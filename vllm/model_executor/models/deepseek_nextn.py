# SPDX-License-Identifier: Apache-2.0
"""NextN speculative decoding model for DeepseekV3 with full MoE.

Uses the real DeepseekV2DecoderLayer as mtp_block (matching the MTP model
architecture exactly) to ensure bit-for-bit computation parity with MTP.
Supports lmsys/DeepSeek-V3-NextN and lmsys/DeepSeek-R1-NextN draft models
which declare architecture 'DeepseekV3ForCausalLMNextN'.
"""

import re
from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import DeepseekV2Config, DeepseekV3Config

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2ForCausalLM,
    DeepseekV2MoE,
)

from vllm.model_executor.layers.fused_moe import FusedMoE

try:
    from vllm.model_executor.layers.fused_moe import SharedFusedMoE
except ImportError:
    SharedFusedMoE = FusedMoE

# On v0.18.0+, deepseek_eagle3.py exists and we inherit from
# Eagle3DeepseekV2ForCausalLM so the isinstance check in eagle.py passes.
try:
    from vllm.model_executor.models.deepseek_eagle3 import (
        Eagle3DeepseekV2ForCausalLM,
    )
    _NEXTN_BASE_CLASS = Eagle3DeepseekV2ForCausalLM
except ImportError:
    _NEXTN_BASE_CLASS = DeepseekV2ForCausalLM
from vllm.multimodal.inputs import NestedTensors

from .utils import (
    AutoWeightsLoader,
    maybe_prefix,
    process_eagle_weight,
)

logger = init_logger(__name__)

# Weight names that belong to the Eagle3 outer layer (NOT inside mtp_block)
_EAGLE3_LAYER_WEIGHTS = ("enorm", "hnorm", "eh_proj", "shared_head")


class SharedHead(nn.Module):
    """Wrapper for shared_head.norm to match checkpoint weight naming
    (model.layers.0.shared_head.norm.weight)."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=eps)


class DeepseekV3NextNDecoderLayer(nn.Module):
    """NextN decoder layer wrapping DeepseekV2DecoderLayer as mtp_block.

    Architecture:
      - enorm, hnorm, eh_proj: Eagle3 input processing
      - mtp_block: real DeepseekV2DecoderLayer (MLA + MoE + layernorms)
      - shared_head.norm: fused residual add + RMSNorm after mtp_block

    Returns (normed, pre_norm) to avoid separate add and norm kernels.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config: DeepseekV2Config | DeepseekV3Config,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        # Eagle3 specific layers (matching MTP exactly)
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )

        self.shared_head = SharedHead(
            config.hidden_size, config.rms_norm_eps
        )

        # Real DeepseekV2DecoderLayer — same class the target model uses.
        # This ensures bit-for-bit computation parity with MTP.
        self.mtp_block = DeepseekV2DecoderLayer(
            vllm_config, prefix, config=config
        )

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeds = self.enorm(embeds)
        hidden_states = self.hnorm(hidden_states)
        hidden_states = self.eh_proj(
            torch.cat([embeds, hidden_states], dim=-1)
        )

        hidden_states, residual = self.mtp_block(
            positions=positions,
            hidden_states=hidden_states,
            residual=None,
        )

        # Separate add + norm (torch.compile compatible).
        # The 2-arg RMSNorm(h, r) fused kernel breaks torch.compile tracing.
        pre_norm = residual + hidden_states
        normed = self.shared_head.norm(pre_norm)
        return normed, pre_norm


@support_torch_compile
class DeepseekV3NextNModel(nn.Module):
    """Inner model for NextN speculative decoding."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = (
            vllm_config.speculative_config.draft_model_config.hf_config
        )
        self.vocab_size = self.config.vocab_size

        current_vllm_config = get_current_vllm_config()

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.layers = nn.ModuleList(
            [
                DeepseekV3NextNDecoderLayer(
                    current_vllm_config,
                    prefix=maybe_prefix(
                        prefix, f"layers.{layer_idx + start_layer_id}"
                    ),
                    config=self.config,
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)

        for layer in self.layers:
            normed, pre_norm = layer(
                positions=positions,
                embeds=input_embeds,
                hidden_states=hidden_states,
            )
            hidden_states = pre_norm

        return normed, pre_norm


class DeepseekV3NextNForCausalLM(_NEXTN_BASE_CLASS):
    """NextN speculative decoding model for DeepseekV3 with full MoE."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = (
            vllm_config.speculative_config.draft_model_config.hf_config
        )

        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(
                self.config, "vocab_size", None
            )

        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.config.target_layer_count = target_layer_num

        self.model = DeepseekV3NextNModel(
            vllm_config=vllm_config,
            prefix="model",
            start_layer_id=target_layer_num,
        )

        # Set up MoE expert mapping (required for weight loading on v0.19.0+)
        self._setup_expert_mapping()

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        self.draft_id_to_target_id = nn.Parameter(
            torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
            requires_grad=False,
        )

    def _setup_expert_mapping(self):
        """Set expert_mapping on all FusedMoE layers for weight loading.

        On v0.19.0+, FusedMoE.load_weights requires expert_mapping.
        We generate it using SharedFusedMoE.make_expert_params_mapping
        and set it on each FusedMoE experts layer.
        """
        n_routed = getattr(self.config, "n_routed_experts", 0) or 0

        expert_mapping = SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=n_routed,
        )

        # Set expert_mapping on all FusedMoE instances in the model
        for module in self.modules():
            if isinstance(module, FusedMoE) and module.expert_mapping is None:
                module.expert_mapping = expert_mapping

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The proposer unpacks as (last_hidden_states, hidden_states):
        #   - normed: already post-norm, goes to compute_logits
        #   - pre_norm: pre-norm sum, propagates to next draft iteration
        normed, pre_norm = self.model(
            input_ids, positions, hidden_states, inputs_embeds
        )
        return normed, pre_norm

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        # hidden_states is already post-norm (fused in the decoder layer).
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            return logits

        base = torch.arange(
            self.config.draft_vocab_size, device=logits.device
        )
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full(
            (logits.shape[0], self.config.vocab_size),
            float("-inf"),
        )
        logits_new[:, targets] = logits
        return logits_new

    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # NextN does not use the Eagle3 'fc' layer for combining 3 auxiliary
        # hidden states. Instead, it uses eh_proj inside the decoder layer
        # to combine embed + hidden. Here we extract the final auxiliary
        # hidden state (the last hidden_size slice) from the concatenated
        # auxiliary states provided by the Eagle3 framework.
        hidden_size = self.config.hidden_size
        return hidden_states[..., -hidden_size:]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load weights using MTP-compatible logic.

        Handles stacked params (gate_up_proj), expert weight mapping,
        and mtp_block prefix remapping — matching DeepSeekMTP.load_weights.
        """
        import typing
        from collections.abc import Callable

        from vllm.model_executor.model_loader.weight_utils import (
            default_weight_loader,
            maybe_remap_kv_scale_name,
        )

        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
        ]

        n_routed = getattr(self.config, "n_routed_experts", 0) or 0
        expert_params_mapping = SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=n_routed,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # Pre-process weights: fix model. prefix and add mtp_block.
        processed_weights = []
        for name, loaded_weight in weights:
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
            elif "lm_head" not in name:
                if not name.startswith("model."):
                    name = "model." + name

            # Add mtp_block. prefix for transformer block weights
            if (
                re.match(r"model\.layers\.\d+\.", name)
                and not any(kw in name for kw in _EAGLE3_LAYER_WEIGHTS)
            ):
                name = re.sub(
                    r"(model\.layers\.\d+\.)",
                    r"\1mtp_block.",
                    name,
                )

            if "embed_tokens" in name:
                process_eagle_weight(self, name)
            process_eagle_weight(self, name)
            processed_weights.append((name, loaded_weight))

        for name, loaded_weight in processed_weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # Phase 1: stacked params (gate_up_proj, fused_qkv_a_proj)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                if (
                    param_name == "fused_qkv_a_proj"
                    and name_mapped not in params_dict
                ):
                    continue
                else:
                    name = name_mapped
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Phase 2: expert weights
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if name_mapped not in params_dict:
                        continue
                    param = params_dict[name_mapped]
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param, loaded_weight, name_mapped,
                        shard_id=shard_id, expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        continue

                    # Phase 3: direct load (norms, Eagle3 layers, etc.)
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

            if name is not None:
                loaded_params.add(name)

        return loaded_params
