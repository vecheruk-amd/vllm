# Eagle3 Speculative Decoding for DeepSeek NextN on vLLM

This branch enables Eagle3 speculative decoding for DeepSeek-V3/R1 using
external NextN draft models (`lmsys/DeepSeek-V3-NextN`, `lmsys/DeepSeek-R1-NextN`).

## Results

| Configuration | Acceptance Rate | Throughput |
|--------------|----------------|------------|
| No speculative decoding | N/A | ~490 tok/s |
| Built-in MTP (1 token) | 90.29% | ~300 tok/s |
| **Eagle3 + NextN (1 token)** | **91.00%** | **~468 tok/s** |
| SGLang Eagle3 (1 token) | 91% | — |

Tested on 8x MI350X, ROCm 7.2, DeepSeek-V3 FP8, TP=8.

## Prerequisites

- 8x AMD MI350X (or MI300X/MI325X) with ROCm 7.2+
- Docker with GPU access (`--device /dev/kfd --device /dev/dri`)
- Target model: `deepseek-ai/DeepSeek-V3` or `deepseek-ai/DeepSeek-V3-0324`
- Draft model: `lmsys/DeepSeek-V3-NextN` (or `lmsys/DeepSeek-R1-NextN` for R1)

## Quick Start

### Step 1: Start Docker Container

```bash
docker run -it --device /dev/kfd --device /dev/dri \
    --network host --shm-size 128g --ipc host \
    -v /path/to/models:/mnt \
    vllm/vllm-openai-rocm:v0.21.0
```

### Step 2: Install Eagle3 Branch

```bash
pip install git+https://github.com/vecheruk-amd/vllm.git@eagle3-deepseek-nextn --no-deps
```

### Step 3: Verify Installation

```bash
python3 -c "
V = '/usr/local/lib/python3.12/dist-packages/vllm'
import os
assert os.path.exists(f'{V}/model_executor/models/deepseek_nextn.py'), 'model class missing'
assert 'DeepseekV3ForCausalLMNextN' in open(f'{V}/model_executor/models/registry.py').read(), 'registry'
assert 'DeepseekV3ForCausalLMNextN' in open(f'{V}/config/speculative.py').read(), 'speculative'
print('All patches verified OK')
"
```

### Step 4: Start Server

```bash
export HSA_NO_SCRATCH_RECLAIM=1
export NCCL_MIN_NCHANNELS=112
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MLA=1
export VLLM_ROCM_USE_AITER_MOE=1

vllm serve deepseek-ai/DeepSeek-V3 \
    --quantization fp8 --kv-cache-dtype fp8 \
    --tensor-parallel-size 8 --block-size 1 \
    --speculative_config '{"method":"eagle3","model":"lmsys/DeepSeek-V3-NextN","num_speculative_tokens":1,"quantization":"fp8"}'
```

If you have a local copy of the draft model, replace the model path:
```bash
--speculative_config '{"method":"eagle3","model":"/mnt/DeepSeek-V3-NextN","num_speculative_tokens":1,"quantization":"fp8"}'
```

### Step 5: Test Output Quality

```bash
curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"deepseek-ai/DeepSeek-V3","prompt":"The capital of France is","max_tokens":32,"temperature":0}'
```

Expected: coherent text starting with " Paris."

### Step 6: Benchmark

```bash
vllm bench serve \
    --base-url http://localhost:8000 \
    --model deepseek-ai/DeepSeek-V3 \
    --dataset-name random \
    --random-input-len 5600 --random-output-len 140 \
    --num-prompts 100 --request-rate inf --temperature 0
```

Expected:
- Acceptance rate: ~91% (greedy)
- Position 0: ~91%
- Output throughput: ~460 tok/s

## What Changed

| File | Change |
|------|--------|
| `vllm/model_executor/models/deepseek_nextn.py` | **New.** Custom NextN model class with full MoE + `enorm/hnorm/eh_proj` fusion |
| `vllm/model_executor/models/registry.py` | Register `DeepseekV3ForCausalLMNextN` and Eagle3 variant architectures |
| `vllm/config/speculative.py` | Skip MTP conversion for NextN; strip `auto_map` to avoid trust_remote_code |
| `vllm/v1/spec_decode/llm_base_proposer.py` | Disable aux hidden states for DeepSeek drafters (uses final hidden state, not 3-layer concat) |

## Technical Details

### Why Aux Hidden States Must Be Disabled

vLLM's Eagle3 path captures 3 intermediate hidden states from the target model
(shape `[N, 3*hidden_size]`) and passes them to the draft model's `combine_hidden_states`.
This works for Llama Eagle3 models which use an `fc` layer to project 3H→H.

DeepSeek NextN models use a different architecture: `eh_proj` fuses token embeddings
with the target's **final** hidden state (shape `[N, hidden_size]`), not intermediate
layers. Without this fix, the draft receives wrong hidden states (from 3 arbitrary
intermediate layers), degrading acceptance from ~91% to ~37%.

### Architecture Name Resolution

The NextN model's `config.json` declares `model_type: "deepseek_v3"` and
`architectures: ["DeepseekV3ForCausalLMNextN"]`. Without our patch:

1. `hf_config_override` sees `deepseek_v3` → converts to `deepseek_mtp` → loads built-in MTP
2. The Eagle3 config wrapper prepends "Eagle3" → `Eagle3DeepSeekMTPModel`
3. This architecture is not registered → crash

Our fix skips the MTP conversion for `DeepseekV3ForCausalLMNextN`, preserving the
original architecture which maps to our custom `DeepseekV3NextNForCausalLM` class.

## Known Limitations

- **`num_speculative_tokens > 1` produces incorrect output.** This is a known upstream
  vLLM issue ([#35288](https://github.com/vllm-project/vllm/issues/35288)) affecting
  ALL multi-token speculative decoding on DeepSeek MLA models (both MTP and Eagle3).
  SGLang achieves 62% with 3 tokens using a different architecture (separate draft-extend phase).

- The NextN model was trained for `DeepSeek-V3-0324`. Using `DeepSeek-V3` (non-0324) works
  but may show slightly different acceptance rates.

- Requires `--block-size 1` for speculative decoding.

## Comparison with SGLang

| Config | vLLM (this branch) | SGLang |
|--------|-------------------|--------|
| Eagle3 1-token | **91%** | 91% |
| Eagle3 3-token | Blocked by #35288 | 62% |
| Built-in MTP 1-token | 90% | N/A |
