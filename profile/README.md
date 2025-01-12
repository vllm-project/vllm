## VLLM TPU Profiling

This guide explains how to profile the TPU performance on VLLM for specific shapes. Note: an actual running server is a mix of both prefill of many shapes and decode of many shapes.

>> In all examples below, we run several warmups before (so `--enforce-eager` is okay)

### Generate Decode Trace

```bash
export XLA_HLO_DEBUG=1
export MODEL=meta-llama/Llama-3.1-70B-Instruct
export VLLM_TPU_PROFILE_DURATION_MS=2000
export VLLM_TPU_PROFILE_DELAY_MS=1000

rm -rf ~/.cache/vllm/xla_cache
python3 profile_tpu.py \
    --model $MODEL \
    --input-len 1 \
    --output-len 128 \
    --batch-size 32 \
    --enforce-eager \
    --profile-result-dir profiles \
    --max-model-len 2048 --tensor-parallel-size 8
```

### Generate Prefill Trace

```bash
export XLA_HLO_DEBUG=1
export MODEL=meta-llama/Llama-3.1-8B-Instruct
export VLLM_TPU_PROFILE_DURATION_MS=3000
export VLLM_TPU_PROFILE_DELAY_MS=0

rm -rf ~/.cache/vllm/xla_cache
python3 profile_tpu.py \
    --model $MODEL \
    --input-len 1024 \
    --output-len 1 \
    --batch-size 1 \
    --enforce-eager \
    --profile-result-dir profiles \
    --max-model-len 2048 --tensor-parallel-size 8
```
