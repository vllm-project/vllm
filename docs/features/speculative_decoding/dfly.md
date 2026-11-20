# DFly

DFly is a checkpoint-native speculative decoder. It drafts multiple tokens
using target hidden states, sparse attention indices, and shared KV storage.
This guide covers DFly with HY4 Flash. DFly is disabled by default; enable it
with `--speculative_config '{"num_speculative_tokens":5,"method":"dfly"}'`.
The HY4 Flash checkpoint must include DFly configuration and draft weights.

DFly is introduced in [AngelSpec: Towards Real-World High Performance Inference
with Speculative Decoding](https://arxiv.org/abs/2607.25852).

## Usage

For HY4 Flash FP8 on eight NVIDIA GPUs, with five speculative tokens:

```bash
vllm serve tencent/Hy4-flash-FP8 \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --speculative_config '{"num_speculative_tokens":5,"method":"dfly"}'
```

## Supported configurations

- NVIDIA GPUs, or AMD MI355X (gfx950) GPUs for HY4 Flash targets, and the
  GPU Model Runner V2.
- HY4 Flash checkpoints with compatible DFly configuration and draft weights.
- BF16 computation with unquantized weights, online MoE-only block FP8
  quantization, or compressed-tensors W8A8 FP8 checkpoints using 128×128
  weight blocks and dynamic group-128 activations. HY4 Flash checkpoints may also
  use weight-only MXFP4 (group-32) weights; draft layers must be BF16 or FP8.
- Matching target and draft KV cache dtype: BF16 (`auto` or `bfloat16`)
  or `fp8_ds_mla`. Omit the draft KV override to inherit the target dtype.
  `fp8_ds_mla` is CUDA-only; ROCm requires BF16.
- Matching target and draft tensor parallelism. Pipeline, prefill context,
  and decode context parallel sizes must be one. HY4 Flash supports data
  parallelism, with each DP replica running a TP-local draft.
- Greedy or probabilistic draft sampling (`draft_sample_method`) with standard
  target verification. Probabilistic drafts sample at each request's
  temperature, which can improve acceptance when HY4 Flash samples at a
  nonzero temperature.

DFly currently rejects LoRA, microbatching, disaggregated prefill/decode,
heterogeneous vocabularies, adaptive verification, and a separate draft
attention backend.

On ROCm, DFly is validated with HY4 Flash BF16 (`tencent/Hy4-flash`) and FP8
on eight MI355X GPUs:

```bash
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1 \
vllm serve tencent/Hy4-flash-FP8 \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --speculative_config '{"num_speculative_tokens":5,"method":"dfly"}'
```

DFly with HY4 (non-Flash) targets is not supported on ROCm.

Measure acceptance and latency on your own workload using the
[acceptance metrics](acceptance_metrics.md) and
[benchmark CLI](../../benchmarking/cli.md). This implementation does not
guarantee a speedup for every workload.
