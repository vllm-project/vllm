# DFly

DFly is a checkpoint-native speculative decoder. It drafts multiple tokens
using target hidden states, sparse attention indices, and shared KV storage.
This guide covers DFly with HY4 and HY4 Flash. DFly is disabled by default; enable it
with `--speculative_config '{"num_speculative_tokens":5,"method":"dfly"}'`.
The checkpoint must include DFly configuration and draft weights.

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

- NVIDIA GPUs or AMD gfx950 GPUs (MI350X, MI355X), and the GPU Model Runner V2.
- HY4 and HY4 Flash checkpoints with compatible DFly configuration and draft
  weights.
- BF16 computation with unquantized weights, online MoE-only block FP8
  quantization, native block FP8 checkpoints, or compressed-tensors checkpoints
  without a KV cache scheme or transforms. HY4 targets can use any layer scheme
  the model supports, such as NVFP4 experts. HY4 Flash targets are limited to
  W8A8 FP8 (128×128 weight blocks, dynamic group-128 activations) and weight-only
  MXFP4 (group-32). The shared embedding and LM head must be BF16.
- Draft layers in BF16 or W8A8 FP8 with 128×128 weight blocks and dynamic
  activations, independent of the target's scheme.
- Matching target and draft KV cache dtype: BF16 (`auto` or `bfloat16`)
  or `fp8_ds_mla`. Omit the draft KV override to inherit the target dtype.
  Both are supported on ROCm (gfx950).
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

On ROCm, DFly is validated with HY4 FP8 (`tencent/Hy4-FP8`), HY4 Flash FP8
(`tencent/Hy4-flash-FP8`) and HY4 Flash BF16 (`tencent/Hy4-flash`) on eight
MI355X GPUs with TP8. Other targets, such as HY4 BF16, NVFP4 experts, or
MXFP4, are accepted on gfx950 but have not been validated. Set
`VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1`, as described in the
[HY4 model notes](../../models/supported_models.md):

```bash
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1 \
vllm serve tencent/Hy4-flash-FP8 \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --speculative_config '{"num_speculative_tokens":5,"method":"dfly"}'
```

Measure acceptance and latency on your own workload using the
[acceptance metrics](acceptance_metrics.md) and
[benchmark CLI](../../benchmarking/cli.md). This implementation does not
guarantee a speedup for every workload.
