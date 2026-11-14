# Experimental ROCm AITER FA attention paths

These opt-in paths use `ROCM_AITER_FA`. They are intended for qualified ROCm/AITER deployments; they do not change the default attention path.

## Multi-token verification with paged attention

`VLLM_ROCM_AITER_FA_PAGED_MTP` is read at process startup:

| Value | Eligible uniform causal multi-token decode batches |
| --- | --- |
| `0` (default) | Keep the existing unified-attention fallback |
| `full` | Use paged MTP only in full-attention layers |
| `1` | Also allow supported sliding-window layers |

Select the backend with `--attention-backend ROCM_AITER_FA`. The builder checks uniform verification length; unsupported batches keep the existing path. Sliding-window metadata includes the keys required by the earliest query in the verification block. This setting is for target verification, not a switch to a different draft backend.

!!! warning
    This draft integration requires an AITER `paged_attention_v1` launcher that accounts for the MTP factor in its `exp_sums`, `max_logits` and `tmp_out` workspace regions. The historical benchmark used a patched launcher; the exact supported public revision is still being qualified. Leave the setting at `0` on an unqualified AITER build: the flag does not repair an incompatible launcher.

Measure complete decode steps and verify all query rows, page/window boundaries and graph padding before enabling the path. The previous implementation already batches verification queries; this option does not turn eight separate API calls into one.

## Paged FP8 extend

`VLLM_ROCM_MUSE_FP8_CTX_ATTN=1` enables the experimental paged FP8 extend kernel. It defaults to off. Despite the development flag's model-specific name, the kernel consumes the backend's paged KV/length metadata.

Eligible layers have an FP8 KV cache, head size 64, supported power-of-two query-head/GQA geometry and the non-shuffled paged layout. Sliding windows, ALiBi, attention sinks and logits soft-capping are excluded. The causal extend path attends new queries over both the cached context and the new chunk. Other configurations retain the existing path.

Q is copied to contiguous storage if needed, viewed as `[tokens, heads * head_size]`, then quantized with one scale for the entire tensor. QK and PV use FP8 matrix operations. Low-parallelism batches may use split-KV plus a merge kernel. FP8 arithmetic can change outputs; this path is not a bitwise replacement or batch-invariant mode.

Evaluate complete copy + Q quantization + attention + merge latency and model quality on representative cached/new-token shapes. Kernel-only timing and the historical head-dim64 results do not establish a benefit for every workload.

## Paged FP8 verification for wider batches

`VLLM_ROCM_MUSE_FP8_DECODE_ATTN=1` enables the FP8 verification variant independently of the extend flag. It defaults to off. `VLLM_ROCM_MUSE_FP8_DECODE_MIN_SEQS` is the minimum number of decode requests (default `8`), not the external client concurrency.

The variant additionally requires uniform causal multi-token queries and the same full-attention FP8/layout/head-size restrictions as the extend kernel. Narrow, single-token and unsupported batches keep their prior paths. The default minimum reflects historical MI355X measurements where quantization and merge overhead made 2–4-request batches slower.

Verification uses 64-row tiles, two warps and 64 KV splits; the fixed split count keeps launch topology independent of growing KV length during graph replay. Split-major grid ordering changes scheduling without changing the mathematical work. Fully masked graph-padding rows produce zero output. These choices are hardware/shape-specific and must be validated on the target deployment rather than inferred from the total client concurrency.
