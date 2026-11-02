# DeepSeek-V4.1 Mono decode on ROCm

On gfx950, `--monokernel` lets DeepSeek-V4.1 use Mono decode with its
528-byte MXFP8 KV record. Enable it with `--monokernel --kv-cache-dtype fp8`.
The record includes all 512 dimensions and 16 scale bytes per token.
Without `--monokernel`, vLLM uses its ordinary attention path.

With `VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1`, aiter uses MXFP8 only if
its installed reader advertises the 528-byte record. Older aiter versions
retain the 584-byte V4 record. On gfx950, aiter sparse MLA also supports NVFP4
when its installed reader advertises both the 528-byte window and 288-byte
compressed records. Otherwise NVFP4 uses the ordinary Triton attention path.

With a KV connector, prefill and decode use the same 528-byte window record
regardless of their Mono/aiter flags. Use the same model and cache dtype on
both peers. Mono can read that record without the previous FFN-only fallback.

The implementation also adds Mono reads of the 288-byte NVFP4 compressed
record with `--kv-cache-dtype nvfp4_ds_mla`. Its sliding-window cache stays
in the 528-byte MXFP8 format. NVFP4 Mono support is limited to gfx950.
