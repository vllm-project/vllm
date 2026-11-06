# DeepSeek-V4.1 KV records on ROCm

On gfx950, `--kv-cache-dtype fp8` uses the 528-byte MXFP8 record. A single
instance with `--monokernel` or aiter sparse MLA enabled retains the 584-byte
V4 record for those readers. The aiter environment variable has no effect on
the record when that backend is disabled.

`--kv-cache-dtype nvfp4_ds_mla` uses a 528-byte MXFP8 sliding-window record
and a 288-byte NVFP4 compressed record. It uses vLLM's Triton attention;
aiter sparse MLA remains disabled for this cache type.

With a KV connector, both prefill and decode use the 528-byte window record,
independent of `--monokernel` and the aiter sparse MLA flag. Use the same
model and cache dtype on both peers. The current Mono reader falls back to
FFN-only acceleration for this record; full Mono support is a follow-up.
