# Kimi-K3 KV-balanced MLA decode (SM100 family)

An opt-in MLA attention backend for Kimi-K3 on Blackwell (SM100/SM103),
`KIMI_K3_KV_BALANCED_MLA`, whose decode splits each request's KV into a number of
chunks planned from its length, so that the GPU's clusters get about the same
amount of KV. It is the `TOKENSPEED_MLA` backend (`backend.py` subclasses it) with
a different decode attention kernel; everything else (prefill, the KV cache, the
decode metadata, the FP8 query, the DCP combine) is TokenSpeed's.

The kernel is TokenSpeed MLA 2.0.3's FP8 MLA decode, vendored in `kernel.py` and
`helpers.py` with these changes:

- **Planned splits:** a planner kernel gives each request 1 to 8 KV splits of at
  least 1024 tokens from its KV length. It runs once per forward, on the device,
  and every MLA layer reuses the plan, so it can be captured in a CUDA graph.
- **Work list:** one wave of persistent 2-CTA clusters walks the list of
  (query tile, request, split) entries, instead of a grid with the same split
  count for every request.
- **Direct output:** a request planned as one split stores its final output, and
  the reduction skips it.
- **P prescale:** P is stored in E4M3 scaled by 2^6, so that the small
  probabilities of a long split don't flush to zero.

## Enabling

```bash
vllm serve <Kimi-K3> --attention-backend KIMI_K3_KV_BALANCED_MLA --kv-cache-dtype fp8 \
  --speculative-config '{..., "attention_backend": "KIMI_K3_KV_BALANCED_MLA"}'
```

The backend is never selected automatically. The DSpark draft takes its attention
backend from the speculative config, so set it there too to run the draft's decode
on the kernel.

## Supported deployments

Those of `TOKENSPEED_MLA` (an FP8 E4M3 KV cache on an SM100-family GPU, DeepSeek
MLA head dimensions) with `kv_lora_rank` 512, more than 64 query rows per request
(heads × query tokens), with or without decode context parallel. Backend selection
rejects any other configuration with the reason.

At run time, a step the kernel can't run uses TokenSpeed's decode: more requests
than the split-KV scratch can address, a strided query, a micro-batch lane first
seen after the workspace is locked, or the first call for a decode shape inside
CUDA graph capture. `tokenspeed_mla_min_split_kv` applies only to
TokenSpeed's decode.

The split-KV scratch (FP16 partial outputs) is reserved when the model is built,
and its size is logged.
