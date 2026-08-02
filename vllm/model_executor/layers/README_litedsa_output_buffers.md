# LiteDSA output-buffer reuse candidate

`VLLM_LITEDSA_REUSE_OUTPUT_BUFS=1` enables an opt-in reuse pool for the
`out`, `max_logits`, and `lse` tensors passed to
`torch.ops._C.litedsa_masked_mla_fp8`. The default is `0`.

## Why the production call chain is ordered safely

- The operator schema marks all three tensors as mutable outputs (`Tensor!`).
  The C++ wrapper passes their storage to the phase-1 kernel; the kernel stores
  `max_logits`/`lse` directly and writes `out` through its output TMA. It does
  not retain a tensor or pointer after the launch.
- `max_logits` and `lse` do not escape `litedsa_masked_mqa`.
- `out` is returned to `MultiHeadLatentAttentionWrapper.forward`, which
  immediately enqueues `o_proj(attn_out)` before the next transformer layer
  can call LiteDSA. CUDA launches on one stream are ordered, so a later LiteDSA
  overwrite cannot start before that projection has finished reading `out`.
- Pools are separated by CUDA device, current-stream handle, host-thread id,
  trailing shape, and dtype. The thread id prevents two Python dispatchers
  targeting the same stream from receiving the same storage concurrently.

The pool grows the leading `ng` capacity geometrically and retains at most four
device/stream/thread pools. A smaller request receives a contiguous prefix
view.

## Fail-closed cases

- CUDA Graph capture raises if this option reaches the low-level helper. The
  current backend already routes away from LiteDSA while capturing.
- Autograd uses fresh tensors because backward may retain the returned output.
- The option must remain disabled for a caller that retains raw attention
  outputs beyond the immediate `o_proj` consumer (for example, a custom debug
  hook). Reusing returned storage intentionally narrows the ordinary PyTorch
  tensor-lifetime contract to the current vLLM inference call chain.

## Validation

The CPU-only helper tests cover geometric growth, contiguous prefix views,
same-pool reuse, stream/thread separation, bounded LRU eviction, and the
autograd/capture guards:

```bash
pytest -q tests/model_executor/layers/test_litedsa_output_buffers.py
```

Before enabling by default, run the strict GLM-5.2 TP8 1M, util=0.90, MTP
end-to-end workload with the option off and on in interleaved A/B order. This
candidate removes three `torch.empty` calls per LiteDSA layer; it does not
change the attention CUDA kernel, so the expected benefit is reduced host
launch/allocation overhead rather than lower kernel time.
