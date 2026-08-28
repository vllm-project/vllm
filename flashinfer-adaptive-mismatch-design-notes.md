# FlashInfer conditional device/CPU query-lens mismatch support (SM120 XQA)

Design notes for the `feat/flashinfer-adaptive-mismatch` branch.
Evidence and validation record:
`docs/active/flashinfer-adaptive-mismatch/validation-20260827.md` in the
author's workspace (summarized below).

## Problem

Adaptive verification trims per-request verify windows ON DEVICE; the
device `query_start_loc` diverges from the CPU copy while totals stay
conserved. The FlashInfer backend opted out
(`supports_device_cpu_query_lens_mismatch() -> False`) because parts of
its decode planning read the CPU copy — so adaptive was forced onto
Triton for full attention, ~2-3x slower than FlashInfer at long context
on SM120 (parent campaign E31b control: Triton static 54.9/30.2 t/s ≈
Triton adaptive 40.7/32.4, vs FlashInfer-static 141 t/s at 32K).

## Why it is safe on SM120 (the founding probe, W0)

FlashInfer 0.6.17's dedicated XQA API takes a CUDA `q_cu_seq_lens` the
kernel dereferences on device, with a scalar max-draft-length bound.
The W0 probe captured a CUDA graph with uniform geometry and replayed it
across in-place-updated ragged layouts — bitwise-identical to eager XQA
and within one bf16 ulp of an SDPA oracle, no host synchronization, on
the production-flags module (fp8 KV + SWA + spec width 8).

## Design

1. **One shared predicate** `_adaptive_xqa_mismatch_safe(vllm_config)`:
   target-side adaptive on + SM120/SM121 + DCP=1 + causal target
   attention + not forced-native. The selector gate, the manager's
   independent scan, and the VARLEN_DECODE cudagraph result all consult
   it, so the two adaptive boot gates cannot disagree.
2. **Deferred finalize** (`_finalize_adaptive_decode(decode_query_len)`):
   the runner's resolved decode width is the authoritative bound
   (model-state data, not on SpeculativeConfig); the group's model role
   comes from a tag recorded at Attention construction (the compilation
   global is restored before builders exist). Allocation + kernel
   warmup are transactional; the mode flag flips last.
3. **Mismatch-safe classification**: the CPU even-distribution decides
   only the provable single-token fast path (all widths exactly 1);
   otherwise device offsets pass through untouched with the fixed
   runner-resolved bound. A threshold clamp routes anything wider than
   the bound through prefill (never through uniform-XQA metadata).
4. **Persistent device-built packed mask** (Triton kernel, preallocated
   buffer, binary search over device offsets, stale-proof tail zeroing)
   rewritten from device offsets every step — never built from CPU data.
5. **Exclusions**: cascade disabled for mismatch batches before any
   wrapper planning; defense-in-depth asserts against native decode,
   DCP, and uniform-only TRTLLM-API metadata.
6. **SM100 prepare-only**: `supports_device_ragged_decode` is the single
   kernel-capability declaration; TRTLLM-gen varlen metadata is wired
   but requires an explicit `trtllm_gen_varlen` authorization that no
   live path can set until real-SM100 hardware validation exists.

## Measurements (deployed wheel, parent W1 harness, ~49K ctx)

| arm | accept rate | warm decode t/s |
|---|---|---|
| FlashInfer + DFlash2 static | 3.36 | 141.0 |
| FlashInfer + DSpark static | 2.04 | 96.6 |
| Triton + DSpark adaptive (before) | 1.78-2.46 | 40.7 / 32.4 |
| **FlashInfer + DSpark adaptive (this)** | ~1.90 | **~82** |

Serving-time evidence: FULL-cudagraph replays with device != CPU widths
(e.g. dev [6,2,1,7,0...] vs cpu [4,4,4,4,0...]) logged immediately
before `graphs[desc].replay()`; identity oracle vs static 8/8
token-identical.

## Known limitations (documented for review)

- Class-level capability cannot see per-group head counts; a hybrid
  with group head counts differing from model-wide can pass the
  selector and fail closed at finalize instead of falling back. The
  upstream-shaped fix is a group-aware capability query.
- The adaptive width POLICY (argmax over confidence-survival /
  boot-profiled cost) trims to width 4 at c=1 on this workload — a
  measured net loss on fast backends (drafter-independent of this
  patch; see the parent campaign's policy-residual analysis).
