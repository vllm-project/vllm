# Key-B recovery draw inside the resample kernel

Branch `watermark-specdec-kernel-residual`, one commit on top of vllm-project/vllm#56122.
This note explains what the commit does and how it was verified, so that it can be cherry-picked with confidence.

## Problem

Under speculative decoding, dual-key Gumbel watermarking draws every token the target model supplies itself with key-B Philox noise. That is the residual draw after the first rejected draft, or the bonus token when all drafts are accepted.

PR 56122 produced that draw by running the stock Triton rejection sampler to completion and then rebuilding the residual distribution in PyTorch for one row per request, about ten full-vocabulary passes, before overwriting the kernel's token. At a vocabulary of 151,936 this made the verification step four to six times slower than the stock sampler and allocated about 1 GiB transiently at 256 requests, outside the memory profiling run. It also left two residual implementations, and the PyTorch one lacked the block-verification and one-hot branches.

## Approach

The stock `_resample_kernel` already holds the residual in registers and draws from it with seeded Gumbel noise. The commit passes three more inputs into that kernel: the per-row watermark contexts, the two 32-bit halves of key B, and a per-request enable mask. On enabled rows the seeded noise is replaced by keyed Philox noise. Everything else in the kernel is untouched, so the residual is computed once and the PyTorch recompute is deleted.

The Philox PRF is factored out of the standalone `_philox_gumbel_kernel` and shared, so for a given context and token the uniform is bit-identical to `philox_gumbel_sample`. Rows that are opted out, greedy, or padded keep the stock seeded draw bit-for-bit. The fp64 Gumbel option does not reach the keyed draw, because the detector's uniform is fp32 and the watermark must stay a function of context and token alone.

Two smaller changes ride along. Key resolution now compares against the target role's key and rejects an unsplit dual-key watermarker, which would otherwise have resolved to key A silently. The spec-decode rejection warmup now compiles the Triton specialization the engine actually launches, matching the runtime dtypes of contexts, index mappings and draft tokens, the real derived key, and whether draft logits are present.

Files touched:

- `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`: kernel arguments and the keyed branch in `_resample_kernel`, new keyword arguments on `rejection_sample`.
- `vllm/v1/worker/gpu/sample/watermark.py`: shared Philox helpers and `philox_gumbel_block_argmax`.
- `vllm/v1/watermarking/spec_decode.py`: key resolution, the sampler hook that supplies contexts, mask and key; the PyTorch recompute is removed.
- `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`: a small hook so the subclass does not copy `_verify`.
- `vllm/model_executor/warmup/spec_decode_rejection_warmup.py`: warm the launched specialization.
- Tests under `tests/v1/spec_decode/` and `tests/watermarking/`.

## Verification

Verification had two tiers. A fast tier ran on every iteration and gated each change. A slow end-to-end tier ran once at the end.

**Parity against the previous implementation.** The PyTorch recovery path from the PR head was frozen as a reference and compared with the new kernel on identical inputs. The grid covered rejection at the first step, rejection later, all drafts accepted, zero drafts, padded rows, opted-out and greedy requests mixed into one batch, draft vocabulary smaller than target vocabulary, masked target logits, batch sizes 1 and 256, K from 1 to 3, bf16 and fp32 logits, several temperatures, context widths from 1 to 16, and several keys. Emitted tokens matched exactly on every row, in eager mode, under chunked verification, under CUDA graph capture and replay, and on the fp64 sub-grid. Planting a one-row context shift or an inverted enable mask made the parity check fail, so the check has power.

**Direct tests for what the old path lacked.** Block verification, one-hot drafts, all-minus-infinity and NaN rows, and the Philox oracle against the PyTorch PRF each have their own check.

**Unit tests in this branch.** The in-kernel draw is compared with `philox_gumbel_sample` across vocabulary sizes that span several resample blocks, because at a single block a dropped per-block offset had left the suite green. Disabled rows are compared with the stock sampler. The fp64 flag is shown not to change the keyed draw. The warmup test pins the runtime dtypes.

**Review.** Independent review passes covered kernel correctness and integration. All medium findings were fixed and re-reviewed. Remaining low findings are listed below.

**End-to-end.** Qwen3.5-2B with its MTP head, 40 prompts and 256 tokens, plus a 256-request variant, on an H200. Seven engine configurations were run on the PR head and on this branch. All 712 generations are token-identical between the two, so acceptance rate, mean accepted length, and every detector statistic under key A only, key B only, both keys and the null are unchanged by construction. The same holds for generation quality: no task benchmark such as GSM8K was run for this commit, because any score the PR head obtains under these configurations, this branch obtains too.

## Numbers

Verification step in isolation, vocabulary 151,936, CUDA events, H200:

| batch, K | stock sampler | PR head | this branch |
|---|---|---|---|
| 1, K=2 | 0.14 ms | 0.75 ms | 0.15 ms |
| 256, K=2 | 0.47 ms | 2.80 ms | 0.42 ms |
| 256, K=5 | 0.65 ms | 2.98 ms | 0.60 ms |

Transient allocation in that step at batch 256: PR head 1076 MiB, this branch 0.9 MiB, same as stock.

End-to-end, serial runs on the same GPU pool:

| configuration | tok/s PR head | tok/s this branch | transient MiB PR head | transient MiB this branch |
|---|---|---|---|---|
| dual key, MTP K=2, batch 40 | 9126 | 10660 | 333 | 113 |
| dual key, MTP K=3, batch 40 | 9402 | 9483 | 352 | 113 |
| dual key, MTP K=2, 256 requests | 19734 | 19655 | 2130 | 721 |

Run-to-run noise on the shared machine is about 9 percent, so the batch-40 gain is real and the 256-request throughput is flat within noise. Single-request latency relative to an unwatermarked run improved from 1.37 to 1.20 times. The remainder is the draft-side Philox draw and per-step context build, which this commit does not touch.

## Open items

Low-severity findings, not addressed here:

- The warmup hardcodes fp64 Gumbel off, so an engine started with that flag still compiles on its first request.
- The warmup compiles the keyed branch but never executes it, so a launch-time fault there would not surface at startup.
- The enable mask is checked for dtype and rank but not for unit stride before the kernel indexes it.

Out of scope and unchanged from PR 56122: the repeated-context deduplication kernel, `alpha` handling under speculative decoding, config validation, and the non-speculative routing-logits allocation.

Not established by this commit or by PR 56122: whether dual-key watermarking under speculative decoding changes task accuracy relative to unwatermarked speculative decoding. The theory gives single-token non-distortion in expectation, and the measured acceptance rate is unchanged, but no GSM8K or similar benchmark has been run for the PR. That is a question for the PR, not for this commit, which cannot move the answer.
