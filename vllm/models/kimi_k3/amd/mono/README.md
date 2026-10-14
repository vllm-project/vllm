# Kimi-K3 mono decode (MI355X)

Enable with `VLLM_ROCM_MONO_DECODE=1` on TP8, gfx950. This experimental
extension is based on #60683 at `9c6c96d8ef`, before its shared MonoKernel
runtime refactor. Its serving results do not apply to that PR's current head.

K1 covers AttnRes, input projection, f_b, speculative convolution, KDA
recurrence and gated RMSNorm. K2 covers output projection, the TP reduction,
MLP AttnRes, routed and shared experts, latent reduction and RMSNorm, up
projection and the final TP reduction. MLA's attention and DCP output merge
finish before K2 consumes the local output; this implementation does not fuse
the MLA prologue or epilogue.

## Serving selection

| Rows in the step | Selected implementation |
| --- | --- |
| 1–16 | K2; eligible KDA steps with verification lengths 2, 4, 6 or 8 use the combined K1/K2 schedule outside FULL graphs |
| 18, 20, 24 | K2 with sparse expert routing and compact LDS |
| 28, 30, 32, 36, 40, 42, 48 | K2 also streams projection panels to their consumers |
| 48 | K2 also stages shared H once and pipelines latent RMSNorm/shared-down into up projection in 16-token panels |
| Other widths | Existing projection and AITER MoE path; KDA may still use an eligible K1 path through 64 rows |

M56 and M64 do not select the wide K2 kernel. Measurements have not established
a complete K1+K2 benefit there. Prefill, mixed steps, unsupported layouts and
unsupported graph metadata retain their existing paths.

## Data and worker scheduling

K1 processes complete requests in token tiles, stages input columns in bounded
LDS slices, and retains projection accumulators across those slices. Selected
small shapes pack projection weights once and use separate worker assignments
for projection and recurrence. Compiled callables and stable argument tuples
are cached; state indices, accepted counts and epochs remain runtime inputs.

The combined small-row schedule reuses workers as K1 output becomes ready.
Epoch-tagged handoffs prevent K2 from consuming incomplete local output, and
the TP reductions wait for the corresponding peer data. Experimental reserved
worker and ring-buffer schedules remain available to the benchmark; serving
selects the worker-reuse schedule.

Wide K2 uses compact routing metadata and a sparse expert-pick list. It keeps
expert intermediates in LDS through 32 rows and streams them above that size.
From 28 rows, projection consumers advance after their input panel is ready.
At M48, each up-projection panel waits for all of its latent-normalization
producers and its shared-expert producer. These local readiness flags have
separate scratch space; peer-buffer layout and reduction ordering are retained.

## Compatibility and validation

Full mono is disabled with a KV connector. Local post-projection fusion has
separate eligibility checks; the serialized-offload K1 experiment additionally
requires an explicit connector option. Neither changes the default offload
gate. FULL-only graph metadata is excluded from these KDA fusion paths.
InferenceX conc8 with CPU offload and FULL graphs therefore does not exercise
the complete pipeline.

The numerical suite is `tests/models/kimi_k3/test_amd_mono.py`, including
state/reset checks and changed-input graph replay. Full-model accuracy
equivalence is not established by kernel-level checks.

The rev022 serving snapshot measured 472.28 output tok/s at 8k/1k, conc8/k5,
GPU KV and PIECEWISE graphs: +5.83% versus saved non-mono and -0.74% versus
rev018. This was one formal round against saved three-round medians. Its
scheduler reported a node failure and some raw artifacts were not archived;
the result is provisional and does not establish an incremental gain from
the M48 tail-panel change. The original-PR comparison includes its overhead
when unsupported widths fall back to ordinary operators.
