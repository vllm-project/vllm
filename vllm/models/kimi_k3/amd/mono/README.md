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
| 28, 30, 32 | K2 also streams projection panels to their consumers |
| Other widths | Existing projection and AITER MoE path; KDA can use an eligible K1 path only through 32 rows |

The scope is conc4 verification, including four requests with up to eight
verification tokens each (M32). Dispatch uses the model step's row count,
not the client's concurrency setting. AgenticX session concurrency can admit
more than four simultaneous HTTP requests; any step above M32 falls back.
K1, K2 and post-projection KDA all stop at M32. M30/M32 KDA retains the
production input projection before post-projection fusion. Prefill, mixed
steps, unsupported layouts and unsupported graph metadata retain their
existing eligibility checks.

## Data and worker scheduling

K1 processes complete requests in token tiles, stages input columns in bounded
LDS slices, and retains projection accumulators across those slices. Selected
small shapes pack projection weights once and use separate worker assignments
for projection and recurrence. Compiled callables and stable argument tuples
are cached. Captured argument-table copies retain their immutable pinned host
storage for subsequent graph replays; state indices, accepted counts and epochs
remain runtime inputs.

The combined small-row schedule reuses workers as K1 output becomes ready.
Epoch-tagged handoffs prevent K2 from consuming incomplete local output, and
the TP reductions wait for the corresponding peer data. Experimental reserved
worker and ring-buffer schedules remain available to the benchmark; serving
selects the worker-reuse schedule.

Wide K2 uses compact routing metadata and a sparse expert-pick list. It keeps
expert intermediates in LDS through 32 rows. From 28 rows, projection consumers
advance after their input panel is ready. The M48 shared-expert and tail-panel
schedules have been removed; scratch and peer buffers cover the retained
shapes through M32.

## Compatibility and validation

Full mono is disabled with a KV connector. Local post-projection fusion has
separate eligibility checks; the serialized-offload K1 experiment additionally
requires an explicit connector option. Neither changes the default offload
gate. FULL-only graph metadata is excluded from these KDA fusion paths.

The numerical suite is `tests/models/kimi_k3/test_amd_mono.py`, including
state/reset checks and changed-input graph replay. Full-model accuracy
equivalence is not established by kernel-level checks.

Historical AgenticX conc4/k5 results with the broader rev023 graph-fixed
candidate measured 190.84 output tok/s versus 181.91 for the original #60683
snapshot (+4.91%), and p90 interactivity of 146.53 versus 123.45 tok/s (+18.70%).
These were single 3600-second runs on different machines (Crusoe control, AAC
candidate), with synthetic acceptance length 3.62, seed 42, TP8, GPU KV and
FULL_AND_PIECEWISE graphs. They do not isolate the kernel effect or establish
performance of the M32-limited revision: the historical recipe admitted eight
HTTP requests and graph sizes through M48. The M32
revision needs a new serving measurement before reusing those gains as a claim.
