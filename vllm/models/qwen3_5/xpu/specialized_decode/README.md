# Qwen3.5 XPU fused MoE decode

This package accelerates small-row routed and sigmoid-gated shared experts in
the dedicated XPU eager model. Eligible calls use the native
`XpuFusedMoeWithSharedExpert` provider from `vllm-xpu-kernels`; other calls use the
existing `Qwen3NextSparseMoeBlock` implementation. Dense models are unaffected.

## Scope and fallback

Eligibility is based on input token rows, including speculative verification
rows, rather than request concurrency. Calls require 1–8 rows and
`rows * top_k <= 512`. An eligible short prefill can also use the same operator.
Inputs must be rank-two, contiguous, 64-byte-aligned FP16/BF16 tensors with the
configured hidden dimension, dtype and weight device.

Static eligibility is checked after weight loading:

- The routed backend is `XPUExpertsFp8`; both shared projections use
  `XPUW8A16FP8LinearKernel`. W8A8 shared projections retain their activation
  quantization through the stock path.
- Routed FP8-e4m3 weights load as `[E, 2I, H]` and `[E, H, I]`, with
  one FP32 scale per expert. The common XPU provider prepares `[E, H, 2I]`
  and `[E, I, H]` before the decode executor binds them.
  Shared weights have matching `[H, 2I]` and `[I, H]`
  shapes, each with one FP32 scale. The shared gate is `[H]` or `[1, H]`, with
  the activation dtype. All tensors are on the same XPU device.
- `1 <= top_k <= E <= 512`; `H` and `I` are positive multiples of 32. The native
  block-2D descriptor limits require `2H <= 2^24 - 1` and `4I <= 2^24 - 1`.
  The installed provider must support the weight device through its native
  Xe2/Xe3 dispatcher.
- Routed and shared experts use SiLU without bias, unit routed scaling and
  router weights applied to expert outputs. The stock router still selects
  the experts.
- Sequence/expert/data/prefill-context parallelism, expert maps, EPLB, DBO,
  LoRA, routed input/output transforms, zero-expert routing, replicated shared
  experts and an already-enabled shared-expert fusion use the stock path.
- Contiguous weight matrices require 64-byte alignment; the contiguous shared
  gate requires 32-byte alignment. Eligible non-contiguous parameters are
  made contiguous during post-load preparation.

Missing providers and failed static checks emit a reason once during
initialization. Runtime row, shape, device, dtype, contiguity and alignment
checks silently select the stock path when a call is ineligible. Forward also
uses stock when the post-load hook has not initialized an executor.

## Default behavior and numerics

Fusion is enabled by default when the provider and model configuration qualify.
It replaces local expert computation and preserves the stock router and final
reduction contract: the combined routed/shared output receives one standard TP
all-reduce when that reduction is required. Under the supported configurations,
either local path reaches the same collective, so no additional TP agreement or
kernel peer-communication protocol is needed. This layer optimization has no
whole-model decode selector or persistent communication kernel.
No additional specialized-decode fail-stop mechanism is needed because the
provider introduces no kernel peer-communication protocol. Native execution
errors retain normal worker error handling.

The fused and stock kernels can differ in accumulation order and intermediate
rounding. Shared projections retain FP16/BF16 activations on both paths;
activation quantization is not bypassed for a W8A8 backend. Bitwise equality or
full-model accuracy parity is not assumed. Correctness tests compare native
outputs against the stock block with numerical tolerances; model validation
also checks greedy outputs and logprob compatibility.

## Lifecycle and graphs

The FP8 experts post-load hook asks the common XPU provider to prepare routed
layouts during per-layer quantization, before checkpoint reload copies values
back into captured parameter storage. It defers provider binding until those
final parameter objects are restored. The model post-load hook then qualifies
static configuration and weights, prepares necessary contiguous model-owned
parameters, initializes the common provider, and binds the decode executor to
the same parameters. The provider owns the routed weight/scale relayout; the
fallback reuses that provider instead of changing parameter storage on its
first forward. Graph
capture and any other consumers binding parameter storage must run after this
preparation. Forward never constructs an executor or
replaces parameters.

`replace_parameter()` preserves parameter objects and reload metadata where
supported; initial layout preparation can change storage or strides. It must
therefore precede address binding. Subsequent in-place kernel-format reloads
are visible through the model-owned tensors, including captured graphs.
Storage, layout or eligibility changes require rebuilding the executor and
recapturing graphs.

The eager model supports graph-off execution and breakable XPU graphs.
`FULL_DECODE_ONLY` selects decode capture; breakable graphs describe the capture
mechanism, and can be used with that mode. This package uses model post-load
hooks and tensor metadata, with no Model Runner V2-only `ModelState` dependency.

## Layout

- `__init__.py`: package scope and token-row limits.
- `hooks.py`: `initialize_specialized_decode()` post-load entry point for the
  target and MTP models.
- `weights.py`: static qualification and model-owned layout preparation.
- `moe.py`: `FusedMoeDecode`, runtime eligibility, stock routing, native dispatch
  and the final TP reduction.
- `../model.py` and `../mtp.py`: eager model integration through post-load hooks
  and the MoE block's forward call.

Layer tests live in `tests/kernels/moe/test_moe.py`; model loading, graph reload
and backend fallback tests live in
`tests/models/qwen3_5/xpu_specialized_decode/test_integration.py`.
