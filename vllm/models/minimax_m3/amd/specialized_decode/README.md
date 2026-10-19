# MiniMax M3 specialized decode

This package contains the gfx950 layer-mode specialized decode selected by the
project-wide `--monokernel` switch. Configuration selection is side-effect
free; the executor is built after weight loading and caches are bound through
the model state.

Milestone one is deliberately narrow:

- MI355X/gfx950 with all 256 CUs and dedicated GPU residency;
- TP4, eager, plain decode widths 1/2/4/8/16;
- layers 3-59 and the FFN half only;
- BF16 expert activations with AITER-shuffled MXFP4 weights and E8M0 scales.

`runtime/` owns refusals, planning, cache identity, health, and shutdown.
`mono/` owns only kernel code. Unsupported static deployments are refused with
all known reasons; unsupported steps run stock. Once a specialized launch
starts, any reported failure stops the step before logits can be sampled.

The FFN executor is not attached yet. The previous compile-only prototype used
an A4W4 kernel and therefore did not match the effective A16W4 contract. The
GLM-5.2 substrate is a source for its tested MXFP4 operand mapping and bounded
peer protocol, but its TP8, top-8, 256-expert geometry is not directly reusable.
`mono.viability.FfnViabilityEvidence` requires a MiniMax-shaped post-link
resource report, one resident CTA per CU with no spills, and measured eager
real-layer speedup before an executor may be selected.
