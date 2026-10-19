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
