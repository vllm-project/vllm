# MiniMax M3 MonoKernel

This directory contains model-specific stages plus a temporary local runtime.
The runtime mirrors the proposed `vllm.models.common.mono` interfaces and is
not attached to model forwards until the FFN viability gate passes.

Temporary, to delete or migrate when the common runtime lands:

- static refusal aggregation and final graph-width extension;
- rank-uniform per-step routing and counters;
- persistent cache/scale identity tracking and eager rebinding;
- peer-memory allocation and TP address exchange;
- bounded-wait sticky health/error state;
- compile-only warm-up key registration.

The BF16 attention, FP32 router, A4W4 expert, index-selection, and layer-stage
implementations remain MiniMax-specific.
