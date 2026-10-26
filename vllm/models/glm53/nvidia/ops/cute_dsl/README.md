# GLM-5.3 CuTe DSL kernel families

This directory holds the GLM-5.3 decode kernels as CuTe DSL modules, one per
family. `registry.py` maps each served entry to its kernel class and derives its launch facts.
Each entry's `@cute.jit` host wrapper (which also builds its TMA descriptors) is compiled
once and kept as an object file under `$VLLM_CACHE_ROOT/glm53_kernels`; later processes load it.

Shared code lives in one file, `_rt.py`. Part 1 holds the PTX primitives as namespace classes
(`Core` with the two inline-asm emitters, `FastMath`, `Convert`, `Sync`, `Memory`, `AsyncCopy`,
`TensorCore`); part 2 the components the families repeat (`Prologue`, `TaggedMailbox`, `Pipeline.Ring`,
`Reduce`, `Mma`); part 3 the served-kernel skeleton (`SmemLayout`/`SmemAlloc`, `DecodeKernel`/`RoleKernel`,
params records). Call sites read `rt.Memory.ld_volatile_v4(...)` or `rt.TaggedMailbox.push(...)`; no
family module defines its own PTX helper. Every family imports only from `_rt.py`, except that the
worker's MoE prefetch reads `fused_moe.py`'s expert layout.

The served math is flush-to-zero: every f32 add, mul, fma, compare, min and max is `.ftz`.

The index-key cache stores each key's scale as four equal UE8M0 bytes (vLLM's SM100/SM103
format, `mqa_logits_use_mx_scales()`); the selector reads and writes that form (`IndexScale` in
`mla_selector.py`).

## Startup

A cold compile takes 0.06-0.08 s per small entry, 0.73 s for
dense_ffn, 1.8-1.9 s for fused_moe, 1.0-3.3 s for mla_worker, and 5.4-6.2 s for mla_selector,
on every rank in parallel. The drafter's entries (`mla_worker_draft_q{1,4}_paged`,
`mtp_preprocess_q{1,4}`, `lm_head_top1_q1`, `prologue_vllm`) compile in the same build.
