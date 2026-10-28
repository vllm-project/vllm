# MiniMax M3 specialized decode

`--monokernel` on MI355X/gfx950 runs each sparse layer (3-59) as one persistent
FlyDSL launch (`mono/`), adapted from ATOM: input norm, FP8 QKV projection,
head norms and RoPE, KV/index cache updates, index scoring and selection,
sparse attention, FP8 output projection, both TP all-reduces, post-attention
norm, routing and the shared/routed experts. The kernels and runtime live in
vLLM, with no ATOM runtime dependency. See the rendered
[deployment documentation](../../../../../docs/design/attention_backends.md#minimax-m3-specialized-decode-on-mi355x)
for the FP8 projection configuration, draft quantization and evaluated recipe.

Supported deployments are deliberately narrow:

- TP4 on dedicated, unpartitioned GPUs (256 CUs, one CTA per CU);
- plain decode of 1-16 rows, eager or replayed from a FULL CUDA graph (graph
  padding rows are inert); PIECEWISE steps, larger and mixed steps run the
  stock layers;
- group-of-32 MXFP8 expert activations against `AITER_MXFP4_MXFP4` post-load
  weights, BF16 router weights/logits and rotary tables;
- dynamic per-token FP8 attention projections with per-channel weight scales.

The previous FFN-only MonoKernel has been removed. `--monokernel` selects
this full-layer implementation; there is no mode selector. Deployments must
configure the FP8 attention projections shown in the deployment documentation.

`runtime/` owns refusals, planning, compilation, launching, health and shutdown;
`mono/` owns the kernels and weight packing. Unsupported static deployments are
refused with all known reasons.

## Fail-stop

In-kernel waits are bounded (10 s) and a timeout or a host abort sets a sticky error word on every rank. The step's last launch
does not finish on any rank until every rank has published its own failure
flag (or the bound expired). After the layer loop of every forward (mono or
stock, eager or replayed from a FULL graph), before logits and sampling, a
device-side assert (`torch._assert_async`) checks that the word is clear. If
it is set, the GPU queue traps: none of the step's later work runs (sampling,
the output copy), the worker's wait for the step's output fails with
`hipErrorLaunchFailure`, the HSA runtime aborts the worker processes and the
engine stops.

Guarantee: no token is returned from a step any of whose launches failed on
any rank, nor from any later step. There is no host synchronization per step;
the check is two small kernels a forward (inside the FULL graph when one is
replayed). The cause (rank, epoch, site) stays in the device word: the worker
reports only the GPU queue exception. `test_host_abort_returns_no_tokens_from_full_graphs` checks this through
`LLM` in `test_layers.py`.

## ATOM provenance

The sparse-layer kernels are adapted from AMD's [ROCm/ATOM](https://github.com/ROCm/ATOM)
project at commit
[`e543f331c78028dfa6ed63987c4fae919734c898`](https://github.com/ROCm/ATOM/commit/e543f331c78028dfa6ed63987c4fae919734c898).
Credit for the fused-layer design, staged FlyDSL kernels, MXFP8 expert arithmetic
and shared kernel support belongs to the ATOM contributors. This is a port of
that work into vLLM's Model Runner V2 lifecycle.

| Local path | Upstream path at that commit |
| --- | --- |
| `mono/config.py`, `layout.py`, `kernels/` | [`atom/models/minimax_m3/mono/`](https://github.com/ROCm/ATOM/tree/e543f331c78028dfa6ed63987c4fae919734c898/atom/models/minimax_m3/mono) |
| `mono/support/` | [`atom/mono/`](https://github.com/ROCm/ATOM/tree/e543f331c78028dfa6ed63987c4fae919734c898/atom/mono) |

Each ported source file identifies its upstream path and retains AMD's
copyright notice. The original ATOM code is MIT-licensed; its complete,
unchanged [license](mono/LICENSE) is included. The vLLM integration uses
the project's Apache-2.0 license.

The vLLM adaptation adds native weight/cache binding, independent main/index
cache mappings, graph padding, persistent-grid residency checks, bounded
mailbox diagnostics and a step-end TP failure fence. Dense layers and prefill
continue to use native vLLM modules. ATOM is not an external runtime dependency.

## Toolchain

FlyDSL and AITER supply the existing compiler and primitive helpers. The
qualified environment uses FlyDSL 0.4.0 and AITER 0.1.26. Startup compiles every
supported width and checks that the persistent grid can reside on the GPU.

The CI suites follow the three test categories used by
[#61482](https://github.com/vllm-project/vllm/pull/61482):

- `test_decode.py`: CPU routing, startup refusals and cache lifecycle, selected
  by the MI355 Basic Models job.
- `test_numerics.py`: independent MXFP8 references, cache mappings, padding and
  the TP failure fence.
- `test_layers.py`: native-versus-MonoKernel layer states on the same seeded
  five-layer model, plus graph replay, accepted EAGLE3 verification and fail-stop
  through `LLM`.

The latter two suites run in the four-GPU MI355 job. The AMD pipeline also
selects each suite explicitly.
