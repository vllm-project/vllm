# MiniMax M3 specialized decode

`--monokernel` on MI355X/gfx950 runs each sparse layer (3-59) as one persistent
FlyDSL launch (`mono/layer/`), adapted from ATOM: input norm, FP8 QKV projection,
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

`--kernel-config '{"monokernel_mode":"ffn"}'` selects the legacy kernel
(`mono/ffn.py`), which leaves input norm and attention on native modules and
uses BF16 expert activations and an FP32 router.

`runtime/` owns refusals, planning, health and shutdown; `mono/` owns the
kernel and its host launcher. Unsupported static deployments are refused with
all known reasons.

## Fail-stop

In-kernel waits are bounded (10 s) and a timeout, a host abort or a failed
epoch reset sets a sticky error word on every rank. The step's last launch
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
reports only the GPU queue exception. `test_failstop.py` checks that the
assert traps on the installed torch build.

## Toolchain

The hand-offs (`mono/sync.py`) rely on LLVM's AMDGPU memory model
(`llvm/docs/AMDGPUMemoryModel.md`: store-available / load-visible operations
and the `amdgcn-av` metadata) as implemented by the LLVM bundled with FlyDSL
0.3.4.1: upstream llvm-project
[`e2a39f504fee`](https://github.com/llvm/llvm-project/commit/e2a39f504fee836e4def9581bed817ecc327b9dc)
(FlyDSL `thirdparty/llvm-build-info.json`; the wheel reports
`24.0.0git e2a39f504fee...`). FlyDSL comes in through AITER (`v0.1.25` in
`docker/Dockerfile.rocm_base`). Like DS V4.1, startup does not check the
FlyDSL version; `test_ops.py` checks the gfx950 lowering of the linked
kernel instead.
