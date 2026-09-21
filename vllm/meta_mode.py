# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for the meta-device CI/CD mode (``VLLM_META_MODE=1``).

This module holds ONLY small, self-contained helpers used by the meta-mode
integration. The behavioural changes live as explicit ``if envs.VLLM_META_MODE:``
branches inside the real vLLM classes (config, worker, model runner, layers),
plus a platform wrapper mixin (see ``vllm/platforms/meta.py``); vLLM's own
classes are never monkeypatched. The one runtime patch is ``disable_triton_launch``
below, which neutralises the third-party Triton launcher (not vLLM code, so it
cannot be edited in-source). When ``VLLM_META_MODE`` is off none of this runs and
there is zero impact (this module is not even imported).

Goal: run the v1 engine end-to-end on a CPU-only box with zero real memory
(weights + KV cache on the ``meta`` device) for functional CI validation, no GPU.
Outputs are meaningless (meta tensors carry no data) -- this validates flow +
shapes, not numerical correctness.

Kernel handling: instead of stubbing the *callers* of GPU kernels, vLLM's opaque
custom ops (``vllm::unified_attention_with_output``, ``vllm::mamba_mixer``,
``vllm::qwen_gdn_attention_core``, ``vllm::moe_forward``, ...) dispatch to their
already-registered ``register_fake`` (meta) kernels; the few direct
``torch.ops._C.*`` ops that lacked one now provide a ``register_fake`` at their
definition site in ``vllm/_custom_ops.py``. Raw Triton kernels that are NOT torch
ops (e.g. ``fused_qk_rmsnorm_rope_gate``) cannot be given a ``torch.library``
meta kernel, so in meta mode the Triton launcher (``disable_triton_launch``) is
hijacked to skip the kernel launch (outputs are host pre-allocated, so shapes are
preserved).
"""

import time

import torch

import vllm.envs as envs
from vllm.v1.outputs import AsyncModelRunnerOutput

# Logical tensor-parallel size to emulate (captured from the requested
# --tensor-parallel-size in ParallelConfig.__post_init__ before the real world
# size is collapsed to 1). Read by the worker to fake the TP/EP group.
_logical_tp = 1
# Logical decode-context-parallel size to emulate (captured before collapse),
# used to fake the DCP group so the CP slot-mapping path is exercised.
_logical_dcp = 1
_triton_disabled = False


def set_logical_tp(tp_size: int) -> None:
    global _logical_tp
    _logical_tp = tp_size


def get_logical_tp() -> int:
    return _logical_tp


def set_logical_dcp(dcp_size: int) -> None:
    global _logical_dcp
    _logical_dcp = dcp_size


def get_logical_dcp() -> int:
    return _logical_dcp



def token_latency_s() -> float:
    return envs.VLLM_META_TOKEN_LATENCY_MS / 1000.0



class MetaAsyncOutput(AsyncModelRunnerOutput):
    """Emulate GPU async output with a token-proportional deadline.

    ``GPUModelRunner.sample_tokens`` returns one in meta mode; the executor
    unwraps it via the blocking ``get_output()``. The deadline is set at creation
    (after the forward), so the engine's pre-output scheduling overlaps the
    simulated latency -- the same overlap the real async-output path gives.
    """

    def __init__(self, output, delay_s: float):
        self._output = output
        self._deadline = time.monotonic() + delay_s

    def get_output(self):
        remaining = self._deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        return self._output


def enable_fake_tp(tp_size: int) -> None:
    """Emulate a size-``tp_size`` TP group inside a single process.

    ``get_tensor_model_parallel_world_size()`` reads ``get_tp_group().world_size``,
    so faking it makes every layer shard its weights by ``tp_size`` at build
    time (and, with ``enable_expert_parallel``, derives ``ep_size`` from it).
    The collectives route through this group object; replace them with
    shape-correct stubs since meta tensors carry no data to communicate.
    """
    if tp_size <= 1:
        return
    from vllm.distributed.parallel_state import get_tp_group

    g = get_tp_group()
    g.world_size = tp_size
    g.rank_in_group = 0
    g.all_reduce = lambda x, *a, **k: x
    g.all_gather = lambda x, dim=-1, *a, **k: torch.cat([x] * tp_size, dim=dim)
    g.gather = lambda x, dst=0, dim=-1, *a, **k: torch.cat([x] * tp_size, dim=dim)


def enable_fake_dcp(dcp_size: int, rank: int = 0) -> None:
    """Emulate a size-``dcp_size`` DCP group in one process, so the CP
    slot-mapping path (``BlockTable`` reads ``get_dcp_group().world_size`` /
    ``rank_in_group``) is exercised.

    Like the TP fake, only sets the reported size/rank + shape-stub collectives
    (no real cross-rank data). KV is sized for the collapsed dcp==1, so this
    validates the CP index math, not CP memory sharding.
    """
    if dcp_size <= 1:
        return
    from vllm.distributed import get_dcp_group

    g = get_dcp_group()
    g.world_size = dcp_size
    g.rank_in_group = rank
    g.all_reduce = lambda x, *a, **k: x
    g.all_gather = lambda x, dim=-1, *a, **k: torch.cat([x] * dcp_size, dim=dim)
    g.gather = lambda x, dst=0, dim=-1, *a, **k: torch.cat([x] * dcp_size, dim=dim)



def disable_triton_launch() -> None:
    """Hijack the Triton launcher so raw ``@triton.jit`` kernels become no-ops.

    Some fused kernels (e.g. ``fused_qk_rmsnorm_rope_gate`` for Qwen3-Next full
    attention) are launched directly as ``kernel[grid](...)`` rather than through
    a torch custom op, so they cannot be given a ``torch.library`` meta kernel.
    On a CPU/meta box the Triton driver has no active device ("0 active
    drivers"). These kernels write in place into host pre-allocated (meta)
    tensors, so skipping the launch preserves output shapes and only avoids the
    GPU compute -- model-agnostically, one layer below the model. Idempotent.
    """
    global _triton_disabled
    if _triton_disabled:
        return
    _triton_disabled = True

    import triton.runtime.jit as _jit

    _jit.JITFunction.run = lambda self, *a, **k: None
    try:
        from triton.runtime.interpreter import InterpretedFunction

        InterpretedFunction.run = lambda self, *a, **k: None
    except Exception:
        pass


_cutedsl_disabled = False


def disable_cutedsl_launch() -> None:
    """Hijack the CuteDSL (cutlass.cute) JIT so its kernels become no-ops.

    Same idea as :func:`disable_triton_launch`, one layer below the model, for
    the CuteDSL kernels used by DeepSeek-V4 / Kimi-K3 (e.g. the sparse-attn
    compressor, latent MoE). ``cute.compile(kernel, *sample_args)`` returns a
    compiled launcher that is later called as ``compiled(*tensors)``; both the
    compile and the launch touch a real GPU (tvm_ffi cannot even pack meta
    tensors). The callers pre-allocate their output/cache tensors (on meta) and
    write in place, so returning a no-op launcher preserves shapes and skips the
    GPU work. Idempotent; no-op if cutlass isn't installed.
    """
    global _cutedsl_disabled
    if _cutedsl_disabled:
        return
    try:
        import cutlass.cute as cute
    except Exception:
        return
    _cutedsl_disabled = True

    def _noop_compile(*args, **kwargs):
        def _noop_launch(*a, **k):
            return None

        return _noop_launch

    cute.compile = _noop_compile


_cuda_primitives_patched = False


class _NullCudaEvent:
    """No-op stand-in for ``torch.cuda.Event`` on meta.

    Events only sequence GPU streams; on a GPU-free box callers (e.g. the
    DeepSeek-V4 / Kimi-K3 overlap bookkeeping) get a harmless object with no-op
    record/wait/synchronize, so those layers need no per-site meta guards.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def record(self, *args, **kwargs) -> None:
        pass

    def wait(self, *args, **kwargs) -> None:
        pass

    def synchronize(self, *args, **kwargs) -> None:
        pass

    def query(self, *args, **kwargs) -> bool:
        return True

    def elapsed_time(self, *args, **kwargs) -> float:
        return 0.0


class _NullCudaStream:
    """``torch.cuda.Stream()`` -> ``None`` on meta.

    Overlap paths gate on ``stream is not None`` (falling back to sequential),
    and ``with torch.cuda.stream(None)`` is a documented no-op, so None disables
    overlap without per-site guards. A class (not a lambda) so annotations like
    ``torch.cuda.Stream | None`` still evaluate.
    """

    def __new__(cls, *args, **kwargs):
        return None


class _NullCudaDeviceProperties:
    """No-GPU stand-in for ``torch.cuda.get_device_properties(...)``.

    The real call needs a CUDA context; on meta only the handful of fields that
    host-side kernel-tiling code reads are needed (dominant one:
    ``multi_processor_count``, used e.g. by the mHC ``compute_num_split``). The
    values are irrelevant to correctness since the kernels themselves are
    no-op'd -- they only need to exist so the host math doesn't crash.
    """

    name = "meta"
    major = 9
    minor = 0
    multi_processor_count = 1
    total_memory = 0
    is_integrated = 0
    is_multi_gpu_board = 0
    warp_size = 32
    uuid = "meta"


def patch_cuda_primitives() -> None:
    """Hijack ``torch.cuda.Event``/``Stream``/``get_device_properties`` and
    ``torch.accelerator.synchronize`` so the model code's unconditional
    GPU-primitive use is meta-safe (one layer below the model, like the
    Triton/CuteDSL launch no-ops). Idempotent.
    """
    global _cuda_primitives_patched
    if _cuda_primitives_patched:
        return
    _cuda_primitives_patched = True

    torch.cuda.Event = _NullCudaEvent
    torch.cuda.Stream = _NullCudaStream
    torch.cuda.get_device_properties = lambda *args, **kwargs: _NullCudaDeviceProperties()
    # No accelerator on meta; synchronize is a no-op (else it would init CUDA).
    torch.accelerator.synchronize = lambda *args, **kwargs: None
