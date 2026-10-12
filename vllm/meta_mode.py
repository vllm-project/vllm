# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for the meta-device CI/CD mode (``VLLM_META_MODE=1``).

This module holds ONLY small, self-contained helpers used by the meta-mode
integration. The behavioural changes live as explicit ``if envs.VLLM_META_MODE:``
branches inside the real vLLM classes (config, worker, model runner, layers);
vLLM's own classes (including the platform) are never monkeypatched. The
runtime patches below only neutralise third-party launch layers (Triton,
CuteDSL, ``torch.cuda`` primitives), which cannot be edited in-source. When
``VLLM_META_MODE`` is off none of this runs and there is zero impact (this
module is not even imported).

Goal: run the v1 engine end-to-end on a CPU-only box with zero real memory
(weights + KV cache on the ``meta`` device) for functional CI validation, no GPU.
Outputs are meaningless (meta tensors carry no data) -- this validates flow +
shapes, not numerical correctness.

Kernel handling: instead of stubbing the *callers* of GPU kernels, vLLM's opaque
custom ops (``vllm::unified_attention_with_output``, ``vllm::mamba_mixer``,
``vllm::qwen_gdn_attention_core``, ``vllm::moe_forward``, ...) dispatch to their
owner-provided fake kernels when one exists. :func:`register_meta_impls` fills
the remaining shape-safe gaps, for both compiled leaf ops and opaque
``vllm::`` ops, from operator schemas (no return or aliased return). Raw Triton
kernels that are NOT torch ops (e.g.
``fused_qk_rmsnorm_rope_gate``) cannot be given a ``torch.library`` meta kernel,
so in meta mode the Triton launcher is hijacked (:func:`install_launch_hooks`)
to skip the kernel launch (outputs are host pre-allocated, so shapes are
preserved).
"""

import contextlib
import functools
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch._C import FunctionSchema
from torch.library import Library

import vllm.envs as envs
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

# Constant token id emitted for every sampled position on meta.
META_TOKEN_ID = 100

_META_DERIVED_NAMESPACES = frozenset(
    {
        "_C",
        "_C_cache_ops",
        "_C_cuda_utils",
        "_C_custom_ar",
        "_flashkda_C",
        "_flashmla_C",
        "_flashmla_extension_C",
        "_moe_C",
        "_qutlass_C",
        "_rocm_C",
        "_vllm_fa2_C",
        "_vllm_fa3_C",
        "_xpu_C",
        # Opaque Python ops; mutate-only ones may omit a fake implementation.
        "vllm",
    }
)
_meta_libraries: dict[str, Library] = {}
_meta_registered: set[str] = set()


def _deepseek_v4_quant_insert(arguments: dict[str, Any]) -> torch.Tensor:
    q = arguments["q_in"]
    return q.new_empty((q.shape[0], arguments["q_head_padded"], q.shape[-1]))


def _fa3_fwd(arguments: dict[str, Any]) -> list[torch.Tensor]:
    q = arguments["q"]
    v = arguments["v"]
    out = arguments.get("out")
    if out is None:
        out = q.new_empty((q.shape[0], q.shape[1], v.shape[-1]))
    softmax_lse = q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32)
    return [out, softmax_lse, q.new_empty(0), q.new_empty(0)]


def _fa2_varlen_fwd(arguments: dict[str, Any]) -> list[torch.Tensor]:
    q = arguments["q"]
    v = arguments["v"]
    out = arguments.get("out")
    if out is None:
        out = q.new_empty((q.shape[0], q.shape[1], v.shape[-1]))
    softmax_lse = q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32)
    return [out, softmax_lse]


def _flashmla_sparse_prefill_fwd(arguments: dict[str, Any]) -> list[torch.Tensor]:
    q = arguments["q"]
    out = arguments.get("out_")
    if out is None:
        out = q.new_empty((q.shape[0], q.shape[1], arguments["d_v"]))
    max_logits = q.new_empty(q.shape[:2], dtype=torch.float32)
    lse = q.new_empty(q.shape[:2], dtype=torch.float32)
    return [out, max_logits, lse]


def _flashmla_sparse_decode_fwd(arguments: dict[str, Any]) -> tuple[Any, ...]:
    q = arguments["q"]
    out = arguments.get("out_")
    if out is None:
        out = q.new_empty((*q.shape[:3], arguments["d_v"]))
    batch_size, seq_len_q, num_heads_q = q.shape[:3]
    lse = q.new_empty((batch_size, num_heads_q, seq_len_q), dtype=torch.float32)
    return (
        out,
        lse,
        arguments.get("tile_scheduler_metadata"),
        arguments.get("num_splits"),
    )


_META_OVERRIDES: dict[str, Callable[[dict[str, Any]], Any]] = {
    "_C::fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert": (
        _deepseek_v4_quant_insert
    ),
    "_flashmla_C::sparse_decode_fwd": _flashmla_sparse_decode_fwd,
    "_flashmla_C::sparse_prefill_fwd": _flashmla_sparse_prefill_fwd,
    "_vllm_fa2_C::varlen_fwd": _fa2_varlen_fwd,
    "_vllm_fa3_C::fwd": _fa3_fwd,
}


def _qualified_name(schema: FunctionSchema) -> str:
    if not schema.overload_name:
        return schema.name
    return f"{schema.name}.{schema.overload_name}"


def _bind_arguments(
    schema: FunctionSchema, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    arguments = dict(zip((argument.name for argument in schema.arguments), args))
    arguments.update(kwargs)
    return arguments


def _aliased_argument(schema: FunctionSchema) -> str | None:
    if len(schema.returns) != 1 or schema.returns[0].alias_info is None:
        return None
    returned = schema.returns[0].alias_info
    for argument in schema.arguments:
        aliased = argument.alias_info
        if aliased is not None and set(aliased.after_set) & set(returned.after_set):
            return argument.name
    return None


def _schema_meta_kernel(schema: FunctionSchema) -> Callable[..., Any] | None:
    override = _META_OVERRIDES.get(_qualified_name(schema))
    if override is not None:
        return lambda *args, **kwargs: override(_bind_arguments(schema, args, kwargs))
    if not schema.returns:
        return lambda *args, **kwargs: None
    aliased = _aliased_argument(schema)
    if aliased is None:
        return None
    return lambda *args, **kwargs: _bind_arguments(schema, args, kwargs)[aliased]


def register_meta_impls() -> frozenset[str]:
    """Register shape-safe Meta kernels for vLLM custom ops that lack one.

    Compiled leaf ops and opaque ``vllm::`` ops are derived only when the schema
    is sufficient: no return, or a return aliasing an input. A small central
    override table covers required independent-output ops; other such ops keep
    their owner-provided fake implementation.
    """
    for schema in torch._C._jit_get_all_schemas():
        namespace = schema.name.partition("::")[0]
        if namespace not in _META_DERIVED_NAMESPACES:
            continue
        qualified_name = _qualified_name(schema)
        if qualified_name in _meta_registered:
            continue
        if torch._C._dispatch_has_kernel_for_dispatch_key(
            qualified_name, "Meta"
        ) or torch._C._dispatch_has_kernel_for_dispatch_key(
            qualified_name, "CompositeImplicitAutograd"
        ):
            continue
        kernel = _schema_meta_kernel(schema)
        if kernel is None:
            continue
        library = _meta_libraries.setdefault(namespace, Library(namespace, "IMPL"))
        library.impl(qualified_name.partition("::")[2], kernel, "Meta")
        _meta_registered.add(qualified_name)
    return frozenset(_meta_registered)


def fake_token_ids(sampled_token_ids: torch.Tensor) -> list[list[int]]:
    """Host token ids standing in for the value-less meta sampler output."""
    return torch.full(sampled_token_ids.shape, META_TOKEN_ID).tolist()


class MetaAsyncOutput(AsyncModelRunnerOutput):
    """Emulate GPU async output with a token-proportional deadline.

    ``GPUModelRunner.sample_tokens`` returns one in meta mode; the executor
    unwraps it via the blocking ``get_output()``. The deadline is set at creation
    (after the forward), so the engine's pre-output scheduling overlaps the
    simulated latency -- the same overlap the real async-output path gives.
    """

    def __init__(self, output: ModelRunnerOutput, scheduler_output: "SchedulerOutput"):
        self._output = output
        self._deadline = time.monotonic() + self._simulated_latency_s(scheduler_output)

    @staticmethod
    def _simulated_latency_s(scheduler_output: "SchedulerOutput") -> float:
        num_tokens = scheduler_output.total_num_scheduled_tokens
        return num_tokens * envs.VLLM_META_TOKEN_LATENCY_MS / 1000.0

    def get_output(self) -> ModelRunnerOutput:
        remaining = self._deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        return self._output


def _disable_triton_launch() -> None:
    """Hijack the Triton launcher so raw ``@triton.jit`` kernels become no-ops.

    Some fused kernels (e.g. ``fused_qk_rmsnorm_rope_gate`` for Qwen3-Next full
    attention) are launched directly as ``kernel[grid](...)`` rather than through
    a torch custom op, so they cannot be given a ``torch.library`` meta kernel.
    On a CPU/meta box the Triton driver has no active device ("0 active
    drivers"). These kernels write in place into host pre-allocated (meta)
    tensors, so skipping the launch preserves output shapes and only avoids the
    GPU compute -- model-agnostically, one layer below the model.
    """
    import triton.runtime.autotuner as _autotuner
    import triton.runtime.jit as _jit

    _jit.JITFunction.run = lambda self, *a, **k: None
    # Autotuners query the driver before reaching the wrapped JIT function.
    _autotuner.Autotuner.run = lambda self, *a, **k: None


def _disable_cutedsl_launch() -> None:
    """Hijack the CuteDSL (cutlass.cute) JIT so its kernels become no-ops.

    Same idea as :func:`_disable_triton_launch`, one layer below the model, for
    the CuteDSL kernels used by DeepSeek-V4 / Kimi-K3 (e.g. the sparse-attn
    compressor, latent MoE). ``cute.compile(kernel, *sample_args)`` returns a
    compiled launcher that is later called as ``compiled(*tensors)``; both the
    compile and the launch touch a real GPU (tvm_ffi cannot even pack meta
    tensors). The callers pre-allocate their output/cache tensors (on meta) and
    write in place, so returning a no-op launcher preserves shapes and skips the
    GPU work. No-op if cutlass isn't installed.
    """
    try:
        import cutlass.cute as cute
    except Exception:
        return

    def _noop_compile(*args, **kwargs):
        def _noop_launch(*a, **k):
            return None

        return _noop_launch

    cute.compile = _noop_compile


class _NullCudaEvent:
    """No-op stand-in for ``torch.cuda.Event`` / ``torch.Event`` on meta.

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
    """No-op stand-in for ``torch.cuda.Stream`` / ``torch.Stream`` on meta.

    Like :class:`_NullCudaEvent`, overlap paths that gate on ``stream is not
    None`` run their production branch (side streams, async copies, DBO) with
    ordering calls reduced to no-ops. The torch entry points that would bind
    the CUDA runtime for a non-None stream (``torch.cuda.stream``,
    ``set_stream``, ``current_stream``, ``Tensor.record_stream``) are
    neutralised alongside in :func:`_patch_cuda_primitives`.
    """

    stream_id = 0
    device_index = 0
    device_type = 0
    cuda_stream = 0
    priority = 0
    device = torch.device("meta")

    def __init__(self, *args, **kwargs) -> None:
        pass

    def wait_stream(self, *args, **kwargs) -> None:
        pass

    def wait_event(self, *args, **kwargs) -> None:
        pass

    def record_event(self, event=None) -> _NullCudaEvent:
        return _NullCudaEvent() if event is None else event

    def synchronize(self) -> None:
        pass

    def query(self) -> bool:
        return True

    def is_capturing(self) -> bool:
        return False

    def __enter__(self) -> "_NullCudaStream":
        return self

    def __exit__(self, *args) -> None:
        pass


# The stream reported as current/default; side streams are fresh instances.
_CURRENT_STREAM = _NullCudaStream()


def _null_stream_context(stream=None) -> contextlib.nullcontext:
    return contextlib.nullcontext(stream)


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


def _patch_cuda_primitives() -> None:
    """Hijack ``torch.cuda`` primitives and ``torch.accelerator.synchronize`` so
    unconditional GPU-primitive use is meta-safe (one layer below the model, like
    the Triton/CuteDSL launch no-ops).
    """
    torch.cuda.Event = _NullCudaEvent
    # Device-generic events (``torch.Event()``) would bind the CUDA runtime.
    torch.Event = _NullCudaEvent
    torch.cuda.Stream = _NullCudaStream
    torch.Stream = _NullCudaStream
    current_stream = lambda *args, **kwargs: _CURRENT_STREAM  # noqa: E731
    torch.cuda.current_stream = current_stream
    torch.cuda.default_stream = current_stream
    torch.accelerator.current_stream = current_stream
    # Stream switching is pure bookkeeping on meta. ``torch.cuda.set_stream``
    # stays wrapped by vLLM's tracker; only its CUDA-runtime tail is dropped.
    torch.cuda.stream = _null_stream_context
    torch.cuda.StreamContext = _null_stream_context
    torch.cuda._set_stream_by_id = lambda *args, **kwargs: None
    torch.accelerator.set_stream = lambda *args, **kwargs: None
    # Allocator lifetime hints have no allocator to inform on meta.
    torch.Tensor.record_stream = lambda self, *args, **kwargs: None
    torch.cuda.set_device = lambda *args, **kwargs: None
    torch.cuda.get_device_properties = lambda *args, **kwargs: (
        _NullCudaDeviceProperties()
    )
    # Without a visible device the real query raises instead of returning False.
    torch.cuda.is_current_stream_capturing = lambda: False
    # Layers place small buffers via current_platform.current_device() (resolved
    # to torch.cuda.current_device); route them to the meta device.
    torch.cuda.current_device = lambda: torch.device("meta")
    # No accelerator on meta; synchronize is a no-op and the device index is
    # the rank-local 0 (else both would init CUDA).
    torch.accelerator.synchronize = lambda *args, **kwargs: None
    torch.accelerator.current_device_index = lambda: 0


@functools.cache
def install_launch_hooks() -> None:
    """Neutralise third-party GPU launch layers for meta execution. Idempotent."""
    _disable_triton_launch()
    _disable_cutedsl_launch()
    _patch_cuda_primitives()
