# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Meta kernels for vLLM's custom ops, so they run on `torch.device("meta")`.

`torch.device("meta")` records an operator and propagates its shapes without
computing anything. vLLM's custom ops have no `Meta` kernel, so a meta-device
forward pass dies on the first one. This module fills that gap in two tiers.

Glue ops are the device-agnostic Python registered through
`direct_register_custom_op` (`vllm::`, `vllm_ir::`, ...). Their bodies are
reused verbatim as meta kernels, so the compiled leaf kernels they dispatch to
still appear in a capture instead of being collapsed into one node.

Leaf ops are the compiled extensions ([`LEAF_NAMESPACES`]
[vllm.profiler.op_capture.meta_ops.LEAF_NAMESPACES]) and are never executed.
Their meta kernel comes from the schema -- nothing to return when the op only
mutates out-arguments, the aliased argument when the return aliases an input --
or from an [`OVERRIDES`][vllm.profiler.op_capture.meta_ops.OVERRIDES] entry. An
op with neither raises `UnsupportedMetaOpError` naming itself, rather than
guessing a shape. Only ops actually reached on the forward path can raise, so
the table holds the kernels the platforms vLLM has been captured on need, and
grows as new ones are covered.

Registrations only add the `Meta` key, so they are inert for real tensors, and
they never replace a kernel vLLM or PyTorch already registered.
"""

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch._C import FunctionSchema
from torch.library import Library

from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON
from vllm.utils.torch_utils import DIRECT_REGISTERED_OPS

logger = init_logger(__name__)

LEAF_NAMESPACES = frozenset(
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
    }
)
"""Namespaces of vLLM's compiled kernel extensions, whose bodies are never run."""

NATIVE_PREFIXES = ("aten::", "prim::", "prims::")
"""Prefixes of PyTorch's own operators, as opposed to vLLM's custom ops."""

MetaKernel = Callable[[dict[str, Any]], Any]
"""An override's signature: schema arguments by name, in to the op's outputs."""


def _first_tensor(values: Iterable[Any]) -> torch.Tensor | None:
    return next((value for value in values if isinstance(value, torch.Tensor)), None)


def _like_first_tensor(arguments: dict[str, Any]) -> torch.Tensor:
    first = _first_tensor(arguments.values())
    assert first is not None
    return torch.empty_like(first)


def _fa2_varlen_fwd(arguments: dict[str, Any]) -> list[torch.Tensor]:
    """`[out, softmax_lse]` for FlashAttention-2's varlen forward.

    `q` is `(total_q, num_heads, head_size)` and the log-sum-exp is kept per
    (head, query) in fp32, but only when asked for: the kernel otherwise leaves
    that return undefined. `out` is an optional pre-allocated buffer.
    """
    query = arguments["q"]
    total_q, num_heads, _ = query.shape
    out = arguments.get("out")
    lse_shape = (num_heads, total_q) if arguments.get("return_softmax") else ()
    return [
        torch.empty_like(query) if out is None else out,
        query.new_empty(lse_shape, dtype=torch.float32),
    ]


def _like_wrapper(op_name: str) -> MetaKernel:
    """Outputs of `vllm::<op_name>`, which only forwards to the kernel.

    On `meta` the wrapper runs its `register_fake` impl, so the shapes stay
    with the op's owner rather than being restated here.
    """
    return lambda arguments: getattr(torch.ops.vllm, op_name)(*arguments.values())


def _mhc_pre_outputs(residual: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """`(post_mix, comb_mix, layer_input)` for an mHC pre block.

    `residual` is `(..., hc_mult, hidden_size)`; both mixes are fp32.
    """
    *outer, hc_mult, hidden_size = residual.shape
    return (
        residual.new_empty((*outer, hc_mult, 1), dtype=torch.float32),
        residual.new_empty((*outer, hc_mult, hc_mult), dtype=torch.float32),
        residual.new_empty((*outer, hidden_size)),
    )


OVERRIDES: dict[str, MetaKernel] = {
    # Both hand back a view of their only tensor argument.
    "_C::weak_ref_tensor": _like_first_tensor,
    "_C::get_xpu_view_from_cpu_tensor": _like_first_tensor,
    # Repacks MXFP scales into the layout the kernel wants; same shape.
    "_moe_C::reorder_mxfp_scales": _like_first_tensor,
    "_vllm_fa2_C::varlen_fwd": _fa2_varlen_fwd,
    # Returns the output buffer it was handed.
    "_xpu_C::cutlass_grouped_gemm_interface": lambda arguments: arguments["ptr_D"],
    "_xpu_C::fp8_bmm": _like_wrapper("xpu_fp8_bmm"),
    "_xpu_C::fp8_mqa_logits": _like_wrapper("xpu_fp8_mqa_logits"),
    "_xpu_C::mhc_pre": lambda arguments: _mhc_pre_outputs(arguments["residual"]),
    # The residual after the post block, then the next pre block's outputs.
    "_xpu_C::mhc_post": lambda arguments: torch.empty_like(arguments["residual"]),
    "_xpu_C::mhc_fused_post_pre": lambda arguments: (
        torch.empty_like(arguments["residual"]),
        *_mhc_pre_outputs(arguments["residual"]),
    ),
    # Rotated query and key.
    "_xpu_C::deepseek_scaling_rope": lambda arguments: (
        torch.empty_like(arguments["query"]),
        torch.empty_like(arguments["key"]),
    ),
}
"""Hand-maintained meta kernels, keyed by qualified op name.

Holds the leaf ops whose output shape cannot be read off the schema. Add an
entry when a capture fails with `UnsupportedMetaOpError`, deriving the shape
from the op's call site in vLLM rather than from the kernel source.
"""


class UnsupportedMetaOpError(NotImplementedError):
    """Raised when an op on the forward path has no derivable meta kernel."""


TritonLaunchCallback = Callable[[str, tuple[Any, ...], dict[str, Any]], None]
"""Receives a skipped Triton launch: kernel name, then its arguments."""


def _refuse_launch(metadata: Any) -> None:
    raise UnsupportedMetaOpError(
        f"Triton kernel {metadata.data['name']!r} reached the device without "
        f"going through `kernel[grid](...)`, so the meta device cannot stand in "
        f"for it. Wrap its launch in a custom op with a fake impl."
    )


def _kernel_name(kernel: Any) -> str:
    from triton.runtime.jit import JITFunction

    # An autotuner or heuristics wrapper holds the kernel it launches in `fn`.
    while not isinstance(kernel, JITFunction) and hasattr(kernel, "fn"):
        kernel = kernel.fn
    return getattr(kernel, "__qualname__", type(kernel).__name__)


@contextmanager
def skip_meta_triton_launches(on_skip: TritonLaunchCallback) -> Iterator[None]:
    """Drop Triton kernel launches that are handed meta tensors.

    A Triton kernel called from Python bypasses the dispatcher, so no meta
    kernel can stand in for it: handed meta tensors, it would compile for and
    run on the accelerator with null pointers. A `kernel[grid](...)` launch with
    a meta tensor argument is instead reported to `on_skip` and dropped before
    it is autotuned or compiled. Triton kernels write into buffers their caller
    allocated, so no shape downstream changes. Any other launch that reaches
    the device raises `UnsupportedMetaOpError` before it is enqueued.

    Args:
        on_skip: Called with the kernel's name and arguments for each launch
            dropped.

    """
    if not HAS_TRITON:
        yield
        return
    from triton import knobs
    from triton.runtime.jit import KernelInterface

    bind_grid = KernelInterface.__getitem__

    def bind_grid_unless_meta(kernel: Any, grid: Any) -> Callable[..., Any]:
        launch = bind_grid(kernel, grid)

        def launch_unless_meta(*args: Any, **kwargs: Any) -> Any:
            values = (*args, *kwargs.values())
            if not any(isinstance(v, torch.Tensor) and v.is_meta for v in values):
                return launch(*args, **kwargs)
            on_skip(_kernel_name(kernel), args, kwargs)
            return None

        return launch_unless_meta

    previous_hook = knobs.runtime.launch_enter_hook
    KernelInterface.__getitem__ = bind_grid_unless_meta
    knobs.runtime.launch_enter_hook = _refuse_launch
    try:
        yield
    finally:
        KernelInterface.__getitem__ = bind_grid
        knobs.runtime.launch_enter_hook = previous_hook


def _qualified_name(schema: FunctionSchema) -> str:
    if not schema.overload_name:
        return schema.name
    return f"{schema.name}.{schema.overload_name}"


def _bind_arguments(
    schema: FunctionSchema, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Map a boxed call's arguments onto their schema argument names."""
    arguments = dict(zip((argument.name for argument in schema.arguments), args))
    arguments.update(kwargs)
    return arguments


def _aliased_argument(schema: FunctionSchema) -> str | None:
    """Name of the argument a single-tensor return aliases, if any."""
    if len(schema.returns) != 1:
        return None
    returned = schema.returns[0].alias_info
    if returned is None:
        return None
    for argument in schema.arguments:
        aliased = argument.alias_info
        if aliased is not None and set(aliased.after_set) & set(returned.after_set):
            return argument.name
    return None


def _make_leaf_kernel(schema: FunctionSchema) -> Callable[..., Any]:
    """Build the meta kernel for one compiled-extension op.

    Args:
        schema: Schema of the op to fake.

    Returns:
        A callable with the op's calling convention that produces its outputs
        without computing them, or one that raises `UnsupportedMetaOpError` when
        no shape can be derived.

    """
    qualified_name = _qualified_name(schema)

    override = OVERRIDES.get(qualified_name)
    if override is not None:
        return lambda *args, **kwargs: override(_bind_arguments(schema, args, kwargs))

    if not schema.returns:
        return lambda *args, **kwargs: None

    aliased = _aliased_argument(schema)
    if aliased is not None:
        return lambda *args, **kwargs: _bind_arguments(schema, args, kwargs)[aliased]

    returns = ", ".join(str(ret.type) for ret in schema.returns)

    def unsupported(*args: Any, **kwargs: Any) -> Any:
        raise UnsupportedMetaOpError(
            f"{qualified_name} returns ({returns}), whose shape is not "
            f"derivable from its schema, so it cannot run on the meta device. "
            f"Add an entry for it to vllm.profiler.op_capture.meta_ops."
            f"OVERRIDES."
        )

    return unsupported


_PORTABLE_DISPATCH_KEYS = (
    "CompositeExplicitAutograd",
    "CompositeExplicitAutogradNonFunctional",
    "CompositeImplicitAutograd",
)


def has_kernel_for(qualified_name: str, dispatch_key: str) -> bool:
    """Whether an op would find a kernel on a device with `dispatch_key`.

    A meta capture never runs the platform's kernels, so an op registered only
    for another backend is captured as if it existed. This tells the two apart.

    Args:
        qualified_name: Op name as recorded, e.g. `"_C::rms_norm"`.
        dispatch_key: The platform's dispatch key, e.g. `"XPU"`.

    Returns:
        Whether the op has a kernel for that key or a device-agnostic one.

    """
    return any(
        torch._C._dispatch_has_kernel_for_dispatch_key(qualified_name, key)
        for key in (dispatch_key, *_PORTABLE_DISPATCH_KEYS)
    )


_SCALAR_PLACEHOLDERS = {"int": 0, "SymInt": 0, "float": 0.0, "bool": False}


def placeholder_outputs(
    schema: FunctionSchema, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """Outputs of the declared types for an op whose output shapes are unknown.

    Every tensor output is shaped like the op's first tensor argument, so shapes
    downstream of a placeholder are guesses.

    Args:
        schema: Schema of the op.
        args: Its positional arguments.
        kwargs: Its keyword arguments.

    Returns:
        One placeholder per declared return, unpacked when there is one, or
        None when the op returns nothing.

    Raises:
        UnsupportedMetaOpError: If the op has no tensor argument to shape by, or
            returns a type with no placeholder.

    """
    like = _first_tensor((*args, *kwargs.values()))
    outputs: list[Any] = []
    for ret in schema.returns:
        return_type = str(ret.type)
        if like is not None and return_type in ("Tensor", "Tensor?"):
            outputs.append(torch.empty_like(like))
        elif return_type == "Tensor[]":
            outputs.append([])
        elif return_type in _SCALAR_PLACEHOLDERS:
            outputs.append(_SCALAR_PLACEHOLDERS[return_type])
        else:
            raise UnsupportedMetaOpError(
                f"No placeholder for {_qualified_name(schema)}, which returns "
                f"{return_type}"
            )
    if not outputs:
        return None
    return outputs[0] if len(outputs) == 1 else tuple(outputs)


_libraries: dict[str, Library] = {}
_registered: set[str] = set()


def register_meta_impls() -> frozenset[str]:
    """Register `Meta` kernels for vLLM custom ops that lack one.

    Idempotent, and additive: an op that already has a `Meta` kernel (a
    `register_fake` impl, or a decomposition) keeps it. Call this before
    building or running a model on the meta device.

    Returns:
        Qualified names of the ops registered in this process, including those
        from earlier calls.

    """
    for schema in torch._C._jit_get_all_schemas():
        namespace = schema.name.partition("::")[0]
        glue_impl = DIRECT_REGISTERED_OPS.get(schema.name)
        if glue_impl is None and namespace not in LEAF_NAMESPACES:
            continue
        qualified_name = _qualified_name(schema)
        if qualified_name in _registered:
            continue
        if torch._C._dispatch_has_kernel_for_dispatch_key(
            qualified_name, "Meta"
        ) or torch._C._dispatch_has_kernel_for_dispatch_key(
            qualified_name, "CompositeImplicitAutograd"
        ):
            continue

        kernel = glue_impl if glue_impl is not None else _make_leaf_kernel(schema)
        library = _libraries.setdefault(namespace, Library(namespace, "IMPL"))
        library.impl(qualified_name.partition("::")[2], kernel, "Meta")
        _registered.add(qualified_name)

    logger.debug("Registered %d meta kernels for vLLM custom ops", len(_registered))
    return frozenset(_registered)
