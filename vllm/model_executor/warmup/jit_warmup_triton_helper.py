# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import inspect
from abc import abstractmethod
from collections.abc import Callable, Hashable, Iterable, Mapping
from contextlib import nullcontext, suppress
from dataclasses import dataclass, field
from functools import cache, cached_property, update_wrapper, wraps
from typing import Any, Generic, ParamSpec, Protocol, TypeVar, cast, overload

from vllm.model_executor.warmup.jit_warmup import (
    VllmJitKernel,
    get_ast_full_name,
    get_function_source_node,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton

CompileKeyT = TypeVar("CompileKeyT")
P = ParamSpec("P")
# ``kernel_launcher`` also supports a ``None`` grid when no launch is needed.
LaunchSpec = (
    tuple[tuple[int, ...] | None, dict[str, Any]]
    | tuple[tuple[int, ...] | None, dict[str, Any], Any]
)
# Dispatchers select a concrete launch configuration for a Triton kernel.
DispatchSpec = (
    tuple[tuple[int, ...], dict[str, Any]] | tuple[tuple[int, ...], dict[str, Any], Any]
)


@dataclass(frozen=True)
class _LaunchBindingPlan:
    input_targets: tuple[tuple[int, str], ...]
    stride_targets: tuple[tuple[str, str, int], ...]


@cache
def _launch_binding_plan(
    arg_names: tuple[str, ...],
    input_names: tuple[str, ...],
    launch_names: tuple[str, ...],
) -> _LaunchBindingPlan:
    arg_name_set = set(arg_names)
    launch_name_set = set(launch_names)

    def kernel_arg(input_name: str) -> str | None:
        pointer_name = f"{input_name}_ptr"
        candidates = (
            input_name,
            pointer_name,
            input_name.upper(),
            pointer_name.upper(),
        )
        return next(
            (candidate for candidate in candidates if candidate in arg_name_set),
            None,
        )

    input_targets = tuple(
        (index, target)
        for index, name in enumerate(input_names)
        if (target := kernel_arg(name)) is not None and target not in launch_name_set
    )
    bound_names = launch_name_set | {target for _, target in input_targets}
    stride_targets = []
    for target in arg_names:
        if target in bound_names:
            continue
        name, separator, dim = target.rpartition("_stride")
        if not separator:
            name, separator, dim = target.rpartition("_STRIDE")
        dim = dim.removeprefix("_")
        if separator and (not dim or dim.isdigit()):
            pointer = f"{name}_PTR" if separator == "_STRIDE" else f"{name}_ptr"
            source = name if name in bound_names else pointer
            if source in bound_names:
                stride_targets.append((target, source, int(dim or 0)))
    return _LaunchBindingPlan(input_targets, tuple(stride_targets))


def triton_warmup_inputs(
    kernel: Any,
    *args: Any,
    grid: tuple[int, ...],
    pointer_dtypes: (
        Mapping[Any, Iterable[str]] | Iterable[tuple[Any, Iterable[str]]] | None
    ) = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build launcher inputs from Triton's native positional argument order."""
    arg_names = tuple(kernel.arg_names)
    if len(args) > len(arg_names):
        raise ValueError(
            f"Received {len(args)} positional inputs for {len(arg_names)} "
            "Triton kernel arguments"
        )
    inputs = dict(zip(arg_names, args))
    duplicate_names = inputs.keys() & kwargs.keys()
    if duplicate_names:
        raise ValueError(
            f"Triton inputs passed twice: {', '.join(sorted(duplicate_names))}"
        )
    inputs.update(kwargs)
    if pointer_dtypes is not None:
        function_def = get_function_source_node(kernel)
        if not isinstance(function_def, ast.FunctionDef):
            raise ValueError("Expected Triton kernel to be defined as a function")
        pointer_names = _pointer_arg_names(function_def, arg_names)
        groups = (
            pointer_dtypes.items()
            if isinstance(pointer_dtypes, Mapping)
            else pointer_dtypes
        )
        for dtype, names in groups:
            for name in names:
                if name not in arg_names:
                    raise ValueError(f"Unknown Triton pointer argument: {name}")
                if name not in pointer_names:
                    raise ValueError(f"Triton argument is not a pointer: {name}")
                if name in inputs:
                    raise ValueError(f"Triton input passed twice: {name}")
                inputs[name] = TritonWarmupTensor(dtype)
        missing_pointers = pointer_names - inputs.keys()
        if missing_pointers:
            names = ", ".join(sorted(missing_pointers))
            raise ValueError(f"Missing Triton pointer inputs: {names}")
    return {"grid": grid, **inputs}


def triton_scalar_specialization_rep(value: int) -> int:
    """Return an integer with the same default Triton JIT specialization.

    For an ordinary integer argument, Triton's cache key contains its inferred
    type (``i32``, ``i64``, or ``u64``) and one of three value classes:

    * ``1`` is specialized as the exact constant ``1``.
    * Multiples of 16 receive a ``tt.divisibility = 16`` attribute.
    * All other values have no value specialization.

    Warmup only needs one concrete value for each cache-key class. This helper
    returns ``1`` for the exact-one class and otherwise returns a divisible or
    generic representative while preserving the inferred integer type.

    This applies only to non-``constexpr`` integer arguments using Triton's
    default specialization. Do not use it for arguments listed in
    ``do_not_specialize`` or ``do_not_specialize_on_alignment``.
    """
    if value == 1:
        return 1

    if -(1 << 31) <= value < (1 << 31):
        divisible_rep = 16
        generic_rep = 2
    elif -(1 << 63) <= value < (1 << 63):
        divisible_rep = 1 << 31
        generic_rep = (1 << 31) + 1
    elif 0 <= value < (1 << 64):
        divisible_rep = 1 << 63
        generic_rep = (1 << 63) + 1
    else:
        raise OverflowError(f"Integer {value} is outside Triton's scalar range")

    return divisible_rep if value % 16 == 0 else generic_rep


@dataclass(frozen=True)
class TritonWarmupTensor:
    """Compile-only tensor metadata used by Triton warmup.

    ``strides=None`` represents compact row-major storage. Pass explicit strides
    whenever the runtime tensor can be padded, transposed, or otherwise strided.
    """

    dtype: Any
    aligned: bool = True
    shape: tuple[int, ...] = (1,)
    strides: tuple[int, ...] | None = None
    init: Any = 0

    def data_ptr(self) -> int:
        return 0 if self.aligned else 1

    def ptr_range(self) -> int:
        return 0

    @property
    def device(self) -> Any:
        import torch

        return torch.device(current_platform.device_type)

    def stride(self, dim: int | None = None) -> int | tuple[int, ...]:
        if self.strides is None:
            strides: list[int] = []
            stride = 1
            for size in reversed(self.shape):
                strides.append(stride)
                stride *= size
            result = tuple(reversed(strides))
        else:
            result = self.strides
        return result if dim is None else result[dim]


class VllmTritonJitKernel(VllmJitKernel[CompileKeyT], Generic[CompileKeyT]):
    """Triton owner whose runtime launch specification is reused for warmup."""

    kernel: Any
    _warming = False
    _warming_compile_key: CompileKeyT | None = None
    _run_autotune = False

    @abstractmethod
    def warmup_inputs(self, compile_key: CompileKeyT) -> dict[str, Any]:
        """Return runtime-shaped inputs that reproduce one compile key."""
        raise NotImplementedError

    def compile(self, compile_key: CompileKeyT) -> None:
        inputs = self.warmup_inputs(compile_key)
        self._warming = True
        self._warming_compile_key = compile_key
        try:
            cast(Callable[..., None], self)(**inputs)
        finally:
            self._warming = False
            self._warming_compile_key = None

    @cached_property
    def _kernel_arg_names(self) -> tuple[str, ...]:
        arg_names = getattr(self.kernel, "arg_names", None)
        if arg_names is not None:
            return tuple(arg_names)
        wrapped = getattr(self.kernel, "func", None)
        if wrapped is not None:
            return tuple(inspect.signature(wrapped).parameters)
        raise TypeError(
            f"Cannot inspect kernel parameters for {type(self.kernel).__name__}"
        )

    def _prepare_launch_kwargs(
        self,
        input_names: tuple[str, ...],
        input_values: tuple[Any, ...],
        launch_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        plan = _launch_binding_plan(
            self._kernel_arg_names, input_names, tuple(launch_kwargs)
        )
        for index, target in plan.input_targets:
            launch_kwargs[target] = input_values[index]
        for target, source, dim in plan.stride_targets:
            launch_kwargs[target] = launch_kwargs[source].stride(dim)
        return launch_kwargs

    def launch(
        self,
        grid: tuple[int, ...],
        input_names: tuple[str, ...],
        input_values: tuple[Any, ...],
        /,
        _runtime_launcher: Any = None,
        _runtime_launcher_arg_count: int = 0,
        **kwargs: Any,
    ) -> Any:
        kwargs = self._prepare_launch_kwargs(input_names, input_values, kwargs)
        if self._warming:
            if self._run_autotune:
                return self.kernel[grid](**kwargs)
            kwargs = {
                name: _triton_metadata_arg(value) for name, value in kwargs.items()
            }
            if (
                self._warming_compile_key is not None
                and "launch_pdl" in kwargs
                and hasattr(self._warming_compile_key, "launch_pdl")
            ):
                kwargs["launch_pdl"] = self._warming_compile_key.launch_pdl
            warmup = getattr(self.kernel, "warmup", None)
            assert warmup is not None
            return warmup(grid=(1,), **kwargs)
        if _runtime_launcher is not None:
            regular_args = [
                kwargs.pop(name)
                for name in self._kernel_arg_names[:_runtime_launcher_arg_count]
            ]
            return _runtime_launcher(self.kernel, grid, *regular_args, **kwargs)
        return self.kernel[grid](**kwargs)


def kernel_launcher(
    call_fn: Callable[..., LaunchSpec],
) -> Callable[..., Any]:
    """Launch a Triton kernel from a declarative ``__call__`` specification.

    ``call_fn`` returns either ``(grid, launch_kwargs)`` or, when it allocates
    its own outputs, ``(grid, launch_kwargs, outputs)``. The declared outputs are
    returned to the caller.
    """
    signature = inspect.signature(call_fn)

    @wraps(call_fn)
    def wrapper(
        self: VllmTritonJitKernel[Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        spec = call_fn(self, *args, **kwargs)
        if len(spec) == 3:
            grid, launch_kwargs, outputs = spec
        else:
            grid, launch_kwargs = spec
            outputs = None
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        inputs = {
            name: value for name, value in bound.arguments.items() if name != "self"
        }
        if grid is not None:
            self.launch(grid, tuple(inputs), tuple(inputs.values()), **launch_kwargs)
        return outputs

    return wrapper


@dataclass(frozen=True)
class TritonJitKey:
    """Process-local identity of one Triton JIT specialization."""

    jit_function_id: int
    jit_function_key: Hashable
    device: Hashable
    cache_key: Hashable


@dataclass(frozen=True)
class TritonCompileKey:
    """Triton-derived compile key with one non-comparing replay input."""

    jit_keys: frozenset[TritonJitKey]
    inputs: tuple[tuple[str, Any], ...] = field(compare=False, hash=False, repr=False)


def _triton_key_deriver(
    kernel: Any,
) -> Callable[[Mapping[str, Any]], set[TritonJitKey]]:
    """Prepare Triton's key derivation once for one kernel and device."""
    knobs = triton.knobs
    Autotuner = triton.runtime.autotuner.Autotuner
    Heuristics = triton.runtime.autotuner.Heuristics
    driver = triton.runtime.driver.driver
    JITFunction = triton.runtime.jit.JITFunction
    compute_cache_key = triton.runtime.jit.compute_cache_key

    if isinstance(kernel, Heuristics):
        derive_inner = _triton_key_deriver(kernel.fn)

        def derive_heuristic(kwargs: Mapping[str, Any]) -> set[TritonJitKey]:
            heuristic_kwargs = dict(kwargs)
            for name, heuristic in kernel.values.items():
                heuristic_kwargs[name] = heuristic(heuristic_kwargs)
            return derive_inner(heuristic_kwargs)

        return derive_heuristic

    if isinstance(kernel, Autotuner):
        derive_inner = _triton_key_deriver(kernel.fn)

        def derive_autotuned(kwargs: Mapping[str, Any]) -> set[TritonJitKey]:
            previous_nargs = getattr(kernel, "nargs", None)
            kernel.nargs = {}
            try:
                configs = kernel.prune_configs(dict(kwargs))
            finally:
                kernel.nargs = previous_nargs
            keys: set[TritonJitKey] = set()
            for config in configs:
                conflicts = kwargs.keys() & config.kwargs.keys()
                if conflicts:
                    names = ", ".join(sorted(conflicts))
                    raise ValueError(f"Conflicting autotune parameters: {names}")
                keys.update(derive_inner(dict(kwargs) | config.all_kwargs()))
            return keys

        return derive_autotuned

    if not isinstance(kernel, JITFunction):
        raise TypeError(f"Unsupported Triton kernel wrapper: {type(kernel).__name__}")

    device = cast(Hashable, driver.active.get_current_device())
    _, kernel_key_cache, _, _, binder = kernel.device_caches[device]
    jit_function_id = id(kernel)
    jit_function_key = cast(Hashable, kernel.cache_key)
    debug = kernel.debug or knobs.runtime.debug
    instrumentation_mode = knobs.compilation.instrumentation_mode
    fpsan_homomorphic_casts = getattr(
        knobs.compilation, "fpsan_homomorphic_casts", None
    )
    inspection_hook = knobs.runtime.add_stages_inspection_hook

    def derive_jit(kwargs: Mapping[str, Any]) -> set[TritonJitKey]:
        binder_kwargs = dict(kwargs)
        binder_kwargs["debug"] = binder_kwargs.get("debug", debug) or debug
        binder_kwargs["instrumentation_mode"] = instrumentation_mode
        if fpsan_homomorphic_casts is not None:
            binder_kwargs["fpsan_homomorphic_casts"] = fpsan_homomorphic_casts
        _, specialization, options = binder(**binder_kwargs)
        if inspection_hook is not None:
            _, inspection_hash = inspection_hook()
            specialization.append(f'("custom_pipeline", {inspection_hash})')
        cache_key = cast(
            Hashable,
            compute_cache_key(kernel_key_cache, specialization, options),
        )
        return {
            TritonJitKey(
                jit_function_id,
                jit_function_key,
                device,
                cache_key,
            )
        }

    return derive_jit


WarmupCases = Mapping[str, Any] | Iterable[Mapping[str, Any]]


class _AutomaticTritonJitKernel(VllmTritonJitKernel[TritonCompileKey]):
    def warmup_inputs(self, compile_key: TritonCompileKey) -> dict[str, Any]:
        return dict(compile_key.inputs)

    def _provider_cases(
        self,
        provider: Callable[..., WarmupCases],
        *args: Any,
        **kwargs: Any,
    ) -> Iterable[Mapping[str, Any]]:
        cases_node = get_function_source_node(provider)
        if any(
            isinstance(node, ast.Call)
            and (get_ast_full_name(node.func) or "").split(".")[-1]
            in {"WarmupIntRange", "WarmupChoices", "_when"}
            for node in ast.walk(cases_node)
        ):
            return self._expand_warmup_cases(provider, *args, **kwargs)
        cases = provider(*args, **kwargs)
        return (cases,) if isinstance(cases, Mapping) else cases


def _is_autotuned(kernel: Any) -> bool:
    Autotuner = triton.runtime.autotuner.Autotuner

    while kernel is not None:
        if isinstance(kernel, Autotuner):
            return True
        kernel = getattr(kernel, "fn", None)
    return False


def _triton_metadata_arg(value: Any) -> Any:
    from torch._subclasses.fake_tensor import FakeTensor

    if not isinstance(value, FakeTensor):
        return value
    return TritonWarmupTensor(
        value.dtype,
        aligned=getattr(value, "_vllm_warmup_aligned", True),
        shape=tuple(value.shape),
        strides=tuple(value.stride()),
    )


def _materialize_warmup_case(
    case: Mapping[str, Any],
    *,
    real: bool,
    mode: Any,
    cache: dict[TritonWarmupTensor, Any],
) -> dict[str, Any]:
    import torch

    def materialize(value: Any) -> Any:
        if not isinstance(value, TritonWarmupTensor):
            return value
        if not real:
            try:
                return cache[value]
            except KeyError:
                pass
            except TypeError:
                # Unhashable initializers are uncommon and safe to materialize directly.
                pass

        strides = cast(tuple[int, ...], value.stride())
        with mode:
            tensor = torch.empty_strided(
                value.shape,
                strides,
                dtype=value.dtype,
                device=value.device,
            )
        if real:
            if callable(value.init):
                value.init(tensor)
            else:
                tensor.fill_(value.init)
        else:
            tensor._vllm_warmup_aligned = value.aligned
            with suppress(TypeError):
                cache[value] = tensor
        return tensor

    return {name: materialize(value) for name, value in case.items()}


class _DecoratedTritonJitKernel(_AutomaticTritonJitKernel):
    def __init__(
        self,
        kernel: Any,
        warmup_inputs: Callable[..., WarmupCases],
        dispatch: Callable[..., DispatchSpec] | None,
    ) -> None:
        self.kernel = kernel
        self._run_autotune = _is_autotuned(kernel)
        self._warmup_inputs_fn = warmup_inputs
        self._dispatch_fn = dispatch
        self._dispatch_arg_names: tuple[str, ...] = ()
        self._dispatch_defaults: dict[str, Any] = {}
        if dispatch is not None:
            parameters = tuple(inspect.signature(dispatch).parameters.values())
            if any(
                parameter.kind
                in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                for parameter in parameters
            ):
                raise TypeError("Triton dispatchers require explicit parameters")
            self._dispatch_arg_names = tuple(parameter.name for parameter in parameters)
            self._dispatch_defaults = {
                parameter.name: parameter.default
                for parameter in parameters
                if parameter.default is not inspect.Parameter.empty
            }
        super().__init__()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._dispatch_fn is None:
            grid = kwargs.pop("grid") if not args else args[0]
            kernel_args = () if not args else args[1:]
            inputs = triton_warmup_inputs(
                self.kernel, *kernel_args, grid=grid, **kwargs
            )
            inputs.pop("grid")
            return self.launch(grid, (), (), **inputs)
        spec = self._dispatch_fn(*args, **kwargs)
        input_values = args
        if len(args) != len(self._dispatch_arg_names):
            input_values += tuple(
                kwargs[name] if name in kwargs else self._dispatch_defaults[name]
                for name in self._dispatch_arg_names[len(args) :]
            )
        grid, launch_kwargs = spec[:2]
        result = self.launch(
            grid, self._dispatch_arg_names, input_values, **launch_kwargs
        )
        return spec[2] if len(spec) == 3 else result

    def __getitem__(self, grid: tuple[int, ...]) -> Callable[..., Any]:
        return cast(Callable[..., Any], self.kernel[grid])

    def __getattr__(self, name: str) -> Any:
        return getattr(self.kernel, name)

    def get_warmup_keys(
        self, *, vllm_config: Any | None = None, **kwargs: Any
    ) -> list[TritonCompileKey]:
        keys: list[TritonCompileKey] = []
        seen_jit_keys: set[frozenset[TritonJitKey]] = set()
        if "vllm_config" in inspect.signature(self._warmup_inputs_fn).parameters:
            kwargs.setdefault("vllm_config", vllm_config)
        cases = self._provider_cases(self._warmup_inputs_fn, **kwargs)
        derive_keys = _triton_key_deriver(self.kernel)
        materialize = self._run_autotune
        mode: Any = nullcontext() if self._run_autotune else None
        materialized: dict[TritonWarmupTensor, Any] = {}
        for case in cases:
            inputs = (
                _materialize_warmup_case(
                    case,
                    real=self._run_autotune,
                    mode=mode,
                    cache=materialized,
                )
                if materialize
                else case
            )
            if self._dispatch_fn is None:
                prepared = dict(inputs)
                prepared.pop("grid")
            else:
                dispatch_inputs = {
                    name: inputs[name]
                    for name in self._dispatch_arg_names
                    if name in inputs
                }
                try:
                    _, launch_kwargs = self._dispatch_fn(**dispatch_inputs)[:2]
                except (AssertionError, AttributeError, TypeError):
                    if materialize:
                        raise
                    from torch._subclasses.fake_tensor import FakeTensorMode

                    materialize = True
                    mode = FakeTensorMode()
                    inputs = _materialize_warmup_case(
                        case,
                        real=False,
                        mode=mode,
                        cache=materialized,
                    )
                    dispatch_inputs = {
                        name: inputs[name]
                        for name in self._dispatch_arg_names
                        if name in inputs
                    }
                    _, launch_kwargs = self._dispatch_fn(**dispatch_inputs)[:2]
                prepared = self._prepare_launch_kwargs(
                    tuple(inputs),
                    tuple(inputs.values()),
                    dict(launch_kwargs),
                )
            if materialize:
                prepared = {
                    name: _triton_metadata_arg(value)
                    for name, value in prepared.items()
                }
            jit_keys = frozenset(derive_keys(prepared))
            if jit_keys and jit_keys not in seen_jit_keys:
                seen_jit_keys.add(jit_keys)
                keys.append(TritonCompileKey(jit_keys, tuple(inputs.items())))
        return keys


class TritonKernelDispatcher(Protocol[P]):
    """Callable Triton launcher with compile-time warmup support."""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Any: ...

    def register_warmup(self, *args: Any, **kwargs: Any) -> None: ...


@overload
def triton_kernel_dispatcher_with_warmup(
    *,
    warmup_inputs: Callable[..., WarmupCases],
    kernel: None = None,
) -> Callable[[Any], Any]: ...


@overload
def triton_kernel_dispatcher_with_warmup(
    *,
    warmup_inputs: Callable[..., WarmupCases],
    kernel: Any,
) -> Callable[[Callable[P, DispatchSpec]], TritonKernelDispatcher[P]]: ...


def triton_kernel_dispatcher_with_warmup(
    *,
    warmup_inputs: Callable[..., WarmupCases],
    kernel: Any | None = None,
) -> Callable[[Any], Any]:
    """Decorate a dispatch function or a native Triton kernel for warmup."""

    def decorate(dispatch: Any) -> Any:
        owner = _DecoratedTritonJitKernel(
            dispatch if kernel is None else kernel,
            warmup_inputs,
            None if kernel is None else dispatch,
        )
        update_wrapper(owner, dispatch, updated=())
        return owner

    return decorate


@dataclass(frozen=True)
class TritonPointerInputVariant:
    # Named pointer-alignment variant for compile-only Triton warmup.
    alignments: tuple[tuple[str, bool], ...]

    @classmethod
    def from_alignment(cls, **aligned: bool) -> "TritonPointerInputVariant":
        return cls(tuple(aligned.items()))

    def is_aligned(self, name: str) -> bool:
        for alignment_name, aligned in self.alignments:
            if alignment_name == name:
                return aligned
        raise KeyError(f"Unknown Triton pointer input variant: {name}")

    def pointer(
        self,
        name: str,
        dtype: Any,
        shape: tuple[int, ...] = (1,),
    ) -> TritonWarmupTensor:
        return TritonWarmupTensor(dtype, aligned=self.is_aligned(name), shape=shape)


def _literal_str_refs(node: ast.AST) -> tuple[str | int, ...]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str | int):
        return (node.value,)
    if isinstance(node, ast.List | ast.Tuple):
        refs: list[str | int] = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str | int):
                refs.append(elt.value)
            else:
                raise ValueError(
                    f"Unsupported Triton specialization ref: {ast.dump(elt)}"
                )
        return tuple(refs)
    raise ValueError(f"Unsupported Triton specialization refs: {ast.dump(node)}")


def _normalize_arg_refs(
    refs: tuple[str | int, ...],
    arg_names: tuple[str, ...],
) -> frozenset[str]:
    names: set[str] = set()
    for ref in refs:
        if isinstance(ref, int):
            names.add(arg_names[ref])
        else:
            names.add(ref)
    return frozenset(names)


def _decorator_keyword_refs(
    function_def: ast.FunctionDef,
    keyword_name: str,
) -> tuple[str | int, ...]:
    for decorator in function_def.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        decorator_name = get_ast_full_name(decorator.func)
        if decorator_name not in ("triton.jit", "jit"):
            continue
        for keyword in decorator.keywords:
            if keyword.arg == keyword_name:
                return _literal_str_refs(keyword.value)
    return ()


def _triton_do_not_specialize_args(
    kernel: Callable[..., Any],
    function_def: ast.FunctionDef,
    arg_names: tuple[str, ...],
) -> frozenset[str]:
    refs = getattr(kernel, "do_not_specialize", None)
    if refs is not None:
        return _normalize_arg_refs(tuple(refs), arg_names)
    return _normalize_arg_refs(
        _decorator_keyword_refs(function_def, "do_not_specialize"),
        arg_names,
    )


def _triton_constexpr_arg_names(
    kernel: Callable[..., Any],
    function_def: ast.FunctionDef,
    arg_names: tuple[str, ...],
) -> frozenset[str]:
    constexprs = getattr(kernel, "constexprs", None)
    if constexprs is not None:
        return frozenset(arg_names[index] for index in constexprs)

    names: set[str] = set()
    for arg in function_def.args.args + function_def.args.kwonlyargs:
        if arg.annotation is None:
            continue
        annotation = get_ast_full_name(arg.annotation)
        if annotation in ("tl.constexpr", "triton.language.constexpr", "constexpr"):
            names.add(arg.arg)
    return frozenset(names)


def _leftmost_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.BinOp):
        return _leftmost_name(node.left)
    return None


def _pointer_arg_names(
    function_def: ast.FunctionDef,
    arg_names: tuple[str, ...],
) -> frozenset[str]:
    candidate_names = set(arg_names)
    pointer_names = {name for name in arg_names if name.endswith("_ptr")}
    for node in ast.walk(function_def):
        if not isinstance(node, ast.Call):
            continue
        if get_ast_full_name(node.func) not in ("tl.load", "tl.store"):
            continue
        if not node.args:
            continue
        name = _leftmost_name(node.args[0])
        if name in candidate_names:
            pointer_names.add(name)
    return frozenset(pointer_names)


def trace_triton_kernel_specialization_args(
    kernel: Callable[..., Any],
) -> tuple[str, ...]:
    function_def = get_function_source_node(kernel)
    if not isinstance(function_def, ast.FunctionDef):
        raise ValueError("Expected Triton kernel to be defined as a function")
    source_fn = getattr(kernel, "fn", kernel)
    arg_names = tuple(inspect.signature(source_fn).parameters)
    constexpr_args = _triton_constexpr_arg_names(kernel, function_def, arg_names)
    do_not_specialize_args = _triton_do_not_specialize_args(
        kernel, function_def, arg_names
    )
    pointer_args = _pointer_arg_names(function_def, arg_names)

    return tuple(
        name
        for name in arg_names
        if name in constexpr_args
        or (name not in pointer_args and name not in do_not_specialize_args)
    )
