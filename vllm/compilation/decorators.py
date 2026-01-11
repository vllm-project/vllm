# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import hashlib
import inspect
import os
import sys
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload
from unittest.mock import patch

import torch
import torch.nn as nn
from packaging import version
from torch._dynamo.symbolic_convert import InliningInstructionTranslator

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import (
    CompilationMode,
    VllmConfig,
    get_current_vllm_config,
    set_current_vllm_config,
)
from vllm.config.compilation import DynamicShapesType
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.torch_utils import is_torch_equal_or_newer, supports_dynamo

from .monitor import start_monitoring_torch_compile

if TYPE_CHECKING:
    # Only added on nightly/2.10 so wrap
    try:
        from torch._dynamo.package import SourceInfo
    except ImportError:
        # Fallback for old versions not supporting
        SourceInfo = Any

logger = init_logger(__name__)

IGNORE_COMPILE_KEY = "_ignore_compile_vllm"

_T = TypeVar("_T", bound=type[nn.Module])


def ignore_torch_compile(cls: _T) -> _T:
    """
    A decorator to ignore support_torch_compile decorator
    on the class. This is useful when a parent class has
    a support_torch_compile decorator, but we don't want to
    compile the class `cls` that inherits the parent class.
    This only ignores compiling the forward of the class the
    decorator is applied to.

    If the parent has ignore_torch_compile but the child has
    support_torch_compile, the child will still be compiled.

    If the class has one or more submodules
    that have support_torch_compile decorator applied, compile will
    not be ignored for those submodules.
    """
    setattr(cls, IGNORE_COMPILE_KEY, True)
    return cls


def _should_ignore_torch_compile(cls: _T) -> bool:
    """
    Check if the class should be ignored for torch.compile.
    """
    return getattr(cls, IGNORE_COMPILE_KEY, False)


@overload
def support_torch_compile(
    *,
    enable_if: Callable[[VllmConfig], bool] | None = None,
) -> Callable[[_T], _T]: ...


@overload
def support_torch_compile(
    *,
    dynamic_arg_dims: dict[str, int | list[int]] | None,
) -> Callable[[_T], _T]: ...


@overload
def support_torch_compile(
    *,
    mark_unbacked_dims: dict[str, int | list[int]] | None,
) -> Callable[[_T], _T]: ...


@overload
def support_torch_compile(
    *,
    dynamic_arg_dims: dict[str, int | list[int]] | None,
    mark_unbacked_dims: dict[str, int | list[int]] | None,
) -> Callable[[_T], _T]: ...


@overload
def support_torch_compile(cls: _T) -> _T: ...


def support_torch_compile(
    cls: _T | None = None,
    *,
    dynamic_arg_dims: dict[str, int | list[int]] | None = None,
    mark_unbacked_dims: dict[str, int | list[int]] | None = None,
    enable_if: Callable[[VllmConfig], bool] | None = None,
    shape_invariants: Callable[..., None] = lambda *args, **kwargs: None,
) -> Callable[[_T], _T] | _T:
    """
    A decorator to add support for compiling the forward method of a class.

    Usage 1: use directly as a decorator without arguments:

    ```python
    @support_torch_compile
    class MyModel(nn.Module):
        def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]): ...
    ```

    Usage 2: use as a decorator with arguments:

    ```python
    @support_torch_compile(dynamic_arg_dims={"x": 0, "y": 0})
    class MyModel(nn.Module):
        def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]): ...
    ```

    `dynamic_arg_dims` is a dictionary that maps argument names to the dynamic
    dimensions of the argument. The dynamic dimensions can be either a single
    integer or a list of integers.

    if `dynamic_arg_dims` is `None`, it is inferred from the type annotation
    of the `forward` method, based on the following default rules:

    - if the argument is annotated as `torch.Tensor` or
        `Optional[torch.Tensor]`, the first dimension will be
        marked as dynamic.
    - if the argument is annotated as `IntermediateTensors`, the first
        dimension of all the tensors in the intermediate tensors
        will be marked as dynamic.

    During runtime, when we actually mark dimensions of tensors,
     it depends on the value of arguments:

    - if it is a single integer (can be negative), the corresponding dimension
        of the argument will be marked as dynamic.
    - if it is `None`, ignored.
    - if it is `IntermediateTensors`, all the tensors in the intermediate
        tensors will be marked as dynamic.
    - otherwise, it will raise an error.

    NOTE: if an argument is `None`, it should always be passed as `None` during
    the lifetime of the model, otherwise, it cannot be captured as a single
    computation graph.

    `enable_if` is a function that takes a `VllmConfig` object as input and
    returns a boolean value indicating whether to compile the model or not.
    This is useful if you want to compile the model only when certain
    conditions are met.

    `mark_unbacked_dims` is a dictionary that maps argument names with a dynamic
    dim to be decorated with `mark_unbacked`.  This is useful if we would like to
    enforce that dynamo does not specialize on 0/1 values in the case of dummy input
    such as for vision model compilation

    `shape_invariants` is a function that gets compiled right before forward.
    The function should have the torch._check calls that are needed to set
    the relationships between different input sizes. For example:
            torch._check(input_ids.size()[0] == inputs_embeds.size()[0])
    This enforces constraints on the symbolic shapes without hardcoding
    specific values. It is needed for some models to avoid data dependent
    errors.
    """

    def cls_decorator_helper(cls: _T) -> _T:
        # helper to pass `dynamic_arg_dims` to `_support_torch_compile`
        # to avoid too much indentation for `_support_torch_compile`
        if not hasattr(cls, "forward"):
            raise TypeError("decorated class should have a forward method.")
        sig = inspect.signature(cls.forward)
        inferred_dynamic_arg_dims = dynamic_arg_dims
        if inferred_dynamic_arg_dims is None:
            inferred_dynamic_arg_dims = {}
            for k, v in sig.parameters.items():
                if v.annotation in [
                    torch.Tensor,
                    torch.Tensor | None,
                    IntermediateTensors,
                    IntermediateTensors | None,
                ]:
                    inferred_dynamic_arg_dims[k] = 0

            logger.debug(
                ("Inferred dynamic dimensions for forward method of %s: %s"),
                cls,
                list(inferred_dynamic_arg_dims.keys()),
            )

        if len(inferred_dynamic_arg_dims) == 0:
            raise ValueError(
                "No dynamic dimensions found in the forward method of "
                f"{cls}. Please provide dynamic_arg_dims explicitly."
            )

        for k in inferred_dynamic_arg_dims:
            if k not in sig.parameters:
                raise ValueError(
                    f"Argument {k} not found in the forward method of {cls}"
                )
        return _support_torch_compile(
            cls,
            inferred_dynamic_arg_dims,
            mark_unbacked_dims,
            enable_if,
            shape_invariants,
        )

    if cls is not None:
        # use `support_torch_compile` as a decorator without arguments
        assert isinstance(cls, type)
        return cls_decorator_helper(cls)

    return cls_decorator_helper


def _model_hash_key(fn: Callable[..., Any]) -> str:
    import vllm

    sha256_hash = hashlib.sha256()
    sha256_hash.update(vllm.__version__.encode())
    sha256_hash.update(fn.__qualname__.encode())
    sha256_hash.update(str(fn.__code__.co_firstlineno).encode())
    return sha256_hash.hexdigest()


def _verify_source_unchanged(
    source_info: "SourceInfo", vllm_config: VllmConfig
) -> None:
    from .caching import _compute_code_hash, _compute_code_hash_with_content

    file_contents = {}
    for source in source_info.inlined_sources:
        module = sys.modules[source.module]
        file = inspect.getfile(module)
        vllm_config.compilation_config.traced_files.add(file)
        file_contents[file] = source.content
    expected_checksum = _compute_code_hash_with_content(file_contents)
    actual_checksum = _compute_code_hash(set(file_contents.keys()))
    if expected_checksum != actual_checksum:
        raise RuntimeError(
            "Source code has changed since the last compilation. Recompiling the model."
        )


def _support_torch_compile(
    cls: _T,
    dynamic_arg_dims: dict[str, int | list[int]],
    mark_unbacked_dims: dict[str, int | list[int]] | None = None,
    enable_if: Callable[[VllmConfig], bool] | None = None,
    shape_invariants: Callable[..., None] = lambda *args, **kwargs: None,
) -> _T:
    """
    A decorator to add support for compiling the forward method of a class.
    """
    if TorchCompileWithNoGuardsWrapper in cls.__bases__:
        # support decorating multiple times
        return cls

    # take care of method resolution order
    # make sure super().__init__ is called on the base class
    #  other than TorchCompileWithNoGuardsWrapper
    cls.__bases__ = cls.__bases__ + (TorchCompileWithNoGuardsWrapper,)

    old_init = cls.__init__

    setattr(cls, IGNORE_COMPILE_KEY, False)

    def __init__(
        self: _T,
        *,
        vllm_config: VllmConfig | None = None,
        prefix: str = "",
        **kwargs: Any,
    ) -> None:
        if vllm_config is None:
            vllm_config = get_current_vllm_config()

        # NOTE: to support multimodal models (such as encoder),
        # we may not have vllm_config so we may need to patch
        # it
        sig = inspect.signature(old_init)
        if "vllm_config" in sig.parameters:
            kwargs["vllm_config"] = vllm_config
        if "prefix" in sig.parameters:
            kwargs["prefix"] = prefix
        old_init(self, **kwargs)

        self.vllm_config = vllm_config
        self.compilation_config = self.vllm_config.compilation_config
        enable_compile = enable_if is None or enable_if(vllm_config)
        # for CompilationMode.STOCK_TORCH_COMPILE , the upper level model runner
        # will handle the compilation, so we don't need to do anything here.
        self.do_not_compile = (
            self.compilation_config.mode
            in [CompilationMode.NONE, CompilationMode.STOCK_TORCH_COMPILE]
            or not supports_dynamo()
            or _should_ignore_torch_compile(self.__class__)
            or not enable_compile
        )
        if self.do_not_compile:
            return

        self._check_shape_invariants = shape_invariants

        compilation_counter.num_models_seen += 1
        self.compiled = False

        # Handled by monkeypatching `TorchCompileWithNoGuardsWrapper` into base class
        TorchCompileWithNoGuardsWrapper.__init__(self)  # type: ignore[arg-type]

    cls.__init__ = __init__

    def _mark_dynamic_inputs(
        mod: _T, ds_type: DynamicShapesType, *args: Any, **kwargs: Any
    ) -> None:
        def mark_dynamic(arg: torch.Tensor, dims: list[int]) -> None:
            if ds_type == DynamicShapesType.UNBACKED:
                if is_torch_equal_or_newer("2.10.0.dev"):
                    for dim in dims:
                        torch._dynamo.decorators.mark_unbacked(
                            arg, dim, hint_override=arg.size()[dim]
                        )
                else:
                    torch._dynamo.decorators.mark_unbacked(arg, dims)
            else:
                torch._dynamo.mark_dynamic(arg, dims)

        sig = inspect.signature(mod.__class__.forward)  # type: ignore[attr-defined]
        bound_args = sig.bind(mod, *args, **kwargs)
        bound_args.apply_defaults()
        for k, dims in dynamic_arg_dims.items():
            arg = bound_args.arguments.get(k)

            if arg is not None:
                dims = [dims] if isinstance(dims, int) else dims
                if isinstance(arg, torch.Tensor):
                    # In case dims is specified with negative indexing
                    dims = [arg.ndim + dim if dim < 0 else dim for dim in dims]
                    mark_dynamic(arg, dims)
                elif isinstance(arg, IntermediateTensors):
                    for tensor in arg.tensors.values():
                        # In case dims is specified with negative indexing
                        dims = [tensor.ndim + dim if dim < 0 else dim for dim in dims]
                        mark_dynamic(tensor, dims)
                else:
                    raise ValueError(
                        "Unsupported dynamic dimensions"
                        f" {dims} for argument {k} with type {type(arg)}."
                    )
        if mark_unbacked_dims:
            for k, dims in mark_unbacked_dims.items():
                arg = bound_args.arguments.get(k)
                if arg is not None:
                    dims = [dims] if isinstance(dims, int) else dims
                    if isinstance(arg, torch.Tensor):
                        # In case dims is specified with negative indexing
                        dims = [arg.ndim + dim if dim < 0 else dim for dim in dims]
                        if is_torch_equal_or_newer("2.10.0.dev"):
                            for dim in dims:
                                torch._dynamo.decorators.mark_unbacked(
                                    arg, dim, hint_override=arg.size()[dim]
                                )
                        else:
                            torch._dynamo.decorators.mark_unbacked(arg, dims)

    def __call__(self: _T, *args: Any, **kwargs: Any) -> Any:
        # torch.compiler.is_compiling() means we are inside the compilation
        # e.g. TPU has the compilation logic in model runner, so we don't
        # need to compile the model inside.
        if self.do_not_compile or torch.compiler.is_compiling():
            return self.forward(*args, **kwargs)

        # if aot_compiled_fn is set, call it with partition wrapper context.
        # The partition wrapper must be active at runtime for CUDA graph
        # capture to work correctly with inductor graph partitioning.
        if getattr(self, "aot_compiled_fn", None) is not None:
            with maybe_use_cudagraph_partition_wrapper(self.vllm_config):
                return self.aot_compiled_fn(self, *args, **kwargs)

        ds_type = self.compilation_config.dynamic_shapes_config.type
        cache_dir = None
        aot_compilation_path = None
        if envs.VLLM_USE_AOT_COMPILE:
            """
            When using torch.compile in AOT mode, we store the cache artifacts
            under VLLM_CACHE_ROOT/torch_aot_compile/{hash}/rank_i_j. The {hash}
            contains all of the factors except for the source files being
            traced through, because we don't actually know which source files
            to check at this point (before dynamo runs).
            On loading we will actually look at the source files being traced
            through. If any source file have changed (compared with the
            serialized backend artifacts), then we need to generate a new AOT
            compile artifact from scratch.
            """
            from .caching import compilation_config_hash_factors

            factors: list[str] = compilation_config_hash_factors(self.vllm_config)

            factors.append(_model_hash_key(self.forward))
            hash_key = hashlib.sha256(str(factors).encode()).hexdigest()
            cache_dir = os.path.join(
                envs.VLLM_CACHE_ROOT,
                "torch_aot_compile",
                hash_key,
            )

            rank = self.vllm_config.parallel_config.rank
            dp_rank = self.vllm_config.parallel_config.data_parallel_index
            cache_dir = os.path.join(cache_dir, f"rank_{rank}_{dp_rank}")
            aot_compilation_path = os.path.join(cache_dir, "model")
            try:
                with (
                    set_current_vllm_config(self.vllm_config),
                    open(aot_compilation_path, "rb") as f,
                ):
                    start_monitoring_torch_compile(self.vllm_config)
                    loaded_fn = torch.compiler.load_compiled_function(
                        f, f_globals=self.forward.__globals__
                    )
                _verify_source_unchanged(loaded_fn.source_info(), self.vllm_config)
                if not self.compilation_config.dynamic_shapes_config.evaluate_guards:
                    loaded_fn.disable_guard_check()
                self.aot_compiled_fn = loaded_fn
            except Exception as e:
                if os.path.exists(aot_compilation_path):
                    logger.warning(
                        "Cannot load aot compilation from path %s, error: %s",
                        aot_compilation_path,
                        str(e),
                    )
                if envs.VLLM_FORCE_AOT_LOAD:
                    raise e
            if getattr(self, "aot_compiled_fn", None) is not None:
                logger.info(
                    "Directly load AOT compilation from path %s", aot_compilation_path
                )
                # Apply partition wrapper context for proper CUDA graph capture
                with maybe_use_cudagraph_partition_wrapper(self.vllm_config):
                    return self.aot_compiled_fn(self, *args, **kwargs)

        if self.compiled:
            assert (
                not envs.VLLM_USE_AOT_COMPILE
                or self.vllm_config.compilation_config.backend == "eager"
            )
            return TorchCompileWithNoGuardsWrapper.__call__(self, *args, **kwargs)  # type: ignore[arg-type]

        # This is the path for the first compilation.
        # the first compilation needs to have dynamic shapes marked
        _mark_dynamic_inputs(
            self,
            ds_type,
            *args,
            **kwargs,
        )

        # here, it is the starting point of the `torch.compile` process
        start_monitoring_torch_compile(self.vllm_config)
        original_code_object = self.original_code_object()
        logger.debug("Start compiling function %s", original_code_object)

        # we do not want tp delete the original code object entries since
        # we depend on them now to look up cached compiled functions.
        # torch._dynamo.eval_frame.remove_from_cache(original_code_object)

        # collect all relevant files traced by Dynamo,
        # so that the compilation cache can trigger re-compilation
        # properly when any of these files change.

        # 1. the file containing the top-level forward function
        self.compilation_config.traced_files.add(original_code_object.co_filename)

        # 2. every time Dynamo sees a function call, it will inline
        # the function by calling InliningInstructionTranslator.inline_call_
        # we hijack this function to know all the functions called
        # during Dynamo tracing, and their corresponding files
        inline_call = InliningInstructionTranslator.inline_call_

        def patched_inline_call(self_: Any) -> Any:
            code = self_.f_code
            self.compilation_config.traced_files.add(code.co_filename)
            return inline_call(self_)

        # Disable the C++ compilation of symbolic shape guards. C++-fication
        # of symbolic shape guards can improve guard overhead. But, since
        # vllm skip guards anyways, setting this flag to False can improve
        # compile time.
        dynamo_config_patches = {}
        try:
            _ = torch._dynamo.config.enable_cpp_symbolic_shape_guards
            dynamo_config_patches["enable_cpp_symbolic_shape_guards"] = False
        except AttributeError:
            # Note: this config is not available in torch 2.6, we can skip
            # if the config doesn't exist
            logger.debug("enable_cpp_symbolic_shape_guards config not available")

        # Prepare backed_size_oblivious config patch if needed
        fx_config_patches = {}
        if ds_type == DynamicShapesType.BACKED_SIZE_OBLIVIOUS:
            fx_config_patches["backed_size_oblivious"] = True

        # Prepare inductor config patches
        # assume_32bit_indexing is only available in torch 2.10.0.dev+
        inductor_config_patches = {}
        if is_torch_equal_or_newer("2.10.0.dev"):
            inductor_config_patches["assume_32bit_indexing"] = (
                self.compilation_config.dynamic_shapes_config.assume_32_bit_indexing
            )

        with (
            patch.object(
                InliningInstructionTranslator, "inline_call_", patched_inline_call
            ),
            torch._dynamo.config.patch(**dynamo_config_patches),
            maybe_use_cudagraph_partition_wrapper(self.vllm_config),
            torch.fx.experimental._config.patch(**fx_config_patches),
            _torch27_patch_tensor_subclasses(),
            torch._inductor.config.patch(**inductor_config_patches),
        ):
            use_aot_compile = envs.VLLM_USE_AOT_COMPILE
            if self.vllm_config.compilation_config.backend == "eager":
                logger.warning("Detected eager backend, disabling AOT compile.")
                use_aot_compile = False
            if use_aot_compile:
                self.aot_compiled_fn = self.aot_compile(*args, **kwargs)
                output = self.aot_compiled_fn(self, *args, **kwargs)
                assert aot_compilation_path is not None
                assert cache_dir is not None
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    self.aot_compiled_fn.save_compiled_function(aot_compilation_path)
                except Exception as e:
                    logger.warning(
                        "Cannot save aot compilation to path %s, error: %s",
                        aot_compilation_path,
                        str(e),
                    )
            else:
                output = TorchCompileWithNoGuardsWrapper.__call__(self, *args, **kwargs)  # type: ignore[arg-type]

        self.compiled = True
        return output

    cls.__call__ = __call__
    return cls


@contextlib.contextmanager
def maybe_use_cudagraph_partition_wrapper(
    vllm_config: VllmConfig,
) -> Generator[None, None, None]:
    """
    Context manager to set/unset customized cudagraph partition wrappers.

    If we're using Inductor-based graph partitioning, we currently have the
    whole `fx.Graph` before Inductor lowering and the piecewise
    splitting happens after all graph passes and fusions. Here, we add
    a custom hook for Inductor to wrap each partition with our static
    graph wrapper class to maintain more control over static graph
    capture and replay.
    """
    from vllm.config import CUDAGraphMode

    compilation_config = vllm_config.compilation_config
    if (
        compilation_config.cudagraph_mode.has_piecewise_cudagraphs()
        and compilation_config.use_inductor_graph_partition
    ):
        from torch._inductor.utils import CUDAGraphWrapperMetadata

        from vllm.compilation.cuda_graph import CUDAGraphOptions
        from vllm.platforms import current_platform

        static_graph_wrapper_class = resolve_obj_by_qualname(
            current_platform.get_static_graph_wrapper_cls()
        )

        def customized_cudagraph_wrapper(
            f: Callable[..., Any], metadata: CUDAGraphWrapperMetadata
        ) -> Any:
            partition_id = metadata.partition_index
            num_partitions = metadata.num_partitions
            return static_graph_wrapper_class(
                runnable=f,
                vllm_config=vllm_config,
                runtime_mode=CUDAGraphMode.PIECEWISE,
                cudagraph_options=CUDAGraphOptions(
                    debug_log_enable=partition_id == 0,
                    gc_disable=partition_id != 0,
                    weak_ref_output=partition_id == num_partitions - 1,
                ),
            )

        torch._inductor.utils.set_customized_partition_wrappers(
            customized_cudagraph_wrapper
        )

    yield

    if (
        compilation_config.cudagraph_mode.has_piecewise_cudagraphs()
        and compilation_config.use_inductor_graph_partition
    ):
        torch._inductor.utils.set_customized_partition_wrappers(None)


@contextlib.contextmanager
def _torch27_patch_tensor_subclasses() -> Generator[None, None, None]:
    """
    Add support for using tensor subclasses (ie `BasevLLMParameter`, ect) when
    using torch 2.7.0. This enables using weight_loader_v2 and the use of
    `BasevLLMParameters` without having to replace them with regular tensors
    before `torch.compile`-time.
    """
    from vllm.model_executor.parameter import (
        BasevLLMParameter,
        ModelWeightParameter,
        RowvLLMParameter,
        _ColumnvLLMParameter,
    )

    def return_false(*args: Any, **kwargs: Any) -> Literal[False]:
        return False

    if version.parse("2.7") <= version.parse(torch.__version__) < version.parse("2.8"):
        yield
        return

    with (
        torch._dynamo.config.patch(
            "traceable_tensor_subclasses",
            [
                BasevLLMParameter,
                ModelWeightParameter,
                _ColumnvLLMParameter,
                RowvLLMParameter,
            ],
        ),
        patch(
            "torch._dynamo.variables.torch.can_dispatch_torch_function", return_false
        ),
    ):
        yield
