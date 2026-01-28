# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import copy
import os
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any, Literal
from unittest.mock import patch

import torch
import torch._inductor.compile_fx
import torch.fx as fx

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.logger import init_logger
from vllm.utils.hashing import safe_hash
from vllm.utils.torch_utils import is_torch_equal_or_newer

logger = init_logger(__name__)


class CompilerInterface:
    """
    The interface for a compiler that can be used by vLLM.
    """

    # The name of the compiler, e.g. inductor.
    # This is a class-level attribute.
    name: str

    def initialize_cache(
        self, cache_dir: str, disable_cache: bool = False, prefix: str = ""
    ) -> None:
        """
        when the vLLM process uses `cache_dir` as the cache directory,
        the compiler should initialize itself with the cache directory,
        e.g. by re-directing its own cache directory to a sub-directory.

        prefix can be used in combination with cache_dir to figure out the base
        cache directory, e.g. there're multiple parts of model being compiled,
        but we want to share the same cache directory for all of them.

        e.g.
        cache_dir = "/path/to/dir/backbone", prefix = "backbone"
        cache_dir = "/path/to/dir/eagle_head", prefix = "eagle_head"
        """
        pass

    def compute_hash(self, vllm_config: VllmConfig) -> str:
        """
        Gather all the relevant information from the vLLM config,
        to compute a hash so that we can cache the compiled model.

        See [`VllmConfig.compute_hash`][vllm.config.VllmConfig.compute_hash]
        to check what information
        is already considered by default. This function should only
        consider the information that is specific to the compiler.
        """
        return ""

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        compile_range: Range,
        key: str | None = None,
    ) -> tuple[Callable[..., Any] | None, Any | None]:
        """
        Compile the graph with the given example inputs and compiler config,
        with a range. The `compile_range` specifies the range of the inputs,
        it could be concrete size (if compile_sizes is provided), e.g. [4, 4]
        or a range [5, 8].
        Right now we only support one variable in ranges for all inputs,
         which is the batchsize (number of tokens) during inference.

        Dynamo will make sure `graph(*example_inputs)` is valid.

        The function should return a compiled callable function, as well as
        a handle that can be used to directly load the compiled function.

        The handle should be a plain Python object, preferably a string or a
        file path for readability.

        If the compiler doesn't support caching, it should return None for the
        handle. If the compiler fails to compile the graph, it should return
        None for the compiled function as well.

        `key` is required for StandaloneInductorAdapter, it specifies where to
        save the compiled artifact. The compiled artifact gets saved to
        `cache_dir/key`.
        """
        return None, None

    def load(
        self,
        handle: Any,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        graph_index: int,
        compile_range: Range,
    ) -> Callable[..., Any]:
        """
        Load the compiled function from the handle.
        Raises an error if the handle is invalid.

        The handle is the second return value of the `compile` function.
        """
        raise NotImplementedError("caching is not supported")


class AlwaysHitShapeEnv:
    """
    Why do we need this class:

    For normal `torch.compile` usage, every compilation will have
    one Dynamo bytecode compilation and one Inductor compilation.
    The Inductor compilation happens under the context of the
    Dynamo bytecode compilation, and that context is used to
    determine the dynamic shape information, etc.

    For our use case, we only run Dynamo bytecode compilation once,
    and run Inductor compilation multiple times with different shapes
    plus a general shape. The compilation for specific shapes happens
    outside of the context of the Dynamo bytecode compilation. At that
    time, we don't have shape environment to provide to Inductor, and
    it will fail the Inductor code cache lookup.

    By providing a dummy shape environment that always hits, we can
    make the Inductor code cache lookup always hit, and we can
    compile the graph for different shapes as needed.

    The following dummy methods are obtained by trial-and-error
    until it works.
    """

    def __init__(self) -> None:
        self.guards: list[Any] = []

    def evaluate_guards_expression(self, *args: Any, **kwargs: Any) -> Literal[True]:
        return True

    def get_pruned_guards(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    def produce_guards_expression(self, *args: Any, **kwargs: Any) -> Literal[""]:
        return ""


def get_inductor_factors() -> list[Any]:
    factors: list[Any] = []
    # summarize system state
    from torch._inductor.codecache import CacheBase

    system_factors = CacheBase.get_system()
    factors.append(system_factors)

    # summarize pytorch state
    from torch._inductor.codecache import torch_key

    torch_factors = torch_key()
    factors.append(torch_factors)
    return factors


def is_compile_cache_enabled(
    vllm_additional_inductor_config: dict[str, Any],
) -> bool:
    vllm_inductor_config_disable_cache = vllm_additional_inductor_config.get(
        "force_disable_caches", False
    )

    # TODO(gmagogsfm): Replace torch._inductor.config.force_disable_caches
    # with torch.compiler.config.force_disable_caches when minimum PyTorch
    # version reaches 2.10
    return (
        not envs.VLLM_DISABLE_COMPILE_CACHE
        and not torch._inductor.config.force_disable_caches
        and not vllm_inductor_config_disable_cache
    )


class InductorStandaloneAdaptor(CompilerInterface):
    """
    The adaptor for the Inductor compiler.
    Requires PyTorch 2.8+.
    This is not on by default yet, but we plan to turn it on by default for
    PyTorch 2.8.

    Use VLLM_USE_STANDALONE_COMPILE to toggle this on or off.
    """

    name = "inductor_standalone"

    def __init__(self, save_format: Literal["binary", "unpacked"]) -> None:
        self.save_format = save_format

    def compute_hash(self, vllm_config: VllmConfig) -> str:
        factors = get_inductor_factors()
        hash_str: str = safe_hash(
            str(factors).encode(), usedforsecurity=False
        ).hexdigest()[:10]
        return hash_str

    def initialize_cache(
        self, cache_dir: str, disable_cache: bool = False, prefix: str = ""
    ) -> None:
        self.cache_dir = cache_dir

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        compile_range: Range,
        key: str | None = None,
    ) -> tuple[Callable[..., Any] | None, Any | None]:
        compilation_counter.num_inductor_compiles += 1
        current_config = {}
        if compiler_config is not None:
            current_config.update(compiler_config)
        set_inductor_config(current_config, compile_range)
        set_functorch_config()

        if compile_range.is_single_size():
            dynamic_shapes = "from_example_inputs"
        else:
            dynamic_shapes = "from_graph"

        from torch._inductor import standalone_compile

        supports_aot = is_torch_equal_or_newer("2.10.0.dev")

        if not supports_aot and envs.VLLM_USE_MEGA_AOT_ARTIFACT:
            logger.error(
                "CRITICAL: VLLM_USE_MEGA_AOT_ARTIFACT "
                "is enabled but PyTorch version does not support 'aot' "
                "parameter in standalone_compile. This requires PyTorch "
                "2.10.0+. Falling back to non-AOT mode."
            )

        compile_kwargs = {
            "dynamic_shapes": dynamic_shapes,
            "options": {
                "config_patches": current_config,
            },
        }

        use_aot: bool = supports_aot and envs.VLLM_USE_MEGA_AOT_ARTIFACT
        # only add 'aot' parameter if both supported and enabled...
        # this will set bundled_autograd_cache
        # https://github.com/pytorch/pytorch/blob/9bbc5b2905c260adf41bc866a732f9c121a2828a/torch/_inductor/standalone_compile.py#L359 # noqa
        if use_aot:
            compile_kwargs["aot"] = True  # type: ignore[assignment]

        compiled_graph = standalone_compile(graph, example_inputs, **compile_kwargs)

        if use_aot:
            from torch._inductor.standalone_compile import AOTCompiledArtifact

            assert isinstance(compiled_graph, AOTCompiledArtifact)
            assert hasattr(compiled_graph, "serialize")
            # just return the compiled graph and a key
            # since we can serialize the bytes using to_bytes
            # and reload it using the key when reading
            return compiled_graph, None

        # Save the compiled artifact to disk in the specified path
        assert key is not None
        path = os.path.join(self.cache_dir, key)

        if is_compile_cache_enabled(compiler_config):
            compiled_graph.save(path=path, format=self.save_format)
            compilation_counter.num_compiled_artifacts_saved += 1
        return compiled_graph, (key, path)

    def load(
        self,
        handle: Any,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        graph_index: int,
        compile_range: Range,
    ) -> Callable[..., Any]:
        assert isinstance(handle, tuple)
        assert isinstance(handle[0], str)
        assert isinstance(handle[1], str)
        path = handle[1]
        inductor_compiled_graph = torch._inductor.CompiledArtifact.load(
            path=path, format=self.save_format
        )
        from torch._inductor.compile_fx import graph_returns_tuple

        returns_tuple = graph_returns_tuple(graph)

        def compiled_graph_wrapper(*args: Any) -> tuple[Any, ...] | Any:
            graph_output = inductor_compiled_graph(*args)
            # unpack the tuple if needed
            # TODO(rzou): the implication is that we're not
            # reading the python bytecode correctly in vLLM?
            if returns_tuple:
                return graph_output
            else:
                return graph_output[0]

        return compiled_graph_wrapper


class InductorAdaptor(CompilerInterface):
    """
    The adaptor for the Inductor compiler, version 2.5, 2.6, 2.7.
    """

    name = "inductor"

    def compute_hash(self, vllm_config: VllmConfig) -> str:
        factors = get_inductor_factors()
        hash_str: str = safe_hash(
            str(factors).encode(), usedforsecurity=False
        ).hexdigest()[:10]
        return hash_str

    def initialize_cache(
        self, cache_dir: str, disable_cache: bool = False, prefix: str = ""
    ) -> None:
        self.cache_dir = cache_dir
        self.prefix = prefix
        self.base_cache_dir = cache_dir[: -len(prefix)] if prefix else cache_dir
        if disable_cache:
            return
        # redirect the cache directory to a subdirectory
        # set flags so that Inductor and Triton store their cache
        # in the cache_dir, then users only need to copy the cache_dir
        # to another machine to reuse the cache.
        inductor_cache = os.path.join(self.base_cache_dir, "inductor_cache")
        os.makedirs(inductor_cache, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache
        triton_cache = os.path.join(self.base_cache_dir, "triton_cache")
        os.makedirs(triton_cache, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = triton_cache

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        compile_range: Range,
        key: str | None = None,
    ) -> tuple[Callable[..., Any] | None, Any | None]:
        compilation_counter.num_inductor_compiles += 1
        from torch._inductor.compile_fx import compile_fx

        current_config = {}
        if compiler_config is not None:
            current_config.update(compiler_config)

        # disable remote cache
        current_config["fx_graph_cache"] = True
        current_config["fx_graph_remote_cache"] = False

        set_inductor_config(current_config, compile_range)
        set_functorch_config()

        # inductor can inplace modify the graph, so we need to copy it
        # see https://github.com/pytorch/pytorch/issues/138980
        graph = copy.deepcopy(graph)

        # it's the first time we compile this graph
        # the assumption is that we don't have nested Inductor compilation.
        # compiled_fx_graph_hash will only be called once, and we can hook
        # it to get the hash of the compiled graph directly.

        hash_str, file_path = None, None
        from torch._inductor.codecache import FxGraphCache, compiled_fx_graph_hash

        if torch.__version__.startswith("2.5"):
            original_load = FxGraphCache.load
            original_load_name = "torch._inductor.codecache.FxGraphCache.load"

            def hijack_load(*args: Any, **kwargs: Any) -> Any:
                inductor_compiled_graph = original_load(*args, **kwargs)
                nonlocal file_path
                compiled_fn = inductor_compiled_graph.current_callable
                file_path = compiled_fn.__code__.co_filename  # noqa
                if (
                    not file_path.startswith(self.base_cache_dir)
                    and compiled_fn.__closure__ is not None
                ):
                    # hooked in the align_inputs_from_check_idxs function
                    # in torch/_inductor/utils.py
                    for cell in compiled_fn.__closure__:
                        if not callable(cell.cell_contents):
                            continue
                        if cell.cell_contents.__code__.co_filename.startswith(
                            self.base_cache_dir
                        ):
                            # this is the real file path compiled from Inductor
                            file_path = cell.cell_contents.__code__.co_filename
                            break
                return inductor_compiled_graph

            hijacked_compile_fx_inner = torch._inductor.compile_fx.compile_fx_inner  # noqa
        elif torch.__version__ >= "2.6":
            # function renamed in 2.6
            original_load_name = None

            def hijacked_compile_fx_inner(*args: Any, **kwargs: Any) -> Any:
                output = torch._inductor.compile_fx.compile_fx_inner(*args, **kwargs)
                nonlocal hash_str
                inductor_compiled_graph = output
                if inductor_compiled_graph is not None:
                    nonlocal file_path
                    compiled_fn = inductor_compiled_graph.current_callable
                    file_path = compiled_fn.__code__.co_filename  # noqa
                    if (
                        not file_path.startswith(self.base_cache_dir)
                        and compiled_fn.__closure__ is not None
                    ):
                        # hooked in the align_inputs_from_check_idxs function
                        # in torch/_inductor/utils.py
                        for cell in compiled_fn.__closure__:
                            if not callable(cell.cell_contents):
                                continue
                            code = cell.cell_contents.__code__
                            if code.co_filename.startswith(self.base_cache_dir):
                                # this is the real file path
                                # compiled from Inductor
                                file_path = code.co_filename
                                break
                    hash_str = inductor_compiled_graph._fx_graph_cache_key
                return output

        def hijack_compiled_fx_graph_hash(*args: Any, **kwargs: Any) -> Any:
            out = compiled_fx_graph_hash(*args, **kwargs)
            nonlocal hash_str
            hash_str = out[0]
            return out

        def _check_can_cache(*args: Any, **kwargs: Any) -> None:
            # no error means it can be cached.
            # Inductor refuses to cache the graph outside of Dynamo
            # tracing context, and also disables caching for graphs
            # with high-order ops.
            # For vLLM, in either case, we want to cache the graph.
            # see https://github.com/pytorch/pytorch/blob/9f5ebf3fc609105a74eab4ccc24932d6353ff566/torch/_inductor/codecache.py#L1221 # noqa
            return

        def _get_shape_env() -> AlwaysHitShapeEnv:
            return AlwaysHitShapeEnv()

        with ExitStack() as stack:
            # hijack to get the compiled graph itself
            if original_load_name is not None:
                stack.enter_context(patch(original_load_name, hijack_load))

            # for hijacking the hash of the compiled graph
            stack.enter_context(
                patch(
                    "torch._inductor.codecache.compiled_fx_graph_hash",
                    hijack_compiled_fx_graph_hash,
                )
            )

            # for providing a dummy shape environment
            stack.enter_context(
                patch(
                    "torch._inductor.codecache.FxGraphCache._get_shape_env",
                    _get_shape_env,
                )
            )

            from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache

            # torch 2.8+ on main uses _get_shape_env in AOTAutogradCache
            if hasattr(AOTAutogradCache, "_get_shape_env"):
                stack.enter_context(
                    patch(
                        "torch._functorch._aot_autograd.autograd_cache.AOTAutogradCache._get_shape_env",
                        _get_shape_env,
                    )
                )

            # for forcing the graph to be cached
            stack.enter_context(
                patch(
                    "torch._inductor.codecache.FxGraphCache._check_can_cache",
                    _check_can_cache,
                )
            )

            # Dynamo metrics context, see method for more details.
            stack.enter_context(self.metrics_context())

            # Disable remote caching. When these are on, on remote cache-hit,
            # the monkey-patched functions never actually get called.
            # vLLM today assumes and requires the monkey-patched functions to
            # get hit.
            # TODO(zou3519): we're going to replace this all with
            # standalone_compile sometime.
            if is_torch_equal_or_newer("2.6"):
                stack.enter_context(
                    torch._inductor.config.patch(fx_graph_remote_cache=False)
                )
                # InductorAdaptor (unfortunately) requires AOTAutogradCache
                # to be turned off to run. It will fail to acquire the hash_str
                # and error if not.
                # StandaloneInductorAdaptor (PyTorch 2.8+) fixes this problem.
                stack.enter_context(
                    torch._functorch.config.patch(enable_autograd_cache=False)
                )
                stack.enter_context(
                    torch._functorch.config.patch(enable_remote_autograd_cache=False)
                )

            compiled_graph = compile_fx(
                graph,
                example_inputs,
                inner_compile=hijacked_compile_fx_inner,
                config_patches=current_config,
            )

        # Turn off the checks if we disable the compilation cache.
        if is_compile_cache_enabled(compiler_config):
            if hash_str is None:
                raise RuntimeError(
                    "vLLM failed to compile the model. The most "
                    "likely reason for this is that a previous compilation "
                    "failed, leading to a corrupted compilation artifact. "
                    "We recommend trying to "
                    "remove ~/.cache/vllm/torch_compile_cache and try again "
                    "to see the real issue. "
                )
            assert file_path is not None, (
                "failed to get the file path of the compiled graph"
            )
        return compiled_graph, (hash_str, file_path)

    def load(
        self,
        handle: Any,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        graph_index: int,
        compile_range: Range,
    ) -> Callable[..., Any]:
        assert isinstance(handle, tuple)
        assert isinstance(handle[0], str)
        assert isinstance(handle[1], str)
        hash_str = handle[0]

        from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
        from torch._inductor.codecache import FxGraphCache

        with ExitStack() as exit_stack:
            exit_stack.enter_context(
                patch(
                    "torch._inductor.codecache.FxGraphCache._get_shape_env",
                    lambda *args, **kwargs: AlwaysHitShapeEnv(),
                )
            )
            # torch 2.8+ on main uses _get_shape_env in AOTAutogradCache
            if hasattr(AOTAutogradCache, "_get_shape_env"):
                exit_stack.enter_context(
                    patch(
                        "torch._functorch._aot_autograd.autograd_cache.AOTAutogradCache._get_shape_env",
                        lambda *args, **kwargs: AlwaysHitShapeEnv(),
                    )
                )

            # Dynamo metrics context, see method for more details.
            exit_stack.enter_context(self.metrics_context())

            if torch.__version__.startswith("2.5"):
                inductor_compiled_graph = FxGraphCache._lookup_graph(
                    hash_str, example_inputs, True, False
                )
                assert inductor_compiled_graph is not None, (
                    "Inductor cache lookup failed. Please remove "
                    f"the cache directory and try again."  # noqa
                )
            elif torch.__version__ >= "2.6":
                from torch._inductor.output_code import CompiledFxGraphConstantsWithGm

                constants = CompiledFxGraphConstantsWithGm(graph)
                inductor_compiled_graph, _ = FxGraphCache._lookup_graph(
                    hash_str, example_inputs, True, None, constants
                )
                assert inductor_compiled_graph is not None, (
                    "Inductor cache lookup failed. Please remove "
                    f"the cache directory and try again."  # noqa
                )

        # Inductor calling convention (function signature):
        # f(list) -> tuple
        # Dynamo calling convention (function signature):
        # f(*args) -> Any

        # need to know if the graph returns a tuple
        from torch._inductor.compile_fx import graph_returns_tuple

        returns_tuple = graph_returns_tuple(graph)

        # this is the callable we return to Dynamo to run
        def compiled_graph(*args: Any) -> tuple[Any, ...] | Any:
            # convert args to list
            list_args = list(args)
            graph_output = inductor_compiled_graph(list_args)
            # unpack the tuple if needed
            if returns_tuple:
                return graph_output
            else:
                return graph_output[0]

        return compiled_graph

    def metrics_context(self) -> contextlib.AbstractContextManager[Any]:
        """
        This method returns the Dynamo metrics context (if it exists,
        otherwise a null context). It is used by various compile components.
        Present in torch>=2.6, it's used inside FxGraphCache in
        torch==2.6 (but not after). It might also be used in various other
        torch.compile internal functions.

        Because it is re-entrant, we always set it (even if entering via Dynamo
        and the context was already entered). We might want to revisit if it
        should be set at a different mode of compilation.

        This is likely a bug in PyTorch: public APIs should not rely on
        manually setting up internal contexts. But we also rely on non-public
        APIs which might not provide these guarantees.
        """
        if is_torch_equal_or_newer("2.6"):
            import torch._dynamo.utils

            return torch._dynamo.utils.get_metrics_context()  # type: ignore[no-any-return]
        else:
            return contextlib.nullcontext()


def set_inductor_config(config: dict[str, Any], compile_range: Range) -> None:
    if compile_range.is_single_size():
        # for a specific batch size, tuning triton kernel parameters
        # can be beneficial
        config["max_autotune"] = envs.VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE
        config["coordinate_descent_tuning"] = (
            envs.VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING
        )


def set_functorch_config() -> None:
    if not envs.VLLM_USE_MEGA_AOT_ARTIFACT:
        torch._functorch.config.bundled_autograd_cache = False


class EagerAdaptor(CompilerInterface):
    name = "eager"

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        compile_range: Range,
        key: str | None = None,
    ) -> tuple[Callable[..., Any] | None, Any | None]:
        compilation_counter.num_eager_compiles += 1
        # we don't need to compile the graph, just return the graph itself.
        # It does not support caching, return None for the handle.
        return graph, None
