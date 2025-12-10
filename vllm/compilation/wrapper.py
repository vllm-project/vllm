# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys
from abc import abstractmethod
from contextlib import contextmanager, nullcontext
from types import CodeType
from typing import Any

import torch
import torch._C._dynamo.guards

import vllm.envs as envs
from vllm.config import CompilationMode, CUDAGraphMode, get_current_vllm_config
from vllm.config.compilation import DynamicShapesType
from vllm.logger import init_logger
from vllm.utils.nvtx_pytorch_hooks import layerwise_nvtx_marker_context

logger = init_logger(__name__)


def _noop_add_global_state_guard(self, *args, **kwargs):
    """No-op to skip the GLOBAL_STATE guard entirely"""
    pass


def _noop_add_torch_function_mode_stack_guard(self, *args, **kwargs):
    """No-op to skip the TORCH_FUNCTION_MODE_STACK guard entirely"""
    pass


@contextmanager
def _compilation_context():
    """Context manager for compilation settings and patches.

    This manager:
    1. Sets higher dynamo cache limits for compilation. (Needed for
        qwen2_5_vl see test_qwen2_5_vl_evs_functionality).
        Generally a recompilation can happen whenever we use a new
        backend instance in torch.compile.
    2. Patches out add_global_state_guard to skip GLOBAL_STATE guards
    3. Patches out add_torch_function_mode_stack_guard to skip
        TORCH_FUNCTION_MODE_STACK guards.
    4. Restores everything when compilation completes
    """
    # Save original values
    original_global_state_guard = (
        torch._C._dynamo.guards.GuardManager.add_global_state_guard
    )
    original_torch_function_mode_stack_guard = (
        torch._C._dynamo.guards.GuardManager.add_torch_function_mode_stack_guard
    )
    original_cache_size = torch._dynamo.config.cache_size_limit
    original_accumulated_cache = torch._dynamo.config.accumulated_cache_size_limit

    try:
        # Set higher cache limits for compilation
        torch._dynamo.config.cache_size_limit = 2048
        torch._dynamo.config.accumulated_cache_size_limit = 8192

        # Patch guard manager
        torch._C._dynamo.guards.GuardManager.add_global_state_guard = (
            _noop_add_global_state_guard
        )
        torch._C._dynamo.guards.GuardManager.add_torch_function_mode_stack_guard = (
            _noop_add_torch_function_mode_stack_guard
        )
        yield
    finally:
        # Restore original values
        torch._C._dynamo.guards.GuardManager.add_global_state_guard = (
            original_global_state_guard
        )
        torch._C._dynamo.guards.GuardManager.add_torch_function_mode_stack_guard = (
            original_torch_function_mode_stack_guard
        )
        torch._dynamo.config.cache_size_limit = original_cache_size
        torch._dynamo.config.accumulated_cache_size_limit = original_accumulated_cache


class TorchCompileWithNoGuardsWrapper:
    """
    A wrapper class for torch.compile, it ensures that all guards are dropped
    when CompilationMode is not CompilationMode.STOCK_TORCH_COMPILE.
    When guards are dropped, the first time __call__ is invoked, a single
    compilation is triggered. Dynamo should never be traced again after that
    since we drop all guards.
    """

    def check_invariants_and_forward(self, *args, **kwargs):
        assert hasattr(self, "_check_shape_invariants")
        self._check_shape_invariants(*args, **kwargs)

        return self.forward(*args, **kwargs)

    def _call_with_optional_nvtx_range(self, callable_fn, *args, **kwargs):
        if self.layerwise_nvtx_tracing_enabled:
            args_list = list(args)
            kwargs_dict = dict(kwargs)
            with layerwise_nvtx_marker_context(
                "Torch Compiled Module (input):{}".format(self.__class__.__name__),
                self,
                in_tensor=args_list,
                kwargs=kwargs_dict,
            ) as ctx:
                ctx.result = callable_fn(*args, **kwargs)
            return ctx.result
        return callable_fn(*args, **kwargs)

    def __init__(self):
        self.compiled = False

        vllm_config = get_current_vllm_config()
        self.vllm_config = vllm_config
        mode = vllm_config.compilation_config.mode
        self.layerwise_nvtx_tracing_enabled = (
            vllm_config.observability_config.enable_layerwise_nvtx_tracing
        )
        if mode is None:
            raise RuntimeError("Compilation mode cannot be NO_COMPILATION")

        backend = vllm_config.compilation_config.init_backend(vllm_config)
        options = {}

        if isinstance(backend, str) and backend == "inductor":
            options = vllm_config.compilation_config.inductor_compile_config

        self.first_compile = True
        self.evaluate_guards = (
            vllm_config.compilation_config.dynamic_shapes_config.evaluate_guards
        )

        ds_type = vllm_config.compilation_config.dynamic_shapes_config.type

        if mode != CompilationMode.STOCK_TORCH_COMPILE:
            # Drop all the guards.
            if self.evaluate_guards:
                assert not envs.VLLM_USE_BYTECODE_HOOK, (
                    "compilation_config.dynamic_shapes_config.evaluate_guards "
                    "requires VLLM_USE_BYTECODE_HOOK=0. "
                )

                if envs.VLLM_USE_AOT_COMPILE:
                    # disabled until https://github.com/pytorch/pytorch/pull/169239
                    # is picked up.
                    assert ds_type != DynamicShapesType.BACKED, (
                        "evaluate_guards for backed shapes requires "
                        "VLLM_USE_AOT_COMPILE=False. "
                    )

                options["guard_filter_fn"] = lambda x: [
                    entry.guard_type == "SHAPE_ENV" for entry in x
                ]
            else:
                options["guard_filter_fn"] = lambda x: [False for _ in x]

        compiled_ptr: Any = self.forward
        # Validate that unbacked dynamic shapes require VLLM_USE_BYTECODE_HOOK=False

        if ds_type == DynamicShapesType.UNBACKED:
            # reason is that bytecode does torch._dynamo.eval_frame.
            # remove_from_cache(self.original_code_object()) to force a new
            # re-compilation. And if we use
            # compiled_ptr = self.check_invariants_and_forward
            # it will reset all entries.
            assert not envs.VLLM_USE_BYTECODE_HOOK, (
                "UNBACKED dynamic shapes requires VLLM_USE_BYTECODE_HOOK=0. "
            )
            assert not self.evaluate_guards, "UNBACKED dynamic shapes do not add guards"

            compiled_ptr = self.check_invariants_and_forward

        if envs.VLLM_USE_AOT_COMPILE:
            if hasattr(torch._dynamo.config, "enable_aot_compile"):
                torch._dynamo.config.enable_aot_compile = True
            else:
                msg = "torch._dynamo.config.enable_aot_compile is not "
                msg += "available. AOT compile is disabled and please "
                msg += "upgrade PyTorch version to use AOT compile."
                logger.warning(msg)

        self._compiled_callable = torch.compile(
            compiled_ptr,
            fullgraph=True,
            dynamic=False,
            backend=backend,
            options=options,
        )

        if envs.VLLM_USE_BYTECODE_HOOK and mode != CompilationMode.STOCK_TORCH_COMPILE:
            torch._dynamo.convert_frame.register_bytecode_hook(self.bytecode_hook)
            self._compiled_bytecode = None

    def aot_compile(self, *args, **kwargs):
        if not hasattr(self._compiled_callable, "aot_compile"):
            raise RuntimeError(
                "aot_compile is not supported by the current configuration. "
                + "Please make sure torch.compile is enabled with the latest "
                + f"version of PyTorch (current using torch: {torch.__version__})"
            )
        return self._compiled_callable.aot_compile((args, kwargs))

    def __call__(self, *args, **kwargs):
        if envs.VLLM_USE_BYTECODE_HOOK:
            if (
                self.vllm_config.compilation_config.mode
                == CompilationMode.STOCK_TORCH_COMPILE
            ):
                return self._compiled_callable(*args, **kwargs)

            if not self._compiled_bytecode:
                # Make sure a compilation is triggered by clearing dynamo
                # cache.
                torch._dynamo.eval_frame.remove_from_cache(self.original_code_object())
                return self._call_with_optional_nvtx_range(
                    self._compiled_callable, *args, **kwargs
                )
            else:
                with self._dispatch_to_compiled_code():
                    return self._call_with_optional_nvtx_range(
                        self.forward, *args, **kwargs
                    )
        else:
            ctx = (
                nullcontext()
                if self.first_compile or not self.evaluate_guards
                else torch.compiler.set_stance("fail_on_recompile")
            )
            self.first_compile = False
            with _compilation_context(), ctx:
                return self._call_with_optional_nvtx_range(
                    self._compiled_callable, *args, **kwargs
                )

    @abstractmethod
    def forward(self, *args, **kwargs): ...

    def original_code_object(self) -> CodeType:
        """Return the original code object of the forward method."""
        return self.__class__.forward.__code__

    def bytecode_hook(self, old_code: CodeType, new_code: CodeType):
        """Hook to save the compiled bytecode for direct execution."""
        if old_code is not self.original_code_object():
            return
        # code borrowed from https://github.com/thuml/depyf/blob/f4ad79fadee27ea113b4c75202db1eb1a11c0dbc/depyf/explain/enable_debugging.py#L25
        frame = sys._getframe()
        while frame and frame.f_back:
            frame = frame.f_back
            code_name = frame.f_code.co_name
            file_name = frame.f_code.co_filename.split(os.path.sep)[-1]
            if code_name == "_compile" and file_name == "convert_frame.py":
                break
        frame = frame.f_locals["frame"]
        assert frame.f_code == old_code

        if frame.f_locals["self"] is not self:
            return

        self._compiled_bytecode = new_code

        path = self.vllm_config.compile_debug_dump_path()
        if path:
            decompiled_file = path / "transformed_code.py"
            if not decompiled_file.exists():
                try:
                    # usually the decompilation will succeed for most models,
                    # as we guarantee a full-graph compilation in Dynamo.
                    # but there's no 100% guarantee, since decompliation is
                    # not a reversible process.
                    import depyf

                    src = depyf.decompile(new_code)

                    with open(decompiled_file, "w") as f:
                        f.write(src)

                    logger.debug("Dynamo transformed code saved to %s", decompiled_file)
                except Exception:
                    pass

        if (
            self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            and "update" in new_code.co_names
        ):
            import depyf

            src = depyf.decompile(new_code)
            msg = (
                "Assigning / modifying buffers of nn.Module during forward pass is not "
                "allowed when using cudagraph inside the compiler because it will "
                "cause silent errors. Please use eager mode or fix the code. The "
                "following code contains clues about which buffer is being modified "
                f"(please search for the usage of the function `update`):\n{src}"
            )
            raise RuntimeError(msg)

    @contextmanager
    def _dispatch_to_compiled_code(self):
        # noqa: E501
        """
        Context manager to dispatch to internally compiled code for torch<2.8.
        Why does this work? Because Dynamo guarantees that the compiled
        bytecode has exactly the same arguments, cell variables, and free
        variables as the original code. Therefore we can directly switch
        the code object in the function and call it.

        See https://dev-discuss.pytorch.org/t/what-is-the-relationship-requirement-among-original-bytecode-transformed-bytecode-and-bytecode-returned-by-hooks-in-dynamo/1693/7 for more details.
        """  # noqa: E501 line too long
        original = self.original_code_object()
        assert self._compiled_bytecode is not None
        self.__class__.forward.__code__ = self._compiled_bytecode
        try:
            yield
        finally:
            self.__class__.forward.__code__ = original
