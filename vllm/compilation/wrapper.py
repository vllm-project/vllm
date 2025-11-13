# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys
from abc import abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from types import CodeType

import torch

import vllm.envs as envs
from vllm.config import CompilationMode, CUDAGraphMode, get_current_vllm_config
from vllm.logger import init_logger

logger = init_logger(__name__)


class TorchCompileWrapperWithCustomDispatcher:
    """
    A wrapper class for torch.compile, with a custom dispatch logic.
    Subclasses should:
    1. Implement the forward method
    2. Implement the dispatch logic in the __call__ method
        It can use `self.compiled_codes` to access the compiled bytecode,
        and `with self.dispatch_to_code(index):` to dispatch to
        the compiled code.
    3. Implement the `__init__` method to determine how to call
        `torch.compile` over the forward method.
    """

    def __init__(
        self,
        compiled_callable: Callable | None = None,
        compilation_mode: CompilationMode = CompilationMode.NONE,
    ):
        vllm_config = get_current_vllm_config()
        self.vllm_config = vllm_config
        if compiled_callable is None:
            # default compilation settings
            # compiling the forward method

            backend = vllm_config.compilation_config.init_backend(vllm_config)
            options = None
            if isinstance(backend, str) and backend == "inductor":
                options = (
                    get_current_vllm_config().compilation_config.inductor_compile_config
                )
            if envs.VLLM_USE_AOT_COMPILE:
                options = options or {}
                # This effectively drop all the guards.
                # We need this because bytecode hook is not used any more to
                # drop guards in the AOT compile mode.
                options["guard_filter_fn"] = lambda guards: [False for _ in guards]
                if hasattr(torch._dynamo.config, "enable_aot_compile"):
                    torch._dynamo.config.enable_aot_compile = True
                else:
                    msg = "torch._dynamo.config.enable_aot_compile is not "
                    msg += "available. AOT compile is disabled and please "
                    msg += "upgrade PyTorch version to use AOT compile."
                    logger.warning(msg)

            compiled_callable = torch.compile(
                self.forward, fullgraph=True, backend=backend, options=options
            )

        self.compiled_callable = compiled_callable
        self.original_code_object = self.__class__.forward.__code__
        self.compiled_codes: list[CodeType] = []
        torch._dynamo.convert_frame.register_bytecode_hook(self.bytecode_hook)

        # read the env var to determine whether to use the custom dispatcher
        # subclasses can use this to switch between the custom dispatcher
        # and the default Dynamo guard mechanism.
        self.use_custom_dispatcher: bool = (
            compilation_mode >= CompilationMode.DYNAMO_TRACE_ONCE
        )

    def aot_compile(self, *args, **kwargs):
        if not hasattr(self.compiled_callable, "aot_compile"):
            raise RuntimeError(
                "aot_compile is not supported by the current configuration. "
                + "Please make sure torch.compile is enabled with the latest "
                + f"version of PyTorch (current using torch: {torch.__version__})"
            )
        return self.compiled_callable.aot_compile((args, kwargs))

    def __call__(self, *args, **kwargs):
        """Implement the dispatch logic here, beyond the torch.compile mode.
        NOTE: this function can have additional arguments beyond the forward
         method, for directly dispatching to the compiled code.
        """
        return self.compiled_callable(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs): ...

    def bytecode_hook(self, old_code: CodeType, new_code: CodeType):
        """Hook to save the compiled bytecode for direct execution."""
        if old_code is not self.original_code_object:
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

        self.compiled_codes.append(new_code)

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
    def dispatch_to_code(self, index: int):
        """Context manager to dispatch to the compiled code.
        Why does this work? Because Dynamo guarantees that the compiled
        bytecode has exactly the same arguments, cell variables, and free
        variables as the original code. Therefore we can directly switch
        the code object in the function and call it.

        See https://dev-discuss.pytorch.org/t/what-is-the-relationship-requirement-among-original-bytecode-transformed-bytecode-and-bytecode-returned-by-hooks-in-dynamo/1693/7
        for more details.
        """
        self.__class__.forward.__code__ = self.compiled_codes[index]
        yield
        self.__class__.forward.__code__ = self.original_code_object
