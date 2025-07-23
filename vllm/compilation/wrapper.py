# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys
from abc import abstractmethod
from contextlib import contextmanager
from types import CodeType
from typing import Optional

import torch

import vllm.envs as envs
from vllm.config import CompilationLevel, get_current_vllm_config
from vllm.logger import init_logger

logger = init_logger(__name__)


class TorchCompileWrapperWithCustomDispatcher:
    """
    A wrapper class around torch.compile that guarantees a single compilation.
    For Torch 2.8 and later, it drops the guards to prevent recompilation and 
    uses the fail_on_recompile stance for further guarantee.

    For Torch versions prior to 2.8, it uses a bytecode hook to access the 
    compiled function and caches it in memory to avoid recompilation.
    
    When compilation_level = DYNAMO_AS_IS, guards are not dropped, and 
    recompilation can occur. Note that in DYNAMO_AS_IS mode, 
    compilation_config.backend is used instead of VllmBackend.

    """

    def __init__(self):

        vllm_config = get_current_vllm_config()
        self.vllm_config = vllm_config

        compilation_config = vllm_config.compilation_config
        self.compilation_config = compilation_config

        assert compilation_config.level != CompilationLevel.NO_COMPILATION

        backend = compilation_config.init_backend(vllm_config)

        options = {}
        if isinstance(backend, str) and backend == "inductor":
            options = compilation_config.inductor_compile_config

        self.eval_shape_guards: bool = compilation_config.eval_shape_guards

        # Whether the first compilation has happened or not.
        self.compiled: bool = False

        # The code object of the original forward function.
        self.original_code_object = self.__class__.forward.__code__

        if (torch.__version__ >= "2.8"
                and compilation_config.level != CompilationLevel.DYNAMO_AS_IS):
            if self.eval_shape_guards:
                options['guard_filter_fn'] = lambda x: [
                    entry.guard_type == "SHAPE_ENV" for entry in x
                ]
            else:
                options['guard_filter_fn'] = lambda x: [False for _ in x]

        else:
            assert not self.eval_shape_guards, (
                "Evaluating Shape guards is only supported in PyTorch 2.8+")
            self.compiled_code: Optional[CodeType] = None

        self._compiled_callable = torch.compile(
            self.forward,
            fullgraph=envs.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
            backend=backend,
            options=options)

        torch._dynamo.convert_frame.register_bytecode_hook(self.bytecode_hook)

    def __call__(self, *args, **kwargs):
        if not self.compiled:
            # Make sure a compilation is triggered by clearing dynamo cache.
            torch._dynamo.eval_frame.remove_from_cache(
                self.original_code_object)
            self.compiled = True

        if (torch.__version__ >= "2.8" or self.compilation_config.level
                == CompilationLevel.DYNAMO_AS_IS):
            return self._compiled_callable(*args, **kwargs)
        else:
            with self._dispatch_to_compiled_code():
                return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        ...

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

        # we do not use compiled_code for torch.__version__ >= 2.8
        if torch.__version__ < "2.8":
            assert self.compiled_code is None
            self.compiled_code = new_code

        local_cache_dir = self.vllm_config.compilation_config.local_cache_dir
        if isinstance(local_cache_dir, str):

            decompiled_file = os.path.join(local_cache_dir,
                                           "transformed_code.py")
            if not os.path.exists(decompiled_file):
                try:
                    # usually the decompilation will succeed for most models,
                    # as we guarantee a full-graph compilation in Dynamo.
                    # but there's no 100% guarantee, since decompliation is
                    # not a reversible process.

                    import depyf
                    src = depyf.decompile(new_code)
                    with open(decompiled_file, "w") as f:
                        f.write(src)

                    logger.debug("Dynamo transformed code saved to %s",
                                 decompiled_file)
                except Exception:
                    pass

        if (self.vllm_config.compilation_config.use_cudagraph
                and "update" in new_code.co_names):
            import depyf
            src = depyf.decompile(new_code)
            msg = (
                "Assigning / modifying buffers of nn.Module during forward "
                "pass is not allowed when using cudagraph inside the compiler "
                "because it will cause silent errors. Please use eager mode "
                "or fix the code. The following code contains clues about "
                "which buffer is being modified (please search for the usage "
                "of the function `update`):\n") + src
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
        """ # noqa: E501 line too long

        self.__class__.forward.__code__ = self.compiled_code
        try:
            yield
        finally:
            self.__class__.forward.__code__ = self.original_code_object
