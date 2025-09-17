# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from types import CodeType

import torch

import vllm.envs as envs
from vllm.config import CompilationMode, get_current_vllm_config
from vllm.logger import init_logger

logger = init_logger(__name__)


class TorchCompileGuardsStripWrapper:
    """
    A wrapper class for torch.compile, it ensures that all guards are dropped
    when CompilationMode is not CompilationMode.STOCK_TORCH_COMPILE.
    When guards are dropped, the first time __call__ is invoked, a single
    compilation is triggered. Dynamo should never be traced again after that
    (Since we drop all guards)
    """

    def __init__(self):
        self.compiled = False

        vllm_config = get_current_vllm_config()
        mode = vllm_config.compilation_config.mode
        if mode is None:
            raise RuntimeError("Compilation mode cannot be NO_COMPILATION")

        backend = vllm_config.compilation_config.init_backend(vllm_config)
        options = {}

        if isinstance(backend, str) and backend == "inductor":
            options = vllm_config.compilation_config.inductor_compile_config

        if mode != CompilationMode.STOCK_TORCH_COMPILE:
            # Drop all the guards.
            options["guard_filter_fn"] = lambda x: [False for _ in x]

        if envs.VLLM_USE_AOT_COMPILE:
            if hasattr(torch._dynamo.config, "enable_aot_compile"):
                torch._dynamo.config.enable_aot_compile = True
            else:
                msg = "torch._dynamo.config.enable_aot_compile is not "
                msg += "available. AOT compile is disabled and please "
                msg += "upgrade PyTorch version to use AOT compile."
                logger.warning(msg)

        self._compiled_callable = torch.compile(
            self.forward,
            fullgraph=True,
            backend=backend,
            options=options,
        )

    def aot_compile(self, *args, **kwargs):
        if not hasattr(self._compiled_callable, "aot_compile"):
            raise RuntimeError(
                "aot_compile is not supported by the current configuration. "
                + "Please make sure torch.compile is enabled with the latest "
                + f"version of PyTorch (current using torch: {torch.__version__})"
            )
        return self._compiled_callable.aot_compile((args, kwargs))

    def __call__(self, *args, **kwargs):
        return self._compiled_callable(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs): ...

    def original_code_object(self) -> CodeType:
        """Return the original code object of the forward method."""
        return self.__class__.forward.__code__
