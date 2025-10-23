# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from contextlib import nullcontext
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
    since we drop all guards.
    """

    def check_invariantes_and_forward(self, *args, **kwargs):
        assert hasattr(self, "_check_shape_invariants")
        self._check_shape_invariants(*args, **kwargs)

        return self.forward(*args, **kwargs)

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
        self.first_compile = True
        if mode != CompilationMode.STOCK_TORCH_COMPILE:
            # Drop all the guards.
            if vllm_config.compilation_config.dynamic_shapes_config.evaluate_guards:
                options["guard_filter_fn"] = lambda x: [
                    entry.guard_type == "SHAPE_ENV" for entry in x
                ]
            else:
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
            self.check_invariantes_and_forward,
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
        prev = self.first_compile
        self.first_compile = False
        ctx = nullcontext() if prev else torch.compiler.set_stance("fail_on_recompile")
        with ctx:
            return self._compiled_callable.aot_compile((args, kwargs))

    def __call__(self, *args, **kwargs):
        prev = self.first_compile
        self.first_compile = False
        ctx = nullcontext() if prev else torch.compiler.set_stance("fail_on_recompile")
        with ctx:
            return self._compiled_callable(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs): ...

    def original_code_object(self) -> CodeType:
        """Return the original code object of the forward method."""
        return self.__class__.forward.__code__
