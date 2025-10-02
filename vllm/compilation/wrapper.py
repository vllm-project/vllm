# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from types import CodeType

import torch

import vllm.envs as envs
from vllm.config import CompilationLevel, get_current_vllm_config
from vllm.logger import init_logger

logger = init_logger(__name__)


class TorchCompileGuardsStripWrapper:
    """
    A wrapper class for torch.compile, it ensures that all guards are dropped
    when CompilationLevel is not CompilationLevel.DYNAMO_AS_IS.
    When guards are dropped, the first time __call__ is invoked, a single
    compilation is triggered. Dynamo should never be traced again after that
    (Since we drop all guards)
    """

    def check_invariantes_and_forward(self, *args, **kwargs):
        self._check_shape_invariants(*args, **kwargs)

        return self.forward(*args, **kwargs)

    def __init__(self):
        self.compiled = False

        vllm_config = get_current_vllm_config()
        level = vllm_config.compilation_config.level
        if level == CompilationLevel.NO_COMPILATION:
            raise RuntimeError("Compilation level cannot be NO_COMPILATION")

        backend = vllm_config.compilation_config.init_backend(vllm_config)

        options = {}

        if isinstance(backend, str) and backend == "inductor":
            options = vllm_config.compilation_config.inductor_compile_config

        if level != CompilationLevel.DYNAMO_AS_IS:
            # Drop all the guards.
            options["guard_filter_fn"] = lambda x: [False for _ in x]

        self._compiled_callable = torch.compile(
            self.check_invariantes_and_forward,
            fullgraph=envs.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
            backend=backend,
            options=options,
        )

    def __call__(self, *args, **kwargs):
        """Implement the dispatch logic here, beyond the torch.compile level.
        NOTE: this function can have additional arguments beyond the forward
         method, for directly dispatching to the compiled code.
        """
        if not self.compiled:
            # We check eagirly on the first compile as well.
            self.check_invariantes_and_forward(*args, **kwargs)

            # Make sure a compilation is triggered by clearing dynamo cache.
            torch._dynamo.eval_frame.remove_from_cache(
                self.original_code_object())

            self.compiled = True

            # Disable the C++ compilation of symbolic shape guards. C++-fication
            # of symbolic shape guards can improve guard overhead. But, since
            # vllm skip guards anyways, setting this flag to False can improve
            # compile time.
            dynamo_config_patches = {}
            try:
                _ = torch._dynamo.config.enable_cpp_symbolic_shape_guards
                dynamo_config_patches[
                    "enable_cpp_symbolic_shape_guards"] = False
            except AttributeError:
                # Note: this config is not available in torch 2.6, we can skip
                # if the config doesn't exist
                logger.debug(
                    "enable_cpp_symbolic_shape_guards config not available")

            with torch._dynamo.config.patch(**dynamo_config_patches):
                return self._compiled_callable(*args, **kwargs)

        return self._compiled_callable(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        ...

    def original_code_object(self) -> CodeType:
        """Return the original code object of the forward method."""
        return self.__class__.forward.__code__
