# SPDX-License-Identifier: Apache-2.0

import sys
import types
from abc import ABC
from importlib.util import find_spec

from vllm.logger import init_logger

logger = init_logger(__name__)

HAS_TRITON = (
    find_spec("triton") is not None
    or find_spec("pytorch-triton-xpu") is not None  # Not compatible
)

if not HAS_TRITON:
    logger.info("Triton not installed or not compatible; certain GPU-related"
                " functions will not be available.")

    class TritonPlaceholder(types.ModuleType):

        def __init__(self):
            super().__init__("triton")
            self.jit = self._dummy_decorator("jit")
            self.autotune = self._dummy_decorator("autotune")
            self.heuristics = self._dummy_decorator("heuristics")
            self.language = TritonLanguagePlaceholder()
            self.Config = self._dummy_decorator("Config")
            self.__version__ = ""
            logger.warning_once(
                "Triton is not installed. Using dummy decorators. "
                "Install it via `pip install triton` to enable kernel"
                "compilation.")

        def _dummy_decorator(self, name):

            def decorator(func=None, **kwargs):
                if func is None:
                    return lambda f: f
                return func

            return decorator

    class TritonLanguagePlaceholder(types.ModuleType):

        def __init__(self):
            super().__init__("triton.language")
            self.constexpr = lambda x: x
            self.dtype = None
            self.extra = None
            self.math = None
            self.tensor = None

    class TritonCompilerPlaceholder(types.ModuleType):

        def __init__(self):
            super().__init__("triton.compiler")
            self.CompiledKernel = ABC

    class TritonRuntimeAutotunerPlaceholder(types.ModuleType):

        def __init__(self):
            super().__init__("triton.runtime.autotuner")
            self.OutOfResources = ABC

    class TritonRuntimeJitPlaceholder(types.ModuleType):

        def __init__(self):
            super().__init__("triton.runtime.jit")
            self.KernelInterface = ABC

    sys.modules['triton'] = TritonPlaceholder()
    sys.modules['triton.language'] = TritonLanguagePlaceholder()
    sys.modules['triton.compiler'] = TritonCompilerPlaceholder()
    sys.modules[
        'triton.runtime.autotuner'] = TritonRuntimeAutotunerPlaceholder()
    sys.modules['triton.runtime.jit'] = TritonRuntimeJitPlaceholder()

if 'triton' in sys.modules:
    logger.info("Triton module has been replaced with a placeholder.")
