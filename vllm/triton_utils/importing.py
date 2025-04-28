# SPDX-License-Identifier: Apache-2.0

import importlib
import pkgutil
import sys
import types
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
            self.constexpr = None
            self.dtype = None

    def init_torch_inductor_runtime():
        name = "torch._inductor.runtime"
        torch_runtime = importlib.import_module(name)
        path = torch_runtime.__path__
        for module_info in pkgutil.iter_modules(path, name + "."):
            if not module_info.ispkg:
                try:
                    importlib.import_module(module_info.name)
                except Exception as e:
                    logger.warning(
                        "Ignore import error when loading " \
                        "%s: %s", module_info.name, e)
                    continue

    # initialize torch inductor without triton placeholder
    # FIXME(Isotr0py): See if we can remove this after bumping torch version
    # to 2.7.0, because torch 2.7.0 has a better triton check.
    init_torch_inductor_runtime()
    # Replace the triton module in sys.modules with the placeholder
    sys.modules['triton'] = TritonPlaceholder()
    sys.modules['triton.language'] = TritonLanguagePlaceholder()

if 'triton' in sys.modules:
    logger.info("Triton module has been replaced with a placeholder.")
