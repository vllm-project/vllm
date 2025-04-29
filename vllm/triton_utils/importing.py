# SPDX-License-Identifier: Apache-2.0
import importlib
import sys
import types
from importlib.util import find_spec
from typing import Optional

from vllm.logger import init_logger

logger = init_logger(__name__)

HAS_TRITON = (
    find_spec("triton") is not None
    or find_spec("pytorch-triton-xpu") is not None  # Not compatible
)

if not HAS_TRITON:
    logger.info("Triton not installed or not compatible; certain GPU-related"
                " functions will not be available.")

    class TritonModulePlaceholder(types.ModuleType):

        def __init__(
            self,
            name: str,
            dummy_objects: Optional[list[str]] = None,
        ):
            super().__init__(name)

            if dummy_objects is not None:
                for obj_name in dummy_objects:
                    setattr(self, obj_name, object)

    class TritonLanguagePlaceholder(TritonModulePlaceholder):

        def __init__(self):
            super().__init__("triton.language")
            self.constexpr = None
            self.dtype = None

    class TritonPlaceholder(TritonModulePlaceholder):

        def __init__(self):
            super().__init__("triton", dummy_objects=["Config"])
            self.jit = self._dummy_decorator("jit")
            self.autotune = self._dummy_decorator("autotune")
            self.heuristics = self._dummy_decorator("heuristics")
            self.language = TritonLanguagePlaceholder()
            logger.warning_once(
                "Triton is not installed. Using dummy decorators. "
                "Install it via `pip install triton` to enable kernel "
                "compilation.")

        def _dummy_decorator(self, name):

            def decorator(func=None, **kwargs):
                if func is None:
                    return lambda f: f
                return func

            return decorator

    # Hack `_is_triton_available` in torch to return False
    torch_hints = importlib.import_module("torch._inductor.runtime.hints")
    torch_hints._is_triton_available = lambda: False  # type: ignore[attr-defined]

    # Replace the triton module and torch triton helpers in sys.modules
    # with the placeholder
    sys.modules['triton'] = TritonPlaceholder()
    sys.modules['triton.language'] = TritonLanguagePlaceholder()
    sys.modules['torch._inductor.runtime.triton_helpers'] = types.ModuleType(
        "triton_helpers")

    # Replace triton submodules with dummy objects to keep compatibility with
    # torch triton check
    triton_modules_with_objects = {
        "triton.compiler": ["CompiledKernel"],
        "triton.runtime.autotuner": ["OutOfResources"],
        "triton.runtime.jit": ["KernelInterface"],
    }
    for module_name, dummy_objects in triton_modules_with_objects.items():
        sys.modules[module_name] = TritonModulePlaceholder(
            module_name, dummy_objects=dummy_objects)

if 'triton' in sys.modules:
    logger.info("Triton module has been replaced with a placeholder.")
