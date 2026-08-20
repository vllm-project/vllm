# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolving `vllm.model_executor.layers.<mod>.X` to
`vllm.model_executor.hw_agnostic.layers.<mod>.X`.

The hw-agnostic layers are self-contained implementations, so a layer name
denotes two unrelated classes, the hw-agnostic and the hw-specific.
An out-of-tree plugin might subclass
`from vllm.model_executor.layers.<mod> import X` or
`from vllm.model_executor.hw_agnostic.layers.<mod> import X`.

In case of `from vllm.model_executor.layers.<mod> import X`,
`hw_agnostic_layer_names()` rebinds the in-tree names to the
hw-agnostic classes.
"""

import importlib
from types import ModuleType, TracebackType

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.hw_agnostic.custom_op import validate_registered_overrides

logger = init_logger(__name__)

# In-tree layer module -> its hw-agnostic counterpart.
# Layers with no hw-agnostic implementation, keep their in-tree class.
_MIRRORED_MODULES: dict[str, str] = {
    "vllm.model_executor.layers.activation": (
        "vllm.model_executor.hw_agnostic.layers.activation"
    ),
    "vllm.model_executor.layers.layernorm": (
        "vllm.model_executor.hw_agnostic.layers.layernorm"
    ),
}


def _own_definitions(module: ModuleType) -> dict[str, object]:
    """Public names `module` defines itself; a re-export is not an implementation."""
    return {
        name: obj
        for name, obj in vars(module).items()
        if not name.startswith("_")
        and getattr(obj, "__module__", None) == module.__name__
    }


class _LayerNameScope:
    """The context manager `hw_agnostic_layer_names()` hands out."""

    def __init__(self) -> None:
        self._saved: list[tuple[ModuleType, str, object]] = []
        self._entered = False

    def __enter__(self) -> None:
        if not envs.VLLM_USE_HW_AGNOSTIC:
            return
        self._entered = True
        try:
            for vllm_name, hw_name in _MIRRORED_MODULES.items():
                vllm_module = importlib.import_module(vllm_name)
                hw_module = importlib.import_module(hw_name)

                rebound = []
                for name, hw_obj in _own_definitions(hw_module).items():
                    if not hasattr(vllm_module, name):
                        continue  # hw-agnostic-only helper
                    self._saved.append((vllm_module, name, getattr(vllm_module, name)))
                    setattr(vllm_module, name, hw_obj)
                    rebound.append(name)

                logger.debug(
                    "Resolving %s to %s for: %s",
                    vllm_name,
                    hw_name,
                    ", ".join(sorted(rebound)) or "(nothing)",
                )
        except BaseException:
            self._restore()  # `__exit__` does not run if `__enter__` raises
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            # Skipped if the block raised: that exception is the useful one.
            if self._entered and exc_type is None:
                validate_registered_overrides(_MIRRORED_MODULES)
        finally:
            self._restore()

    def _restore(self) -> None:
        for module, name, original in reversed(self._saved):
            setattr(module, name, original)
        self._saved.clear()
        self._entered = False


def hw_agnostic_layer_names() -> _LayerNameScope:
    """Resolve in-tree layer names to the hw-agnostic classes inside this block.

    A no-op unless `VLLM_USE_HW_AGNOSTIC` is set. Restores every rebound name on
    exit, including when the block raises.
    """
    return _LayerNameScope()
