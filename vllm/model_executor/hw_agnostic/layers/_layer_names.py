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

`validate_registered_overrides()` is the backstop: on leaving the scope it
turns any override that would still be skipped or dropped into a startup
`TypeError`, rather than a silently half-built or in-tree layer.
"""

import importlib
from types import ModuleType, TracebackType

import vllm.envs as envs
from vllm.logger import init_logger

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


def _own_classes(module: ModuleType) -> dict[str, type]:
    """Public classes `module` defines itself; a re-export is not an implementation.

    Classes only. A layer name denotes a class on both paths, and a class is what a
    plugin subclasses and what `register_oot` keys on, so a class is the only thing
    rebinding a name can usefully redirect. A module-level *function* under a
    mirrored name is a different implementation of a different signature -- the two
    `get_act_and_mul_fn`s disagree on both the keyword arguments they take and the
    exception they raise for an unsupported activation -- and no override can reach
    it, so swapping it would only break callers that run while plugins load. Sharing
    this filter with `validate_registered_overrides` is also what keeps the
    rebinding no wider than the backstop that checks it.
    """
    return {
        name: obj
        for name, obj in vars(module).items()
        if not name.startswith("_")
        and isinstance(obj, type)
        and obj.__module__ == module.__name__
    }


def _published_classes(hw_modules: dict[str, str]) -> dict[str, type]:
    """Hw-agnostic classes, by the name an override would register under."""
    published: dict[str, type] = {}
    for hw_name in hw_modules.values():
        published.update(_own_classes(importlib.import_module(hw_name)))
    return published


def validate_registered_overrides(hw_modules: dict[str, str]) -> None:
    """Fail loudly on an override the hw-agnostic path would quietly ignore.

    The two paths keep separate registries, so a plugin that captured the
    hw-specific class goes wrong in one of two ways, neither of which raises by
    itself:

    * *registered here, but not derived from the class we instantiate* -- the
      lookup in `__new__` matches on the bare name and hands back the override,
      then `super().__new__(override)` returns an object that is not an instance
      of `cls`, so Python silently skips `cls.__init__`. Half-built module.
    * *registered against the hw-specific class* -- the registration lands in the
      hw-specific registry, which the hw-agnostic `__new__` never reads, so the
      override is dropped and the in-tree layer runs instead.

    `hw_agnostic_layer_names()` is what keeps either from happening; this is the
    backstop for a registration route the rebinding does not reach. Raising here
    keeps the traceback pointed at plugin loading, where the cause is.
    """
    from vllm.model_executor.custom_op import op_registry_oot as hw_specific_registry
    from vllm.model_executor.hw_agnostic.custom_op import (
        op_registry_oot as hw_agnostic_registry,
    )

    for name, hw_agnostic_cls in _published_classes(hw_modules).items():
        override = hw_agnostic_registry.get(name)

        if override is None:
            if name not in hw_specific_registry:
                continue  # no override for this layer at all
            raise TypeError(
                f"Out-of-tree override {hw_specific_registry[name].__name__!r} is "
                f"registered for {name!r} against the hw-specific class, so it is "
                f"invisible to the hw-agnostic layers and "
                f"{hw_agnostic_cls.__module__}.{name} would run instead. Register "
                f"it against the class the hw-agnostic path instantiates, which "
                f"'from vllm.model_executor.layers.<mod> import {name}' resolves "
                f"to while plugins load."
            )

        if not issubclass(override, hw_agnostic_cls):
            raise TypeError(
                f"Out-of-tree override {override.__name__!r} for {name!r} does not "
                f"derive from {hw_agnostic_cls.__module__}.{name}, the class the "
                f"hw-agnostic path instantiates, so its '__init__' would be skipped "
                f"without error. It most likely subclasses the hw-specific "
                f"{name}; import it from vllm.model_executor.layers.<mod> while "
                f"plugins load, or from vllm.model_executor.hw_agnostic.layers.<mod> "
                f"directly."
            )


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
                for name, hw_obj in _own_classes(hw_module).items():
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
