# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolving `vllm.model_executor.layers.<mod>.X` to
`vllm.model_executor.hw_agnostic.layers.<mod>.X`.

The hw-agnostic layers are standalone implementations, so a layer name denotes
two unrelated classes, the in-tree and the hw-agnostic one. An out-of-tree
plugin might subclass `from vllm.model_executor.layers.<mod> import X` or
`from vllm.model_executor.hw_agnostic.layers.<mod> import X`.

In the first case, `hw_agnostic_layer_names()` rebinds the in-tree names to the
hw-agnostic classes. There is no explicit mapping between the in-tree and the
hw-agnostic modules, but `_mirrored_modules()` derives it from the package layout,
following the rules listed there.

`validate_registered_overrides()` is the backstop: on leaving the scope it turns
any override that would still be skipped or dropped into a startup `TypeError`.
"""

import importlib
import importlib.util
import pkgutil
from types import ModuleType, TracebackType

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_HW_PKG = "vllm.model_executor.hw_agnostic.layers"
_VLLM_PKG = "vllm.model_executor.layers"


def _raise_import_error(name: str) -> None:
    # ImportErrors are deliberately ignored,
    # which would hide everything under it from the rebinding.
    raise


def _mirrored_modules() -> dict[str, str]:
    """In-tree layer module -> its hw-agnostic counterpart, by package layout.

    The hw-agnostic package tree mirrors the in-tree one: the counterpart of
    `vllm.model_executor.layers.<path>` is
    `vllm.model_executor.hw_agnostic.layers.<path>`, the convention
    `transformers.layers._resolve` uses too. The following rules apply:

    * A port keeps the in-tree path, `__name__` (the out-of-tree registry key)
      and registered `name` (the `custom_ops` key); otherwise `_is_counterpart`
      skips it with a warning.
    * Every directory needs an `__init__.py`, or it and everything under it is
      invisible.
    * `_`-prefixed modules and packages are private and never mirrored.
    * Extra hw-agnostic modules and classes are allowed, but an extra class must
      not reuse an in-tree class name: registries key on the bare name.
    * Only classes are rebound (`_hw_agnostic_classes`), not functions.
    * Layers with no hw-agnostic implementation keep their in-tree class.

    """
    hw_pkg = importlib.import_module(_HW_PKG)
    mirrored: dict[str, str] = {}
    for info in pkgutil.walk_packages(
        hw_pkg.__path__, prefix=f"{_HW_PKG}.", onerror=_raise_import_error
    ):
        rel = info.name.removeprefix(_HW_PKG)
        if any(part.startswith("_") for part in rel.split(".") if part):
            continue
        vllm_name = _VLLM_PKG + rel
        try:
            spec = importlib.util.find_spec(vllm_name)
        except ModuleNotFoundError:  # in-tree parent package is missing
            spec = None
        if spec is None:
            continue  # hw-agnostic-only module
        mirrored[vllm_name] = info.name
    return mirrored


def _is_counterpart(vllm_obj: object, hw_cls: type, name: str) -> bool:
    """Whether in-tree `vllm_obj` is the class `hw_cls` ports.

    It is if `vllm_obj` is a class, both classes are called `name` (so not an
    alias such as `Foo = Bar`), and both registered the same op name (e.g.
    `"rms_norm"`).
    """
    return (
        isinstance(vllm_obj, type)
        and vllm_obj.__name__ == hw_cls.__name__ == name
        and getattr(vllm_obj, "name", None) == getattr(hw_cls, "name", None)
    )


def _hw_agnostic_classes(module: ModuleType) -> dict[str, type]:
    """Returns a dict mapping attribute name -> class for every public class that
    `module` may be rebound to.

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
        published.update(_hw_agnostic_classes(importlib.import_module(hw_name)))
    return published


def validate_registered_overrides(hw_modules: dict[str, str]) -> None:
    """Fail loudly on an override the hw-agnostic path would silently ignore.

    The in-tree and hw-agnostic layers keep separate registries, so a plugin that
    subclassed the in-tree class goes wrong in one of two ways, neither of which
    raises by itself:

    * *In the hw-agnostic registry, but not derived from the hw-agnostic class*,
      e.g. `@HwCustomOp.register_oot(name="RMSNorm") class MyNorm(InTreeRMSNorm)`:
      `__new__` matches on the bare name and returns a `MyNorm`, which is not a
      hw-agnostic `RMSNorm`, so Python skips `__init__` and leaves a half-built
      module.
    * *In the in-tree registry*, e.g. `@InTreeRMSNorm.register_oot` in code
      imported before plugins load: the hw-agnostic `__new__` never reads that
      registry, so the override is dropped and the plain hw-agnostic `RMSNorm`
      runs.

    Plugins that import from `vllm.model_executor.layers.<mod>` while
    `hw_agnostic_layer_names()` is active cannot hit either case. This check
    catches overrides registered any other way, and raises during plugin loading
    so the error points at the plugin.
    """
    from vllm.model_executor.custom_op import op_registry_oot as in_tree_registry
    from vllm.model_executor.hw_agnostic.custom_op import (
        op_registry_oot as hw_agnostic_registry,
    )

    for name, hw_agnostic_cls in _published_classes(hw_modules).items():
        override = hw_agnostic_registry.get(name)

        if override is None:
            if name not in in_tree_registry:
                continue  # no override for this layer at all
            raise TypeError(
                f"Out-of-tree override {in_tree_registry[name].__name__!r} is "
                f"registered for {name!r} against the in-tree class, so it is "
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
                f"without error. It most likely subclasses the in-tree "
                f"{name}; import it from vllm.model_executor.layers.<mod> while "
                f"plugins load, or from vllm.model_executor.hw_agnostic.layers.<mod> "
                f"directly."
            )


class _LayerNameScope:
    """The context manager `hw_agnostic_layer_names()` hands out."""

    def __init__(self) -> None:
        self._saved: list[tuple[ModuleType, str, object]] = []
        self._mirrored: dict[str, str] = {}
        self._entered = False

    def __enter__(self) -> None:
        if not envs.VLLM_USE_HW_AGNOSTIC:
            return
        self._entered = True
        try:
            self._mirrored = _mirrored_modules()
            for vllm_name, hw_name in self._mirrored.items():
                vllm_module = importlib.import_module(vllm_name)
                hw_module = importlib.import_module(hw_name)

                rebound = []
                for name, hw_obj in _hw_agnostic_classes(hw_module).items():
                    if not hasattr(vllm_module, name):
                        continue  # hw-agnostic-only class
                    vllm_obj = getattr(vllm_module, name)
                    if not _is_counterpart(vllm_obj, hw_obj, name):
                        # The hw-agnostic class shares its name with an in-tree
                        # object it does not port, e.g. an in-tree function or a
                        # class with a different op name. That is a bug in the
                        # hw-agnostic layers, not in the plugin, so keep the
                        # in-tree object and warn instead of failing plugin
                        # loading; `test_mirrored_names_are_counterparts`
                        # catches it in CI.
                        logger.warning_once(
                            "Not resolving %s.%s to %s.%s: the in-tree object is "
                            "not the class it ports (see the layout rules in "
                            "%s._mirrored_modules).",
                            vllm_name,
                            name,
                            hw_name,
                            name,
                            __name__,
                        )
                        continue
                    self._saved.append((vllm_module, name, vllm_obj))
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
                validate_registered_overrides(self._mirrored)
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
