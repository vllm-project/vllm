# SPDX-License-Identifier: Apache-2.0
"""Device detection helpers, decoupled from ``lmcache.v1.platform``.

Kept in its own module (rather than in ``platform/__init__.py``) so that
peers such as :mod:`lmcache.v1.platform.torch_ops` can import the
detection primitives at the top of the file without introducing an
import cycle -- ``platform/__init__.py`` itself pulls in
``base_device_ops``, which in turn pulls in ``torch_ops``, so any name
that ``torch_ops`` needs from the platform package must live *outside*
that init chain.

The registry of :class:`DeviceSpec` subclasses and the detected torch
device module are both built lazily on first access and then cached
process-wide.
"""

# Standard
from functools import lru_cache
from typing import TYPE_CHECKING, Any
import os

# First Party
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base.device_spec import DeviceSpec

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _build_device_registry() -> "dict[str, DeviceSpec]":
    """Discover and instantiate every :class:`DeviceSpec` subclass.

    Discovery is deferred until first use so importing this module
    stays cheap and side-effect free.
    """
    # First Party
    from lmcache.v1.platform.base.device_spec import DeviceSpec
    from lmcache.v1.utils.subclass_discovery import discover_subclasses

    return {
        spec.device_type: spec
        for spec in [
            cls()
            for cls in discover_subclasses(
                "lmcache.v1.platform",
                DeviceSpec,  # type: ignore[type-abstract]
                module_filter=lambda name: not name.startswith(("_", "base")),
                require_defined_in_module=True,
                on_import_error=lambda name, exc: None,
            )
        ]
    }


def _detect_device() -> tuple[Any, str]:
    """Detect the available accelerator via the device registry.

    Returns:
        tuple[Any, str]: A tuple of (torch_device_module, device_type_string).
            When torch is not installed (CLI-only mode), returns
            ``(None, "cpu")``.
    """
    try:
        # Third Party
        import torch
    except ImportError as e:
        logger.warning("load torch failed, error is %s", e)
        return None, "cpu"  # fallback for CLI-only environments

    registry = _build_device_registry()

    # Check DEVICE_TYPE environment variable for forced device selection.
    env_device_type = os.environ.get("DEVICE_TYPE")
    if env_device_type is not None:
        env_device_type = env_device_type.strip().lower()
        spec = registry.get(env_device_type)
        if spec is not None and spec.is_available():
            torch_module = getattr(torch, spec.torch_module_name, None)
            if torch_module is not None:
                return torch_module, spec.device_type
            else:
                logger.warning(
                    "DEVICE_TYPE=%r is available but torch module [%s] not found, "
                    "falling back to auto-detection.",
                    env_device_type,
                    spec.torch_module_name,
                )
        else:
            logger.warning(
                "DEVICE_TYPE=%r is not available or not registered, "
                "falling back to auto-detection.",
                env_device_type,
            )

    for spec in registry.values():
        if not spec.is_available():
            continue

        torch_module = getattr(torch, spec.torch_module_name, None)
        if torch_module is not None:
            return torch_module, spec.device_type
        else:
            logger.warning(
                "device [%s] is available, but torch module [%s] is not found.",
                spec.device_type,
                spec.torch_module_name,
            )

    # No accelerator found -- fall back to CPU stub
    # First Party
    from lmcache.v1.platform.cpu.stub_cpu_device import StubCPUDevice

    return StubCPUDevice("cpu"), "cpu"


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def get_device_spec(device_type: str) -> "DeviceSpec | None":
    """Return the :class:`DeviceSpec` registered for *device_type*, if any."""
    return _build_device_registry().get(device_type)


@lru_cache(maxsize=1)
def get_torch_device() -> tuple[Any, str]:
    """Return the cached ``(torch_dev, torch_device_type)`` pair.

    Lazy + memoized so that peers like :mod:`torch_ops` can safely
    import this helper at module top level: no work is performed until
    the tuple is actually needed.
    """
    torch_dev, torch_device_type = _detect_device()
    logger.info("torch_dev=%s, torch_device_type=%s", torch_dev, torch_device_type)
    return torch_dev, torch_device_type


@lru_cache(maxsize=1)
def current_device_spec() -> "DeviceSpec":
    """Return the :class:`DeviceSpec` for the detected device.

    Falls back to a bare ``DeviceSpec()`` (no-op / all False semantics)
    when no accelerator sub-package matches.
    """
    # First Party
    from lmcache.v1.platform.base.device_spec import DeviceSpec

    _, device_type = get_torch_device()
    spec = get_device_spec(device_type)
    if spec is None:
        if device_type != "cpu":
            logger.warning(
                "No DeviceSpec registered for %r; using fallback"
                " with no-op capabilities.",
                device_type,
            )
        return DeviceSpec()
    return spec
