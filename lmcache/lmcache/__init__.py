# SPDX-License-Identifier: Apache-2.0

# Standard
import sys
import types

# First Party
from lmcache.logging import init_logger

# --------------------------
# Backend instance & Device detection
# --------------------------
from lmcache.v1.platform import torch_dev as torch_dev
from lmcache.v1.platform import torch_device_type as torch_device_type

try:
    # First Party
    from lmcache._version import __version__
except ImportError:
    __version__ = "unknown"

logger = init_logger(__name__)

__all__ = ["__version__", "torch_dev", "torch_device_type"]


# --------------------------
# Backward-compat ``lmcache.c_ops`` shim
# --------------------------
def _install_c_ops_shim() -> None:
    """Register ``lmcache.c_ops`` as the resolved :class:`DeviceOps` instance.

    Resolves the singleton :class:`DeviceOps` instance for the detected
    device (which calls :meth:`ensure_native` internally), then registers
    a PEP 562 shim module that forwards attribute access to it.
    """
    # First Party
    from lmcache.v1.platform import resolve_device_ops

    ops = resolve_device_ops(torch_device_type)

    shim = types.ModuleType("lmcache.c_ops")
    shim.__getattr__ = lambda name: getattr(ops, name)  # type: ignore[method-assign]
    shim.__dir__ = lambda: dir(ops)  # type: ignore[method-assign]
    sys.modules["lmcache.c_ops"] = shim
    globals()["c_ops"] = shim  # parent attr for IMPORT_FROM bytecode


try:
    _install_c_ops_shim()
except Exception as exc:
    logger.warning(
        "No compute backend loaded; CLI-only mode (torch/numba not installed). "
        "Reason: %s",
        exc,
    )
