# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import shutil
import subprocess
import types
from importlib.util import find_spec

from vllm.logger import init_logger

logger = init_logger(__name__)


def _configure_triton_ptxas_for_new_gpus():
    """
    Configure TRITON_PTXAS_PATH for GPUs that may not be supported by
    Triton's bundled ptxas (e.g., Jetson Thor sm_110a, DGX Spark sm_121a).

    Triton bundles a ptxas binary (currently CUDA 12.8) that may not support
    the newest GPU architectures. When running on such GPUs, Triton kernel
    compilation fails with errors like:
        ptxas fatal: Value 'sm_121a' is not defined for option 'gpu-name'

    This function uses Triton's native GPU detection to check the architecture
    and configures Triton to use the system's CUDA toolkit ptxas instead,
    which typically has broader architecture support (e.g., CUDA 13.0+).
    """
    # Don't override if already set by user
    if os.environ.get("TRITON_PTXAS_PATH"):
        return

    # Try to find system ptxas
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    system_ptxas_paths = [
        os.path.join(cuda_home, "bin", "ptxas"),
        "/usr/local/cuda/bin/ptxas",
        shutil.which("ptxas"),  # Check PATH
    ]

    system_ptxas = None
    for path in system_ptxas_paths:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            system_ptxas = path
            break

    if not system_ptxas:
        # No system ptxas found, can't help
        return

    # Use Triton's native GPU detection to get the architecture.
    # This is how Triton itself determines the target GPU.
    try:
        from triton.backends import backends

        nvidia_backend = backends.get("nvidia")
        if nvidia_backend is None or nvidia_backend.driver is None:
            return

        if not nvidia_backend.driver.is_active():
            return

        # Get the current GPU target using Triton's driver
        driver_instance = nvidia_backend.driver()
        target = driver_instance.get_current_target()
        arch = target.arch  # e.g., 121 for sm_121a (CC 12.1)

        # GPUs with arch >= 110 (compute capability >= 11.0) may need system ptxas
        # - arch 110: Jetson Thor (sm_110a, CC 11.0)
        # - arch 120: Blackwell B100/B200 (sm_120, CC 12.0)
        # - arch 121: DGX Spark GB10 (sm_121a, CC 12.1)
        if arch >= 110:
            # Check if system ptxas is functional
            try:
                result = subprocess.run(
                    [system_ptxas, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    # System ptxas is available, use it
                    os.environ["TRITON_PTXAS_PATH"] = system_ptxas
                    major, minor = divmod(arch, 10)
                    logger.info(
                        "Detected GPU with compute capability %d.%d (arch=%d). "
                        "Configuring TRITON_PTXAS_PATH=%s to ensure "
                        "Triton kernel compilation compatibility.",
                        major,
                        minor,
                        arch,
                        system_ptxas,
                    )
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
                logger.debug("Cannot use system ptxas: %s", e)

    except Exception as e:
        # Don't fail if detection doesn't work - user can still set
        # TRITON_PTXAS_PATH manually
        logger.debug("Failed to auto-configure TRITON_PTXAS_PATH: %s", e)


# Configure ptxas before importing Triton to ensure kernels can compile
# on new GPU architectures (Thor, GB10, etc.)
_configure_triton_ptxas_for_new_gpus()

HAS_TRITON = (
    find_spec("triton") is not None
    or find_spec("pytorch-triton-xpu") is not None  # Not compatible
)
if HAS_TRITON:
    try:
        from triton.backends import backends

        # It's generally expected that x.driver exists and has
        # an is_active method.
        # The `x.driver and` check adds a small layer of safety.
        active_drivers = [
            x.driver for x in backends.values() if x.driver and x.driver.is_active()
        ]

        # Check if we're in a distributed environment where CUDA_VISIBLE_DEVICES
        # might be temporarily empty (e.g., Ray sets it to "" during actor init)
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        is_distributed_env = (
            cuda_visible_devices is not None and len(cuda_visible_devices.strip()) == 0
        )

        # Apply lenient driver check for distributed environments
        if is_distributed_env and len(active_drivers) == 0:
            # Allow 0 drivers in distributed environments - they may become
            # active later when CUDA context is properly initialized
            logger.debug(
                "Triton found 0 active drivers in distributed environment. "
                "This is expected during initialization."
            )
        elif not is_distributed_env and len(active_drivers) != 1:
            # Strict check for non-distributed environments
            logger.info(
                "Triton is installed but %d active driver(s) found "
                "(expected 1). Disabling Triton to prevent runtime errors.",
                len(active_drivers),
            )
            HAS_TRITON = False
    except ImportError:
        # This can occur if Triton is partially installed or triton.backends
        # is missing.
        logger.warning(
            "Triton is installed, but `triton.backends` could not be imported. "
            "Disabling Triton."
        )
        HAS_TRITON = False
    except Exception as e:
        # Catch any other unexpected errors during the check.
        logger.warning(
            "An unexpected error occurred while checking Triton active drivers:"
            " %s. Disabling Triton.",
            e,
        )
        HAS_TRITON = False

if not HAS_TRITON:
    logger.info(
        "Triton not installed or not compatible; certain GPU-related"
        " functions will not be available."
    )


class TritonPlaceholder(types.ModuleType):
    def __init__(self):
        super().__init__("triton")
        self.__version__ = "3.4.0"
        self.jit = self._dummy_decorator("jit")
        self.autotune = self._dummy_decorator("autotune")
        self.heuristics = self._dummy_decorator("heuristics")
        self.Config = self._dummy_decorator("Config")
        self.language = TritonLanguagePlaceholder()

    def _dummy_decorator(self, name):
        def decorator(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            return lambda f: f

        return decorator


class TritonLanguagePlaceholder(types.ModuleType):
    def __init__(self):
        super().__init__("triton.language")
        self.constexpr = None
        self.dtype = None
        self.int64 = None
        self.int32 = None
        self.tensor = None
        self.exp = None
        self.log = None
        self.log2 = None
