from typing import Any

from .interface import _Backend  # noqa: F401
from .interface import CpuArchEnum, Platform, PlatformEnum, UnspecifiedPlatform
from .registry import PlatformRegistry, detect_current_platform

_current_platform: Platform = UnspecifiedPlatform()


def initialize_current_platform():
    """Initialize the current platform. This function is called when loading
    the vllm plugin."""
    # Get the current platform from the registry first. If the current
    # platform is not set, try to detect the current platform.
    global _current_platform
    if PlatformRegistry.current_platform is not None:
        _current_platform = PlatformRegistry.get_current_platform_cls()()
    else:
        _current_platform = detect_current_platform()

    # Register custom ops for the current platform.
    from vllm.attention.layer import register_custom_ops
    register_custom_ops()


class CurrentPlatform(Platform):
    """A wrapper that provides an interface to the current platform.
    
    `current_platform` is imported to many modules once vLLM is imported.
    Updating `current_platform` value directly will not work in those modules.
    So it needs the wrapper here to provide a dynamic platform loading
    mechanism.

    This class can make sure that the `current_platform` is always up-to-date.
    """

    def __getattribute__(self, name: str) -> Any:
        """If the attribute is not found, go pass to the current platform."""
        # Use __getattribute__ to here to get the attribute from the current
        # platform. It doesn't work to use __getattr__ because it will be called
        # only when the attribute is not found. Since CurrentPlatform inherits
        # from Platform, __getattr__ will not be called.
        global _current_platform
        # Go pass to the current platform.
        return _current_platform.__getattribute__(name)


# The global variable for other modules to use.
current_platform = CurrentPlatform()

__all__ = ['Platform', 'PlatformEnum', 'current_platform', 'CpuArchEnum']
