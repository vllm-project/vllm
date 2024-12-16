from .interface import _Backend  # noqa: F401
from .interface import CpuArchEnum, Platform, PlatformEnum, UnspecifiedPlatform
from .registry import PlatformRegistry, detect_current_platform

_current_platform: Platform = UnspecifiedPlatform()


def initialize_current_platform():
    """Initialize the current platform. This function is called when loading
    the vllm plugin."""
    global _current_platform
    # Get the current platform from the registry first. If the current platform
    # is not set, try to detect the current platform.
    if PlatformRegistry.current_platform is not None:
        _current_platform = PlatformRegistry.get_current_platform_cls()
    else:
        _current_platform = detect_current_platform()


def update_current_platform(device_name: str):
    """Update the current platform. This function is used by users to set the
    current platform by hand."""
    global _current_platform
    PlatformRegistry.set_current_platform(device_name)
    _current_platform = PlatformRegistry.get_current_platform_cls()


class CurrentPlatform:
    """A wrapper that provides an interface to the current platform.
    
    `current_platform` is imported to many modules once vLLM is imported.
    Updating `current_platform` value directly will not work in those modules.
    So it needs the wrapper here to provide a dynamic platform loading
    mechanism.

    This class can make sure that the `current_platform` is always up-to-date.
    """

    def __init__(self):
        self.platform = _current_platform

    def _refresh_current_platform(self):
        """Refresh the current platform dynamically."""
        global _current_platform
        if _current_platform is not self.platform:
            self.platform = _current_platform

    def __getattr__(self, name):
        """Go pass to the current platform."""
        self._refresh_current_platform()
        return getattr(self.platform, name)


# The global variable for other modules to use.
current_platform: CurrentPlatform = CurrentPlatform()

__all__ = ['Platform', 'PlatformEnum', 'current_platform', 'CpuArchEnum']
