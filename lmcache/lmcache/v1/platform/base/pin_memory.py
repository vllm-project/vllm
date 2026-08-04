# SPDX-License-Identifier: Apache-2.0
"""Platform-abstraction base classes.

This module defines abstract base classes for platform-specific capabilities.
:class:`PinMemoryBackend` is the first; future platform abstractions should
be added here as well.
"""


class PinMemoryBackend:
    """Base class for host-memory pinning per platform.

    The default implementation is a no-op that always returns ``False``,
    so platforms that do not support pinning do not need to subclass this.
    """

    def pin_memory(self, ptr: int, size: int, flags: int = 0) -> bool:
        """Pin a host memory region for DMA access.

        Args:
            ptr: Raw pointer (data_ptr) to the memory region.
            size: Size in bytes of the region to pin.
            flags: Platform-specific registration flags (e.g.
                ``cudaHostRegisterDefault = 0``).

        Returns:
            True if pinning succeeded, False otherwise.
        """
        return False

    def unpin_memory(self, ptr: int) -> bool:
        """Unpin a previously pinned host memory region.

        Args:
            ptr: Raw pointer (data_ptr) to the memory region.

        Returns:
            True if unpinning succeeded, False otherwise.
        """
        return False

    @property
    def is_pin_supported(self) -> bool:
        """Whether the current platform supports memory pinning.

        Returns:
            True if pinning is supported, False otherwise.
        """
        return False
