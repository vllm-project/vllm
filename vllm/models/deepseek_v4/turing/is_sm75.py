from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.platforms.interface import DeviceCapability


def is_turing_target(capability: DeviceCapability | None) -> bool:
    """True when we should use the Turing (SM75) DeepSeek-V4 backend."""
    return capability is not None and capability.major == 7
