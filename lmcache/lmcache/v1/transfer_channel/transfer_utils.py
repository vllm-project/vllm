# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Union

# Third Party
import msgspec


def get_correct_device(device: str, worker_id: int) -> str:
    """
    Get the correct device based on the given device string.

    Args:
        device (str): The device string, could be cpu or the accelerator type.
        worker_id (int): The worker id to determine the accelerator device.

    Returns:
        str: The correct device string with device id.
    """
    if device == "cpu":
        return "cpu"
    base_device = device.split(":")[0]
    return f"{base_device}:{worker_id}"


class SideMsgBase(msgspec.Struct, tag=True):
    """Base class for all side-related messages during initialization"""

    pass


# Side messages during initialization
class InitSideMsgBase(SideMsgBase):
    """Base class for all side-related messages during initialization"""

    pass


class P2PInitSideMsg(InitSideMsgBase):
    """P2P specific initialization message"""

    pass


# Side return messages during initialization
class InitSideRetMsgBase(SideMsgBase):
    """Base class for all side-related messages during initialization"""

    pass


class P2PInitSideRetMsg(InitSideRetMsgBase):
    """P2P specific initialization return message"""

    peer_lookup_url: str


SideMsg = Union[
    P2PInitSideMsg,
    P2PInitSideRetMsg,
]
