# SPDX-License-Identifier: Apache-2.0

from .base_device_communicator import DeviceCommunicatorBase


class XpuCommunicator(DeviceCommunicatorBase):
    # no special logic for XPU communicator
    pass
