# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
from torch.distributed import ProcessGroup

from .base_device_communicator import DeviceCommunicatorBase


class CpuCommunicator(DeviceCommunicatorBase):

    def __init__(self,
                 cpu_group: ProcessGroup,
                 device: Optional[torch.device] = None,
                 device_group: Optional[ProcessGroup] = None,
                 unique_name: str = ""):
        super().__init__(cpu_group, device, device_group, unique_name)
        self.ipex_available = False
        self.dist_module = torch.distributed
        try:
            import intel_extension_for_pytorch as ipex
            self.ipex_available = True
            self.dist_module = ipex.distributed
        except ImportError:
            """
            Intel IPEX not found. Falling back to PyTorch native 
            all_reduce for CPU (e.g. MacOS)
            """
            pass

    def all_reduce(self, input_):
        self.dist_module.all_reduce(input_, group=self.device_group)
        return input_
