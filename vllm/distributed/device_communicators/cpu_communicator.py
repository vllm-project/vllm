# SPDX-License-Identifier: Apache-2.0

import torch

from .base_device_communicator import DeviceCommunicatorBase


class CpuCommunicator(DeviceCommunicatorBase):

    def all_reduce(self, input_):
        try:
            import intel_extension_for_pytorch as ipex
            ipex.distributed.all_reduce(input_, group=self.device_group)
            return input_
        except ImportError:
            """
            Intel IPEX not found. Falling back to PyTorch native 
            all_reduce for CPU
            """
            torch.distributed.all_reduce(input_, group=self.device_group)
            return input_
