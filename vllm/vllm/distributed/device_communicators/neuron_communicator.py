# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase)
from vllm.platforms import current_platform

if current_platform.is_neuron():
    import torch_xla.core.xla_model as xm


class NeuronCommunicator(DeviceCommunicatorBase):

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return xm.all_reduce(xm.REDUCE_SUM, x)

    def all_gather(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == -1, "Neuron only supports dim=-1 for all-gather."
        return xm.all_gather(x, dim=dim)
