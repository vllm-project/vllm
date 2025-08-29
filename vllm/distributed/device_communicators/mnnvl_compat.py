# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# from vllm.distributed import get_dp_group
import torch
from flashinfer.comm.mnnvl import CommBackend as CommBackend

from vllm.utils.flashinfer import has_flashinfer_all2all

assert has_flashinfer_all2all(), "Flashinfer alltoallv module cannot be found"


class CustomCommunicator(CommBackend):

    def __init__(self, group):
        self._group = group

    def Get_rank(self) -> int:
        return self._group.rank_in_group

    def Get_size(self) -> int:
        return self._group.world_size

    def allgather(self, data: int | bytes):
        device = f"cuda:{torch.cuda.current_device()}"

        if isinstance(data, int):
            local_tensor = torch.tensor([data],
                                        device=device,
                                        dtype=torch.int32)
            gathered = self._group.all_gather(local_tensor)
            return [int(x.item()) for x in gathered]
        elif isinstance(data, bytes):
            device_id = torch.cuda.current_device()
            device_str = f"cuda:{device_id}"
            local_tensor = torch.ByteTensor(
                list(data)).unsqueeze(0).to(device_str)
            gathered = self._group.all_gather(local_tensor, dim=0)
            result = [
                bytes(gathered[i].cpu().tolist())
                for i in range(self.Get_size())
            ]
            return result
        else:
            raise TypeError(f"Unsupported type for allgather: {type(data)}")

    def Split(self, color: int, key: int) -> 'CustomCommunicator':
        return self
