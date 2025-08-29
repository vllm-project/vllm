# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# from vllm.distributed import get_dp_group
import torch
import torch.distributed as dist
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

    def allgather(self, data: int):
        gathered = [None] * self.Get_size()
        dist.all_gather_object(gathered, data, group=self._group)
        return gathered

    def Split(self, color: int, key: int) -> 'CustomCommunicator':
        return self


    def allgather_bytes(self, data: bytes):
        # return torch.distributed.broadcast_object_list(
        result = [data] * self.Get_size()
        torch.distributed.all_gather_object(result, data)
        # print(result)
        return result