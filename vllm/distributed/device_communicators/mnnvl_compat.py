# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch.distributed as dist
from flashinfer.comm.mnnvl import CommBackend as CommBackend

from vllm.utils.flashinfer import has_flashinfer_all2all

assert has_flashinfer_all2all(), "Flashinfer alltoallv module cannot be found"


class CustomCommunicator(CommBackend):
    def __init__(self, group):
        self._group = group

    def Get_rank(self) -> int:
        return self._group.rank()

    def Get_size(self) -> int:
        return self._group.size()

    def allgather(self, data: int):
        gathered = [None] * self.Get_size()
        dist.all_gather_object(gathered, data, group=self._group)
        return gathered

    def Split(self, color: int, key: int) -> "CustomCommunicator":
        return self
