# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections import deque

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.v1.kv_offload.mediums import BlockIDsLoadStoreSpec
from vllm.v1.kv_offload.worker.worker import OffloadingHandler, TransferResult, TransferSpec

from .cpu_gpu import expand_block_ids


class WeaveDramCxlOffloadingHandler(OffloadingHandler):
    def __init__(
        self,
        src_tensors: list[torch.Tensor],
        dst_tensors: list[torch.Tensor],
        kv_dim_before_num_blocks: list[bool],
        src_block_size_factor: int,
        dst_block_size_factor: int,
    ):
        assert len(src_tensors) == len(dst_tensors) == len(kv_dim_before_num_blocks)
        self.src_tensors = src_tensors
        self.dst_tensors = dst_tensors
        self.kv_dim_before_num_blocks = kv_dim_before_num_blocks
        self.src_block_size_factor = src_block_size_factor
        self.dst_block_size_factor = dst_block_size_factor

        self._finished: deque[TransferResult] = deque()

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        src_spec, dst_spec = spec
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        src_sub_block_count = src_blocks.size * self.src_block_size_factor
        dst_sub_block_count = dst_blocks.size * self.dst_block_size_factor
        src_sub_blocks_to_skip = -dst_blocks.size % self.src_block_size_factor

        if dst_sub_block_count != src_sub_block_count - src_sub_blocks_to_skip:
            self._finished.append((job_id, False))
            return False

        src_to_dst = np.empty((dst_sub_block_count, 2), dtype=np.int64)
        expand_block_ids(
            src_blocks,
            self.src_block_size_factor,
            src_to_dst[:, 0],
            skip_count=src_sub_blocks_to_skip,
        )
        expand_block_ids(dst_blocks, self.dst_block_size_factor, src_to_dst[:, 1])
        mapping = torch.from_numpy(src_to_dst)

        for src_tensor, dst_tensor, kv_dim in zip(
            self.src_tensors, self.dst_tensors, self.kv_dim_before_num_blocks
        ):
            if kv_dim:
                src_key_cache, src_value_cache = src_tensor
                dst_key_cache, dst_value_cache = dst_tensor
                ops.swap_blocks(src_key_cache, dst_key_cache, mapping)
                ops.swap_blocks(src_value_cache, dst_value_cache, mapping)
            else:
                ops.swap_blocks(src_tensor, dst_tensor, mapping)

        self._finished.append((job_id, True))
        return True

    def get_finished(self) -> list[TransferResult]:
        finished = list(self._finished)
        self._finished.clear()
        return finished

    def wait(self, job_ids: set[int]) -> None:
        return
