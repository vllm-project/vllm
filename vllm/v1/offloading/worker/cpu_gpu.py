# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator

import torch

from vllm.attention import AttentionBackend
from vllm.logger import init_logger
from vllm.utils import is_pin_memory_available
from vllm.v1.offloading.abstract import LoadStoreSpec
from vllm.v1.offloading.mediums import (BlockIDLoadStoreSpec, CPULoadStoreSpec,
                                        GPULoadStoreSpec)
from vllm.v1.offloading.worker.worker import (OffloadingHandler,
                                              TransferResult, TransferSpec)

logger = init_logger(__name__)


def block_ids(specs_list: list[LoadStoreSpec],
              block_size_factor: int,
              skip_count: int = 0) -> Iterator[int]:
    """
    Convert a list of BlockIDLoadStoreSpec to a list of matching block ids,
    assuming each spec is composed of actual block_size_factor blocks.
    The first skip_count blocks will be skipped.
    Note that skip_count must be less than block_size_factor.

    For example, if spec_list = [0, 1, 3] and block_size_factor =  4,
    then it yields [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15]
    since 0 maps to [0, 1, 2, 3]
    1 maps to [4, 5, 6, 7]
    and 3 maps to [12, 13, 14, 15]
    """
    assert skip_count < block_size_factor

    for spec in specs_list:
        assert isinstance(spec, BlockIDLoadStoreSpec)
        base_block_id = spec.block_id * block_size_factor
        for i in range(skip_count, block_size_factor):
            yield base_block_id + i

        # finished skipping
        skip_count = 0


class CpuGpuOffloadingHandler(OffloadingHandler):

    def __init__(self, attn_backend: type[AttentionBackend],
                 gpu_block_size: int, cpu_block_size: int, num_cpu_blocks: int,
                 gpu_caches: list[torch.Tensor]):
        assert cpu_block_size % gpu_block_size == 0
        self.block_size_factor = cpu_block_size // gpu_block_size

        self.attn_backend = attn_backend
        self.gpu_caches: list[torch.Tensor] = gpu_caches

        # cuda streams for gpu->cpu and cpu->gpu
        self.d2h_stream = torch.cuda.Stream()
        self.h2d_stream = torch.cuda.Stream()

        # job_id -> transfer cuda event
        self.transfer_events: dict[int, torch.cuda.Event] = {}
        # list of cuda events available for re-use
        self.events_pool: list[torch.cuda.Event] = []

        pin_memory = is_pin_memory_available()

        # allocate cpu tensors
        logger.info("Allocating %d CPU tensors...", len(gpu_caches))
        self.cpu_caches: list[torch.Tensor] = []
        for gpu_tensor in gpu_caches:
            gpu_shape = gpu_tensor.shape
            assert len(gpu_shape) >= 4  # (2, num_blocks, ..., ...)
            assert gpu_shape[0] == 2

            cpu_shape = list(gpu_shape)
            cpu_shape[1] = num_cpu_blocks * self.block_size_factor

            logger.debug("Allocating CPU tensor of shape %r", cpu_shape)
            self.cpu_caches.append(
                torch.zeros(cpu_shape,
                            dtype=gpu_tensor.dtype,
                            device="cpu",
                            pin_memory=pin_memory))

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        src_specs, dst_specs = spec
        assert src_specs and dst_specs
        if src_specs[0].medium() == CPULoadStoreSpec.medium():
            assert dst_specs[0].medium() == GPULoadStoreSpec.medium()
            stream = self.h2d_stream
            src_caches = self.cpu_caches
            dst_caches = self.gpu_caches
            src_block_size_factor = self.block_size_factor
            dst_block_size_factor = 1
        else:
            assert src_specs[0].medium() == GPULoadStoreSpec.medium()
            assert dst_specs[0].medium() == CPULoadStoreSpec.medium()
            stream = self.d2h_stream
            src_caches = self.gpu_caches
            dst_caches = self.cpu_caches
            src_block_size_factor = 1
            dst_block_size_factor = self.block_size_factor

        dst_sub_blocks_to_skip = (-len(src_specs) % dst_block_size_factor)

        assert (
            len(src_specs) *
            src_block_size_factor == len(dst_specs) * dst_block_size_factor -
            dst_sub_blocks_to_skip)

        src_to_dst_list: list[tuple[int, int]] = list(
            zip(
                block_ids(src_specs, src_block_size_factor),
                block_ids(dst_specs, dst_block_size_factor,
                          dst_sub_blocks_to_skip)))
        src_to_dst = torch.tensor(src_to_dst_list,
                                  device="cpu",
                                  dtype=torch.int64).view(-1, 2)

        event = self.events_pool.pop() if self.events_pool \
            else torch.cuda.Event()
        with torch.cuda.stream(stream):
            for src_cache, dst_cache in zip(src_caches, dst_caches):
                self.attn_backend.swap_blocks(src_cache, dst_cache, src_to_dst)
            event.record(stream)

        self.transfer_events[job_id] = event

        # success
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        for job_id, event in list(self.transfer_events.items()):
            if event.query():
                results.append((job_id, True))
                del self.transfer_events[job_id]
                self.events_pool.append(event)
        return results
