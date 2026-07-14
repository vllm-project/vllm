#!/usr/bin/env python3
"""Validate aligned and unaligned Ascend NPU/CPU KV block transfers."""

from __future__ import annotations

import torch

from vllm.platforms import current_platform
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.npu_worker import AscendCPUOffloadingWorker
from vllm.v1.kv_offload.cpu.worker_factory import (
    create_cpu_offloading_worker,
    is_ascend_platform,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    from vllm_ascend.utils import enable_custom_op

    require(current_platform.device_type == "npu", "Ascend NPU is required")
    require(is_ascend_platform(), "Ascend platform was not detected")
    require(enable_custom_op(), "Failed to enable vllm-ascend custom ops")
    require(
        hasattr(torch.ops._C_ascend, "swap_blocks_batch"),
        "swap_blocks_batch is unavailable",
    )

    device = "npu:0"
    num_npu_blocks = 32
    num_cpu_blocks = 16
    page_size = 1024
    block_size_factor = 8
    npu_tensors: list[torch.Tensor] = []
    originals: list[torch.Tensor] = []
    for tensor_idx in range(6):
        tensor = torch.empty(
            (num_npu_blocks, page_size), dtype=torch.int8, device=device
        )
        for block_idx in range(num_npu_blocks):
            pattern = (
                torch.arange(page_size, dtype=torch.int16)
                + tensor_idx * 17
                + block_idx * 3
            ) % 127
            tensor[block_idx].copy_(pattern.to(torch.int8).to(device))
        npu_tensors.append(tensor)
        originals.append(tensor.cpu().clone())

    kv_caches = CanonicalKVCaches(
        tensors=[
            CanonicalKVCacheTensor(tensor=tensor, page_size_bytes=page_size)
            for tensor in npu_tensors
        ],
        group_data_refs=[
            [
                CanonicalKVCacheRef(tensor_idx=idx, page_size_bytes=page_size)
                for idx in range(3)
            ],
            [
                CanonicalKVCacheRef(tensor_idx=idx, page_size_bytes=page_size)
                for idx in range(3, 6)
            ],
        ],
    )
    worker = create_cpu_offloading_worker(
        kv_caches=kv_caches,
        block_size_factor=block_size_factor,
        num_cpu_blocks=num_cpu_blocks,
    )
    assert isinstance(worker, AscendCPUOffloadingWorker)

    cpu_tensors = worker._store_handler.dst_tensors
    for tensor in cpu_tensors:
        tensor.fill_(-9)

    # Group 0 is aligned. Group 1 begins at logical block 5 and crosses
    # three CPU offload blocks, exercising sub-block pointer calculations.
    npu_blocks = list(range(16)) + list(range(5, 18))
    group_sizes = [16, 13]
    block_indices = [0, 5]
    cpu_blocks = [0, 1, 2, 3, 4]
    npu_spec = GPULoadStoreSpec(
        block_ids=npu_blocks,
        group_sizes=group_sizes,
        block_indices=block_indices,
    )

    require(
        worker.submit_store(9001, npu_spec, CPULoadStoreSpec(cpu_blocks)),
        "NPU to CPU submission failed",
    )
    worker.wait({9001})
    finished = worker.get_finished()
    require(len(finished) == 1 and finished[0].success, "Store did not finish")

    cpu_views = [
        tensor.view(num_cpu_blocks, block_size_factor, page_size)
        for tensor in cpu_tensors
    ]
    for tensor_idx in range(3):
        for logical_block in range(16):
            torch.testing.assert_close(
                cpu_views[tensor_idx][
                    logical_block // block_size_factor,
                    logical_block % block_size_factor,
                ],
                originals[tensor_idx][logical_block],
            )
    for tensor_idx in range(3, 6):
        for logical_block in range(5, 18):
            torch.testing.assert_close(
                cpu_views[tensor_idx][
                    2 + logical_block // block_size_factor,
                    logical_block % block_size_factor,
                ],
                originals[tensor_idx][logical_block],
            )

    for tensor_idx in range(3):
        npu_tensors[tensor_idx][:16].zero_()
    for tensor_idx in range(3, 6):
        npu_tensors[tensor_idx][5:18].zero_()
    torch.npu.synchronize()

    require(
        worker.submit_load(9002, CPULoadStoreSpec(cpu_blocks), npu_spec),
        "CPU to NPU submission failed",
    )
    worker.wait({9002})
    finished = worker.get_finished()
    require(len(finished) == 1 and finished[0].success, "Load did not finish")
    torch.npu.synchronize()

    for tensor_idx in range(3):
        torch.testing.assert_close(
            npu_tensors[tensor_idx][:16].cpu(), originals[tensor_idx][:16]
        )
    for tensor_idx in range(3, 6):
        torch.testing.assert_close(
            npu_tensors[tensor_idx][5:18].cpu(), originals[tensor_idx][5:18]
        )

    worker.shutdown()
    print("PASS: Ascend NPU/CPU KV offload roundtrip is correct")


if __name__ == "__main__":
    main()
