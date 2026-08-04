# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Tuple
import argparse
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_allocators.ad_hoc_memory_allocator import AdHocMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.transfer_channel.nixl_channel import NixlChannel
from lmcache.v1.transfer_channel.transfer_utils import get_correct_device

pytest.importorskip("nixl", reason="nixl package is required for nixl tests")

logger = init_logger(__name__)


def generate_test_data(
    num_objs: int, shape: torch.Size, dtype: torch.dtype = torch.bfloat16
) -> Tuple[List[CacheEngineKey], List[MemoryObj]]:
    keys = []
    objs = []
    allocator = AdHocMemoryAllocator(
        device="cuda",  # Assuming we are using CUDA for the test
    )
    for i in range(num_objs):
        keys.append(
            CacheEngineKey(
                model_name="test_model",
                world_size=1,
                worker_id=0,
                chunk_hash=i,
                dtype=dtype,
            )
        )
        obj = allocator.allocate(shape, dtype, fmt=MemoryFormat.KV_2LTD)
        obj.tensor.fill_(i + 1)  # Fill with some test data, e.g., the index
        objs.append(obj)
    return keys, objs


def calculate_throughput(total_bytes: int, elapsed_time: float) -> float:
    """Calculate throughput in GB/s"""
    if elapsed_time == 0:
        return float("inf")
    gb = total_bytes / (1024 * 1024 * 1024)
    return gb / elapsed_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test NixlChannel with sender/receiver roles"
    )
    parser.add_argument(
        "--role",
        type=str,
        required=True,
        choices=["sender", "receiver"],
        help="Role of this instance (sender or receiver)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host name/IP for connection",
    )
    parser.add_argument(
        "--port", type=int, default=5555, help="Port number for connection"
    )
    parser.add_argument(
        "--num-objs", type=int, default=100, help="Number of objects to send"
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=1,
        help="Number of rounds to run the experiment",
    )
    args = parser.parse_args()

    # Generate test data
    keys, objs = generate_test_data(args.num_objs, torch.Size([32, 2, 256, 1024]))
    total_size = sum(obj.get_size() for obj in objs)
    logger.info(
        f"Generated {len(objs)} objects with total size "
        f"{total_size / (1024 * 1024):.2f} MB"
    )

    # Common configuration
    buffer_size = 2**32  # 4GB
    buffer_device = get_correct_device("cuda", 0)  # Use first GPU

    # Get buffer pointer from first object
    buffer_ptr = objs[0].metadata.address
    align_bytes = 4096  # Standard page size

    # Create the NixlChannel
    channel = NixlChannel(
        async_mode=False,
        role=args.role,
        buffer_ptr=buffer_ptr,
        buffer_size=buffer_size,
        align_bytes=align_bytes,
        tp_rank=0,
        peer_init_url=f"tcp://{args.host}:{args.port}",
        backends=["UCX"],
    )

    if args.role == "sender":
        throughputs = []
        for i in range(args.num_rounds):
            logger.info(f"Round {i + 1}/{args.num_rounds}")
            start_time = time.time()
            num_sent = channel.batched_send(objs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            throughput = calculate_throughput(total_size, elapsed_time)
            logger.info(f"Sent {num_sent} objects in {elapsed_time:.6f} seconds")
            logger.info(f"Throughput: {throughput:.2f} GB/s")
            throughputs.append(throughput)
        avg_throughput = sum(throughputs) / len(throughputs)
        logger.info(f"Average throughput: {avg_throughput:.2f} GB/s")
    else:  # receiver
        for i in range(args.num_rounds):
            logger.info(f"Round {i + 1}/{args.num_rounds}")
            start_time = time.time()
            num_received = channel.batched_recv(objs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            throughput = calculate_throughput(total_size, elapsed_time)
            logger.info(
                f"Received {num_received} objects in {elapsed_time:.6f} seconds"
            )
            logger.info(f"Throughput: {throughput:.2f} GB/s")

            # Verify data
            for i, (received_obj, original_obj) in enumerate(
                zip(objs, objs, strict=False)
            ):
                assert received_obj.tensor is not None, (
                    f"Received tensor is None at index {i}"
                )
                assert original_obj.tensor is not None, (
                    f"Original tensor is None at index {i}"
                )
                assert torch.allclose(received_obj.tensor, original_obj.tensor), (
                    f"Data mismatch at index {i}: "
                    f"received {received_obj.tensor.mean().item()}"
                    f" but expected {original_obj.tensor.mean().item()}"
                )

    # Wait a bit before closing
    time.sleep(2)
    channel.close()
    logger.info("Test completed")
