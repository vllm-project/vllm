# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch.multiprocessing import spawn

from vllm.distributed.utils import (
    stateless_destroy_torch_distributed_process_group,
    stateless_init_torch_distributed_process_group)


def worker_process(rank: int, world_size: int, host: str, port1: int,
                   port2: int):
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Create first process group with all workers
    pg1 = stateless_init_torch_distributed_process_group(host=host,
                                                         port=port1,
                                                         rank=rank,
                                                         world_size=world_size,
                                                         backend="gloo")

    # Create second process group with worldsize-1 workers (excluding last rank)
    pg2 = None
    if rank < world_size - 1:
        pg2 = stateless_init_torch_distributed_process_group(
            host=host,
            port=port2,
            rank=rank,
            world_size=world_size - 1,
            backend="gloo")

    # Test both groups work simultaneously
    tensor1 = torch.tensor([rank], dtype=torch.float32)
    torch.distributed.all_reduce(tensor1, group=pg1)
    expected1 = sum(range(world_size))
    assert tensor1.item(
    ) == expected1, f"PG1 failed: got {tensor1.item()}, expected {expected1}"
    print(f"Rank {rank}: PG1 all_reduce passed")

    if pg2 is not None:
        tensor2 = torch.tensor([rank], dtype=torch.float32)
        torch.distributed.all_reduce(tensor2, group=pg2)
        expected2 = sum(range(world_size - 1))
        assert tensor2.item() == expected2, (
            f"PG2 failed: got {tensor2.item()}, expected {expected2}")
        print(f"Rank {rank}: PG2 all_reduce passed")

    # Destroy first process group
    stateless_destroy_torch_distributed_process_group(pg1)
    print(f"Rank {rank}: PG1 destroyed")

    # Last rank exits here
    if rank == world_size - 1:
        print(f"Rank {rank}: Exiting")
        return

    # Test second group still works after destroying
    # first group and last rank exit
    tensor3 = torch.tensor([rank * 10], dtype=torch.float32)
    torch.distributed.all_reduce(tensor3, group=pg2)
    expected3 = sum(i * 10 for i in range(world_size - 1))
    assert tensor3.item() == expected3, (
        f"PG2 after PG1 destroy failed: got {tensor3.item()}, "
        f"expected {expected3}")
    print(f"Rank {rank}: PG2 after PG1 destroy passed")

    # Clean up
    if pg2 is not None:
        stateless_destroy_torch_distributed_process_group(pg2)
    print(f"Rank {rank}: PG2 destroyed")


def test_stateless_process_groups():
    assert not torch.distributed.is_initialized(
    ), "torch.distributed should not be initialized"

    world_size = 4
    host = "127.0.0.1"
    port1 = 29600
    port2 = 29601

    print(f"Testing stateless process groups with world_size={world_size}")

    spawn(worker_process,
          args=(world_size, host, port1, port2),
          nprocs=world_size,
          join=True)

    print("Test completed successfully!")


if __name__ == "__main__":
    test_stateless_process_groups()
