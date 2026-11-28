import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.distributed.device_communicators.cpu_communicator import (
    CpuCommunicator,
)


def worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
    )

    try:
        communicator = CpuCommunicator(
            cpu_group=dist.group.WORLD,
            device=torch.device("cpu"),
            device_group=dist.group.WORLD,
        )

        sizes = [2, 3]
        local = torch.arange(sizes[rank], dtype=torch.float32) + rank * 10

        result = communicator.all_gatherv(
            local,
            dim=0,
            sizes=sizes,
        )

        expected = torch.tensor(
            [0, 1, 10, 11, 12],
            dtype=torch.float32,
        )

        torch.testing.assert_close(result, expected)

        # Test gathering multiple tensors with different feature dimensions
        hidden_states = (
            torch.arange(sizes[rank] * 4, dtype=torch.float32).reshape(sizes[rank], 4)
            + rank * 100
        )

        router_logits = torch.full(
            (sizes[rank], 3),
            float(rank + 1),
        )

        gathered = communicator.all_gatherv(
            [hidden_states, router_logits],
            dim=0,
            sizes=sizes,
        )

        expected_hidden = torch.cat(
            [
                torch.arange(8, dtype=torch.float32).reshape(2, 4),
                torch.arange(12, dtype=torch.float32).reshape(3, 4) + 100,
            ],
            dim=0,
        )

        expected_logits = torch.cat(
            [
                torch.ones((2, 3)),
                torch.full((3, 3), 2.0),
            ],
            dim=0,
        )

        assert isinstance(gathered, list)
        assert len(gathered) == 2

        torch.testing.assert_close(gathered[0], expected_hidden)
        torch.testing.assert_close(gathered[1], expected_logits)

        # Case 1: Equal sizes with sizes explicitly provided
        equal_local = torch.tensor([rank * 10.0, rank * 10.0 + 1])
        equal_result = communicator.all_gatherv(equal_local, dim=0, sizes=[2, 2])
        torch.testing.assert_close(equal_result, torch.tensor([0.0, 1.0, 10.0, 11.0]))

        # Case 2: Equal sizes with sizes=None
        uniform_result = communicator.all_gatherv(equal_local, dim=0, sizes=None)
        torch.testing.assert_close(uniform_result, torch.tensor([0.0, 1.0, 10.0, 11.0]))

        # Case 3: One rank contributes zero elements
        zero_sizes = [0, 3]
        zero_local = (
            torch.empty(0, dtype=torch.float32)
            if rank == 0
            else torch.tensor([10.0, 11.0, 12.0])
        )
        zero_result = communicator.all_gatherv(zero_local, dim=0, sizes=zero_sizes)
        torch.testing.assert_close(zero_result, torch.tensor([10.0, 11.0, 12.0]))

        # Case 4: Both ranks contribute zero elements
        empty_local = torch.empty((0, 4), dtype=torch.float32)

        empty_result = communicator.all_gatherv(
            empty_local,
            dim=0,
            sizes=[0, 0],
        )

        assert empty_result.shape == (0, 4)
        assert empty_result.numel() == 0

        print(f"Rank {rank}: PASS", flush=True)

    finally:
        dist.destroy_process_group()


def test_cpu_all_gatherv():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    mp.spawn(worker, args=(2, port), nprocs=2, join=True)
