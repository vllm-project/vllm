# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Multi-node integration test for MessageQueue TCP fallback.

Verifies that when writer and readers span separate nodes (Docker containers
with isolated /dev/shm), `create_from_process_group` correctly detects
cross-node ranks via `in_the_same_node_as()` and falls back to ZMQ TCP
transport — and that data actually arrives.
"""

import numpy as np
import torch.distributed as dist

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.distributed.parallel_state import in_the_same_node_as


def main():
    dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size >= 2, (
        f"Need at least 2 ranks across nodes, got world_size={world_size}"
    )

    # Verify that in_the_same_node_as detects cross-node correctly
    status = in_the_same_node_as(dist.group.WORLD, source_rank=0)
    local_count = sum(status)
    print(
        f"[Rank {rank}] in_the_same_node_as(source=0): {status}  "
        f"(local={local_count}/{world_size})"
    )
    # With 2 Docker containers (1 proc each), rank 0 and rank 1 should be on different nodes.
    assert local_count < world_size, (
        f"Expected cross-node ranks but all {world_size} ranks appear local."
    )

    # Create MessageQueue
    writer_rank = 0
    mq = MessageQueue.create_from_process_group(
        dist.group.WORLD,
        max_chunk_bytes=1024 * 1024,  # 1 MiB
        max_chunks=10,
        writer_rank=writer_rank,
    )

    # Verify the transport path selection
    if rank == writer_rank:
        print(
            f"[Rank {rank}] Writer: n_local_reader={mq.n_local_reader}, "
            f"n_remote_reader={mq.n_remote_reader}"
        )
        assert mq.n_remote_reader > 0, (
            "Writer should have at least 1 remote (TCP) reader in a "
            "multi-node setup."
        )
    else:
        if status[rank]:
            assert mq._is_local_reader, (
                f"Rank {rank} is on the same node as writer but is not a "
                "local reader."
            )
            print(f"[Rank {rank}] Reader: local (shared memory)")
        else:
            assert mq._is_remote_reader, (
                f"Rank {rank} is on a different node but is not a remote "
                "(TCP) reader."
            )
            print(f"[Rank {rank}] Reader: remote (TCP)")

    # Test data transfer: simple objects
    dist.barrier()
    if rank == writer_rank:
        mq.enqueue("hello_from_node0")
    else:
        msg = mq.dequeue(timeout=10)
        assert msg == "hello_from_node0"
    dist.barrier()
    print(f"[Rank {rank}] Simple object test passed")

    # Test data transfer: numpy arrays
    np.random.seed(42)
    arrays = [np.random.randint(0, 100, size=np.random.randint(100, 5000))
              for _ in range(100)]

    dist.barrier()
    if rank == writer_rank:
        for arr in arrays:
            mq.enqueue(arr)
    else:
        for i, expected in enumerate(arrays):
            received = mq.dequeue(timeout=10)
            assert np.array_equal(expected, received), (
                f"Array mismatch at index {i}: "
                f"expected shape {expected.shape}, got shape {received.shape}"
            )
    dist.barrier()
    print(f"[Rank {rank}] Numpy array test passed")

    # Test data transfer: large payload (> max_chunk_bytes)
    dist.barrier()
    big_array = np.zeros(200_000, dtype=np.int64)  # ~1.6 MiB > 1 MiB chunk
    if rank == writer_rank:
        mq.enqueue(big_array)
    else:
        received = mq.dequeue(timeout=10)
        assert np.array_equal(big_array, received)
    dist.barrier()
    print(f"[Rank {rank}] Large payload test passed")

    # Done -- cleanup
    dist.barrier()
    print(f"[Rank {rank}] All MessageQueue TCP multi-node tests passed!")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
