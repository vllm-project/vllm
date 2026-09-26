# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest.mock as mock
from multiprocessing import shared_memory
import pytest
import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.distributed.utils import StatelessProcessGroup


def test_in_the_same_node_as_retains_shm_until_after_barrier():
    """
    Ensure the creator's shared memory handle is not closed before the
    process-group barrier executes, preventing a race condition where peer
    ranks open a named shared memory segment on Windows after the creator
    closed its handle.
    """
    events = []

    class MockStatelessProcessGroup:
        rank = 0
        world_size = 1

        def broadcast_obj(self, obj, src=0):
            return obj

        def barrier(self):
            events.append("barrier")

    original_shm_init = shared_memory.SharedMemory.__init__
    original_shm_close = shared_memory.SharedMemory.close
    original_shm_unlink = shared_memory.SharedMemory.unlink

    def mock_close(self):
        events.append("shm_close")
        return original_shm_close(self)

    def mock_unlink(self):
        events.append("shm_unlink")
        return original_shm_unlink(self)

    with mock.patch.object(shared_memory.SharedMemory, "close", mock_close), \
         mock.patch.object(shared_memory.SharedMemory, "unlink", mock_unlink):
        pg = MockStatelessProcessGroup()
        result = in_the_same_node_as(pg, source_rank=0)

    assert result == [True]
    # Barrier MUST occur before shm_close and shm_unlink
    assert "barrier" in events
    assert "shm_close" in events
    barrier_idx = events.index("barrier")
    close_idx = events.index("shm_close")
    assert barrier_idx < close_idx, f"Barrier at {barrier_idx} was not before close at {close_idx}. Events: {events}"
