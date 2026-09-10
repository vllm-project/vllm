# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FSOffloadingWorker._submit_io block_indices handling."""

import numpy as np

from vllm.v1.kv_offload.base import DevicePointers
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.fs.worker import FSOffloadingWorker


class RecordingFSWorker(FSOffloadingWorker):
    """Concrete FSOffloadingWorker that records I/O ops instead of doing I/O."""

    def __init__(self, block_size_factor: int):
        mapper = FileMapper("/tmp/test", ".bin")
        super().__init__(file_mapper=mapper, block_size_factor=block_size_factor)
        self.recorded: list[tuple[str, list[tuple[int, int, int]]]] = []

    def write_block(self, file_path, ops):
        self.recorded.append((file_path, ops))

    def read_block(self, file_path, ops):
        self.recorded.append((file_path, ops))


def _make_device_ptrs(
    n_blocks: int,
    n_data_refs: int,
    block_size: int,
    block_indices: tuple[int, ...],
) -> DevicePointers:
    """Build a single-group DevicePointers with sequential fake pointers."""
    n_ops = n_blocks * n_data_refs
    ptrs = np.arange(1000, 1000 + n_ops, dtype=np.uint64) * block_size
    sizes = np.full(n_ops, block_size, dtype=np.uint64)
    return DevicePointers(
        ptrs=ptrs,
        sizes=sizes,
        group_block_counts=(n_blocks,),
        group_data_ref_counts=(n_data_refs,),
        block_indices=block_indices,
    )


def test_offset_within_single_file():
    """block_size_factor=3, block_indices=(1,), 2 blocks → 1 file at positions 1,2."""
    worker = RecordingFSWorker(block_size_factor=3)
    block_size = 64
    device_ptrs = _make_device_ptrs(
        n_blocks=2, n_data_refs=1, block_size=block_size, block_indices=(1,)
    )
    keys = ["key0"]

    futures, total_bytes = worker._submit_io(device_ptrs, keys, is_store=True)
    for f in futures:
        f.result()

    assert len(worker.recorded) == 1
    _, ops = worker.recorded[0]
    assert len(ops) == 2
    # file offsets should be at positions 1 and 2, not 0 and 1
    assert ops[0][2] == 1 * block_size
    assert ops[1][2] == 2 * block_size


def test_offset_spanning_two_files():
    """block_size_factor=3, block_indices=(2,), 3 blocks → 2 files.

    First file: 1 block at position 2.
    Second file: 2 blocks at positions 0,1.
    """
    worker = RecordingFSWorker(block_size_factor=3)
    block_size = 64
    device_ptrs = _make_device_ptrs(
        n_blocks=3, n_data_refs=1, block_size=block_size, block_indices=(2,)
    )
    keys = ["key0", "key1"]

    futures, total_bytes = worker._submit_io(device_ptrs, keys, is_store=True)
    for f in futures:
        f.result()

    assert len(worker.recorded) == 2

    # First file: 1 block at file position 2
    _, ops0 = worker.recorded[0]
    assert len(ops0) == 1
    assert ops0[0][2] == 2 * block_size

    # Second file: 2 blocks at file positions 0,1
    _, ops1 = worker.recorded[1]
    assert len(ops1) == 2
    assert ops1[0][2] == 0 * block_size
    assert ops1[1][2] == 1 * block_size
