# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV checksums on CUDA: the Triton kernel matches the definition, and the
copy to host is ordered after the checksums for any reader. The definition and
helpers are those of test_kv_checksum.py (CPU tests)."""

import threading

import numpy as np
import pytest
import torch

from vllm.distributed.kv_transfer.kv_checksum import worker as checksum_worker
from vllm.distributed.kv_transfer.kv_checksum.checksum import (
    MAX_KV_HEADS_PER_LAYER,
    KVChecksumBlocks,
    KVChecksumScheduled,
)
from vllm.triton_utils import HAS_TRITON

from .test_kv_checksum import (
    BLOCKS,
    MAMBA_BLOCKS,
    NUM_BLOCKS,
    _cache,
    _check_requests_and_groups_of_one_step,
    _config,
    _group,
    _logical_kv,
    _mamba_cache,
    _mamba_name,
    _mamba_reference,
    _mamba_spec,
    _name,
    _reference,
    _send,
    _spec,
    _split_layer,
    _split_reference,
    _split_states,
    _worker,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

needs_triton = pytest.mark.skipif(not HAS_TRITON, reason="needs Triton")


@needs_triton
@pytest.mark.parametrize("token_major", [False, True])
@pytest.mark.parametrize("kernel_split", [1, 2])
def test_triton_kernel_matches_definition(token_major, kernel_split):
    """On CUDA the per-layer sums come from the Triton kernel, which must give
    the definition's values for any layout, kernel block split and partial
    last block, including a layer with a large slice factor."""
    layers = {0: _logical_kv(heads=2, seed=1), 3: _logical_kv(heads=2, seed=2)}
    caches = {
        _name(number): _cache(
            kv, token_major=token_major, kernel_split=kernel_split
        ).cuda()
        for number, kv in layers.items()
    }
    worker = _worker(_config(_group(_spec(2), *caches)), caches)
    assert all(view.use_triton for view in worker._layer_views[0])
    [group] = _send(worker).values()
    np.testing.assert_array_equal(group.values[:, 0], _reference(layers))


@needs_triton
def test_triton_kernel_reads_rows_of_several_requests_and_groups():
    """The kernel reads each group's rows from one shared copy, after a group
    with an odd number of rows."""
    _check_requests_and_groups_of_one_step("cuda")


@needs_triton
def test_triton_kernel_tiles_large_single_head_blocks():
    """One head of 32 tokens x 36 pieces is more than one 1024-piece tile and
    not a power of two (as MLA-like caches are), so the kernel's tiling and
    tail mask matter."""
    head_size, block_size = 72, 32  # K + V of 72 bf16 = 288 bytes = 36 pieces.
    logical = _logical_kv(heads=1, tokens=block_size, cell_bytes=4 * head_size)
    caches = {_name(0): _cache(logical).cuda()}
    spec = _spec(1, block_size=block_size, head_size=head_size)
    worker = _worker(_config(_group(spec)), caches, uniform_kv_heads=1)
    # 40 tokens: the second block is partial, the third empty.
    blocks = KVChecksumBlocks(num_tokens=40, groups={0: (0, [3, 1, 4])})
    [group] = _send(worker, blocks).values()
    np.testing.assert_array_equal(group.values[:, 0], _reference({0: logical}, blocks))


@needs_triton
def test_triton_kernel_accumulates_into_a_strided_column():
    """With several replica keys a group's sums form a matrix, each layer
    adding into one column: the kernel must honor the column's offset and
    stride and leave the other columns alone."""
    logical = _logical_kv(heads=2, seed=3)
    pieces = checksum_worker._as_pieces(_cache(logical).cuda(), NUM_BLOCKS)
    view = checksum_worker._LayerView(pieces, first_slice=0, replica_key=(1, 0))
    block_ids = torch.tensor([3, 1, 4], device="cuda")
    first_tokens = torch.tensor([0, 16, 32], device="cuda")
    valid_tokens = torch.tensor([16, 16, 8], device="cuda")
    out = torch.zeros(3, 3, dtype=torch.int64, device="cuda")
    view.accumulate(block_ids, first_tokens, valid_tokens, out[:, 1])
    values = out.cpu().numpy().view(np.uint64)
    np.testing.assert_array_equal(values[:, 1], _reference({0: logical}))
    assert not values[:, 0].any() and not values[:, 2].any()


@needs_triton
def test_triton_kernel_sums_multi_chunk_mamba_states():
    """A Mamba state of 24K pieces (192 KiB) spans two kernel chunks, the
    last partial, whose sums add up atomically into one value."""
    shapes, dtypes = ((256, 128), (128, 128)), (torch.float32, torch.float32)
    covered = 4 * (256 + 128) * 128
    cache, pages = _mamba_cache(covered, covered + 64)
    worker = _worker(
        _config(_group(_mamba_spec(shapes, dtypes), _mamba_name(2))),
        {_mamba_name(2): cache.cuda()},
    )
    expected = _mamba_reference({2 * MAX_KV_HEADS_PER_LAYER: pages}, 3, covered)
    assert _send(worker, MAMBA_BLOCKS)[0].values[0, 0] == expected


@needs_triton
@pytest.mark.parametrize("dim_first", [True, False])
def test_triton_kernel_sums_split_mamba_rows(dim_first):
    """Rows of TP-split Mamba state are slices of 2-byte pieces, which the
    kernel reads unsigned: TP2 shards with replicated B and C sum to the
    definition over the global state."""
    states = _split_states()
    total = np.uint64(0)
    for rank in range(2):
        spec, cache = _split_layer(states, 2, rank, one_group=True, dim_first=dim_first)
        worker = _worker(
            _config(_group(spec, _mamba_name(2))),
            {_mamba_name(2): cache.cuda()},
            tp_rank=rank,
            tp_size=2,
        )
        assert all(view.use_triton for view in worker._layer_views[0])
        for group in _send(worker, MAMBA_BLOCKS).values():
            with np.errstate(over="ignore"):
                total += group.values.sum(dtype=np.uint64)
    assert total == _split_reference(states)


def test_copy_is_ordered_after_checksums_for_any_reader():
    """The copy is queued with the checksums, so another thread finalizing
    (as the async output thread does) reads complete values even while the
    main stream is still busy."""
    logical = _logical_kv(heads=2)
    worker = _worker(_config(_group(_spec(2))), {_name(0): _cache(logical).cuda()})
    main_stream = torch.cuda.Stream()
    torch.cuda.set_stream(main_stream)
    try:
        torch.cuda._sleep(200_000_000)
        worker.bind(KVChecksumScheduled(reqs_to_send={"req": BLOCKS}))
        output = worker.post_forward(None, set(), set())
    finally:
        torch.cuda.set_stream(torch.cuda.default_stream())

    records = []
    reader = threading.Thread(target=lambda: records.extend(output.finalize()))
    reader.start()
    reader.join()
    np.testing.assert_array_equal(
        records[0].checksums["req"][0].values[:, 0], _reference({0: logical})
    )
