# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.sparse_utils import (
    build_rotated_dcp_peer_block_table,
    convert_global_indices_to_dcp_peer_slots,
)


def _build_rotated_dcp_peer_block_table_ref(
    gathered_block_tables: torch.Tensor,
    *,
    local_rank: int,
    peer_block_stride: int,
) -> torch.Tensor:
    dcp_size, num_requests, max_owner_pages = gathered_block_tables.shape
    expected = torch.full(
        (num_requests, dcp_size * max_owner_pages),
        -1,
        dtype=torch.int32,
    )
    for request in range(num_requests):
        for logical_page in range(dcp_size * max_owner_pages):
            owner = logical_page % dcp_size
            owner_local_page = logical_page // dcp_size
            physical_block = int(
                gathered_block_tables[owner, request, owner_local_page]
            )
            rotated_owner = (owner - local_rank) % dcp_size
            peer_block = rotated_owner * peer_block_stride + physical_block
            if (
                0 <= physical_block < peer_block_stride
                and peer_block <= torch.iinfo(torch.int32).max
            ):
                expected[request, logical_page] = peer_block
    return expected


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("local_rank", [0, 1, 3])
def test_build_rotated_dcp_peer_block_table(local_rank: int):
    gathered_block_tables = torch.tensor(
        [
            [[4, 8, -1], [12, 2, 7]],
            [[1, 5, -1], [10, -1, -1]],
            [[3, 6, -1], [11, 6, 1]],
            [[9, 0, -1], [5, 9, -1]],
        ],
        dtype=torch.int32,
    )
    expected = _build_rotated_dcp_peer_block_table_ref(
        gathered_block_tables,
        local_rank=local_rank,
        peer_block_stride=16,
    )

    actual = build_rotated_dcp_peer_block_table(
        gathered_block_tables.cuda(),
        local_rank=local_rank,
        peer_block_stride=16,
        cp_kv_cache_interleave_size=64,
        block_size=64,
        BLOCK_N=4,
    )

    torch.testing.assert_close(actual.cpu(), expected)
    assert actual.dtype == torch.int32
    if local_rank == 0:
        # Columns are global logical-page order (owner = p % 4), not
        # owner-major order.
        torch.testing.assert_close(
            actual[0].cpu(),
            torch.tensor(
                [4, 17, 35, 57, 8, 21, 38, 48, -1, -1, -1, -1],
                dtype=torch.int32,
            ),
        )
    # The first request pads every owner's third local page.
    assert (actual[0, 8:] == -1).all()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
def test_build_rotated_dcp_peer_block_table_uses_strides_and_fails_closed():
    storage = torch.full((3, 2, 2, 8), -99, dtype=torch.int32)
    gathered_block_tables = storage[:, :, 1, 1::2]
    gathered_block_tables.copy_(
        torch.tensor(
            [
                [[0, 7, -1, -1], [2, 3, 4, -1]],
                [[1, 8, -1, -1], [5, -1, 6, -1]],
                [[6, torch.iinfo(torch.int32).max, -1, -1], [7, 1, 0, -1]],
            ],
            dtype=torch.int32,
        )
    )
    assert not gathered_block_tables.is_contiguous()
    expected = _build_rotated_dcp_peer_block_table_ref(
        gathered_block_tables,
        local_rank=2,
        peer_block_stride=8,
    )

    storage_gpu = storage.cuda()
    gathered_gpu = storage_gpu[:, :, 1, 1::2]
    assert not gathered_gpu.is_contiguous()
    actual = build_rotated_dcp_peer_block_table(
        gathered_gpu,
        local_rank=2,
        peer_block_stride=8,
        cp_kv_cache_interleave_size=32,
        block_size=32,
        BLOCK_N=8,
    )

    torch.testing.assert_close(actual.cpu(), expected)
    assert actual[0, 4].item() == -1  # physical block equals peer stride
    assert actual[0, 5].item() == -1  # very large physical block


@pytest.mark.parametrize(
    ("tensor", "kwargs", "error", "match"),
    [
        (
            torch.empty((2, 1, 1), dtype=torch.int64),
            {},
            TypeError,
            "dtype int32",
        ),
        (
            torch.empty((2, 1), dtype=torch.int32),
            {},
            ValueError,
            "must have shape",
        ),
        (
            torch.empty((2, 1, 1), dtype=torch.int32),
            {"local_rank": 2},
            ValueError,
            "local_rank must be",
        ),
        (
            torch.empty((2, 1, 1), dtype=torch.int32),
            {"peer_block_stride": 0},
            ValueError,
            "peer_block_stride must be positive",
        ),
        (
            torch.empty((2, 1, 1), dtype=torch.int32),
            {"peer_block_stride": torch.iinfo(torch.int64).max},
            ValueError,
            "address space exceeds int64",
        ),
        (
            torch.empty((2, 1, 1), dtype=torch.int32),
            {"cp_kv_cache_interleave_size": 32},
            ValueError,
            "interleave_size == block_size",
        ),
        (
            torch.empty((2, 1, 1), dtype=torch.int32),
            {"BLOCK_N": 7},
            ValueError,
            "positive power of two",
        ),
    ],
)
def test_build_rotated_dcp_peer_block_table_validates_contract(
    tensor: torch.Tensor,
    kwargs: dict[str, int],
    error: type[Exception],
    match: str,
):
    arguments = {
        "local_rank": 0,
        "peer_block_stride": 8,
        "cp_kv_cache_interleave_size": 64,
        "block_size": 64,
    }
    arguments.update(kwargs)
    with pytest.raises(error, match=match):
        build_rotated_dcp_peer_block_table(tensor, **arguments)


def _convert_global_indices_to_dcp_peer_slots_ref(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    dcp_size: int,
    blocks_per_peer: int,
    interleave: int,
    block_size: int,
) -> torch.Tensor:
    expected = torch.full_like(token_indices, -1)
    for row in range(token_indices.shape[0]):
        request = int(req_id[row])
        if request < 0 or request >= block_table.shape[0]:
            continue
        for column in range(token_indices.shape[1]):
            token = int(token_indices[row, column])
            if token < 0:
                continue
            owner = (token // interleave) % dcp_size
            local = (token // (dcp_size * interleave)) * interleave + token % interleave
            logical_block = local // block_size
            if logical_block >= block_table.shape[1]:
                continue
            physical_block = int(block_table[request, logical_block])
            if physical_block < 0 or physical_block >= blocks_per_peer:
                continue
            expected[row, column] = (
                owner * blocks_per_peer + physical_block
            ) * block_size + local % block_size
    return expected


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("interleave", [1, 2, 4])
def test_convert_global_indices_to_dcp_peer_slots(interleave: int):
    dcp_size = 2
    block_size = 4
    blocks_per_peer = 16
    req_id = torch.tensor([0, 1], dtype=torch.int32)
    block_table = torch.tensor(
        [
            [3, 7, 9],
            [5, 1, 6],
        ],
        dtype=torch.int32,
    )
    # An odd width spanning several tiles exercises masked tail stores and
    # cross-tile valid-count atomics.
    token_indices = torch.full((2, 257), -1, dtype=torch.int32)
    token_indices[0, :8] = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11])
    token_indices[1, :8] = torch.tensor([4, 5, 6, 7, 12, 13, 14, 15])
    expected = _convert_global_indices_to_dcp_peer_slots_ref(
        req_id,
        block_table,
        token_indices,
        dcp_size,
        blocks_per_peer,
        interleave,
        block_size,
    )

    actual, valid_counts = convert_global_indices_to_dcp_peer_slots(
        req_id.cuda(),
        block_table.cuda(),
        token_indices.cuda(),
        dcp_size=dcp_size,
        blocks_per_peer=blocks_per_peer,
        cp_kv_cache_interleave_size=interleave,
        block_size=block_size,
        BLOCK_N=64,
        return_valid_counts=True,
    )

    torch.testing.assert_close(actual.cpu(), expected)
    torch.testing.assert_close(
        valid_counts.cpu(),
        (expected >= 0).sum(dim=1, dtype=torch.int32),
    )
    assert actual.dtype == token_indices.dtype
    assert (actual[:, 8:] == -1).all()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
def test_convert_global_indices_to_dcp_peer_slots_reuses_outputs():
    req_id = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    block_table = torch.tensor([[3, 7], [5, 1]], dtype=torch.int32, device="cuda")
    token_indices = torch.tensor(
        [[0, 1, -1, -1], [4, 5, 6, -1]],
        dtype=torch.int32,
        device="cuda",
    )
    out = torch.full_like(token_indices, -99)
    valid_counts_out = torch.full((2,), 99, dtype=torch.int32, device="cuda")

    actual, valid_counts = convert_global_indices_to_dcp_peer_slots(
        req_id,
        block_table,
        token_indices,
        dcp_size=2,
        blocks_per_peer=16,
        cp_kv_cache_interleave_size=2,
        block_size=4,
        BLOCK_N=4,
        return_valid_counts=True,
        out=out,
        valid_counts_out=valid_counts_out,
    )

    assert actual.data_ptr() == out.data_ptr()
    assert valid_counts.data_ptr() == valid_counts_out.data_ptr()
    torch.testing.assert_close(
        valid_counts,
        torch.tensor([2, 3], dtype=torch.int32, device="cuda"),
    )
    assert (actual[token_indices < 0] == -1).all()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
def test_convert_global_indices_to_dcp_peer_slots_reuses_empty_outputs():
    req_id = torch.empty(0, dtype=torch.int32, device="cuda")
    block_table = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    token_indices = torch.empty((0, 4), dtype=torch.int32, device="cuda")
    out = torch.empty_like(token_indices)
    valid_counts_out = torch.empty(0, dtype=torch.int32, device="cuda")

    actual, valid_counts = convert_global_indices_to_dcp_peer_slots(
        req_id,
        block_table,
        token_indices,
        dcp_size=4,
        blocks_per_peer=16,
        cp_kv_cache_interleave_size=64,
        block_size=64,
        return_valid_counts=True,
        out=out,
        valid_counts_out=valid_counts_out,
    )

    assert actual is out
    assert valid_counts is valid_counts_out


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
def test_convert_global_indices_to_dcp_peer_slots_uses_strides():
    req_id_storage = torch.tensor([9, 0, 9, 1], dtype=torch.int32)
    req_id = req_id_storage[1::2]
    block_table_storage = torch.tensor(
        [
            [3, 99, 7, 99],
            [5, 99, 1, 99],
        ],
        dtype=torch.int32,
    )
    block_table = block_table_storage[:, ::2]
    token_storage = torch.full((2, 256), -1, dtype=torch.int32)
    token_storage[0, :16:2] = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11])
    token_storage[1, :16:2] = torch.tensor([4, 5, 6, 7, 12, 13, 14, 15])
    token_indices = token_storage[:, ::2]
    assert not req_id.is_contiguous()
    assert not block_table.is_contiguous()
    assert not token_indices.is_contiguous()

    expected = _convert_global_indices_to_dcp_peer_slots_ref(
        req_id,
        block_table,
        token_indices,
        dcp_size=2,
        blocks_per_peer=16,
        interleave=2,
        block_size=4,
    )
    req_id_gpu = req_id_storage.cuda()[1::2]
    block_table_gpu = block_table_storage.cuda()[:, ::2]
    token_storage_gpu = token_storage.cuda()
    token_indices_gpu = token_storage_gpu[:, ::2]
    assert not req_id_gpu.is_contiguous()
    assert not block_table_gpu.is_contiguous()
    assert not token_indices_gpu.is_contiguous()
    actual = convert_global_indices_to_dcp_peer_slots(
        req_id_gpu,
        block_table_gpu,
        token_indices_gpu,
        dcp_size=2,
        blocks_per_peer=16,
        cp_kv_cache_interleave_size=2,
        block_size=4,
    )

    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
def test_convert_global_indices_to_dcp_peer_slots_rejects_invalid_entries():
    req_id = torch.tensor([0, 1, 2, -1], dtype=torch.int32)
    block_table = torch.tensor(
        [
            [2, -1],
            [3, 8],
        ],
        dtype=torch.int32,
    )
    token_indices = torch.full((4, 128), -1, dtype=torch.int32)
    token_indices[:, :2] = torch.tensor(
        [
            [-1, 8],  # -1 token and out-of-range logical block
            [0, 2],  # valid block and physical block outside peer stride
            [0, 1],  # request beyond block-table rows
            [0, 1],  # negative request
        ],
        dtype=torch.int32,
    )

    actual, valid_counts = convert_global_indices_to_dcp_peer_slots(
        req_id.cuda(),
        block_table.cuda(),
        token_indices.cuda(),
        dcp_size=1,
        blocks_per_peer=8,
        block_size=2,
        return_valid_counts=True,
    )

    expected = torch.full_like(token_indices, -1)
    expected[1, 0] = 6
    torch.testing.assert_close(actual.cpu(), expected)
    torch.testing.assert_close(
        valid_counts.cpu(),
        torch.tensor([0, 1, 0, 0], dtype=torch.int32),
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dcp_size": 0}, "dcp_size must be positive"),
        ({"blocks_per_peer": 0}, "blocks_per_peer must be positive"),
        (
            {"cp_kv_cache_interleave_size": 3},
            "block_size .* must be .* divisible",
        ),
        ({"BLOCK_N": 3}, "BLOCK_N must be a positive power of two"),
    ],
)
def test_convert_global_indices_to_dcp_peer_slots_validates_layout(
    kwargs: dict[str, int], match: str
):
    arguments = {
        "dcp_size": 2,
        "blocks_per_peer": 4,
        "cp_kv_cache_interleave_size": 1,
        "block_size": 4,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=match):
        convert_global_indices_to_dcp_peer_slots(
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([[0]], dtype=torch.int32),
            torch.tensor([[0]], dtype=torch.int32),
            **arguments,
        )
