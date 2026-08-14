# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
import torch

from vllm.models.deepseek_v4.amd import rocm as rocm_module
from vllm.models.deepseek_v4.amd.rocm import (
    DeepseekV4ROCMAiterMLASparseMetadataBuilder,
    DeepseekV4ROCMAiterSparseSWAMetadata,
    DeepseekV4ROCMAiterSparseSWAMetadataBuilder,
)
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWAMetadataBuilder,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    build_ragged_indices_from_dense_out,
)


def _build_ragged_reference(
    indices: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = lengths.clamp(min=0, max=indices.shape[-1]).to(torch.int32)
    rows = [row[:length] for row, length in zip(indices, lengths.tolist())]
    ragged = torch.cat(rows) if rows else torch.empty(0, dtype=torch.int32)
    indptr = torch.zeros(len(rows) + 1, dtype=torch.int32)
    torch.cumsum(lengths, dim=0, out=indptr[1:])
    return ragged, indptr


def _build_ragged_reference_out(
    indices: torch.Tensor,
    lengths: torch.Tensor,
    out_indices: torch.Tensor,
    out_indptr: torch.Tensor,
) -> None:
    ragged, indptr = _build_ragged_reference(indices, lengths)
    out_indptr.copy_(indptr)
    out_indices[: ragged.numel()].copy_(ragged)


@pytest.mark.parametrize(
    ("updated_indices", "updated_lens"),
    [
        (
            [
                [63, 64, 65, 66],
                [127, 128, 129, 130],
                [191, 192, 193, 194],
                [255, 256, 257, 258],
            ],
            [4, 3, 1, 0],
        ),
        (
            [
                [60, 61, 62, 63],
                [124, 125, 126, 127],
                [188, 189, 190, 191],
                [252, 253, 254, 255],
            ],
            [2, 0, 4, 0],
        ),
    ],
)
def test_rocm_swa_draft_update_refreshes_ragged_in_place(
    updated_indices: list[list[int]],
    updated_lens: list[int],
):
    builder = object.__new__(DeepseekV4ROCMAiterSparseSWAMetadataBuilder)
    dense_indices = torch.zeros((4, 1, 4), dtype=torch.int32)
    dense_lens = torch.zeros(4, dtype=torch.int32)
    ragged_indices = torch.full((16,), -1, dtype=torch.int32)
    ragged_indptr = torch.full((5,), -1, dtype=torch.int32)
    metadata = DeepseekV4ROCMAiterSparseSWAMetadata(
        block_table=torch.empty((4, 4), dtype=torch.int32),
        slot_mapping=torch.empty(4, dtype=torch.int64),
        block_size=64,
        decode_swa_indices=dense_indices,
        decode_swa_lens=dense_lens,
        decode_swa_ragged_indices=ragged_indices,
        decode_swa_ragged_indptr=ragged_indptr,
        num_decodes=4,
        num_decode_tokens=4,
    )
    ragged_indices_ptr = ragged_indices.data_ptr()
    ragged_indptr_ptr = ragged_indptr.data_ptr()

    def update_dense(_builder, updated_metadata):
        updated_metadata.decode_swa_indices.copy_(
            torch.tensor(updated_indices, dtype=torch.int32).view(4, 1, 4)
        )
        updated_metadata.decode_swa_lens.copy_(
            torch.tensor(updated_lens, dtype=torch.int32)
        )

    with (
        patch.object(
            DeepseekSparseSWAMetadataBuilder,
            "update_draft_decode_metadata",
            update_dense,
        ),
        patch.object(
            rocm_module,
            "build_ragged_indices_from_dense_out",
            _build_ragged_reference_out,
        ),
        patch.object(
            rocm_module,
            "build_ragged_indices_from_dense",
            side_effect=AssertionError("allocating ragged builder must not be called"),
        ),
    ):
        builder.update_draft_decode_metadata(metadata)

    expected_indices, expected_indptr = _build_ragged_reference(
        torch.tensor(updated_indices, dtype=torch.int32),
        torch.tensor(updated_lens, dtype=torch.int32),
    )
    assert metadata.decode_swa_ragged_indices is ragged_indices
    assert metadata.decode_swa_ragged_indptr is ragged_indptr
    assert ragged_indices.data_ptr() == ragged_indices_ptr
    assert ragged_indptr.data_ptr() == ragged_indptr_ptr
    torch.testing.assert_close(
        ragged_indices[: expected_indices.numel()], expected_indices
    )
    torch.testing.assert_close(ragged_indptr, expected_indptr)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize(
    "lengths",
    [
        [4, 3, 1, 0],
        [2, 0, 4, 0],
        [-1, 2, 5, 0],
    ],
)
def test_build_ragged_indices_from_dense_out_matches_reference(
    lengths: list[int],
):
    indices = torch.tensor(
        [
            [63, 64, 65, 66],
            [127, 128, 129, 130],
            [191, 192, 193, 194],
            [255, 256, 257, 258],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    out_indices = torch.full((indices.numel(),), -1, dtype=torch.int32, device="cuda")
    out_indptr = torch.full(
        (indices.shape[0] + 1,), -1, dtype=torch.int32, device="cuda"
    )
    indices_ptr = out_indices.data_ptr()
    indptr_ptr = out_indptr.data_ptr()

    build_ragged_indices_from_dense_out(indices, lens, out_indices, out_indptr)
    expected_indices, expected_indptr = _build_ragged_reference(
        indices.cpu(), torch.tensor(lengths, dtype=torch.int32)
    )

    assert out_indices.data_ptr() == indices_ptr
    assert out_indptr.data_ptr() == indptr_ptr
    torch.testing.assert_close(out_indptr.cpu(), expected_indptr)
    nnz = int(expected_indptr[-1].item())
    torch.testing.assert_close(out_indices[:nnz].cpu(), expected_indices[:nnz])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("num_rows", [1, 63, 64, 65, 127, 128, 129, 256])
def test_build_ragged_indices_from_dense_out_scan_sizes(num_rows: int):
    row_width = 4
    indices = torch.arange(
        num_rows * row_width,
        dtype=torch.int32,
        device="cuda",
    ).reshape(num_rows, row_width)
    lengths_cpu = (torch.arange(num_rows, dtype=torch.int32) % (row_width + 2)) - 1
    lens = lengths_cpu.to(device="cuda")
    out_indices = torch.full((indices.numel(),), -1, dtype=torch.int32, device="cuda")
    out_indptr = torch.full((num_rows + 1,), -1, dtype=torch.int32, device="cuda")

    build_ragged_indices_from_dense_out(indices, lens, out_indices, out_indptr)
    expected_indices, expected_indptr = _build_ragged_reference(
        indices.cpu(), lengths_cpu
    )

    torch.testing.assert_close(out_indptr.cpu(), expected_indptr)
    nnz = int(expected_indptr[-1].item())
    torch.testing.assert_close(out_indices[:nnz].cpu(), expected_indices)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_build_ragged_indices_from_dense_out_empty_rows():
    indices = torch.empty((0, 4), dtype=torch.int32, device="cuda")
    lengths = torch.empty((0,), dtype=torch.int32, device="cuda")
    out_indices = torch.empty((0,), dtype=torch.int32, device="cuda")
    out_indptr = torch.full((1,), -1, dtype=torch.int32, device="cuda")

    build_ragged_indices_from_dense_out(indices, lengths, out_indices, out_indptr)

    torch.testing.assert_close(out_indptr.cpu(), torch.zeros(1, dtype=torch.int32))


def test_rocm_swa_builder_enables_fused_draft_decode():
    builder_cls = DeepseekV4ROCMAiterSparseSWAMetadataBuilder
    assert builder_cls.supports_draft_decode_metadata_update


def test_rocm_mla_sparse_builder_keeps_fused_draft_decode_disabled():
    builder_cls = DeepseekV4ROCMAiterMLASparseMetadataBuilder
    assert not builder_cls.supports_draft_decode_metadata_update
