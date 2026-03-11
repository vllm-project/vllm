# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the fused EAGLE slot mapping kernel."""

import pytest
import torch

from vllm.v1.spec_decode.utils import (
    PADDING_SLOT_ID,
    eagle_step_update_slot_mapping_and_metadata,
)

# Skip if no CUDA - Triton kernel requires GPU
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for EAGLE kernel tests", allow_module_level=True)


def _reference_eagle_step_slot_mapping(
    positions_1d: torch.Tensor,
    block_table_tensor: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_model_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Python reference for eagle_step_update_slot_mapping_and_metadata."""
    new_positions = positions_1d + 1
    exceeds_max = new_positions >= max_model_len
    clamped_positions = torch.where(
        exceeds_max, torch.zeros_like(positions_1d), new_positions
    )
    block_numbers = (clamped_positions // block_size).clamp(
        max=block_table_tensor.shape[1] - 1
    )
    block_ids = block_table_tensor[
        torch.arange(positions_1d.shape[0], device=positions_1d.device),
        block_numbers.long(),
    ].long()
    slot_mapping = block_ids * block_size + (clamped_positions % block_size)
    slot_mapping = torch.where(
        exceeds_max, torch.full_like(slot_mapping, PADDING_SLOT_ID), slot_mapping
    )
    new_seq_lens = torch.where(exceeds_max, torch.ones_like(seq_lens), seq_lens + 1)
    new_seq_lens = new_seq_lens.clamp(max=max_model_len)
    return clamped_positions, slot_mapping, new_seq_lens


def test_eagle_step_slot_mapping_kernel():
    """Test fused kernel matches Python reference for slot mapping and metadata."""
    device = torch.device("cuda")
    batch_size = 32
    block_size = 16
    max_model_len = 4096
    n_blocks_per_req = (max_model_len + block_size - 1) // block_size

    positions_1d = torch.randint(
        0, max_model_len - 10, (batch_size,), dtype=torch.int64, device=device
    )
    block_table_tensor = torch.randint(
        0, 1000, (batch_size, n_blocks_per_req), dtype=torch.int32, device=device
    )
    seq_lens = torch.randint(1, 100, (batch_size,), dtype=torch.int32, device=device)

    ref_clamped, ref_slot, ref_seq_lens = _reference_eagle_step_slot_mapping(
        positions_1d.clone(),
        block_table_tensor,
        seq_lens.clone(),
        block_size,
        max_model_len,
    )

    out_clamped = torch.zeros(batch_size, dtype=torch.int64, device=device)
    out_slot = torch.zeros(batch_size, dtype=torch.int64, device=device)
    seq_lens_copy = seq_lens.clone()
    eagle_step_update_slot_mapping_and_metadata(
        positions_1d=positions_1d,
        block_table_tensor=block_table_tensor,
        seq_lens=seq_lens_copy,
        block_size=block_size,
        max_model_len=max_model_len,
        out_clamped_positions=out_clamped,
        out_slot_mapping=out_slot,
    )

    assert torch.equal(out_clamped, ref_clamped), (
        f"clamped: {out_clamped} vs {ref_clamped}"
    )
    assert torch.equal(out_slot, ref_slot), f"slot: {out_slot} vs {ref_slot}"
    assert torch.equal(seq_lens_copy, ref_seq_lens), (
        f"seq_lens: {seq_lens_copy} vs {ref_seq_lens}"
    )


def test_eagle_step_slot_mapping_kernel_exceeds_max():
    """Test fused kernel when position exceeds max_model_len."""
    device = torch.device("cuda")
    batch_size = 4
    block_size = 16
    max_model_len = 100
    n_blocks_per_req = (max_model_len + block_size - 1) // block_size

    positions_1d = torch.tensor([50, 98, 99, 100], dtype=torch.int64, device=device)
    block_table_tensor = torch.randint(
        0, 100, (batch_size, n_blocks_per_req), dtype=torch.int32, device=device
    )
    seq_lens = torch.tensor([51, 99, 100, 101], dtype=torch.int32, device=device)

    out_clamped = torch.zeros(batch_size, dtype=torch.int64, device=device)
    out_slot = torch.zeros(batch_size, dtype=torch.int64, device=device)
    eagle_step_update_slot_mapping_and_metadata(
        positions_1d=positions_1d,
        block_table_tensor=block_table_tensor,
        seq_lens=seq_lens,
        block_size=block_size,
        max_model_len=max_model_len,
        out_clamped_positions=out_clamped,
        out_slot_mapping=out_slot,
    )

    assert out_clamped[0].item() == 51
    assert out_clamped[1].item() == 99
    assert out_clamped[2].item() == 0
    assert out_clamped[3].item() == 0
    assert out_slot[2].item() == PADDING_SLOT_ID
    assert out_slot[3].item() == PADDING_SLOT_ID
    assert seq_lens[2].item() == 1
    assert seq_lens[3].item() == 1


def test_eagle_step_slot_mapping_kernel_cudagraph_padding():
    """Test that padding threads write PADDING_SLOT_ID when
    input_batch_size > batch_size (cudagraph padding)."""
    device = torch.device("cuda")
    batch_size = 4
    input_batch_size = 8
    block_size = 16
    max_model_len = 4096
    n_blocks_per_req = (max_model_len + block_size - 1) // block_size

    positions_1d = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device=device)
    block_table_tensor = torch.randint(
        0, 100, (batch_size, n_blocks_per_req), dtype=torch.int32, device=device
    )
    seq_lens = torch.tensor([11, 21, 31, 41], dtype=torch.int32, device=device)

    ref_clamped, ref_slot, ref_seq_lens = _reference_eagle_step_slot_mapping(
        positions_1d.clone(),
        block_table_tensor,
        seq_lens.clone(),
        block_size,
        max_model_len,
    )

    out_clamped = torch.zeros(batch_size, dtype=torch.int64, device=device)
    out_slot = torch.full((input_batch_size,), -999, dtype=torch.int64, device=device)
    seq_lens_copy = seq_lens.clone()
    eagle_step_update_slot_mapping_and_metadata(
        positions_1d=positions_1d,
        block_table_tensor=block_table_tensor,
        seq_lens=seq_lens_copy,
        block_size=block_size,
        max_model_len=max_model_len,
        out_clamped_positions=out_clamped,
        out_slot_mapping=out_slot,
        input_batch_size=input_batch_size,
    )

    # Real slots should match the reference
    assert torch.equal(out_clamped, ref_clamped)
    assert torch.equal(out_slot[:batch_size], ref_slot)
    assert torch.equal(seq_lens_copy, ref_seq_lens)

    # Padding slots should be PADDING_SLOT_ID
    for i in range(batch_size, input_batch_size):
        assert out_slot[i].item() == PADDING_SLOT_ID
