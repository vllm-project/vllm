# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for M-RoPE speculative decoding slot mapping and position updates."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.step3p5 import Step3p5MTPProposer
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID

DEVICE_TYPE = current_platform.device_type

# Skip if no CUDA - Triton kernel requires GPU
pytest.importorskip("triton")
if not current_platform.is_cuda_alike() and not current_platform.is_xpu():
    pytest.skip("CUDA/XPU required for EAGLE kernel tests", allow_module_level=True)


def _make_common_attn_metadata(
    block_table_tensor: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
) -> CommonAttentionMetadata:
    meta = CommonAttentionMetadata.__new__(CommonAttentionMetadata)
    meta.block_table_tensor = block_table_tensor
    meta.seq_lens = seq_lens
    meta.max_seq_len = max_seq_len
    meta.slot_mapping = None
    meta._seq_lens_cpu = seq_lens.cpu().clone()
    meta._num_computed_tokens_cpu = seq_lens.cpu().clone()
    meta.seq_lens_cpu_upper_bound = max_seq_len
    return meta


def _create_mock_mrope_proposer(
    batch_size: int,
    max_model_len: int = 4096,
    device: str = DEVICE_TYPE,
) -> SpecDecodeBaseProposer:
    """Create a mock proposer with M-RoPE enabled."""
    proposer = SpecDecodeBaseProposer.__new__(SpecDecodeBaseProposer)
    proposer.device = torch.device(device)
    proposer.max_positions = max_model_len
    proposer.max_model_len = max_model_len
    proposer.uses_mrope = True
    proposer.uses_xdrope_dim = 0
    proposer.draft_uses_xdrope_dim = 0
    proposer.mrope_positions = torch.zeros(
        (3, proposer.max_positions + 1),
        dtype=torch.int64,
        device=proposer.device,
    )
    proposer._slot_positions = torch.zeros(
        proposer.max_positions,
        dtype=torch.int64,
        device=proposer.device,
    )
    proposer._slot_mapping_buffer = torch.zeros(
        proposer.max_positions,
        dtype=torch.int64,
        device=proposer.device,
    )
    return proposer


def test_mrope_slot_mapping_uses_seq_lens():
    """Verify that M-RoPE slot mapping derives from sequence lengths,
    not 3D temporal coordinates, and preserves spatial dimensions."""
    device = torch.device(DEVICE_TYPE)
    batch_size = 2
    block_size = 16
    max_model_len = 4096
    n_blocks_per_req = (max_model_len + block_size - 1) // block_size

    proposer = _create_mock_mrope_proposer(batch_size, max_model_len)

    # Initial 3D M-RoPE positions (temporal, height, width)
    # Request 0: 100 text + 256 image + 9 text = 365 tokens.
    # Temporal=110, Height=115, Width=120.
    # Request 1: 200 text + 390 image + 10 text = 600 tokens.
    # Temporal=210, Height=225, Width=235.
    initial_positions = torch.tensor(
        [
            [110, 210],  # Temporal
            [115, 225],  # Height
            [120, 235],  # Width
        ],
        dtype=torch.int64,
        device=device,
    )

    # Sequence lengths before drafting the next token
    seq_lens = torch.tensor([365, 600], dtype=torch.int32, device=device)

    # Construct block table with distinct block IDs
    block_table = torch.arange(
        batch_size * n_blocks_per_req, dtype=torch.int32, device=device
    ).reshape(batch_size, n_blocks_per_req)

    common_metadata = _make_common_attn_metadata(
        block_table_tensor=block_table,
        seq_lens=seq_lens,
        max_seq_len=600,
    )

    # Execute update
    updated_positions = proposer._update_positions_dependent_metadata(
        positions=initial_positions,
        common_attn_metadata=common_metadata,
        batch_size=batch_size,
        input_batch_size=batch_size,
        block_size=block_size,
    )

    # Expected slot mapping:
    # Req 0: seq_len=365 -> block_num=365 // 16 = 22, offset=365 % 16 = 13
    # Expected slot = block_table[0, 22] * 16 + 13
    # Req 1: seq_len=600 -> block_num=600 // 16 = 37, offset=600 % 16 = 8
    # Expected slot = block_table[1, 37] * 16 + 8
    expected_slot_0 = block_table[0, 365 // block_size].item() * block_size + (
        365 % block_size
    )
    expected_slot_1 = block_table[1, 600 // block_size].item() * block_size + (
        600 % block_size
    )

    actual_slots = common_metadata.slot_mapping
    assert actual_slots is not None
    assert actual_slots[0].item() == expected_slot_0
    assert actual_slots[1].item() == expected_slot_1

    # Verify seq_lens incremented in place
    assert common_metadata.seq_lens[0].item() == 366
    assert common_metadata.seq_lens[1].item() == 601

    # Verify 3D M-RoPE positions incremented all dimensions independently
    expected_positions = torch.tensor(
        [
            [111, 211],  # Temporal
            [116, 226],  # Height
            [121, 236],  # Width
        ],
        dtype=torch.int64,
        device=device,
    )
    assert torch.equal(updated_positions, expected_positions)
    assert torch.equal(proposer.mrope_positions[:, :batch_size], expected_positions)


def test_mrope_multi_step_draft_loop():
    """Verify multi-step speculative draft loop for M-RoPE.

    Ensures slot mapping and 3D positions advance across consecutive steps.
    """
    device = torch.device(DEVICE_TYPE)
    batch_size = 2
    block_size = 16
    max_model_len = 4096
    n_blocks_per_req = (max_model_len + block_size - 1) // block_size
    num_draft_steps = 5

    proposer = _create_mock_mrope_proposer(batch_size, max_model_len)

    current_positions = torch.tensor(
        [
            [100, 250],  # Temporal
            [105, 260],  # Height
            [110, 270],  # Width
        ],
        dtype=torch.int64,
        device=device,
    )
    initial_seq_lens = [320, 512]
    seq_lens = torch.tensor(initial_seq_lens, dtype=torch.int32, device=device)

    block_table = torch.arange(
        batch_size * n_blocks_per_req, dtype=torch.int32, device=device
    ).reshape(batch_size, n_blocks_per_req)

    common_metadata = _make_common_attn_metadata(
        block_table_tensor=block_table,
        seq_lens=seq_lens,
        max_seq_len=512,
    )

    for step in range(num_draft_steps):
        expected_seq_len_req0 = initial_seq_lens[0] + step
        expected_seq_len_req1 = initial_seq_lens[1] + step

        # Verify expected slot before step
        expected_slot_0 = block_table[
            0, expected_seq_len_req0 // block_size
        ].item() * block_size + (expected_seq_len_req0 % block_size)
        expected_slot_1 = block_table[
            1, expected_seq_len_req1 // block_size
        ].item() * block_size + (expected_seq_len_req1 % block_size)

        current_positions = proposer._update_positions_dependent_metadata(
            positions=current_positions,
            common_attn_metadata=common_metadata,
            batch_size=batch_size,
            input_batch_size=batch_size,
            block_size=block_size,
        )

        # Assert slot mapping is correct for this step
        assert common_metadata.slot_mapping[0].item() == expected_slot_0
        assert common_metadata.slot_mapping[1].item() == expected_slot_1

        # Assert seq_lens incremented
        assert common_metadata.seq_lens[0].item() == expected_seq_len_req0 + 1
        assert common_metadata.seq_lens[1].item() == expected_seq_len_req1 + 1

        # Assert 3D positions advanced without collapsing spatial coordinates
        expected_positions_step = torch.tensor(
            [
                [100 + step + 1, 250 + step + 1],
                [105 + step + 1, 260 + step + 1],
                [110 + step + 1, 270 + step + 1],
            ],
            dtype=torch.int64,
            device=device,
        )
        assert torch.equal(current_positions, expected_positions_step)


def test_mrope_mixed_batch_with_cudagraph_padding():
    """Verify mixed batch with cudagraph padding."""
    device = torch.device(DEVICE_TYPE)
    batch_size = 3
    input_batch_size = 8  # cudagraph padding slots
    block_size = 16
    max_model_len = 4096
    n_blocks_per_req = (max_model_len + block_size - 1) // block_size

    proposer = _create_mock_mrope_proposer(input_batch_size, max_model_len)

    # Req 0: Text-only (temporal=height=width=49)
    # Req 1: Single image (temporal=60, height=75, width=80, seq_len=180)
    # Req 2: Multi image (temporal=120, height=180, width=210, seq_len=450)
    positions = torch.tensor(
        [
            [49, 60, 120],
            [49, 75, 180],
            [49, 80, 210],
        ],
        dtype=torch.int64,
        device=device,
    )
    seq_lens = torch.tensor([50, 180, 450], dtype=torch.int32, device=device)

    block_table = torch.arange(
        batch_size * n_blocks_per_req, dtype=torch.int32, device=device
    ).reshape(batch_size, n_blocks_per_req)

    common_metadata = _make_common_attn_metadata(
        block_table_tensor=block_table,
        seq_lens=seq_lens,
        max_seq_len=450,
    )

    updated_positions = proposer._update_positions_dependent_metadata(
        positions=positions,
        common_attn_metadata=common_metadata,
        batch_size=batch_size,
        input_batch_size=input_batch_size,
        block_size=block_size,
    )

    # Verify real requests
    assert common_metadata.slot_mapping[0].item() == (
        block_table[0, 50 // block_size].item() * block_size + (50 % block_size)
    )
    assert common_metadata.slot_mapping[1].item() == (
        block_table[1, 180 // block_size].item() * block_size + (180 % block_size)
    )
    assert common_metadata.slot_mapping[2].item() == (
        block_table[2, 450 // block_size].item() * block_size + (450 % block_size)
    )

    # Verify padding slots wrote PADDING_SLOT_ID
    for pad_idx in range(batch_size, input_batch_size):
        assert proposer._slot_mapping_buffer[pad_idx].item() == PADDING_SLOT_ID

    # Verify positions
    expected_positions = torch.tensor(
        [
            [50, 61, 121],
            [50, 76, 181],
            [50, 81, 211],
        ],
        dtype=torch.int64,
        device=device,
    )
    assert torch.equal(updated_positions, expected_positions)


def test_mrope_step3p5_per_group_slot_mapping():
    """Verify Step3p5MTPProposer multi-group slot mapping for M-RoPE models."""
    device = torch.device(DEVICE_TYPE)
    batch_size = 2
    block_size = 16
    max_model_len = 4096
    n_blocks_per_req = (max_model_len + block_size - 1) // block_size

    # Create mock Step3p5 proposer
    proposer = Step3p5MTPProposer.__new__(Step3p5MTPProposer)
    proposer.device = torch.device(device)
    proposer.max_positions = max_model_len
    proposer.max_model_len = max_model_len
    proposer.uses_mrope = True
    proposer.uses_xdrope_dim = 0
    proposer.draft_uses_xdrope_dim = 0
    proposer.mrope_positions = torch.zeros(
        (3, proposer.max_positions + 1), dtype=torch.int64, device=proposer.device
    )
    proposer._slot_positions = torch.zeros(
        proposer.max_positions, dtype=torch.int64, device=proposer.device
    )
    proposer._slot_mapping_buffer = torch.zeros(
        proposer.max_positions, dtype=torch.int64, device=proposer.device
    )

    # Multi-group setup: primary group (gid=0) and auxiliary draft group (gid=1)
    proposer.kv_cache_gid = 0
    aux_group = SimpleNamespace(kv_cache_group_id=1, layer_names=["draft_layer_0"])
    proposer.draft_attn_groups = [aux_group]

    # Distinct block tables for each group
    primary_block_table = torch.arange(
        batch_size * n_blocks_per_req, dtype=torch.int32, device=device
    ).reshape(batch_size, n_blocks_per_req)
    aux_block_table = (
        torch.arange(
            batch_size * n_blocks_per_req, dtype=torch.int32, device=device
        ).reshape(batch_size, n_blocks_per_req)
        + 1000
    )

    proposer._per_group_block_tables = {1: aux_block_table}
    proposer._per_group_slot_mappings = {}
    proposer._per_group_slot_mapping_buffers = {
        1: torch.zeros(proposer.max_positions, dtype=torch.int64, device=device)
    }

    initial_positions = torch.tensor(
        [
            [110, 210],
            [115, 225],
            [120, 235],
        ],
        dtype=torch.int64,
        device=device,
    )
    seq_lens = torch.tensor([365, 600], dtype=torch.int32, device=device)

    common_metadata = _make_common_attn_metadata(
        block_table_tensor=primary_block_table,
        seq_lens=seq_lens,
        max_seq_len=600,
    )

    # Execute step3p5 update
    updated_positions = proposer._update_positions_dependent_metadata(
        positions=initial_positions,
        common_attn_metadata=common_metadata,
        batch_size=batch_size,
        input_batch_size=batch_size,
        block_size=block_size,
    )

    expected_positions = torch.tensor(
        [
            [111, 211],
            [116, 226],
            [121, 236],
        ],
        dtype=torch.int64,
        device=device,
    )
    assert torch.equal(updated_positions, expected_positions)

    # Verify primary group slot mapping (gid=0)
    expected_primary_0 = primary_block_table[
        0, 365 // block_size
    ].item() * block_size + (365 % block_size)
    expected_primary_1 = primary_block_table[
        1, 600 // block_size
    ].item() * block_size + (600 % block_size)
    assert proposer._per_group_slot_mappings[0][0].item() == expected_primary_0
    assert proposer._per_group_slot_mappings[0][1].item() == expected_primary_1

    # Verify auxiliary draft group slot mapping (gid=1)
    expected_aux_0 = aux_block_table[0, 365 // block_size].item() * block_size + (
        365 % block_size
    )
    expected_aux_1 = aux_block_table[1, 600 // block_size].item() * block_size + (
        600 % block_size
    )
    assert proposer._per_group_slot_mappings[1][0].item() == expected_aux_0
    assert proposer._per_group_slot_mappings[1][1].item() == expected_aux_1


def test_mrope_slot_mapping_exceeds_max_model_len():
    """Verify that M-RoPE positions clamp correctly when exceeding max_model_len."""
    device = torch.device(DEVICE_TYPE)
    batch_size = 2
    block_size = 16
    max_model_len = 100
    n_blocks_per_req = (max_model_len + block_size - 1) // block_size

    proposer = _create_mock_mrope_proposer(batch_size, max_model_len)

    # Request 0 is within bounds, Request 1 reaches max_model_len - 1
    initial_positions = torch.tensor(
        [
            [50, 99],
            [55, 99],
            [60, 99],
        ],
        dtype=torch.int64,
        device=device,
    )
    seq_lens = torch.tensor([51, 99], dtype=torch.int32, device=device)
    block_table = torch.arange(
        batch_size * n_blocks_per_req, dtype=torch.int32, device=device
    ).reshape(batch_size, n_blocks_per_req)

    common_metadata = _make_common_attn_metadata(
        block_table_tensor=block_table,
        seq_lens=seq_lens,
        max_seq_len=max_model_len,
    )

    updated_positions = proposer._update_positions_dependent_metadata(
        positions=initial_positions,
        common_attn_metadata=common_metadata,
        batch_size=batch_size,
        input_batch_size=batch_size,
        block_size=block_size,
    )

    # Request 0 should increment normally
    assert updated_positions[0, 0].item() == 51
    assert updated_positions[1, 0].item() == 56
    assert updated_positions[2, 0].item() == 61

    # Request 1 positions (99 + 1 = 100 >= max_model_len) should clamp to 0
    assert updated_positions[0, 1].item() == 0
    assert updated_positions[1, 1].item() == 0
    assert updated_positions[2, 1].item() == 0
