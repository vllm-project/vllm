# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.models.qwen3_ksa.common.cache import build_ksa_summary_slot_mapping


def test_compressed_summary_slots_use_physical_pages() -> None:
    # Page-size unification expands a 16-token Summary page to 128 tokens
    # while preserving one physical state per eight logical tokens.
    token_positions = torch.tensor([6, 7, 15, 127, 128, 135])
    boundary = (token_positions + 1).remainder(8) == 0
    slots = build_ksa_summary_slot_mapping(
        token_positions=token_positions,
        token_to_request=torch.zeros(6, dtype=torch.int32),
        boundary_mask=boundary,
        block_table=torch.tensor([[3, 5]], dtype=torch.int32),
        manager_block_size=128,
        states_per_block=16,
        summary_chunk_size=8,
    )

    torch.testing.assert_close(slots, torch.tensor([-1, 48, 49, 63, -1, 80]))
