# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M-RoPE positions for a P/D decode request whose media stayed on prefill."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.mm.rope import RopeState

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="RopeState needs CUDA"
)

PROMPT_LEN = 10


class FakeMRoPEModel:
    """Text-only positions are the token indices; with media, a fixed layout
    whose delta is -2."""

    def get_mrope_input_positions(self, input_tokens, mm_features):
        positions = torch.arange(len(input_tokens)).repeat(3, 1)
        if not mm_features:
            return positions, 0
        return positions - 2, -2


def _rope_state() -> RopeState:
    return RopeState(
        num_dims=3,
        max_num_reqs=3,
        max_num_tokens=16,
        max_model_len=32,
        device=torch.device("cuda"),
    )


def _positions(state: RopeState, num_computed_tokens: list[int]) -> list[int]:
    """Position of the single token each request computes this step."""
    num_reqs = len(num_computed_tokens)

    def i32(values):
        return torch.tensor(values, dtype=torch.int32, device="cuda")

    state.prepare_positions(
        idx_mapping=i32(list(range(num_reqs))),
        query_start_loc=i32(list(range(num_reqs + 1))),
        prefill_lens=i32([PROMPT_LEN] * num_reqs),
        input_positions=torch.tensor(
            num_computed_tokens, dtype=torch.int64, device="cuda"
        ),
    )
    positions = state.get_positions(num_reqs).cpu()
    # Text positions: every channel holds the same value.
    assert (positions == positions[0]).all()
    return positions[0].tolist()


def test_transferred_delta_shifts_prefill_and_decode_positions():
    state = _rope_state()
    model = FakeMRoPEModel()
    tokens = list(range(PROMPT_LEN))

    # A decode instance that received the prompt without its media.
    state.init_prefill_positions(0, model, tokens, [], position_delta=-4)
    # A text-only prompt.
    state.init_prefill_positions(1, model, tokens, [])
    # The media did arrive: the instance computes positions itself and
    # ignores the reported delta.
    state.init_prefill_positions(2, model, tokens, [object()], position_delta=-4)
    state.apply_staged_writes()

    assert [state.get_prefill_delta(i) for i in range(3)] == [-4, 0, -2]
    shifted = (torch.arange(PROMPT_LEN) - 4).clamp(min=0).repeat(3, 1)
    torch.testing.assert_close(
        state.read_prefill_positions(0, PROMPT_LEN).cpu().long(), shifted
    )

    # The KV for the first PROMPT_LEN - 1 tokens arrived from prefill; each
    # request recomputes its last prompt token.
    assert _positions(state, [PROMPT_LEN - 1] * 3) == [5, 9, 7]
    # First decoded token.
    assert _positions(state, [PROMPT_LEN] * 3) == [6, 10, 8]
