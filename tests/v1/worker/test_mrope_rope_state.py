# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Text-only M-RoPE positions computed by the Model Runner V2 position kernel.

For models that set `supports_linear_text_mrope`, RopeState skips staging
per-request prefill positions for text-only requests; the kernel output must
match the staged path.
"""

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.models.interfaces import SupportsMRoPE
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.mm.rope import RopeState

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires CUDA or ROCm"
)

NUM_DIMS = 3
IMAGE = MultiModalFeatureSpec(
    data=None,
    modality="image",
    identifier="image",
    mm_position=PlaceholderRange(offset=0, length=1),
)


class _MRoPEModel(nn.Module, SupportsMRoPE):
    """Sequential positions for text-only inputs; with multimodal features,
    distinct per-channel positions and a nonzero delta."""

    supports_linear_text_mrope = True
    MM_DELTA = -7

    def __init__(self):
        super().__init__()
        self.num_calls = 0

    def get_mrope_input_positions(self, input_tokens, mm_features):
        self.num_calls += 1
        positions = torch.arange(len(input_tokens))
        if not mm_features:
            return positions.unsqueeze(0).expand(NUM_DIMS, -1), 0
        return (
            torch.stack([positions // (d + 1) for d in range(NUM_DIMS)]),
            self.MM_DELTA,
        )


def _rope_state(**mode) -> RopeState:
    return RopeState(
        num_dims=NUM_DIMS,
        max_num_reqs=4,
        max_num_tokens=4096,
        max_model_len=4096,
        device=torch.device("cuda"),
        **mode,
    )


def test_linear_mode_skips_host_positions_and_rejects_multimodal_inputs():
    model = _MRoPEModel()
    state = _rope_state(linear_prefill_positions=True)
    assert state.prefill_positions is None

    state.init_prefill_positions(0, model, [1, 2, 3], mm_features=[])
    state.apply_staged_writes()
    assert model.num_calls == 0

    with pytest.raises(RuntimeError, match="multimodal"):
        state.init_prefill_positions(0, model, [1, 2, 3], mm_features=[IMAGE])


def _batch_positions(
    state: RopeState,
    model: _MRoPEModel,
    prompt_lens: list[int],
    has_mm: list[bool],
) -> list[torch.Tensor]:
    """Positions for chunked prefill (including a chunk longer than the
    kernel block), the prefill tail, and multi-token decode steps, with the
    batch order differing from the request-slot order."""
    device = torch.device("cuda")
    for req_idx, (length, mm) in enumerate(zip(prompt_lens, has_mm)):
        mm_features = [IMAGE] if mm else []
        state.init_prefill_positions(req_idx, model, [7] * length, mm_features)
    state.apply_staged_writes()

    num_reqs = len(prompt_lens)
    order = list(reversed(range(num_reqs)))
    idx_mapping = torch.tensor(order, dtype=torch.int32, device=device)
    prefill_lens = torch.zeros(state.max_num_reqs, dtype=torch.int32, device=device)
    prefill_lens[:num_reqs] = torch.tensor(prompt_lens, dtype=torch.int32)
    outs = []
    for step, (frac, max_query_len) in enumerate(
        [(0.0, 1200), (0.99, 200), (1.0, 4), (1.0, 4)]
    ):
        computed = [min(int(n * frac), n) + step for n in prompt_lens]
        query_lens = [
            max_query_len
            if computed[i] >= prompt_lens[i]
            else min(max_query_len, prompt_lens[i] - computed[i])
            for i in order
        ]
        num_computed = torch.zeros(state.max_num_reqs, dtype=torch.int32, device=device)
        num_computed[:num_reqs] = torch.tensor(computed, dtype=torch.int32)
        query_start_loc = torch.tensor(
            [0] + query_lens, dtype=torch.int32, device=device
        ).cumsum(0, dtype=torch.int32)
        state.prepare_positions(
            idx_mapping, query_start_loc, prefill_lens, num_computed
        )
        outs.append(state.get_positions(int(query_start_loc[-1])).clone())
    return outs


def test_linear_positions_match_staged_positions():
    prompt_lens = [100, 1500, 9, 777]
    has_mm = [False] * len(prompt_lens)
    staged = _batch_positions(_rope_state(), _MRoPEModel(), prompt_lens, has_mm)
    model = _MRoPEModel()
    state = _rope_state(linear_prefill_positions=True)
    computed = _batch_positions(state, model, prompt_lens, has_mm)
    assert model.num_calls == 0
    for ref, out in zip(staged, computed):
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
