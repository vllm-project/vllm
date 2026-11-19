# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.utils import gather_rejected_tokens

WIDTH = 3
NUM_LAYERS = 2
HIDDEN = 4


def _aux(num_tokens: int) -> list[torch.Tensor]:
    # Every (token row, layer) holds a distinct value.
    return [
        (torch.arange(num_tokens, dtype=torch.float32)[:, None] * 10 + layer)
        .expand(num_tokens, HIDDEN)
        .contiguous()
        for layer in range(NUM_LAYERS)
    ]


def _batch(order: str):
    """Two decode requests, 3 drafts each: state 5 (anchor 40) accepts 1,
    state 2 (anchor 70) accepts 2."""
    reqs = {
        5: dict(positions=[40, 41, 42, 43], num_rejected=2),
        2: dict(positions=[70, 71, 72, 73], num_rejected=1),
    }
    states = [5, 2] if order == "AB" else [2, 5]
    positions, qsl, num_rejected = [], [0], []
    for s in states:
        positions += reqs[s]["positions"]
        qsl.append(len(positions))
        num_rejected.append(reqs[s]["num_rejected"])
    return dict(
        query_start_loc=torch.tensor(qsl, dtype=torch.int32),
        num_rejected=torch.tensor(num_rejected, dtype=torch.int32),
        positions=torch.tensor(positions, dtype=torch.int64),
        idx_mapping=torch.tensor(states, dtype=torch.int32),
        num_reqs=2,
        width=WIDTH,
    )


def test_rejected_suffix_rows_and_positions():
    # Row 0: decode, 2 rejected. Row 1: prefill, none. Row 2: all accepted.
    # Row 3: graph padding (beyond num_reqs).
    out = gather_rejected_tokens(
        query_start_loc=torch.tensor([0, 4, 9, 13, 13], dtype=torch.int32),
        num_rejected=torch.tensor([2, 0, 0], dtype=torch.int32),
        positions=torch.arange(100, 113, dtype=torch.int64),
        idx_mapping=torch.tensor([7, 3, 1], dtype=torch.int32),
        num_reqs=3,
        width=WIDTH,
    )
    assert out.rows.tolist() == [[2, 3, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    assert out.positions.tolist() == [
        [102, 103, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    assert out.valid.tolist() == [
        [True, True, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
    ]
    assert out.num_rejected.tolist() == [2, 0, 0, 0]
    assert out.req_state_idx.tolist() == [7, 3, 1, -1]
    assert out.features is None


def test_features_are_the_raw_aux_rows():
    out = gather_rejected_tokens(**_batch("AB"), aux_hidden_states=_aux(8))
    assert out.features.shape == (2, WIDTH, NUM_LAYERS, HIDDEN)
    # AB: state 5 rejects packed rows 2, 3; state 2 rejects packed row 7.
    for layer in range(NUM_LAYERS):
        assert out.features[0, 0, layer].tolist() == [20.0 + layer] * HIDDEN
        assert out.features[0, 1, layer].tolist() == [30.0 + layer] * HIDDEN
        assert out.features[1, 0, layer].tolist() == [70.0 + layer] * HIDDEN
    assert not out.features[0, 2].any()
    assert not out.features[1, 1:].any()


def test_swapped_batch_order_keeps_each_request():
    ab = gather_rejected_tokens(**_batch("AB"), aux_hidden_states=_aux(8))
    ba = gather_rejected_tokens(**_batch("BA"), aux_hidden_states=_aux(8))
    assert ab.req_state_idx.tolist() == [5, 2]
    assert ba.req_state_idx.tolist() == [2, 5]
    for state, (i, j) in {5: (0, 1), 2: (1, 0)}.items():
        assert ab.req_state_idx[i] == ba.req_state_idx[j] == state
        assert ab.num_rejected[i] == ba.num_rejected[j]
        assert torch.equal(ab.positions[i], ba.positions[j])
        assert torch.equal(ab.valid[i], ba.valid[j])
    assert ab.positions[0].tolist() == [42, 43, 0]
    assert ab.positions[1].tolist() == [73, 0, 0]
    # BA packs state 2 first: its rejected row moves from 7 to 3.
    assert ba.rows.tolist() == [[3, 0, 0], [6, 7, 0]]
    assert ba.features[1, 0, 1].tolist() == [61.0] * HIDDEN
    assert ba.features[0, 0, 0].tolist() == [30.0] * HIDDEN


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="requires CUDA")
def test_boundary_from_sampler_counts():
    from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected

    device = torch.device(current_platform.device_type)
    # Row 0: 4 logits, 2 sampled -> 2 rejected. Row 1: chunked prefill (seq_len
    # 5 < prefill_len 9) -> 0 rejected. Row 2: 4 logits, 4 sampled -> 0.
    num_sampled, num_rejected = get_num_sampled_and_rejected(
        num_sampled=torch.tensor([2, 1, 4], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([44, 5, 74], dtype=torch.int32, device=device),
        cu_num_logits=torch.tensor([0, 4, 5, 9], dtype=torch.int32, device=device),
        idx_mapping=torch.tensor([0, 1, 2], dtype=torch.int32, device=device),
        prefill_len=torch.tensor([10, 9, 10], dtype=torch.int32, device=device),
    )
    assert num_rejected.tolist() == [2, 0, 0]
    out = gather_rejected_tokens(
        query_start_loc=torch.tensor([0, 4, 9, 13], dtype=torch.int32, device=device),
        num_rejected=num_rejected,
        positions=torch.arange(100, 113, dtype=torch.int64, device=device),
        idx_mapping=torch.tensor([0, 1, 2], dtype=torch.int32, device=device),
        num_reqs=3,
        width=WIDTH,
        aux_hidden_states=[h.to(device) for h in _aux(13)],
    )
    assert out.positions.tolist() == [[102, 103, 0], [0, 0, 0], [0, 0, 0]]
    assert out.features[0, :2, 0, 0].tolist() == [20.0, 30.0]
    assert not out.features[1:].any()
