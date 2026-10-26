# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The drafter's step-word kernels, per request of a tier."""

import pytest
import torch

from vllm.models.glm53.nvidia import draft as glm53_draft

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

T = 4
WIDTH = 1024


def _table(live: int) -> torch.Tensor:
    return torch.tensor([T * live, live, 0, 0], dtype=torch.int32, device="cuda")


def _words(requests: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    return (
        torch.zeros(2 * T * requests, dtype=torch.int32, device=device),
        torch.zeros(requests, dtype=torch.int32, device=device),
        torch.ones(2 * requests, dtype=torch.int32, device=device),
    )


@pytest.mark.parametrize(("requests", "live"), [(1, 1), (4, 3), (12, 12)])
def test_a_captured_catchup_counts_a_generation_and_sets_the_live_words(requests, live):
    device = torch.device("cuda")
    positions = torch.zeros(T * requests, dtype=torch.int64, device=device)
    positions[::T] = 100 * torch.arange(1, requests + 1, device=device)
    accepts = [(b + 1) % T for b in range(requests)]
    last = torch.tensor(
        [T * b + accepts[b] for b in range(requests)], dtype=torch.int64, device=device
    )
    position, accepted, context = _words(requests)
    for _ in range(3):
        glm53_draft._catchup_words_kernel[(requests,)](
            positions, last, _table(live), position, accepted, context, T=T
        )
    for b in range(requests):
        if b < live:
            assert position[2 * T * b].item() == 100 * (b + 1)
            assert accepted[b].item() == accepts[b]
            assert context[2 * b : 2 * b + 2].tolist() == [
                100 * (b + 1) + accepts[b] + 1,
                4,
            ]
        else:
            assert position[2 * T * b].item() == 0 and accepted[b].item() == 0
            assert context[2 * b : 2 * b + 2].tolist() == [1, 1]
    # The positions are an int64-style array: only each first row's low word is set.
    assert position.view(-1, 2 * T)[:, 1:].count_nonzero().item() == 0


@pytest.mark.parametrize(
    ("starts", "lasts"),
    [
        ([0], [6]),  # one request: a chunk of seven rows
        ([0, 5], [4, 10]),  # two requests of five and six rows
        ([0, 5, 7], [4, 6, 8]),  # two requests of two rows and one of five
    ],
)
def test_an_eager_catchup_gathers_each_requests_last_rows(starts, lasts):
    live = len(starts)
    requests = 4
    device = torch.device("cuda")
    tokens = max(lasts) + 1
    idle = [0] * (requests - live)
    last = torch.tensor(lasts + idle, dtype=torch.int64, device=device)
    query_start_loc = torch.tensor(
        starts + [tokens] * (requests + 1 - live), dtype=torch.int32, device=device
    )
    positions = torch.arange(tokens, dtype=torch.int64, device=device) + 50
    token_ids = torch.arange(tokens, dtype=torch.int32, device=device)
    rows = torch.randn(tokens, WIDTH, device=device).to(torch.bfloat16)
    rows_scratch = torch.zeros(T * requests, WIDTH, dtype=torch.bfloat16, device=device)
    token_scratch = torch.zeros(T * requests, dtype=torch.int32, device=device)
    position, accepted, context = _words(requests)
    glm53_draft._gather_inputs_kernel[(WIDTH // 1024, requests)](
        last,
        query_start_loc,
        positions,
        token_ids,
        rows,
        token_scratch,
        rows_scratch,
        position,
        accepted,
        context,
        _table(live),
        tokens,
        T=T,
        WIDTH=WIDTH,
        BLOCK=1024,
    )
    for b in range(live):
        a = min(lasts[b] - starts[b], T - 1)
        first = lasts[b] - a
        assert accepted[b].item() == a
        assert position[2 * T * b].item() == 50 + first
        assert context[2 * b : 2 * b + 2].tolist() == [50 + first + a + 1, 2]
        for i in range(a + 1):
            assert torch.equal(rows_scratch[T * b + i], rows[first + i])
            assert token_scratch[T * b + i].item() == first + i
    for b in range(live, requests):
        assert context[2 * b : 2 * b + 2].tolist() == [1, 1]


def test_the_proposal_inputs_of_the_live_requests_are_copied():
    requests, live = 4, 3
    device = torch.device("cuda")
    positions = torch.arange(requests, dtype=torch.int64, device=device) + 70
    input_ids = torch.arange(requests, dtype=torch.int32, device=device) + 5
    hidden = torch.randn(requests, WIDTH, device=device).to(torch.bfloat16)
    position = torch.zeros(2 * requests, dtype=torch.int32, device=device)
    token = torch.zeros(requests, dtype=torch.int32, device=device)
    row = torch.zeros(requests, WIDTH, dtype=torch.bfloat16, device=device)
    glm53_draft._proposal_inputs_kernel[(WIDTH // 1024, requests)](
        positions,
        input_ids,
        hidden,
        _table(live),
        position,
        token,
        row,
        WIDTH=WIDTH,
        BLOCK=1024,
    )
    assert position.view(-1, 2)[:, 0].tolist() == [70, 71, 72, 0]
    assert token.tolist() == [5, 6, 7, 0]
    assert torch.equal(row[:live], hidden[:live]) and not row[live:].any()


BLOCK_SIZE = 64


def _stage(drafts: int, window: int, positions: list[int], accepts: list[int]):
    """Stage one window of ``len(positions)`` live requests of a tier of four; return
    the staged tensors, vLLM's source rows and the block table before the step.
    """
    device = torch.device("cuda")
    requests, live, rows_per = 4, len(positions), drafts + 1
    tokens = torch.arange(rows_per * requests, dtype=torch.int32, device=device) + 9
    rows = torch.randn(rows_per * requests, WIDTH, device=device).to(torch.bfloat16)
    blocks = torch.arange(1, 1 + requests * 64, dtype=torch.int32, device=device)
    tensors = {
        glm53_draft.BLOCK_TABLE: blocks.view(requests, 64),
        "draft.t.position": torch.zeros(
            2 * rows_per * requests, dtype=torch.int32, device=device
        ),
        "draft.t.accept": torch.zeros(requests, dtype=torch.int32, device=device),
        "token_ids": torch.zeros(4 * requests, dtype=torch.int32, device=device),
        "draft.rows": torch.zeros(
            4 * requests, WIDTH, dtype=torch.bfloat16, device=device
        ),
        "position": torch.zeros(8 * requests, dtype=torch.int32, device=device),
        "mtp.accept": torch.zeros(requests, dtype=torch.int32, device=device),
        "draft.context": torch.ones(2 * requests, dtype=torch.int32, device=device),
    }
    for b, (p, a) in enumerate(zip(positions, accepts)):
        tensors["draft.t.position"][2 * rows_per * b] = p
        tensors["draft.t.accept"][b] = a
    before = blocks.clone()
    table = torch.tensor([4 * live, live, 0, 0], dtype=torch.int32, device=device)
    glm53_draft.stage_window(
        tensors, tokens, rows, table, requests, drafts, window, BLOCK_SIZE
    )
    return tensors, tokens, rows, before.view(requests, 64)


@pytest.mark.parametrize("drafts", [1, 2])
def test_a_short_step_pads_its_window_with_its_last_row(drafts):
    rows_per = drafts + 1
    positions, accepts = [100, 200, 300], [0, drafts, 1 % rows_per]
    tensors, tokens, rows, _ = _stage(drafts, 1, positions, accepts)
    for b, (p, a) in enumerate(zip(positions, accepts)):
        for i in range(4):
            source = rows_per * b + min(i, rows_per - 1)
            assert torch.equal(tensors["draft.rows"][4 * b + i], rows[source])
            assert tensors["token_ids"][4 * b + i].item() == tokens[source].item()
        assert tensors["position"][8 * b].item() == p
        assert tensors["mtp.accept"][b].item() == a
        assert tensors["draft.context"][2 * b : 2 * b + 2].tolist() == [p + a + 1, 2]
    assert not tensors["draft.rows"][12:].any()  # the idle request
    assert tensors["draft.context"][6:].tolist() == [1, 1]


@pytest.mark.parametrize(
    ("drafts", "position", "nulled"),
    [
        (1, 61, 1),  # p + 3 = 64 starts block 1, past the reservation p..p + 2
        (1, 60, None),  # p + 3 = 63: the reserved block
        (1, 62, None),  # p + 2 = 64 is reserved: block 1 is the request's
        (2, 61, None),  # p..p + 4 reserved: every dead row is
    ],
)
def test_a_dead_row_past_the_reservation_gets_the_null_block(drafts, position, nulled):
    tensors, _, _, before = _stage(drafts, 1, [position, position], [0, 0])
    after = tensors[glm53_draft.BLOCK_TABLE]
    expected = before.clone()
    if nulled is not None:
        expected[:2, nulled] = 0  # both live requests, not the idle ones
    assert torch.equal(after, expected)


@pytest.mark.parametrize("window", [0, 1])
def test_two_windows_take_four_true_rows_and_null_no_block(window):
    drafts, position = 4, 61
    tensors, tokens, rows, before = _stage(drafts, window, [position], [1])
    # Accepted row 1 lies in the low window: pass 0 copies rows 1..4, pass 1 rows 0..3.
    start = 1 if window == 0 else 0
    for i in range(4):
        assert torch.equal(tensors["draft.rows"][i], rows[start + i])
        assert tensors["token_ids"][i].item() == tokens[start + i].item()
    assert tensors["position"][0].item() == position + start
    assert tensors["mtp.accept"][0].item() == (0 if window == 0 else 1)
    assert torch.equal(tensors[glm53_draft.BLOCK_TABLE], before)
