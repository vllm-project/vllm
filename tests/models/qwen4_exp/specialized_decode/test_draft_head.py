# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The exact FP8-prefiltered MTP draft head (`logits/draft_head.py`) against
the stock draft head on GB300: `F.linear` (cuBLAS) and the first index of the
largest BF16 logit, NaN as -inf, padding never chosen.

On a head of the real shape with planted winners, `DraftHead` must return the
stock token for 1 to 8 rows: rows with one clear winner, exact ties (the lower
index wins), near ties that only the BF16 rescoring separates (rows one ulp
apart), a planted winner in the padding, and rows that take the full
rescoring (NaN, an infinity, `max|h|` below `2^-20`) or the zero-row
shortcut; eagerly and under CUDA graph replay with new rows each replay.
Rescoring every entry must reproduce every logit of the stock GEMM bit for bit
at every row count (the premise that cuBLAS adds 16-column MMA steps in k
order), and the FP8 GEMM must stay within the bound's premise on adversarial
code rows. On CPU, the draft passes' `get_top_tokens` takes the head for up
to 8 rows of its own LM head and keeps the stock path otherwise, and a model
of the rescore grid's fold (atomic maxima into the rows' words, arrivals, the
last CTA's swap and reset) over every interleaving of three CTAs and two
launches yields the largest keys and leaves the words reset; it must catch
folding after arriving, a missing reset and an off-by-one last CTA.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.models.qwen4_exp.nvidia.specialized_decode.logits import processor
from vllm.platforms import current_platform

VOCAB = 248320
VALID = 248077  # the rest of the rows is padding
HIDDEN = 2560
DIRECTIONS = 4
WINNER = [1000 + 50000 * k for k in range(DIRECTIONS)]  # planted winners
TIE = 700  # an exact copy of winner 0 at a lower index
NEAR = [WINNER[1] + 3, WINNER[1] - 5]  # winner 1 with one weight one ulp off
PADDED = VALID + 11  # a better copy of winner 2, in the padding

requires_sm103 = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(103),
    reason="the FP8-prefiltered draft head is exact on GB300 (SM103)",
)


def _stock_tokens(weight: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    """The stock draft token of every row."""
    logits = F.linear(hidden, weight).float()
    logits = torch.where(torch.isnan(logits), float("-inf"), logits)
    logits[:, VALID:] = float("-inf")
    return logits.argmax(dim=1)


@pytest.fixture(scope="module")
def head():
    """A random head of the real shape with planted winners, its directions
    and its `DraftHead`."""
    from vllm.models.qwen4_exp.nvidia.specialized_decode.logits.draft_head import (
        DraftHead,
    )

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(0)
    weight = torch.randn(VOCAB, HIDDEN, generator=generator, device=device) * 0.02
    directions = torch.randn(DIRECTIONS, HIDDEN, generator=generator, device=device)
    directions = directions / directions.norm(dim=1, keepdim=True)
    for k, index in enumerate(WINNER):
        weight[index] = directions[k] * 0.5
    weight = weight.to(torch.bfloat16)
    weight[TIE] = weight[WINNER[0]]
    for i, index in enumerate(NEAR):
        weight[index] = weight[WINNER[1]]
        bits = weight[index].view(torch.int16)
        bits[17 + 900 * i] += 1 - 2 * i
    weight[PADDED] = weight[WINNER[2]] * 2
    return weight, directions, DraftHead(weight, VALID)


def _rows(directions: torch.Tensor, seed: int) -> torch.Tensor:
    """16 draft rows: planted directions at several scales, then the special
    rows (zeros, NaN, an infinity, a tiny row)."""
    generator = torch.Generator(device=directions.device).manual_seed(seed)
    noise = torch.randn(16, HIDDEN, generator=generator, device=directions.device)
    rows = []
    for i in range(12):
        scale = (1.0, 7.0, 1e-3, 300.0)[i % 4]
        rows.append(scale * (directions[i % DIRECTIONS] + 0.3 * noise[i] / 50.0))
    hidden = torch.stack(rows + [noise[12]] * 4).to(torch.bfloat16)
    hidden[12] = 0.0
    hidden[13, 5] = float("nan")
    hidden[14, 2000] = float("inf")
    hidden[15] = (noise[15] * 2.0**-30).to(torch.bfloat16)
    return hidden


@requires_sm103
@pytest.mark.parametrize("rows", range(1, 9))
def test_tokens_are_the_stock_tokens(head, rows: int):
    weight, directions, draft = head
    hidden = _rows(directions, seed=rows)
    for first in range(0, 16 - rows + 1, max(1, rows // 2)):
        h = hidden[first : first + rows]
        assert draft.takes(h)
        torch.testing.assert_close(draft(h), _stock_tokens(weight, h), rtol=0, atol=0)


@requires_sm103
def test_ties_and_padding(head):
    """The planted rows' answers: the exact tie goes to the lower index, the
    padding copy never wins, the near ties follow the stock logits."""
    weight, directions, draft = head
    hidden = _rows(directions, seed=1)[:8]
    tokens = draft(hidden)
    assert tokens[0].item() == TIE
    assert tokens[2].item() == WINNER[2]
    assert tokens[1].item() in (WINNER[1], *NEAR)
    torch.testing.assert_close(tokens, _stock_tokens(weight, hidden), rtol=0, atol=0)
    # The prefilter leaves few candidates for planted rows.
    assert int(draft.candidates()[:8].max()) <= 64


@requires_sm103
def test_tokens_under_graph_replay(head):
    """Captured once, replayed with new rows each time: the per-row words the
    kernels accumulate into are reset by every call."""
    weight, directions, draft = head
    for rows in (3, 8):
        static = _rows(directions, seed=100)[:rows].clone()
        draft(static)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            tokens = draft(static)
        for seed in range(101, 107):
            fresh = _rows(directions, seed=seed)
            start = (seed * 5) % (16 - rows + 1)
            static.copy_(fresh[start : start + rows])
            graph.replay()
            torch.testing.assert_close(
                tokens, _stock_tokens(weight, static), rtol=0, atol=0
            )


@requires_sm103
@pytest.mark.parametrize("rows", range(1, 9))
def test_full_rescoring_reproduces_every_stock_logit(head, rows: int):
    """Every BF16 logit of the MMA chain equals cuBLAS's (-0 as +0)."""
    weight, directions, draft = head
    hidden = _rows(directions, seed=200 + rows)[:rows]
    hidden[hidden.isnan() | hidden.isinf()] = 0.25
    hidden[(hidden == 0).all(dim=1)] = 0.5
    tokens, logits = draft.rescore_all(hidden)
    stock = F.linear(hidden, weight)[:, :VALID]
    ours = logits[:, :VALID]
    zero = torch.zeros((), dtype=torch.bfloat16, device=ours.device)
    ours = torch.where(ours == 0, zero, ours).view(torch.int16)
    stock = torch.where(stock == 0, zero, stock).view(torch.int16)
    differ = (ours != stock).sum().item()
    assert differ == 0, f"{differ} of {ours.numel()} logits differ"
    torch.testing.assert_close(tokens, _stock_tokens(weight, hidden), rtol=0, atol=0)


@requires_sm103
@pytest.mark.parametrize("rows", [1, 4, 8])
def test_the_fp8_gemm_keeps_the_bound_premise(rows: int):
    """CuBLASLt's FP8 GEMM at the head's shape, on codes whose products are of
    one sign and span every E4M3 binade (subnormals included), is within
    `8 gamma_K |a| |c|` of the exact sums (module docstring, (2))."""
    from vllm.models.qwen4_exp.nvidia.specialized_decode.logits.draft_head import (
        GAMMA_FP8,
    )

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(rows)

    def codes(count: int) -> torch.Tensor:
        exponent = torch.rand(count, HIDDEN, generator=generator, device=device)
        return torch.exp2(exponent * 17.8 - 9.0).to(torch.float8_e4m3fn)

    a = codes(2 * rows)
    c = codes(VOCAB)
    one = torch.ones((), device=device)
    sums = torch._scaled_mm(a, c.t(), one, one, out_dtype=torch.float32)
    columns = 16384
    exact = a.double() @ c[:columns].double().t()
    norms = a.double().norm(dim=1)[:, None] * c[:columns].double().norm(dim=1)[None, :]
    ratio = ((sums[:, :columns].double() - exact).abs() / norms).max().item()
    assert ratio <= GAMMA_FP8, ratio


class _Head:
    """A stand-in DraftHead that takes up to 8 rows."""

    vocab = 1000
    valid = 990

    def takes(self, hidden: torch.Tensor) -> bool:
        return hidden.shape[0] <= 8

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.full((hidden.shape[0],), 7)


@pytest.mark.parametrize(
    "rows, shard, bias, expected",
    [
        (1, (1000, 990, 0), None, 7),
        (8, (1000, 990, 0), None, 7),
        (9, (1000, 990, 0), None, -1),
        (4, (1000, 989, 0), None, -1),
        (4, (1000, 990, 0), "bias", -1),
    ],
)
def test_get_top_tokens_takes_the_draft_head(monkeypatch, rows, shard, bias, expected):
    """Up to 8 rows of the head's own LM head and shard take it; larger
    passes, another shard or a bias keep the stock path."""
    lm_head = object()
    monkeypatch.setattr(processor, "_DRAFT_HEADS", {lm_head: _Head()})
    monkeypatch.setattr(processor, "_shard", lambda proc, head: shard)
    monkeypatch.setattr(
        LogitsProcessor,
        "get_top_tokens",
        lambda self, head, hidden, bias=None: torch.full((hidden.shape[0],), -1),
    )
    proc = processor.SpecializedDecodeLogitsProcessor.__new__(
        processor.SpecializedDecodeLogitsProcessor
    )
    torch.nn.Module.__init__(proc)
    proc.token_choice = None
    embedding_bias = None if bias is None else torch.zeros(1000)
    tokens = proc.get_top_tokens(lm_head, torch.zeros(rows, 16), embedding_bias)
    assert tokens.tolist() == [expected] * rows


def _interleavings(programs: list[list[str]]):
    """Every interleaving of the CTAs' step lists, as (CTA, step) sequences."""
    if all(not program for program in programs):
        yield []
        return
    for cta, program in enumerate(programs):
        if program:
            rest = programs[:cta] + [program[1:]] + programs[cta + 1 :]
            for tail in _interleavings(rest):
                yield [(cta, program[0])] + tail


def _rescore_launches(keys, publish_first=True, last_rule=None, reset_words=True):
    """The rescore grid's fold over consecutive launches (one per entry of
    `keys`: per CTA, per row, the CTA's best key or 0) over every
    interleaving of the CTAs' steps in each launch. A CTA folds its keys into
    the rows' words (atomic max), then arrives; the one that sees `ctas - 1`
    arrivals before it swaps every row's word for 0 into the tokens and resets
    the counter. A launch starts after the previous one completed. Returns,
    per launch, the set of token tuples the interleavings produced, and
    whether every launch left the words and the counter at 0."""
    ctas, rows = len(keys[0]), len(keys[0][0])
    last_rule = last_rule or (lambda old: old == ctas - 1)
    clean = True
    results = []
    words, arrivals = [0] * rows, 0
    for launch in keys:
        seen = set()
        start = (list(words), arrivals)
        steps = [f"max{r}" for r in range(rows)] + ["arrive"]
        if not publish_first:
            steps = steps[-1:] + steps[:-1]
        for order in _interleavings([list(steps) for _ in range(ctas)]):
            words, arrivals = list(start[0]), start[1]
            tokens = None
            for cta, step in order:
                if step == "arrive":
                    old, arrivals = arrivals, arrivals + 1
                    if last_rule(old):
                        tokens = tuple(words)
                        if reset_words:
                            words = [0] * rows
                        arrivals = 0
                else:
                    r = int(step[3:])
                    words[r] = max(words[r], launch[cta][r])
            seen.add(tokens)
            clean &= words == [0] * rows and arrivals == 0
        results.append(seen)
    return results, clean


def test_rescore_fold_and_reset_over_every_interleaving():
    """Every interleaving of three CTAs gives each launch the rows' largest
    keys and leaves the words and the counter reset for the next launch. The
    model catches folding after arriving, a missing reset of the words and an
    off-by-one last CTA."""
    keys = [
        [[5, 0], [9, 2], [0, 7]],  # tokens (9, 7)
        [[1, 3], [0, 0], [2, 1]],  # smaller keys: a stale word would win
    ]
    expected = [{(9, 7)}, {(2, 3)}]
    assert _rescore_launches(keys) == (expected, True)
    late, _ = _rescore_launches(keys, publish_first=False)
    assert late != expected
    stale, clean = _rescore_launches(keys, reset_words=False)
    assert stale != expected and not clean
    off, clean = _rescore_launches(keys, last_rule=lambda old: old == len(keys[0]))
    assert off[0] == {None} and off != expected and not clean
