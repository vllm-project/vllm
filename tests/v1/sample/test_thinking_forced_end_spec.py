# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V1 thinking-budget forced end under speculative decoding.

The harness drives ``ThinkingBudgetStateHolder`` the way V1 calls it
(``update_state``, the bonus-row call in ``Sampler``, then the target-row call in
``RejectionSampler``) and verifies drafts greedily, so no GPU is needed.
"""

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.interface import BatchUpdate
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder

THINK_START = 100
# A 6-token forced end sequence: a transition phrase plus the parser's marker.
END = [40, 41, 42, 43, 44, 200]
OFF_SCRIPT = 77  # the target's own preference where it disagrees with the phrase
FILLER = 5
VOCAB = 256
NUM_SPEC = 3


class _Config:
    reasoning_start_token_ids = [THINK_START]
    reasoning_end_token_ids = END


def _start(budget: int) -> tuple[ThinkingBudgetStateHolder, list[int]]:
    holder = ThinkingBudgetStateHolder(
        _Config(), 8, NUM_SPEC, torch.device("cpu"), False
    )
    output: list[int] = []
    params = SamplingParams(thinking_token_budget=budget)
    holder.sync_batch(
        BatchUpdate(
            batch_size=1, removed=(), added=[(0, params, None, output)], moved=()
        )
    )
    output.append(THINK_START)
    return holder, output


def _phrase_progress(tokens: list[int]) -> int:
    """END tokens emitted since the first END[0], if the tail is exactly them."""
    if END[0] not in tokens:
        return -1
    tail = tokens[tokens.index(END[0]) :]
    n = 0
    while n < len(tail) and n < len(END) and tail[n] == END[n]:
        n += 1
    return n if n == len(tail) else -2


def _natural(breaks_at: int):
    """Target argmax: FILLER before the end sequence starts, then the phrase,
    except right after END[:breaks_at] where it prefers OFF_SCRIPT."""

    def natural(prefix: list[int]) -> int:
        n = _phrase_progress(prefix)
        if n == breaks_at:
            return OFF_SCRIPT
        if 0 <= n < len(END):
            return END[n]
        return FILLER

    return natural


def _phrase_drafter(output: list[int]) -> list[int]:
    """Proposes the next END tokens once the end sequence has started."""
    n = _phrase_progress(output)
    if n < 0:
        return [9] * NUM_SPEC
    nxt = END[n : n + NUM_SPEC]
    return nxt + [9] * (NUM_SPEC - len(nxt))


def _step(holder, output, drafts, natural):
    """One V1 verification step, greedy."""
    holder.update_state([output], [drafts])
    bonus = torch.zeros(1, VOCAB)
    bonus[0, natural(output + drafts)] = 1.0
    bonus = holder.apply_to_logits(bonus, True, [drafts])
    target = torch.zeros(len(drafts), VOCAB)
    for j in range(len(drafts)):
        target[j, natural(output + drafts[:j])] = 1.0
    target = holder.apply_to_logits(target, False, [drafts])
    for j, d in enumerate(drafts):
        t = int(torch.argmax(target[j]))
        output.append(t)
        if t != d:
            return
    output.append(int(torch.argmax(bonus[0])))


def _run_until_closed(holder, output, drafter, natural, max_steps=40):
    for _ in range(max_steps):
        if END[0] in output and END[-1] in output[output.index(END[0]) :]:
            return
        _step(holder, output, drafter(output), natural)
    raise AssertionError(f"end sequence never closed: {output}")


def _assert_exact_end(output):
    start = output.index(END[0])
    emitted = output[start : start + len(END)]
    assert emitted == END, f"forced end sequence corrupted: {output[start:]}"
    assert output[start:].count(END[0]) == 1, f"end sequence restarted: {output}"


def test_forced_end_survives_target_rejecting_a_matching_draft():
    holder, output = _start(budget=12)
    _run_until_closed(holder, output, _phrase_drafter, _natural(3))
    _assert_exact_end(output)


def test_forcing_from_inside_the_draft_window_covers_the_rest():
    """The budget runs out one position into the window while the drafter
    already proposes the phrase."""
    holder, output = _start(budget=10)

    def drafter(out):
        if END[0] in out:
            return _phrase_drafter(out)
        return [FILLER, END[0], END[1]] if len(out) - 1 >= 9 else [FILLER] * 2 + [9]

    _run_until_closed(holder, output, drafter, _natural(1))
    _assert_exact_end(output)


def test_forced_end_exact_when_no_draft_matches():
    """Control: no draft ever matches, so one end token is forced per step."""
    holder, output = _start(budget=12)
    _run_until_closed(holder, output, lambda o: [9] * NUM_SPEC, _natural(3))
    _assert_exact_end(output)
