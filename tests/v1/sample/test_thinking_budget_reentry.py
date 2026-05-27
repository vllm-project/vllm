"""Tests for ThinkingBudgetStateHolder re-entry after forced end.

Regression test for https://github.com/vllm-project/vllm/issues/43708:
After the budget enforcer forces the complete end-of-thinking token sequence,
the state machine must detect and enforce budget on subsequent thinking blocks.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock

import torch

from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder

THINK_START = 100
THINK_END_MULTI = [200, 201, 202]
THINK_END_SINGLE = [200]
BUDGET = 5
CONTENT_TOKEN = 50
THINK_TOKEN = 60


@dataclass
class FakeReasoningConfig:
    reasoning_start_token_ids: list[int]
    reasoning_end_token_ids: list[int]
    enabled: bool = True


def _make_holder(end_token_ids: list[int]) -> ThinkingBudgetStateHolder:
    cfg = FakeReasoningConfig(
        reasoning_start_token_ids=[THINK_START],
        reasoning_end_token_ids=end_token_ids,
    )
    return ThinkingBudgetStateHolder(
        reasoning_config=cfg,
        max_num_seqs=8,
        num_spec_tokens=0,
        device=torch.device("cpu"),
        is_pin_memory=False,
    )


def _make_batch_update(index, prompt_tok_ids, budget):
    params = MagicMock()
    params.thinking_token_budget = budget
    return MagicMock(
        removed=[],
        added=[(index, params, prompt_tok_ids, [])],
        moved=[],
    )


def _step(holder, output_tok_ids):
    """Run one update_state + apply_to_logits cycle."""
    holder.update_state(
        output_token_ids=[output_tok_ids],
        spec_token_ids=None,
        repeat_indices=None,
    )
    logits = torch.zeros(1, 300)
    return holder.apply_to_logits(logits, predict_bonus_token=False, spec_token_ids=[])


class TestThinkingBudgetReentry:
    """After budget forces end tokens, a second <think> block must also be
    budget-enforced."""

    def test_single_token_end_reentry(self):
        """With a single-token end string, verify second think block is caught.

        Regression test for issue #43708.
        """
        holder = _make_holder(THINK_END_SINGLE)
        batch_update = _make_batch_update(0, None, BUDGET)
        holder.sync_batch(batch_update)

        output = []

        # First thinking block: <think> + BUDGET tokens
        output.append(THINK_START)
        _step(holder, list(output))
        for _ in range(BUDGET):
            output.append(THINK_TOKEN)
            _step(holder, list(output))

        state = holder._state[0]
        assert state["in_end"], "Budget should trigger end forcing"

        # Force the single end token
        output.append(THINK_END_SINGLE[0])
        _step(holder, list(output))

        state = holder._state[0]
        assert not state["in_end"]
        assert state["start_thinking"] == -1, "start_thinking must reset"
        assert state["end_thinking"] == -1, "end_thinking must reset"

        # Generate some content tokens
        for _ in range(3):
            output.append(CONTENT_TOKEN)
            _step(holder, list(output))

        # Second thinking block: <think> + BUDGET tokens
        output.append(THINK_START)
        _step(holder, list(output))
        for _ in range(BUDGET):
            output.append(THINK_TOKEN)
            _step(holder, list(output))

        state = holder._state[0]
        assert state["in_end"], "Second thinking block must also be budget-enforced"

    def test_multi_token_end_reentry(self):
        """With a multi-token end string (like custom reasoning_end_str),
        verify re-entry detection after full sequence is forced.

        Regression test for issue #43708.
        """
        holder = _make_holder(THINK_END_MULTI)
        batch_update = _make_batch_update(0, None, BUDGET)
        holder.sync_batch(batch_update)

        output = []

        # First thinking block
        output.append(THINK_START)
        _step(holder, list(output))
        for _ in range(BUDGET):
            output.append(THINK_TOKEN)
            _step(holder, list(output))

        state = holder._state[0]
        assert state["in_end"]

        # Force all tokens of the multi-token end sequence
        for tok in THINK_END_MULTI:
            output.append(tok)
            _step(holder, list(output))

        state = holder._state[0]
        assert not state["in_end"]
        assert state["end_count"] == 0
        assert state["start_thinking"] == -1
        assert state["end_thinking"] == -1
        assert state["think_count"] == 0

        # Second thinking block (no content in between — immediate re-entry)
        output.append(THINK_START)
        _step(holder, list(output))
        for _ in range(BUDGET):
            output.append(THINK_TOKEN)
            _step(holder, list(output))

        state = holder._state[0]
        assert state["in_end"], (
            "Immediate re-entry after multi-token end must be enforced"
        )

    def test_single_block_still_works(self):
        """Regression: single-block usage (the common case) is not broken."""
        holder = _make_holder(THINK_END_SINGLE)
        batch_update = _make_batch_update(0, None, BUDGET)
        holder.sync_batch(batch_update)

        output = []

        output.append(THINK_START)
        _step(holder, list(output))
        for _ in range(BUDGET):
            output.append(THINK_TOKEN)
            _step(holder, list(output))

        state = holder._state[0]
        assert state["in_end"]

        output.append(THINK_END_SINGLE[0])
        _step(holder, list(output))

        assert not holder._state[0]["in_end"]

        # Content generation should NOT trigger enforcement
        for _ in range(20):
            output.append(CONTENT_TOKEN)
            _step(holder, list(output))

        state = holder._state[0]
        assert not state["in_end"], "Content generation should not trigger end"
        assert not state["in_think"], "Content should not be seen as thinking"

    def test_logits_forcing_on_second_block(self):
        """Verify that apply_to_logits forces the correct token on the
        second thinking block."""
        holder = _make_holder(THINK_END_SINGLE)
        batch_update = _make_batch_update(0, None, BUDGET)
        holder.sync_batch(batch_update)

        output = []

        # First block: exhaust budget and force end
        output.append(THINK_START)
        _step(holder, list(output))
        for _ in range(BUDGET):
            output.append(THINK_TOKEN)
            _step(holder, list(output))
        output.append(THINK_END_SINGLE[0])
        _step(holder, list(output))

        # Content gap
        for _ in range(5):
            output.append(CONTENT_TOKEN)
            _step(holder, list(output))

        # Second block: exhaust budget
        output.append(THINK_START)
        _step(holder, list(output))
        for _ in range(BUDGET):
            output.append(THINK_TOKEN)
            _step(holder, list(output))

        # apply_to_logits should force the end token
        logits = torch.zeros(1, 300)
        result = holder.apply_to_logits(
            logits, predict_bonus_token=False, spec_token_ids=[]
        )
        assert result[0, THINK_END_SINGLE[0]].item() > 1e8, (
            "End token must be forced in logits on second block"
        )
