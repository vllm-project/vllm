# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ToolChoiceRequiredLogitsProcessor.

This processor suppresses stop/EOS tokens from the start of generation until
the model produces a release token (e.g. <|tool_call_end|>), forcing thinking
models to generate tool calls when tool_choice=required.
"""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.builtin import (
    ToolChoiceRequiredLogitsProcessor,
)
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
)

pytestmark = pytest.mark.cpu_test

DEVICE = torch.device("cpu")
VOCAB_SIZE = 256

# Token IDs for testing
THINK_END = 100  # </think>
IM_END = 101  # <|im_end|>
EOS_TOKEN = 102  # [EOS]
TOOL_CALL_END = 103  # <|tool_call_end|>


def _make_vllm_config():
    config = MagicMock()
    config.scheduler_config.max_num_seqs = 32
    return config


def _make_params(*, extra_args=None, eos_token_id=None, stop_token_ids=None):
    kwargs = {}
    if extra_args is not None:
        kwargs["extra_args"] = extra_args
    if stop_token_ids is not None:
        kwargs["stop_token_ids"] = stop_token_ids
    params = SamplingParams(**kwargs)
    if eos_token_id is not None:
        params._eos_token_id = eos_token_id
    return params


def _make_batch_update(added=None, removed=None, moved=None, batch_size=4):
    update = BatchUpdate(
        batch_size=batch_size,
        removed=removed or [],
        added=added or [],
        moved=moved or [],
    )
    return update


class TestAddRequest:
    def test_returns_none_without_extra_args(self):
        """Normal requests without extra_args → processor not activated."""
        params = _make_params()
        result = ToolChoiceRequiredLogitsProcessor.add_request(params, None, [])
        assert result is None

    def test_returns_none_with_partial_extra_args(self):
        """Only think_end without stop → not activated."""
        params = _make_params(
            extra_args={
                "tool_choice_required_think_end": THINK_END,
            }
        )
        result = ToolChoiceRequiredLogitsProcessor.add_request(params, None, [])
        assert result is None

    def test_collects_stop_token(self):
        """Basic activation with stop token."""
        params = _make_params(
            extra_args={
                "tool_choice_required_think_end": THINK_END,
                "tool_choice_required_stop": IM_END,
            }
        )
        stop_toks, release, out_ids = ToolChoiceRequiredLogitsProcessor.add_request(
            params, None, []
        )
        assert IM_END in stop_toks
        assert release == -1  # No section_end → never releases

    def test_collects_eos_and_stop_token_ids(self):
        """Collects eos_token_id + stop_token_ids in addition to the main stop."""
        params = _make_params(
            extra_args={
                "tool_choice_required_think_end": THINK_END,
                "tool_choice_required_stop": IM_END,
            },
            eos_token_id=EOS_TOKEN,
            stop_token_ids=[200, 201],
        )
        stop_toks, release, out_ids = ToolChoiceRequiredLogitsProcessor.add_request(
            params, None, []
        )
        assert IM_END in stop_toks
        assert EOS_TOKEN in stop_toks
        assert 200 in stop_toks
        assert 201 in stop_toks
        assert len(stop_toks) == 4

    def test_no_duplicate_when_eos_equals_stop(self):
        """When eos_token_id == stop, it should not be duplicated."""
        params = _make_params(
            extra_args={
                "tool_choice_required_think_end": THINK_END,
                "tool_choice_required_stop": IM_END,
            },
            eos_token_id=IM_END,  # Same as stop
        )
        stop_toks, _, _ = ToolChoiceRequiredLogitsProcessor.add_request(
            params, None, []
        )
        assert stop_toks.count(IM_END) == 1

    def test_section_end_sets_release_token(self):
        """section_end sets the release token."""
        params = _make_params(
            extra_args={
                "tool_choice_required_think_end": THINK_END,
                "tool_choice_required_stop": IM_END,
                "tool_choice_required_section_end": TOOL_CALL_END,
            }
        )
        _, release, _ = ToolChoiceRequiredLogitsProcessor.add_request(params, None, [])
        assert release == TOOL_CALL_END


class TestUpdateStateAndApply:
    def _make_processor(self):
        return ToolChoiceRequiredLogitsProcessor(_make_vllm_config(), DEVICE, False)

    def test_no_op_when_inactive(self):
        """When no requests have extra_args, apply is a no-op."""
        proc = self._make_processor()
        logits = torch.randn(4, VOCAB_SIZE)
        original = logits.clone()

        params = _make_params()  # No extra_args
        update = _make_batch_update(
            added=[(0, params, None, [])],
        )
        proc.update_state(update)
        result = proc.apply(logits)

        assert torch.equal(result, original)

    def test_suppresses_stop_tokens(self):
        """Active request → stop tokens set to -inf."""
        proc = self._make_processor()
        logits = torch.zeros(4, VOCAB_SIZE)

        out_ids: list[int] = []  # Empty — no tokens generated yet
        params = _make_params(
            extra_args={
                "tool_choice_required_think_end": THINK_END,
                "tool_choice_required_stop": IM_END,
                "tool_choice_required_section_end": TOOL_CALL_END,
            },
            eos_token_id=EOS_TOKEN,
        )
        update = _make_batch_update(
            added=[(0, params, None, out_ids)],
        )
        proc.update_state(update)
        result = proc.apply(logits)

        # Stop tokens should be -inf for request 0
        assert result[0, IM_END].item() == float("-inf")
        assert result[0, EOS_TOKEN].item() == float("-inf")
        # Other tokens unaffected
        assert result[0, 50].item() == 0.0
        # Other requests unaffected
        assert result[1, IM_END].item() == 0.0

    def test_release_after_tool_call_end(self):
        """After release token appears in output_tok_ids, suppression lifts."""
        proc = self._make_processor()

        out_ids: list[int] = []
        params = _make_params(
            extra_args={
                "tool_choice_required_think_end": THINK_END,
                "tool_choice_required_stop": IM_END,
                "tool_choice_required_section_end": TOOL_CALL_END,
            },
        )
        update = _make_batch_update(added=[(0, params, None, out_ids)])
        proc.update_state(update)

        # Initially suppressing
        logits = torch.zeros(4, VOCAB_SIZE)
        proc.apply(logits)
        assert logits[0, IM_END].item() == float("-inf")

        # Simulate model generating the release token
        out_ids.append(TOOL_CALL_END)

        # Update state — should detect release
        proc.update_state(_make_batch_update())
        logits = torch.zeros(4, VOCAB_SIZE)
        proc.apply(logits)

        # Suppression lifted — stop tokens no longer -inf
        assert logits[0, IM_END].item() == 0.0

    def test_no_release_without_section_end(self):
        """When section_end not set (release=-1), suppression never lifts."""
        proc = self._make_processor()

        out_ids = [50, 60, 70]  # Some tokens but not the release token
        params = _make_params(
            extra_args={
                "tool_choice_required_think_end": THINK_END,
                "tool_choice_required_stop": IM_END,
                # No section_end
            },
        )
        update = _make_batch_update(added=[(0, params, None, out_ids)])
        proc.update_state(update)

        logits = torch.zeros(4, VOCAB_SIZE)
        proc.apply(logits)
        assert logits[0, IM_END].item() == float("-inf")

    def test_multiple_requests(self):
        """Multiple active requests each get their stop tokens suppressed."""
        proc = self._make_processor()

        out_ids_0: list[int] = []
        params_0 = _make_params(
            extra_args={
                "tool_choice_required_think_end": THINK_END,
                "tool_choice_required_stop": IM_END,
            },
        )
        out_ids_1: list[int] = []
        params_1 = _make_params(
            extra_args={
                "tool_choice_required_think_end": THINK_END,
                "tool_choice_required_stop": 150,  # Different stop token
            },
        )
        update = _make_batch_update(
            added=[
                (0, params_0, None, out_ids_0),
                (2, params_1, None, out_ids_1),
            ]
        )
        proc.update_state(update)

        logits = torch.zeros(4, VOCAB_SIZE)
        proc.apply(logits)

        assert logits[0, IM_END].item() == float("-inf")
        assert logits[2, 150].item() == float("-inf")
        # Request 1 unaffected
        assert logits[1, IM_END].item() == 0.0

    def test_removed_request(self):
        """Removed request no longer suppressed."""
        proc = self._make_processor()

        out_ids: list[int] = []
        params = _make_params(
            extra_args={
                "tool_choice_required_think_end": THINK_END,
                "tool_choice_required_stop": IM_END,
            },
        )
        proc.update_state(_make_batch_update(added=[(0, params, None, out_ids)]))

        # Verify active
        logits = torch.zeros(4, VOCAB_SIZE)
        proc.apply(logits)
        assert logits[0, IM_END].item() == float("-inf")

        # Remove
        proc.update_state(_make_batch_update(removed=[0]))

        logits = torch.zeros(4, VOCAB_SIZE)
        proc.apply(logits)
        assert logits[0, IM_END].item() == 0.0
