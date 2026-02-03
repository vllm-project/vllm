# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for PatternForcedSequenceLogitsProcessor."""

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    MoveDirectionality,
)
from vllm.v1.sample.logits_processor.pattern_forced_sequence import (
    ForcingState,
    PatternForcedSequenceLogitsProcessor,
)

VOCAB_SIZE = 256
DEVICE = "cpu"
TRIGGER_PATTERN = [100, 101, 102, 103]
FORCED_SEQUENCE = [200, 201, 202]


def _make_processor() -> PatternForcedSequenceLogitsProcessor:
    return PatternForcedSequenceLogitsProcessor(None, torch.device(DEVICE), False)


def _make_params(
    trigger: list[int] | None = None,
    forced: list[int] | None = None,
) -> SamplingParams:
    return SamplingParams(
        extra_args={
            "harmony_tool_required": {
                "trigger_pattern": trigger or TRIGGER_PATTERN,
                "forced_sequence": forced or FORCED_SEQUENCE,
            }
        }
    )


def _make_batch_update(
    added: list[tuple[int, SamplingParams, list[int] | None, list[int]]],
    removed: list[int] | None = None,
    moved: list[tuple[int, int, MoveDirectionality]] | None = None,
    batch_size: int | None = None,
) -> BatchUpdate:
    if batch_size is None:
        batch_size = max((idx for idx, *_ in added), default=-1) + 1
    return BatchUpdate(
        batch_size=batch_size,
        removed=removed or [],
        added=added,
        moved=moved or [],
    )


def _make_logits(batch_size: int) -> torch.Tensor:
    return torch.ones(batch_size, VOCAB_SIZE, dtype=torch.float32)


class TestAddRequest:
    """Tests for add_request static method."""

    def test_returns_state_with_valid_config(self) -> None:
        output_ids = [1, 2, 3]
        params = _make_params()
        state = PatternForcedSequenceLogitsProcessor.add_request(
            params, None, output_ids
        )
        assert state is not None
        assert state.state == ForcingState.NORMAL
        assert state.forcing_pos == 0
        assert state.output_ids is output_ids
        assert state.trigger_pattern == TRIGGER_PATTERN
        assert state.forced_sequence == FORCED_SEQUENCE

    def test_returns_none_without_extra_args(self) -> None:
        params = SamplingParams()
        state = PatternForcedSequenceLogitsProcessor.add_request(params, None, [])
        assert state is None

    def test_returns_none_with_empty_extra_args(self) -> None:
        params = SamplingParams(extra_args={})
        state = PatternForcedSequenceLogitsProcessor.add_request(params, None, [])
        assert state is None

    def test_returns_none_with_non_dict_config(self) -> None:
        params = SamplingParams(extra_args={"harmony_tool_required": "invalid"})
        state = PatternForcedSequenceLogitsProcessor.add_request(params, None, [])
        assert state is None

    def test_returns_none_with_missing_trigger(self) -> None:
        params = SamplingParams(
            extra_args={
                "harmony_tool_required": {
                    "forced_sequence": FORCED_SEQUENCE,
                }
            }
        )
        state = PatternForcedSequenceLogitsProcessor.add_request(params, None, [])
        assert state is None

    def test_returns_none_with_missing_forced(self) -> None:
        params = SamplingParams(
            extra_args={
                "harmony_tool_required": {
                    "trigger_pattern": TRIGGER_PATTERN,
                }
            }
        )
        state = PatternForcedSequenceLogitsProcessor.add_request(params, None, [])
        assert state is None


class TestApply:
    """Tests for apply method: trigger detection and forced sequence."""

    def test_empty_req_states_returns_logits_unchanged(self) -> None:
        proc = _make_processor()
        logits = _make_logits(2)
        original = logits.clone()
        result = proc.apply(logits)
        assert torch.equal(result, original)

    def test_trigger_detection_forces_first_token(self) -> None:
        proc = _make_processor()
        output_ids: list[int] = [10, 11] + TRIGGER_PATTERN
        params = _make_params()
        batch_update = _make_batch_update(
            added=[(0, params, None, output_ids)],
            batch_size=1,
        )
        proc.update_state(batch_update)

        logits = _make_logits(1)
        result = proc.apply(logits)

        # Only the first forced token should have non-neg-inf logit
        allowed_token = FORCED_SEQUENCE[0]
        for tok in range(VOCAB_SIZE):
            if tok == allowed_token:
                assert result[0, tok] == 1.0
            else:
                assert result[0, tok] == float("-inf")

    def test_forced_sequence_progresses_through_all_tokens(self) -> None:
        proc = _make_processor()
        output_ids: list[int] = list(TRIGGER_PATTERN)
        params = _make_params()
        batch_update = _make_batch_update(
            added=[(0, params, None, output_ids)],
            batch_size=1,
        )
        proc.update_state(batch_update)

        # Step through each forced token
        for i, expected_token in enumerate(FORCED_SEQUENCE):
            logits = _make_logits(1)
            result = proc.apply(logits)

            for tok in range(VOCAB_SIZE):
                if tok == expected_token:
                    assert result[0, tok] == 1.0
                else:
                    assert result[0, tok] == float("-inf"), (
                        f"Step {i}: token {tok} should be -inf"
                    )

            # Simulate appending the forced token to output_ids
            output_ids.append(expected_token)

    def test_returns_to_normal_after_forced_sequence_complete(self) -> None:
        proc = _make_processor()
        output_ids: list[int] = list(TRIGGER_PATTERN)
        params = _make_params()
        batch_update = _make_batch_update(
            added=[(0, params, None, output_ids)],
            batch_size=1,
        )
        proc.update_state(batch_update)

        # Walk through entire forced sequence
        for token in FORCED_SEQUENCE:
            proc.apply(_make_logits(1))
            output_ids.append(token)

        # Now the state should be NORMAL; logits should be unchanged
        logits = _make_logits(1)
        original = logits.clone()
        result = proc.apply(logits)
        assert torch.equal(result, original)

    def test_incomplete_trigger_does_not_activate(self) -> None:
        proc = _make_processor()
        # Only partial trigger pattern
        output_ids: list[int] = TRIGGER_PATTERN[:-1]
        params = _make_params()
        batch_update = _make_batch_update(
            added=[(0, params, None, output_ids)],
            batch_size=1,
        )
        proc.update_state(batch_update)

        logits = _make_logits(1)
        original = logits.clone()
        result = proc.apply(logits)
        assert torch.equal(result, original)

    def test_no_config_request_logits_unchanged(self) -> None:
        proc = _make_processor()
        # Request without harmony config should not be tracked
        params = SamplingParams()
        output_ids: list[int] = list(TRIGGER_PATTERN)
        batch_update = _make_batch_update(
            added=[(0, params, None, output_ids)],
            batch_size=1,
        )
        proc.update_state(batch_update)
        assert len(proc.req_states) == 0

        logits = _make_logits(1)
        original = logits.clone()
        result = proc.apply(logits)
        assert torch.equal(result, original)

    def test_trigger_fires_again_after_completion(self) -> None:
        """Trigger should re-activate each time the pattern appears."""
        proc = _make_processor()
        output_ids: list[int] = list(TRIGGER_PATTERN)
        params = _make_params()
        batch_update = _make_batch_update(
            added=[(0, params, None, output_ids)],
            batch_size=1,
        )
        proc.update_state(batch_update)

        # Complete first forced sequence
        for token in FORCED_SEQUENCE:
            proc.apply(_make_logits(1))
            output_ids.append(token)

        # Consume the "return to normal" step
        proc.apply(_make_logits(1))

        # Add trigger pattern again
        output_ids.extend(TRIGGER_PATTERN)

        # Should activate forcing again
        logits = _make_logits(1)
        result = proc.apply(logits)
        allowed_token = FORCED_SEQUENCE[0]
        assert result[0, allowed_token] == 1.0
        assert result[0, (allowed_token + 1) % VOCAB_SIZE] == float("-inf")


class TestStateManagement:
    """Tests for request add/remove state management."""

    def test_add_and_remove_requests(self) -> None:
        proc = _make_processor()
        params = _make_params()
        output_ids_0: list[int] = [1, 2]
        output_ids_1: list[int] = [3, 4]

        # Add two requests
        batch_update = _make_batch_update(
            added=[
                (0, params, None, output_ids_0),
                (1, params, None, output_ids_1),
            ],
            batch_size=2,
        )
        proc.update_state(batch_update)
        assert len(proc.req_states) == 2

        # Remove request 0
        batch_update = BatchUpdate(
            batch_size=1,
            removed=[0],
            added=[],
            moved=[],
        )
        proc.update_state(batch_update)
        assert 0 not in proc.req_states
        assert 1 in proc.req_states

    def test_move_request(self) -> None:
        proc = _make_processor()
        params = _make_params()
        output_ids: list[int] = [1, 2]

        batch_update = _make_batch_update(
            added=[(0, params, None, output_ids)],
            batch_size=1,
        )
        proc.update_state(batch_update)
        assert 0 in proc.req_states

        # Move request from index 0 to index 5
        batch_update = BatchUpdate(
            batch_size=6,
            removed=[],
            added=[],
            moved=[(0, 5, MoveDirectionality.UNIDIRECTIONAL)],
        )
        proc.update_state(batch_update)
        assert 0 not in proc.req_states
        assert 5 in proc.req_states

    def test_mixed_batch_only_tracks_configured_requests(self) -> None:
        proc = _make_processor()
        params_with = _make_params()
        params_without = SamplingParams()

        batch_update = _make_batch_update(
            added=[
                (0, params_with, None, [1, 2]),
                (1, params_without, None, [3, 4]),
                (2, params_with, None, [5, 6]),
            ],
            batch_size=3,
        )
        proc.update_state(batch_update)
        assert len(proc.req_states) == 2
        assert 0 in proc.req_states
        assert 1 not in proc.req_states
        assert 2 in proc.req_states
