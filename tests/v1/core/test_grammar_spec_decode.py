# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-level tests for grammar speculative decoding (jump-forward).

With speculative method "grammar", the scheduler computes grammar-forced
fast-forward tokens when processing model output and proposes them as draft
tokens for the next step, where the target model verifies them. Drafts are
padded to num_speculative_tokens with -1 so decode batches stay uniform.
"""

from unittest.mock import Mock

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

NUM_SPEC_TOKENS = 4
PAD = -1


@pytest.fixture(autouse=True)
def _use_v2_model_runner(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")


def _create_grammar_scheduler(**kwargs):
    return create_scheduler(
        num_speculative_tokens=NUM_SPEC_TOKENS,
        speculative_method="grammar",
        **kwargs,
    )


def _make_ff_grammar(ff_tokens: list[int]):
    """Mock grammar that forces the given fast-forward continuation."""
    grammar = Mock()
    grammar.accept_tokens = Mock(return_value=True)
    grammar.is_terminated = Mock(return_value=False)
    grammar.compute_ff_tokens = Mock(side_effect=lambda: list(ff_tokens))
    grammar.validate_tokens = Mock(side_effect=lambda tokens: list(tokens))
    return grammar


def _make_running_request(scheduler, ff_tokens: list[int], max_tokens: int = 32):
    req = create_requests(num_requests=1, max_tokens=max_tokens)[0]
    req.num_computed_tokens = req.num_tokens
    req.status = RequestStatus.RUNNING
    req.structured_output_request = Mock(
        grammar=_make_ff_grammar(ff_tokens), reasoning_ended=True
    )
    scheduler.requests[req.request_id] = req
    scheduler.running.append(req)
    return req


def _make_step_io(req, sampled_token_id: int = 7):
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {req.request_id: 1}
    scheduler_output.total_num_scheduled_tokens = 1
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id],
        req_id_to_index={req.request_id: 0},
        sampled_token_ids=[[sampled_token_id]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    return scheduler_output, model_output


def test_ff_tokens_become_spec_tokens():
    """Forced tokens are proposed as padded draft tokens for the next step."""
    scheduler = _create_grammar_scheduler()
    scheduler.structured_output_manager.should_advance = Mock(return_value=True)
    req = _make_running_request(scheduler, ff_tokens=[100, 101, 102])

    scheduler_output, model_output = _make_step_io(req)
    scheduler.update_from_output(scheduler_output, model_output)

    # The sampled token is the request's output; the forced continuation
    # becomes the draft proposal (nothing is force-injected).
    assert list(req.output_token_ids)[-1] == 7
    assert req.spec_token_ids == [100, 101, 102, PAD]


def test_ff_tokens_capped_at_num_spec_tokens():
    scheduler = _create_grammar_scheduler()
    scheduler.structured_output_manager.should_advance = Mock(return_value=True)
    ff_tokens = list(range(100, 110))
    req = _make_running_request(scheduler, ff_tokens=ff_tokens)

    scheduler.update_from_output(*_make_step_io(req))

    assert req.spec_token_ids == ff_tokens[:NUM_SPEC_TOKENS]


def test_ff_tokens_skip_in_flight_real_drafts():
    """With async scheduling, the in-flight step consumes its bonus token
    plus its real (non-pad) drafts, so the drafts must start after them."""
    scheduler = _create_grammar_scheduler()
    scheduler.structured_output_manager.should_advance = Mock(return_value=True)
    ff_tokens = list(range(100, 110))
    req = _make_running_request(scheduler, ff_tokens=ff_tokens)
    req.num_output_placeholders = 1 + NUM_SPEC_TOKENS
    scheduler._inflight_grammar_drafts[req.request_id] = 2

    scheduler.update_from_output(*_make_step_io(req))

    # skip = 1 bonus + 2 real in-flight drafts.
    assert req.spec_token_ids == ff_tokens[3 : 3 + NUM_SPEC_TOKENS]


def test_all_pads_when_ff_consumed_by_in_flight_steps():
    scheduler = _create_grammar_scheduler()
    scheduler.structured_output_manager.should_advance = Mock(return_value=True)
    req = _make_running_request(scheduler, ff_tokens=[100])
    req.num_output_placeholders = 1 + NUM_SPEC_TOKENS
    scheduler._inflight_grammar_drafts[req.request_id] = 2

    scheduler.update_from_output(*_make_step_io(req))

    assert req.spec_token_ids == [PAD] * NUM_SPEC_TOKENS


def test_no_drafts_for_non_structured_request():
    scheduler = _create_grammar_scheduler()
    scheduler.structured_output_manager.should_advance = Mock(return_value=False)
    req = _make_running_request(scheduler, ff_tokens=[100, 101, 102])

    scheduler.update_from_output(*_make_step_io(req))

    assert req.spec_token_ids == []


def test_spec_tokens_are_scheduled():
    """Draft tokens set from ff tokens flow into the next SchedulerOutput,
    and the scheduler tracks the in-flight real-draft count."""
    scheduler = _create_grammar_scheduler()
    scheduler.structured_output_manager.should_advance = Mock(return_value=True)

    req = create_requests(num_requests=1, max_tokens=32)[0]
    req.structured_output_request = Mock(
        grammar=_make_ff_grammar([100, 101, 102]), reasoning_ended=True
    )
    scheduler.add_request(req)
    prefill_output = scheduler.schedule()
    model_output = ModelRunnerOutput(
        req_ids=[req.request_id],
        req_id_to_index={req.request_id: 0},
        sampled_token_ids=[[7]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(prefill_output, model_output)
    assert req.spec_token_ids == [100, 101, 102, PAD]

    decode_output = scheduler.schedule()
    assert decode_output.scheduled_spec_decode_tokens == {
        req.request_id: [100, 101, 102, PAD]
    }
    # One sampled token plus the padded drafts to verify.
    assert decode_output.num_scheduled_tokens[req.request_id] == 1 + NUM_SPEC_TOKENS
    # Only the real drafts count toward the next step's skip.
    assert scheduler._inflight_grammar_drafts[req.request_id] == 3
    # Consumed: not re-proposed until the next ff computation.
    assert req.spec_token_ids == []


def test_revalidate_trims_stale_drafts():
    """Scheduled drafts invalidated by in-flight rejections are trimmed and
    padded with -1 before the grammar bitmask walk."""
    scheduler = _create_grammar_scheduler()
    scheduler.structured_output_manager.should_advance = Mock(return_value=True)
    req = _make_running_request(scheduler, ff_tokens=[])
    grammar = req.structured_output_request.grammar
    grammar.validate_tokens = Mock(side_effect=lambda tokens: tokens[:1])

    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {req.request_id: 4}
    scheduler_output.scheduled_spec_decode_tokens = {
        req.request_id: [100, 101, 102, PAD]
    }

    scheduler._revalidate_grammar_spec_tokens(scheduler_output, [req.request_id])

    # Only the real prefix is validated; the result keeps the padded length.
    assert scheduler_output.scheduled_spec_decode_tokens == {
        req.request_id: [100, PAD, PAD, PAD]
    }
    assert scheduler_output.num_invalid_spec_tokens == {req.request_id: 2}
    grammar.validate_tokens.assert_called_once_with([100, 101, 102])


def test_revalidate_skips_all_pad_drafts():
    scheduler = _create_grammar_scheduler()
    scheduler.structured_output_manager.should_advance = Mock(return_value=True)
    req = _make_running_request(scheduler, ff_tokens=[])
    grammar = req.structured_output_request.grammar

    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {req.request_id: 4}
    scheduler_output.scheduled_spec_decode_tokens = {
        req.request_id: [PAD] * NUM_SPEC_TOKENS
    }

    scheduler._revalidate_grammar_spec_tokens(scheduler_output, [req.request_id])

    assert scheduler_output.scheduled_spec_decode_tokens == {
        req.request_id: [PAD] * NUM_SPEC_TOKENS
    }
    grammar.validate_tokens.assert_not_called()


def test_grammar_spec_requires_v2_model_runner(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    with pytest.raises(ValueError, match="V2 model runner"):
        _create_grammar_scheduler()
