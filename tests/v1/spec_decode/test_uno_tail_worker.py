# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU_OBSERVED/CUDA_UNVERIFIED contracts for the production Uno worker path.

The worker consumes a scheduler snapshot, emits sampled output before any
postprocessing, and retains K-wide collective buffers with per-request draft
validity. These tests drive sample_tokens with CPU collaborators; CUDA numerical
equivalence and latency remain device gates.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tests.v1.spec_decode.test_uno_mrv2 import _uno_sample_tokens_runner
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.uno_step_timing import UnoStepTimingTracer


def _enable_exact_tail(
    runner, *, outputs, caps, stops=None, prompt_len=1, eos=None, ignore_eos=False
):
    """Use the real tracker when present; old-head runs reach the skip assertion."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.worker.gpu import model_runner

    tracker_type = getattr(model_runner, "UnoTailState", None)
    if tracker_type is None:
        # Old code has no detector. It must fail on proposing for a finished
        # request, rather than fail importing a class introduced by this fix.
        return
    runner._uno_tail = tracker_type(4096)
    for i, (output_len, cap) in enumerate(zip(outputs, caps)):
        params = SamplingParams(
            max_tokens=cap, stop_token_ids=stops or [], ignore_eos=ignore_eos
        )
        params.update_from_generation_config({}, eos_token_id=eos)
        runner._uno_tail.add_request(
            NewRequestData(
                req_id=f"request-{i}",
                prompt_token_ids=[1] * prompt_len,
                mm_features=[],
                sampling_params=params,
                pooling_params=None,
                block_ids=([],),
                num_computed_tokens=prompt_len + output_len,
                lora_request=None,
                prefill_token_ids=[1] * (prompt_len + output_len),
            )
        )


def test_exact_tail_worker_lifecycle_reseeds_resumed_and_streaming_requests():
    """Production add/remove wiring must not reuse a previous turn's finish."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
    from vllm.v1.worker.gpu import model_runner

    runner = object.__new__(GPUModelRunner)
    tracker_type = getattr(model_runner, "UnoTailState", None)
    runner._uno_tail = (
        tracker_type(4096) if tracker_type else SimpleNamespace(requests={})
    )
    slots = {}
    runner.req_states = Mock(req_id_to_index=slots)
    runner.req_states.add_request.side_effect = lambda **kwargs: slots.update(
        {kwargs["req_id"]: 0}
    )
    runner.req_states.remove_request.side_effect = lambda req_id: slots.pop(
        req_id, None
    )
    runner.model_state = Mock()
    runner.block_tables = Mock()
    runner.lora_state = Mock()
    runner.sampler = Mock()
    runner.prompt_logprobs_worker = Mock()
    runner.pooling_runner = None
    runner.encoder_cache = None
    runner.pp_handler = None
    runner.adaptive_verification = None
    runner.is_last_pp_rank = True

    def add(prompt_len, output_len, cap):
        output = SchedulerOutput.make_empty()
        output.scheduled_new_reqs = [
            NewRequestData(
                req_id="request-0",
                prompt_token_ids=[1] * prompt_len,
                prefill_token_ids=[1] * prompt_len + [7] * output_len,
                mm_features=[],
                sampling_params=SamplingParams(max_tokens=cap),
                pooling_params=None,
                block_ids=([],),
                num_computed_tokens=prompt_len + output_len,
                lora_request=None,
            )
        ]
        runner.add_requests(output)
        # The old worker reaches this behavioral assertion without importing
        # the new tracker or reading a field that does not exist on old code.
        assert "request-0" in runner._uno_tail.requests, (
            "add_requests must seed progress"
        )
        return runner._uno_tail.requests["request-0"]

    initial = add(3, 0, 5)
    assert initial.output_len == 0
    assert not initial.finished
    preempt = SchedulerOutput.make_empty()
    preempt.preempted_req_ids = {"request-0"}
    runner.finish_requests(preempt)
    assert not runner._uno_tail.requests
    assert not slots

    resumed = add(3, 4, 5)
    assert resumed is not initial
    assert resumed.output_len == 4
    assert resumed.output_cap == 5
    resumed.finished = True
    streaming = add(6, 0, 9)
    assert streaming is not resumed
    assert streaming.output_len == 0
    assert streaming.output_cap == 9
    assert not streaming.finished
    assert runner._remove_request("request-0")
    assert not runner._uno_tail.requests
    assert not slots


@pytest.mark.parametrize(
    "output_len,cap,tokens,accepted,stop_ids,expected_skip",
    [
        (3, 5, [7, 8, 9], 3, [], True),  # accepted-draft leap
        (120, 128, [7] * 9, 1, [], False),  # all eight drafts rejected
        (0, 16, [7] * 9, 9, [], False),
        (0, 128, [7, 8, 9], 3, [8], True),  # stop inside accepted prefix
        (0, 128, [7, 8, 9], 1, [8], False),  # rejected stop is unusable
    ],
)
def test_exact_tail_checks_accepted_prefix_before_proposing(
    monkeypatch, output_len, cap, tokens, accepted, stop_ids, expected_skip
):
    runner, _, events, _ = _uno_sample_tokens_runner(monkeypatch)
    _enable_exact_tail(runner, outputs=[output_len], caps=[cap], stops=stop_ids)
    sampled, counts, rejected = runner.sample(None, None, None)
    sampled.sampled_token_ids = torch.tensor([tokens])
    counts[0] = accepted

    runner.sample_tokens(None)

    assert ("propose-step-7" not in events) is expected_skip
    assert events[:3] == ["output", "postprocess", "copy-ready"]
    assert runner._zero_next_draft_req_ids == (
        frozenset({"request-0"}) if expected_skip else frozenset()
    )


@pytest.mark.parametrize("ignore_eos", [False, True])
def test_exact_tail_eos_prefix_respects_ignore_eos(monkeypatch, ignore_eos):
    runner, _, events, _ = _uno_sample_tokens_runner(monkeypatch)
    _enable_exact_tail(runner, outputs=[0], caps=[128], eos=8, ignore_eos=ignore_eos)
    sampled, counts, _ = runner.sample(None, None, None)
    sampled.sampled_token_ids = torch.tensor([[7, 8, 9]])
    counts[0] = 3
    runner.sample_tokens(None)
    assert ("propose-step-7" in events) is ignore_eos
    assert events[:3] == ["output", "postprocess", "copy-ready"]


def test_exact_tail_context_cap_skips_and_followup_uses_zero_drafts(monkeypatch):
    from vllm.v1.core.sched.output import SchedulerOutput

    runner, _, events, _ = _uno_sample_tokens_runner(monkeypatch)
    _enable_exact_tail(runner, outputs=[95], caps=[4096], prompt_len=4000)
    state = runner.execute_model_state
    runner.sample_tokens(None)
    assert "propose-step-7" not in events, "context-terminal sample must skip"

    original = SchedulerOutput.make_empty()
    original.num_scheduled_tokens = {"request-0": 9}
    original.total_num_scheduled_tokens = 9
    original.scheduled_spec_decode_tokens = {"request-0": [-1] * 8}
    runner._uno_step_timing = None
    runner.update_pp_decode_requests = lambda: None
    for name in ("finish_requests", "free_states", "add_requests", "update_requests"):
        monkeypatch.setattr(runner, name, lambda _output: None)
    runner.block_tables = SimpleNamespace(apply_staged_writes=lambda: None)
    snapshots = []

    class BeforeDeviceDispatch(Exception):
        pass

    def record_gather(output, _dummy):
        snapshots.append(output)
        raise BeforeDeviceDispatch

    runner.gather_batch_req_state = record_gather
    with pytest.raises(BeforeDeviceDispatch):
        runner.execute_model(original)
    followup = snapshots[0]
    assert followup.num_scheduled_tokens == {"request-0": 1}
    assert followup.scheduled_spec_decode_tokens == {}
    assert followup.zero_next_draft_req_ids == {"request-0"}
    assert original.num_scheduled_tokens == {"request-0": 9}
    assert len(original.scheduled_spec_decode_tokens["request-0"]) == 8

    # Exercise the real sampler dispatch on the compacted batch. The previous
    # assertion, through execute_model, proves K=0 came from production policy.
    runner.execute_model_state = state._replace(
        skip_speculator_proposal=True,
        zero_next_draft_req_ids=frozenset(followup.zero_next_draft_req_ids),
    )
    runner.batch_sharder = None
    runner.model.compute_logits = lambda hidden: hidden
    sampled, _, _ = runner.sample(None, None, None)
    runner.sampler = lambda *_args: sampled
    runner.rejection_sampler = lambda *_args: pytest.fail("finished row verified")
    runner.sample = GPUModelRunner.sample.__get__(runner)
    runner.sample_tokens(None)


@pytest.mark.parametrize("all_finished", [False, True])
def test_exact_tail_finished_rows_preserve_peer_grammar_and_collective_shape(
    monkeypatch, all_finished
):
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

    runner, _, events, _ = _uno_sample_tokens_runner(monkeypatch, num_reqs=2)
    _enable_exact_tail(runner, outputs=[4, 4 if all_finished else 0], caps=[5, 5])
    saved = runner.execute_model_state
    broadcasts = []
    runner.pp_handler = SimpleNamespace(
        broadcast=lambda *_args: None,
        broadcast_drafts=lambda tokens, _batch: broadcasts.append(tuple(tokens.shape)),
    )
    runner.sample_tokens(None)
    assert ("propose-step-7" not in events) is all_finished
    assert runner._zero_next_draft_req_ids == (
        frozenset({"request-0", "request-1"})
        if all_finished
        else frozenset({"request-0"})
    )
    assert broadcasts == [(2, 2)]

    original = SchedulerOutput.make_empty()
    original.num_scheduled_tokens = {"request-0": 3, "request-1": 3}
    original.total_num_scheduled_tokens = 6
    original.scheduled_spec_decode_tokens = {
        "request-0": [-1, -1],
        "request-1": [-1, -1],
    }
    followup, original_drafts = runner._uno_tail.prepare_followup(original)
    batch = saved.input_batch
    counts = np.array(
        [len(followup.scheduled_spec_decode_tokens.get(r, [])) for r in batch.req_ids]
    )
    batch.num_draft_tokens_per_req = counts
    batch.num_draft_tokens = int(counts.sum())
    runner.execute_model_state = saved._replace(
        uno_original_drafts=original_drafts,
        zero_next_draft_req_ids=frozenset(followup.zero_next_draft_req_ids),
        skip_speculator_proposal=followup.skip_speculator_proposal,
    )
    sampled_result = runner.sample(None, None, None)
    masks = []

    def sample_with_grammar(_hidden, _batch, grammar):
        masks.append(grammar.grammar_bitmask[:, 0].tolist())
        return sampled_result

    runner.sample = sample_with_grammar
    # Reverse request ordering in the grammar to catch peer-offset corruption.
    grammar = GrammarOutput(
        ["request-1", "request-0"], np.arange(6, dtype=np.int32).reshape(6, 1)
    )
    runner.sample_tokens(grammar)
    assert masks == ([[0, 3]] if all_finished else [[0, 1, 2, 3]])


@pytest.mark.parametrize(
    "is_uno,zero_rows,skip,expected_proposal",
    [
        (True, {"request-0", "request-1"}, True, False),
        (True, {"request-0"}, True, True),
        (False, {"request-0", "request-1"}, True, True),
        (True, {"request-0", "request-1"}, False, True),
    ],
)
def test_sample_tokens_suppresses_only_all_uno_rows(
    monkeypatch, is_uno, zero_rows, skip, expected_proposal
):
    """One tail row must not suppress a peer's dense proposal or another method."""
    runner, proposer, events, _ = _uno_sample_tokens_runner(monkeypatch, num_reqs=2)
    runner.execute_model_state = runner.execute_model_state._replace(
        skip_speculator_proposal=skip,
        zero_next_draft_req_ids=frozenset(zero_rows),
    )
    if not is_uno:
        runner.speculator = SimpleNamespace(
            supports_mm_inputs=False,
            propose=proposer.propose,
        )

    runner.sample_tokens(None)

    assert ("propose-step-7" in events) is expected_proposal
    assert events[:2] == ["output", "postprocess"]
    assert events[-2:] == ["publish", "kv"]


@pytest.mark.parametrize("all_tail", [False, True])
def test_tail_draft_retrieval_masks_rows_without_changing_collective_shape(
    monkeypatch, all_tail
):
    """Published stale storage remains K-wide but CPU consumers receive zero drafts."""
    runner, _, _, _ = _uno_sample_tokens_runner(monkeypatch, num_reqs=2)
    zero_rows = {"request-0", "request-1"} if all_tail else {"request-0"}
    runner.execute_model_state = runner.execute_model_state._replace(
        skip_speculator_proposal=all_tail,
        zero_next_draft_req_ids=frozenset(zero_rows),
    )
    published = []
    broadcasts = []

    def publish(input_batch, draft_tokens):
        assert draft_tokens.shape == (2, 2)
        published.append(DraftTokenIds(input_batch.req_ids, draft_tokens.tolist()))

    runner.draft_tokens_handler = SimpleNamespace(
        set_draft_tokens=publish,
        get_draft_tokens=lambda: published[-1],
    )
    runner.pp_handler = SimpleNamespace(
        broadcast=lambda *_args: None,
        broadcast_drafts=lambda tokens, _batch: broadcasts.append(tokens.clone()),
    )
    runner.req_states.draft_tokens[:] = torch.tensor([[88, 89], [98, 99]])

    runner.sample_tokens(None)
    drafts = runner.take_draft_token_ids()

    assert len(broadcasts) == 1
    assert broadcasts[0].shape == (2, 2)
    assert drafts.req_ids == ["request-0", "request-1"]
    assert drafts.draft_token_ids == ([[], []] if all_tail else [[], [8, 9]])
    if all_tail:
        assert broadcasts[0].tolist() == [[88, 89], [98, 99]]


@pytest.mark.parametrize("tail_mode", ["early", "exact"])
def test_execute_model_timing_captures_late_tail_and_its_followup(
    monkeypatch, tail_mode
):
    """Exact diagnostics start events before an unknown late terminal sample."""
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu import uno_step_timing

    monkeypatch.setattr(uno_step_timing, "UNO_TAIL_MODE", tail_mode, raising=False)
    tracer = UnoStepTimingTracer(capture_cuda_events=False)
    runner, _, _, _ = _uno_sample_tokens_runner(monkeypatch)
    runner._uno_step_timing = tracer
    runner.req_states.num_reqs = 1
    runner.update_pp_decode_requests = lambda: None
    for name in ("finish_requests", "free_states", "add_requests", "update_requests"):
        monkeypatch.setattr(runner, name, lambda _output: None)
    runner.block_tables = SimpleNamespace(apply_staged_writes=lambda: None)
    traces = []
    begin = tracer.begin

    def record_begin(*args):
        trace = begin(*args)
        traces.append(trace)
        return trace

    monkeypatch.setattr(tracer, "begin", record_begin)

    class BeforeDeviceDispatch(Exception):
        pass

    def stop_before_device_dispatch(*_args):
        raise BeforeDeviceDispatch

    runner.gather_batch_req_state = stop_before_device_dispatch
    output = SchedulerOutput.make_empty()
    output.debug_uno_step_id = 1
    output.debug_schedule_wall_ms = 0.5
    output.num_scheduled_tokens = {"request-0": 1}
    output.total_num_scheduled_tokens = 1

    def execute_until_dispatch():
        with pytest.raises(BeforeDeviceDispatch):
            runner.execute_model(output)
        return traces[-1]

    for _ in range(3):
        assert execute_until_dispatch() is not None
    late_trace = execute_until_dispatch()
    assert (late_trace is not None) is (tail_mode == "exact"), (
        "exact diagnostics must trace verification before a late finish is known"
    )
    output.zero_next_draft_req_ids = {"request-0"}
    output.skip_speculator_proposal = True

    for expected_step in (5, 6):
        trace = execute_until_dispatch()
        assert trace is not None, "late draft-free tail stage must be observable"
        payload = trace.payload()
        assert payload["tail_mode_rows"] == 1
        assert payload["current_draft_counts"] == (0,)
        assert payload["request_steps"] == (expected_step,)
        assert payload["proposal_skipped_terminal"]

    output.debug_uno_step_id = None
    assert execute_until_dispatch() is None, "debug-off steps create no trace"


@pytest.mark.parametrize("all_finished", [False, True])
def test_exact_tail_trace_records_post_sample_skip_without_zero_draft_claim(
    monkeypatch, all_finished
):
    """A K=8 terminal sample updates actual skip without becoming a K=0 trace."""
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu import model_runner

    runner, _, events, _ = _uno_sample_tokens_runner(monkeypatch, num_reqs=2)
    _enable_exact_tail(runner, outputs=[0, 0], caps=[5, 5 if all_finished else 128])
    sampled, counts, _ = runner.sample(None, None, None)
    sampled.sampled_token_ids = torch.full((2, 9), 7, dtype=torch.int64)
    counts.fill_(9)
    tracer = UnoStepTimingTracer(capture_cuda_events=False)
    output = SchedulerOutput.make_empty()
    output.debug_uno_step_id = 1
    output.num_scheduled_tokens = {"request-0": 9, "request-1": 9}
    output.total_num_scheduled_tokens = 18
    output.scheduled_spec_decode_tokens = {
        "request-0": [7] * 8,
        "request-1": [7] * 8,
    }
    trace = tracer.begin(output, 2)
    assert trace is not None
    runner.execute_model_state = runner.execute_model_state._replace(
        uno_step_trace=trace
    )
    runner._uno_step_timing = tracer
    published = []
    monkeypatch.setattr(tracer, "_log", lambda trace: published.append(trace.payload()))
    worker_logs = []
    monkeypatch.setattr(model_runner, "UNO_STEP_TIMING_DEBUG", True)
    monkeypatch.setattr(
        model_runner.logger,
        "info",
        lambda message, *args: worker_logs.append(message % args),
    )

    runner.sample_tokens(None)

    assert ("propose-step-7" not in events) is all_finished
    payload = published[0]
    assert payload["proposal_skipped_terminal"] is all_finished
    assert payload["tail_mode_rows"] == 0
    assert payload["current_draft_counts"] == (8, 8)
    expected_finished = 2 if all_finished else 1
    assert (
        f"UNO_TAIL_STEP proposals_skipped={int(all_finished)} tail_mode_rows=0 "
        f"post_sample_finished_rows={expected_finished} scheduled_rows=2"
    ) in worker_logs, "the engagement line must distinguish finishing K=8 from K=0"
    assert payload["post_sample_finished_rows"] == expected_finished
    assert payload["finish_check_ms"] >= 0
    assert ("propose" not in trace.wall_ms) is all_finished
    assert events[:3] == ["output", "postprocess", "copy-ready"]
