# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU_OBSERVED/CUDA_UNVERIFIED contracts for the production Uno worker path.

The worker consumes a scheduler snapshot, emits sampled output before any
postprocessing, and retains K-wide collective buffers with per-request draft
validity. These tests drive sample_tokens with CPU collaborators; CUDA numerical
equivalence and latency remain device gates.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.spec_decode.test_uno_mrv2 import _uno_sample_tokens_runner
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.uno_step_timing import UnoStepTimingTracer


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
    runner.execute_model_state.skip_speculator_proposal = skip
    runner.execute_model_state.zero_next_draft_req_ids = frozenset(zero_rows)
    if not is_uno:
        runner.speculator = SimpleNamespace(
            supports_mm_inputs=False,
            propose=proposer.propose,
        )

    runner.sample_tokens(None)

    assert ("propose-step-7" in events) is expected_proposal
    assert events[:2] == ["output", "postprocess"]
    assert events[-2:] == ["publish", "kv"]


@pytest.mark.parametrize("num_reqs", [1, 2])
def test_tail_sample_tokens_reuses_regular_sampler(monkeypatch, num_reqs):
    """Both serial and batched K=0 dispatch use the real regular-sampler branch."""
    runner, _, events, _ = _uno_sample_tokens_runner(monkeypatch, num_reqs=num_reqs)
    batch = runner.execute_model_state.input_batch
    runner.execute_model_state.skip_speculator_proposal = True
    runner.execute_model_state.zero_next_draft_req_ids = frozenset(batch.req_ids)
    runner.batch_sharder = None
    runner.model.compute_logits = lambda hidden_states: hidden_states
    output = SimpleNamespace(
        sampled_token_ids=torch.full((num_reqs, 1), 7),
        num_sampled=torch.ones(num_reqs, dtype=torch.int32),
        num_rejected=torch.zeros(num_reqs, dtype=torch.int32),
    )

    def regular_sampler(logits, input_batch):
        assert logits.shape == (num_reqs, 2)
        assert input_batch.num_draft_tokens == 0
        events.append("regular-sampler")
        return output

    def rejection_sampler(*_args):
        pytest.fail("K=0 must not enter rejection verification")

    runner.sampler = regular_sampler
    runner.rejection_sampler = rejection_sampler
    runner.sample = GPUModelRunner.sample.__get__(runner)

    runner.sample_tokens(None)

    assert events == ["regular-sampler", "output", "postprocess", "publish", "kv"]


@pytest.mark.parametrize("all_tail", [False, True])
def test_tail_draft_retrieval_masks_rows_without_changing_collective_shape(
    monkeypatch, all_tail
):
    """Published stale storage remains K-wide but CPU consumers receive zero drafts."""
    runner, _, _, _ = _uno_sample_tokens_runner(monkeypatch, num_reqs=2)
    zero_rows = {"request-0", "request-1"} if all_tail else {"request-0"}
    runner.execute_model_state.skip_speculator_proposal = all_tail
    runner.execute_model_state.zero_next_draft_req_ids = frozenset(zero_rows)
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


def test_execute_model_timing_captures_late_tail_and_its_followup(monkeypatch):
    """The engagement record must include length-tail GPU stages after step three."""
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
    output = SimpleNamespace(
        debug_uno_step_id=1,
        debug_schedule_wall_ms=0.5,
        num_scheduled_tokens={"request-0": 1},
        skip_speculator_proposal=False,
        zero_next_draft_req_ids=set(),
        scheduled_spec_decode_tokens={},
        total_num_scheduled_tokens=1,
    )

    def execute_until_dispatch():
        with pytest.raises(BeforeDeviceDispatch):
            runner.execute_model(output)
        return traces[-1]

    for _ in range(3):
        assert execute_until_dispatch() is not None
    assert execute_until_dispatch() is None
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
