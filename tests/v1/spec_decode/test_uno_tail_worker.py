# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU_OBSERVED/CUDA_UNVERIFIED contracts for the production Uno worker path.

The worker consumes a scheduler snapshot, emits sampled output before any
postprocessing, and retains K-wide collective buffers with per-request draft
validity. These tests drive sample_tokens with CPU collaborators; CUDA numerical
equivalence and latency remain device gates.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.v1.spec_decode.test_uno_mrv2 import _uno_sample_tokens_runner
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.uno_step_timing import UnoStepTimingTracer


def test_tail_worker_uses_scheduler_zero_draft_batch(monkeypatch):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu import model_runner
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

    runner, _, events, _ = _uno_sample_tokens_runner(monkeypatch)
    state = runner.execute_model_state

    original = SchedulerOutput.make_empty()
    original.num_scheduled_tokens = {"request-0": 1}
    original.total_num_scheduled_tokens = 1
    original.zero_next_draft_req_ids = {"request-0"}
    original.skip_speculator_proposal = True
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
    assert followup is original, "worker must consume the scheduler's tail layout"

    # Derive the actual InputBatch from the worker snapshot. Only device-copy
    # and Triton kernels are replaced; row counts/logits offsets are production.
    runner.device = torch.device("cpu")
    runner.max_num_reqs = 1
    runner.decode_query_len = 9
    runner.input_buffers = InputBuffers(1, 9, runner.device)
    runner.fast_prefill = None
    runner.model_config = SimpleNamespace(rswa_window=None)
    runner.model_state.num_new_sampled_tokens_per_step = 1
    runner.req_states.req_id_to_index = {"request-0": 0}
    runner.req_states.prefill_len = SimpleNamespace(
        np=np.array([4000], dtype=np.int32), gpu=torch.tensor([4000])
    )
    runner.req_states.num_computed_prefill_tokens = np.array([4000], dtype=np.int32)
    runner.req_states.num_computed_tokens_np = np.array([4095], dtype=np.int32)
    runner.req_states.num_computed_tokens.gpu = torch.tensor([4095])

    def copy_to_cpu(values, *, device=None, out=None):
        tensor = torch.as_tensor(values)
        return tensor.clone() if out is None else out.copy_(tensor)

    def prepare_positions(indices, starts, computed, positions, seq_lens):
        positions[0] = computed[indices[0]]
        seq_lens[: len(indices)] = computed[indices] + starts[1:] - starts[:-1]

    logits_rows = []

    def combine_tokens(
        input_ids,
        indices,
        sampled,
        starts,
        seq_lens,
        prefill,
        drafts,
        cu_logits,
        num_logits,
        num_bonus,
    ):
        logits_rows.append((cu_logits.tolist(), num_logits, num_bonus))
        input_ids[: len(indices)] = sampled[indices, 0]
        return starts[1:].long() - 1

    monkeypatch.setattr(model_runner, "async_copy_to_gpu", copy_to_cpu)
    monkeypatch.setattr(model_runner, "prepare_pos_seq_lens", prepare_positions)
    monkeypatch.setattr(
        model_runner, "combine_sampled_and_draft_tokens", combine_tokens
    )
    batch_state, uniform_tokens = GPUModelRunner.gather_batch_req_state(
        runner, followup, False
    )
    assert batch_state is not None and uniform_tokens == 1
    batch = GPUModelRunner.prepare_inputs(
        runner,
        followup,
        batch_state,
        BatchExecutionDescriptor(CUDAGraphMode.NONE, 1, 1, uniform_tokens),
    )
    assert isinstance(batch, InputBatch) and batch is not state.input_batch
    assert batch.num_scheduled_tokens.tolist() == [1]
    assert batch.num_draft_tokens == 0
    assert batch.query_start_loc_np.tolist() == [0, 1]
    assert batch.cu_num_logits_np.tolist() == [0, 1]
    assert logits_rows == [([0, 1], 1, 1)]

    # Feed that derived batch to the real sampler dispatch.
    runner.execute_model_state = state._replace(
        input_batch=batch,
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
    assert "propose-step-7" not in events


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


def test_execute_model_timing_captures_late_tail_and_its_followup(monkeypatch):
    """Late tail diagnostics do not require tracing every verification step."""
    from vllm.v1.core.sched.output import SchedulerOutput

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
    assert late_trace is None, "ordinary late verification needs no timing trace"
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
