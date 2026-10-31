# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``BaseSpeculator.observe_step``: one ``StepObservation`` per sampled step.

The runner calls it only for a speculator that overrides it, on every sampled
step (speculative, no-draft and target-only ones), and builds no record
otherwise.
"""

import timeit
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.model_runner as model_runner_module
import vllm.v1.worker.gpu.spec_decode.round_runner as round_runner
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.ngram.speculator import NgramGPUSpeculator
from vllm.v1.worker.gpu.spec_decode.round import SpecResources
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator
from vllm.v1.worker.gpu.spec_decode.step_observation import StepObservation


class _Speculator(BaseSpeculator):
    def init_cudagraph_manager(self, cudagraph_mode):
        pass

    def capture(self):
        pass

    def propose(self, input_batch, *args, **kwargs):
        return torch.zeros(input_batch.num_reqs, 3, dtype=torch.int64)


class _Observer(_Speculator):
    def __init__(self):
        self.seen: list[StepObservation] = []

    def observe_step(self, step: StepObservation) -> None:
        self.seen.append(step)


# --- the record ---------------------------------------------------------------


def _observation() -> StepObservation:
    return StepObservation(
        step=0,
        start_ns=0,
        req_ids=["a", "b"],
        num_reqs=2,
        num_reqs_padded=2,
        num_tokens=2,
        num_tokens_padded=2,
        cudagraph_mode=CUDAGraphMode.NONE,
        uniform_token_count=1,
        num_ubatches=1,
        num_prefill_tokens=0,
        num_spec_tokens=0,
        num_scheduled_tokens=np.ones(2, dtype=np.int32),
        num_computed_tokens=np.ones(2, dtype=np.int32),
        is_prefilling=np.zeros(2, dtype=bool),
        num_draft_tokens=None,
        num_sampled=torch.ones(2, dtype=torch.int32),
        num_rejected=torch.zeros(2, dtype=torch.int32),
    )


# --- which speculators are called -----------------------------------------------


def test_observe_step_is_a_hook_only_when_overridden():
    assert "observe_step" in _Observer().overridden_hooks()
    assert "observe_step" not in _Speculator().overridden_hooks()
    assert (
        "observe_step"
        not in NgramGPUSpeculator.__new__(NgramGPUSpeculator).overridden_hooks()
    )
    BaseSpeculator.observe_step(_Speculator(), _observation())


# --- the runner -----------------------------------------------------------------


def _batch(
    num_reqs: int,
    num_reqs_padded: int,
    scheduled: list[int],
    drafts: list[int] | None,
    prefilling: list[bool],
    computed: list[int],
) -> InputBatch:
    num_tokens = sum(scheduled)
    batch = InputBatch.make_dummy(
        num_reqs_padded, num_tokens + 2, InputBuffers(8, 32, torch.device("cpu"))
    )
    batch.req_ids = [f"r{i}" for i in range(num_reqs)]
    batch.num_reqs = num_reqs
    batch.num_reqs_after_padding = num_reqs_padded
    batch.num_tokens = num_tokens
    batch.num_tokens_after_padding = num_tokens + 2
    batch.num_scheduled_tokens = np.array(scheduled, dtype=np.int32)
    batch.num_computed_tokens_np = np.array(computed, dtype=np.int32)
    batch.is_prefilling_np = np.array(prefilling)
    batch.has_prefill = any(prefilling)
    batch.num_draft_tokens_per_req = (
        None if drafts is None else np.array(drafts, dtype=np.int32)
    )
    batch.num_draft_tokens = 0 if drafts is None else sum(drafts)
    batch.idx_mapping = torch.arange(num_reqs, dtype=torch.int32)
    return batch


def _runner(speculator, monkeypatch) -> GPUModelRunner:
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.is_last_pp_rank = True
    runner.pcp_manager = runner.pp_handler = runner.aux_output_connector = None
    runner.speculator = speculator
    runner.speculator_hooks = (
        frozenset() if speculator is None else speculator.overridden_hooks()
    )
    runner.model = SimpleNamespace(compute_logits=lambda x: x)
    runner.prompt_logprobs_worker = SimpleNamespace(
        compute_prompt_logprobs=lambda *args: {},
        compute_prompt_token_id_logprobs=lambda *args: {},
    )
    runner.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=None),
        num_computed_tokens=SimpleNamespace(gpu=None),
        prompt_len=SimpleNamespace(np=None),
        draft_tokens=torch.zeros(8, 3, dtype=torch.int64),
        last_sampled_tokens=None,
        next_prefill_tokens=None,
    )
    runner.main_stream = runner.output_copy_stream = None
    runner.check_ep_fault = False
    runner.kv_connector = Mock()
    runner.postprocess_sampled = lambda *args, **kw: None  # type: ignore[method-assign]
    runner.spec_round = None
    if speculator is not None:
        runner.spec_round = round_runner.SpecRound(
            None,  # type: ignore[arg-type]
            SpecResources(max_candidates=4, capture_shapes=()),
            max_num_reqs=8,
            max_num_tokens=32,
            num_speculative_steps=3,
            device=torch.device("cpu"),
        )
    runner.sampler = SimpleNamespace(
        sampling_states=SimpleNamespace(
            temperature=SimpleNamespace(gpu=None), seeds=SimpleNamespace(gpu=None)
        )
    )
    runner.num_speculative_steps = 3
    runner.adaptive_verification = None
    runner._draft_workspace_lane = 0
    runner.draft_tokens_handler = SimpleNamespace(set_draft_tokens=lambda *a: None)
    runner.eplb = Mock()
    monkeypatch.setattr(model_runner_module, "AsyncOutput", lambda **kw: kw)
    monkeypatch.setattr(
        model_runner_module, "get_drafter_hidden_states", lambda model, h: h
    )
    monkeypatch.setattr(
        model_runner_module, "use_workspace_lane", lambda lane: _NullContext()
    )
    return runner


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _run_step(
    runner: GPUModelRunner,
    batch: InputBatch,
    desc: BatchExecutionDescriptor,
    num_spec_tokens: int = 3,
    skip_drafts_dp: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_sampled = torch.arange(1, batch.num_reqs + 1, dtype=torch.int32)
    num_rejected = torch.zeros(batch.num_reqs, dtype=torch.int32)
    output = SimpleNamespace(
        sampled_token_ids=torch.ones(batch.num_reqs, 1, dtype=torch.int64)
    )
    runner.sample = lambda *args: (  # type: ignore[method-assign]
        output,
        num_sampled,
        num_rejected,
    )
    runner.execute_model_state = model_runner_module.ExecuteModelState(
        input_batch=batch,
        attn_metadata={},
        slot_mappings_by_layer={},
        hidden_states=torch.zeros(batch.num_tokens, 4),
        aux_hidden_states=None,
        dp_sync_state=None,
        finished_req_ids=set(),
        ec_connector_output=None,
        cudagraph_stats=None,
        num_spec_tokens_to_schedule=num_spec_tokens,
        batch_desc=desc,
        start_ns=12345,
    )
    GPUModelRunner.sample_tokens(runner, None)
    return num_sampled, num_rejected


def test_chain_step_is_observed_once_with_its_batch(monkeypatch):
    speculator = _Observer()
    runner = _runner(speculator, monkeypatch)
    batch = _batch(
        num_reqs=3,
        num_reqs_padded=4,
        scheduled=[4, 4, 4],
        drafts=[3, 3, 3],
        prefilling=[False, False, False],
        computed=[10, 20, 30],
    )
    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL, num_tokens=16, num_reqs=4, uniform_token_count=4
    )
    num_sampled, num_rejected = _run_step(runner, batch, desc)

    assert len(speculator.seen) == 1
    obs = speculator.seen[0]
    assert obs.step == 0
    assert obs.start_ns == 12345
    assert obs.req_ids == ["r0", "r1", "r2"]
    assert (obs.num_reqs, obs.num_reqs_padded) == (3, 4)
    assert (obs.num_tokens, obs.num_tokens_padded) == (12, 14)
    assert obs.cudagraph_mode == CUDAGraphMode.FULL
    assert obs.uniform_token_count == 4
    assert obs.num_ubatches == 1
    assert obs.num_prefill_tokens == 0
    assert obs.num_spec_tokens == 3
    assert obs.num_scheduled_tokens.tolist() == [4, 4, 4]
    assert obs.num_computed_tokens.tolist() == [10, 20, 30]
    assert obs.is_prefilling.tolist() == [False, False, False]
    assert obs.num_draft_tokens is not None
    assert obs.num_draft_tokens.tolist() == [3, 3, 3]
    # Device results are passed as they are, not copied.
    assert obs.num_sampled is num_sampled
    assert obs.num_rejected is num_rejected

    _run_step(runner, batch, desc)
    assert [o.step for o in speculator.seen] == [0, 1]


def test_no_draft_steps_are_observed(monkeypatch):
    """A step at candidate budget 1 (skip_drafts) is observed like any other."""
    speculator = _Observer()
    runner = _runner(speculator, monkeypatch)
    batch = _batch(
        num_reqs=2,
        num_reqs_padded=2,
        scheduled=[5, 1],
        drafts=None,
        prefilling=[True, False],
        computed=[0, 7],
    )
    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE, num_tokens=8, num_reqs=None
    )
    _run_step(runner, batch, desc, num_spec_tokens=0)

    assert len(speculator.seen) == 1
    obs = speculator.seen[0]
    assert obs.num_draft_tokens is None
    assert obs.num_prefill_tokens == 5
    assert obs.is_prefilling.tolist() == [True, False]
    assert obs.num_spec_tokens == 0
    assert obs.cudagraph_mode == CUDAGraphMode.PIECEWISE
    assert obs.uniform_token_count is None


def test_scheduler_skipped_drafts_show_as_zero_spec_tokens(monkeypatch):
    speculator = _Observer()
    runner = _runner(speculator, monkeypatch)
    batch = _batch(2, 2, [1, 1], None, [False, False], [3, 4])
    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL, num_tokens=2, num_reqs=2
    )
    _run_step(runner, batch, desc, num_spec_tokens=0)
    assert speculator.seen[0].num_spec_tokens == 0


def test_step_is_observed_after_the_commit_and_before_propose(monkeypatch):
    """The speculator sees the step once its path is committed (KV and state
    along the path, ``input_batch.accept_result`` set) and before it drafts."""
    order: list[str] = []
    speculator = _Observer()
    speculator.observe_step = lambda step: order.append(  # type: ignore[method-assign]
        "observe" if batch.accept_result is order else "observe-before-accept"
    )
    propose = speculator.propose

    def record_propose(*a, **kw):
        order.append("propose")
        return propose(*a, **kw)

    speculator.propose = record_propose  # type: ignore[method-assign]
    runner = _runner(speculator, monkeypatch)
    spec_round = runner.spec_round
    assert spec_round is not None
    spec_round.commit = lambda *a: order.append("commit")  # type: ignore[method-assign]
    spec_round.accept_result = lambda *a: order  # type: ignore[method-assign]
    batch = _batch(2, 2, [4, 4], [3, 3], [False, False], [1, 2])
    batch.verify_plan = object()
    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL, num_tokens=8, num_reqs=2, uniform_token_count=4
    )
    _run_step(runner, batch, desc)
    assert order == ["commit", "observe", "propose"]


@pytest.mark.parametrize("with_speculator", [True, False])
def test_no_record_is_built_unless_observe_step_is_overridden(
    monkeypatch, with_speculator
):
    speculator = _Speculator() if with_speculator else None
    runner = _runner(speculator, monkeypatch)

    def unexpected(*args, **kwargs):
        pytest.fail("built a StepObservation for a speculator that does not observe")

    monkeypatch.setattr(round_runner, "StepObservation", unexpected)
    monkeypatch.setattr(BaseSpeculator, "observe_step", unexpected)
    if not with_speculator:
        runner.num_speculative_steps = 0
    batch = _batch(2, 2, [4, 4], [3, 3], [False, False], [1, 2])
    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL, num_tokens=8, num_reqs=2
    )
    _run_step(runner, batch, desc)


@pytest.mark.parametrize("observing", [True, False])
def test_step_start_is_stamped_only_for_an_observer(monkeypatch, observing):
    from contextlib import nullcontext

    from vllm.v1.core.sched.output import SchedulerOutput

    batch = _batch(2, 2, [1, 1], None, [False, False], [3, 4])
    scheduled = SchedulerOutput.make_empty()
    scheduled.num_scheduled_tokens = {"r0": 1, "r1": 1}
    scheduled.total_num_scheduled_tokens = 2
    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE, num_tokens=2, num_reqs=2
    )
    outputs = torch.zeros(2, 4)
    runner = SimpleNamespace(
        update_pp_decode_requests=lambda: None,
        finish_requests=lambda s: None,
        free_states=lambda s: None,
        add_requests=lambda s: None,
        update_requests=lambda s: None,
        block_tables=Mock(),
        draft_tokens_handler=Mock(),
        aux_output_connector=None,
        cudagraph_manager=object(),
        attn_groups=[],
        gather_batch_req_state=lambda *a: (
            SimpleNamespace(num_tokens=2, has_prefill=False),
            1,
        ),
        pcp_manager=None,
        lora_config=None,
        is_encoder_decoder=False,
        encoder_cache=None,
        dp_size=1,
        dp_rank=0,
        parallel_config=object(),
        ubatch_runner=None,
        decode_query_len=4,
        observability_config=SimpleNamespace(cudagraph_metrics=False),
        prepare_inputs=lambda *a: batch,
        prepare_attn=lambda b: ([], []),
        spec_round=None,
        model_state=SimpleNamespace(
            preprocess_state=lambda *a: None,
            prepare_attn=lambda *a, **kw: {},
            prepare_inputs=lambda *a: {},
        ),
        kv_cache_config=object(),
        req_states=SimpleNamespace(num_computed_tokens=SimpleNamespace(gpu=None)),
        dcp_size=1,
        uses_inputs_embeds=False,
        is_first_pp_rank=True,
        is_last_pp_rank=True,
        speculator=None,
        speculator_hooks=frozenset({"observe_step"} if observing else ()),
        eplb=Mock(),
        step_timing=Mock(),
        kv_connector=Mock(),
        vllm_config=object(),
        model_config=object(),
        model=lambda **kw: outputs,
        use_aux_hidden_state_outputs=False,
    )
    monkeypatch.setattr(
        model_runner_module, "dispatch_cg_and_sync_dp", lambda *a, **kw: (desc, None)
    )
    monkeypatch.setattr(
        model_runner_module, "build_slot_mappings_by_layer", lambda *a: {}
    )
    monkeypatch.setattr(
        model_runner_module, "set_forward_context", lambda *a, **kw: nullcontext()
    )
    monkeypatch.setattr(model_runner_module.time, "perf_counter_ns", lambda: 777)
    GPUModelRunner.execute_model(runner, scheduled)
    state = runner.execute_model_state
    assert state.start_ns == (777 if observing else 0)
    assert state.batch_desc is desc


# --- cost -----------------------------------------------------------------------


def test_observe_step_cost(monkeypatch):
    """The disabled path is one frozenset test; the enabled one stays in
    single-digit microseconds of host time (no sync, no copies)."""
    observer = _Observer()
    runner = _runner(observer, monkeypatch)
    batch = _batch(4, 4, [4, 4, 4, 4], [3, 3, 3, 3], [False] * 4, [1, 2, 3, 4])
    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL, num_tokens=16, num_reqs=4, uniform_token_count=4
    )
    sampled = torch.ones(4, dtype=torch.int32)
    rejected = torch.zeros(4, dtype=torch.int32)
    spec_round = runner.spec_round
    assert spec_round is not None
    n = 20_000
    off = frozenset({"accept"})
    disabled_s = min(timeit.repeat(lambda: "observe_step" in off, number=n, repeat=5))
    observer.seen = _Discard()  # type: ignore[assignment]
    enabled_s = min(
        timeit.repeat(
            lambda: spec_round.observe(observer, batch, desc, 0, 3, sampled, rejected),
            number=n,
            repeat=5,
        )
    )
    disabled_ns = disabled_s / n * 1e9
    enabled_us = enabled_s / n * 1e6
    print(f"observe_step: disabled {disabled_ns:.0f} ns, enabled {enabled_us:.2f} us")
    # Generous bounds: a shared CPU runner must not make this flaky.
    assert disabled_ns < 1_000
    assert enabled_us < 50


class _Discard:
    def append(self, item):
        pass
