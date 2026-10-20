# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hooks that let a speculator own verification (tree or other non-chain drafts).

Every hook is a no-op by default: a speculator that does not override it gets
the behavior vLLM had before the hook existed.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.model_runner as model_runner_module
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.ngram.speculator import NgramGPUSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import (
    BaseSpeculator,
)


def test_native_decode_is_sticky_by_request_identity_and_drains_pending_state():
    from vllm.v1.worker.gpu.spec_decode.native_decode import NativeDecodeState

    state = NativeDecodeState(lambda num_reqs, req_ids: 4 <= num_reqs <= 8)
    # A speculative forward leaves an accepted state to consume on the next step.
    assert state.select(["a", "b"], set(), {"a", "b"}) == (False, False)
    # Crossing the policy boundary stops drafting, but finishes pending verification.
    assert state.select(["b", "a", "c", "d"], set(), {"a", "b"}) == (True, False)
    # One target-only transition consumes the accepted recurrent-state offset.
    assert state.select(["a", "b"], {"c", "d"}, set()) == (True, False)
    # Filtering/compaction cannot resume a stale drafter when the batch shrinks.
    assert state.select(["b"], {"a"}, set()) == (True, True)
    assert state.select(["b"], set(), set()) == (True, True)
    # A new request merged into an OFF batch is also disabled, without a draft prefill.
    assert state.select(["new", "b"], set(), set()) == (True, True)
    assert state.select(["new"], {"b"}, set()) == (True, True)
    # Only request finish resets the contract (including reuse of the same ID).
    assert state.select(["new"], {"new"}, set()) == (False, False)


@pytest.mark.parametrize("native_target", [False, True])
def test_suspended_batch_publishes_no_drafts_without_calling_the_speculator(
    monkeypatch, native_target
):
    from unittest.mock import Mock

    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

    batch = InputBatch.make_dummy(2, 2, InputBuffers(2, 4, torch.device("cpu")))
    batch.skip_draft = True
    batch.native_target = native_target
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.execute_model_state = SimpleNamespace(
        input_batch=batch,
        attn_metadata={},
        slot_mappings_by_layer={},
        hidden_states=torch.zeros(2, 4),
        aux_hidden_states=None,
        dp_sync_state=None,
        finished_req_ids=set(),
        ec_connector_output=None,
        cudagraph_stats=None,
        num_spec_tokens_to_schedule=3,
    )
    runner.is_last_pp_rank = True
    runner.pcp_manager = runner.pp_handler = runner.aux_output_connector = None
    runner.speculator = _MinimalSpeculator()
    runner.speculator.supports_mm_inputs = False  # type: ignore[attr-defined]
    runner.model = SimpleNamespace(compute_logits=lambda x: x)
    output = SimpleNamespace(sampled_token_ids=torch.ones(2, 1, dtype=torch.int32))
    runner.sample = lambda *args: (  # type: ignore[method-assign]
        output,
        torch.ones(2),
        torch.zeros(2),
    )
    runner.prompt_logprobs_worker = SimpleNamespace(
        compute_prompt_logprobs=lambda *args: {},
        compute_prompt_token_id_logprobs=lambda *args: {},
    )
    runner.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=None),
        num_computed_tokens=SimpleNamespace(gpu=None),
        prompt_len=SimpleNamespace(np=None),
        draft_tokens=torch.full((2, 3), 123),
    )
    runner.main_stream = runner.output_copy_stream = None
    runner.check_ep_fault = False
    runner.kv_connector = Mock()
    runner.postprocess_sampled = lambda *args: None  # type: ignore[method-assign]
    runner.sampler = object()
    runner.num_speculative_steps = 3
    published: list[torch.Tensor] = []
    runner.draft_tokens_handler = SimpleNamespace(
        set_draft_tokens=lambda batch, tokens: published.append(tokens.clone())
    )
    runner.eplb = Mock()

    monkeypatch.setattr(model_runner_module, "AsyncOutput", lambda **kw: kw)

    def unexpected_draft(*args):
        pytest.fail("OFF must not prepare hidden states or run the drafter")

    monkeypatch.setattr(
        model_runner_module, "get_drafter_hidden_states", unexpected_draft
    )
    commits = []
    runner.speculator.post_accept = lambda *args: commits.append(  # type: ignore[method-assign]
        "verify"
    )
    runner.speculator_hooks = frozenset({"post_accept"})
    GPUModelRunner.sample_tokens(runner, None)
    assert commits == ([] if native_target else ["verify"])
    assert len(published) == 1
    assert published[0].shape == (2, 0)
    assert torch.all(runner.req_states.draft_tokens == 123)


def test_native_decode_builds_separate_target_metadata_and_graphs(monkeypatch):
    from unittest.mock import Mock

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.speculator = SimpleNamespace(
        may_suspend_drafting=True, suspend_drafting=lambda num_reqs, req_ids: False
    )
    runner.scheduler_config = SimpleNamespace(async_scheduling=False)
    runner.parallel_config = SimpleNamespace(
        pipeline_parallel_size=1,
        data_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_dbo=False,
    )
    runner.cache_config = SimpleNamespace(
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        use_replayssm=False,
        use_kda_recoverssm=False,
    )
    runner.vllm_config = SimpleNamespace(
        speculative_config=object(),
        kv_transfer_config=None,
        parallel_config=runner.parallel_config,
        cache_config=runner.cache_config,
        scheduler_config=runner.scheduler_config,
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY
        ),
    )
    runner.model_state = SimpleNamespace(num_new_sampled_tokens_per_step=1)
    runner.adaptive_verification = None
    runner.kv_cache_config = object()
    runner.device = "cpu"
    runner.kernel_block_sizes = [16, 16]
    runner.lora_capture_cases = [0]
    original_spec = runner.vllm_config.speculative_config
    # An excluded draft-only group need not use the target kernel block size.
    init = Mock(return_value=([["native"], []], None, [16, 32]))
    monkeypatch.setattr(model_runner_module, "init_attn_backend", init)
    manager = Mock(return_value="native graphs")
    monkeypatch.setattr(model_runner_module, "ModelCudaGraphManager", manager)
    GPUModelRunner._init_native_decode(runner, {"target"}, "mode")
    assert init.called, "A speculator that may suspend must get native metadata"
    assert init.call_args.args[1].speculative_config is None
    assert init.call_args.kwargs["active_layer_names"] == {"target"}
    assert manager.call_args.kwargs["decode_query_len"] == 1
    assert runner.native_cudagraph_manager == "native graphs"
    assert runner.native_attn_groups == [["native"], []]
    assert runner.vllm_config.speculative_config is original_spec
    assert runner.native_decode_state is not None
    assert runner.native_decode_state.suspend == runner.speculator.suspend_drafting


@pytest.mark.parametrize("speculator", [None, "default"])
def test_native_decode_is_not_set_up_unless_the_speculator_opts_in(
    monkeypatch, speculator
):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.speculator = (
        None if speculator is None else _MinimalSpeculator.__new__(_MinimalSpeculator)
    )
    runner.native_decode_state = None
    unexpected = Mock(side_effect=AssertionError("native setup must stay off"))
    monkeypatch.setattr(model_runner_module, "native_decode_config", unexpected)
    monkeypatch.setattr(model_runner_module, "init_attn_backend", unexpected)
    GPUModelRunner._init_native_decode(runner, {"target"}, "mode")
    assert runner.native_decode_state is None


def test_suspend_drafting_is_asked_once_per_batch_with_its_requests():
    from vllm.v1.worker.gpu.spec_decode.native_decode import NativeDecodeState

    calls: list[tuple[int, list[str]]] = []

    def suspend(num_reqs, req_ids):
        calls.append((num_reqs, list(req_ids)))
        return req_ids == ["c"]

    state = NativeDecodeState(suspend)
    assert state.select(["a", "b"], set(), set()) == (False, False)
    assert state.select(["c"], set(), set()) == (True, True)
    # The policy is still asked while a suspended request keeps the batch OFF.
    assert state.select(["c", "d"], set(), set()) == (True, True)
    assert calls == [(2, ["a", "b"]), (1, ["c"]), (2, ["c", "d"])]
    # An empty batch is never suspended and the policy is not asked.
    assert state.select([], {"c", "d"}, set()) == (False, False)
    assert len(calls) == 3


@pytest.mark.parametrize("full_graph", [False, True])
def test_native_execution_uses_native_metadata_graphs_and_no_verify_hook(
    monkeypatch, full_graph
):
    from unittest.mock import Mock

    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.spec_decode.native_decode import NativeDecodeState

    batch = InputBatch.make_dummy(2, 2, InputBuffers(2, 4, torch.device("cpu")))
    batch.req_ids = ["a", "b"]
    scheduled = SchedulerOutput.make_empty()
    scheduled.num_scheduled_tokens = {"a": 1, "b": 1}
    scheduled.total_num_scheduled_tokens = 2
    desc = SimpleNamespace(
        num_tokens=2,
        cg_mode=CUDAGraphMode.FULL if full_graph else CUDAGraphMode.NONE,
        num_ubatches=1,
        num_active_loras=0,
    )
    outputs = torch.zeros(2, 4)
    groups_seen = []
    graphs_seen = []
    native_manager = SimpleNamespace(run_fullgraph=lambda desc: outputs)

    def prepare_attn(*args, **kwargs):
        groups_seen.append(args[4])
        return {}

    runner = SimpleNamespace(
        update_pp_decode_requests=lambda: None,
        finish_requests=lambda s: None,
        free_states=lambda s: None,
        add_requests=lambda s: None,
        update_requests=lambda s: None,
        block_tables=Mock(),
        aux_output_connector=None,
        native_decode_state=NativeDecodeState(lambda num_reqs, req_ids: True),
        native_cudagraph_manager=native_manager,
        cudagraph_manager=object(),
        native_attn_groups=["native"],
        attn_groups=["verify"],
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
        model_state=SimpleNamespace(
            preprocess_state=lambda *a: None,
            prepare_attn=prepare_attn,
            prepare_inputs=lambda *a: {},
        ),
        kv_cache_config=object(),
        req_states=SimpleNamespace(num_computed_tokens=SimpleNamespace(gpu=None)),
        dcp_size=1,
        uses_inputs_embeds=False,
        is_first_pp_rank=True,
        is_last_pp_rank=True,
        speculator=SimpleNamespace(
            prepare_verify_inputs=lambda *a, **kw: pytest.fail(
                "Native batch ran verify hook"
            )
        ),
        speculator_hooks=frozenset({"prepare_verify_inputs"}),
        eplb=Mock(),
        step_timing=Mock(),
        kv_connector=Mock(),
        vllm_config=object(),
        model_config=object(),
        model=lambda **kw: outputs,
        use_aux_hidden_state_outputs=False,
    )

    def dispatch(manager, *args, **kwargs):
        graphs_seen.append(manager)
        return desc, None

    from contextlib import nullcontext

    monkeypatch.setattr(model_runner_module, "dispatch_cg_and_sync_dp", dispatch)
    monkeypatch.setattr(
        model_runner_module, "build_slot_mappings_by_layer", lambda *a: {}
    )
    monkeypatch.setattr(
        model_runner_module, "set_forward_context", lambda *a, **kw: nullcontext()
    )
    GPUModelRunner.execute_model(runner, scheduled)
    assert groups_seen == [["native"]]
    assert graphs_seen == [native_manager]
    assert runner.execute_model_state.input_batch.native_target
    assert runner.execute_model_state.input_batch.skip_draft


def test_native_graphs_capture_even_when_speculative_graphs_are_already_ready(
    monkeypatch,
):
    from contextlib import nullcontext
    from unittest.mock import Mock

    captures = []
    native = SimpleNamespace(
        needs_capture=lambda: True,
        capture=lambda *args, **kwargs: captures.append(args[5]),
    )
    runner = SimpleNamespace(
        cudagraph_manager=SimpleNamespace(needs_capture=lambda: False),
        native_cudagraph_manager=native,
        native_attn_groups=["native"],
        attn_groups=["verify"],
        model_state=SimpleNamespace(
            supports_mm_inputs=False, capture_inner_cudagraphs=lambda *a: None
        ),
        maybe_setup_dummy_loras=lambda *a: nullcontext(),
        lora_config=None,
        input_buffers=object(),
        pcp_manager=None,
        model=object(),
        intermediate_tensors=None,
        block_tables=object(),
        kv_cache_config=object(),
        use_aux_hidden_state_outputs=False,
        speculator=None,
        adaptive_verification=None,
        kv_connector=Mock(),
    )
    monkeypatch.setattr(
        model_runner_module, "freeze_gc_for_cudagraph_capture", nullcontext
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda: (1, 1))
    monkeypatch.setattr(
        model_runner_module, "create_lora_capture_hook", lambda *a: None
    )
    assert GPUModelRunner.needs_cudagraph_capture(runner)
    GPUModelRunner.capture_model(runner, profile_only=True)
    assert captures == [["native"]]


@pytest.mark.parametrize(
    "section, field, value",
    [
        ("scheduler_config", "async_scheduling", True),
        ("cache_config", "enable_prefix_caching", True),
        ("cache_config", "mamba_cache_mode", "align"),
        ("cache_config", "use_replayssm", True),
        ("cache_config", "use_kda_recoverssm", True),
        ("parallel_config", "pipeline_parallel_size", 2),
        ("parallel_config", "data_parallel_size", 2),
        ("parallel_config", "prefill_context_parallel_size", 2),
        ("parallel_config", "decode_context_parallel_size", 2),
        ("parallel_config", "enable_dbo", True),
        ("compilation_config", "cudagraph_mode", CUDAGraphMode.PIECEWISE),
        ("compilation_config", "cudagraph_mode", CUDAGraphMode.FULL_AND_PIECEWISE),
    ],
)
def test_native_decode_rejects_uncoordinated_state_and_cache_reuse(
    section, field, value
):
    from vllm.v1.worker.gpu.spec_decode.native_decode import native_decode_config

    config = SimpleNamespace(
        speculative_config=object(),
        kv_transfer_config=None,
        scheduler_config=SimpleNamespace(async_scheduling=False),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=False,
            mamba_cache_mode="none",
            use_replayssm=False,
            use_kda_recoverssm=False,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_dbo=False,
        ),
    )
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError, match="Native draft suspension requires"):
        native_decode_config(config)


def test_native_decode_preserves_unscheduled_and_preempted_request_lifetimes():
    from vllm.v1.worker.gpu.spec_decode.native_decode import NativeDecodeState

    state = NativeDecodeState(lambda num_reqs, req_ids: 2 <= num_reqs <= 8)
    assert state.select(["a", "b"], set(), set()) == (True, True)
    # Not scheduled is not finished. An unrelated batch may still speculate.
    assert state.select(["other"], set(), {"other"}) == (False, False)
    assert state.select([], set(), set()) == (False, False)
    # Resume the preempted ID without resurrecting its stale draft KV.
    assert state.select(["a"], set(), set()) == (True, True)
    # Merging a speculative request drains its own most recent accepted state.
    assert state.select(["a", "other"], set(), set()) == (True, False)
    assert state.select(["other"], {"a", "b"}, set()) == (True, True)
    assert state.select([], {"other"}, set()) == (False, False)
    assert not state.disabled and not state.pending_state


# --- make_rejection_sampler -------------------------------------------------


class _FakeRejectionSampler:
    def __init__(
        self,
        sampler,
        speculative_config,
        device,
        watermark_key=None,
        draft_logits_num_shards=1,
    ):
        self.args = (sampler, speculative_config, device, watermark_key)
        self.draft_logits_num_shards = draft_logits_num_shards


def _runner(speculator, monkeypatch) -> GPUModelRunner:
    monkeypatch.setattr(model_runner_module, "RejectionSampler", _FakeRejectionSampler)
    monkeypatch.setattr(
        model_runner_module, "speculative_target_watermark_key", lambda config: "key"
    )
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.sampler = object()
    runner.speculator = speculator
    runner.speculative_config = object()
    runner.device = "cpu"
    runner.vllm_config = SimpleNamespace(watermark_config=None)
    return runner


class _Declines:
    def make_rejection_sampler(self, sampler):
        return None


@pytest.mark.parametrize("speculator", [_Declines(), None])
def test_default_rejection_sampler_is_built_unless_the_speculator_supplies_one(
    speculator, monkeypatch
):
    runner = _runner(speculator, monkeypatch)
    built = GPUModelRunner._build_rejection_sampler(runner, 2)
    assert isinstance(built, _FakeRejectionSampler)
    assert built.args == (runner.sampler, runner.speculative_config, "cpu", "key")
    assert built.draft_logits_num_shards == 2


def test_speculator_supplied_rejection_sampler_replaces_the_default(monkeypatch):
    custom = object()
    seen = []

    class Supplies:
        def make_rejection_sampler(self, sampler):
            seen.append(sampler)
            return custom

    runner = _runner(Supplies(), monkeypatch)
    assert GPUModelRunner._build_rejection_sampler(runner, 1) is custom
    assert seen == [runner.sampler]


# --- prepare_verify_inputs / post_accept ------------------------------------


class _MinimalSpeculator(BaseSpeculator):
    def init_cudagraph_manager(self, cudagraph_mode):
        pass

    def capture(self):
        pass

    def propose(self, *args, **kwargs):  # pragma: no cover - not called
        raise NotImplementedError


class _VerifySpeculator(_MinimalSpeculator):
    def prepare_verify_inputs(self, input_batch, model_inputs, *, dummy_run=False):
        pass

    def post_accept(self, input_batch, sampler_output, num_sampled, num_rejected):
        pass


def test_only_overridden_hooks_are_called():
    assert _MinimalSpeculator.__new__(_MinimalSpeculator).overridden_hooks() == set()
    assert NgramGPUSpeculator.__new__(NgramGPUSpeculator).overridden_hooks() == set()
    assert _VerifySpeculator.__new__(_VerifySpeculator).overridden_hooks() == {
        "prepare_verify_inputs",
        "post_accept",
    }
    # A runner built without a speculator calls none.
    assert GPUModelRunner.__new__(GPUModelRunner).speculator_hooks == set()


# --- verify_budget ------------------------------------------------------------


def test_trim_draft_tokens_keeps_the_first_verified_drafts():
    from vllm.v1.worker.gpu.model_runner import trim_draft_tokens

    tokens, drafts = trim_draft_tokens(
        np.array([4, 4, 9], dtype=np.int32),
        np.array([3, 3, 0], dtype=np.int32),
        np.array([1, 3, 0]),
    )
    assert tokens.tolist() == [2, 4, 9]
    assert drafts.tolist() == [1, 3, 0]
    assert drafts.dtype == np.int32
    for bad in ([4, 3, 0], [-1, 3, 0], [1, 3]):
        with pytest.raises(ValueError, match="verify_budget must return"):
            trim_draft_tokens(
                np.array([4, 4, 9], dtype=np.int32),
                np.array([3, 3, 0], dtype=np.int32),
                np.array(bad),
            )


class _BudgetSpeculator(_MinimalSpeculator):
    def __init__(self, width: int):
        self.width = width
        self.calls: list[tuple[list[str], list[int]]] = []

    def verify_budget(self, req_ids, num_draft_tokens):
        self.calls.append((list(req_ids), num_draft_tokens.tolist()))
        return np.minimum(num_draft_tokens, self.width)


def _budget_runner(speculator, adaptive=None) -> GPUModelRunner:
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.decode_query_len = 4
    runner.adaptive_verification = adaptive
    runner.speculator = speculator
    runner.speculator_hooks = speculator.overridden_hooks()
    runner.pcp_manager = None
    runner.use_pp = False
    runner.req_states = SimpleNamespace(
        req_id_to_index={"a": 0, "b": 1, "p": 2},
        num_computed_prefill_tokens=np.array([16, 16, 8], dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.array([16, 16, 40], dtype=np.int32)),
    )
    return runner


def _decode_batch(with_prefill: bool = False) -> SimpleNamespace:
    tokens = {"a": 4, "b": 4}
    if with_prefill:
        tokens["p"] = 6
    return SimpleNamespace(
        num_scheduled_tokens=tokens,
        total_num_scheduled_tokens=sum(tokens.values()),
        scheduled_spec_decode_tokens={"a": [7, 8, 9], "b": [5, 6, 7]},
    )


def _gather(runner: GPUModelRunner, batch: Any) -> tuple[Any, int | None]:
    state, uniform_tok_count = runner.gather_batch_req_state(batch, False)
    assert state is not None
    return state, uniform_tok_count


def test_verify_budget_trims_the_batch_before_it_is_dispatched():
    speculator = _BudgetSpeculator(width=1)
    state, uniform_tok_count = _gather(_budget_runner(speculator), _decode_batch())
    assert speculator.calls == [(["a", "b"], [3, 3])]
    assert state.num_draft_tokens_np is not None
    assert state.num_draft_tokens_np.tolist() == [1, 1]
    assert state.num_scheduled_tokens.tolist() == [2, 2]
    assert state.num_tokens == 4
    # A uniform trim is a uniform decode batch of the narrower width.
    assert uniform_tok_count == 2


def test_verify_budget_keeps_prefill_rows_and_the_batch_when_unchanged():
    speculator = _BudgetSpeculator(width=3)
    state, uniform_tok_count = _gather(_budget_runner(speculator), _decode_batch())
    assert state.num_tokens == 8 and uniform_tok_count == 4
    speculator = _BudgetSpeculator(width=2)
    state, _ = _gather(_budget_runner(speculator), _decode_batch(with_prefill=True))
    assert dict(zip(state.req_ids, state.num_scheduled_tokens.tolist())) == {
        "a": 3,
        "b": 3,
        "p": 6,
    }
    assert state.num_tokens == 12


def test_verify_budget_is_not_asked_without_drafts_or_with_adaptive_verification():
    speculator = _BudgetSpeculator(width=1)
    batch = _decode_batch()
    batch.scheduled_spec_decode_tokens = {}
    _gather(_budget_runner(speculator), batch)
    adaptive = SimpleNamespace(get_num_tokens=lambda tokens, drafts: 6)
    state, _ = _gather(_budget_runner(speculator, adaptive), _decode_batch())
    assert speculator.calls == []
    assert state.num_tokens == 6
