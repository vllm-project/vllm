# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A step whose next round verifies no drafts (candidate budget 1) does not run
the drafter: ``propose`` is not called, the batch's draft rows are -1, and the
draft handoff still runs."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.model_runner as model_runner_module
import vllm.v1.worker.gpu.spec_decode.round_runner as round_runner
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.round import SpecResources
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator


class _Drafter(BaseSpeculator):
    def __init__(self):
        self.num_proposed = 0

    def init_cudagraph_manager(self, cudagraph_mode):
        pass

    def capture(self):
        pass

    def propose(self, input_batch, *args, **kwargs):
        self.num_proposed += 1
        return torch.zeros(input_batch.num_reqs, 3, dtype=torch.int64)


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


def _runner(speculator: _Drafter, monkeypatch) -> GPUModelRunner:
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.is_last_pp_rank = True
    runner.pcp_manager = runner.pp_handler = runner.aux_output_connector = None
    runner.speculator = speculator
    runner.speculator_hooks = speculator.overridden_hooks()
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
    num_spec_tokens: int,
) -> None:
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
    )
    GPUModelRunner.sample_tokens(runner, None)


def _step(monkeypatch, num_spec_tokens: int, all_dp_ranks_skip: bool | None = None):
    drafter = _Drafter()
    runner = _runner(drafter, monkeypatch)
    runner.req_states.draft_tokens.fill_(7)
    handed_off: list[torch.Tensor] = []
    runner.draft_tokens_handler = SimpleNamespace(
        set_draft_tokens=lambda batch, drafts: handed_off.append(drafts.clone())
    )
    if all_dp_ranks_skip is not None:
        state = model_runner_module.ExecuteModelState
        dp_sync_state = SimpleNamespace(skip_drafts=all_dp_ranks_skip)
        monkeypatch.setattr(
            model_runner_module,
            "ExecuteModelState",
            lambda **kw: state(**{**kw, "dp_sync_state": dp_sync_state}),
        )
    batch = _batch(2, 2, [1, 1], None, [False, False], [3, 4])
    _run_step(runner, batch, num_spec_tokens=num_spec_tokens)
    return drafter, runner, handed_off


@pytest.mark.parametrize("all_dp_ranks_skip", [None, True])
def test_budget_1_does_not_run_the_drafter(monkeypatch, all_dp_ranks_skip):
    drafter, runner, handed_off = _step(
        monkeypatch, num_spec_tokens=0, all_dp_ranks_skip=all_dp_ranks_skip
    )
    assert drafter.num_proposed == 0
    # The batch's rows hold no drafts; rows outside the batch are untouched.
    assert torch.all(runner.req_states.draft_tokens[:2] == -1)
    assert torch.all(runner.req_states.draft_tokens[2:] == 7)
    assert [d.shape for d in handed_off] == [torch.Size([2, 0])]


def test_a_drafting_step_still_runs_the_drafter(monkeypatch):
    drafter, runner, handed_off = _step(monkeypatch, num_spec_tokens=3)
    assert drafter.num_proposed == 1
    assert torch.all(runner.req_states.draft_tokens[:2] == 0)
    assert [d.shape for d in handed_off] == [torch.Size([2, 3])]


def test_budget_1_runs_the_drafter_while_another_dp_rank_drafts(monkeypatch):
    """The drafter's forward runs in lockstep across DP ranks: a rank without
    drafts to propose still runs it unless every rank skips."""
    drafter, _, _ = _step(monkeypatch, num_spec_tokens=0, all_dp_ranks_skip=False)
    assert drafter.num_proposed == 1
