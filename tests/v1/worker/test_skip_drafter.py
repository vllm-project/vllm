# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A step whose next round verifies no drafts (candidate budget 1) does not run
the drafter: ``propose`` is not called, the batch's draft rows are -1, and the
rest of the step (``observe_step``, the draft handoff) still runs."""

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.worker.gpu.model_runner as model_runner_module
from tests.v1.worker.test_gpu_speculator_observe_step import (
    _batch,
    _Observer,
    _run_step,
    _runner,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor


class _Drafter(_Observer):
    def __init__(self):
        super().__init__()
        self.num_proposed = 0

    def propose(self, input_batch, *args, **kwargs):
        self.num_proposed += 1
        return super().propose(input_batch, *args, **kwargs)


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
    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL, num_tokens=2, num_reqs=2
    )
    _run_step(runner, batch, desc, num_spec_tokens=num_spec_tokens)
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
    assert len(drafter.seen) == 1 and drafter.seen[0].num_spec_tokens == 0
    assert [d.shape for d in handed_off] == [torch.Size([2, 0])]


def test_a_drafting_step_still_runs_the_drafter(monkeypatch):
    drafter, runner, handed_off = _step(monkeypatch, num_spec_tokens=3)
    assert drafter.num_proposed == 1
    assert torch.all(runner.req_states.draft_tokens[:2] == 0)
    assert len(drafter.seen) == 1
    assert [d.shape for d in handed_off] == [torch.Size([2, 3])]


def test_budget_1_runs_the_drafter_while_another_dp_rank_drafts(monkeypatch):
    """The drafter's forward runs in lockstep across DP ranks: a rank without
    drafts to propose still runs it unless every rank skips."""
    drafter, _, _ = _step(monkeypatch, num_spec_tokens=0, all_dp_ranks_skip=False)
    assert drafter.num_proposed == 1
