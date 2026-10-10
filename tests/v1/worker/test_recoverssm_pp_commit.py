# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RecoverSSM commit deferred to the PP postprocess on non-last ranks.

A non-last pipeline-parallel rank commits a step ``pp_size`` steps later, after
the other micro-batch has recorded its own step. ``detach_step`` hands the step
over with its commit index tensors cloned, and ``commit_step(step=...)`` commits
that step, skipping rows whose request was freed in between.
"""

from dataclasses import dataclass, field

import torch

from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState


@dataclass
class _Commit:
    state_indices: torch.Tensor
    query_start_loc: torch.Tensor


@dataclass
class _Metadata:
    num_spec_decodes: int
    recoverssm_commit: _Commit | None
    seen: list = field(default_factory=list)

    def commit_recoverssm_state(self, num_accepted_tokens):
        self.seen.append((num_accepted_tokens.clone(), self.recoverssm_commit))
        return None


def _state_with_step(md):
    state = RecoverSSMState()
    state._step = (md,)
    return state


def test_detach_clones_commit_tensors_and_clears_the_record():
    commit = _Commit(torch.tensor([[3], [5]]), torch.tensor([0, 2, 4]))
    state = _state_with_step(_Metadata(2, commit))
    step = state.detach_step()
    assert state._step is None
    # A later step rebuilding the persistent buffers must not change the record.
    commit.state_indices.fill_(-7)
    commit.query_start_loc.fill_(-7)
    assert step[0].recoverssm_commit.state_indices.tolist() == [[3], [5]]
    assert step[0].recoverssm_commit.query_start_loc.tolist() == [0, 2, 4]


def test_detach_without_a_recorded_step_returns_none():
    assert RecoverSSMState().detach_step() is None


def test_deferred_commit_uses_the_given_step_not_the_current_one():
    deferred = _Metadata(2, _Commit(torch.tensor([[1], [2]]), torch.tensor([0, 2, 4])))
    current = _Metadata(1, _Commit(torch.tensor([[9]]), torch.tensor([0, 2])))
    state = _state_with_step(current)
    num_sampled = torch.tensor([2, 1, 0, 0], dtype=torch.int32)  # padded past 2 rows
    state.commit_step(
        num_sampled,
        torch.tensor([4, 6]),
        state_indices=None,
        num_accepted_tokens=torch.ones(8, dtype=torch.int32),
        step=(deferred,),
    )
    assert len(deferred.seen) == 1 and not current.seen
    assert state._step == (current,)  # the current record stays for its own commit
    assert deferred.seen[0][0].tolist() == [2, 1, 0, 0]


def test_deferred_commit_skips_rows_freed_in_between():
    md = _Metadata(
        3, _Commit(torch.tensor([[1], [2], [3]]), torch.tensor([0, 2, 4, 6]))
    )
    num_sampled = torch.tensor([2, 2, 1, 0], dtype=torch.int32)  # 3 rows + padding
    RecoverSSMState().commit_step(
        num_sampled,
        torch.tensor([4, -1, 6]),  # the second request was freed
        state_indices=None,
        num_accepted_tokens=torch.ones(8, dtype=torch.int32),
        step=(md,),
    )
    assert md.seen[0][0].tolist() == [2, 0, 1, 0]
    assert num_sampled.tolist() == [2, 2, 1, 0]  # the broadcast buffer is not modified
