# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.v1.worker.gpu.spec_decode.speculator import select_estimator_observations


def test_select_estimator_observations_keeps_unmasked_rows():
    idx_mapping = torch.tensor([4, 7, 9])
    num_sampled = torch.tensor([1, 2, 3])
    num_rejected = torch.tensor([4, 0, 1])

    selected = select_estimator_observations(
        idx_mapping,
        num_sampled,
        num_rejected,
        skip_np=np.array([True, False, True]),
    )

    assert selected is not None
    keep_idx, keep_sampled, keep_rejected = selected
    assert torch.equal(keep_idx, torch.tensor([7]))
    assert torch.equal(keep_sampled, torch.tensor([2]))
    assert torch.equal(keep_rejected, torch.tensor([0]))


def test_select_estimator_observations_returns_none_when_all_skipped():
    idx_mapping = torch.tensor([4, 7])
    num_sampled = torch.tensor([1, 2])
    num_rejected = torch.tensor([4, 0])

    selected = select_estimator_observations(
        idx_mapping,
        num_sampled,
        num_rejected,
        skip_np=np.array([True, True]),
    )
    assert selected is None


def test_select_estimator_observations_passthrough_when_none_skipped():
    idx_mapping = torch.tensor([4, 7])
    num_sampled = torch.tensor([1, 2])
    num_rejected = torch.tensor([4, 0])

    selected = select_estimator_observations(
        idx_mapping, num_sampled, num_rejected, skip_np=np.array([False, False])
    )
    assert selected is not None
    keep_idx, keep_sampled, keep_rejected = selected
    assert keep_idx is idx_mapping
    assert keep_sampled is num_sampled
    assert keep_rejected is num_rejected


def test_select_estimator_observations_passthrough_when_skip_is_none():
    idx_mapping = torch.tensor([4, 7])
    num_sampled = torch.tensor([1, 2])
    num_rejected = torch.tensor([4, 0])

    selected = select_estimator_observations(
        idx_mapping, num_sampled, num_rejected, skip_np=None
    )
    assert selected is not None
    assert selected[0] is idx_mapping
    assert selected[1] is num_sampled
    assert selected[2] is num_rejected
