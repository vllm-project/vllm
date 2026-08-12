# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mamba_attn import (
    ReplaySSMAlignCommitMetadata,
    ReplaySSMSpecMetadata,
)
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(("num_sampled", "expected_value"), [(0, 1), (3, 3)])
def test_postprocess_state_scalar_with_int32_mapping(
    num_sampled: int, expected_value: int
) -> None:
    state = object.__new__(MambaHybridModelState)
    state.num_accepted_tokens_gpu = torch.full(
        (4,), 9, dtype=torch.int32, device="cuda"
    )
    state.cache_config = SimpleNamespace(use_replayssm_spec=False)
    state._align_mode = False
    state._replayssm_align = False
    state._mamba_ctx = None
    idx_mapping = torch.tensor([2, -1, 0], dtype=torch.int32, device="cuda")

    state.postprocess_state(idx_mapping, num_sampled)

    expected = torch.tensor(
        [expected_value, 9, expected_value, 9], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_close(state.num_accepted_tokens_gpu, expected)


def test_replayssm_commits_accepted_window_after_v2_sampling() -> None:
    state = object.__new__(MambaHybridModelState)
    state.cache_config = SimpleNamespace(use_replayssm_spec=True)
    metadata = Mock(spec=ReplaySSMSpecMetadata)
    metadata.get_replayssm_align_commit_metadata.return_value = None
    num_sampled = torch.tensor([3, 1], dtype=torch.int32)

    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    state._replayssm_align = False
    state._replayssm_step = (metadata,)
    state._commit_replayssm_state(num_sampled, idx_mapping)

    metadata.commit_replayssm_state.assert_called_once_with(num_sampled)
    assert state._replayssm_step is None


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("mixed_batch", [False, True])
def test_replayssm_align_tracks_final_running_state_and_neutralizes_copy_bias(
    mixed_batch: bool,
) -> None:
    state = object.__new__(MambaHybridModelState)
    state.cache_config = SimpleNamespace(use_replayssm_spec=True)
    state._replayssm_align = True
    state._align_mode = False
    state._mamba_ctx = None
    state._mamba_state_idx_gpu = torch.full((5,), -1, dtype=torch.int32, device="cuda")
    state._replayssm_committed_gpu = torch.zeros(5, dtype=torch.bool, device="cuda")
    state.num_accepted_tokens_gpu = torch.full(
        (5,), 9, dtype=torch.int32, device="cuda"
    )
    metadata = Mock(spec=ReplaySSMSpecMetadata)
    metadata.get_replayssm_align_commit_metadata.return_value = (
        ReplaySSMAlignCommitMetadata(
            num_spec_decodes=1 if mixed_batch else 2,
            request_indices=(
                torch.tensor([1], dtype=torch.int32, device="cuda")
                if mixed_batch
                else None
            ),
            num_computed_tokens=torch.tensor([6, 7], dtype=torch.int32, device="cuda"),
            block_size=8,
            block_table=torch.zeros((2, 4), dtype=torch.int32, device="cuda"),
        )
    )
    num_sampled = torch.tensor([2, 3], dtype=torch.int32, device="cuda")
    idx_mapping = torch.tensor([3, 1], dtype=torch.int32, device="cuda")

    state._replayssm_step = (metadata,)

    state.postprocess_state(idx_mapping, num_sampled)

    expected_state_indices = [-1, 1, -1, -1 if mixed_batch else 1, -1]
    assert state._mamba_state_idx_gpu.tolist() == expected_state_indices
    expected_accepted = [9, 1, 9, 2 if mixed_batch else 1, 9]
    assert state.num_accepted_tokens_gpu.tolist() == expected_accepted
    assert not state._replayssm_committed_gpu.any().item()
