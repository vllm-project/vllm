# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(("num_sampled", "expected_value"), [(0, 1), (3, 3)])
def test_postprocess_state_scalar_with_int32_mapping(
    num_sampled: int, expected_value: int
) -> None:
    state = object.__new__(MambaHybridModelState)
    state.num_accepted_tokens_gpu = torch.full(
        (4,), 9, dtype=torch.int32, device="cuda"
    )
    state._align_mode = False
    state._needs_prefix_state_migration = False
    state._use_flashinfer_replayssm = False
    state.recoverssm = None
    state._mamba_ctx = None
    idx_mapping = torch.tensor([2, -1, 0], dtype=torch.int32, device="cuda")

    state.postprocess_state(idx_mapping, num_sampled)

    expected = torch.tensor(
        [expected_value, 9, expected_value, 9], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_close(state.num_accepted_tokens_gpu, expected)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_flashinfer_replayssm_none_postprocess_skips_prefix_migration() -> None:
    state = object.__new__(MambaHybridModelState)
    state._align_mode = False
    state._needs_prefix_state_migration = False
    state._use_flashinfer_replayssm = True
    state.recoverssm = None
    state.num_accepted_tokens_gpu = torch.ones(4, dtype=torch.int32, device="cuda")
    state._replayssm_live_cols_gpu = torch.zeros(
        4, dtype=torch.int32, device="cuda"
    )
    state._is_prefilling_gpu = torch.zeros(4, dtype=torch.bool, device="cuda")
    replayssm = Mock()
    ctx = Mock(
        is_initialized=True,
        replayssm=replayssm,
        materialize_src_cols=torch.full(
            (4,), -1, dtype=torch.int32, device="cuda"
        ),
        materialize_dst_cols=torch.full(
            (4,), -1, dtype=torch.int32, device="cuda"
        ),
        materialize_token_counts=torch.zeros(4, dtype=torch.int32, device="cuda"),
        block_size=1024,
    )
    state._mamba_ctx = ctx
    idx_mapping = torch.tensor([2], dtype=torch.int32, device="cuda")
    num_computed = torch.tensor([0, 0, 20, 0], dtype=torch.int32, device="cuda")
    query_start_loc = torch.tensor([0, 4], dtype=torch.int32, device="cuda")

    state.postprocess_state(
        idx_mapping,
        2,
        num_computed_tokens=num_computed,
        query_start_loc=query_start_loc,
    )

    ctx.run_fused_postprocess_align.assert_not_called()
    assert replayssm.postprocess_and_materialize.call_count == 1
    kwargs = replayssm.postprocess_and_materialize.call_args.kwargs
    assert kwargs["num_accepted_tokens"] is state.num_accepted_tokens_gpu
    assert kwargs["live_cols"] is state._replayssm_live_cols_gpu


def test_recoverssm_commits_accepted_window_after_v2_sampling() -> None:
    state = RecoverSSMState()
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = None
    num_sampled = torch.tensor([3, 1], dtype=torch.int32)
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    num_accepted_tokens = torch.ones(2, dtype=torch.int32)
    group = SimpleNamespace(layer_names=["layer"])

    state.record_step({"layer": metadata}, [[group]], for_capture=False)
    state.commit_step(
        num_sampled,
        idx_mapping,
        state_indices=None,
        num_accepted_tokens=num_accepted_tokens,
    )
    state.commit_step(
        num_sampled,
        idx_mapping,
        state_indices=None,
        num_accepted_tokens=num_accepted_tokens,
    )

    metadata.commit_recoverssm_state.assert_called_once_with(num_sampled)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_recoverssm_align_tracks_mixed_batch_state_and_neutralizes_copy_bias() -> None:
    state = object.__new__(MambaHybridModelState)
    state._align_mode = True
    state._needs_prefix_state_migration = True
    state._use_flashinfer_replayssm = False
    state._mamba_ctx = None
    state._mamba_state_idx_gpu = torch.full((5,), -1, dtype=torch.int32, device="cuda")
    state.recoverssm = RecoverSSMState()
    state.num_accepted_tokens_gpu = torch.full(
        (5,), 9, dtype=torch.int32, device="cuda"
    )
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = RecoverSSMPostprocessMetadata(
        num_spec_decodes=1,
        request_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
        num_computed_tokens=torch.tensor([6, 7], dtype=torch.int32, device="cuda"),
        block_size=8,
        block_table=torch.zeros((2, 4), dtype=torch.int32, device="cuda"),
    )
    num_sampled = torch.tensor([2, 3], dtype=torch.int32, device="cuda")
    idx_mapping = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    group = SimpleNamespace(layer_names=["layer"])

    state.recoverssm.record_step({"layer": metadata}, [[group]], for_capture=False)

    state.postprocess_state(idx_mapping, num_sampled)

    expected_state_indices = [-1, 1, -1, -1, -1]
    assert state._mamba_state_idx_gpu.tolist() == expected_state_indices
    expected_accepted = [9, 1, 9, 2, 9]
    assert state.num_accepted_tokens_gpu.tolist() == expected_accepted
