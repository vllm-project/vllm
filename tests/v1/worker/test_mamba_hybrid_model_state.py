# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.worker.gpu.model_states import mamba_hybrid
from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
    MambaHybridAttnMetadata,
    MambaHybridModelState,
)
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState


@pytest.mark.parametrize(
    ("use_flashinfer_replayssm", "expected_state_idx"), [(False, 1), (True, 2)]
)
def test_add_request_seeds_state_with_scoped_block_size(
    use_flashinfer_replayssm: bool, expected_state_idx: int
) -> None:
    state = object.__new__(MambaHybridModelState)
    state.rope_state = None
    state.prompt_embeds_state = None
    state.cache_config = SimpleNamespace(
        block_size=16,
        mamba_block_size=8,
        mamba_cache_mode="align",
    )
    state._needs_prefix_state_migration = True
    state._use_flashinfer_replayssm = use_flashinfer_replayssm
    state.num_accepted_tokens_gpu = torch.full((2,), 9, dtype=torch.int32)
    state._mamba_state_idx_gpu = torch.full((2,), -1, dtype=torch.int32)
    state._mamba_prev_last_scheduled_idx_gpu = torch.full((2,), 9, dtype=torch.int32)

    state.add_request(1, Mock(num_computed_tokens=17))

    assert state.num_accepted_tokens_gpu.tolist() == [9, 1]
    assert state._mamba_state_idx_gpu.tolist() == [-1, expected_state_idx]
    assert state._mamba_prev_last_scheduled_idx_gpu.tolist() == [9, -1]


def test_all_spec_tracks_previous_scheduled_page_by_request() -> None:
    state = object.__new__(MambaHybridModelState)
    state.cache_config = SimpleNamespace(mamba_block_size=16)
    state.vllm_config = SimpleNamespace(num_speculative_tokens=3)
    state._mamba_prev_last_scheduled_idx_gpu = torch.full((4,), -1, dtype=torch.int32)

    first_batch = SimpleNamespace(
        num_reqs=2,
        idx_mapping=torch.tensor([2, 0], dtype=torch.int32),
        seq_lens=torch.tensor([18, 33], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
    )
    first_prev = state._stage_prev_last_scheduled_idx(first_batch, num_reqs=3)

    assert first_prev is not None
    assert first_prev.tolist() == [-1, -1, -1]
    assert state._mamba_prev_last_scheduled_idx_gpu.tolist() == [2, -1, 1, -1]

    # Request 2 previously started with N=14 tokens and scheduled q=4, so its
    # state window is anchored at page 1. Accepting one token leaves N=15 and
    # logical last-computed page 0, which must not replace that physical anchor.
    logical_last_computed = (15 - 1) // 16
    state._mamba_prev_last_scheduled_idx_gpu[1] = 7
    second_batch = SimpleNamespace(
        num_reqs=3,
        idx_mapping=torch.tensor([0, 2, 1], dtype=torch.int32),
        seq_lens=torch.tensor([34, 19, 5], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4, 8, 9], dtype=torch.int32),
    )
    second_prev = state._stage_prev_last_scheduled_idx(second_batch, num_reqs=4)

    assert second_prev is not None
    assert second_prev.tolist() == [2, 1, 7, -1]
    assert second_prev[1].item() != logical_last_computed
    # The one-token row may consume its old anchor in this step, but must clear
    # it rather than advertising a speculative window to the following step.
    assert state._mamba_prev_last_scheduled_idx_gpu.tolist() == [2, -1, 1, -1]


def test_previous_scheduled_page_is_passed_only_to_mamba2() -> None:
    prev_last_scheduled_idx = torch.tensor([3, 5], dtype=torch.int32)
    metadata = MambaHybridAttnMetadata(
        is_prefilling=torch.zeros(2, dtype=torch.bool),
        prev_last_scheduled_idx=prev_last_scheduled_idx,
    )

    mamba2_args = metadata.get_extra_attn_kwargs(
        Mock(spec=Mamba2AttentionMetadataBuilder), 2
    )
    gdn_args = metadata.get_extra_attn_kwargs(Mock(spec=GDNAttentionMetadataBuilder), 2)

    mamba2_prev = mamba2_args["prev_last_scheduled_idx"]
    assert torch.equal(mamba2_prev, prev_last_scheduled_idx)
    assert mamba2_prev.data_ptr() == prev_last_scheduled_idx.data_ptr()
    assert "prev_last_scheduled_idx" not in gdn_args


@pytest.mark.parametrize(
    ("computed", "scheduled", "drafts", "prefilling", "expected_prefilling"),
    [
        (256, 4, 3, True, False),  # Cached prompt tail plus placeholders.
        (1, 4, 3, True, False),  # Rejection can exceed the cached prefix length.
        (256, 1, 0, True, False),
        (0, 4, 3, True, True),  # No prior state: must stay a prefill.
        (256, 4, 0, True, True),
        (256, 4, 3, False, False),
    ],
)
def test_prepare_attn_forwards_positions_and_stages_replayssm_prefill(
    monkeypatch: pytest.MonkeyPatch,
    computed,
    scheduled,
    drafts,
    prefilling,
    expected_prefilling,
) -> None:
    state = object.__new__(MambaHybridModelState)
    state.vllm_config = SimpleNamespace(num_speculative_tokens=3)
    state.max_model_len = 8192
    state._align_mode = False
    state._use_flashinfer_replayssm = True
    state._is_prefilling_gpu = torch.zeros(1, dtype=torch.bool)
    state.num_accepted_tokens_gpu = torch.ones(1, dtype=torch.int32)
    state._get_mamba_group_info = Mock(return_value=([], None))
    state._ensure_mamba_postprocess_ctx = Mock()
    state._mamba_prev_last_scheduled_idx_gpu = None
    state.recoverssm = None

    positions = torch.arange(computed, computed + scheduled, dtype=torch.int64)
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=scheduled,
        num_reqs_after_padding=1,
        num_tokens_after_padding=scheduled,
        query_start_loc_np=torch.tensor([0, scheduled], dtype=torch.int32).numpy(),
        query_start_loc=torch.tensor([0, scheduled], dtype=torch.int32),
        num_scheduled_tokens=torch.tensor([scheduled], dtype=torch.int32).numpy(),
        num_draft_tokens_per_req=torch.tensor([drafts], dtype=torch.int32).numpy(),
        idx_mapping=torch.tensor([0], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([computed + scheduled]),
        seq_lens=torch.tensor([computed + scheduled]),
        is_prefilling_np=torch.tensor([prefilling]).numpy(),
        dcp_local_seq_lens=None,
        positions=positions,
        prompt_lens=torch.tensor([1024], dtype=torch.int32),
    )
    expected_metadata = {"layer": object()}
    build_attn_metadata = Mock(return_value=expected_metadata)
    monkeypatch.setattr(mamba_hybrid, "build_attn_metadata", build_attn_metadata)

    metadata = state.prepare_attn(
        input_batch=input_batch,
        cudagraph_mode=CUDAGraphMode.NONE,
        block_tables=(),
        slot_mappings=torch.empty(0, dtype=torch.int64),
        attn_groups=[],
        kv_cache_config=Mock(),
    )

    assert metadata is expected_metadata
    assert build_attn_metadata.call_args.kwargs["positions"] is positions
    assert state._is_prefilling_gpu.item() == expected_prefilling


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
def test_flashinfer_replayssm_prefix_uses_original_accepted_counts() -> None:
    state = object.__new__(MambaHybridModelState)
    state._align_mode = True
    state._needs_prefix_state_migration = True
    state._use_flashinfer_replayssm = True
    state.recoverssm = None
    state.num_accepted_tokens_gpu = torch.ones(4, dtype=torch.int32, device="cuda")
    state._mamba_state_idx_gpu = torch.zeros(4, dtype=torch.int32, device="cuda")
    state._is_prefilling_gpu = torch.zeros(4, dtype=torch.bool, device="cuda")
    state._replayssm_query_start_loc = torch.tensor(
        [0, 4], dtype=torch.int32, device="cuda"
    )
    replayssm = Mock(materialize_prefixes=True)
    accepted_snapshot = torch.zeros(4, dtype=torch.int32, device="cuda")
    ctx = Mock(
        is_initialized=True,
        replayssm=replayssm,
        num_accepted_tokens_snapshot=accepted_snapshot,
    )

    def normalize_live(*_args) -> None:
        accepted_snapshot.copy_(state.num_accepted_tokens_gpu)
        state.num_accepted_tokens_gpu[2] = 1

    ctx.run_fused_postprocess_align.side_effect = normalize_live
    state._mamba_ctx = ctx

    state.postprocess_state(
        torch.tensor([2], dtype=torch.int32, device="cuda"),
        torch.tensor([3], dtype=torch.int32, device="cuda"),
        num_computed_tokens=torch.tensor([0, 0, 8, 0], device="cuda"),
    )

    kwargs = replayssm.postprocess.call_args.kwargs
    assert kwargs["num_accepted_tokens"] is accepted_snapshot
    assert accepted_snapshot[2].item() == 3
    assert state.num_accepted_tokens_gpu[2].item() == 1
    assert state._replayssm_query_start_loc is None


def test_recoverssm_commits_accepted_window_without_align_mapping() -> None:
    state = RecoverSSMState()
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = None
    num_sampled = torch.tensor([3, 1], dtype=torch.int32)
    num_accepted_tokens = torch.ones(2, dtype=torch.int32)
    group = SimpleNamespace(layer_names=["layer"])

    state.record_step({"layer": metadata}, [[group]], for_capture=False)
    state.commit_step(
        num_sampled,
        None,
        state_indices=None,
        num_accepted_tokens=num_accepted_tokens,
    )
    state.commit_step(
        num_sampled,
        None,
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
