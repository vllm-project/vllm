# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import vllm.utils.gpu_sync_debug as gsd
from tests.v1.attention.test_gdn_metadata_builder import (
    BLOCK_SIZE,
    DEVICE,
    _create_gdn_builder,
)
from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.config.compilation import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.worker.gpu.model_states import mamba_hybrid
from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
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

    state.add_request(1, Mock(num_computed_tokens=17))

    assert state.num_accepted_tokens_gpu.tolist() == [9, 1]
    assert state._mamba_state_idx_gpu.tolist() == [-1, expected_state_idx]


@pytest.mark.parametrize(
    "staging_device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
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
    staging_device,
) -> None:
    state = object.__new__(MambaHybridModelState)
    state.vllm_config = SimpleNamespace(num_speculative_tokens=3)
    state.model_config = SimpleNamespace(max_model_len=8192)
    state._align_mode = False
    state._use_flashinfer_replayssm = True
    if staging_device == "cpu":
        monkeypatch.setattr("vllm.utils.torch_utils.PIN_MEMORY", False)
    state._is_prefilling_gpu = torch.zeros(1, dtype=torch.bool, device=staging_device)
    state._mamba_ctx = None
    state.num_accepted_tokens_gpu = torch.ones(1, dtype=torch.int32)
    state._get_mamba_group_info = Mock(return_value=([], None))
    state._ensure_mamba_postprocess_ctx = Mock()
    state.recoverssm = None

    positions = torch.arange(computed, computed + scheduled, dtype=torch.int64)
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=scheduled,
        num_reqs_after_padding=1,
        num_tokens_after_padding=scheduled,
        max_query_len=None,
        prefill_runs_as_decode_np=np.array(
            [prefilling and computed > 0 and drafts > 0]
        ),
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

    prepare_attn = state.prepare_attn
    if staging_device == "cuda":
        monkeypatch.setattr(gsd, "_SYNC_CHECK_MODE", "error")
        monkeypatch.setattr(gsd, "_sync_check_enabled", True)
        gsd.enable_gpu_sync_check()
        prepare_attn = gsd.with_gpu_sync_check(prepare_attn)

    metadata = prepare_attn(
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_replayssm_prefill_staging_retains_inflight_masks(monkeypatch) -> None:
    """A ramp can stage another mask while the previous H2D is still queued."""
    state = object.__new__(MambaHybridModelState)
    state.vllm_config = SimpleNamespace(num_speculative_tokens=3)
    state.model_config = SimpleNamespace(max_model_len=8192)
    state._align_mode = False
    state._use_flashinfer_replayssm = True
    state._is_prefilling_gpu = torch.zeros(2, dtype=torch.bool, device="cuda")
    state._get_mamba_group_info = Mock(return_value=([], None))
    state._ensure_mamba_postprocess_ctx = Mock()
    state.num_accepted_tokens_gpu = torch.ones(2, dtype=torch.int32, device="cuda")
    state.recoverssm = None
    query_start_loc = torch.tensor([0, 8, 16], dtype=torch.int32)
    input_batch = SimpleNamespace(
        num_reqs=2,
        num_tokens=16,
        num_reqs_after_padding=2,
        num_tokens_after_padding=16,
        max_query_len=8,
        prefill_runs_as_decode_np=None,
        query_start_loc_np=query_start_loc.numpy(),
        query_start_loc=query_start_loc.cuda(),
        num_scheduled_tokens=np.array([8, 8], dtype=np.int32),
        num_draft_tokens_per_req=np.array([0, 0], dtype=np.int32),
        idx_mapping=torch.arange(2, device="cuda"),
        seq_lens_cpu_upper_bound=torch.tensor([264, 264]),
        seq_lens=torch.tensor([264, 264], device="cuda"),
        is_prefilling_np=np.array([False, True]),
        dcp_local_seq_lens=None,
        positions=torch.arange(16, device="cuda"),
        prompt_lens=torch.tensor([1024, 1024], device="cuda"),
    )
    monkeypatch.setattr(mamba_hybrid, "build_attn_metadata", Mock(return_value={}))
    kwargs = dict(
        cudagraph_mode=CUDAGraphMode.NONE,
        block_tables=(),
        slot_mappings=torch.empty(0, dtype=torch.int64, device="cuda"),
        attn_groups=[],
        kv_cache_config=Mock(),
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    first = torch.empty(2, dtype=torch.bool, device="cuda")
    second = torch.empty_like(first)
    blocker = torch.cuda.Event()
    with torch.cuda.stream(stream):
        # Warm the sleep kernel and prepare allocations on this stream before
        # queuing the deliberate GPU backlog.
        torch.cuda._sleep(1)
        state.prepare_attn(input_batch=input_batch, **kwargs)
        stream.synchronize()
        torch.cuda._sleep(1_000_000_000)
        blocker.record(stream)
        state.prepare_attn(input_batch=input_batch, **kwargs)
        first.copy_(state._is_prefilling_gpu)
        input_batch.is_prefilling_np = np.array([True, False])
        state.prepare_attn(input_batch=input_batch, **kwargs)
        second.copy_(state._is_prefilling_gpu)
        # Both CPU masks were staged before the first DMA could begin.
        assert not blocker.query(), "GPU backlog ended before the staging overlap"
    stream.synchronize()
    assert first.tolist() == [False, True]
    assert second.tolist() == [True, False]


def test_padded_prompt_tail_builds_as_spec_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-token prompt tail over prior state, padded with K placeholder
    drafts (e.g. a P/D decode-node arrival), must reach the GDN builder as a
    spec-decode row. Built as a prefill, the placeholder tokens are folded into
    the recurrent state and can't be rolled back.
    """
    k = 3
    state = object.__new__(MambaHybridModelState)
    state.vllm_config = SimpleNamespace(num_speculative_tokens=k)
    state.model_config = SimpleNamespace(max_model_len=8192)
    state._align_mode = False
    state._needs_prefix_state_migration = False
    state._use_flashinfer_replayssm = False
    state.recoverssm = None
    state.num_accepted_tokens_gpu = torch.ones(4, dtype=torch.int32)

    # A verify decode, a padded prompt tail (128 of 129 prompt tokens already
    # computed), and a fresh prompt chunk of the same length.
    query_lens = [k + 1] * 3
    seq_lens = [50, 128 + k + 1, k + 1]
    is_prefilling = [False, True, True]
    query_start_loc = np.array([0, 4, 8, 12], dtype=np.int32)
    input_batch = SimpleNamespace(
        num_reqs=3,
        num_tokens=12,
        num_reqs_after_padding=3,
        num_tokens_after_padding=12,
        idx_mapping=torch.arange(3),
        query_start_loc_np=query_start_loc,
        query_start_loc=torch.from_numpy(query_start_loc),
        num_scheduled_tokens=np.array(query_lens, dtype=np.int32),
        num_draft_tokens_per_req=np.array([k, k, 0], dtype=np.int32),
        max_query_len=None,
        seq_lens_cpu_upper_bound=torch.tensor(seq_lens, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        is_prefilling_np=np.array(is_prefilling),
        prefill_runs_as_decode_np=np.array([False, True, False]),
        dcp_local_seq_lens=None,
        positions=torch.zeros(12, dtype=torch.int64),
        prompt_lens=None,
    )
    build_attn_metadata = Mock(return_value={})
    monkeypatch.setattr(mamba_hybrid, "build_attn_metadata", build_attn_metadata)
    state.prepare_attn(
        input_batch=input_batch,
        cudagraph_mode=CUDAGraphMode.NONE,
        block_tables=(),
        slot_mappings=torch.empty(0, dtype=torch.int64),
        attn_groups=[],
        kv_cache_config=Mock(),
    )
    mamba_metadata = build_attn_metadata.call_args.kwargs[
        "model_specific_attn_metadata"
    ]
    assert mamba_metadata.is_prefilling.tolist() == [False, False, True]

    builder = _create_gdn_builder(num_speculative_tokens=k)
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens), BLOCK_SIZE, DEVICE
    ).replace(**mamba_metadata.get_extra_common_attn_kwargs(0, 3))
    meta = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_accepted_tokens=mamba_metadata.num_accepted_tokens,
        num_decode_draft_tokens_cpu=mamba_metadata.num_decode_draft_tokens_cpu,
    )

    # Only the fresh prompt chunk needs the prefill kernels.
    assert meta.num_spec_decodes == 2
    assert meta.num_prefills == 1
    assert meta.num_prefill_tokens == k + 1


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
