# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

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
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState
from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext


def test_prepare_attn_forwards_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    state = object.__new__(MambaHybridModelState)
    state.vllm_config = SimpleNamespace(num_speculative_tokens=0)
    state.model_config = SimpleNamespace(max_model_len=8192)
    state._align_mode = False
    state.recoverssm = None

    positions = torch.tensor([1536], dtype=torch.int64)
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=1,
        num_reqs_after_padding=1,
        num_tokens_after_padding=1,
        query_start_loc_np=torch.tensor([0, 1], dtype=torch.int32).numpy(),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        num_scheduled_tokens=torch.tensor([1], dtype=torch.int32),
        max_query_len=None,
        seq_lens_cpu_upper_bound=torch.tensor([1537], dtype=torch.int32),
        seq_lens=torch.tensor([1537], dtype=torch.int32),
        is_prefilling_np=torch.tensor([False]).numpy(),
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
        prefill_len_np=np.array([40, 129, 100], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([40, 128, 0], dtype=np.int32),
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

    builder = _create_gdn_builder(num_speculative_tokens=k)
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens), BLOCK_SIZE, DEVICE
    ).replace(is_prefilling=torch.tensor(is_prefilling))
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
    state.recoverssm = None
    state._mamba_ctx = None
    idx_mapping = torch.tensor([2, -1, 0], dtype=torch.int32, device="cuda")

    state.postprocess_state(idx_mapping, num_sampled)

    expected = torch.tensor(
        [expected_value, 9, expected_value, 9], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_close(state.num_accepted_tokens_gpu, expected)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_pp_warmup_postprocess_is_state_neutral() -> None:
    device = torch.device("cuda")
    recurrent_state = torch.full((2, 8), 7, dtype=torch.float32, device=device)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32, device=device)

    def tensor(values: list[int], dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype, device=device)

    ctx = MambaSpecDecodeGPUContext(
        state_base_addrs=tensor([recurrent_state.data_ptr()], torch.int64),
        state_block_strides=tensor(
            [recurrent_state.stride(0) * recurrent_state.element_size()], torch.int64
        ),
        state_elem_sizes=tensor([recurrent_state.element_size()], torch.int32),
        state_inner_sizes=tensor([recurrent_state.shape[1]], torch.int64),
        state_conv_widths=tensor([0], torch.int32),
        state_group_indices=tensor([0], torch.int32),
        state_dim_row_count=tensor([0], torch.int32),
        state_dim_row_stride=tensor([0], torch.int64),
        block_size=8,
        num_states=1,
        mamba_group_ids=[0],
        num_groups=1,
        num_accepted_tokens_out=torch.zeros(4, dtype=torch.int32, device=device),
        block_table_ptrs=tensor([block_table.data_ptr()], torch.int64),
        block_table_stride_req=block_table.stride(0),
        is_initialized=True,
    )
    state = object.__new__(MambaHybridModelState)
    state.num_accepted_tokens_gpu = tensor([9, 8, 7, 6], torch.int32)
    state._align_mode = True
    state._mamba_ctx = ctx
    state._mamba_state_idx_gpu = tensor([0, 0, 0, 0], torch.int32)

    accepted_before = state.num_accepted_tokens_gpu.clone()
    recurrent_before = recurrent_state.clone()
    state.warmup_postprocess_state(
        tensor([-1], torch.int64),
        tensor([3], torch.int32),
        tensor([8, 8, 8, 8], torch.int32),
    )

    torch.testing.assert_close(state.num_accepted_tokens_gpu, accepted_before)
    torch.testing.assert_close(recurrent_state, recurrent_before)


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
