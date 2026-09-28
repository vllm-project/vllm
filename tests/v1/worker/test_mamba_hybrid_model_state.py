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
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.worker.gpu.model_states import mamba_hybrid
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState


@pytest.mark.parametrize("cache_mode", ["none", "align"])
def test_mamba_group_lookup_without_prefix_caching(monkeypatch, cache_mode):
    def init_base(self, *args):
        self.max_num_reqs = 2
        self.device = torch.device("cpu")

    monkeypatch.setattr(mamba_hybrid.DefaultModelState, "__init__", init_base)
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            mamba_cache_mode=cache_mode,
            use_kda_recoverssm=False,
        )
    )
    state = MambaHybridModelState(
        config, torch.nn.Identity(), None, torch.device("cpu")
    )
    spec = MambaSpec(
        block_size=16,
        shapes=((4, 4),),
        dtypes=(torch.float32,),
        mamba_cache_mode=cache_mode,
    )
    cache = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["linear"], spec)],
    )
    assert state._get_mamba_group_info(cache) == ([0], spec)


@pytest.mark.parametrize("mode", ["tp", "pcp", "dummy", "capture"])
@pytest.mark.parametrize("spec_tokens", [0, 2])
@pytest.mark.parametrize("align", [False, True])
def test_hybrid_metadata_preserves_request_order(monkeypatch, mode, spec_tokens, align):
    """KDA state and acceptance use global requests, MLA uses local segments."""

    def batch(lengths, indices):
        lengths = np.array(lengths, dtype=np.int32)
        starts = np.r_[np.int32(0), lengths.cumsum(dtype=np.int32)]
        return SimpleNamespace(
            num_reqs=len(lengths),
            num_reqs_after_padding=len(lengths),
            num_tokens=int(starts[-1]),
            num_tokens_after_padding=int(starts[-1]),
            query_start_loc=torch.from_numpy(starts),
            query_start_loc_np=starts,
            num_scheduled_tokens=lengths,
            max_query_len=None,
            seq_lens=torch.from_numpy(lengths + 16),
            seq_lens_cpu_upper_bound=torch.from_numpy(lengths + 16),
            is_prefilling_np=np.array([False] + [True] * (len(lengths) - 1)),
            prefill_runs_as_decode_np=None,
            idx_mapping=torch.tensor(indices),
            idx_mapping_np=np.array(indices),
            num_draft_tokens_per_req=np.array([2] + [0] * (len(lengths) - 1)),
            dcp_local_seq_lens=None,
            dcp_local_seq_lens_cpu_upper_bound=None,
            positions=torch.arange(int(starts[-1])),
            prompt_lens=None,
            fast_prefill=None,
        )

    local = batch([3, 2, 2], [2, 0, 0])
    global_batch = batch([3, 8], [2, 0])
    local_tables = (torch.tensor([[12], [10], [10]]),) * 2
    global_tables = (torch.tensor([[12], [10]]),) * 2
    state = object.__new__(MambaHybridModelState)
    state.vllm_config = SimpleNamespace(num_speculative_tokens=spec_tokens)
    state.model_config = SimpleNamespace(max_model_len=8192)
    state.supports_mm_inputs = False
    state._align_mode = align
    state.model = Mock()
    state.num_accepted_tokens_gpu = torch.tensor([4, 5, 6], dtype=torch.int32)
    layout = Mock()
    build_layout = Mock(return_value=layout)
    state.pcp_manager = (
        None
        if mode == "tp"
        else SimpleNamespace(
            hybrid=True,
            global_batch=None if mode == "dummy" else global_batch,
            global_block_tables=global_tables,
            build_hybrid_layout=build_layout,
        )
    )
    spec = SimpleNamespace(shapes=((12, 3), (2, 4, 4)))
    state._get_mamba_group_info = Mock(return_value=([1], spec))
    state._ensure_align_ctx = Mock()
    state._ensure_align_ctx.return_value.compute_aligned_state_indices.return_value = (
        torch.tensor([1, 2]),
    )
    state.recoverssm = Mock()
    mla = Mock()
    kda = Mock(spec=mamba_hybrid.GDNAttentionMetadataBuilder)
    kda.mamba_aligned_state_indices = None
    kda_metadata = mamba_hybrid.GDNAttentionMetadata(
        0, 0, 0, 0, 0, 0, 0, non_spec_state_indices_tensor=torch.tensor([3, 5])
    )
    kda.build.return_value = kda.build_for_cudagraph_capture.return_value = kda_metadata
    groups = [
        [
            SimpleNamespace(
                layer_names=[name],
                kv_cache_spec=None,
                get_metadata_builder=Mock(return_value=b),
            )
        ]
        for name, b in [("mla", mla), ("kda", kda)]
    ]
    config = SimpleNamespace(kv_cache_groups=[None, None])
    capture = mode == "capture"

    metadata = state.prepare_attn(
        local,
        CUDAGraphMode.NONE,
        local_tables,
        torch.zeros((2, 14), dtype=torch.int64),
        groups,
        config,
        for_capture=capture,
    )

    assert set(metadata) == {"mla", "kda"}
    expected_kda = global_batch if mode == "pcp" else local
    expected_tables = global_tables if mode == "pcp" else local_tables
    for builder, expected, table in [
        (mla, local, local_tables[0]),
        (kda, expected_kda, expected_tables[1]),
    ]:
        call = builder.build_for_cudagraph_capture if capture else builder.build
        call.assert_called_once()
        common = (
            call.call_args.args[0]
            if capture
            else call.call_args.kwargs["common_attn_metadata"]
        )
        assert common.num_reqs == expected.num_reqs
        assert common.num_actual_tokens == expected.num_tokens
        assert common.positions is expected.positions
        assert common.block_table_tensor is table
        torch.testing.assert_close(common.query_start_loc, expected.query_start_loc)
        torch.testing.assert_close(common.seq_lens, expected.seq_lens)
        torch.testing.assert_close(
            common.is_prefilling, torch.from_numpy(expected.is_prefilling_np)
        )
    if not capture:
        kwargs = kda.build.call_args.kwargs
        if spec_tokens:
            torch.testing.assert_close(
                kwargs["num_accepted_tokens"],
                state.num_accepted_tokens_gpu[expected_kda.idx_mapping],
            )
            assert kwargs["num_decode_draft_tokens_cpu"].tolist() == (
                [2] + [-1] * (expected_kda.num_reqs - 1)
            )
        else:
            assert kwargs["num_accepted_tokens"] is None
            assert kwargs["num_decode_draft_tokens_cpu"] is None
    if align:
        state._ensure_align_ctx.assert_called_once_with(config, [1], expected_tables)
        align_ctx = state._ensure_align_ctx.return_value
        align_ctx.compute_aligned_state_indices.assert_called_once_with(
            expected_kda.seq_lens, expected_kda.num_reqs
        )
    state.recoverssm.record_step.assert_called_once()
    if mode == "pcp":
        build_layout.assert_called_once_with()
        layout.with_state_indices.assert_called_once_with(
            kda_metadata.non_spec_state_indices_tensor
        )
        assert metadata["kda"].pcp_layout is layout.with_state_indices.return_value
        state.model.prepare_hybrid_pcp.assert_called_once_with(layout)
    else:
        build_layout.assert_not_called()
        state.model.prepare_hybrid_pcp.assert_not_called()
        assert metadata["kda"].pcp_layout is None


def test_prepare_attn_forwards_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    state = object.__new__(MambaHybridModelState)
    state.pcp_manager = None
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
        prefill_runs_as_decode_np=None,
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
    state.pcp_manager = None
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
    state.pcp_manager = None
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
    state.pcp_manager = None
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
