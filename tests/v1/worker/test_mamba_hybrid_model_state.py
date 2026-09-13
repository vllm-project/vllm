# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.worker.gpu.model_states import mamba_hybrid
from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
    MambaHybridModelState,
    compute_num_decode_draft_tokens,
)
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState


def test_prepare_attn_forwards_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    state = object.__new__(MambaHybridModelState)
    state.vllm_config = SimpleNamespace(num_speculative_tokens=0)
    state.max_model_len = 8192
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


def _mamba_hybrid_state(num_speculative_tokens: int) -> MambaHybridModelState:
    state = object.__new__(MambaHybridModelState)
    state.vllm_config = SimpleNamespace(num_speculative_tokens=num_speculative_tokens)
    state._align_mode = False
    state.recoverssm = None
    # A request that accepted 3 drafts (plus the bonus token) on the previous
    # step; the other has never had a draft accepted.
    state.num_accepted_tokens_gpu = torch.tensor([4, 1], dtype=torch.int32)
    return state


def _input_batch_with_no_scheduled_drafts() -> SimpleNamespace:
    """A two-request decode batch where nobody proposed a draft this step
    (`num_draft_tokens_per_req is None`), e.g. because ngram found no match
    for anyone."""
    return SimpleNamespace(
        num_reqs=2,
        num_reqs_after_padding=2,
        num_tokens=2,
        num_tokens_after_padding=2,
        query_start_loc_np=np.array([0, 1, 2], dtype=np.int32),
        num_scheduled_tokens=np.array([1, 1], dtype=np.int32),
        seq_lens_cpu_upper_bound=torch.tensor([10, 10], dtype=torch.int32),
        is_prefilling_np=np.array([False, False]),
        num_draft_tokens_per_req=None,
        idx_mapping=torch.tensor([0, 1], dtype=torch.int64),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([10, 10], dtype=torch.int32),
        dcp_local_seq_lens=None,
        positions=torch.tensor([9, 9], dtype=torch.int64),
        prompt_lens=None,
    )


def _prepare_attn_metadata(
    state: MambaHybridModelState, monkeypatch
) -> SimpleNamespace:
    captured = {}

    def fake_build_attn_metadata(**kwargs):
        captured.update(kwargs)
        return Mock()

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.model_states.mamba_hybrid.build_attn_metadata",
        fake_build_attn_metadata,
    )

    state.prepare_attn(
        input_batch=_input_batch_with_no_scheduled_drafts(),
        cudagraph_mode=CUDAGraphMode.NONE,
        block_tables=(),
        slot_mappings={},
        attn_groups=[],
        kv_cache_config=Mock(),
    )
    return captured["model_specific_attn_metadata"]


def test_prepare_attn_keeps_the_accepted_offset_when_the_batch_drafted_nothing(
    monkeypatch,
):
    """`num_accepted_tokens` must reach the attention builders whenever the
    speculative config is on, even when *this* batch scheduled no draft
    tokens at all. Before the fix, `use_spec_decode` was gated on this step's
    scheduled drafts (`len(scheduled_spec_decode_tokens) > 0` in the V1
    runner) rather than on the speculative config, so a request that accepted
    drafts on a previous step would silently lose that offset the moment
    ngram found no match for anyone in the batch.
    """
    state = _mamba_hybrid_state(num_speculative_tokens=3)

    metadata = _prepare_attn_metadata(state, monkeypatch)

    assert metadata.num_accepted_tokens is not None
    assert metadata.num_accepted_tokens.tolist() == [4, 1]
    assert metadata.num_decode_draft_tokens_cpu is not None


def test_prepare_attn_omits_the_offset_without_a_speculative_config(monkeypatch):
    """Mirrors the check above: with speculative decoding off entirely, there
    is no accepted-token offset to carry, and the metadata must say so
    explicitly (`None`) rather than a stale or default value."""
    state = _mamba_hybrid_state(num_speculative_tokens=0)

    metadata = _prepare_attn_metadata(state, monkeypatch)

    assert metadata.num_accepted_tokens is None
    assert metadata.num_decode_draft_tokens_cpu is None


def test_zero_draft_decode_row_is_marked_as_a_speculative_row():
    """A decode row that got no drafts still has to reach the speculative path.

    The recurrent backends only apply a row's accepted-token offset on that
    path, and a row that accepted drafts on the previous step keeps that offset
    even when the current step proposes nothing (the scheduler drops drafts
    whenever the token budget truncates the step). Marking such a row with the
    -1 sentinel sends it to the plain decode path, which reads the state at
    column 0 and ignores the offset.
    """
    num_decode_draft_tokens = compute_num_decode_draft_tokens(
        num_padded_reqs=3,
        num_scheduled_tokens=np.array([3, 1, 1], dtype=np.int32),
        num_draft_tokens_per_req=np.array([2, 0, 0], dtype=np.int32),
        is_prefilling=np.array([False, False, True]),
    )

    assert num_decode_draft_tokens.tolist() == [
        2,  # decode row with drafts
        0,  # decode row with no drafts: speculative path, offset applied
        -1,  # one-token prefill tail: no offset to apply, stays a plain row
    ]


def test_batch_without_any_drafts_still_marks_its_decode_rows():
    """`num_draft_tokens_per_req` is None whenever no row in the batch drafted
    anything. Those rows are ordinary decode rows carrying an offset from the
    previous step, so the absence of drafts in this step must not be read as
    speculative decoding being off."""
    num_decode_draft_tokens = compute_num_decode_draft_tokens(
        num_padded_reqs=2,
        num_scheduled_tokens=np.array([1, 1], dtype=np.int32),
        num_draft_tokens_per_req=None,
        is_prefilling=np.array([False, True]),
    )

    assert num_decode_draft_tokens.tolist() == [0, -1]


def test_padded_rows_keep_the_sentinel():
    """Full CUDA-graph capture pads the batch; padded rows describe no request
    and must not be picked up as speculative rows."""
    num_decode_draft_tokens = compute_num_decode_draft_tokens(
        num_padded_reqs=4,
        num_scheduled_tokens=np.array([1, 1], dtype=np.int32),
        num_draft_tokens_per_req=np.array([0, 0], dtype=np.int32),
        is_prefilling=np.array([False, False]),
    )

    assert num_decode_draft_tokens.tolist() == [0, 0, -1, -1]


def test_chunked_prefill_tail_of_two_or_three_tokens_keeps_the_sentinel():
    """A prefill continuation is excluded by `is_prefilling`, not by its token
    count, so tails wider than one token are covered by the same rule."""
    num_decode_draft_tokens = compute_num_decode_draft_tokens(
        num_padded_reqs=2,
        num_scheduled_tokens=np.array([2, 3], dtype=np.int32),
        num_draft_tokens_per_req=np.array([0, 0], dtype=np.int32),
        is_prefilling=np.array([True, True]),
    )

    assert num_decode_draft_tokens.tolist() == [-1, -1]


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
