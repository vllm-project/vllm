# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import vllm.envs as envs
from vllm.model_executor.layers.hybrid_nvfp4_lm_head import (
    HybridNvfp4LmHead,
    _attach_state,
    release_hybrid_nvfp4_lm_head,
    refresh_hybrid_nvfp4_lm_head,
    select_lm_head_candidates,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.warmup.hybrid_nvfp4_lm_head_warmup import _row_shapes
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS


class _RecordingHybridState:
    def __init__(self) -> None:
        self.can_use_called = False

    def can_use(self, *args, **kwargs) -> bool:
        self.can_use_called = True
        return True


class _FallbackHybridState(_RecordingHybridState):
    def coarse_logits(self, hidden_states, bias):
        return torch.zeros((hidden_states.shape[0], 8), dtype=torch.bfloat16)

    def select_candidates(self, coarse_logits, *, top_k):
        return torch.tensor([[5, 2, 7]], dtype=torch.int64).expand(
            coarse_logits.shape[0], -1
        )

    def refine_logits(self, hidden_states, weight, candidate_indices, bias):
        return torch.tensor([[1.0, 1.0, float("nan")]], dtype=torch.float32).expand(
            hidden_states.shape[0], -1
        )


class _TopKHybridState(_RecordingHybridState):
    def __init__(self) -> None:
        super().__init__()
        self.requested_top_k: list[int] = []

    def can_use(self, *args, **kwargs) -> bool:
        self.can_use_called = True
        return True

    def candidate_count_for_topk(self, top_k: int) -> int:
        return 4 if top_k > 1 else 2

    def coarse_logits(self, hidden_states, bias):
        return torch.tensor(
            [[8.0, 7.0, 6.0, 5.0, 1.0, 0.0, -1.0, -2.0]],
            dtype=torch.bfloat16,
        ).expand(hidden_states.shape[0], -1)

    def select_candidates(self, coarse_logits, *, top_k):
        self.requested_top_k.append(top_k)
        count = self.candidate_count_for_topk(top_k)
        return torch.arange(count, dtype=torch.int64).expand(
            coarse_logits.shape[0], -1
        )

    def refine_logits(self, hidden_states, weight, candidate_indices, bias):
        exact = torch.tensor(
            [[0.0, 4.0, 3.0, 2.0]], dtype=torch.float32
        )
        return exact[:, : candidate_indices.shape[1]].expand(
            hidden_states.shape[0], -1
        )


class _PenaltyAwareHybridState(_RecordingHybridState):
    """Small state that exposes candidate ordering to the penalty test."""

    def can_use(self, *args, **kwargs) -> bool:
        self.can_use_called = True
        return True

    def candidate_count_for_topk(self, top_k: int) -> int:
        del top_k
        return 2

    def coarse_logits(self, hidden_states, bias):
        del bias
        return torch.tensor(
            [[5.0, 4.0, 3.0, 2.0]],
            dtype=torch.bfloat16,
            device=hidden_states.device,
        ).expand(hidden_states.shape[0], -1)

    def select_candidates(self, coarse_logits, *, top_k):
        del top_k
        return torch.argsort(
            coarse_logits, dim=-1, descending=True, stable=True
        )[:, :2]

    def refine_logits(self, hidden_states, weight, candidate_indices, bias):
        del hidden_states, weight, bias
        # Match the coarse values so the only ordering change comes from the
        # presence penalty applied before candidate selection.
        return 5.0 - candidate_indices.to(torch.float32)


def _make_lm_head(*, added_embeddings: int = 0) -> SimpleNamespace:
    vocab_size = 8
    org_vocab_size = vocab_size - added_embeddings - (1 if added_embeddings else 0)
    padded_org_vocab_size = org_vocab_size if not added_embeddings else 6
    return SimpleNamespace(
        weight=torch.zeros(vocab_size, 4, dtype=torch.bfloat16),
        shard_indices=SimpleNamespace(
            num_org_vocab_padding=padded_org_vocab_size - org_vocab_size,
            num_elements_padded=vocab_size,
            org_vocab_start_index=0,
            padded_org_vocab_start_index=0,
            padded_org_vocab_end_index=padded_org_vocab_size,
            num_org_elements=org_vocab_size,
            num_org_elements_padded=padded_org_vocab_size,
            added_vocab_start_index=org_vocab_size,
            added_vocab_end_index=org_vocab_size + added_embeddings,
            padded_added_vocab_start_index=org_vocab_size,
            num_added_elements=added_embeddings,
        ),
        num_added_embeddings=added_embeddings,
        tp_size=1,
    )


def _make_processor() -> LogitsProcessor:
    processor = LogitsProcessor(8)
    processor.head_dtype = torch.bfloat16
    return processor


def test_batch_invariant_disables_hybrid_state(monkeypatch, default_vllm_config):
    processor = _make_processor()
    lm_head = _make_lm_head()
    state = _RecordingHybridState()
    lm_head._hybrid_nvfp4_lm_head_state = state
    logits = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    processor._apply_head = lambda *args: logits  # type: ignore[method-assign]

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    hidden_states = torch.zeros(4, 4, dtype=torch.bfloat16)
    top_tokens = processor.get_top_tokens(lm_head, hidden_states)

    assert torch.equal(top_tokens, logits.argmax(dim=-1))
    assert not state.can_use_called


def test_candidate_fallback_uses_lower_index_for_ties():
    coarse_logits = torch.ones((2, 4), dtype=torch.float32)

    candidate_indices = select_lm_head_candidates(coarse_logits, candidates=2)

    assert torch.equal(candidate_indices, torch.tensor([[0, 1], [0, 1]]))


def test_candidate_selection_masks_nan_rows():
    coarse_logits = torch.tensor([[float("nan"), 2.0, float("nan"), 1.0]])

    candidate_indices = select_lm_head_candidates(coarse_logits, candidates=3)

    assert torch.equal(candidate_indices, torch.tensor([[1, 3, 0]]))
    assert torch.isneginf(coarse_logits[0, 0])


def test_tp_argmax_reduction_handles_ties_and_nan(monkeypatch, default_vllm_config):
    processor = _make_processor()
    gathered = torch.tensor([[1.0, 7.0, 1.0, 3.0], [float("nan"), 5.0, 2.0, 4.0]])
    monkeypatch.setattr(
        "vllm.model_executor.layers.logits_processor.tensor_model_parallel_all_gather",
        lambda pair, dim: gathered,
    )

    result = processor.reduce_local_argmax(
        torch.tensor([1.0, float("nan")]),
        torch.tensor([7, 5]),
        tp_size=2,
    )

    assert torch.equal(result, torch.tensor([3, 4], dtype=torch.int64))


def test_tp_argmax_reduction_all_negative_inf_prefers_lower_token_id(
    monkeypatch, default_vllm_config
):
    processor = _make_processor()
    gathered = torch.tensor([[-float("inf"), 7.0, -float("inf"), 3.0]])
    monkeypatch.setattr(
        "vllm.model_executor.layers.logits_processor.tensor_model_parallel_all_gather",
        lambda pair, dim: gathered,
    )

    result = processor.reduce_local_argmax(
        torch.tensor([-float("inf"), -float("inf")]),
        torch.tensor([7, 3]),
        tp_size=2,
    )

    assert torch.equal(result, torch.tensor([3], dtype=torch.int64))


def test_large_vocab_argmax_reduction_keeps_int64_ids(monkeypatch, default_vllm_config):
    processor = LogitsProcessor(1 << 24)
    gathered_values = torch.tensor([[2.0, 2.0], [float("nan"), -float("inf")]])
    gathered_ids = torch.tensor(
        [[1 << 24, (1 << 24) + 1], [1 << 24, 3]], dtype=torch.int64
    )
    calls = []

    def gather(tensor, dim):
        del dim
        calls.append(tensor.dtype)
        return gathered_values if tensor.dtype.is_floating_point else gathered_ids

    monkeypatch.setattr(
        "vllm.model_executor.layers.logits_processor.tensor_model_parallel_all_gather",
        gather,
    )
    result = processor.reduce_local_argmax(
        torch.tensor([2.0, float("nan")]),
        torch.tensor([1 << 24, 3], dtype=torch.int64),
        tp_size=2,
    )

    assert calls == [torch.float32, torch.int64]
    assert torch.equal(result, torch.tensor([1 << 24, 3], dtype=torch.int64))


def test_head_dtype_mismatch_disables_hybrid_state(monkeypatch, default_vllm_config):
    processor = _make_processor()
    processor.head_dtype = torch.float32
    lm_head = _make_lm_head()
    state = _RecordingHybridState()
    lm_head._hybrid_nvfp4_lm_head_state = state
    logits = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    processor._apply_head = lambda *args: logits  # type: ignore[method-assign]

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    hidden_states = torch.zeros(4, 4, dtype=torch.bfloat16)
    top_tokens = processor.get_top_tokens(lm_head, hidden_states)

    assert torch.equal(top_tokens, logits.argmax(dim=-1))
    assert not state.can_use_called


def test_nan_diagnostics_disables_hybrid_state(monkeypatch, default_vllm_config):
    processor = _make_processor()
    lm_head = _make_lm_head()
    state = _RecordingHybridState()
    lm_head._hybrid_nvfp4_lm_head_state = state
    logits = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    processor._apply_head = lambda *args: logits  # type: ignore[method-assign]

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(envs, "VLLM_COMPUTE_NANS_IN_LOGITS", True)
    hidden_states = torch.zeros(4, 4, dtype=torch.bfloat16)
    top_tokens = processor.get_top_tokens(lm_head, hidden_states)

    assert torch.equal(top_tokens, logits.argmax(dim=-1))
    assert not state.can_use_called


def test_hybrid_fallback_argmax_is_stable_and_nan_safe(
    monkeypatch, default_vllm_config
):
    processor = _make_processor()
    lm_head = _make_lm_head()
    lm_head._hybrid_nvfp4_lm_head_state = _FallbackHybridState()
    hidden_states = torch.zeros(2, 4, dtype=torch.bfloat16)

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    result = processor.get_top_tokens(lm_head, hidden_states)

    assert torch.equal(result, torch.full((2,), 2, dtype=torch.int64))


def test_mixed_prefill_row_mask_keeps_decode_rows_compact(
    monkeypatch, default_vllm_config
):
    processor = _make_processor()
    lm_head = _make_lm_head()
    state = _FallbackHybridState()
    lm_head._hybrid_nvfp4_lm_head_state = state

    def apply_head(_lm_head, states, _bias=None):
        return torch.arange(states.shape[0] * 8, dtype=torch.float32).reshape(
            states.shape[0], 8
        )

    processor._apply_head = apply_head  # type: ignore[method-assign]
    processor.hybrid_lm_head_row_mask = torch.tensor([True, False])
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)

    hidden_states = torch.zeros(2, 4, dtype=torch.bfloat16)
    result = processor.get_top_tokens(lm_head, hidden_states)

    assert torch.equal(result, torch.tensor([2, 7], dtype=torch.int64))
    assert state.can_use_called


def test_mixed_prefill_presence_rows_keep_exact_fallback(
    monkeypatch, default_vllm_config
):
    processor = _make_processor()
    lm_head = _make_lm_head()
    state = _FallbackHybridState()
    lm_head._hybrid_nvfp4_lm_head_state = state

    def apply_head(_lm_head, states, _bias=None):
        rows = []
        for state in states:
            if state[0] > 0:
                rows.append([0.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0])
            else:
                rows.append([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        return torch.tensor(rows, dtype=torch.float32)

    processor._apply_head = apply_head  # type: ignore[method-assign]
    processor.hybrid_lm_head_row_mask = torch.tensor([True, False])
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)

    hidden_states = torch.zeros(2, 4, dtype=torch.bfloat16)
    hidden_states[1, 0] = 1
    values, token_ids = processor.get_topk_candidates(
        lm_head,
        hidden_states,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
        presence_penalties=torch.tensor([0.5, 0.5]),
        output_token_ids=torch.tensor([[5], [1]], dtype=torch.int64),
    )

    # The prompt row (row 1) must use the full head: the fake hybrid state
    # never proposes token 1, while the exact row's top token is 1.
    assert token_ids[1, 0] == 1
    assert values[1, 0] == pytest.approx(9.5)
    assert state.can_use_called


def test_added_vocab_uses_layout_aware_path(monkeypatch, default_vllm_config):
    processor = _make_processor()
    lm_head = _make_lm_head(added_embeddings=2)
    state = _RecordingHybridState()
    lm_head._hybrid_nvfp4_lm_head_state = state
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 100.0, 6.0, 7.0]]).expand(4, -1)
    processor._apply_head = lambda *args: logits  # type: ignore[method-assign]

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    hidden_states = torch.zeros(4, 4, dtype=torch.bfloat16)
    top_tokens = processor.get_top_tokens(lm_head, hidden_states)

    assert torch.equal(top_tokens, torch.full((4,), 6, dtype=torch.int64))
    assert not state.can_use_called


def test_added_vocab_shard_metadata_disables_hybrid_without_layer_alias(
    monkeypatch, default_vllm_config
):
    processor = _make_processor()
    lm_head = _make_lm_head(added_embeddings=2)
    # Older/custom heads may not expose num_added_embeddings on the module,
    # while shard_indices still carries the authoritative layout.
    del lm_head.num_added_embeddings
    state = _RecordingHybridState()
    lm_head._hybrid_nvfp4_lm_head_state = state
    logits = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    processor._apply_head = lambda *args: logits  # type: ignore[method-assign]

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    hidden_states = torch.zeros(4, 4, dtype=torch.bfloat16)
    top_tokens = processor.get_top_tokens(lm_head, hidden_states)

    assert torch.equal(top_tokens, torch.full((4,), 6, dtype=torch.int64))
    assert not state.can_use_called


def test_warmup_row_shapes_respect_memory_cap(monkeypatch, default_vllm_config):
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=2048),
        speculative_config=SimpleNamespace(num_speculative_tokens=2),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 512, 1024]),
    )
    worker = SimpleNamespace(vllm_config=config)
    monkeypatch.setattr(envs, "VLLM_HYBRID_NVFP4_LM_HEAD_MAX_AUTOTUNE_ROWS", 512)
    monkeypatch.setattr(envs, "VLLM_HYBRID_NVFP4_LM_HEAD_MAX_ROWS", 512)

    assert _row_shapes(worker) == (1, 512)

    config.compilation_config.cudagraph_capture_sizes = []
    monkeypatch.setattr(envs, "VLLM_HYBRID_NVFP4_LM_HEAD_MAX_AUTOTUNE_ROWS", 128)
    monkeypatch.setattr(envs, "VLLM_HYBRID_NVFP4_LM_HEAD_MAX_ROWS", 64)
    assert max(_row_shapes(worker)) <= 64


def test_sampler_workspace_warmup_handles_small_vocab():
    sampler = object.__new__(Sampler)
    sampler.device = torch.device("cpu")
    sampler.sampling_states = SimpleNamespace(vocab_size=16)

    # The production warmup uses top_k=20; clamp it for toy vocabularies used
    # by CPU/unit-test runners instead of indexing beyond the logits width.
    sampler.warmup_top_k_top_p_buffer(8)


def test_sampler_workspace_warmup_respects_byte_cap(monkeypatch):
    sampler = object.__new__(Sampler)
    sampler.device = torch.device("cpu")
    sampler.sampling_states = SimpleNamespace(vocab_size=1024)
    monkeypatch.setattr(envs, "VLLM_HYBRID_NVFP4_LM_HEAD_MAX_WARMUP_BYTES", 1024)

    def fail_allocation(*args, **kwargs):
        raise AssertionError("warmup allocation exceeded byte cap")

    monkeypatch.setattr(torch, "zeros", fail_allocation)
    sampler.warmup_top_k_top_p_buffer(8)


def test_compact_topk_native_path_applies_top_p(monkeypatch, default_vllm_config):
    processor = _make_processor()
    lm_head = _make_lm_head()
    logits = torch.tensor(
        [[5.0, 4.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]],
        dtype=torch.bfloat16,
    )
    processor._apply_head = lambda *args: logits  # type: ignore[method-assign]
    hidden_states = torch.zeros(1, 4, dtype=torch.bfloat16)

    values, token_ids = processor.get_topk_candidates(
        lm_head,
        hidden_states,
        top_k=2,
        top_p=0.7,
        temperature=1.0,
    )

    assert torch.equal(token_ids, torch.tensor([[0, 1]], dtype=torch.int64))
    assert values[0, 0] == 5.0
    assert torch.isneginf(values[0, 1])


def test_full_vocab_sampling_uses_local_gumbel_path(default_vllm_config):
    processor = _make_processor()
    lm_head = _make_lm_head()
    logits = torch.tensor(
        [[5.0, 4.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]],
        dtype=torch.bfloat16,
    )
    processor._apply_head = lambda *args: logits  # type: ignore[method-assign]
    hidden_states = torch.zeros(1, 4, dtype=torch.bfloat16)

    sampled = processor.sample_full_tokens(
        lm_head,
        hidden_states,
        temperature=0.7,
    )

    assert sampled.shape == (1,)
    assert 0 <= int(sampled[0]) < 8


def test_compact_topk_hybrid_widens_transient_candidates(
    monkeypatch, default_vllm_config
):
    processor = _make_processor()
    lm_head = _make_lm_head()
    state = _TopKHybridState()
    lm_head._hybrid_nvfp4_lm_head_state = state
    hidden_states = torch.zeros(1, 4, dtype=torch.bfloat16)

    values, token_ids = processor.get_topk_candidates(
        lm_head,
        hidden_states,
        top_k=2,
        top_p=1.0,
        temperature=1.0,
    )

    assert state.requested_top_k == [2]
    assert torch.equal(token_ids, torch.tensor([[1, 2]], dtype=torch.int64))
    assert torch.equal(values, torch.tensor([[4.0, 3.0]]))


def test_compact_topk_tie_breaks_by_token_id(default_vllm_config):
    processor = _make_processor()
    values, token_ids = processor._select_compact_topk_values_ids(
        torch.tensor([[1.0, 1.0, 0.0]]),
        torch.tensor([[7, 3, 5]], dtype=torch.int64),
        top_k=2,
        top_p=1.0,
    )
    assert torch.equal(values, torch.tensor([[1.0, 1.0]]))
    assert torch.equal(token_ids, torch.tensor([[3, 7]], dtype=torch.int64))


def test_large_vocab_compact_selection_keeps_integer_ids(default_vllm_config):
    processor = LogitsProcessor(1 << 24)
    values, token_ids = processor._select_compact_topk_values_ids(
        torch.tensor([[2.0, 2.0, 1.0]]),
        torch.tensor([[1 << 24, (1 << 24) + 1, 3]], dtype=torch.int64),
        top_k=2,
        top_p=1.0,
    )
    assert torch.equal(
        token_ids, torch.tensor([[1 << 24, (1 << 24) + 1]], dtype=torch.int64)
    )


def test_tied_hybrid_state_releases_after_last_attachment():
    weight = torch.zeros((4, 4), dtype=torch.bfloat16)
    layer_a = nn.Module()
    layer_b = nn.Module()
    layer_a.weight = weight
    layer_b.weight = weight
    state = HybridNvfp4LmHead(
        weight=torch.ones((4, 2), dtype=torch.uint8),
        scale=torch.ones((4, 1), dtype=torch.float32),
        global_scale=torch.ones((), dtype=torch.float32),
        input_size=4,
        output_size=4,
        candidates=2,
    )
    _attach_state(layer_a, state)
    _attach_state(layer_b, state)
    setattr(weight, "_hybrid_nvfp4_lm_head_shared_state", state)

    assert release_hybrid_nvfp4_lm_head(layer_a) == 0
    assert getattr(layer_b, "_hybrid_nvfp4_lm_head_state") is state
    assert getattr(weight, "_hybrid_nvfp4_lm_head_shared_state") is state

    released = release_hybrid_nvfp4_lm_head(layer_b)
    assert released == (
        state.weight.nbytes + state.scale.nbytes + state.global_scale.nbytes
    )
    assert not hasattr(weight, "_hybrid_nvfp4_lm_head_shared_state")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_refresh_hybrid_state_updates_derived_buffers_in_place(monkeypatch):
    device = torch.device("cuda")
    layer = nn.Module()
    layer.weight = nn.Parameter(
        torch.zeros((4, 4), dtype=torch.bfloat16, device=device),
        requires_grad=False,
    )
    state = HybridNvfp4LmHead(
        weight=torch.ones((4, 2), dtype=torch.uint8, device=device),
        scale=torch.ones((4, 1), dtype=torch.float32, device=device),
        global_scale=torch.ones((), dtype=torch.float32, device=device),
        input_size=4,
        output_size=4,
        candidates=2,
    )
    _attach_state(layer, state)
    old_ptrs = tuple(value.data_ptr() for value in (state.weight, state.scale))

    def fake_quantize(weight):
        assert weight is layer.weight
        return (
            torch.full_like(state.weight, 7),
            torch.full_like(state.scale, 2.0),
            torch.tensor(3.0, dtype=torch.float32, device=device),
        )

    monkeypatch.setattr(
        "vllm.model_executor.layers.hybrid_nvfp4_lm_head._quantize_lm_head_weight",
        fake_quantize,
    )

    try:
        assert refresh_hybrid_nvfp4_lm_head(layer, candidates=3)
        assert tuple(
            value.data_ptr() for value in (state.weight, state.scale)
        ) == old_ptrs
        assert torch.equal(state.weight, torch.full_like(state.weight, 7))
        assert torch.equal(state.scale, torch.full_like(state.scale, 2.0))
        assert torch.equal(
            state.global_scale,
            torch.tensor(3.0, dtype=torch.float32, device=device),
        )
        assert state.candidates == 3
    finally:
        release_hybrid_nvfp4_lm_head(layer)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_refresh_hybrid_state_releases_on_shape_change():
    device = torch.device("cuda")
    layer = nn.Module()
    layer.weight = nn.Parameter(
        torch.zeros((4, 4), dtype=torch.bfloat16, device=device),
        requires_grad=False,
    )
    state = HybridNvfp4LmHead(
        weight=torch.ones((4, 2), dtype=torch.uint8, device=device),
        scale=torch.ones((4, 1), dtype=torch.float32, device=device),
        global_scale=torch.ones((), dtype=torch.float32, device=device),
        input_size=4,
        output_size=4,
        candidates=2,
    )
    _attach_state(layer, state)
    layer.weight = nn.Parameter(
        torch.zeros((5, 4), dtype=torch.bfloat16, device=device),
        requires_grad=False,
    )

    assert not refresh_hybrid_nvfp4_lm_head(layer)
    assert not hasattr(layer, "_hybrid_nvfp4_lm_head_state")
    assert not hasattr(layer, "_hybrid_nvfp4_lm_head_weight")
    assert not hasattr(layer, "_hybrid_nvfp4_lm_head_scale")
    assert not hasattr(layer, "_hybrid_nvfp4_lm_head_global_scale")


def test_hybrid_state_row_limit_fails_closed():
    state = HybridNvfp4LmHead(
        weight=torch.ones((4, 2), dtype=torch.uint8),
        scale=torch.ones((4, 1), dtype=torch.float32),
        global_scale=torch.ones((), dtype=torch.float32),
        input_size=4,
        output_size=4,
        candidates=2,
        max_rows=2,
    )
    hidden_states = SimpleNamespace(
        ndim=2,
        dtype=torch.bfloat16,
        is_cuda=True,
        is_contiguous=lambda: True,
        shape=(3, 4),
        device=torch.device("cpu"),
    )

    assert not state.can_use(
        hidden_states,
        bf16_weight=torch.zeros((4, 4), dtype=torch.bfloat16),
        active_vocab_size=4,
        top_k=1,
    )
    assert state.can_use_failure_counts["rows_exceed_limit"] == 1


def test_presence_penalty_compact_helper_uses_output_counts(default_vllm_config):
    processor = _make_processor()
    lm_head = _make_lm_head()
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    counts = torch.tensor([[0, 1, 0, 2]], dtype=torch.int32)
    penalties = torch.tensor([0.5], dtype=torch.float32)
    request_indices = torch.tensor([0], dtype=torch.int64)

    processor._apply_presence_penalty_from_counts(
        logits,
        penalties,
        counts,
        request_indices,
        shard_indices=lm_head.shard_indices,
    )

    assert torch.equal(logits, torch.tensor([[1.0, 1.5, 3.0, 3.5]]))


def test_presence_penalty_compact_helper_uses_unique_output_ids(default_vllm_config):
    processor = _make_processor()
    lm_head = _make_lm_head()
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    output_token_ids = torch.tensor([[1, 3, 8]], dtype=torch.int64)
    penalties = torch.tensor([0.5], dtype=torch.float32)

    processor._apply_presence_penalty_from_token_ids(
        logits,
        penalties,
        output_token_ids,
        shard_indices=lm_head.shard_indices,
    )

    assert torch.equal(logits, torch.tensor([[1.0, 1.5, 3.0, 3.5]]))

    candidate_logits = torch.tensor([[4.0, 3.0]], dtype=torch.float32)
    candidate_ids = torch.tensor([[3, 6]], dtype=torch.int64)
    processor._apply_presence_penalty_from_token_ids(
        candidate_logits,
        penalties,
        output_token_ids,
        shard_indices=lm_head.shard_indices,
        local_token_ids=candidate_ids,
    )
    assert torch.equal(candidate_logits, torch.tensor([[3.5, 3.0]]))


def test_presence_penalty_is_applied_before_hybrid_candidate_selection(
    monkeypatch, default_vllm_config
):
    processor = _make_processor()
    lm_head = _make_lm_head()
    state = _PenaltyAwareHybridState()
    lm_head._hybrid_nvfp4_lm_head_state = state
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)

    values, token_ids = processor.get_topk_candidates(
        lm_head,
        torch.zeros(1, 4, dtype=torch.bfloat16),
        top_k=1,
        top_p=1.0,
        temperature=1.0,
        presence_penalties=torch.tensor([3.0]),
        output_token_ids=torch.tensor([[0, 1]], dtype=torch.int64),
    )

    # Without the pre-selection penalty, candidates would be [0, 1] and the
    # result would remain token 0.  Applying it first admits token 2.
    assert torch.equal(token_ids, torch.tensor([[2]], dtype=torch.int64))
    assert torch.equal(values, torch.tensor([[3.0]]))
    assert state.can_use_called


def _sampling_gate_stub(
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    presence: float = 0.0,
    repetition: float = 1.0,
    frequency: float = 0.0,
    explicit_seed: bool = False,
):
    sampler = SimpleNamespace(
        compute_nans=False,
        return_sampling_mask=False,
        trace_replay_state=None,
        use_fp64_gumbel=False,
        sampling_states=SimpleNamespace(
            vocab_size=128,
            temperature=SimpleNamespace(np=np.array([temperature], np.float32)),
            top_k=SimpleNamespace(np=np.array([top_k], np.int32)),
            top_p=SimpleNamespace(np=np.array([top_p], np.float32)),
            min_p=SimpleNamespace(np=np.array([0.0], np.float32)),
            max_num_logprobs=lambda _: NO_LOGPROBS,
            any_explicit_seed=lambda _: explicit_seed,
        ),
        logprob_token_ids_state=SimpleNamespace(max_num_token_ids=lambda _: 0),
        logit_bias_state=SimpleNamespace(use_logit_bias=np.array([False])),
        bad_words_state=SimpleNamespace(
            num_bad_words=SimpleNamespace(np=np.array([0], np.int32))
        ),
        thinking_budget_state=SimpleNamespace(enabled=False),
        penalties_state=SimpleNamespace(
            use_penalty=np.array(
                [presence != 0.0 or repetition != 1.0 or frequency != 0.0]
            ),
            repetition_penalty=SimpleNamespace(np=np.array([repetition], np.float32)),
            frequency_penalty=SimpleNamespace(np=np.array([frequency], np.float32)),
        ),
    )
    input_batch = SimpleNamespace(idx_mapping_np=np.array([0], np.int32))
    return sampler, input_batch


def test_sampling_gate_matches_supported_penalty_modes():
    sampler, input_batch = _sampling_gate_stub(temperature=0.7, top_k=8, top_p=0.9)
    result = Sampler.get_vocab_parallel_sampling_params(sampler, input_batch)
    assert result is not None
    assert result[0] == "topk" and result[1] == 8 and not result[-1]
    assert result[2] == pytest.approx(0.9)
    assert result[3] == pytest.approx(0.7)

    sampler, input_batch = _sampling_gate_stub(
        temperature=0.7, top_k=8, top_p=0.9, presence=0.5
    )
    result = Sampler.get_vocab_parallel_sampling_params(sampler, input_batch)
    assert result is not None and result[0] == "topk" and result[-1]

    sampler, input_batch = _sampling_gate_stub(
        temperature=0.7, top_k=8, top_p=0.9, frequency=0.1
    )
    assert Sampler.get_vocab_parallel_sampling_params(sampler, input_batch) is None


def test_sampling_gate_keeps_presence_penalty_greedy_on_exact_path():
    sampler, input_batch = _sampling_gate_stub(
        temperature=0.0, top_k=1, top_p=1.0, presence=0.5
    )

    # The current hybrid greedy implementation has no penalty metadata.  Do
    # not silently ignore presence penalty by entering the approximate path.
    assert Sampler.get_vocab_parallel_sampling_params(sampler, input_batch) is None


def test_sampling_gate_rejects_top_k_above_compact_limit():
    sampler, input_batch = _sampling_gate_stub(
        temperature=0.7, top_k=65, top_p=1.0
    )

    assert Sampler.get_vocab_parallel_sampling_params(sampler, input_batch) is None


def test_sampling_gate_accepts_explicit_seed_for_keyed_fast_path():
    sampler, input_batch = _sampling_gate_stub(
        temperature=0.7, top_k=8, top_p=0.9, explicit_seed=True
    )
    result = Sampler.get_vocab_parallel_sampling_params(sampler, input_batch)
    assert result is not None and result[0] == "topk"

    sampler, input_batch = _sampling_gate_stub(temperature=0.7, top_k=128, top_p=1.0)
    result = Sampler.get_vocab_parallel_sampling_params(sampler, input_batch)
    assert result is not None and result[0] == "full"

    sampler, input_batch = _sampling_gate_stub(
        temperature=0.7, top_k=128, top_p=1.0, presence=0.5
    )
    # Presence-only penalties are not representable by unrestricted local
    # Gumbel-max; they must stay on the regular full-logits sampler.
    assert Sampler.get_vocab_parallel_sampling_params(sampler, input_batch) is None


def test_sampling_gate_rejects_fp64_gumbel():
    sampler, input_batch = _sampling_gate_stub(
        temperature=0.7, top_k=8, top_p=0.9
    )
    sampler.use_fp64_gumbel = True

    assert Sampler.get_vocab_parallel_sampling_params(sampler, input_batch) is None


def test_sampling_gate_fails_closed_without_triton(monkeypatch):
    sampler, input_batch = _sampling_gate_stub(
        temperature=0.7, top_k=8, top_p=0.9, explicit_seed=True
    )
    sampler.device = torch.device("cuda")
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.sample.sampler.HAS_TRITON", False
    )

    assert Sampler.get_vocab_parallel_sampling_params(sampler, input_batch) is None
