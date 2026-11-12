# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup sizing policy for the FlashInfer workspace.

The GPU upper-bound call is FlashInfer's. These tests cover the vLLM policy
around it: explicit settings win, unsupported routes stay native, and a
smaller bound is what gets reserved.
"""

import pytest
import torch

from vllm.v1.attention.backends.flashinfer_workspace import (
    WorkspaceGroup,
    WorkspaceLimits,
    maybe_reserve_flashinfer_workspace,
    native_workspace_bytes,
    reserved_flashinfer_workspace,
    reset_flashinfer_workspace_state,
    select_startup_workspace,
)

_DEFAULT = 394 * 1024 * 1024
_MIB = 1024 * 1024


def _group(**overrides) -> WorkspaceGroup:
    values = dict(
        num_qo_heads=12,
        num_kv_heads=2,
        head_dim_qk=128,
        head_dim_vo=128,
        window_left=-1,
        logits_soft_cap=None,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    values.update(overrides)
    return WorkspaceGroup(**values)


def _limits(**overrides) -> WorkspaceLimits:
    # Qwen2.5-1.5B study shape: 4 sequences, 512 batched tokens, page size 16.
    values = dict(
        max_batch_size=4,
        max_total_num_rows=512,
        max_num_pages_per_request=1032,
        page_size=16,
        kv_layout="NHD",
        decode_cuda_graph=False,
    )
    values.update(overrides)
    return WorkspaceLimits(**values)


def _bound(float_bytes: int, int_bytes: int = 1024):
    def query(kind, group, use_cuda_graph):
        query.calls.append((kind, use_cuda_graph, group.num_qo_heads))
        return float_bytes, int_bytes

    query.calls = []
    return query


def test_native_size_grows_and_does_not_shrink():
    # 8192 * 64 * 128 * 16 = 1 GiB, above the 394 MiB default.
    grown = native_workspace_bytes(
        _DEFAULT,
        max_num_batched_tokens=8192,
        num_qo_heads=64,
        head_dim=128,
        batch_invariant=False,
    )
    assert grown == 8192 * 64 * 128 * 16
    small = native_workspace_bytes(
        _DEFAULT,
        max_num_batched_tokens=512,
        num_qo_heads=12,
        head_dim=128,
        batch_invariant=False,
    )
    assert small == _DEFAULT
    assert (
        native_workspace_bytes(
            1024,
            max_num_batched_tokens=1,
            num_qo_heads=1,
            head_dim=1,
            batch_invariant=True,
        )
        == 2048 * _MIB
    )


def test_auto_size_off_does_not_inspect_the_model(monkeypatch):
    monkeypatch.delenv("VLLM_FLASHINFER_WORKSPACE_AUTO_SIZE", raising=False)
    reset_flashinfer_workspace_state()
    assert maybe_reserve_flashinfer_workspace(object(), torch.device("cpu")) is None


def test_explicit_setting_wins_even_at_the_historical_default():
    query = _bound(32 * _MIB)
    decision = select_startup_workspace(
        configured_bytes=_DEFAULT,
        explicit=True,
        batch_invariant=False,
        groups=[_group()],
        limits=_limits(),
        unsupported_reason=None,
        query=query,
    )
    assert decision.source == "explicit"
    assert decision.selected_bytes == _DEFAULT
    assert decision.fallback_reason is None
    assert query.calls == []


def test_study_geometry_selects_32_mib():
    # ~21.164 MiB float bound from the SM89 FA2 study. Margin and the 32 MiB
    # floor land on 32 MiB, under the 394 MiB default. The prefill floor for
    # this shape is 12 MiB, so it does not keep the larger reservation.
    query = _bound(22_192_668)
    decision = select_startup_workspace(
        configured_bytes=_DEFAULT,
        explicit=False,
        batch_invariant=False,
        groups=[_group()],
        limits=_limits(),
        unsupported_reason=None,
        query=query,
    )
    assert decision.source == "flashinfer"
    assert decision.selected_bytes == 32 * _MIB
    assert decision.float_bound_bytes == 22_192_668
    assert [call[0] for call in query.calls] == ["prefill", "decode"]


def test_cuda_graph_decode_is_queried_when_enabled():
    query = _bound(32 * _MIB)
    select_startup_workspace(
        configured_bytes=_DEFAULT,
        explicit=False,
        batch_invariant=False,
        groups=[_group()],
        limits=_limits(decode_cuda_graph=True),
        unsupported_reason=None,
        query=query,
    )
    assert query.calls == [
        ("prefill", False, 12),
        ("decode", False, 12),
        ("decode", True, 12),
    ]


def test_missing_upper_bound_keeps_native_allocation():
    def query(kind, group, use_cuda_graph):
        raise NotImplementedError("workspace_size_upper_bound is unavailable")

    decision = select_startup_workspace(
        configured_bytes=_DEFAULT,
        explicit=False,
        batch_invariant=False,
        groups=[_group()],
        limits=_limits(),
        unsupported_reason=None,
        query=query,
    )
    assert decision.source == "configured"
    assert decision.selected_bytes == _DEFAULT
    assert "no workspace upper bound" in decision.fallback_reason


def test_unsupported_route_does_not_query():
    query = _bound(32 * _MIB)
    decision = select_startup_workspace(
        configured_bytes=_DEFAULT,
        explicit=False,
        batch_invariant=False,
        groups=[_group()],
        limits=_limits(),
        unsupported_reason="decode context parallel is not supported",
        query=query,
    )
    assert decision.selected_bytes == _DEFAULT
    assert decision.source == "configured"
    assert query.calls == []


def test_does_not_shrink_below_the_prefill_floor():
    # 64 query heads at 8192 tokens is a 1 GiB floor, above both the bound
    # and the 394 MiB default. Native allocation grows; auto-size must not
    # pull it back down to the bound.
    wide = _group(num_qo_heads=64, num_kv_heads=8)
    query = _bound(32 * _MIB)
    decision = select_startup_workspace(
        configured_bytes=_DEFAULT,
        explicit=False,
        batch_invariant=False,
        groups=[_group(), wide],
        limits=_limits(max_total_num_rows=8192),
        unsupported_reason=None,
        query=query,
    )
    assert decision.source == "configured"
    assert decision.selected_bytes == 8192 * 64 * 128 * 16
    assert "does not reduce" in decision.fallback_reason


def test_int_workspace_above_wrapper_scratch_falls_back():
    query = _bound(32 * _MIB, int_bytes=8 * _MIB + 1)
    decision = select_startup_workspace(
        configured_bytes=_DEFAULT,
        explicit=False,
        batch_invariant=False,
        groups=[_group()],
        limits=_limits(),
        unsupported_reason=None,
        query=query,
    )
    assert decision.source == "configured"
    assert decision.selected_bytes == _DEFAULT
    assert "int workspace bound" in decision.fallback_reason


def test_reserved_buffer_is_reused_and_not_grown():
    import vllm.v1.attention.backends.flashinfer_workspace as workspace

    reset_flashinfer_workspace_state()
    workspace._reserved = torch.zeros(128, dtype=torch.uint8)
    try:
        reused = reserved_flashinfer_workspace(torch.device("cpu"), 128)
        assert reused.data_ptr() == workspace._reserved.data_ptr()
        with pytest.raises(RuntimeError, match="VLLM_FLASHINFER_WORKSPACE_AUTO_SIZE"):
            reserved_flashinfer_workspace(torch.device("cpu"), 129)
    finally:
        reset_flashinfer_workspace_state()
