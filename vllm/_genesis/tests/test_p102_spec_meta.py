# SPDX-License-Identifier: Apache-2.0
"""Unit tests for P102 — Genesis spec_meta unified metadata module.

Pure-Python tests on the dataclass + predicates + thread-local context.
No GPU/vLLM dependency — tests dispatch logic in isolation.

Covers:
  - GenesisSpecMeta defaults
  - current() / set_step() lifecycle
  - is_active() env var recognition
  - should_dispatch_p67 — full truth table
  - should_use_perlayer_workspace
  - should_skip_tolist
  - should_use_workspace_cache
  - Phase 1 disagreement detection
  - get_telemetry() / log_telemetry_summary
"""
from __future__ import annotations

import pytest

from vllm._genesis.spec_meta import (
    GenesisSpecMeta,
    current,
    get_telemetry,
    is_active,
    reset_for_tests,
    set_step,
    should_dispatch_p67,
    should_skip_tolist,
    should_use_perlayer_workspace,
    should_use_workspace_cache,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_for_tests()
    yield
    reset_for_tests()


# ─── Dataclass defaults ─────────────────────────────────────────────────


def test_genesis_spec_meta_defaults():
    m = GenesisSpecMeta()
    assert m.is_cuda_graph_capture is False
    assert m.is_cuda_graph_replay is False
    assert m.current_query_len == 1
    assert m.spec_method == "none"
    assert m.runtime_K == 0
    assert m.batch_size == 0
    assert m.is_chunked_prefill is False
    assert m.step_index == 0


def test_current_lazy_init():
    """current() returns default GenesisSpecMeta if unset."""
    m = current()
    assert isinstance(m, GenesisSpecMeta)
    assert m.spec_method == "none"


def test_set_step_replaces_context():
    new_meta = GenesisSpecMeta(
        spec_method="mtp", runtime_K=3, current_query_len=4, step_index=42
    )
    set_step(new_meta)
    m = current()
    assert m.spec_method == "mtp"
    assert m.runtime_K == 3
    assert m.current_query_len == 4
    assert m.step_index == 42


# ─── is_active ──────────────────────────────────────────────────────────


def test_is_active_truthy(monkeypatch):
    for v in ("1", "true", "yes", "on", "True"):
        monkeypatch.setenv("GENESIS_ENABLE_P102", v)
        assert is_active(), f"{v!r} should activate"
    for v in ("0", "", "off", "no"):
        monkeypatch.setenv("GENESIS_ENABLE_P102", v)
        assert not is_active(), f"{v!r} should NOT activate"


# ─── should_dispatch_p67 — happy path + each guard ──────────────────────


def _spec_active() -> GenesisSpecMeta:
    """Common 'spec-decode active' state for happy-path tests."""
    return GenesisSpecMeta(
        runtime_K=3,
        spec_method="mtp",
        current_query_len=4,
        is_chunked_prefill=False,
    )


def test_p67_happy_path():
    set_step(_spec_active())
    assert should_dispatch_p67(
        Hq=64, Hk=8, head_size=128,
        max_query_len=4, max_seq_len=2048, N=8,
    )


def test_p67_off_when_runtime_K_zero():
    set_step(GenesisSpecMeta(runtime_K=0, spec_method="none"))
    assert not should_dispatch_p67(
        Hq=64, Hk=8, head_size=128,
        max_query_len=4, max_seq_len=2048, N=8,
    )


def test_p67_off_when_chunked_prefill():
    """v756 regression class: chunked-prefill must NEVER dispatch P67."""
    m = _spec_active()
    m.is_chunked_prefill = True
    set_step(m)
    assert not should_dispatch_p67(
        Hq=64, Hk=8, head_size=128,
        max_query_len=4, max_seq_len=2048, N=8,
    )


def test_p67_off_when_unsupported_layer():
    set_step(_spec_active())
    assert not should_dispatch_p67(
        Hq=64, Hk=8, head_size=128,
        max_query_len=4, max_seq_len=2048, N=8,
        layer_kind="flashinfer",  # not turboquant
    )


def test_p67_off_when_unsupported_method():
    m = _spec_active()
    m.spec_method = "synthetic"  # not in supported set
    set_step(m)
    assert not should_dispatch_p67(
        Hq=64, Hk=8, head_size=128,
        max_query_len=4, max_seq_len=2048, N=8,
    )


def test_p67_off_when_Hq_lt_8():
    set_step(_spec_active())
    assert not should_dispatch_p67(
        Hq=4, Hk=2, head_size=128,
        max_query_len=4, max_seq_len=2048, N=8,
    )


def test_p67_off_when_unsupported_head_size():
    set_step(_spec_active())
    for D in (32, 64, 96, 192, 512):
        assert not should_dispatch_p67(
            Hq=64, Hk=8, head_size=D,
            max_query_len=4, max_seq_len=2048, N=8,
        ), f"head_size={D} should be rejected"


def test_p67_on_for_head_size_256():
    set_step(_spec_active())
    assert should_dispatch_p67(
        Hq=64, Hk=8, head_size=256,
        max_query_len=4, max_seq_len=2048, N=8,
    )


def test_p67_off_when_GQA_lt_2():
    set_step(_spec_active())
    assert not should_dispatch_p67(
        Hq=8, Hk=8, head_size=128,  # GQA=1
        max_query_len=4, max_seq_len=2048, N=8,
    )


def test_p67_off_for_pure_decode():
    """max_query_len=1 means pure decode — no spec-verify."""
    set_step(_spec_active())
    assert not should_dispatch_p67(
        Hq=64, Hk=8, head_size=128,
        max_query_len=1, max_seq_len=2048, N=2,
    )


def test_p67_off_above_kp1_cap():
    set_step(_spec_active())
    assert not should_dispatch_p67(
        Hq=64, Hk=8, head_size=128,
        max_query_len=17, max_seq_len=2048, N=17,
        max_kp1=16,
    )


def test_p67_off_when_no_prior_cache():
    """max_seq_len == max_query_len → first-chunk prefill — no cached KV."""
    set_step(_spec_active())
    assert not should_dispatch_p67(
        Hq=64, Hk=8, head_size=128,
        max_query_len=4, max_seq_len=4, N=8,
    )


def test_p67_off_when_prior_exceeds_baked_max():
    """prior_len > max_prior → reject. Boundary: prior_len == max_prior is OK."""
    set_step(_spec_active())
    # max_seq_len=4101, max_query=4 → prior=4097 > 4096 → reject
    assert not should_dispatch_p67(
        Hq=64, Hk=8, head_size=128,
        max_query_len=4, max_seq_len=4101, N=8,
        max_prior=4096,
    )
    # max_seq_len=4100, prior=4096 == max_prior → accept (boundary inclusive)
    assert should_dispatch_p67(
        Hq=64, Hk=8, head_size=128,
        max_query_len=4, max_seq_len=4100, N=8,
        max_prior=4096,
    )


def test_p67_off_when_N_zero():
    set_step(_spec_active())
    assert not should_dispatch_p67(
        Hq=64, Hk=8, head_size=128,
        max_query_len=4, max_seq_len=2048, N=0,
    )


def test_p67_off_when_N_not_uniform_kp1():
    """Uniform K+1 layout requires N % max_query_len == 0."""
    set_step(_spec_active())
    assert not should_dispatch_p67(
        Hq=64, Hk=8, head_size=128,
        max_query_len=4, max_seq_len=2048, N=10,  # 10 % 4 = 2
    )


# ─── Other predicates ──────────────────────────────────────────────────


def test_should_use_perlayer_workspace():
    set_step(GenesisSpecMeta(is_cuda_graph_replay=True))
    assert should_use_perlayer_workspace() is True

    set_step(GenesisSpecMeta(is_cuda_graph_replay=False))
    assert should_use_perlayer_workspace() is False


def test_should_skip_tolist():
    set_step(GenesisSpecMeta(is_cuda_graph_capture=True))
    assert should_skip_tolist() is True

    set_step(GenesisSpecMeta(is_cuda_graph_capture=False))
    assert should_skip_tolist() is False


def test_should_use_workspace_cache():
    """P99 logic: cache when NOT in capture."""
    set_step(GenesisSpecMeta(is_cuda_graph_capture=False))
    assert should_use_workspace_cache() is True

    set_step(GenesisSpecMeta(is_cuda_graph_capture=True))
    assert should_use_workspace_cache() is False


# ─── Phase 1 disagreement detection ────────────────────────────────────


def test_disagreement_logged_via_telemetry():
    """When inline check disagrees with predicate, telemetry must record."""
    set_step(_spec_active())
    # predicate says True (happy path), but pretend inline says False
    should_dispatch_p67(
        Hq=64, Hk=8, head_size=128,
        max_query_len=4, max_seq_len=2048, N=8,
        inline_decision=False,
    )
    t = get_telemetry()
    assert t["disagreement_count"] == 1
    assert t["predicate_calls"] >= 1


def test_no_disagreement_when_aligned():
    set_step(_spec_active())
    should_dispatch_p67(
        Hq=64, Hk=8, head_size=128,
        max_query_len=4, max_seq_len=2048, N=8,
        inline_decision=True,
    )
    t = get_telemetry()
    assert t["disagreement_count"] == 0


# ─── Telemetry ─────────────────────────────────────────────────────────


def test_telemetry_reflects_current_state():
    set_step(GenesisSpecMeta(
        spec_method="mtp", runtime_K=3, current_query_len=4,
        is_cuda_graph_capture=True, step_index=99,
    ))
    t = get_telemetry()
    assert t["spec_method"] == "mtp"
    assert t["runtime_K"] == 3
    assert t["current_query_len"] == 4
    assert t["is_cuda_graph_capture"] is True
    assert t["step_index"] == 99
