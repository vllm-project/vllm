# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.worker import cp_utils


class _DummyImplNoDCP:
    """Simulates an attention impl that does NOT support DCP."""

    need_to_return_lse_for_decode = False
    supports_pcp = True
    supports_mtp_with_cp_non_trivial_interleave_size = True


class _DummyImplNoPCP:
    """Simulates an attention impl that does NOT support PCP."""

    need_to_return_lse_for_decode = True
    supports_pcp = False
    supports_mtp_with_cp_non_trivial_interleave_size = True


class _DummyImplNoMTPInterleave:
    """Simulates an attention impl that does NOT support MTP with
    cp_kv_cache_interleave_size > 1."""

    need_to_return_lse_for_decode = True
    supports_pcp = True
    supports_mtp_with_cp_non_trivial_interleave_size = False


class _DummyImplFullSupport:
    """Simulates an attention impl that supports all CP features."""

    need_to_return_lse_for_decode = True
    supports_pcp = True
    supports_mtp_with_cp_non_trivial_interleave_size = True


class _DummyLayer:
    def __init__(self, impl):
        self.impl = impl


def _make_vllm_config(
    dcp_size=1,
    pcp_size=1,
    interleave_size=1,
    speculative_config=None,
):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            decode_context_parallel_size=dcp_size,
            cp_kv_cache_interleave_size=interleave_size,
        ),
        speculative_config=speculative_config,
    )


def test_dcp_error_includes_backend_guidance(monkeypatch: pytest.MonkeyPatch):
    """DCP error should mention --attention-backend and list compatible
    backends."""
    vllm_config = _make_vllm_config(dcp_size=2)
    monkeypatch.setattr(
        cp_utils,
        "get_layers_from_vllm_config",
        lambda *_: {"layer.0": _DummyLayer(_DummyImplNoDCP())},
    )

    with pytest.raises(AssertionError, match="--attention-backend") as exc:
        cp_utils.check_attention_cp_compatibility(vllm_config)

    msg = str(exc.value)
    # Should mention the failing impl name
    assert "_DummyImplNoDCP" in msg
    # Should mention the CLI flag
    assert "--attention-backend" in msg
    # Should list at least one compatible backend
    assert "FLASH_ATTN" in msg
    assert "FLASHINFER" in msg


def test_pcp_error_includes_backend_guidance(monkeypatch: pytest.MonkeyPatch):
    """PCP error should mention --attention-backend and the impl name."""
    vllm_config = _make_vllm_config(pcp_size=2)
    monkeypatch.setattr(
        cp_utils,
        "get_layers_from_vllm_config",
        lambda *_: {"layer.0": _DummyLayer(_DummyImplNoPCP())},
    )

    with pytest.raises(AssertionError, match="--attention-backend") as exc:
        cp_utils.check_attention_cp_compatibility(vllm_config)

    msg = str(exc.value)
    assert "_DummyImplNoPCP" in msg
    assert "--attention-backend" in msg


def test_mtp_interleave_error_includes_backend_guidance(
    monkeypatch: pytest.MonkeyPatch,
):
    """MTP interleave error should mention --attention-backend and the
    impl name."""
    vllm_config = _make_vllm_config(
        dcp_size=2,
        interleave_size=2,
        speculative_config=SimpleNamespace(),
    )
    monkeypatch.setattr(
        cp_utils,
        "get_layers_from_vllm_config",
        lambda *_: {"layer.0": _DummyLayer(_DummyImplNoMTPInterleave())},
    )

    with pytest.raises(AssertionError, match="--attention-backend") as exc:
        cp_utils.check_attention_cp_compatibility(vllm_config)

    msg = str(exc.value)
    assert "_DummyImplNoMTPInterleave" in msg
    assert "--attention-backend" in msg


def test_no_error_when_backend_supports_all_cp(
    monkeypatch: pytest.MonkeyPatch,
):
    """No error should be raised when the backend supports all CP
    features."""
    vllm_config = _make_vllm_config(
        dcp_size=2,
        pcp_size=2,
        interleave_size=2,
        speculative_config=SimpleNamespace(),
    )
    monkeypatch.setattr(
        cp_utils,
        "get_layers_from_vllm_config",
        lambda *_: {"layer.0": _DummyLayer(_DummyImplFullSupport())},
    )

    # Should not raise any exception
    cp_utils.check_attention_cp_compatibility(vllm_config)
