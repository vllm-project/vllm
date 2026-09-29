# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-runnable selection tests for MiniMax-M3 AITER sparse PA (#52860).

MiniMax-M3 package init imports the NVIDIA model, which needs compiled
``vllm._C``. Register a namespace package first so these tests can import the
selection helpers without that extension.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skip_global_cleanup


def _ensure_minimax_m3_importable() -> None:
    """Import MiniMax-M3 helpers without requiring compiled ``vllm._C``.

    Package init loads the NVIDIA model, which needs the CUDA/ROCm extension.
    If that import fails, register a namespace package so the selection
    helpers can still be unit-tested on CPU.
    """
    name = "vllm.models.minimax_m3"
    try:
        import vllm.models.minimax_m3  # noqa: F401

        return
    except Exception:
        pass
    import vllm.models  # noqa: F401

    root = Path(__file__).resolve().parents[3] / "vllm" / "models" / "minimax_m3"
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(root)]
    pkg.__file__ = str(root / "__init__.py")
    pkg.__package__ = name
    sys.modules[name] = pkg


_ensure_minimax_m3_importable()

from vllm.models.minimax_m3.amd.sparse_attention_msa import (  # noqa: E402
    MiniMaxM3SparseAiterPAImpl,
)
from vllm.models.minimax_m3.common.sparse_attention import (  # noqa: E402
    MiniMaxM3SparseBackend,
    MiniMaxM3SparseTritonImpl,
    minimax_m3_use_aiter_sparse_pa,
    select_main_backend_and_impl_cls,
)
from vllm.v1.kv_cache_interface import KVCacheLayout  # noqa: E402


def _enable_aiter_sparse_pa(monkeypatch, *, speculative: bool) -> None:
    import vllm.models.minimax_m3.common.sparse_attention as sparse_attn_mod

    monkeypatch.setattr(sparse_attn_mod.rocm_aiter_ops, "is_enabled", lambda: True)
    monkeypatch.setattr(
        sparse_attn_mod.rocm_aiter_ops,
        "is_shuffle_kv_cache_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        sparse_attn_mod,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(speculative_config=object() if speculative else None),
    )


def test_aiter_sparse_pa_used_without_speculative_decoding(monkeypatch):
    """AITER + shuffle KV still selects the prototype when spec decode is off."""
    _enable_aiter_sparse_pa(monkeypatch, speculative=False)

    assert minimax_m3_use_aiter_sparse_pa(1) is True
    assert MiniMaxM3SparseBackend.supported_kv_cache_layouts() == (
        KVCacheLayout.LBHNC,
        KVCacheLayout.LHBNC,
    )
    with pytest.raises(ValueError, match="num_kv_heads == 1"):
        minimax_m3_use_aiter_sparse_pa(2)

    _, impl_cls = select_main_backend_and_impl_cls(
        topk_blocks=16,
        kv_cache_dtype="fp8",
        num_kv_heads=1,
    )
    assert impl_cls is MiniMaxM3SparseAiterPAImpl


def test_aiter_sparse_pa_skipped_under_speculative_decoding(monkeypatch):
    """Spec decode must not silently use the AITER sparse PA prototype (#52860).

    KV layout and impl selection share this gate; both must fall back together
    so the Triton path is not paired with AITER's separated K/V storage.
    """
    _enable_aiter_sparse_pa(monkeypatch, speculative=True)

    assert minimax_m3_use_aiter_sparse_pa(1) is False
    assert MiniMaxM3SparseBackend.supported_kv_cache_layouts() is None
    # Triton supports GQA; do not inherit the AITER num_kv_heads == 1 check.
    assert minimax_m3_use_aiter_sparse_pa(2) is False

    backend, impl_cls = select_main_backend_and_impl_cls(
        topk_blocks=16,
        kv_cache_dtype="fp8",
        num_kv_heads=1,
    )
    assert backend is MiniMaxM3SparseBackend
    assert impl_cls is MiniMaxM3SparseTritonImpl
