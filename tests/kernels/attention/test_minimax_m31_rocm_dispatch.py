# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm MiniMax-M3.1 dispatch: FP8 MSA/indexer, reject NVFP4 KV/indexer."""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("uvloop")

from vllm.models.minimax_m3.amd.indexer_msa import msa_indexer_unsupported_reason
from vllm.models.minimax_m3.common.indexer import select_indexer_impl_cls
from vllm.models.minimax_m3.common.rocm_compat import MINIMAX_M31_ROCM_RECIPE
from vllm.models.minimax_m3.common.sparse_attention import (
    select_main_backend_and_impl_cls,
)


@pytest.mark.cpu_test
@pytest.mark.parametrize("kv_cache_dtype", ["nvfp4", "nvfp4_4over6"])
def test_nvfp4_kv_rejected_on_rocm(
    monkeypatch: pytest.MonkeyPatch, kv_cache_dtype: str
) -> None:
    monkeypatch.setattr(
        "vllm.models.minimax_m3.common.sparse_attention.current_platform.is_rocm",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.models.minimax_m3.common.sparse_attention.current_platform.is_cuda",
        lambda: False,
    )
    with pytest.raises(ValueError, match="not supported on ROCm"):
        select_main_backend_and_impl_cls(
            topk_blocks=16,
            kv_cache_dtype=kv_cache_dtype,
            num_kv_heads=1,
        )


@pytest.mark.cpu_test
def test_fp8_kv_is_not_rejected_as_nvfp4_on_rocm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm.models.minimax_m3.common.sparse_attention.current_platform.is_rocm",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.models.minimax_m3.common.sparse_attention.current_platform.is_cuda",
        lambda: False,
    )
    monkeypatch.setattr(
        "vllm.models.minimax_m3.common.sparse_attention.minimax_m3_use_aiter_sparse_pa",
        lambda *args, **kwargs: False,
    )
    backend, impl = select_main_backend_and_impl_cls(
        topk_blocks=16,
        kv_cache_dtype="fp8",
        num_kv_heads=1,
    )
    assert backend.__name__ == "MiniMaxM3SparseBackend"
    assert impl.__name__ == "MiniMaxM3SparseTritonImpl"


@pytest.mark.cpu_test
def test_nvfp4_indexer_reason_on_rocm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "vllm.models.minimax_m3.amd.indexer_msa.current_platform.is_rocm",
        lambda: True,
    )
    reason = msa_indexer_unsupported_reason(
        topk_blocks=16,
        sparse_block_size=128,
        index_head_dim=128,
        indexer_kv_dtype="nvfp4",
        max_model_len=32768,
        score_type="max",
    )
    assert reason is not None
    assert "not supported on ROCm" in reason
    assert MINIMAX_M31_ROCM_RECIPE in reason


@pytest.mark.cpu_test
def test_fp8_indexer_reason_is_not_the_nvfp4_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm.models.minimax_m3.amd.indexer_msa.current_platform.is_rocm",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.models.minimax_m3.amd.indexer_msa._minimax_m3_aiter_sparse_pa_requested",
        lambda: False,
    )
    reason = msa_indexer_unsupported_reason(
        topk_blocks=16,
        sparse_block_size=128,
        index_head_dim=128,
        indexer_kv_dtype="fp8",
        max_model_len=32768,
        score_type="max",
    )
    assert reason is not None
    assert MINIMAX_M31_ROCM_RECIPE not in reason
    assert "AITER sparse PA" in reason


@pytest.mark.cpu_test
def test_select_indexer_impl_rejects_nvfp4_on_rocm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm.models.minimax_m3.common.indexer.current_platform.is_rocm",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.models.minimax_m3.common.indexer.current_platform.is_cuda",
        lambda: False,
    )
    fake_cfg = MagicMock()
    fake_cfg.attention_config.minimax_m3_indexer_cp = False
    monkeypatch.setattr(
        "vllm.models.minimax_m3.common.indexer.get_current_vllm_config",
        lambda: fake_cfg,
    )
    with pytest.raises(NotImplementedError, match="not supported on ROCm"):
        select_indexer_impl_cls(topk_blocks=16, indexer_kv_dtype="nvfp4")
