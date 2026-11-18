# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm._aiter_ops import is_aiter_found_and_supported
from vllm.platforms import current_platform

if not current_platform.is_rocm() or not is_aiter_found_and_supported():
    pytest.skip(
        "ROCm AITER sparse indexer test requires ROCm with AITER.",
        allow_module_level=True,
    )

from aiter.ops.triton.attention import sparse_mla as aiter_sparse_mla

import vllm.platforms.rocm as rocm_platform
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.attention import sparse_mla_attention as sparse_mla_mod
from vllm.v1.attention.backends.mla import rocm_aiter_mla_sparse as sparse_backend_mod
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as sparse_mod

UNSET = object()


def test_rocm_fp8_mqa_logits_disables_clean_logits(monkeypatch):
    captured = {}
    expected = torch.empty((1, 1))

    def fake_fp8_mqa_logits(*args, clean_logits):
        captured["clean_logits"] = clean_logits
        return expected

    monkeypatch.setattr(sparse_mod, "_ON_GFX942", False)
    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: True)
    monkeypatch.setattr(
        sparse_mod,
        "mqa_logits_module",
        lambda: SimpleNamespace(fp8_mqa_logits=fake_fp8_mqa_logits),
    )

    q = torch.empty((1, 1, 1))
    k_fp8 = torch.empty((1, 1))
    scale = torch.empty((1,))
    weights = torch.empty((1, 1))
    cu_seqlen_ks = torch.zeros((1,), dtype=torch.int32)
    cu_seqlen_ke = torch.ones((1,), dtype=torch.int32)

    result = sparse_mod.rocm_fp8_mqa_logits(
        q,
        (k_fp8, scale),
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )

    assert result is expected
    assert captured["clean_logits"] is False


@pytest.fixture
def fp8_ds_mla_impl_deps(monkeypatch):
    monkeypatch.setattr(
        sparse_backend_mod.AiterMLAHelper,
        "check_num_heads_validity",
        staticmethod(lambda num_heads: None),
    )
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        speculative_config=None,
        attention_config=SimpleNamespace(hisparse_config=None),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        compilation_config=SimpleNamespace(cudagraph_mode=None),
    )
    monkeypatch.setattr(
        sparse_backend_mod, "get_current_vllm_config", lambda: vllm_config
    )
    monkeypatch.setattr(sparse_mla_mod, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(
        sparse_mla_mod, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(
        sparse_backend_mod,
        "current_workspace_manager",
        lambda: SimpleNamespace(
            get_simultaneous=lambda *specs: tuple(
                torch.zeros(shape, dtype=dtype, device="cpu") for shape, dtype in specs
            )
        ),
    )
    monkeypatch.setattr(rocm_aiter_ops, "is_triton_sparse_mla_enabled", lambda: False)
    sparse_backend_mod._INDEX_EPOCHS.clear()
    yield
    sparse_backend_mod._INDEX_EPOCHS.clear()


def _build_fp8_ds_mla_impl(monkeypatch, cdna_version: int, fnuz_archs):
    monkeypatch.setattr(rocm_platform, "get_cdna_version", lambda: cdna_version)
    if fnuz_archs is UNSET:
        monkeypatch.delattr(aiter_sparse_mla, "FNUZ_DSV32_ARCHS", raising=False)
    else:
        monkeypatch.setattr(
            aiter_sparse_mla, "FNUZ_DSV32_ARCHS", fnuz_archs, raising=False
        )
    return sparse_backend_mod.ROCMAiterMLASparseImpl(
        num_heads=16,
        head_size=576,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8_ds_mla",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        topk_indices_buffer=None,
        indexer=None,
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        v_head_dim=128,
        kv_b_proj=None,
    )


@pytest.mark.parametrize(
    ("cdna_version", "fnuz_archs"),
    [
        (3, ("gfx942",)),
        (4, UNSET),
        (4, ()),
    ],
    ids=["gfx942-fnuz-aiter", "gfx950-old-aiter", "gfx950-new-aiter"],
)
def test_fp8_ds_mla_accepted(
    monkeypatch, fp8_ds_mla_impl_deps, cdna_version, fnuz_archs
):
    impl = _build_fp8_ds_mla_impl(monkeypatch, cdna_version, fnuz_archs)

    assert impl.kv_cache_dtype == "fp8_ds_mla"


@pytest.mark.parametrize(
    ("cdna_version", "fnuz_archs"),
    [
        (3, UNSET),
        (3, ()),
        (3, ("gfx950",)),
        (2, ("gfx942",)),
        (0, ("gfx942",)),
    ],
    ids=[
        "gfx942-old-aiter",
        "gfx942-empty-archs",
        "gfx942-other-archs",
        "gfx90a",
        "non-cdna",
    ],
)
def test_fp8_ds_mla_rejected(
    monkeypatch, fp8_ds_mla_impl_deps, cdna_version, fnuz_archs
):
    with pytest.raises(NotImplementedError, match="fp8_ds_mla"):
        _build_fp8_ds_mla_impl(monkeypatch, cdna_version, fnuz_archs)
