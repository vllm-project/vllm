# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    return module


def _load_rocm_aiter_mla_sparse(monkeypatch):
    class _FakeCurrentPlatform:
        def is_rocm(self) -> bool:
            return False

        def fp8_dtype(self) -> torch.dtype:
            return torch.float8_e4m3fn

    class _FakeTriton:
        def jit(self, fn=None, **kwargs):
            def decorator(inner_fn):
                return inner_fn

            return decorator(fn) if fn is not None else decorator

    tl = SimpleNamespace(constexpr=object)
    stubs = {
        "vllm": _package("vllm"),
        "vllm.forward_context": SimpleNamespace(get_forward_context=lambda: None),
        "vllm.platforms": SimpleNamespace(current_platform=_FakeCurrentPlatform()),
        "vllm.triton_utils": SimpleNamespace(tl=tl, triton=_FakeTriton()),
        "vllm.utils": _package("vllm.utils"),
        "vllm.utils.torch_utils": SimpleNamespace(LayerNameType=str),
        "vllm.v1": _package("vllm.v1"),
        "vllm.v1.attention": _package("vllm.v1.attention"),
        "vllm.v1.attention.backends": _package("vllm.v1.attention.backends"),
        "vllm.v1.attention.backends.mla": _package("vllm.v1.attention.backends.mla"),
        "vllm.v1.attention.backends.mla.indexer": SimpleNamespace(
            DeepseekV32IndexerMetadata=object
        ),
        "vllm.v1.attention.ops": _package("vllm.v1.attention.ops"),
        "vllm.v1.attention.ops.common": SimpleNamespace(
            pack_seq_triton=lambda *args, **kwargs: None,
            unpack_seq_triton=lambda *args, **kwargs: None,
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    source_path = (
        Path(__file__).resolve().parents[3]
        / "vllm"
        / "v1"
        / "attention"
        / "ops"
        / "rocm_aiter_mla_sparse.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_test_rocm_aiter_mla_sparse", source_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_forward_decode_fallback_dequantizes_only_referenced_cache_blocks(monkeypatch):
    module = _load_rocm_aiter_mla_sparse(monkeypatch)

    block_size = 4
    cache_width = 32
    swa_k_cache = torch.arange(6 * block_size * cache_width, dtype=torch.uint8).reshape(
        6, block_size, cache_width
    )
    kv_cache = torch.arange(8 * block_size * cache_width, dtype=torch.uint8).reshape(
        8, block_size, cache_width
    )

    q = torch.zeros((2, 2, 6), dtype=torch.bfloat16)
    output = torch.empty((2, 2, 4), dtype=torch.bfloat16)
    swa_indices = torch.tensor(
        [[2 * block_size + 1, 4 * block_size + 3, -1], [2 * block_size, -1, -1]],
        dtype=torch.int64,
    )
    topk_indices = torch.tensor(
        [[[6 * block_size + 2, -1]], [[3 * block_size + 1, 6 * block_size]]],
        dtype=torch.int64,
    )
    swa_lens = torch.tensor([2, 1], dtype=torch.int32)
    topk_lens = torch.tensor([1, 2], dtype=torch.int32)

    dequant_inputs = []
    captured_decode = {}

    def fake_dequantize(quant_k_cache, *, head_dim, nope_head_dim, rope_head_dim):
        dequant_inputs.append(quant_k_cache.clone())
        return torch.empty(
            (quant_k_cache.shape[0], quant_k_cache.shape[1], 1, head_dim),
            dtype=torch.bfloat16,
        )

    def fake_sparse_decode(
        *,
        q,
        blocked_k,
        indices_in_kvcache,
        topk_length,
        scale,
        head_dim,
        attn_sink,
        extra_blocked_k=None,
        extra_indices_in_kvcache=None,
        extra_topk_length=None,
    ):
        captured_decode["indices_in_kvcache"] = indices_in_kvcache.clone()
        captured_decode["extra_indices_in_kvcache"] = (
            None
            if extra_indices_in_kvcache is None
            else extra_indices_in_kvcache.clone()
        )
        return torch.full((q.shape[0], q.shape[2], head_dim), 7, dtype=torch.bfloat16)

    monkeypatch.setattr(module, "rocm_dequantize_blocked_k_cache", fake_dequantize)
    monkeypatch.setattr(module, "rocm_ref_sparse_attn_decode", fake_sparse_decode)

    module.rocm_forward_decode_fallback(
        q=q,
        kv_cache=kv_cache,
        swa_k_cache=swa_k_cache,
        swa_only=False,
        topk_indices=topk_indices,
        topk_lens=topk_lens,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        attn_sink=None,
        scale=1.0,
        head_dim=4,
        nope_head_dim=4,
        rope_head_dim=2,
        output=output,
    )

    assert len(dequant_inputs) == 2
    torch.testing.assert_close(dequant_inputs[0], swa_k_cache[[2, 4]])
    torch.testing.assert_close(dequant_inputs[1], kv_cache[[3, 6]])

    expected_swa_indices = torch.tensor(
        [[[1, 7, -1]], [[0, -1, -1]]], dtype=torch.int64
    )
    expected_topk_indices = torch.tensor([[[6, -1]], [[1, 4]]], dtype=torch.int64)
    torch.testing.assert_close(
        captured_decode["indices_in_kvcache"], expected_swa_indices
    )
    torch.testing.assert_close(
        captured_decode["extra_indices_in_kvcache"], expected_topk_indices
    )
    torch.testing.assert_close(output, torch.full_like(output, 7))
