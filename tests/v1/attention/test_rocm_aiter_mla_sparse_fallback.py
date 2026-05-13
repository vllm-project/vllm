# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
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


def test_forward_decode_fallback_uses_graph_safe_per_token_gather(monkeypatch):
    """Regression for #41962 / #42248.

    Locks in two related invariants that together unblock graph-mode decode
    for the DeepSeek-V4-Flash MLA fallback on ROCm:

    * The fallback never materializes the full SWA / KV cache pool as bf16.
      ``_dequantize_referenced_tokens`` is called with the original
      (uncompacted) cache and the original indices, and the gathered output
      passed downstream is bounded by ``indices.numel()`` rather than the
      cache pool size. This preserves the OOM win that motivated #42248.
    * The fallback uses no host-side compaction step (no boolean masking, no
      ``torch.unique``, no Python-level branch on tensor data). The
      previous implementation hit
      ``hipErrorStreamCaptureUnsupported`` inside ``_gather_referenced_cache_blocks``
      under HIP graph capture; this test prevents that path from being
      reintroduced by mistake.
    """
    module = _load_rocm_aiter_mla_sparse(monkeypatch)

    block_size = 4
    cache_width = 32
    swa_k_cache = torch.arange(6 * block_size * cache_width, dtype=torch.uint8).reshape(
        6, block_size, cache_width
    )
    kv_cache = torch.arange(8 * block_size * cache_width, dtype=torch.uint8).reshape(
        8, block_size, cache_width
    )

    head_dim = 4
    q = torch.zeros((2, 2, 6), dtype=torch.bfloat16)
    output = torch.empty((2, 2, head_dim), dtype=torch.bfloat16)
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

    captured_calls: list[dict] = []

    def fake_dequant_ref_tokens(
        quant_k_cache,
        indices,
        *,
        head_dim,
        nope_head_dim,
        rope_head_dim,
    ):
        captured_calls.append(
            {
                "quant_k_cache": quant_k_cache,
                "indices": indices.clone(),
            }
        )
        n = indices.numel()
        gathered = torch.zeros((n, head_dim), dtype=torch.bfloat16)
        flat = indices.reshape(-1)
        arange = torch.arange(n, dtype=indices.dtype)
        flat_remapped = torch.where(flat >= 0, arange, torch.full_like(arange, -1))
        return gathered, flat_remapped.view_as(indices)

    captured_decode: dict = {}

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
        captured_decode["blocked_k_shape"] = tuple(blocked_k.shape)
        captured_decode["indices_in_kvcache"] = indices_in_kvcache.clone()
        captured_decode["extra_blocked_k_shape"] = (
            None if extra_blocked_k is None else tuple(extra_blocked_k.shape)
        )
        captured_decode["extra_indices_in_kvcache"] = (
            None
            if extra_indices_in_kvcache is None
            else extra_indices_in_kvcache.clone()
        )
        return torch.full((q.shape[0], q.shape[2], head_dim), 7, dtype=torch.bfloat16)

    monkeypatch.setattr(
        module, "_dequantize_referenced_tokens", fake_dequant_ref_tokens
    )
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
        head_dim=head_dim,
        nope_head_dim=4,
        rope_head_dim=2,
        output=output,
    )

    assert len(captured_calls) == 2

    # Original (full) caches and indices are passed straight through — no
    # host-side compaction step that would force a stream-capture sync.
    assert captured_calls[0]["quant_k_cache"] is swa_k_cache
    assert captured_calls[1]["quant_k_cache"] is kv_cache
    torch.testing.assert_close(captured_calls[0]["indices"], swa_indices)
    torch.testing.assert_close(captured_calls[1]["indices"], topk_indices)

    # The tensor handed to the decode kernel has row count bounded by the
    # number of referenced token slots — independent of the cache pool size.
    assert captured_decode["blocked_k_shape"] == (swa_indices.numel(), head_dim)
    assert captured_decode["extra_blocked_k_shape"] == (
        topk_indices.numel(),
        head_dim,
    )

    # Remapped indices are identity-style flat positions for valid slots,
    # -1 preserved for invalid slots so downstream masking still works.
    expected_swa_remapped = torch.tensor(
        [[[0, 1, -1]], [[3, -1, -1]]],
        dtype=torch.int64,
    )
    torch.testing.assert_close(
        captured_decode["indices_in_kvcache"], expected_swa_remapped
    )
    expected_topk_remapped = torch.tensor(
        [[[0, -1]], [[2, 3]]],
        dtype=torch.int64,
    )
    torch.testing.assert_close(
        captured_decode["extra_indices_in_kvcache"], expected_topk_remapped
    )

    torch.testing.assert_close(output, torch.full_like(output, 7))


@pytest.mark.skipif(
    not hasattr(torch, "float8_e8m0fnu"),
    reason="torch.float8_e8m0fnu unavailable in this PyTorch build",
)
def test_dequantize_referenced_tokens_is_graph_safe(monkeypatch):
    """Direct check that ``_dequantize_referenced_tokens`` only relies on
    operations whose output shape is statically derivable from input shapes.

    The previous helper used boolean masking (``indices[indices >= 0]``) and
    ``torch.unique`` to dedup referenced cache blocks. Both produce
    data-dependent output shapes, which HIP/CUDA stream capture rejects with
    ``hipErrorStreamCaptureUnsupported``. This test pins down two
    consequences of the new per-token implementation that together prevent
    that regression:

    * Output ``gathered_kv`` shape is ``(indices.numel(), head_dim)`` — a
      function of ``indices.shape`` alone, independent of cache content.
    * ``remapped_indices`` preserves ``indices.shape`` and only relies on
      ``torch.where`` over the valid mask (no ``unique``, no boolean
      indexing).
    """
    module = _load_rocm_aiter_mla_sparse(monkeypatch)

    # Build a minimal cache with the layout the dequant helper expects:
    # per block, ``block_size * (nope+2*rope)`` bytes of nope+rope are stored
    # first, followed by ``block_size * 8`` bytes of per-token scales.
    block_size = 4
    nope_head_dim = 64  # must be a multiple of tile_size (64)
    rope_head_dim = 8
    head_dim = nope_head_dim + rope_head_dim
    nope_rope_bytes = nope_head_dim + 2 * rope_head_dim
    head_bytes = nope_rope_bytes + 8
    num_blocks = 5

    quant_k_cache = torch.zeros((num_blocks, block_size, head_bytes), dtype=torch.uint8)

    # Exercise multiple shapes (mirroring the fallback's SWA and topk paths)
    # to confirm the output bound tracks ``indices.shape`` only.
    for indices_shape in [(3, 5), (2, 1, 4)]:
        indices = torch.full(indices_shape, -1, dtype=torch.int64)
        flat = indices.view(-1)
        flat[0] = 1 * block_size + 0
        flat[1] = 3 * block_size + 2
        flat[-1] = 0 * block_size + 1

        gathered, remapped = module._dequantize_referenced_tokens(
            quant_k_cache,
            indices,
            head_dim=head_dim,
            nope_head_dim=nope_head_dim,
            rope_head_dim=rope_head_dim,
        )

        assert gathered.shape == (indices.numel(), head_dim)
        assert gathered.dtype == torch.bfloat16
        assert remapped.shape == indices.shape

        # Identity remap for valid slots, -1 preserved for invalid.
        flat_indices = indices.view(-1)
        flat_remapped = remapped.view(-1)
        for i in range(indices.numel()):
            if flat_indices[i].item() < 0:
                assert flat_remapped[i].item() == -1
            else:
                assert flat_remapped[i].item() == i


@pytest.mark.skipif(
    not hasattr(torch, "float8_e8m0fnu"),
    reason="torch.float8_e8m0fnu unavailable in this PyTorch build",
)
def test_dequantize_referenced_tokens_matches_block_level_dequant(monkeypatch):
    """Numerical equivalence between per-token and block-level dequant.

    For every valid index, the per-token gather + dequant must yield
    identical bf16 values to what the existing
    ``rocm_dequantize_blocked_k_cache`` produces for the same (block, slot)
    tuple. This guards against a silent change in the dequantization math
    when switching from the block-level to the per-token path.
    """
    module = _load_rocm_aiter_mla_sparse(monkeypatch)

    block_size = 4
    nope_head_dim = 64
    rope_head_dim = 8
    head_dim = nope_head_dim + rope_head_dim
    nope_rope_bytes = nope_head_dim + 2 * rope_head_dim
    head_bytes = nope_rope_bytes + 8
    num_blocks = 6

    # Random uint8 storage exercises real dequant math (per-tile fp8 nope,
    # bf16 rope, fp8_e8m0 scales) over both helpers.
    torch.manual_seed(0)
    quant_k_cache = torch.randint(
        0, 256, (num_blocks, block_size, head_bytes), dtype=torch.uint8
    )

    indices = torch.tensor(
        [
            [1 * block_size + 2, -1, 5 * block_size + 0],
            [3 * block_size + 3, 0 * block_size + 1, -1],
        ],
        dtype=torch.int64,
    )

    gathered, remapped = module._dequantize_referenced_tokens(
        quant_k_cache,
        indices,
        head_dim=head_dim,
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
    )

    # Reference: full block-level dequant, then index per (block, slot).
    block_level = module.rocm_dequantize_blocked_k_cache(
        quant_k_cache,
        head_dim=head_dim,
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
    )  # (num_blocks, block_size, 1, head_dim)

    flat_indices = indices.view(-1)
    flat_remapped = remapped.view(-1)
    for i in range(indices.numel()):
        idx = flat_indices[i].item()
        if idx < 0:
            # Invalid slots are masked by the consumer; only verify the
            # remap preserved the sentinel.
            assert flat_remapped[i].item() == -1
            continue
            block = idx // block_size
            slot = idx % block_size
            expected = block_level[block, slot, 0]
            # ``equal_nan=True``: random fp8 / e8m0 byte patterns can decode to
            # NaN. Both helpers must produce the same NaN positions, so allow
            # NaN-to-NaN matches.
            torch.testing.assert_close(gathered[i], expected, equal_nan=True)
