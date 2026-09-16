# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone CPU interpreter correctness tests (no GPU or model weights).

TRITON_INTERPRET=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -m pytest --noconftest -o addopts= -v -s <this file>

The public launcher must preserve each token's selected, causal keys despite
union reordering, different membership, partial tiles and paged/scaled caches.
Nearby suites require CUDA; this file can also be copied outside the checkout
and pointed at the patched module with MINIMAX_SPARSE_ATTN_SOURCE.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch

if os.getenv("TRITON_INTERPRET") != "1":
    pytest.skip("Run with TRITON_INTERPRET=1 on CPU", allow_module_level=True)

if source_override := os.getenv("MINIMAX_SPARSE_ATTN_SOURCE"):
    SOURCE = Path(source_override)
else:
    SOURCE = (
        Path(__file__).resolve().parents[3]
        / "vllm/models/minimax_m3/common/ops/sparse_attn.py"
    )
spec = importlib.util.spec_from_file_location("sparse_attn_under_test", SOURCE)
assert spec is not None and spec.loader is not None
attn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(attn)
torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def _bf16_interpreter_support(monkeypatch):
    """Adapt Triton 3.7.1's unsupported bf16 dot and default downcast.

    create_dot treats bf16 storage as uint16; cast_impl truncates fp32->bf16
    instead of the GPU's default round-to-nearest-even. Adapt these two primitive
    operations for BOTH unchanged and tiled kernels. All loads, control flow,
    masks, softmax and stores still run in the interpreter.
    Set MINIMAX_TEST_RAW_BF16=1 to reproduce the upstream interpreter failure.
    This fixture never changes the production kernel or a GPU execution path.
    """
    if os.getenv("MINIMAX_TEST_RAW_BF16") == "1":
        return
    from triton.runtime.interpreter import TensorHandle, interpreter_builder

    original = interpreter_builder.create_dot
    original_cast = interpreter_builder.cast_impl

    def decode(value):
        if value.dtype == attn.tl.bfloat16:
            data = (value.data.astype(np.uint32) << 16).view(np.float32)
            return TensorHandle(data, attn.tl.float32)
        return value

    def dot(a, b, d, input_precision, max_num_imprecise_acc):
        return original(decode(a), decode(b), d, input_precision, max_num_imprecise_acc)

    def cast(src, dst_type):
        if src.dtype.scalar == attn.tl.float32 and dst_type.scalar == attn.tl.bfloat16:
            data = torch.from_numpy(src.data.copy()).to(torch.bfloat16)
            return TensorHandle(data.view(torch.uint16).numpy(), attn.tl.bfloat16)
        return original_cast(src, dst_type)

    # Check exact storage decoding independently against PyTorch before use.
    sample = torch.tensor([-3.5, -0.125, 0.0, 0.75, 2.0], dtype=torch.bfloat16)
    handle = TensorHandle(sample.view(torch.uint16).numpy(), attn.tl.bfloat16)
    np.testing.assert_array_equal(decode(handle).data, sample.float().numpy())
    monkeypatch.setattr(interpreter_builder, "create_dot", dot)
    monkeypatch.setattr(interpreter_builder, "cast_impl", cast)


def _case(tile, group, dtype, pattern, kv_dtype=None, scale_mode=0):
    torch.manual_seed(100 + group + tile)
    if pattern == "overlap":
        prefixes, lengths, topk, init, local = (
            [0, 123, 4093],
            [tile + 3, 9, 5],
            16,
            0,
            1,
        )
    elif pattern == "varied":
        prefixes, lengths, topk, init, local = [0, 255, 8191], [3, tile + 1, 7], 7, 2, 2
    elif pattern == "long":
        prefixes, lengths, topk, init, local = [65531, 1048573], [7, 3], 16, 0, 1
    else:
        raise AssertionError(pattern)
    heads, dim = (2 if group == 4 else 1), 128
    total = sum(lengths)
    # Slice all metadata and data to exercise the mixed-batch call site's
    # nonzero offsets, transposed top-k layout, and noncontiguous strides.
    q = torch.randn(total + 2, heads * group, dim * 2, dtype=dtype)[1:-1, :, ::2]
    out = torch.full_like(q, float("nan"), memory_format=torch.contiguous_format)
    seq = torch.tensor([p + n for p, n in zip(prefixes, lengths)], dtype=torch.int32)
    prefix = torch.tensor([99, *prefixes], dtype=torch.int32)[1:]
    seq = torch.cat((torch.tensor([99], dtype=torch.int32), seq))[1:]
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    max_blocks = (int(seq.max()) + 127) // 128
    # Long-context tests reuse physical pages to keep CPU memory bounded.
    pages = min(max_blocks * len(lengths), 192)
    bt = torch.stack([torch.randperm(max_blocks) % pages for _ in lengths]).int()
    # Both launchers assume unit stride within a block-table row.
    bt = torch.cat((bt, bt), dim=1)[:, :max_blocks]
    cache = torch.randn(pages, heads, 128, 2 * dim).to(kv_dtype or dtype)
    ks = vs = None
    if scale_mode == 1:
        ks, vs = torch.tensor(0.7), torch.tensor(1.3)
    elif scale_mode == 2:
        ks = (torch.rand(heads, pages * 128 * 2) * 0.6 + 0.5)[:, ::2]
        vs = (torch.rand(heads, pages * 128 * 2) * 0.8 + 0.7)[:, ::2]
    top = torch.full((total + 2, heads, topk * 2), -1, dtype=torch.int32)
    top = top[1:-1, :, ::2].transpose(0, 1)
    start = 0
    for pref, length in zip(prefixes, lengths):
        common = torch.randn(heads, max_blocks)
        for i in range(length):
            valid = (pref + i + 128) // 128
            for h in range(heads):
                score = torch.randn(valid)
                if pattern == "overlap":
                    score = common[h, :valid] + score * 0.05
                # Same forced-block precedence as index_topk: local overwrites
                # init scores, selection is unique and ordered by score, not ID.
                score[:init] = 1e30
                score[max(0, valid - local) :] = 1e29
                ids = score.topk(min(topk, valid)).indices.int()
                assert len(ids.unique()) == len(ids)
                top[h, start + i, : len(ids)] = ids
        start += length
    return dict(
        q=q,
        kv_cache=cache,
        topk_idx=top,
        block_table=bt,
        cu_seqlens_q=cu,
        seq_lens=seq,
        prefix_lens=prefix,
        max_query_len=max(lengths),
        num_kv_heads=heads,
        sm_scale=dim**-0.5,
        output=out,
        k_scale=ks,
        v_scale=vs,
    )


def _reference(args):
    q, cache, top = (args[k] for k in ("q", "kv_cache", "topk_idx"))
    cu, seq, prefix = (args[k] for k in ("cu_seqlens_q", "seq_lens", "prefix_lens"))
    heads, dim = args["num_kv_heads"], q.shape[-1]
    group = q.shape[1] // heads
    # Match the kernel's dequantization rounding, then compute true fp32
    # softmax over the exact selected key multiset (no union in the reference).
    cache = cache.to(q.dtype)
    k, v = cache[..., :dim], cache[..., dim:]
    for name, val in (("k_scale", k), ("v_scale", v)):
        scale = args[name]
        if scale is not None:
            if scale.numel() > 1:
                scale = scale.reshape(heads, cache.shape[0], 128).permute(1, 0, 2)
                scale = scale[..., None]
            val.copy_((val.float() * scale).to(q.dtype))
    result = torch.empty_like(q, dtype=torch.float32)
    for b in range(len(seq)):
        for t in range(int(cu[b]), int(cu[b + 1])):
            absolute = int(prefix[b]) + t - int(cu[b])
            count = min(top.shape[-1], (absolute + 128) // 128)
            for h in range(heads):
                ids = top[h, t, :count].long()
                ids = ids[ids >= 0]
                pos = (ids[:, None] * 128 + torch.arange(128)).flatten()
                pages = args["block_table"][b, ids].long()
                mask = (pos <= absolute) & (pos < seq[b])
                keys = k[pages, h].reshape(-1, dim)[mask].float()
                values = v[pages, h].reshape(-1, dim)[mask].float()
                rows = slice(h * group, (h + 1) * group)
                scores = q[t, rows].float() @ keys.T * args["sm_scale"]
                result[t, rows] = scores.softmax(-1) @ values
    return result


def _compare(args, tile, monkeypatch):
    # Reference first, with a private cache because dequantization is in-place.
    reference = _reference({**args, "kv_cache": args["kv_cache"].clone()})
    monkeypatch.setattr(attn, "_PREFILL_TILE_Q", tile)
    attn.minimax_m3_sparse_attn(**args)
    output = args["output"].float()
    tol = 1e-4 if args["q"].dtype == torch.float32 else 2e-2
    error = (output - reference).abs().max().item()
    print(f"dtype={args['q'].dtype} tile={tile} max_abs(tiled/reference)={error}")
    torch.testing.assert_close(output, reference, rtol=0, atol=tol)


@pytest.mark.parametrize("tile", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_scalar_fallback(tile, dtype, monkeypatch):
    _compare(_case(2, 8, dtype, "overlap"), tile, monkeypatch)


@pytest.mark.parametrize("tile", [16, 32])
@pytest.mark.parametrize("group", [4, 8, 16])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("pattern", ["overlap", "varied"])
def test_selected_keys_match_original_and_reference(
    tile, group, dtype, pattern, monkeypatch
):
    _compare(_case(tile, group, dtype, pattern), tile, monkeypatch)


@pytest.mark.parametrize("tile", [16, 32])
def test_block_ids_at_64k_and_one_million_tokens(tile, monkeypatch):
    _compare(_case(tile, 8, torch.float32, "long"), tile, monkeypatch)


@pytest.mark.parametrize("tile", [16, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_disjoint_lists_and_large_logits(tile, dtype, monkeypatch):
    """Rows must survive entirely masked union blocks before their own block."""
    args = _case(tile, 4, dtype, "long")
    args["topk_idx"] = args["topk_idx"][..., :1]
    for h in range(args["num_kv_heads"]):
        args["topk_idx"][h, :, 0] = torch.arange(len(args["q"])) * 3 + h
    args["q"].mul_(8)
    _compare(args, tile, monkeypatch)


@pytest.mark.parametrize("tile", [16, 32])
def test_tiled_skips_sentinels_inside_valid_count(tile, monkeypatch):
    """Defensive sentinel handling; the old kernel cannot consume these holes.

    Normal index_topk padding is already tested against both kernels above.
    """
    args = _case(tile, 4, torch.float32, "long")
    args["topk_idx"][:, :, 1::2] = -1
    args["topk_idx"][:, 0, :] = -1
    reference = _reference({**args, "kv_cache": args["kv_cache"].clone()})
    monkeypatch.setattr(attn, "_PREFILL_TILE_Q", tile)
    attn.minimax_m3_sparse_attn(**args)
    torch.testing.assert_close(args["output"], reference, rtol=0, atol=1e-4)


@pytest.mark.parametrize("tile", [16, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("scale_mode", [0, 1, 2])
def test_fp8_cache_scale_modes(tile, dtype, kv_dtype, scale_mode, monkeypatch):
    _compare(_case(tile, 4, dtype, "overlap", kv_dtype, scale_mode), tile, monkeypatch)
