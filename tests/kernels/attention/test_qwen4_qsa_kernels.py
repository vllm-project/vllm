# Copyright 2026, The FlagOS Contributors.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness coverage for the self-developed Qwen4 QSA kernels.

Torch references live in this test module only. The production wrappers are
required to fail closed instead of dispatching a Torch compute fallback.
"""

from __future__ import annotations

import math

import pytest
import torch

from vllm.platforms import current_platform

pytest.importorskip("triton")

from vllm.model_executor.layers.qsa import (  # noqa: E402
    qwen4_compress_norm_mrope_store_groups,
    qwen4_qsa_mqa_paged_dot,
    qwen4_store_qsa_kv_rows,
)

DEVICE = current_platform.device_type
EPS = 1.0e-6

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Qwen4 QSA Triton kernels require CUDA/ROCm.",
)


def _qsa_mqa_reference(q, cache, table, token_to_req, positions, lengths, ratio):
    columns = table.shape[1] * cache.shape[1]
    req = token_to_req.to(torch.int64)
    valid_req = (req >= 0) & (req < table.shape[0])
    safe_req = req.clamp(0, table.shape[0] - 1)
    seq = lengths.index_select(0, safe_req)
    seq = torch.where(valid_req, seq, torch.zeros_like(seq))
    visible = torch.minimum((positions.to(torch.int64) + 1) // ratio, seq // ratio)
    col = torch.arange(columns, device=q.device, dtype=torch.int64)
    logical_page = col // cache.shape[1]
    offset = col % cache.shape[1]
    safe_logical = logical_page.clamp(0, table.shape[1] - 1)
    physical = table[safe_req[:, None], safe_logical[None, :]].to(torch.int64)
    valid = (
        valid_req[:, None]
        & (col[None, :] < visible[:, None])
        & (physical >= 0)
        & (physical < cache.shape[0])
    )
    safe_physical = physical.clamp(0, cache.shape[0] - 1)
    keys = cache[safe_physical, offset[None, :], 0].float()
    dots = torch.matmul(q.float(), keys.transpose(1, 2))
    logits = torch.relu(dots).sum(1) / math.sqrt(q.shape[-1])
    logits = logits.masked_fill(~valid, -float("inf"))
    return logits, visible.to(torch.int32)


def test_qwen4_qsa_mqa_paged_dot_matches_torch_and_invalid_pages():
    rows, page_size, pages, columns = 8, 16, 4, 48
    torch.manual_seed(202)
    q = torch.randn((rows, 4, 128), device=DEVICE, dtype=torch.bfloat16)
    cache = torch.randn((pages, page_size, 1, 128), device=DEVICE, dtype=torch.bfloat16)
    table = torch.tensor(
        [[0, 1, 2], [1, 2, 3], [2, 3, 0], [3, 0, 1]],
        device=DEVICE,
        dtype=torch.int32,
    )
    table[2, 1] = -1
    token_to_req = torch.tensor(
        [0, 1, 2, 3, 0, 1, -1, 3], device=DEVICE, dtype=torch.int32
    )
    positions = torch.tensor(
        [47, 39, 31, 23, 47, 35, 47, 15], device=DEVICE, dtype=torch.int64
    )
    lengths = torch.tensor([48, 40, 32, 24], device=DEVICE, dtype=torch.int64)

    ref_logits, ref_visible = _qsa_mqa_reference(
        q, cache, table, token_to_req, positions, lengths, 4
    )
    out_logits, out_visible = qwen4_qsa_mqa_paged_dot(
        q,
        cache,
        table,
        token_to_req,
        positions,
        lengths,
        compress_ratio=4,
        num_columns=columns,
    )
    torch.testing.assert_close(out_visible, ref_visible, atol=0, rtol=0)
    torch.testing.assert_close(out_logits, ref_logits, atol=0.2, rtol=0.02)
    assert torch.isneginf(out_logits[6]).all()
    assert torch.isneginf(out_logits[2, 8:]).all()

    snapshot = out_logits.clone()
    for _ in range(10):
        again, visible = qwen4_qsa_mqa_paged_dot(
            q,
            cache,
            table,
            token_to_req,
            positions,
            lengths,
            compress_ratio=4,
            num_columns=columns,
        )
        torch.testing.assert_close(again, snapshot, atol=0, rtol=0)
        torch.testing.assert_close(visible, out_visible, atol=0, rtol=0)


def test_qwen4_qsa_kv_store_matches_torch_and_skips_invalid_slots():
    blocks, page_size, heads, dim, rows = 3, 4, 2, 17, 5
    torch.manual_seed(303)
    initial_k = torch.randn(
        (blocks, page_size, heads, dim), device=DEVICE, dtype=torch.bfloat16
    )
    initial_v = torch.randn_like(initial_k)
    key = torch.randn((rows, heads, dim), device=DEVICE, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    slots = torch.tensor([0, 3, 4, 11, -1], device=DEVICE, dtype=torch.int64)

    ref_k, ref_v = initial_k.clone(), initial_v.clone()
    valid = (slots >= 0) & (slots < blocks * page_size)
    safe = slots[valid]
    ref_k.index_put_(
        (safe // page_size, safe % page_size), key[valid], accumulate=False
    )
    ref_v.index_put_(
        (safe // page_size, safe % page_size), value[valid], accumulate=False
    )

    out_k, out_v = initial_k.clone(), initial_v.clone()
    qwen4_store_qsa_kv_rows(out_k, out_v, slots, key, value)
    torch.testing.assert_close(out_k, ref_k, atol=0, rtol=0)
    torch.testing.assert_close(out_v, ref_v, atol=0, rtol=0)


def _compress_reference(
    raw, rope, table, requests, positions, slots, weight, cos_sin, out
):
    ratio = 4
    for row in range(requests.numel()):
        request = int(requests[row].item())
        end = int(positions[row].item())
        slot = int(slots[row].item())
        raw_rows = []
        for offset in range(ratio):
            pos = end - (ratio - 1 - offset)
            page = int(table[request, pos // raw.shape[1]].item())
            raw_rows.append(raw[page, pos % raw.shape[1], 0].float())
        pooled = (torch.stack(raw_rows).sum(0) / ratio).to(torch.bfloat16)
        pooled_fp32 = pooled.float()
        variance = pooled_fp32.square().mean() / 1.0
        normalized = (
            pooled_fp32 * torch.rsqrt(variance + EPS) * (weight.float() + 1.0)
        ).to(torch.bfloat16)
        first_pos = end - ratio + 1
        page = int(table[request, first_pos // raw.shape[1]].item())
        axes = rope[page, first_pos % rope.shape[1], 0, :3]
        freq = torch.arange(32, device=raw.device)
        use_h = (freq % 3 == 1) & (freq < 3 * 11)
        use_w = (freq % 3 == 2) & (freq < 3 * 10)
        axis_pos = torch.where(
            use_h,
            axes[1],
            torch.where(use_w, axes[2], axes[0]),
        )
        cos = cos_sin[axis_pos, freq].float()
        sin = cos_sin[axis_pos, 32 + freq].float()
        first = normalized[:32].float()
        second = normalized[32:64].float()
        rotated_first = (first * cos - second * sin).to(torch.bfloat16)
        rotated_second = (second * cos + first * sin).to(torch.bfloat16)
        stored = torch.cat((rotated_first, rotated_second, normalized[64:]), dim=0)
        block, token = slot // out.shape[1], slot % out.shape[1]
        out[block, token, 0].copy_(stored)


def test_qwen4_qsa_fused_compress_norm_mrope_interleaved_matches_torch():
    page_size, raw_blocks = 8, 2
    torch.manual_seed(404)
    raw = torch.randn(
        (raw_blocks, page_size, 1, 128), device=DEVICE, dtype=torch.bfloat16
    )
    rope = torch.empty((raw_blocks, page_size, 1, 3), device=DEVICE, dtype=torch.int64)
    rope[..., 0] = torch.arange(page_size, device=DEVICE).view(1, page_size, 1) % 16
    rope[..., 1] = (
        torch.arange(page_size, device=DEVICE).view(1, page_size, 1) + 1
    ) % 16
    rope[..., 2] = (
        torch.arange(page_size, device=DEVICE).view(1, page_size, 1) + 2
    ) % 16
    table = torch.tensor([[0, 1]], device=DEVICE, dtype=torch.int32)
    requests = torch.tensor([0, 0], device=DEVICE, dtype=torch.int32)
    positions = torch.tensor([3, 7], device=DEVICE, dtype=torch.int64)
    slots = torch.tensor([0, 1], device=DEVICE, dtype=torch.int64)
    weight = torch.randn((128,), device=DEVICE, dtype=torch.bfloat16) * 0.01
    t = torch.arange(16, device=DEVICE, dtype=torch.float32)
    freq = torch.arange(32, device=DEVICE, dtype=torch.float32)
    inv_freq = 1.0 / (10000.0 ** (freq / 32.0))
    angles = t[:, None] * inv_freq[None, :]
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)
    initial = torch.full_like(raw, 7)
    expected = initial.clone()
    _compress_reference(
        raw, rope, table, requests, positions, slots, weight, cos_sin, expected
    )

    actual = initial.clone()
    qwen4_compress_norm_mrope_store_groups(
        raw,
        table,
        requests,
        positions,
        slots,
        actual,
        weight,
        cos_sin,
        compress_ratio=4,
        norm_eps=EPS,
        rotary_dim=64,
        mrope_section=(11, 11, 10),
        mrope_interleaved=True,
        rope_cache=rope,
    )
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.02)

    snapshot = actual.clone()
    for _ in range(3):
        repeated = initial.clone()
        qwen4_compress_norm_mrope_store_groups(
            raw,
            table,
            requests,
            positions,
            slots,
            repeated,
            weight,
            cos_sin,
            compress_ratio=4,
            norm_eps=EPS,
            rotary_dim=64,
            mrope_section=(11, 11, 10),
            mrope_interleaved=True,
            rope_cache=rope,
        )
        torch.testing.assert_close(repeated, snapshot, atol=0, rtol=0)


def test_qsa_exports_are_explicit():
    from vllm.model_executor.layers import qsa as qsa_module

    names = set(qsa_module.__all__)
    assert {
        "_qsa_mqa_paged_dot_kernel",
        "_store_qsa_kv_rows_kernel",
        "_compress_norm_mrope_store_qsa_groups_kernel",
    } <= names


def test_qwen4_qsa_cpu_guards_fail_closed():
    cache = torch.empty((1, 16, 1, 128), dtype=torch.bfloat16)
    q = torch.empty((1, 4, 128), dtype=torch.bfloat16)
    table = torch.zeros((1, 1), dtype=torch.int32)
    metadata = torch.zeros((1,), dtype=torch.int32)
    with pytest.raises(RuntimeError):
        qwen4_qsa_mqa_paged_dot(q, cache, table, metadata, metadata, metadata)
