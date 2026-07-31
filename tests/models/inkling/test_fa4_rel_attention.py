# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness test for the Inkling FA4 relative-attention score-mod kernel.

Checks ``inkling_fa4_rel_attention`` against a pure-PyTorch reference that
implements the relative bias exactly as documented in the Inkling architecture
guide::

    logit(i, j, h) = (1 / head_dim) * dot(q[i, h], k[j, h]) + rel_bias(i, j, h)
    rel_bias(i, j, h) = rel_logits[i, h, i - j]   if 0 <= i - j < rel_extent
                      = 0                          otherwise

with causal (and optionally sliding-window) masking handled by the backend.
"""

import importlib

import pytest
import torch

from vllm.models.inkling.nvidia.attention import (
    InklingAttention,
    compute_log_scaling_tau,
)
from vllm.models.inkling.nvidia.ops.fa4_rel_attention import (
    _use_sheared_bias,
    inkling_fa4_num_splits,
    inkling_fa4_rel_attention,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability

_cap = current_platform.get_device_capability() if current_platform.is_cuda() else None

NUM_HEADS = [(4, 4), (8, 2)]  # (num_heads, num_kv_heads)
GLOBAL_REL_EXTENTS = [128, 1024]
LOCAL_REL_EXTENTS = [128, 256]
HEAD_DIM = 128
BLOCK_SIZE = 16
DTYPE = torch.bfloat16


def test_log_scaling_tau_matches_reference():
    positions = torch.tensor([0, 127999, 128000, 999999], dtype=torch.int64)
    actual = compute_log_scaling_tau(positions, 128000, 0.1)
    expected = 1.0 + 0.1 * torch.log(
        torch.clamp((positions + 1).float() / 128000.0, min=1.0)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_split_packed_kv_cache():
    attention = InklingAttention.__new__(InklingAttention)
    torch.nn.Module.__init__(attention)
    attention.head_dim = 8
    attention.kv_cache = torch.arange(2 * 3 * 4 * 16).reshape(2, 3, 4, 16)

    key_cache, value_cache = attention._split_kv_cache()

    assert key_cache.shape == value_cache.shape == (2, 4, 3, 8)
    torch.testing.assert_close(key_cache, attention.kv_cache[..., :8].transpose(1, 2))
    torch.testing.assert_close(value_cache, attention.kv_cache[..., 8:].transpose(1, 2))


def test_num_splits_hopper_is_unsplit(monkeypatch):
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda: DeviceCapability(major=9, minor=0),
    )
    assert (
        inkling_fa4_num_splits(
            is_local=False,
            batch_size=1,
            max_query_len=1,
            num_heads=16,
            num_kv_heads=2,
            max_kv_len=1_048_576,
        )
        == 1
    )


@pytest.mark.parametrize(
    ("major", "expected"),
    [(9, False), (10, True), (11, True), (12, False)],
)
def test_sheared_bias_architecture_selection(monkeypatch, major, expected):
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda: DeviceCapability(major=major, minor=0),
    )
    _use_sheared_bias.cache_clear()
    try:
        assert _use_sheared_bias() is expected
    finally:
        _use_sheared_bias.cache_clear()


@pytest.fixture
def blackwell_platform(monkeypatch):
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda: DeviceCapability(major=10, minor=0),
    )


@pytest.mark.parametrize(
    ("batch_size", "max_query_len", "expected"),
    [
        (1, 1, (16, 32, 128, 128)),
        (8, 1, (2, 4, 8, 16)),
        (32, 1, (1, 1, 2, 4)),
        (1, 128, (2, 4, 8, 16)),
        (1, 2048, (1, 1, 1, 1)),
    ],
)
def test_num_splits_all_tp(blackwell_platform, batch_size, max_query_len, expected):
    actual = tuple(
        inkling_fa4_num_splits(
            is_local=False,
            batch_size=batch_size,
            max_query_len=max_query_len,
            num_heads=64 // tp,
            num_kv_heads=8 // tp,
            max_kv_len=131072,
        )
        for tp in (1, 2, 4, 8)
    )
    assert actual == expected


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_num_splits_local_is_unsplit(tp):
    assert (
        inkling_fa4_num_splits(
            is_local=True,
            batch_size=1,
            max_query_len=1,
            num_heads=64 // tp,
            num_kv_heads=16 // tp,
            max_kv_len=512,
        )
        == 1
    )


@pytest.mark.parametrize(
    ("max_kv_len", "expected"),
    [(8192, 32), (65536, 64), (1048576, 128)],
)
@pytest.mark.parametrize("tp", [4, 8])
def test_num_splits_long_context_bound(blackwell_platform, tp, max_kv_len, expected):
    assert (
        inkling_fa4_num_splits(
            is_local=False,
            batch_size=1,
            max_query_len=1,
            num_heads=64 // tp,
            num_kv_heads=8 // tp,
            max_kv_len=max_kv_len,
        )
        == expected
    )


def _ref_rel_attn(
    q: torch.Tensor,  # [total_q, H, D]
    key_cache: torch.Tensor,  # [num_blocks, block, Hkv, D]
    value_cache: torch.Tensor,
    rel_logits: torch.Tensor,  # [total_q, H, rel_extent]
    *,
    q_lens: list[int],
    kv_lens: list[int],
    block_table: torch.Tensor,
    scale: float,
    rel_extent: int,
    window_left: int | None,
) -> torch.Tensor:
    num_kv_heads = key_cache.shape[2]
    num_heads = q.shape[1]
    g = num_heads // num_kv_heads
    bt = block_table.cpu().numpy()
    out = torch.empty_like(q)

    start = 0
    for i, (ql, kl) in enumerate(zip(q_lens, kv_lens)):
        qi = q[start : start + ql].float()  # [ql, H, D]
        rl = rel_logits[start : start + ql].float()  # [ql, H, rel_extent]

        nblk = (kl + BLOCK_SIZE - 1) // BLOCK_SIZE
        blk = bt[i, :nblk]
        k = key_cache[blk].reshape(-1, num_kv_heads, HEAD_DIM)[:kl].float()
        v = value_cache[blk].reshape(-1, num_kv_heads, HEAD_DIM)[:kl].float()
        k = k.repeat_interleave(g, dim=1)  # [kl, H, D]
        v = v.repeat_interleave(g, dim=1)

        # [H, ql, kl]
        scores = torch.einsum("qhd,khd->hqk", qi, k) * scale

        dev = q.device
        qpos = torch.arange(ql, device=dev).view(ql, 1) + (kl - ql)  # query pos
        kpos = torch.arange(kl, device=dev).view(1, kl)
        dist = qpos - kpos  # [ql, kl] = i - j

        # Relative bias: rel_logits[i, h, dist] when 0 <= dist < rel_extent.
        in_rng = (dist >= 0) & (dist < rel_extent)  # [ql, kl]
        idx = dist.clamp(0, rel_extent - 1)
        # gather per head: bias[h, i, j] = rl[i, h, idx[i, j]]
        bias = rl.permute(1, 0, 2).gather(  # [H, ql, rel_extent]
            2, idx.unsqueeze(0).expand(num_heads, -1, -1)
        )  # [H, ql, kl]
        bias = torch.where(in_rng.unsqueeze(0), bias, torch.zeros_like(bias))
        scores = scores + bias

        mask = dist < 0  # causal
        if window_left is not None:
            mask = mask | (dist > window_left)
        scores.masked_fill_(mask.unsqueeze(0), float("-inf"))

        probs = torch.softmax(scores, dim=-1)
        out[start : start + ql] = torch.einsum("hqk,khd->qhd", probs, v).to(q.dtype)
        start += ql
    return out


def _run_case(seq_lens, num_heads, num_kv_heads, rel_extent, window_left, seed=0):
    torch.manual_seed(seed)
    device = "cuda"
    q_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    total_q = sum(q_lens)
    num_seqs = len(seq_lens)

    scale = 1.0 / HEAD_DIM

    # q/k are RMS-normed in the model (unit-ish norm); normalize here so the
    # logit magnitudes are realistic and the bias is not numerically dwarfed.
    q = torch.randn(total_q, num_heads, HEAD_DIM, device=device, dtype=DTYPE)
    q = torch.nn.functional.normalize(q.float(), dim=-1).to(DTYPE)

    # Paged KV cache.
    max_blocks = (max(kv_lens) + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = num_seqs * max_blocks + 1
    key_cache = torch.randn(
        num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM, device=device, dtype=DTYPE
    )
    key_cache = torch.nn.functional.normalize(key_cache.float(), dim=-1).to(DTYPE)
    value_cache = torch.randn(
        num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM, device=device, dtype=DTYPE
    )

    # Distinct blocks per sequence (block 0 left as a never-referenced pad).
    block_table = torch.zeros(num_seqs, max_blocks, dtype=torch.int32, device=device)
    for i in range(num_seqs):
        block_table[i] = torch.arange(
            1 + i * max_blocks, 1 + (i + 1) * max_blocks, dtype=torch.int32
        )

    cu_seqlens_q = torch.tensor(
        [0, *torch.cumsum(torch.tensor(q_lens), 0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    cache_seqlens = torch.tensor(kv_lens, dtype=torch.int32, device=device)

    rel_logits = torch.randn(total_q, num_heads, rel_extent, device=device, dtype=DTYPE)

    window_size = (-1, -1) if window_left is None else (window_left, 0)

    preallocated_out = torch.empty_like(q)
    num_splits = inkling_fa4_num_splits(
        is_local=window_left is not None,
        batch_size=num_seqs,
        max_query_len=max(q_lens),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_kv_len=max(kv_lens),
    )
    out = inkling_fa4_rel_attention(
        q,
        key_cache,
        value_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max(q_lens),
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        rel_extent=rel_extent,
        rel_logits=rel_logits,
        num_splits=num_splits,
        out=preallocated_out,
    )
    assert out.data_ptr() == preallocated_out.data_ptr()
    out = out.view(total_q, num_heads, HEAD_DIM)

    ref = _ref_rel_attn(
        q,
        key_cache,
        value_cache,
        rel_logits,
        q_lens=q_lens,
        kv_lens=kv_lens,
        block_table=block_table,
        scale=scale,
        rel_extent=rel_extent,
        window_left=window_left,
    )

    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@torch.inference_mode()
def test_score_mod_relative_attention(monkeypatch):
    module = importlib.import_module("vllm.models.inkling.nvidia.ops.fa4_rel_attention")
    monkeypatch.setattr(module, "_use_sheared_bias", lambda: False)
    _run_case(
        [(64, 64), (1, 80)],
        num_heads=4,
        num_kv_heads=4,
        rel_extent=128,
        window_left=None,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize(
    "seq_lens",
    [
        [(64, 64)],  # single full prefill
        [(64, 64), (33, 33), (17, 17)],  # ragged prefill batch
        [(512, 512)],  # seq_len >> rel_extent (most keys get zero bias)
        [(300, 300), (512, 512), (129, 129)],  # large ragged batch
    ],
)
@pytest.mark.parametrize("rel_extent", GLOBAL_REL_EXTENTS)
@torch.inference_mode()
def test_full_attention(seq_lens, num_heads, rel_extent):
    # rel_extent=128 exercises the out-of-range (zero bias) path; 1024 covers all.
    # With the 512-token cases and rel_extent=128, query/seq lengths are far
    # larger than rel_extent so the vast majority of (i, j) pairs are out of
    # range and must contribute zero bias.
    _run_case(seq_lens, num_heads[0], num_heads[1], rel_extent, window_left=None)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize(
    "seq_lens",
    [
        [(200, 512)],  # chunked prefill: q_len=200 (> rel_extent), 312 cached
        [(200, 512), (50, 300), (1, 400)],  # mixed chunked + decode
    ],
)
@pytest.mark.parametrize("rel_extent", GLOBAL_REL_EXTENTS)
@torch.inference_mode()
def test_chunked_prefill(seq_lens, num_heads, rel_extent):
    # q_len < kv_len with q_len itself larger than rel_extent (for the 128 case):
    # exercises the seqlen_k - seqlen_q offset together with the out-of-range path.
    _run_case(seq_lens, num_heads[0], num_heads[1], rel_extent, window_left=None)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize(
    "seq_lens",
    [
        [(64, 64), (40, 40)],  # seq_len > window
        [(512, 512), (300, 300)],  # seq_len/query_len >> window
        [(1, 512)],  # decode with kv_len >> window
    ],
)
@pytest.mark.parametrize("local_extent", LOCAL_REL_EXTENTS)
@torch.inference_mode()
def test_sliding_window(seq_lens, num_heads, local_extent):
    # Local layers use window_size=(local_extent-1, 0) and rel_extent==local_extent.
    # With the 512-token cases, query/seq lengths far exceed the window so most
    # keys are masked out by the sliding window.
    _run_case(
        seq_lens,
        num_heads[0],
        num_heads[1],
        rel_extent=local_extent,
        window_left=local_extent - 1,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.skipif(
    _cap is None or _cap.major < 9,
    reason="FA4 score-mod requires Hopper+ (SM90+)",
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 50)],
        [(1, 50), (1, 7), (1, 200)],
        [(1, 512), (1, 333)],  # kv_len >> rel_extent
    ],
)
@pytest.mark.parametrize("rel_extent", GLOBAL_REL_EXTENTS)
@torch.inference_mode()
def test_decode(seq_lens, num_heads, rel_extent):
    # q_len=1 with kv_len>q_len: the score-mod's seqlen_k - seqlen_q offset path.
    _run_case(seq_lens, num_heads[0], num_heads[1], rel_extent, window_left=None)
