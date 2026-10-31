# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionMetadataBuilder

if not current_platform.is_rocm():
    pytest.skip("ROCm AITER only", allow_module_level=True)

UA = pytest.importorskip("aiter.ops.triton.attention.unified_attention")
UAU = importlib.import_module("aiter.ops.triton.utils.unified_attention_utils")

W, BS, HQ, HKV, D = 2048, 16, 16, 2, 64
LENS = [100, 2047, 2048, 2055, 4096, 32768, 200000]


def _force_split(monkeypatch):
    grid, cfg = UA._gfx950_split_grid, UA.get_unified_attention_config
    seg, run3d = UAU.compute_segment_params, UA._unified_attention_3d_gfx950
    lift, segments = [False], []

    def get_config(op, params, *a, **kw):
        lift[0] = op == "kv_split" and params.sliding_window > 0
        out = cfg(op, params, *a, **kw)
        lift[0] = False
        return out

    def segment_params(config, params):
        if lift[0]:
            config.pop("MAX_SEGMENTS", None)
        return seg(config, params)

    def record_3d(params, o, m, e, num_segments, *a):
        segments.append(num_segments)
        return run3d(params, o, m, e, num_segments, *a)

    monkeypatch.setattr(
        UA, "_gfx950_split_grid", lambda p: p.sliding_window > 0 or grid(p)
    )
    monkeypatch.setattr(UA, "get_unified_attention_config", get_config)
    monkeypatch.setattr(UAU, "compute_segment_params", segment_params)
    monkeypatch.setattr(UA, "_unified_attention_3d_gfx950", record_3d)
    return segments


@pytest.mark.parametrize("qlens", [[8] * 7, [8, 1, 3, 8, 2, 8, 5]])
@pytest.mark.parametrize("split", [False, True])
def test_trimmed_matches_full(monkeypatch, split, qlens):
    if split and getattr(UA, "DEVICE_ARCH", None) != "gfx950":
        pytest.skip("forced split targets the gfx950 gluon path")
    segments = _force_split(monkeypatch) if split else None
    g = torch.Generator(device="cuda").manual_seed(0)
    n, nblk = len(LENS), sum(cdiv(L, BS) for L in LENS)
    k, v = torch.randn(2, nblk, BS, HKV, D, device="cuda", generator=g).bfloat16()
    bt = torch.randperm(nblk, device="cuda", generator=g).int().repeat(n, 1)
    q = torch.randn(sum(qlens), HQ, D, device="cuda", generator=g).bfloat16()
    cu = torch.tensor([0] + qlens, device="cuda").cumsum(0).int()
    sk = torch.tensor(LENS, dtype=torch.int32, device="cuda")
    width = cdiv(W + 7, BS) + 1
    builder = SimpleNamespace(
        aot_sliding_window=(W - 1, 0), block_size=BS,
        _swa_block_table=torch.zeros(n, width, dtype=torch.int32, device="cuda"),
        _swa_seq_lens=torch.zeros(n, dtype=torch.int32, device="cuda"),
        _swa_block_offsets=torch.arange(width, device="cuda"),
    )  # fmt: skip
    table, lens, max_len = AiterFlashAttentionMetadataBuilder._build_swa_decode_tables(
        builder, bt, sk, n, max(LENS), cu, max(qlens)
    )
    assert max_len < max(LENS)
    outs = []
    for seqused_k, max_k, block_table in ((sk, max(LENS), bt), (lens, max_len, table)):
        out = torch.empty_like(q)
        UA.unified_attention(
            q=q, k=k, v=v, out=out, cu_seqlens_q=cu, max_seqlen_q=max(qlens),
            seqused_k=seqused_k, max_seqlen_k=max_k, softmax_scale=D**-0.5,
            causal=True, window_size=(W - 1, 0), block_table=block_table,
            softcap=0, q_descale=None, k_descale=None, v_descale=None,
        )  # fmt: skip
        outs.append(out.float())
    full, trimmed = outs
    if split:
        assert len(segments) == 2 and min(segments) > 1, segments
    assert (full - trimmed).abs().max() <= (1e-3 if split else 2**-7)
    for r, (L, ql) in enumerate(zip(LENS, qlens)):
        rows = slice(int(cu[r]), int(cu[r + 1]))
        pos = torch.arange(max(0, L - ql - W + 1), L, device="cuda")
        slots = bt[r, pos // BS].long() * BS + pos % BS
        kk = k.view(-1, HKV, D)[slots].float().repeat_interleave(HQ // HKV, 1)
        vv = v.view(-1, HKV, D)[slots].float().repeat_interleave(HQ // HKV, 1)
        s = torch.einsum("qhd,khd->hqk", q[rows].float(), kk) * D**-0.5
        qpos = (L - ql + torch.arange(ql, device="cuda"))[:, None]
        s.masked_fill_(~((pos <= qpos) & (qpos - pos < W))[None], float("-inf"))
        ref = torch.einsum("hqk,khd->qhd", s.softmax(-1), vv)
        assert max((o[rows] - ref).abs().max() for o in (full, trimmed)) <= 1e-2
