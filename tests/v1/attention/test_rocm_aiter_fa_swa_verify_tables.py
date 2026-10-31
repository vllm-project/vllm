# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends import rocm_aiter_fa as fa
from vllm.v1.kv_cache_interface import SlidingWindowSpec

W, BS, H, KVH, D = 2048, 16, 4, 2, 64
MIXED = ([100, 2047, 2055, 2056, 4096, 32768, 200000, 9], [1, 8, 3, 8, 5, 2, 8, 7])
BUILD_CASES = [(0, 1, 0), (1, 0, 0), (1, 1, 1), (1, 1, 0)]
Impl, Builder = fa.AiterFlashAttentionImpl, fa.AiterFlashAttentionMetadataBuilder


def _slots(row, L, q, w, bs):
    ks = [torch.arange(max(0, L - q + i - w + 1), L - q + i + 1) for i in range(q)]
    return [row[k // bs].long() * bs + k % bs for k in ks]


def _assert_same_keys(full, lens, table, tlens, qlens, w, bs):
    for r, (L, q) in enumerate(zip(lens, qlens)):
        Lt = int(tlens[r])
        assert q <= Lt <= L and (L - Lt) % bs == 0
        for a, b in zip(_slots(full[r], L, q, w, bs), _slots(table[r], Lt, q, w, bs)):
            assert torch.equal(a, b), (L, q)


def _trim(lens, qlens, w, bs, max_q=8, rows=None):
    n, width = len(lens), cdiv(w + max_q - 1, bs) + 1
    b = SimpleNamespace(
        aot_sliding_window=(w - 1, 0), block_size=bs,
        _swa_block_table=torch.zeros(rows or n, width, dtype=torch.int32),
        _swa_seq_lens=torch.zeros(rows or n, dtype=torch.int32),
        _swa_block_offsets=torch.arange(width),
    )  # fmt: skip
    full = torch.randperm(n * (cdiv(max(lens), bs) + 3)).view(n, -1).int()
    qsl = torch.tensor([0] + qlens).cumsum(0).int()
    lens_t = torch.tensor(lens, dtype=torch.int32)
    out = Builder._build_swa_decode_tables(
        b, full, lens_t, n, max(lens), qsl, max(qlens)
    )
    return full, lens_t, out


@pytest.mark.parametrize("w, bs", [(100, 16), (7, 4)])
def test_trimmed_table_sees_same_keys(w, bs):
    for q in [*range(1, 9), None]:
        lens = list(range(q or 1, 3 * w + 3 * bs)) + [200000]
        qlens = [q or min(L, 1 + L % 8) for L in lens]
        full, _, (table, tlens, max_len) = _trim(lens, qlens, w, bs)
        assert max(tlens) <= max_len <= max(lens)
        _assert_same_keys(full, lens, table, tlens, qlens, w, bs)


def test_q1_matches_old_formula():
    lens = list(range(1, 3 * W, 37)) + [200000]
    full, lens_t, (table, tlens, max_len) = _trim(lens, [1] * len(lens), W, BS, 1)
    first = (lens_t - W).clamp(min=0) // BS
    idx = (first[:, None] + torch.arange(cdiv(W, BS) + 1)).clamp(max=full.shape[1] - 1)
    assert torch.equal(table, full.gather(1, idx))
    assert torch.equal(tlens, lens_t - first * BS)
    assert max_len == W + BS - 1


def test_capacity_fallback():
    assert _trim([5000] * 4, [2] * 4, W, BS, max_q=1)[2] == (None, None, 0)
    assert _trim([5000] * 4, [2] * 4, W, BS, rows=3)[2] == (None, None, 0)


@pytest.fixture
def cfg():
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with set_current_vllm_config(config):
        yield config


def _builder(monkeypatch, cfg, flag, alibi=False, k=7):
    monkeypatch.setenv("VLLM_ROCM_AITER_FA_SWA_VERIFY_TRIM", str(int(flag)))
    layer = Attention.__new__(Attention)
    torch.nn.Module.__init__(layer)
    layer.impl = Impl(H, D, 1.0, KVH, [0.5] * H if alibi else None, W, "auto")
    layer.kv_sharing_target_layer_name = None
    layer.kv_cache_update_bypass = False
    monkeypatch.setattr(cfg.compilation_config, "static_forward_context", {"a": layer})
    model = SimpleNamespace(
        dtype=torch.bfloat16, get_head_size=lambda: D,
        get_num_attention_heads=lambda _: H, get_num_kv_heads=lambda _: KVH,
    )  # fmt: skip
    monkeypatch.setattr(cfg, "model_config", model)
    spec_cfg = SimpleNamespace(num_speculative_tokens=k, parallel_drafting=False)
    monkeypatch.setattr(cfg, "speculative_config", spec_cfg if k else None)
    spec = SlidingWindowSpec(
        block_size=BS, num_kv_heads=KVH, head_size=D, dtype=torch.bfloat16,
        sliding_window=W,
    )  # fmt: skip
    return Builder(spec, ["a"], cfg, torch.device("cpu"))


def _common(lens, qlens):
    spec = BatchSpec(seq_lens=lens, query_lens=qlens)
    cpu = torch.device("cpu")
    return create_common_attn_metadata(spec, BS, cpu, arange_block_indices=True)


@pytest.mark.parametrize("flag, causal, alibi", BUILD_CASES)
def test_build_verify_trim(monkeypatch, cfg, flag, causal, alibi):
    b = _builder(monkeypatch, cfg, flag, alibi)
    wide = bool(flag and not alibi)
    on = wide and bool(causal)
    assert b._swa_block_table.shape[1] == cdiv(W + 7 * wide, BS) + 1
    b._build_swa_decode_tables = Mock(wraps=b._build_swa_decode_tables)
    common = _common(*MIXED)
    common.causal = bool(causal)
    dmd = b.build(0, common).decode_metadata
    assert b._build_swa_decode_tables.call_count == on
    assert (dmd.swa_block_table is not None) is on
    if on:
        assert dmd.swa_max_seq_len == W + 7 + BS - 1
        args = (dmd.swa_block_table, dmd.swa_seq_lens, MIXED[1], W, BS)
        _assert_same_keys(common.block_table_tensor, MIXED[0], *args)
    dmd = b.build(0, _common(MIXED[0], [1] * 8)).decode_metadata
    assert dmd.swa_block_table is not None
    if on:
        no_spec = _builder(monkeypatch, cfg, True, k=0)
        assert no_spec._swa_block_table.shape[1] == cdiv(W, BS) + 1


@pytest.mark.parametrize("case", ["swa_verify", "full", "alibi", "q1_sinks"])
def test_forward_routing(monkeypatch, cfg, case):
    lens, qlens = ([100, 4096, 200000], [1] * 3) if case == "q1_sinks" else MIXED
    md = _builder(monkeypatch, cfg, True).build(0, _common(lens, qlens))
    alibi = [0.5] * H if case == "alibi" else None
    sinks = torch.zeros(H) if case == "q1_sinks" else None
    window = None if case == "full" else W
    impl = Impl(H, D, 1.0, KVH, alibi, window, "auto", sinks=sinks)
    calls: list[dict] = []
    fake = types.ModuleType("aiter.ops.triton.unified_attention")
    ua = lambda **kw: calls.append(kw)
    monkeypatch.setattr(fake, "unified_attention", ua, raising=False)
    monkeypatch.setitem(sys.modules, "aiter.ops.triton.unified_attention", fake)
    kv = torch.zeros(int(md.block_table.max()) + 1, KVH, BS, 2 * D)
    q = torch.zeros(sum(qlens), H, D)
    layer = SimpleNamespace(_k_scale=torch.ones(1), _v_scale=torch.ones(1))
    impl.forward(layer, q, None, None, kv, md, torch.zeros_like(q))
    [call] = calls
    dmd, n = md.decode_metadata, len(lens)
    if case == "swa_verify":
        want = (dmd.swa_block_table, dmd.swa_seq_lens, dmd.swa_max_seq_len)
    else:
        want = (md.block_table[:n], md.seq_lens[:n], md.max_seq_len)
    assert torch.equal(call["block_table"], want[0])
    assert torch.equal(call["seqused_k"], want[1])
    assert call["max_seqlen_k"] == want[2]
