# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers import dsv4_packed_attn


def test_layer_specific_attention_sink_is_refreshed(monkeypatch):
    dsv4_packed_attn._SCRATCH.clear()
    seen_sinks = []

    def fake_kernel(*args):
        seen_sinks.append(args[-1].clone())

    plan = (
        torch.zeros(1, 1, 64, dtype=torch.int32),
        torch.zeros(1, 8, dtype=torch.int32),
        torch.zeros(1, 8, dtype=torch.int32),
        torch.zeros(1, 8, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
    )
    monkeypatch.setattr(dsv4_packed_attn, "_fn", lambda: fake_kernel)
    monkeypatch.setattr(dsv4_packed_attn, "_build_ranges", lambda *args: plan)
    monkeypatch.setattr(dsv4_packed_attn, "_is_sm100a_device", lambda _device: True)

    q = torch.zeros(8, 64, 512, dtype=torch.bfloat16)
    kv = torch.zeros(64, 512, dtype=torch.bfloat16)
    out = torch.empty_like(q)
    owner = SimpleNamespace()
    common = dict(
        q=q,
        kv=kv,
        out=out,
        sm_scale=1.0,
        query_start_loc=torch.tensor([0, 8], dtype=torch.int32),
        seq_lens=torch.tensor([8], dtype=torch.int32),
        gather_lens=torch.tensor([8], dtype=torch.int32),
        chunk_M=1,
        chunk_N=1,
        window_size=1,
        compress_ratio=128,
        top_k=1,
        n_local_heads=16,
        cache_owner=owner,
        cache_key=(0, 1),
    )

    sink_a = torch.arange(64, dtype=torch.float32)
    sink_b = sink_a + 100
    assert dsv4_packed_attn.try_packed_prefill(attn_sink=sink_a, **common)
    assert dsv4_packed_attn.try_packed_prefill(attn_sink=sink_b, **common)

    assert torch.equal(seen_sinks[0].view(8, 16)[0], sink_a[:16])
    assert torch.equal(seen_sinks[1].view(8, 16)[0], sink_b[:16])
    assert not torch.equal(seen_sinks[0], seen_sinks[1])


def test_validation_rejects_nonfinite_output():
    packed = torch.zeros(1, 16, 2)
    stock = torch.zeros_like(packed)
    packed[0, 0, 0] = torch.nan

    with pytest.raises(RuntimeError, match="non-finite"):
        dsv4_packed_attn.report_check(packed, stock, n_local_heads=16)
