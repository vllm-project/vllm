# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the indexer_only trim (mono/indexer_only.py): every value the indexer
consumes is unchanged, the MQA dummies are persistent zero buffers, and the q-row slice
of qkv_a falls back to the full GEMM unless it is bit-identical."""

from types import SimpleNamespace as NS

import pytest
import torch

from vllm.models.deepseek_v32.amd.mono import indexer_only as IO

QL, KVL, PE, H, NH, HD = 24, 16, 8, 32, 4, 8
CALLS: list = []


@pytest.fixture(autouse=True)
def _fake_kernels(monkeypatch):
    def fused_norm_rope(
        positions, q_c, qw, qe, kv_c, kw_, ke, k_pe, cs, index_k, *a, **kw
    ):
        CALLS.append(
            (
                "fnr",
                q_c.clone(),
                index_k.clone(),
                kv_c.shape,
                k_pe.shape,
                kw.get("mla_kv_cache"),
            )
        )
        return q_c * 2

    def fused_q(positions, q_pe, cs, index_q, ics, ql_nope, *a, **kw):
        CALLS.append(
            (
                "fq",
                q_pe.data_ptr(),
                ql_nope.data_ptr(),
                bool((q_pe == 0).all()),
                bool((ql_nope == 0).all()),
                index_q.clone(),
            )
        )
        return index_q + 1, index_q.sum(-1), None

    monkeypatch.setattr(IO, "fused_norm_rope", fused_norm_rope)
    monkeypatch.setattr(IO, "fused_q", fused_q)
    fc = NS(attn_metadata={"L": object()}, slot_mapping={"L": torch.arange(4)})
    monkeypatch.setattr("vllm.forward_context.get_forward_context", lambda: fc)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *a, **k: None)


class Lin:
    def __init__(self, rows, noise=0.0, raises=False):
        g = torch.Generator().manual_seed(rows)
        # small integers: exact sums, so a row-slice GEMM == the full one
        self.weight = torch.randint(-3, 4, (rows, H), generator=g).float()
        self.noise, self.raises = noise, raises
        lin = self

        class QM:
            def apply(self, layer, x, bias=None):
                if lin.raises:
                    raise RuntimeError("needs more than .weight")
                return x @ layer.weight.T + lin.noise

        self.quant_method = QM()

    def __call__(self, x):
        return x @ self.weight.T, None


def make_attn(noise=0.0, raises=False):
    ind = NS(
        wk_weights_proj=Lin(HD + NH),
        head_dim=HD,
        n_head=NH,
        k_norm=NS(weight=None, bias=None, eps=1e-6),
        k_cache=NS(kv_cache=None, uses_shuffled_layout=False),
        softmax_scale=1.0,
        wq_b=lambda q: (q[:, : NH * HD].contiguous(), None),
    )
    return NS(
        indexer=ind,
        skip_topk=False,
        fused_qkv_a_proj=Lin(QL + KVL + PE, noise, raises),
        q_lora_rank=QL,
        kv_lora_rank=KVL,
        qk_rope_head_dim=PE,
        layer_name="L",
        q_a_layernorm=NS(weight=None, variance_epsilon=0),
        kv_a_layernorm=NS(weight=None, variance_epsilon=0),
        rotary_emb=NS(cos_sin_cache=None),
        indexer_rope_emb=NS(cos_sin_cache=None),
        topk_indices_buffer=None,
        kv_cache_dtype="auto",
        _index_rope_interleave=True,
        num_local_heads=2,
        _q_scale=None,
        _fp8_kv=False,
        _run_indexer=lambda q_c, iq, iw: CALLS.append(
            ("run", q_c.clone(), iq.clone(), iw.clone())
        ),
    )


def run(attn, x, trim):
    CALLS.clear()
    IO.refresh_indexer(attn, torch.arange(x.shape[0]), x, trim=trim)
    return list(CALLS)


def test_trim_identical_outputs():
    x = torch.randint(-3, 4, (4, H)).float()
    attn = make_attn()
    base = run(attn, x, False)
    t1 = run(attn, x, True)
    t2 = run(attn, x, True)
    # q_c into fused_norm_rope, index_k, index_q and the _run_indexer args are equal
    assert torch.equal(base[0][1], t1[0][1]) and torch.equal(base[0][2], t1[0][2])
    assert torch.equal(base[1][5], t1[1][5])
    assert all(torch.equal(a, b) for a, b in zip(base[2][1:], t1[2][1:]))
    assert base[0][5] is None and t1[0][5] is None  # the MLA write stays disabled
    # persistent dummies: the same zero-filled buffers on every call
    assert t1[1][1] == t2[1][1] and t1[1][2] == t2[1][2] and t1[1][3] and t1[1][4]
    assert IO._QSLICE[(id(attn), 4)] is True


def test_trim_fallbacks():
    x = torch.randint(-3, 4, (4, H)).float()
    # the sliced GEMM is not bit-identical -> full GEMM for this width from now on
    a1 = make_attn(noise=1e-3)
    base = run(make_attn(noise=1e-3), x, False)
    t = run(a1, x, True)
    assert IO._QSLICE[(id(a1), 4)] is False and torch.equal(base[0][1], t[0][1])
    # the quant method needs more than the weight -> full GEMM
    a2 = make_attn(raises=True)
    t = run(a2, x, True)
    assert IO._QSLICE[(id(a2), 4)] is False and t[0][3] == (4, KVL)
