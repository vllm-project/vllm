# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash specialized decode, layer side (eligibility, weight views, row
tables), without a GPU."""

import pytest
import torch

from vllm.models.glm5next.amd.specialized_decode.runtime import layers, spec
from vllm.models.glm5next.amd.specialized_decode.runtime.context import (
    DecodeContext,
    DecodeStep,
)


def _ctx(tp: int) -> DecodeContext:
    return DecodeContext(None, tp, 0, None, torch.device("cpu"))


def _step(width: int) -> DecodeStep:
    return DecodeStep(width, width, torch.zeros(width, dtype=torch.long), {})


@pytest.mark.parametrize("tp", spec.TP_SIZES)
@pytest.mark.parametrize("width", spec.WIDTHS)
def test_eligible_caps_width_per_tp(tp: int, width: int):
    cap = spec.TP_MAX_WIDTH.get(tp, max(spec.WIDTHS))
    reason = layers.eligible(_ctx(tp), _step(width))
    assert (reason is None) == (width <= cap), (tp, width, reason)


def test_layer_kinds():
    from types import SimpleNamespace

    assert spec.is_kda(SimpleNamespace(layer_kind="kda"))
    assert spec.is_dsa(SimpleNamespace(layer_kind="mla"))
    assert not spec.is_dsa(SimpleNamespace(layer_kind="kda"))
    hf = SimpleNamespace(
        num_hidden_layers=6, mlp_layer_types=["dense"] + ["sparse"] * 5
    )
    assert spec.kernel_layers(hf) == [1, 2, 3, 4]  # not the last (mHC contraction)


def _linear(w, scale=None):
    from types import SimpleNamespace

    return SimpleNamespace(weight=w, weight_scale_inv=scale)


def test_projection_views_accept_fp8_or_bf16_and_reject_the_rest():
    from vllm.models.glm5next.amd.specialized_decode.runtime.weights import _proj

    w8 = torch.zeros(256, 512, dtype=torch.float8_e4m3fn)
    s8 = torch.zeros(2, 4)
    assert _proj("p", _linear(w8, s8), 256, 512)[1] is s8
    assert (
        _proj("p", _linear(torch.zeros(256, 512, dtype=torch.bfloat16)), 256, 512)[1]
        is None
    )
    bad = (
        _linear(w8, None),  # FP8 without scales
        _linear(w8, torch.zeros(2, 3)),  # scales of the wrong grid
        _linear(w8, torch.zeros(2, 4, dtype=torch.float16)),  # scale dtype
        _linear(torch.zeros(256, 512, dtype=torch.float16)),  # weight dtype
        _linear(torch.zeros(128, 512, dtype=torch.bfloat16)),  # shard shape
        _linear(torch.zeros(512, 256, dtype=torch.float8_e4m3fn).T, s8),  # strides
    )
    for lin in bad:
        with pytest.raises(ValueError, match="needs p as row-major"):
            _proj("p", lin, 256, 512)


def test_dsa_weights_name_every_mismatch():
    from types import SimpleNamespace

    from vllm.models.glm5next.amd.specialized_decode.runtime.weights import dsa_weights

    def lin(*shape, dtype=torch.bfloat16):
        return SimpleNamespace(weight=torch.zeros(*shape, dtype=dtype))

    attn = SimpleNamespace(
        fused_qkv_a_proj=lin(2048, 4096),
        q_b_proj=lin(16 * 256, 1536),
        o_proj=lin(4096, 16 * 256),
        kv_b_proj=lin(16 * 512, 512),
        q_a_layernorm=lin(1536),
        kv_a_layernorm=lin(512),
        indexer=SimpleNamespace(
            wk_weights_proj=lin(160, 4096),
            index_kpool_compress_gate=torch.zeros(128, 4096, dtype=torch.bfloat16),
            k_norm=SimpleNamespace(weight=torch.ones(128), bias=torch.zeros(128)),
            index_kpool_compress_ape=torch.zeros(4, 128),
            wq_b=lin(4096, 1536),
        ),
    )
    layer = SimpleNamespace(
        self_attn=attn,
        hc_attn_fn=torch.zeros(24, 16384),
        hc_attn_scale=torch.zeros(3),
        hc_attn_base=torch.zeros(24),
        input_layernorm=lin(4096),
    )
    w = dsa_weights(layer, 4)
    assert w.fp8 == (False, False, False)
    assert w.w_uk.shape == (16, 512, 256) and w.w_uk.is_contiguous()
    attn.indexer.wq_b = lin(4096, 1024)
    with pytest.raises(ValueError, match="indexer.wq_b.weight"):
        dsa_weights(layer, 4)


def test_dsa_row_table_pads_and_follows_the_step():
    from types import SimpleNamespace

    from vllm.models.glm5next.amd.specialized_decode.mono.ffn.layout import (
        ROW_BT,
        ROW_POOL_SLOT,
        ROW_POS,
        ROW_SLOT,
        ROW_TAIL_SLOT,
    )

    w = layers._Width(None, 4, torch.device("cpu"))
    md = {
        "mla": SimpleNamespace(slot_mapping=torch.tensor([70, 9])),
        "ic": SimpleNamespace(slot_mapping=torch.tensor([17, -1])),
        "tail": SimpleNamespace(slot_mapping=torch.tensor([7, 13])),
    }
    step = DecodeStep(2, 4, torch.tensor([71, 9]), md)
    rows = w.dsa_rows(step, ("mla", "ic", "tail"))
    assert rows[:2, ROW_POS].tolist() == [71, 9]
    assert rows[:2, ROW_SLOT].tolist() == [70, 9]
    assert rows[:2, ROW_POOL_SLOT].tolist() == [17, -1]
    assert rows[:2, ROW_TAIL_SLOT].tolist() == [7, 13]
    assert (rows[2:, ROW_POS : ROW_TAIL_SLOT + 1] == -1).all()
    assert rows[:, ROW_BT].tolist() == [0, 1, 2, 3]
    # computed once a step, shared by every sparse-attention layer
    md["mla"].slot_mapping[0] = 5
    assert w.dsa_rows(step, ("mla", "ic", "tail"))[0, ROW_SLOT] == 70


def test_narrow_step_pads_once_then_chains_padded_outputs():
    from vllm.models.glm5next.amd.specialized_decode.mono.config import (
        HC_MULT,
        HIDDEN,
    )

    ctx = _ctx(2)
    w = layers._Width(None, 2, torch.device("cpu"))
    a, b = (layers.FlashLayer(ctx, None, None, w, (), i) for i in range(2))
    step = DecodeStep(1, 2, torch.zeros(1, dtype=torch.long), {})

    def mhc_state(n):
        return (
            torch.randn(n, HIDDEN).bfloat16(),
            torch.randn(n, HC_MULT, HIDDEN).bfloat16(),
            torch.rand(n, HC_MULT, 1),
            torch.rand(n, HC_MULT, HC_MULT),
        )

    # from vLLM's layer: copied into the padding buffers
    first = mhc_state(1)
    ins = a.inputs(step, *first)
    assert ins is w.pad and all(torch.equal(p[:1], t) for p, t in zip(ins, first))
    # from the previous kernel layer: its padded buffers, no copy
    out = mhc_state(2)
    rows = a.finish(step, out)
    assert all(r.shape[0] == 1 for r in rows)
    assert all(x is y for x, y in zip(b.inputs(step, *rows), out))
    # anything else (a new step's first layer) is padded again
    assert b.inputs(step, *mhc_state(1)) is w.pad
