# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FX-graph-shape test for the q-quant-separation refactor.

This is the test that pins the **whole point** of Phase 1: the lifted
decode q-prep must appear as discrete nodes in the FX graph so a
downstream pattern matcher (Phase 2) can rewrite the chain into the
AITER fused kernel.

We compile ``_maybe_prepare_decode_mqa_q`` end-to-end with vLLM's
``TestBackend`` and assert that the post-Dynamo / pre-pass graph contains
the four lifted ops as separate ``call_function`` nodes, in the right
order:

* ``vllm::mla_decode_q_take``  (slices q to ``num_decode_tokens`` rows)
* ``vllm::unified_mla_q_absorb``
* ``aten::cat``
* the static FP8 quant op

``vllm::unified_mla_attention_with_output`` is *not* exercised here (we
test the prep in isolation), but its ``prepared_mqa_q`` kwarg is
validated separately by the wiring test.

If this test ever starts failing, **stop**: it means the lift has
collapsed back into the opaque attention custom op and Phase 2's
pattern matcher will have nothing to match against.
"""

from __future__ import annotations

import pytest
import torch

from tests.compile.backend import TestBackend
from vllm.compilation.passes.fx_utils import find_op_nodes
from vllm.forward_context import override_forward_context
from vllm.model_executor.layers.attention.mla_attention import (
    MLAAttention,
    _DecodeConcatQuantFP8,
)
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.utils.torch_utils import LayerNameType, _encode_layer_name, set_random_seed

_QK_NOPE = 128
_QK_ROPE = 64
_KV_LORA = 512
_NUM_HEADS = 16
_SEQ_LEN = 4
_LAYER_NAME = "fxtest_layer.0"


def _make_stub_layer(
    device: torch.device, dtype: torch.dtype, *, lift: bool
) -> MLAAttention:
    layer: MLAAttention = object.__new__(MLAAttention)
    torch.nn.Module.__init__(layer)
    layer.qk_nope_head_dim = _QK_NOPE
    layer.qk_rope_head_dim = _QK_ROPE
    layer.kv_lora_rank = _KV_LORA
    layer._lift_q_decode_quant = lift
    layer.layer_name = _LAYER_NAME
    layer.is_aiter_triton_fp4_bmm_enabled = False
    layer.is_aiter_triton_fp8_bmm_enabled = False
    layer.W_UK_T = torch.randn(
        _NUM_HEADS, _QK_NOPE, _KV_LORA, dtype=dtype, device=device
    )
    layer._q_scale = torch.tensor(0.125, dtype=torch.float32, device=device)
    layer._quant_fp8_op = QuantFP8(
        static=True, group_shape=GroupShape.PER_TENSOR, compile_native=True
    )
    layer._decode_concat_quant_fp8_op = _DecodeConcatQuantFP8(
        static=True, group_shape=GroupShape.PER_TENSOR, compile_native=True
    )
    return layer


class _LiftedQPrep(torch.nn.Module):
    """Thin wrapper around ``MLAAttention._maybe_prepare_decode_mqa_q`` so
    we can ``torch.compile`` it in isolation."""

    def __init__(self, layer: MLAAttention, encoded: LayerNameType) -> None:
        super().__init__()
        self.layer = layer
        self.encoded = encoded

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        out = self.layer._maybe_prepare_decode_mqa_q(q, self.encoded)
        assert out is not None
        return out


def _make_forward_context(layer: MLAAttention, num_decode_tokens: int):
    """Build a forward context with a stub attn_metadata that only carries
    ``num_decode_tokens``. The lifted slice op (``mla_decode_q_take``) is
    the only thing in this test that reads from attn_metadata.
    """
    import types

    from vllm.forward_context import ForwardContext

    fake_attn_meta = types.SimpleNamespace(num_decode_tokens=num_decode_tokens)
    return ForwardContext(
        no_compile_layers={layer.layer_name: layer},
        all_moe_layers=None,
        attn_metadata={layer.layer_name: fake_attn_meta},
        slot_mapping={},
        dp_metadata=None,
    )


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="requires GPU")
@torch.inference_mode()
def test_lifted_prep_emits_discrete_fx_nodes(default_vllm_config) -> None:
    """The lifted prep, after Dynamo trace, must show the decode-rows
    slice op, the q-absorb BMM op, the cat, and the FP8 quant op as
    separate ``call_function`` nodes.
    """
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    set_random_seed(0)

    layer = _make_stub_layer(device, dtype, lift=True)
    encoded = _encode_layer_name(_LAYER_NAME)
    module = _LiftedQPrep(layer, encoded).eval()

    backend = TestBackend()  # no custom passes; we only want the captured graph
    compiled = torch.compile(module, backend=backend, fullgraph=True)

    q = torch.randn(
        _SEQ_LEN, _NUM_HEADS, _QK_NOPE + _QK_ROPE, dtype=dtype, device=device
    )

    # Tell the slice op to keep all rows so the rest of the prep runs on a
    # non-empty tensor; the FX-node assertions are independent of this size.
    fctx = _make_forward_context(layer, num_decode_tokens=_SEQ_LEN)
    with override_forward_context(fctx):
        out = compiled(q)

    assert out is not None and out.dtype == current_platform.fp8_dtype()
    assert out.shape == (_SEQ_LEN, _NUM_HEADS, _KV_LORA + _QK_ROPE), out.shape
    assert backend.graph_pre_pass is not None, (
        "TestBackend never captured a graph; torch.compile may have fallen "
        "through without invoking the post_grad pass hook."
    )
    graph = backend.graph_pre_pass

    # --- The discrete ops we expect to see as separate FX nodes ---
    q_take = torch.ops.vllm.mla_decode_q_take
    q_absorb = torch.ops.vllm.unified_mla_q_absorb
    high_level_fp8_quant = torch.ops._C.static_scaled_fp8_quant
    fp8_dtype = current_platform.fp8_dtype()

    q_take_nodes = list(find_op_nodes(q_take, graph))
    assert len(q_take_nodes) >= 1, (
        "vllm::mla_decode_q_take is missing from the post-Dynamo graph. "
        "The decode-rows slice collapsed back into the opaque attention "
        "op or was inlined by Dynamo, which would re-introduce the "
        "prefill-rows BMM waste. Graph dump:\n"
        f"{graph}"
    )

    q_absorb_nodes = list(find_op_nodes(q_absorb, graph))
    assert len(q_absorb_nodes) >= 1, (
        "vllm::unified_mla_q_absorb is missing from the post-Dynamo graph. "
        "The lift collapsed back into something else; Phase 2's pattern "
        "matcher will have nothing to match. Graph dump:\n"
        f"{graph}"
    )

    cat_nodes = list(find_op_nodes(torch.ops.aten.cat, graph))
    assert len(cat_nodes) >= 1, (
        f"aten::cat is missing from the post-Dynamo graph:\n{graph}"
    )

    # The FP8 quant is visible in the graph in one of two shapes,
    # depending on whether QuantFP8 lowered to its high-level op or got
    # decomposed by ``compile_native=True``:
    #
    # 1. As a single ``vllm._C.static_scaled_fp8_quant`` call, or
    # 2. As a chain ending in ``prims.convert_element_type`` to the
    #    platform's FP8 dtype (preceded by ``clamp_min`` / ``clamp_max``).
    #
    # Either way it must NOT be folded into ``unified_mla_q_absorb`` or
    # any other opaque op — it has to be matchable by the Phase 2 pass.
    high_level_quant_nodes = list(find_op_nodes(high_level_fp8_quant, graph))
    fp8_cast_nodes = [
        n
        for n in graph.find_nodes(
            op="call_function",
            target=torch.ops.prims.convert_element_type.default,
        )
        if len(n.args) >= 2 and n.args[1] == fp8_dtype
    ]
    assert high_level_quant_nodes or fp8_cast_nodes, (
        "Neither static_scaled_fp8_quant nor a prims.convert_element_type "
        f"to {fp8_dtype} is present in the post-Dynamo graph. The lifted "
        "FP8 quant has been fused away or eliminated, which would defeat "
        "the purpose of the lift.\n"
        f"{graph}"
    )

    # Topology: mla_decode_q_take ->  q_absorb -> cat -> FP8 quant.
    # The pattern matcher in Phase 2 will rely on this ordering.
    quant_nodes = high_level_quant_nodes or fp8_cast_nodes
    indexed = list(graph.nodes)
    q_take_idx = min(indexed.index(n) for n in q_take_nodes)
    q_absorb_idx = min(indexed.index(n) for n in q_absorb_nodes)
    cat_idx = min(indexed.index(n) for n in cat_nodes)
    quant_idx = min(indexed.index(n) for n in quant_nodes)

    assert q_take_idx < q_absorb_idx, (
        "Expected mla_decode_q_take to appear before unified_mla_q_absorb "
        f"in the FX node ordering. Graph dump:\n{graph}"
    )
    assert q_absorb_idx < cat_idx, (
        f"Expected unified_mla_q_absorb to appear before aten.cat:\n{graph}"
    )
    assert q_absorb_idx < quant_idx, (
        "Expected unified_mla_q_absorb to appear before the FP8 quant in "
        "the FX node ordering, but they are reversed. Graph dump:\n"
        f"{graph}"
    )
