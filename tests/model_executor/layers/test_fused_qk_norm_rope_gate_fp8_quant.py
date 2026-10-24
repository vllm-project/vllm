# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm.model_executor.layers import (
    fused_qk_norm_rope_gate_fp8_quant as prep_module,
)
from vllm.model_executor.layers.attention import attention as attention_module
from vllm.model_executor.models import qwen3_next as qwen3_next_model
from vllm.v1.attention.backend import PrequantizedQKV

HEAD_DIM = 256
ROTARY_DIM = 64
MAX_NUM_SEQS = 256

# Query and KV heads per rank for a 16-query-head, 2-KV-head model at TP1, TP2
# and TP4 (KV heads replicate once TP exceeds them). All have a GQA ratio the
# backend admits.
HEADS = pytest.mark.parametrize(
    ("num_heads", "num_kv_heads"),
    [(16, 2), (8, 1), (4, 1)],
    ids=["tp1", "tp2", "tp4"],
)
# Prefill and extend tokens in one step: a short prompt, an odd count and a
# full 2048-token chunk. Pure decode batches are in DECODE_BATCHES.
PREFILL_TOKENS = pytest.mark.parametrize("tokens", [2, 7, 129, 2048])
DECODE_BATCHES = pytest.mark.parametrize("tokens", [1, 7, 128, MAX_NUM_SEQS])


@pytest.mark.parametrize(
    ("max_num_seqs", "expected"),
    [
        (256, True),
        (257, False),
        (None, False),
    ],
)
def test_supports_prequantized_scheduler(
    max_num_seqs,
    expected,
):
    assert prep_module.supports_prequantized_scheduler(max_num_seqs) is expected


def _make_inputs(tokens, num_heads, num_kv_heads):
    q_gate = torch.randn(tokens, num_heads * 2 * HEAD_DIM)
    key = torch.randn(tokens, num_kv_heads * HEAD_DIM)
    return (
        q_gate,
        key,
        torch.randn_like(key),
        torch.randn(HEAD_DIM),
        torch.randn(HEAD_DIM),
        torch.randn(tokens, ROTARY_DIM),
        torch.arange(tokens),
    )


def _make_outputs(inputs, num_heads, num_kv_heads):
    q_gate, key, *_ = inputs
    tokens = q_gate.shape[0]
    device = q_gate.device
    query = torch.randn(tokens, num_heads * HEAD_DIM)
    key_fp8 = torch.zeros(
        tokens, num_kv_heads, HEAD_DIM, dtype=torch.float8_e4m3fn, device=device
    )
    query_descale = torch.randn(MAX_NUM_SEQS, num_kv_heads, device=device)
    return (
        query,
        torch.randn_like(key),
        torch.randn_like(query),
        torch.zeros(
            tokens, num_heads, HEAD_DIM, dtype=torch.float8_e4m3fn, device=device
        ),
        key_fp8,
        torch.zeros_like(key_fp8),
        query_descale,
        torch.randn_like(query_descale),
        torch.randn_like(query_descale),
    )


def _run_impl(inputs, outputs, num_heads, num_kv_heads) -> None:
    prep_module._fused_qk_norm_rope_gate_fp8_quant_impl(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
        inputs[6],
        "layer",
        1.0e-6,
        num_heads,
        num_kv_heads,
        HEAD_DIM,
        ROTARY_DIM,
        *outputs,
    )


@HEADS
@PREFILL_TOKENS
def test_fused_qk_norm_rope_gate_fp8_quant_translates_attention_metadata(
    monkeypatch, tokens, num_heads, num_kv_heads
):
    inputs = _make_inputs(tokens, num_heads, num_kv_heads)
    expected_outputs = _make_outputs(inputs, num_heads, num_kv_heads)
    # One decode token, then one extend sequence with the rest.
    query_start_loc = torch.tensor([0, 1, tokens], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_actual_tokens=tokens,
        query_start_loc=query_start_loc,
        num_decodes=1,
        num_decode_tokens=1,
        num_extends=1,
        num_prefills=0,
    )
    monkeypatch.setattr(
        attention_module,
        "get_attention_context",
        lambda layer_name: (metadata, None, None, None),
    )
    recorded_kwargs = None

    def fake_aiter_prep(*args, **kwargs):
        nonlocal recorded_kwargs
        recorded_kwargs = kwargs
        for name, source in zip(
            (
                "query_out",
                "key_out",
                "gate_out",
                "query_fp8_out",
                "key_fp8_out",
                "value_fp8_out",
                "query_descale_out",
                "key_descale_out",
                "value_descale_out",
            ),
            expected_outputs,
        ):
            kwargs[name].copy_(source)
        return expected_outputs

    monkeypatch.setattr(
        prep_module.rocm_aiter_ops,
        "fused_qk_norm_rope_gate_fp8_quant",
        fake_aiter_prep,
    )

    actual_outputs = _make_outputs(inputs, num_heads, num_kv_heads)
    _run_impl(inputs, actual_outputs, num_heads, num_kv_heads)

    for actual, expected in zip(actual_outputs, expected_outputs):
        torch.testing.assert_close(actual, expected)
    assert recorded_kwargs is not None
    assert recorded_kwargs["num_actual_tokens"] == tokens
    assert recorded_kwargs["quant_token_start"] == 1
    assert recorded_kwargs["quant_sequence_start"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph support")
@HEADS
@PREFILL_TOKENS
def test_fused_qk_norm_rope_gate_fp8_quant_without_metadata_is_cuda_graph_safe(
    monkeypatch, tokens, num_heads, num_kv_heads
):
    inputs = tuple(
        tensor.to(device="cuda")
        for tensor in _make_inputs(tokens, num_heads, num_kv_heads)
    )
    monkeypatch.setattr(
        attention_module,
        "get_attention_context",
        lambda layer_name: (None, None, None, None),
    )
    recorded_cu_seqlens = None
    outputs = _make_outputs(inputs, num_heads, num_kv_heads)
    expected_outputs = _make_outputs(inputs, num_heads, num_kv_heads)

    def fake_aiter_prep(*args, **kwargs):
        nonlocal recorded_cu_seqlens
        recorded_cu_seqlens = args[7]
        for name, source in zip(
            (
                "query_out",
                "key_out",
                "gate_out",
                "query_fp8_out",
                "key_fp8_out",
                "value_fp8_out",
                "query_descale_out",
                "key_descale_out",
                "value_descale_out",
            ),
            expected_outputs,
        ):
            kwargs[name].copy_(source)
        return expected_outputs

    monkeypatch.setattr(
        prep_module.rocm_aiter_ops,
        "fused_qk_norm_rope_gate_fp8_quant",
        fake_aiter_prep,
    )

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        _run_impl(inputs, outputs, num_heads, num_kv_heads)
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        _run_impl(inputs, outputs, num_heads, num_kv_heads)
    graph.replay()
    torch.accelerator.synchronize()

    assert recorded_cu_seqlens is not None
    torch.testing.assert_close(
        recorded_cu_seqlens,
        torch.tensor([0, tokens], dtype=torch.int32, device="cuda"),
    )
    for actual, expected in zip(outputs, expected_outputs):
        torch.testing.assert_close(actual, expected)


@HEADS
@DECODE_BATCHES
def test_fused_qk_norm_rope_gate_fp8_quant_pure_decode_uses_bf16_fallback(
    monkeypatch, tokens, num_heads, num_kv_heads
):
    inputs = _make_inputs(tokens, num_heads, num_kv_heads)
    metadata = SimpleNamespace(
        num_actual_tokens=tokens,
        query_start_loc=torch.arange(tokens + 1, dtype=torch.int32),
        num_decodes=tokens,
        num_decode_tokens=tokens,
        num_extends=0,
        num_prefills=0,
    )
    monkeypatch.setattr(
        attention_module,
        "get_attention_context",
        lambda layer_name: (metadata, None, None, None),
    )
    expected_query = torch.randn(tokens, num_heads * HEAD_DIM)
    expected_key = torch.randn(tokens, num_kv_heads * HEAD_DIM)
    expected_gate = torch.randn_like(expected_query)
    recorded_weights = None
    recorded_kwargs = None

    def fake_bf16_prep(
        q_gate,
        key,
        query_norm_weight,
        key_norm_weight,
        *args,
        **kwargs,
    ):
        nonlocal recorded_kwargs, recorded_weights
        recorded_weights = (query_norm_weight, key_norm_weight)
        recorded_kwargs = kwargs
        kwargs["query_out"].copy_(expected_query)
        kwargs["key_out"].copy_(expected_key)
        kwargs["gate_out"].copy_(expected_gate)

    monkeypatch.setattr(
        prep_module,
        "fused_qk_rmsnorm_rope_gate",
        fake_bf16_prep,
    )
    monkeypatch.setattr(
        prep_module.rocm_aiter_ops,
        "fused_qk_norm_rope_gate_fp8_quant",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("AITER producer must not run for pure decode")
        ),
    )

    outputs = _make_outputs(inputs, num_heads, num_kv_heads)
    _run_impl(inputs, outputs, num_heads, num_kv_heads)

    torch.testing.assert_close(outputs[0], expected_query)
    torch.testing.assert_close(outputs[1], expected_key)
    torch.testing.assert_close(outputs[2], expected_gate)
    assert recorded_weights is not None
    assert recorded_weights[0] is inputs[3]
    assert recorded_weights[1] is inputs[4]
    assert recorded_kwargs is not None
    assert recorded_kwargs["norm_beta"] == 1.0
    assert recorded_kwargs["query_out"] is outputs[0]
    assert recorded_kwargs["key_out"] is outputs[1]
    assert recorded_kwargs["gate_out"] is outputs[2]
    assert outputs[3].shape == (tokens, num_heads, HEAD_DIM)
    assert outputs[4].shape == (tokens, num_kv_heads, HEAD_DIM)
    assert outputs[6].shape == (MAX_NUM_SEQS, num_kv_heads)


@HEADS
@PREFILL_TOKENS
def test_fused_qk_norm_rope_gate_fp8_quant_custom_op_meta_shapes(
    tokens, num_heads, num_kv_heads
):
    inputs = tuple(
        tensor.to(device="meta")
        for tensor in _make_inputs(tokens, num_heads, num_kv_heads)
    )
    outputs = prep_module._allocate_outputs(
        inputs[0], inputs[1], inputs[2], num_heads, num_kv_heads, HEAD_DIM
    )
    result = torch.ops.vllm.fused_qk_norm_rope_gate_fp8_quant(
        *inputs,
        "layer",
        1.0e-6,
        num_heads,
        num_kv_heads,
        HEAD_DIM,
        ROTARY_DIM,
        *outputs,
    )

    assert result is None
    assert outputs[0].shape == (tokens, num_heads * HEAD_DIM)
    assert outputs[1].shape == (tokens, num_kv_heads * HEAD_DIM)
    assert outputs[3].shape == (tokens, num_heads, HEAD_DIM)
    assert outputs[4].shape == (tokens, num_kv_heads, HEAD_DIM)
    assert outputs[6].shape == (MAX_NUM_SEQS, num_kv_heads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@HEADS
def test_fused_qk_norm_rope_gate_fp8_quant_opcheck(
    monkeypatch, num_heads, num_kv_heads
):
    tokens = 7
    inputs = tuple(
        tensor.to(device="cuda")
        for tensor in _make_inputs(tokens, num_heads, num_kv_heads)
    )
    outputs = _make_outputs(inputs, num_heads, num_kv_heads)
    metadata = SimpleNamespace(
        num_actual_tokens=tokens,
        query_start_loc=torch.tensor([0, tokens], dtype=torch.int32, device="cuda"),
        num_decodes=0,
        num_decode_tokens=0,
    )
    monkeypatch.setattr(
        attention_module,
        "get_attention_context",
        lambda layer_name: (metadata, None, None, None),
    )
    # The AITER kernel itself is not under test; the op's schema, fake
    # implementation and AOT dispatch are.
    monkeypatch.setattr(
        prep_module.rocm_aiter_ops,
        "fused_qk_norm_rope_gate_fp8_quant",
        lambda *args, **kwargs: None,
    )

    op = torch.ops.vllm.fused_qk_norm_rope_gate_fp8_quant
    # Piecewise CUDA graphs would replay capture-time metadata offsets.
    assert torch.Tag.cudagraph_unsafe in op.default.tags
    opcheck(
        op,
        (
            *inputs,
            "layer",
            1.0e-6,
            num_heads,
            num_kv_heads,
            HEAD_DIM,
            ROTARY_DIM,
            *outputs,
        ),
    )


@HEADS
@PREFILL_TOKENS
def test_qwen3_next_model_builds_prequantized_bundle(
    monkeypatch, tokens, num_heads, num_kv_heads
):
    inputs = _make_inputs(tokens, num_heads, num_kv_heads)
    expected_outputs = _make_outputs(inputs, num_heads, num_kv_heads)
    attention = object.__new__(qwen3_next_model.Qwen3NextAttention)
    torch.nn.Module.__init__(attention)
    attention.use_prequantized_qkv = True
    attention.use_fused_qk_norm_rope_gate = True
    attention.q_size = num_heads * HEAD_DIM
    attention.kv_size = num_kv_heads * HEAD_DIM
    attention.num_heads = num_heads
    attention.num_kv_heads = num_kv_heads
    attention.head_dim = HEAD_DIM
    attention.q_norm = SimpleNamespace(
        weight=inputs[3],
        variance_epsilon=1.0e-6,
    )
    attention.k_norm = SimpleNamespace(weight=inputs[4])
    attention.rotary_emb = SimpleNamespace(
        cos_sin_cache=inputs[5],
        rotary_dim=ROTARY_DIM,
    )
    attention.attn = SimpleNamespace(layer_name="layer")
    monkeypatch.setattr(
        qwen3_next_model,
        "fused_qk_norm_rope_gate_fp8_quant",
        lambda *args, **kwargs: expected_outputs,
    )

    qkv = torch.cat((inputs[0], inputs[1], inputs[2]), dim=-1)
    query, key, value, gate, prequantized_qkv = (
        attention._project_prequantized_qkv_gate(qkv, inputs[6])
    )

    assert query is expected_outputs[0]
    assert key is expected_outputs[1]
    torch.testing.assert_close(value, inputs[2])
    assert gate is expected_outputs[2]
    assert isinstance(prequantized_qkv, PrequantizedQKV)
    assert prequantized_qkv.query is expected_outputs[3]
    assert prequantized_qkv.key is expected_outputs[4]
    assert prequantized_qkv.value is expected_outputs[5]


@pytest.mark.parametrize("use_prequantized_qkv", [False, True])
def test_qwen3_next_forward_dispatches_projection(use_prequantized_qkv):
    # Qwen4Exp's attention subclasses Qwen3NextAttention and unpacks four
    # values from _project_qkv_gate, so the prequantized path must not change it.
    q, k, v, gate = (torch.randn(2, 4) for _ in range(4))
    bundle = object()
    attn_calls = []

    def attn(*args, **kwargs):
        attn_calls.append((args, kwargs))
        return q

    attention: Any = SimpleNamespace(
        use_prequantized_qkv=use_prequantized_qkv,
        qkv_proj=lambda hidden_states: (hidden_states, None),
        o_proj=lambda attn_output: (attn_output, None),
        attn=attn,
        _project_qkv_gate=lambda qkv, positions: (q, k, v, gate),
        _project_prequantized_qkv_gate=lambda qkv, positions: (q, k, v, gate, bundle),
    )

    qwen3_next_model.Qwen3NextAttention.forward(
        attention, torch.arange(2), torch.randn(2, 4)
    )

    ((args, kwargs),) = attn_calls
    assert args == (q, k, v)
    assert kwargs == ({"prequantized_qkv": bundle} if use_prequantized_qkv else {})
