# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers import qwen3_next_fp8_qkv as prep_module
from vllm.model_executor.layers.attention import attention as attention_module
from vllm.model_executor.models import qwen3_next as qwen3_next_model
from vllm.v1.attention.backend import PrequantizedQKV

NUM_QUERY_HEADS = 8
NUM_KV_HEADS = 1
HEAD_DIM = 256
ROTARY_DIM = 64


def _make_inputs(tokens: int = 8):
    q_gate = torch.randn(tokens, NUM_QUERY_HEADS * 2 * HEAD_DIM)
    key = torch.randn(tokens, NUM_KV_HEADS * HEAD_DIM)
    value = torch.randn_like(key)
    query_norm_weight = torch.randn(HEAD_DIM)
    key_norm_weight = torch.randn(HEAD_DIM)
    cos_sin_cache = torch.randn(tokens, ROTARY_DIM)
    positions = torch.arange(tokens)
    return (
        q_gate,
        key,
        value,
        query_norm_weight,
        key_norm_weight,
        cos_sin_cache,
        positions,
    )


def _make_outputs(inputs):
    q_gate, key, value, *_ = inputs
    tokens = q_gate.shape[0]
    query = torch.randn(tokens, NUM_QUERY_HEADS * HEAD_DIM)
    output_key = torch.randn_like(key)
    gate = torch.randn_like(query)
    query_fp8 = torch.empty(
        tokens, NUM_QUERY_HEADS, HEAD_DIM, dtype=torch.float8_e4m3fn
    )
    key_fp8 = torch.empty(tokens, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float8_e4m3fn)
    value_fp8 = torch.empty_like(key_fp8)
    query_descale = torch.randn(256, NUM_KV_HEADS)
    key_descale = torch.randn_like(query_descale)
    value_descale = torch.randn_like(query_descale)
    return (
        query,
        output_key,
        gate,
        query_fp8,
        key_fp8,
        value_fp8,
        query_descale,
        key_descale,
        value_descale,
    )


def test_qwen3_next_fp8_prep_translates_attention_metadata(monkeypatch):
    inputs = _make_inputs()
    expected_outputs = _make_outputs(inputs)
    query_start_loc = torch.tensor([0, 1, 7], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_actual_tokens=7,
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
        return expected_outputs

    monkeypatch.setattr(
        prep_module.rocm_aiter_ops,
        "qwen3_next_fp8_qkv_prep",
        fake_aiter_prep,
    )

    actual_outputs = prep_module._qwen3_next_fp8_qkv_prep_impl(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
        inputs[6],
        "layer",
        1.0e-6,
        NUM_QUERY_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        ROTARY_DIM,
    )

    assert all(
        actual is expected for actual, expected in zip(actual_outputs, expected_outputs)
    )
    assert recorded_kwargs is not None
    assert recorded_kwargs["num_actual_tokens"] == 7
    assert recorded_kwargs["quant_token_start"] == 1
    assert recorded_kwargs["quant_sequence_start"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph support")
def test_qwen3_next_fp8_prep_without_metadata_is_cuda_graph_safe(monkeypatch):
    inputs = tuple(tensor.to(device="cuda") for tensor in _make_inputs())
    monkeypatch.setattr(
        attention_module,
        "get_attention_context",
        lambda layer_name: (None, None, None, None),
    )
    recorded_cu_seqlens = None

    def fake_aiter_prep(*args, **kwargs):
        nonlocal recorded_cu_seqlens
        recorded_cu_seqlens = args[7]
        return (
            inputs[0],
            inputs[1],
            inputs[0],
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[3],
        )

    monkeypatch.setattr(
        prep_module.rocm_aiter_ops,
        "qwen3_next_fp8_qkv_prep",
        fake_aiter_prep,
    )

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        prep_module._qwen3_next_fp8_qkv_prep_impl(
            *inputs,
            "layer",
            1.0e-6,
            NUM_QUERY_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            ROTARY_DIM,
        )
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        prep_module._qwen3_next_fp8_qkv_prep_impl(
            *inputs,
            "layer",
            1.0e-6,
            NUM_QUERY_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            ROTARY_DIM,
        )
    graph.replay()
    torch.accelerator.synchronize()

    assert recorded_cu_seqlens is not None
    torch.testing.assert_close(
        recorded_cu_seqlens,
        torch.tensor([0, inputs[0].shape[0]], dtype=torch.int32, device="cuda"),
    )


def test_qwen3_next_fp8_prep_pure_decode_uses_bf16_fallback(monkeypatch):
    inputs = _make_inputs(tokens=4)
    metadata = SimpleNamespace(
        num_actual_tokens=4,
        query_start_loc=torch.arange(5, dtype=torch.int32),
        num_decodes=4,
        num_decode_tokens=4,
        num_extends=0,
        num_prefills=0,
    )
    monkeypatch.setattr(
        attention_module,
        "get_attention_context",
        lambda layer_name: (metadata, None, None, None),
    )
    expected_query = torch.randn(4, NUM_QUERY_HEADS * HEAD_DIM)
    expected_key = torch.randn(4, NUM_KV_HEADS * HEAD_DIM)
    expected_gate = torch.randn_like(expected_query)
    recorded_weights = None

    def fake_bf16_prep(
        q_gate,
        key,
        query_norm_weight,
        key_norm_weight,
        *args,
        **kwargs,
    ):
        nonlocal recorded_weights
        recorded_weights = (query_norm_weight, key_norm_weight)
        return expected_query, expected_key, expected_gate

    monkeypatch.setattr(
        prep_module,
        "fused_qk_rmsnorm_rope_gate",
        fake_bf16_prep,
    )
    monkeypatch.setattr(
        prep_module.rocm_aiter_ops,
        "qwen3_next_fp8_qkv_prep",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("AITER producer must not run for pure decode")
        ),
    )

    outputs = prep_module._qwen3_next_fp8_qkv_prep_impl(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
        inputs[6],
        "layer",
        1.0e-6,
        NUM_QUERY_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        ROTARY_DIM,
    )

    assert outputs[0] is expected_query
    assert outputs[1] is expected_key
    assert outputs[2] is expected_gate
    assert recorded_weights is not None
    torch.testing.assert_close(recorded_weights[0], inputs[3].float() + 1.0)
    torch.testing.assert_close(recorded_weights[1], inputs[4].float() + 1.0)
    assert outputs[3].shape == (4, NUM_QUERY_HEADS, HEAD_DIM)
    assert outputs[4].shape == (4, NUM_KV_HEADS, HEAD_DIM)
    assert outputs[6].shape == (256, NUM_KV_HEADS)


def test_qwen3_next_fp8_prep_custom_op_meta_shapes():
    inputs = tuple(tensor.to(device="meta") for tensor in _make_inputs())
    outputs = torch.ops.vllm.qwen3_next_fp8_qkv_prep(
        *inputs,
        "layer",
        1.0e-6,
        NUM_QUERY_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        ROTARY_DIM,
    )

    assert outputs[0].shape == (8, NUM_QUERY_HEADS * HEAD_DIM)
    assert outputs[1].shape == (8, NUM_KV_HEADS * HEAD_DIM)
    assert outputs[3].shape == (8, NUM_QUERY_HEADS, HEAD_DIM)
    assert outputs[4].shape == (8, NUM_KV_HEADS, HEAD_DIM)
    assert outputs[6].shape == (256, NUM_KV_HEADS)


def test_qwen3_next_model_builds_prequantized_bundle(monkeypatch):
    inputs = _make_inputs()
    expected_outputs = _make_outputs(inputs)
    attention = object.__new__(qwen3_next_model.Qwen3NextAttention)
    torch.nn.Module.__init__(attention)
    attention.use_prequantized_qkv = True
    attention.use_fused_qk_norm_rope_gate = True
    attention.q_size = NUM_QUERY_HEADS * HEAD_DIM
    attention.kv_size = NUM_KV_HEADS * HEAD_DIM
    attention.num_heads = NUM_QUERY_HEADS
    attention.num_kv_heads = NUM_KV_HEADS
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
        "qwen3_next_fp8_qkv_prep",
        lambda *args, **kwargs: expected_outputs,
    )

    qkv = torch.cat((inputs[0], inputs[1], inputs[2]), dim=-1)
    query, key, value, gate, prequantized_qkv = attention._project_qkv_gate(
        qkv,
        inputs[6],
    )

    assert query is expected_outputs[0]
    assert key is expected_outputs[1]
    torch.testing.assert_close(value, inputs[2])
    assert gate is expected_outputs[2]
    assert isinstance(prequantized_qkv, PrequantizedQKV)
    assert prequantized_qkv.query is expected_outputs[3]
    assert prequantized_qkv.key is expected_outputs[4]
    assert prequantized_qkv.value is expected_outputs[5]
