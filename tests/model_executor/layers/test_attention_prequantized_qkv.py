# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import regex as re
import torch

from vllm.model_executor.layers.attention import attention as attention_module
from vllm.model_executor.layers.attention import prequantized_qkv_attention as module
from vllm.model_executor.layers.attention.prequantized_qkv_attention import (
    PrequantizedQKV,
    PrequantizedQKVAttention,
)

# (tokens, query heads, KV heads, head size) as seen by one rank: decode, an odd
# mixed batch and a full prefill chunk, over the GQA ratios the backend admits
# (TP1 to TP8 of a 16-query-head, 2-KV-head model, and a single KV head).
WORKLOADS = pytest.mark.parametrize(
    ("tokens", "num_heads", "num_kv_heads", "head_size"),
    [
        pytest.param(1, 16, 2, 256, id="decode-tp1"),
        pytest.param(7, 8, 1, 256, id="mixed-tp2"),
        pytest.param(129, 4, 1, 256, id="mixed-tp4"),
        pytest.param(2048, 16, 1, 256, id="chunk-ratio16"),
    ],
)


def _make_tensors(tokens, num_heads, num_kv_heads, head_size):
    num_seqs = max(1, tokens // 4)
    return (
        torch.randn(tokens, num_heads, head_size),
        torch.randn(tokens, num_kv_heads, head_size),
        torch.randn(tokens, num_kv_heads, head_size),
        torch.randn(num_seqs, num_kv_heads),
        torch.randn(num_seqs, num_kv_heads),
        torch.randn(num_seqs, num_kv_heads),
    )


class _RecordingAttentionImpl:
    def __init__(self, supports_prequantized_qkv_input: bool):
        self.supports_prequantized_qkv_input = supports_prequantized_qkv_input
        self.prequantized_calls = 0
        self.received_prequantized_qkv: PrequantizedQKV | None = None

    def forward_with_prequantized_qkv(
        self, *args, prequantized_qkv: PrequantizedQKV, **kwargs
    ):
        self.prequantized_calls += 1
        self.received_prequantized_qkv = prequantized_qkv


def _make_layer(impl, num_heads, num_kv_heads, head_size, cls=PrequantizedQKVAttention):
    layer = object.__new__(cls)
    torch.nn.Module.__init__(layer)
    layer.impl = impl
    layer.layer_name = "layer"
    layer.num_heads = num_heads
    layer.num_kv_heads = num_kv_heads
    layer.head_size = head_size
    layer.head_size_v = head_size
    layer.use_direct_call = True
    layer.query_quant = None
    layer.kv_sharing_target_layer_name = None
    layer.attn_backend = SimpleNamespace(forward_includes_kv_cache_update=True)
    return layer


def _patch_attention_context(monkeypatch, layer):
    monkeypatch.setattr(
        module,
        "get_attention_context",
        lambda layer_name: (object(), layer, torch.empty(0), None),
    )


@WORKLOADS
def test_forward_dispatches_prequantized_qkv_to_impl(
    monkeypatch, tokens, num_heads, num_kv_heads, head_size
):
    impl = _RecordingAttentionImpl(supports_prequantized_qkv_input=True)
    layer = _make_layer(impl, num_heads, num_kv_heads, head_size)
    _patch_attention_context(monkeypatch, layer)
    tensors = _make_tensors(tokens, num_heads, num_kv_heads, head_size)

    output = layer(*tensors[:3], prequantized_qkv=PrequantizedQKV(*tensors))

    assert output.shape == (tokens, num_heads * head_size)
    assert impl.prequantized_calls == 1
    assert impl.received_prequantized_qkv is not None
    assert all(
        actual.shape == expected.shape and actual.data_ptr() == expected.data_ptr()
        for actual, expected in zip(impl.received_prequantized_qkv, tensors)
    )


def test_forward_updates_kv_cache_before_prequantized_attention(monkeypatch):
    impl = _RecordingAttentionImpl(supports_prequantized_qkv_input=True)
    layer = _make_layer(impl, 4, 1, 8)
    layer.attn_backend = SimpleNamespace(forward_includes_kv_cache_update=False)
    _patch_attention_context(monkeypatch, layer)
    calls = []
    dep = torch.empty(0)

    def fake_update(key, value, layer_name):
        calls.append("update")
        return dep

    monkeypatch.setattr(attention_module, "unified_kv_cache_update", fake_update)
    original = module.unified_prequantized_attention_with_output

    def recording_op(*args, kv_cache_dummy_dep=None, **kwargs):
        calls.append(("attention", kv_cache_dummy_dep))

    monkeypatch.setattr(
        module, "unified_prequantized_attention_with_output", recording_op
    )
    tensors = _make_tensors(2, 4, 1, 8)
    layer(*tensors[:3], prequantized_qkv=PrequantizedQKV(*tensors))
    assert calls == ["update", ("attention", dep)]
    assert original is not recording_op

    # A layer sharing another layer's KV cache must not write to it.
    calls.clear()
    layer.kv_sharing_target_layer_name = "other"
    layer(*tensors[:3], prequantized_qkv=PrequantizedQKV(*tensors))
    assert calls == [("attention", None)]


def test_custom_op_schema():
    schema = str(
        torch.ops.vllm.unified_prequantized_attention_with_output.default._schema
    )

    for argument in (
        "prequantized_query",
        "prequantized_key",
        "prequantized_value",
        "prequantized_query_descale",
        "prequantized_key_descale",
        "prequantized_value_descale",
    ):
        assert f"Tensor {argument}" in schema
    assert re.search(r"Tensor\(a\d*!\) output", schema)


def test_forward_rejects_unsupported_backend():
    impl = _RecordingAttentionImpl(supports_prequantized_qkv_input=False)
    layer = _make_layer(impl, 4, 1, 8)
    tensors = _make_tensors(2, 4, 1, 8)

    assert not layer.supports_prequantized_qkv_input
    with pytest.raises(ValueError, match="does not support it"):
        layer(*tensors[:3], prequantized_qkv=PrequantizedQKV(*tensors))
    assert impl.prequantized_calls == 0


def test_forward_without_bundle_uses_standard_dispatch(monkeypatch):
    layer = _make_layer(_RecordingAttentionImpl(True), 4, 1, 8)
    seen = {}

    def fake_dispatch(self, query, key, value, output):
        seen["called"] = True

    monkeypatch.setattr(module.Attention, "_dispatch_attention", fake_dispatch)
    query, key, value, *_ = _make_tensors(2, 4, 1, 8)

    assert layer(query, key, value).shape == (2, 4 * 8)
    assert seen == {"called": True}


def test_plain_attention_rejects_prequantized_qkv():
    layer = _make_layer(_RecordingAttentionImpl(True), 4, 1, 8, cls=module.Attention)
    tensors = _make_tensors(2, 4, 1, 8)

    with pytest.raises(TypeError, match="prequantized_qkv"):
        layer(*tensors[:3], prequantized_qkv=PrequantizedQKV(*tensors))
