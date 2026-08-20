# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention import attention as attention_module
from vllm.v1.attention.backend import PrequantizedQKV


def _make_tensors():
    query = torch.randn(2, 4, 8)
    key = torch.randn(2, 1, 8)
    value = torch.randn(2, 1, 8)
    query_descale = torch.randn(2, 1)
    key_descale = torch.randn(2, 1)
    value_descale = torch.randn(2, 1)
    return (
        query,
        key,
        value,
        query_descale,
        key_descale,
        value_descale,
    )


def test_make_prequantized_qkv_requires_all_tensors():
    assert (
        attention_module._make_prequantized_qkv(None, None, None, None, None, None)
        is None
    )

    tensors = _make_tensors()
    with pytest.raises(ValueError, match="requires query, key, value"):
        attention_module._make_prequantized_qkv(
            tensors[0],
            tensors[1],
            tensors[2],
            tensors[3],
            tensors[4],
            None,
        )


def test_make_prequantized_qkv_preserves_tensor_identity():
    tensors = _make_tensors()
    result = attention_module._make_prequantized_qkv(*tensors)

    assert isinstance(result, PrequantizedQKV)
    assert result is not None
    assert all(actual is expected for actual, expected in zip(result, tensors))


class _RecordingAttentionImpl:
    def __init__(self, supports_prequantized_qkv_input: bool):
        self.supports_prequantized_qkv_input = supports_prequantized_qkv_input
        self.standard_calls = 0
        self.prequantized_calls = 0
        self.received_prequantized_qkv: PrequantizedQKV | None = None

    def forward(self, *args, **kwargs):
        self.standard_calls += 1

    def forward_with_prequantized_qkv(
        self, *args, prequantized_qkv: PrequantizedQKV, **kwargs
    ):
        self.prequantized_calls += 1
        self.received_prequantized_qkv = prequantized_qkv


def _patch_attention_context(monkeypatch, impl):
    layer = SimpleNamespace(impl=impl)
    monkeypatch.setattr(
        attention_module,
        "get_attention_context",
        lambda layer_name: (object(), layer, torch.empty(0), None),
    )


def test_unified_attention_dispatches_prequantized_qkv(monkeypatch):
    impl = _RecordingAttentionImpl(supports_prequantized_qkv_input=True)
    _patch_attention_context(monkeypatch, impl)
    tensors = _make_tensors()

    attention_module.unified_attention_with_output(
        tensors[0],
        tensors[1],
        tensors[2],
        torch.empty_like(tensors[0]),
        "layer",
        prequantized_query=tensors[0],
        prequantized_key=tensors[1],
        prequantized_value=tensors[2],
        prequantized_query_descale=tensors[3],
        prequantized_key_descale=tensors[4],
        prequantized_value_descale=tensors[5],
    )

    assert impl.standard_calls == 0
    assert impl.prequantized_calls == 1
    assert impl.received_prequantized_qkv is not None
    assert all(
        actual is expected
        for actual, expected in zip(impl.received_prequantized_qkv, tensors)
    )


def test_unified_attention_custom_op_schema_has_prequantized_qkv():
    schema = str(torch.ops.vllm.unified_attention_with_output.default._schema)

    for argument in (
        "prequantized_query",
        "prequantized_key",
        "prequantized_value",
        "prequantized_query_descale",
        "prequantized_key_descale",
        "prequantized_value_descale",
    ):
        assert f"Tensor? {argument}=None" in schema


def test_unified_attention_custom_op_meta_accepts_prequantized_qkv():
    tensors = tuple(tensor.to(device="meta") for tensor in _make_tensors())

    torch.ops.vllm.unified_attention_with_output(
        tensors[0],
        tensors[1],
        tensors[2],
        torch.empty_like(tensors[0]),
        "layer",
        prequantized_query=tensors[0],
        prequantized_key=tensors[1],
        prequantized_value=tensors[2],
        prequantized_query_descale=tensors[3],
        prequantized_key_descale=tensors[4],
        prequantized_value_descale=tensors[5],
    )


def test_unified_attention_rejects_unsupported_prequantized_qkv(monkeypatch):
    impl = _RecordingAttentionImpl(supports_prequantized_qkv_input=False)
    _patch_attention_context(monkeypatch, impl)
    tensors = _make_tensors()

    with pytest.raises(ValueError, match="does not support prequantized QKV"):
        attention_module.unified_attention_with_output(
            tensors[0],
            tensors[1],
            tensors[2],
            torch.empty_like(tensors[0]),
            "layer",
            prequantized_query=tensors[0],
            prequantized_key=tensors[1],
            prequantized_value=tensors[2],
            prequantized_query_descale=tensors[3],
            prequantized_key_descale=tensors[4],
            prequantized_value_descale=tensors[5],
        )

    assert impl.standard_calls == 0
    assert impl.prequantized_calls == 0


def test_unified_attention_standard_path_is_unchanged(monkeypatch):
    impl = _RecordingAttentionImpl(supports_prequantized_qkv_input=False)
    _patch_attention_context(monkeypatch, impl)
    query, key, value, *_ = _make_tensors()

    attention_module.unified_attention_with_output(
        query,
        key,
        value,
        torch.empty_like(query),
        "layer",
    )

    assert impl.standard_calls == 1
    assert impl.prequantized_calls == 0
