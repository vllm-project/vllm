from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from vllm.models.deepseek_v4 import training_deepep as route
from vllm.models.deepseek_v4 import training_metadata as metadata
from vllm.models.deepseek_v4 import training_moe as moe
from vllm.models.deepseek_v4.training_checkpoint_quantization import (
    BlockFP8CheckpointDequantAdapter,
)
from vllm.models.deepseek_v4.training_cp import c128_all_visible_topk


def _config(**overrides):
    values = dict(
        num_hidden_layers=2,
        compress_ratios=[1, 4],
        head_dim=512,
        index_head_dim=128,
        num_attention_heads=4,
        sliding_window=3,
        rms_norm_eps=1e-6,
        qk_rope_head_dim=64,
        index_topk=2,
        max_position_embeddings=128,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_prefill_builder_owns_request_local_pages(monkeypatch) -> None:
    gather = Mock()
    monkeypatch.setattr(
        metadata,
        "_symbol",
        lambda _module, name: gather if name == "dequantize_and_gather_k_cache" else None,
    )
    builder = metadata.DS4PrefillMetadataBuilder(
        _config(), layer_idx=0, device="cpu", cos_sin_cache=torch.empty(16, 64)
    )
    value = builder.build_prefill_batch([2, 3])
    assert value.positions.tolist() == [0, 1, 0, 1, 2]
    assert value.runtime_layout.block_table[:, 0].tolist() == [0, 1]
    assert value.slot_mapping.tolist() == [0, 1, 256, 257, 258]
    value.prepare_flash()
    assert gather.call_count == 1


def test_block_fp8_checkpoint_dequant_uses_release_scales() -> None:
    qweight = torch.ones((128, 128), dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[2.0]], dtype=torch.float32)
    output = BlockFP8CheckpointDequantAdapter()(qweight, scale)
    assert output.dtype == torch.bfloat16
    assert torch.equal(output, torch.full_like(output, 2))


def test_route_order_restores_primary_deepep_receive_order() -> None:
    received = torch.arange(64, dtype=torch.bfloat16).reshape(4, 16)
    ids = torch.tensor([[0], [1], [0], [1]])
    weights = torch.tensor([[0.1], [0.2], [0.3], [0.4]])
    output_index = torch.tensor([[0], [2], [1], [3]])
    rows = route._validate_and_order_route_preserving_outputs(
        torch.tensor([[10.0], [30.0], [20.0], [40.0]]),
        received,
        ids,
        weights,
        output_index,
        received.clone(),
        ids.flatten(),
        weights.flatten(),
        return_route_rows=True,
    )
    assert rows.tolist() == [0, 2, 1, 3]


def _reference(hidden, counts, limit, w13, w2, _pack=None):
    output, offset = [], 0
    for count, fc1, fc2 in zip(counts, w13, w2, strict=True):
        gate, up = F.linear(hidden[offset : offset + count], fc1).chunk(2, -1)
        output.append(F.linear(F.silu(gate) * up, fc2))
        offset += count
    return torch.cat(output)


def test_grouped_moe_visible_forward_keeps_bf16_master_backward(monkeypatch) -> None:
    monkeypatch.setattr(moe, "_vllm_grouped_forward", _reference)
    hidden = torch.randn(3, 4, requires_grad=True)
    w13 = tuple(torch.randn(6, 4, requires_grad=True) for _ in range(2))
    w2 = tuple(torch.randn(4, 3, requires_grad=True) for _ in range(2))
    counts = torch.tensor([2, 1], dtype=torch.int32)
    output = moe.VLLMGroupedMoEWithBF16Backward.apply(
        hidden,
        counts,
        torch.ones(3),
        0.0,
        None,
        lambda value, _probs, _limit: F.silu(value.chunk(2, -1)[0])
        * value.chunk(2, -1)[1],
        *w13,
        *w2,
    )
    grad = torch.randn_like(output)
    output.backward(grad)
    reference_inputs = [item.detach().requires_grad_(True) for item in (hidden, *w13, *w2)]
    expected = _reference(
        reference_inputs[0],
        (2, 1),
        0.0,
        tuple(reference_inputs[1:3]),
        tuple(reference_inputs[3:]),
    )
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad)
    for actual, target in zip(
        (hidden.grad, *(item.grad for item in w13 + w2)), expected_grads, strict=True
    ):
        torch.testing.assert_close(actual, target)


def test_c128_topk_is_causal_and_padded() -> None:
    result = c128_all_visible_topk(torch.tensor([127, 128, 255]), width=4, ratio=128)
    assert result.tolist() == [[0, -1, -1, -1], [0, -1, -1, -1], [0, 1, -1, -1]]
