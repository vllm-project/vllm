# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP slicing of expert LoRA factors when a backend pads the expert
intermediate dim: the adapter is sharded by the unpadded per-rank size."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.config.lora import LoRAConfig
from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA

NUM_EXPERTS = 3
HIDDEN = 5
RANK = 8
# Unpadded per-rank intermediate size.
UNPADDED = 6


def _make_layer(cls, *, tp_size: int, tp_rank: int, padded: int, base_model=None):
    layer = cls.__new__(cls)
    nn.Module.__init__(layer)
    layer.moe_config = SimpleNamespace(
        hidden_dim=HIDDEN,
        num_local_experts=NUM_EXPERTS,
        intermediate_size_per_partition=padded,
        intermediate_size_per_partition_unpadded=UNPADDED,
    )
    layer.tp_size = tp_size
    layer.tp_rank = tp_rank
    layer.device = torch.device("cpu")
    layer.fully_sharded = False
    layer.enable_moe_shared_loras = False
    layer._w13_slices = 1 if cls is FusedMoE3DWithLoRA else 2
    layer._base_model = base_model

    lora_config = LoRAConfig(max_lora_rank=RANK, max_loras=1, lora_dtype=torch.float32)
    layer._create_lora_a_weights(1, lora_config)
    layer._create_lora_b_weights(1, lora_config)
    layer.adapter_enabled = torch.zeros(2, dtype=torch.int)
    return layer


def _rand(*shape: int) -> torch.Tensor:
    # Nonzero everywhere, so a row taken from the wrong place cannot compare
    # equal by accident and the padded tail can be checked against zero.
    return torch.rand(*shape) + 1.0


# padded == UNPADDED is the unpadded case, where the result must not change.
@pytest.mark.parametrize("padded", [UNPADDED, 8], ids=["unpadded", "padded"])
@pytest.mark.parametrize("tp_size", [2, 4])
def test_per_expert_lora_tp_slice(tp_size: int, padded: int):
    full = UNPADDED * tp_size
    w1_a, w3_a = _rand(NUM_EXPERTS, RANK, HIDDEN), _rand(NUM_EXPERTS, RANK, HIDDEN)
    w1_b, w3_b = _rand(NUM_EXPERTS, full, RANK), _rand(NUM_EXPERTS, full, RANK)
    w2_a = _rand(NUM_EXPERTS, RANK, full)
    w2_b = _rand(NUM_EXPERTS, HIDDEN, RANK)

    for tp_rank in range(tp_size):
        layer = _make_layer(
            FusedMoEWithLoRA, tp_size=tp_size, tp_rank=tp_rank, padded=padded
        )
        layer.set_lora(0, [w1_a, w2_a, w3_a], [w1_b, w2_b, w3_b])

        rows = slice(tp_rank * UNPADDED, (tp_rank + 1) * UNPADDED)
        for stacked, full_b in zip(layer.w13_lora_b_stacked, (w1_b, w3_b)):
            assert stacked.shape == (1, NUM_EXPERTS, padded, RANK)
            torch.testing.assert_close(stacked[0, :, :UNPADDED], full_b[:, rows])
            assert not stacked[0, :, UNPADDED:].any()

        w2_a_stacked = layer.w2_lora_a_stacked[0]
        assert w2_a_stacked.shape == (1, NUM_EXPERTS, RANK, padded)
        torch.testing.assert_close(w2_a_stacked[0, :, :, :UNPADDED], w2_a[:, :, rows])
        assert not w2_a_stacked[0, :, :, UNPADDED:].any()

        # Factors that are not split along the intermediate dim are untouched.
        torch.testing.assert_close(layer.w13_lora_a_stacked[0][0], w1_a)
        torch.testing.assert_close(layer.w13_lora_a_stacked[1][0], w3_a)
        torch.testing.assert_close(layer.w2_lora_b_stacked[0][0], w2_b)


@pytest.mark.parametrize("padded", [UNPADDED, 8], ids=["unpadded", "padded"])
@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize(
    "base_model", ["GptOssForCausalLM", "OtherForCausalLM"], ids=["interleaved", "cat"]
)
def test_3d_lora_tp_slice(base_model: str, tp_size: int, padded: int):
    full = UNPADDED * tp_size
    gate_b, up_b = _rand(NUM_EXPERTS, full, RANK), _rand(NUM_EXPERTS, full, RANK)
    if base_model == "GptOssForCausalLM":
        # (gate_0, up_0, gate_1, up_1, ...)
        w13_b = torch.stack([gate_b, up_b], dim=2).flatten(1, 2)
    else:
        w13_b = torch.cat([gate_b, up_b], dim=1)
    w13_a = _rand(NUM_EXPERTS, RANK, HIDDEN)
    w2_a = _rand(NUM_EXPERTS, RANK, full)
    w2_b = _rand(NUM_EXPERTS, HIDDEN, RANK)

    for tp_rank in range(tp_size):
        layer = _make_layer(
            FusedMoE3DWithLoRA,
            tp_size=tp_size,
            tp_rank=tp_rank,
            padded=padded,
            base_model=base_model,
        )
        layer.set_lora(0, [w13_a, w2_a], [w13_b, w2_b])

        rows = slice(tp_rank * UNPADDED, (tp_rank + 1) * UNPADDED)
        stacked = layer.w13_lora_b_stacked[0]
        assert stacked.shape == (1, NUM_EXPERTS, 2 * padded, RANK)
        if base_model == "GptOssForCausalLM":
            # Pairs stay adjacent; the padding is at the end.
            gate, up = stacked[0, :, 0::2], stacked[0, :, 1::2]
        else:
            # Each half starts at the allocated per-rank size, like w13.
            gate, up = stacked[0, :, :padded], stacked[0, :, padded:]
        for got, full_b in ((gate, gate_b), (up, up_b)):
            torch.testing.assert_close(got[:, :UNPADDED], full_b[:, rows])
            assert not got[:, UNPADDED:].any()

        w2_a_stacked = layer.w2_lora_a_stacked[0]
        torch.testing.assert_close(w2_a_stacked[0, :, :, :UNPADDED], w2_a[:, :, rows])
        assert not w2_a_stacked[0, :, :, UNPADDED:].any()
