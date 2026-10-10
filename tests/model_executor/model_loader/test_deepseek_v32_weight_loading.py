# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 indexer WK pair fusion across incremental ``load_weights`` calls.

Reordering weight loaders (e.g. ``runai_streamer``) yield tensors in
IO-completion order, so a layer's FP8 ``indexer.wk`` weight /
``weight_scale_inv`` pair can be split by ``lm_head.weight``:
``AutoWeightsLoader`` splits the checkpoint stream at every ``model`` <->
non-``model`` prefix transition, issuing two consecutive ``load_weights()``
calls to the inner model. The pending pair buffer must therefore survive
across calls, or one half of the pair is silently dropped and the WK shard
of the fused ``wk_weights_proj`` parameter keeps its initialization.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.platforms import current_platform

if current_platform.is_cuda():
    from vllm.models.deepseek_v32.nvidia import model as deepseek_v32_model
    from vllm.models.deepseek_v32.nvidia import mtp as deepseek_v32_mtp
else:
    # The nvidia module chain is not importable on other platforms; the
    # shared model-executor CI job also collects this file on ROCm mirrors.
    deepseek_v32_model = None
    deepseek_v32_mtp = None

pytestmark = [
    pytest.mark.cpu_test,
    pytest.mark.skipif(
        not current_platform.is_cuda(), reason="Tests the CUDA deepseek_v32 loader"
    ),
]

WK_ROWS, WP_ROWS, IN_FEATURES = 4, 6, 8

MODEL_WK = "model.layers.1.self_attn.indexer.wk.weight"
MODEL_SCALE = "model.layers.1.self_attn.indexer.wk.weight_scale_inv"
MODEL_WP = "model.layers.1.self_attn.indexer.weights_proj.weight"
MTP_WK = "model.layers.2.self_attn.indexer.wk.weight"
MTP_SCALE = "model.layers.2.self_attn.indexer.wk.weight_scale_inv"
MTP_WP = "model.layers.2.self_attn.indexer.weights_proj.weight"


def _fused_weight_loader(param, loaded_weight, shard_id):
    rows = loaded_weight.shape[0]
    if shard_id == 0:
        param.data[:rows] = loaded_weight
    else:
        param.data[WK_ROWS : WK_ROWS + rows] = loaded_weight


class _StubDSAWeights(deepseek_v32_model.DeepseekV32Model):
    """Real ``load_weights`` over a single fused ``wk_weights_proj`` param."""

    def __init__(self, prefix: str = "model.layers.1"):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(n_routed_experts=0, num_hidden_layers=2)
        self.num_redundant_experts = 0
        self.prefix = prefix
        self.weight = nn.Parameter(
            torch.zeros(WK_ROWS + WP_ROWS, IN_FEATURES, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.weight.weight_loader = _fused_weight_loader

    def named_parameters(self, *, remove_duplicate=True, **kwargs):
        yield f"{self.prefix}.self_attn.indexer.wk_weights_proj.weight", self.weight


class _StubMTPWeights(deepseek_v32_mtp.DeepseekV32MTP):
    def __init__(self):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(
            n_routed_experts=0, num_hidden_layers=2, num_nextn_predict_layers=1
        )
        self.model = SimpleNamespace(mtp_start_layer_idx=2, num_mtp_layers=1)
        self.weight = nn.Parameter(
            torch.zeros(WK_ROWS + WP_ROWS, IN_FEATURES, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.weight.weight_loader = _fused_weight_loader

    def named_parameters(self, *, remove_duplicate=True, **kwargs):
        yield (
            "model.layers.2.mtp_block.self_attn.indexer.wk_weights_proj.weight",
            self.weight,
        )


class _StubLMHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(WK_ROWS + WP_ROWS, IN_FEATURES, dtype=torch.bfloat16),
            requires_grad=False,
        )


class _StubCausalLM(nn.Module):
    """Causal-LM shell mirroring GlmMoeDsaForCausalLM's module layout."""

    def __init__(self):
        super().__init__()
        self.model = _StubDSAWeights(prefix="layers.1")
        self.lm_head = _StubLMHead()

    def load_weights(self, weights):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


def _wk_pair(scale_layout: str = "scalar"):
    """FP8 wk weight + scale with distinct values per quantization group,
    so a wrong scale broadcast produces a different expected tensor."""
    weight = (
        torch.arange(WK_ROWS * IN_FEATURES, dtype=torch.float32)
        .reshape(WK_ROWS, IN_FEATURES)
        .to(torch.float8_e4m3fn)
    )
    if scale_layout == "scalar":
        scale = torch.full((1, 1), 0.5, dtype=torch.float32)
    elif scale_layout == "block":
        # Same layout as the GLM-5.3 checkpoint: one scale per column block.
        scale = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    elif scale_layout == "per_channel":
        scale = torch.tensor([0.5, 0.25, 0.75, 0.125], dtype=torch.float32)
    else:
        raise ValueError(scale_layout)
    return weight, scale, _dequant_expected(weight, scale)


def _dequant_expected(weight, scale):
    """Dequantized bf16 values computed from first principles (value * scale
    broadcast over its quantization group), independently of the loader."""
    if scale.ndim == 1:
        broadcast = scale.view(-1, 1)
    elif scale.shape == (1, 1):
        broadcast = scale
    else:
        g0 = WK_ROWS // scale.shape[0]
        g1 = IN_FEATURES // scale.shape[1]
        broadcast = scale.repeat_interleave(g0, 0).repeat_interleave(g1, 1)
    return (weight.to(torch.float32) * broadcast).to(torch.bfloat16)


@pytest.mark.parametrize("scale_layout", ["scalar", "block", "per_channel"])
@pytest.mark.parametrize("split_half_first", ["weight", "scale"])
def test_model_fuses_fp8_wk_pair_split_across_load_weights_calls(
    split_half_first,
    scale_layout,
):
    weight, scale, expected = _wk_pair(scale_layout)
    wp = torch.full((WP_ROWS, IN_FEATURES), 2.0, dtype=torch.bfloat16)
    model = _StubDSAWeights()

    halves = {
        "weight": (MODEL_WK, weight),
        "scale": (MODEL_SCALE, scale),
    }
    first_half = halves.pop(split_half_first)
    second_half = halves.popitem()[1]

    model.load_weights(iter([(MODEL_WP, wp), first_half]))
    assert torch.equal(
        model.weight.data[:WK_ROWS],
        torch.zeros_like(expected),
    ), "WK shard must not be written before both halves of the pair arrived"

    loaded = model.load_weights(iter([second_half]))
    assert "model.layers.1.self_attn.indexer.wk_weights_proj.weight" in loaded
    torch.testing.assert_close(model.weight.data[:WK_ROWS], expected)
    torch.testing.assert_close(model.weight.data[WK_ROWS:], wp)


@pytest.mark.parametrize("scale_layout", ["scalar", "block", "per_channel"])
@pytest.mark.parametrize("split_half_first", ["weight", "scale"])
def test_mtp_fuses_fp8_wk_pair_split_across_load_weights_calls(
    split_half_first,
    scale_layout,
):
    weight, scale, expected = _wk_pair(scale_layout)
    wp = torch.full((WP_ROWS, IN_FEATURES), 2.0, dtype=torch.bfloat16)
    model = _StubMTPWeights()

    halves = {
        "weight": (MTP_WK, weight),
        "scale": (MTP_SCALE, scale),
    }
    first_half = halves.pop(split_half_first)
    second_half = halves.popitem()[1]

    model.load_weights(iter([(MTP_WP, wp), first_half]))
    assert torch.equal(
        model.weight.data[:WK_ROWS],
        torch.zeros_like(expected),
    ), "WK shard must not be written before both halves of the pair arrived"

    model.load_weights(iter([second_half]))
    torch.testing.assert_close(model.weight.data[:WK_ROWS], expected)
    torch.testing.assert_close(model.weight.data[WK_ROWS:], wp)


def test_auto_weights_loader_lm_head_split_does_not_drop_wk_pair():
    """lm_head.weight between the two halves must not silently drop them."""
    weight, scale, expected = _wk_pair("block")
    wp = torch.full((WP_ROWS, IN_FEATURES), 2.0, dtype=torch.bfloat16)
    lm_head = torch.full((WK_ROWS + WP_ROWS, IN_FEATURES), 3.0, dtype=torch.bfloat16)
    causal = _StubCausalLM()

    causal.load_weights(
        iter(
            [
                (MODEL_SCALE, scale),
                ("lm_head.weight", lm_head),
                (MODEL_WK, weight),
                (MODEL_WP, wp),
            ]
        )
    )
    torch.testing.assert_close(causal.model.weight.data[:WK_ROWS], expected)
    torch.testing.assert_close(causal.model.weight.data[WK_ROWS:], wp)
    torch.testing.assert_close(causal.lm_head.weight.data, lm_head)
