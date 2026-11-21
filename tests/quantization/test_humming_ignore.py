# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for humming's is_layer_skipped handling of compressed-tensors
regex ("re:") ignore entries.

Regression test: compressed-tensors checkpoints (e.g. Kimi-K2.6) list ignored
layers as regex patterns prefixed with "re:" (e.g. "re:vision_tower.*").
humming previously substring-matched these literals, so ignored layers were
never skipped and got (incorrectly) quantized -- which then diverged between
the real weight-load path (falls back to unquantized) and the dummy-load path
(stays quantized), breaking any consumer that byte-compares the two.
"""

import pytest
import torch

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.humming import (
    HummingConfig,
    HummingLayerQuantizationConfig,
    HummingLinearMethod,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

# The real ignore list from Kimi-K2.6's compressed-tensors quantization_config.
K26_IGNORE = [
    "re:.*self_attn.*",
    "re:.*shared_experts.*",
    r"re:.*mlp\.(gate|up|gate_up|down)_proj.*",
    "re:.*lm_head.*",
    "re:vision_tower.*",
    "re:mm_projector.*",
]


@pytest.mark.parametrize(
    "prefix,expected",
    [
        # ignored layers (must be skipped -> stay unquantized)
        ("vision_tower.encoder.blocks.0.mlp.fc0", True),
        ("vision_tower.encoder.blocks.16.wo", True),
        ("mm_projector.linear_1", True),
        ("language_model.model.layers.0.self_attn.o_proj", True),
        ("language_model.model.layers.0.mlp.gate_up_proj", True),
        ("language_model.model.layers.0.mlp.down_proj", True),
        ("language_model.model.layers.3.mlp.shared_experts.gate_proj", True),
        ("model.lm_head", True),
        # routed experts are NOT ignored -> must be quantized
        ("language_model.model.layers.3.mlp.experts.383.up_proj", False),
        ("language_model.model.layers.3.mlp.experts.0.down_proj", False),
    ],
)
def test_is_layer_skipped_regex_ignore(prefix, expected):
    cfg = HummingConfig(full_config={"ignore": K26_IGNORE})
    assert cfg.is_layer_skipped({"ignore": K26_IGNORE}, prefix) is expected


def test_plain_substring_entries_still_work():
    # bitsandbytes-style modules_to_not_convert use plain substrings.
    cfg = HummingConfig()
    cfg_dict = {"modules_to_not_convert": ["lm_head", "vision"]}
    assert cfg.is_layer_skipped(cfg_dict, "model.vision.encoder.fc") is True
    assert cfg.is_layer_skipped(cfg_dict, "model.layers.0.mlp.up_proj") is False


def test_unquantized_weight_falls_back_to_working_linear(monkeypatch):
    # A layer that the checkpoint left unquantized (e.g. AutoRound extra_config
    # entries) reaches the Humming weight loader with a float weight. The
    # method then turns itself into UnquantizedLinearMethod, which must be
    # fully initialized so that process_weights_after_loading and apply work.
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_size, in_size = 8, 16
    method = HummingLinearMethod(
        HummingLayerQuantizationConfig(weight_schema=None, input_schema=object())
    )
    layer = torch.nn.Module()
    layer.is_fallback = False
    layer.param_dtype = torch.float32
    layer.input_size_per_partition = in_size
    layer.output_partition_sizes_sum = out_size
    layer.extra_weight_attrs = {"weight_loader": default_weight_loader}
    layer.tp_size, layer.tp_rank = 1, 0
    placeholder = torch.nn.Parameter(torch.empty(0, device=device), requires_grad=False)
    placeholder.param_name = "weight"
    layer.weight = placeholder

    weight = torch.randn(out_size, in_size, device=device)
    method.prepare_weight_loader(layer, default_weight_loader)(placeholder, weight)

    assert layer.is_fallback
    assert isinstance(method, UnquantizedLinearMethod)
    assert hasattr(method, "_gemm_impl")
    torch.testing.assert_close(layer.weight.data, weight)

    method.process_weights_after_loading(layer)
    x = torch.randn(3, in_size, device=device)
    torch.testing.assert_close(
        method.apply(layer, x), torch.nn.functional.linear(x, weight)
    )
