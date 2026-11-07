# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AMD Flint model initialization smoke test."""

import json

import pytest
import torch

import vllm.models.flint.amd.model as flint_amd_model
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.config.load import LoadConfig
from vllm.config.quantization import QuantizationConfigArgs
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
from vllm.model_executor.layers.quantization.online.mxfp8 import (
    Mxfp8OnlineLinearMethod,
    Mxfp8OnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Static
from vllm.model_executor.model_loader import get_model_loader
from vllm.models.flint.amd.model import (
    FlintForCausalLM,
    _flint_gguf_quant_config,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only test")


def _tiny_vllm_config(tmp_path, *, quant_config=None):
    config = {
        "architectures": ["FlintForCausalLM"],
        "model_type": "flint",
        "vocab_size": 128,
        "hidden_size": 128,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "max_position_embeddings": 128,
        "layer_types": ["sliding_attention", "full_attention"],
        "no_rope_layer_indices": [1],
        "sliding_window": 64,
        "moe_intermediate_size": 128,
        "shared_expert_intermediate_size": 128,
        "num_experts_per_tok": 2,
        "num_experts": 4,
        "torch_dtype": "bfloat16",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    model_config = ModelConfig(
        model=str(tmp_path),
        dtype="bfloat16",
        max_model_len=128,
        skip_tokenizer_init=True,
    )
    return VllmConfig(
        model_config=model_config,
        load_config=LoadConfig(load_format="dummy"),
        quant_config=quant_config,
    )


def test_flint_amd_tiny_model_initializes_with_dummy_weights(tmp_path, dist_init):
    vllm_config = _tiny_vllm_config(tmp_path)
    model_config = vllm_config.model_config
    model = get_model_loader(vllm_config.load_config).load_model(
        vllm_config, model_config
    )

    assert model.__class__.__module__ == "vllm.models.flint.amd.model"
    assert len(model.model.layers) == 2
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


@torch.inference_mode()
def test_flint_amd_tiny_model_runs_moe_and_logits(tmp_path, dist_init, workspace_init):
    vllm_config = _tiny_vllm_config(tmp_path)
    model = get_model_loader(vllm_config.load_config).load_model(
        vllm_config, vllm_config.model_config
    )
    num_tokens = 32
    hidden_states = torch.randn(
        num_tokens,
        model.config.hidden_size,
        device=next(model.parameters()).device,
        dtype=model.config.torch_dtype,
    )

    with set_forward_context(None, vllm_config, num_tokens=hidden_states.shape[0]):
        hidden_states = model.model.layers[0].mlp(hidden_states)
    logits = model.compute_logits(hidden_states)

    assert hidden_states.shape == (num_tokens, model.config.hidden_size)
    assert logits is not None
    assert logits.shape == (num_tokens, model.config.vocab_size)
    assert torch.isfinite(hidden_states).all()
    assert torch.isfinite(logits).all()


def test_flint_gguf_constructor_wires_mxfp8_and_exclusions(
    tmp_path, dist_init, monkeypatch
):
    monkeypatch.setattr(flint_amd_model, "find_gguf_file", lambda *args: "flint.gguf")
    target = OnlineQuantizationConfig(
        QuantizationConfigArgs(linear="mxfp8", moe="mxfp8")
    )
    vllm_config = _tiny_vllm_config(tmp_path, quant_config=target)

    with set_current_vllm_config(vllm_config):
        model = FlintForCausalLM(vllm_config=vllm_config)
    layer = model.model.layers[0]
    source = model.quant_config

    assert isinstance(source, ModelOptNvFp4Config)
    assert source.online_quantization_config is target
    assert source.exclude_modules == ["model.embed_tokens", "*.gate", "lm_head"]
    assert isinstance(layer.self_attn.o_proj.quant_method, Mxfp8OnlineLinearMethod)
    assert isinstance(
        layer.mlp.experts.routed_experts.quant_method, Mxfp8OnlineMoEMethod
    )
    assert isinstance(layer.mlp.gate.quant_method, UnquantizedLinearMethod)
    assert isinstance(model.lm_head.quant_method, UnquantizedLinearMethod)

    weights = list(
        model._prepare_gguf_weights(
            [
                ("lm_head.weight_scale_2", torch.tensor(4.0)),
                ("model.weight", torch.ones(1)),
            ]
        )
    )
    assert model.logits_processor.scale == 4.0
    assert len(weights) == 1
    assert weights[0][0] == "model.weight"
    torch.testing.assert_close(weights[0][1], torch.ones(1))


@pytest.mark.parametrize(
    "wrong_target",
    [
        OnlineQuantizationConfig(QuantizationConfigArgs(linear="fp8_per_tensor")),
        OnlineQuantizationConfig(
            QuantizationConfigArgs(linear="mxfp8", moe="fp8_per_tensor")
        ),
    ],
)
def test_flint_gguf_requires_mxfp8_online_target(wrong_target):
    with pytest.raises(ValueError, match="requires online MXFP8 conversion"):
        _flint_gguf_quant_config(wrong_target)


def test_flint_gguf_wraps_mxfp8_target():
    target = OnlineQuantizationConfig(
        QuantizationConfigArgs(linear="mxfp8", moe="mxfp8")
    )

    source = _flint_gguf_quant_config(target)
    assert isinstance(source, ModelOptNvFp4Config)
    assert source.online_quantization_config is target
    assert source.exclude_modules == ["model.embed_tokens", "*.gate", "lm_head"]
    assert target.args.linear is not None
    assert target.args.linear.weight == kMxfp8Static
    assert target.args.moe is not None
    assert target.args.moe.weight == kMxfp8Static
