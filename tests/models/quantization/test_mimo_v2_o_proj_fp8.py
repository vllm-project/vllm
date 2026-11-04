# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MiMo-V2's opt-in online FP8 ``o_proj`` gate.

``VLLM_MIMO_OPROJ_FP8=1`` must swap the checkpoint's bf16 ``o_proj`` for an
online per-tensor FP8 quant config (dynamic activations), while the default
(off) must pass the model's quant config through untouched and MTP layers
must stay bf16 in both modes.
"""

import pytest

from vllm import envs
from vllm.config.model import ModelConfig
from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.quantization.online.base import (
    Fp8PerTensorOnlineLinearMethod,
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
)
from vllm.model_executor.models import mimo_v2

pytestmark = pytest.mark.cpu_test

PREFIX = "model.layers.0.self_attn"


def test_env_registration(monkeypatch):
    getter = envs.environment_variables["VLLM_MIMO_OPROJ_FP8"]
    monkeypatch.delenv("VLLM_MIMO_OPROJ_FP8", raising=False)
    assert getter() is False
    assert envs.VLLM_MIMO_OPROJ_FP8 is False
    monkeypatch.setenv("VLLM_MIMO_OPROJ_FP8", "1")
    assert getter() is True
    monkeypatch.setenv("VLLM_MIMO_OPROJ_FP8", "0")
    assert getter() is False


def _o_proj_quant_config(monkeypatch, prefix=PREFIX, model_quant_config=None):
    """Build MiMoV2Attention with layer deps stubbed; return o_proj's config."""
    captured = {}

    def recorder(key):
        def record(*args, **kwargs):
            captured[key] = kwargs
            return object()

        return record

    monkeypatch.setattr(mimo_v2, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mimo_v2, "QKVParallelLinear", recorder("qkv"))
    monkeypatch.setattr(mimo_v2, "RowParallelLinear", recorder("o_proj"))
    monkeypatch.setattr(mimo_v2, "Attention", recorder("attn"))
    monkeypatch.setattr(mimo_v2, "get_rope", lambda **kwargs: object())

    mimo_v2.MiMoV2Attention(
        hidden_size=64,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        prefix=prefix,
        quant_config=model_quant_config,
    )
    return captured["o_proj"]["quant_config"]


def test_gate_off_passes_quant_config_through(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_MIMO_OPROJ_FP8", False)
    sentinel = object()
    assert _o_proj_quant_config(monkeypatch, model_quant_config=sentinel) is sentinel
    assert _o_proj_quant_config(monkeypatch) is None


def test_gate_on_builds_online_fp8_config(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_MIMO_OPROJ_FP8", True)
    config = _o_proj_quant_config(monkeypatch)
    assert isinstance(config, OnlineQuantizationConfig)
    assert config.args.linear.weight == kFp8StaticTensorSym
    assert config.args.moe is None


def test_gate_on_keeps_mtp_layers_unquantized(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_MIMO_OPROJ_FP8", True)
    config = _o_proj_quant_config(monkeypatch, prefix="model.mtp.layers.0.self_attn")
    assert config is None


def test_online_config_builds_fp8_o_proj(default_vllm_config, dist_init):
    default_vllm_config.model_config = ModelConfig()
    config = OnlineQuantizationConfig(
        QuantizationConfigArgs(linear=QuantSpec(weight=kFp8StaticTensorSym))
    )
    layer = RowParallelLinear(
        64,
        64,
        bias=False,
        quant_config=config,
        prefix=f"{PREFIX}.o_proj",
    )
    assert isinstance(layer.quant_method, Fp8PerTensorOnlineLinearMethod)
    assert layer.weight.device.type == "meta"
