# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed-expert dispatch of the DeepSeek-V4 quant configs for NVFP4 exports.

CPU only: the MoE method classes are stubbed, so no kernel is built.
"""

import importlib
from types import SimpleNamespace

import pytest

_IGNORE = ["*.attn.*", "*.ffn.shared_experts.*", "head", "mtp.*"]


def _config(monkeypatch, module_name: str, ckpt_quant_cfg: dict):
    import vllm.model_executor.layers.quantization.modelopt as modelopt

    qc = importlib.import_module(f"vllm.models.{module_name}.quant_config")
    hf_config = SimpleNamespace(
        expert_dtype="fp4",
        quantization_config={"moe_quant_algo": ckpt_quant_cfg.get("moe_quant_algo")},
    )
    model_config = SimpleNamespace(
        hf_config=hf_config, hf_text_config=SimpleNamespace(num_hidden_layers=40)
    )
    monkeypatch.setattr(
        qc,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=model_config),
    )
    built: dict = {}

    def fake_nvfp4_moe(**kwargs):
        built.update(kwargs)
        return "nvfp4"

    monkeypatch.setattr(modelopt, "ModelOptNvFp4FusedMoE", fake_nvfp4_moe)
    if hasattr(qc, "_mxfp4_moe_method"):
        monkeypatch.setattr(qc, "_mxfp4_moe_method", lambda moe_config: "mxfp4")
    else:
        monkeypatch.setattr(qc, "Mxfp4MoEMethod", lambda *args, **kwargs: "mxfp4")
    config = qc.DeepseekV4FP8Config.from_config(
        {"quant_method": "fp8", "activation_scheme": "dynamic", **ckpt_quant_cfg}
    )
    return config, built


def _experts():
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    layer = RoutedExperts.__new__(RoutedExperts)
    layer.moe_config = None
    return layer


@pytest.mark.parametrize("module_name", ["deepseek_v4", "deepseek_v41"])
@pytest.mark.parametrize(
    ("moe_quant_algo", "expected_method"),
    [
        ("NVFP4", "NVFP4"),
        ("W4A16_NVFP4", "W4A16_NVFP4"),
        ("w4a16_nvfp4", "W4A16_NVFP4"),
    ],
)
def test_nvfp4_experts_get_matching_quant_method(
    monkeypatch, module_name, moe_quant_algo, expected_method
):
    """Weight-only NVFP4 exports (moe_quant_algo W4A16_NVFP4, no calibrated
    input_scale) build their routed experts with a W4A16_NVFP4 sub-config, which
    selects Marlin. W4A4 NVFP4 exports keep the NVFP4 sub-config."""
    config, built = _config(
        monkeypatch, module_name, {"moe_quant_algo": moe_quant_algo}
    )
    assert config.get_quant_method(_experts(), "model.layers.1.ffn.experts") == "nvfp4"
    assert built["quant_config"].quant_method == expected_method


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        pytest.param("model.layers.1.ffn.experts", "nvfp4", id="target"),
        pytest.param("model.layers.40.ffn.experts", "mxfp4", id="ignored-mtp0"),
    ],
)
def test_w4a16_export_keeps_ignored_experts_mxfp4(monkeypatch, prefix, expected):
    """A W4A16_NVFP4 export honours its ignore list like an NVFP4 one: the
    bundled DSpark draft (`mtp.*`, built as `layers.{40 + k}`) stays MXFP4."""
    config, _ = _config(
        monkeypatch,
        "deepseek_v41",
        {"moe_quant_algo": "W4A16_NVFP4", "ignore": _IGNORE},
    )
    monkeypatch.setattr(config, "_get_nvfp4_config", lambda: None)
    assert config.get_quant_method(_experts(), prefix) == expected
