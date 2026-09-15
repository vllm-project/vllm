# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shadow-format validation and INT4 config cloning."""

import logging
import os
from dataclasses import fields

import pytest

from vllm.model_executor.dual_precision.loader import (
    clone_init_dataclass,
    load_int4_shadow_model,
    make_int4_vllm_config,
    make_shadow_load_config,
    validate_shadow_quantization,
)

# The measurement host kept its cache at /data/huggingface/hub; other hosts set
# DUAL_PRECISION_HF_HUB. Tests skip on whatever is missing, so either works.
HF_HUB = os.environ.get("DUAL_PRECISION_HF_HUB", "/data/huggingface/hub")
QWEN35_9B_BF16 = (
    f"{HF_HUB}/models--Qwen--Qwen3.5-9B/snapshots/"
    "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
QWEN35_9B_AUTOROUND = (
    f"{HF_HUB}/models--Intel--Qwen3.5-9B-int4-AutoRound/snapshots/"
    "29688b8959bebb6d019ddd8f174a5b4bfd670456"
)
QWEN35_9B_NVFP4 = (
    f"{HF_HUB}/models--AxionML--Qwen3.5-9B-NVFP4/snapshots/"
    "97aef92393f126bf649f310cd40861be8dad3279"
)

AUTOROUND_GPTQ = {
    "quant_method": "auto-round",
    "bits": 4,
    "group_size": 128,
    "sym": True,
    "packing_format": "auto_round:auto_gptq",
}
CT_PACK_QUANTIZED = {
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {"group_0": {"format": "pack-quantized", "targets": ["Linear"]}},
}
MODELOPT_NVFP4 = {
    "quant_method": "modelopt",
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {"num_bits": 4, "type": "float", "group_size": 16},
            "input_activations": {"num_bits": 4, "type": "float", "dynamic": False},
        }
    },
}
MODELOPT_FP8 = {
    "quant_method": "modelopt",
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {"num_bits": 8, "type": "float"},
        }
    },
}


@pytest.mark.parametrize(
    ("hf_quant_config", "resolved", "label"),
    [
        (AUTOROUND_GPTQ, "inc", "auto-round:auto_round:auto_gptq"),
        ({**AUTOROUND_GPTQ, "packing_format": None}, "inc", "auto-round:None"),
        (CT_PACK_QUANTIZED, "compressed-tensors", "compressed-tensors:pack-quantized"),
        ({"quant_method": "gptq", "bits": 4}, "gptq_marlin", "gptq:gptq_marlin"),
    ],
)
def test_accepts_gptq_packings(hf_quant_config, resolved, label):
    assert validate_shadow_quantization(hf_quant_config, resolved) == label


@pytest.mark.parametrize(
    ("hf_quant_config", "resolved"),
    [
        (MODELOPT_NVFP4, "modelopt_fp4"),
        # Older exports name the format only in the weight spec or quant_algo.
        (MODELOPT_NVFP4, None),
        ({"quant_method": "modelopt", "quant_algo": "NVFP4"}, None),
        ({"quant_method": "modelopt_fp4"}, "modelopt_fp4"),
    ],
)
def test_accepts_modelopt_nvfp4_shadow(hf_quant_config, resolved):
    assert validate_shadow_quantization(hf_quant_config, resolved) == "modelopt:nvfp4"


def test_rejects_modelopt_fp8_shadow():
    with pytest.raises(ValueError, match="NVFP4"):
        validate_shadow_quantization(MODELOPT_FP8, "modelopt")


@pytest.mark.parametrize(
    ("hf_quant_config", "resolved", "match"),
    [
        ({"quant_method": "awq", "bits": 4}, "awq_marlin", "AWQ"),
        ({"quant_method": "awq", "bits": 4}, None, "AWQ"),
        ({**AUTOROUND_GPTQ, "packing_format": "auto_round:auto_awq"}, "inc", "AWQ"),
        ({**AUTOROUND_GPTQ, "backend": "awq:marlin"}, "inc", "AWQ"),
        (
            {"quant_method": "compressed-tensors", "format": "float-quantized"},
            "compressed-tensors",
            "pack-quantized",
        ),
        ({"quant_method": "fp8"}, "fp8", "GPTQ-packed"),
        (None, None, "quantized checkpoint"),
        ({}, None, "quantized checkpoint"),
    ],
)
def test_rejects_non_gptq_shadows(hf_quant_config, resolved, match):
    with pytest.raises(ValueError, match=match):
        validate_shadow_quantization(hf_quant_config, resolved)


def test_clone_init_dataclass_keeps_every_init_field():
    from vllm.config import CompilationConfig

    original = CompilationConfig(
        cudagraph_num_of_warmups=3, cudagraph_capture_sizes=[8, 16]
    )
    original.static_forward_context["x"] = object()
    clone = clone_init_dataclass(original)

    assert clone is not original
    for field in fields(original):
        if field.init:
            assert getattr(clone, field.name) == getattr(original, field.name), (
                field.name
            )
    # Non-init state is fresh: the shadow model registers its own layers.
    assert clone.static_forward_context == {}


def test_shadow_load_config_is_auto_even_when_engine_loads_dummy(caplog):
    """verl's ``rollout.load_format: dummy`` must never reach the shadow: the
    INT4 checkpoint is real weights by definition, so its clone always says
    ``auto`` and the dummy engine is announced at WARNING."""
    from vllm.config.load import LoadConfig

    engine = LoadConfig(load_format="dummy", download_dir="/nonexistent/cache")
    with caplog.at_level(logging.WARNING, logger="vllm"):
        shadow = make_shadow_load_config(engine)
    assert shadow is not engine
    assert shadow.load_format == "auto"
    assert engine.load_format == "dummy"
    assert shadow.download_dir == engine.download_dir
    assert any(
        "shadow" in rec.message and "dummy" in rec.message and "auto" in rec.message
        for rec in caplog.records
        if rec.levelno == logging.WARNING
    ), caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="vllm"):
        plain = make_shadow_load_config(LoadConfig(load_format="auto"))
    assert plain.load_format == "auto"
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    # Every other explicit format is coerced too (the shadow is a checkpoint).
    assert (
        make_shadow_load_config(LoadConfig(load_format="safetensors")).load_format
        == "auto"
    )


def test_load_int4_shadow_model_refuses_dummy_loader():
    """Belt and braces below :func:`make_shadow_load_config`."""
    from types import SimpleNamespace

    from vllm.config.load import LoadConfig

    config = SimpleNamespace(
        load_config=LoadConfig(load_format="dummy"),
        model_config=SimpleNamespace(model="/nonexistent/int4"),
    )
    with pytest.raises(RuntimeError, match="dummy"):
        load_int4_shadow_model(config)


def _require(path: str) -> str:
    if not os.path.isdir(path):
        pytest.skip(f"checkpoint not available: {path}")
    return path


def test_make_int4_vllm_config_clone_fields():
    from vllm.config import ModelConfig, VllmConfig
    from vllm.config.lora import LoRAConfig

    bf16 = _require(QWEN35_9B_BF16)
    int4 = _require(QWEN35_9B_AUTOROUND)
    from vllm.config.load import LoadConfig

    model_config = ModelConfig(
        model=bf16, tokenizer=bf16, dtype="bfloat16", max_model_len=512, seed=0
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        lora_config=LoRAConfig(max_lora_rank=16),
        load_config=LoadConfig(load_format="dummy"),
    )
    vllm_config.compilation_config.static_forward_context["attn"] = object()

    int4_config = make_int4_vllm_config(vllm_config, int4)

    assert int4_config is not vllm_config
    assert int4_config.model_config.model == int4
    assert int4_config.model_config.hf_config_path == int4
    assert int4_config.model_config.model_weights == ""
    # Quantization auto-detected from the shadow checkpoint (AutoRound -> inc).
    assert int4_config.model_config.quantization == "inc"
    assert vllm_config.model_config.quantization is None
    for field in fields(model_config):
        if field.init and field.name not in {
            "model",
            "model_weights",
            "hf_config_path",
            "quantization",
        }:
            assert getattr(int4_config.model_config, field.name) == getattr(
                model_config, field.name
            ), field.name
    for field in fields(vllm_config):
        if field.init and field.name not in {
            "model_config",
            "compilation_config",
            "load_config",
            "quant_config",
            "instance_id",
        }:
            assert getattr(int4_config, field.name) == getattr(
                vllm_config, field.name
            ), field.name
    assert int4_config.lora_config is vllm_config.lora_config
    # The shadow never inherits the engine's dummy loader (integration defect 1).
    assert int4_config.load_config is not vllm_config.load_config
    assert int4_config.load_config.load_format == "auto"
    assert vllm_config.load_config.load_format == "dummy"
    # The shadow config resolves its own quant config; the BF16 one stays None.
    assert type(int4_config.quant_config).__name__ == "INCConfig"
    assert vllm_config.quant_config is None
    assert int4_config.compilation_config is not vllm_config.compilation_config
    assert int4_config.compilation_config.static_forward_context == {}
    assert int4_config.model_config.get_total_num_hidden_layers() == 32

    with pytest.raises(ValueError, match="VLLM_DUAL_PRECISION_INT4_MODEL"):
        make_int4_vllm_config(vllm_config, "")


def test_make_int4_vllm_config_accepts_an_nvfp4_shadow():
    """The gate passes an NVFP4 checkpoint and vLLM resolves it to modelopt_fp4.

    Config-level only: no weights are loaded and no GPU is touched.
    """
    from vllm.config import ModelConfig, VllmConfig
    from vllm.config.load import LoadConfig
    from vllm.config.lora import LoRAConfig

    bf16 = _require(QWEN35_9B_BF16)
    nvfp4 = _require(QWEN35_9B_NVFP4)

    model_config = ModelConfig(
        model=bf16, tokenizer=bf16, dtype="bfloat16", max_model_len=512, seed=0
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        lora_config=LoRAConfig(max_lora_rank=16),
        load_config=LoadConfig(load_format="dummy"),
    )

    shadow_config = make_int4_vllm_config(vllm_config, nvfp4)

    assert shadow_config.model_config.model == nvfp4
    assert shadow_config.model_config.quantization == "modelopt_fp4"
    # The shadow never inherits a dummy loader, whatever the engine does.
    assert str(shadow_config.load_config.load_format).lower() == "auto"
    assert vllm_config.model_config.quantization is None
