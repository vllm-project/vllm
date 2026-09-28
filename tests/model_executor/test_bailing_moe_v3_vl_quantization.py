# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from torch import nn

from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.models import bailing_moe_v3, bailing_moe_v3_vl
from vllm.transformers_utils.configs.bailing_moe_v3_vl import BailingMoeV3VLConfig

_LING_MXFP4_QUANT_CONFIG = {
    "quant_method": "fp8",
    "fmt": "e4m3",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128],
    "scale_fmt": "ue8m0",
    "modules_to_not_convert": ["model.visual", "linear_proj", "lm_head"],
    "routed_experts_quant_method": "mxfp4",
    "routed_experts_group_size": 32,
}


class _Stub(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


def _make_vl_config() -> BailingMoeV3VLConfig:
    return BailingMoeV3VLConfig(
        text_config={"num_hidden_layers": 6, "layer_group_size": 6},
        quantization_config=copy.deepcopy(_LING_MXFP4_QUANT_CONFIG),
    )


def _make_quant_config() -> Fp8Config:
    return Fp8Config.from_config(copy.deepcopy(_LING_MXFP4_QUANT_CONFIG))


def _set_current_hf_config(monkeypatch: pytest.MonkeyPatch, hf_config) -> None:
    current = (
        None
        if hf_config is None
        else SimpleNamespace(model_config=SimpleNamespace(hf_config=hf_config))
    )
    monkeypatch.setattr(
        bailing_moe_v3,
        "get_current_vllm_config_or_none",
        lambda: current,
        raising=False,
    )


def test_ling_quant_config_falls_back_to_top_level_hf_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_vl_config()
    quant_config = _make_quant_config()
    _set_current_hf_config(monkeypatch, config)

    bailing_moe_v3._configure_ling_fp8_quant_config(quant_config, config.text_config)

    assert quant_config.store_dtype == "mxfp4"
    assert quant_config.is_scale_e8m0 is True


def test_ling_quant_config_without_current_vllm_config_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_vl_config()
    quant_config = _make_quant_config()
    _set_current_hf_config(monkeypatch, None)

    bailing_moe_v3._configure_ling_fp8_quant_config(quant_config, config.text_config)

    assert quant_config.store_dtype is None
    assert getattr(quant_config, "is_scale_e8m0", False) is False


def test_ling_quant_config_prefers_sub_config_quantization_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_vl_config()
    config.text_config.quantization_config = {
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }
    quant_config = _make_quant_config()
    _set_current_hf_config(monkeypatch, config)

    bailing_moe_v3._configure_ling_fp8_quant_config(quant_config, config.text_config)

    assert quant_config.store_dtype is None
    assert quant_config.is_scale_e8m0 is False


def test_bailing_v3_vl_language_model_uses_ling_mxfp4_quant_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_vl_config()
    quant_config = _make_quant_config()
    _set_current_hf_config(monkeypatch, config)

    seen: dict[str, object] = {}

    class _TextModel(nn.Module):
        def __init__(self, *, vllm_config, prefix: str = "") -> None:
            super().__init__()
            seen["hf_config"] = vllm_config.model_config.hf_config
            seen["store_dtype"] = vllm_config.quant_config.store_dtype
            seen["is_scale_e8m0"] = getattr(
                vllm_config.quant_config, "is_scale_e8m0", False
            )

    monkeypatch.setattr(bailing_moe_v3, "BailingMoeV3Model", _TextModel)
    monkeypatch.setattr(
        bailing_moe_v3, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False)
    )
    monkeypatch.setattr(bailing_moe_v3_vl, "BailingMoeV3VisionTransformer", _Stub)
    monkeypatch.setattr(bailing_moe_v3_vl, "BailingMoeV3VLProjector", _Stub)
    vl_cls = bailing_moe_v3_vl.BailingMoeV3VLForConditionalGeneration
    monkeypatch.setattr(vl_cls, "_mark_tower_model", lambda *a, **k: nullcontext())
    monkeypatch.setattr(vl_cls, "_mark_language_model", lambda *a, **k: nullcontext())

    def make_vllm_config(hf_config):
        return SimpleNamespace(
            model_config=SimpleNamespace(hf_config=hf_config),
            quant_config=quant_config,
            with_hf_config=make_vllm_config,
        )

    vl_cls(vllm_config=make_vllm_config(config), prefix="")

    assert seen["hf_config"] is config.text_config
    assert seen["store_dtype"] == "mxfp4"
    assert seen["is_scale_e8m0"] is True
