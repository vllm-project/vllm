# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.model_loader import get_model_loader, register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.utils import AutoWeightsLoader


@register_model_loader("custom_load_format")
class CustomModelLoader(BaseModelLoader):
    def __init__(self, load_config: LoadConfig) -> None:
        super().__init__(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        pass


def test_register_model_loader():
    load_config = LoadConfig(load_format="custom_load_format")
    assert isinstance(get_model_loader(load_config), CustomModelLoader)


def test_invalid_model_loader():
    with pytest.raises(ValueError):

        @register_model_loader("invalid_load_format")
        class InValidModelLoader:
            pass


def test_default_loader_rejects_zero_num_threads():
    # num_threads=0 used to fail late in ThreadPoolExecutor ("max_workers must be > 0").
    with pytest.raises(ValueError, match="num_threads"):
        DefaultModelLoader(
            LoadConfig(
                model_loader_extra_config={
                    "enable_multithread_load": True,
                    "num_threads": 0,
                }
            )
        )


def test_default_loader_rejects_multithread_with_non_lazy_strategy():
    # The multi-thread loader ignores safetensors_load_strategy; reject the
    # combination instead of silently dropping the requested strategy.
    with pytest.raises(ValueError, match="does not support"):
        DefaultModelLoader(
            LoadConfig(
                safetensors_load_strategy="torchao",
                model_loader_extra_config={"enable_multithread_load": True},
            )
        )


def test_default_loader_explicit_safetensors_does_not_misread_pt(tmp_path):
    # Explicit safetensors must not fall back to a .pt and open it as safetensors.
    (tmp_path / "model.pt").write_bytes(b"\x00\x00\x00\x00")
    loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
    with pytest.raises(RuntimeError, match="Cannot find any model weights"):
        loader._prepare_weights(
            str(tmp_path),
            None,
            None,
            fall_back_to_pt=True,
            allow_patterns_overrides=None,
        )


def test_default_loader_hf_still_falls_back_to_pt(tmp_path):
    # Control: load_format="hf" still picks up .pt weights via fallback.
    (tmp_path / "model.pt").write_bytes(b"\x00\x00\x00\x00")
    loader = DefaultModelLoader(LoadConfig(load_format="hf"))
    _, files, use_safetensors, _ = loader._prepare_weights(
        str(tmp_path),
        None,
        None,
        fall_back_to_pt=True,
        allow_patterns_overrides=None,
    )
    assert use_safetensors is False
    assert any(f.endswith("model.pt") for f in files)


class _QuantMethod:
    uses_meta_device = False

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        pass


def _layer(quant_method=None, **shapes) -> nn.Module:
    layer = nn.Module()
    for name, shape in shapes.items():
        param = nn.Parameter(torch.zeros(shape), requires_grad=False)
        layer.register_parameter(name, param)
    layer.quant_method = quant_method
    return layer


def _track(model: nn.Module, loaded: set[str], quantized: bool) -> str:
    loader = DefaultModelLoader(LoadConfig())
    try:
        loader.track_weights_loading(model, set(loaded), quantized=quantized)
    except ValueError as e:
        return str(e)
    return ""


def test_weights_track_reports_missing_quantized_weights():
    # Quantized checkpoints may only omit KV-cache quantization params, empty
    # placeholders, fbgemm_fp8's input_scale_ub and online-quant params.
    online = _QuantMethod()
    online.uses_meta_device = True
    model = nn.Module()
    model.proj = _layer(_QuantMethod(), weight=(4, 4), weight_scale_inv=(1, 1))
    model.unquantized = _layer(UnquantizedLinearMethod(), weight=(4, 4), bias=(4,))
    model.gptq = _layer(_QuantMethod(), qweight=(4, 4), qzeros=(0,))
    model.fbgemm = _layer(_QuantMethod(), weight=(4, 4), input_scale_ub=())
    model.attn = _layer(BaseKVCacheMethod(None), k_scale=(), k_zero_point=())
    model.online = _layer(online, weight=(4, 4))
    loaded = {"proj.weight", "unquantized.weight", "gptq.qweight", "fbgemm.weight"}
    error = _track(model, loaded, quantized=True)
    assert "proj.weight_scale_inv" in error and "unquantized.bias" in error
    assert not any(n in error for n in ("gptq", "fbgemm", "attn", "online"))


def test_weights_track_keeps_unquantized_models_unchanged():
    # The check is on by default here, so unquantized layers stay exempt.
    model = nn.Module()
    model.score = _layer(_QuantMethod(), weight=(2, 4))
    model.norm = _layer(weight=(4,))
    assert _track(model, {"norm.weight"}, quantized=False) == ""
    assert "norm.weight" in _track(model, set(), quantized=False)


def test_moe_runner_reports_registered_expert_names():
    # AutoWeightsLoader prefixes these names, so they must include routed_experts.
    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.w13_weight = nn.Parameter(torch.zeros(2, 8, 4), requires_grad=False)

        def load_weights(self, weights):
            for _ in weights:
                yield "w13_weight"

    class Runner(nn.Module):
        load_weights = MoERunner.load_weights

        def __init__(self):
            super().__init__()
            self.routed_experts = Experts()

    model = nn.Module()
    model.experts = Runner()
    weights = [("experts.w13_weight", torch.zeros(2, 8, 4))]
    loaded = AutoWeightsLoader(model).load_weights(weights)
    assert loaded == {"experts.routed_experts.w13_weight"}
