# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.models.minimax_m3.nvidia.model as minimax_m3
from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config
from vllm.model_executor.model_loader.utils import configure_quant_config
from vllm.model_executor.models.interfaces import supports_lora
from vllm.model_executor.utils import get_packed_modules_mapping
from vllm.models.minimax_m3.nvidia.model import (
    MiniMaxM3Model,
    MiniMaxM3SparseForCausalLM,
    MiniMaxM3SparseForConditionalGeneration,
)

QKV = ["q_proj", "k_proj", "v_proj"]
INDEXER = ["index_q_proj", "index_k_proj"]
MODEL_CLASSES = pytest.mark.parametrize(
    "model_cls,layers",
    [
        (MiniMaxM3SparseForCausalLM, "model.layers"),
        (MiniMaxM3SparseForConditionalGeneration, "language_model.model.layers"),
    ],
    ids=["text", "vl"],
)


def _language_model(**kwargs):
    model = nn.Module()
    model.make_empty_intermediate_tensors = None
    return model


def _build(model_cls, monkeypatch, lora_config):
    monkeypatch.setattr(minimax_m3, "MiniMaxM3Model", _language_model)
    monkeypatch.setattr(minimax_m3, "MiniMaxVLVisionModel", _language_model)
    monkeypatch.setattr(minimax_m3, "init_vllm_registered_model", _language_model)
    monkeypatch.setattr(
        minimax_m3, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False)
    )
    monkeypatch.setattr(
        model_cls, "_mark_tower_model", lambda *a, **k: nullcontext(), raising=False
    )
    text = SimpleNamespace(vocab_size=8, hidden_size=8)
    model_config = SimpleNamespace(
        hf_text_config=text,
        hf_config=SimpleNamespace(text_config=text, vision_config={}),
        multimodal_config=SimpleNamespace(mm_encoder_tp_mode="weights"),
    )
    return model_cls(
        vllm_config=SimpleNamespace(
            model_config=model_config, quant_config=None, lora_config=lora_config
        )
    )


@MODEL_CLASSES
@pytest.mark.parametrize(
    "lora_config",
    [None, SimpleNamespace(fully_sharded_loras=False)],
    ids=["lora-off", "lora-on"],
)
def test_lora_packs_indexer_and_quant_checks_real_projections(
    default_vllm_config, monkeypatch, model_cls, layers, lora_config
):
    dense, sparse = f"{layers}.0.self_attn", f"{layers}.3.self_attn"
    quant_config = ModelOptMxFp8Config.from_config(
        {
            "quant_method": "mxfp8",
            "ignored_layers": [f"{dense}.{p}" for p in QKV]
            + [f"{sparse}.{p}" for p in QKV + INDEXER],
        }
    )
    configure_quant_config(quant_config, model_cls)
    model = _build(model_cls, monkeypatch, lora_config)

    assert supports_lora(model)
    assert get_packed_modules_mapping(model)["qkv_proj"] == QKV + INDEXER
    assert quant_config.is_layer_excluded(f"{dense}.qkv_proj")
    assert quant_config.is_layer_excluded(f"{sparse}.qkv_proj")
    assert not quant_config.is_layer_excluded(f"{layers}.1.self_attn.qkv_proj")


class _Recorder(nn.Module):
    def __init__(self, calls: list):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.weight.weight_loader = lambda param, weight, *args, **kwargs: (
            calls.append((param, args, kwargs))
        )


class _Experts(nn.Module):
    def __init__(self, calls: list):
        super().__init__()
        self.routed_experts = nn.Module()
        for name in ("w13_weight", "w2_weight"):
            param = nn.Parameter(torch.zeros(1), requires_grad=False)
            param.weight_loader = lambda param, weight, name, shard_id, expert_id: (
                calls.append((param, shard_id, expert_id))
            )
            self.routed_experts.register_parameter(name, param)


class _LoRAWrapped(nn.Module):
    def __init__(self, base_layer: nn.Module):
        super().__init__()
        self.base_layer = base_layer


class _TinyM3(nn.Module):
    get_expert_mapping = MiniMaxM3Model.get_expert_mapping
    load_weights = MiniMaxM3Model.load_weights

    def __init__(self, wrapped: bool):
        super().__init__()
        self.config = SimpleNamespace(num_local_experts=2)
        self.calls: list = []
        wrap = _LoRAWrapped if wrapped else (lambda module: module)
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.qkv_proj = wrap(_Recorder(self.calls))
        layer.self_attn.o_proj = wrap(_Recorder(self.calls))
        layer.block_sparse_moe = nn.Module()
        layer.block_sparse_moe.experts = wrap(_Experts(self.calls))
        self.layers = nn.ModuleList([layer])


@pytest.mark.parametrize("wrapped", [False, True], ids=["plain", "lora-wrapped"])
def test_load_weights_resolves_lora_wrapped_params(wrapped):
    model = _TinyM3(wrapped)
    weights = [
        ("layers.0.self_attn.q_proj.weight", torch.zeros(1)),
        ("layers.0.self_attn.index_k_proj.weight", torch.zeros(1)),
        ("layers.0.self_attn.o_proj.weight", torch.zeros(1)),
        ("layers.0.block_sparse_moe.experts.1.w1.weight", torch.zeros(1)),
        ("layers.0.block_sparse_moe.experts.0.w2.weight", torch.zeros(1)),
    ]

    loaded = model.load_weights(weights)

    params = dict(model.named_parameters())
    prefix = "base_layer." if wrapped else ""
    qkv = params[f"layers.0.self_attn.qkv_proj.{prefix}weight"]
    o_proj = params[f"layers.0.self_attn.o_proj.{prefix}weight"]
    experts = f"layers.0.block_sparse_moe.experts.{prefix}routed_experts."
    assert [(p is qkv, args) for p, args, _ in model.calls[:2]] == [
        (True, ("q",)),
        (True, ("index_k",)),
    ]
    assert model.calls[2][0] is o_proj
    assert model.calls[3:] == [
        (params[experts + "w13_weight"], "w1", 1),
        (params[experts + "w2_weight"], "w2", 0),
    ]
    assert loaded == {
        "layers.0.self_attn.qkv_proj.weight",
        "layers.0.self_attn.o_proj.weight",
        "layers.0.block_sparse_moe.experts.routed_experts.w13_weight",
        "layers.0.block_sparse_moe.experts.routed_experts.w2_weight",
    }
