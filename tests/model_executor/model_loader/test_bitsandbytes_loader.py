# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from typing import Any

import pytest
import torch

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.bitsandbytes_loader import (
    BitsAndBytesModelLoader,
)
from vllm.model_executor.model_loader.utils import ParamMapping
from vllm.model_executor.models.gemma4 import (
    duplicate_gemma4_k_eq_v_bnb_quant_states,
)


def _install_fake_bitsandbytes(monkeypatch):
    functional = types.ModuleType("bitsandbytes.functional")

    class FakeQuantState:
        shape = (2, 3)
        dtype = torch.float16

        @classmethod
        def from_dict(cls, quant_state, device):
            return cls()

    functional_any: Any = functional
    functional_any.QuantState = FakeQuantState

    bitsandbytes = types.ModuleType("bitsandbytes")
    bitsandbytes_any: Any = bitsandbytes
    bitsandbytes_any.__version__ = "99.0.0"
    bitsandbytes_any.functional = functional

    monkeypatch.setitem(sys.modules, "bitsandbytes", bitsandbytes)
    monkeypatch.setitem(sys.modules, "bitsandbytes.functional", functional)


def _bnb_4bit_entries(name: str, packed_weight: torch.Tensor):
    return [
        (f"{name}.absmax", f"{name}.absmax", torch.ones(1)),
        (f"{name}.quant_map", f"{name}.quant_map", torch.ones(16)),
        (
            f"{name}.quant_state.bitsandbytes__nf4",
            f"{name}.quant_state.bitsandbytes__nf4",
            torch.ones(1, dtype=torch.uint8),
        ),
        (name, name, packed_weight),
    ]


def _loader_for_entries(entries, param_dict):
    loader = BitsAndBytesModelLoader(LoadConfig(use_tqdm_on_load=False))
    loader.param_dict = param_dict
    loader.modules_mapping = ParamMapping({})
    loader._hf_weight_iter = lambda *args, **kwargs: iter(entries)
    return loader


def test_prequant_4bit_plain_param_fails_fast(monkeypatch):
    packed_weight = torch.arange(3, dtype=torch.uint8).reshape(3, 1)
    target_param = torch.nn.Parameter(torch.empty(2, 3, dtype=torch.float16))
    _install_fake_bitsandbytes(monkeypatch)

    loader = _loader_for_entries(
        _bnb_4bit_entries("plain.weight", packed_weight),
        {"plain.weight": target_param},
    )
    quant_state_dict: dict[str, Any] = {}

    with pytest.raises(ValueError, match="not initialized as a vLLM BNB"):
        list(loader._quantized_4bit_generator([], True, quant_state_dict))

    assert quant_state_dict == {}


def test_prequant_4bit_bnb_param_keeps_packed_weight(monkeypatch):
    packed_weight = torch.arange(3, dtype=torch.uint8).reshape(3, 1)
    target_param = torch.nn.Parameter(
        torch.empty_like(packed_weight), requires_grad=False
    )
    target_param.use_bitsandbytes_4bit = True
    target_param.pack_factor = 2
    _install_fake_bitsandbytes(monkeypatch)

    loader = _loader_for_entries(
        _bnb_4bit_entries("bnb.weight", packed_weight),
        {"bnb.weight": target_param},
    )
    quant_state_dict: dict[str, Any] = {}

    loaded = list(loader._quantized_4bit_generator([], True, quant_state_dict))

    assert len(loaded) == 1
    assert loaded[0] == ("bnb.weight", packed_weight)
    assert "bnb.weight" in quant_state_dict


def test_prequant_4bit_renames_only_last_packed_module_match(monkeypatch):
    packed_weight = torch.arange(3, dtype=torch.uint8).reshape(3, 1)
    target_param = torch.nn.Parameter(
        torch.empty_like(packed_weight), requires_grad=False
    )
    target_param.use_bitsandbytes_4bit = True
    target_param.pack_factor = 2
    _install_fake_bitsandbytes(monkeypatch)

    loader = _loader_for_entries(
        _bnb_4bit_entries("q_proj_adapter.layer.q_proj.weight", packed_weight),
        {"q_proj_adapter.layer.qkv_proj.weight": target_param},
    )
    loader.modules_mapping = ParamMapping({"qkv_proj": ["q_proj", "k_proj", "v_proj"]})
    quant_state_dict: dict[str, Any] = {}

    loaded = list(loader._quantized_4bit_generator([], True, quant_state_dict))

    assert len(loaded) == 1
    assert loaded[0][0] == "q_proj_adapter.layer.q_proj.weight"
    assert loaded[0][1] is packed_weight
    assert "q_proj_adapter.layer.q_proj.weight" in quant_state_dict


def test_prequant_4bit_packed_bnb_param_keeps_packed_weight(monkeypatch):
    packed_weight = torch.arange(3, dtype=torch.uint8).reshape(3, 1)
    target_param = torch.nn.Parameter(
        torch.empty_like(packed_weight), requires_grad=False
    )
    target_param.use_bitsandbytes_4bit = True
    target_param.pack_factor = 2
    _install_fake_bitsandbytes(monkeypatch)

    loader = _loader_for_entries(
        _bnb_4bit_entries("layer.q_proj.weight", packed_weight),
        {"layer.qkv_proj.weight": target_param},
    )
    loader.modules_mapping = ParamMapping({"qkv_proj": ["q_proj", "k_proj", "v_proj"]})
    quant_state_dict: dict[str, Any] = {}

    loaded = list(loader._quantized_4bit_generator([], True, quant_state_dict))

    assert len(loaded) == 1
    assert loaded[0] == ("layer.q_proj.weight", packed_weight)
    assert "layer.q_proj.weight" in quant_state_dict


def test_gemma4_bnb_quant_state_hook_duplicates_k_eq_v_state():
    target_param = torch.nn.Parameter(
        torch.empty(4, 1, dtype=torch.uint8), requires_grad=False
    )
    target_param.use_bitsandbytes_4bit = True
    target_param.pack_factor = 2

    loader = _loader_for_entries(
        [],
        {"language_model.model.layers.5.self_attn.qkv_proj.weight": target_param},
    )
    loader.modules_mapping = ParamMapping({"qkv_proj": ["q_proj", "k_proj", "v_proj"]})

    q_state = object()
    k_state = object()
    quant_state_dict = {
        "language_model.model.layers.5.self_attn.q_proj.weight": q_state,
        "language_model.model.layers.5.self_attn.k_proj.weight": k_state,
    }
    model = torch.nn.Module()
    model.config = types.SimpleNamespace(
        text_config=types.SimpleNamespace(
            attention_k_eq_v=True,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
        )
    )

    duplicate_gemma4_k_eq_v_bnb_quant_states(
        model.config,
        quant_state_dict,
        loader._resolve_target_param_name,
    )
    stacked = loader._stack_quantization_states(model, quant_state_dict)

    assert stacked["language_model.model.layers.5.self_attn.qkv_proj.weight"] == {
        0: q_state,
        1: k_state,
        2: k_state,
    }


def test_gemma4_bnb_quant_state_hook_does_not_duplicate_non_k_eq_v_state():
    target_param = torch.nn.Parameter(
        torch.empty(4, 1, dtype=torch.uint8), requires_grad=False
    )
    target_param.use_bitsandbytes_4bit = True
    target_param.pack_factor = 2

    loader = _loader_for_entries(
        [],
        {"language_model.model.layers.5.self_attn.qkv_proj.weight": target_param},
    )
    loader.modules_mapping = ParamMapping({"qkv_proj": ["q_proj", "k_proj", "v_proj"]})

    q_state = object()
    k_state = object()
    quant_state_dict = {
        "language_model.model.layers.5.self_attn.q_proj.weight": q_state,
        "language_model.model.layers.5.self_attn.k_proj.weight": k_state,
    }
    model = torch.nn.Module()
    model.config = types.SimpleNamespace(
        text_config=types.SimpleNamespace(
            attention_k_eq_v=False,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
        )
    )

    duplicate_gemma4_k_eq_v_bnb_quant_states(
        model.config,
        quant_state_dict,
        loader._resolve_target_param_name,
    )
    stacked = loader._stack_quantization_states(model, quant_state_dict)

    assert stacked["language_model.model.layers.5.self_attn.qkv_proj.weight"] == {
        0: q_state,
        1: k_state,
    }
