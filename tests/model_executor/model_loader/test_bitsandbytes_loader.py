# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types

import torch

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.bitsandbytes_loader import (
    BitsAndBytesModelLoader,
)
from vllm.model_executor.model_loader.utils import ParamMapping


def _install_fake_bitsandbytes(monkeypatch, dequantized_weight: torch.Tensor):
    functional = types.ModuleType("bitsandbytes.functional")

    class FakeQuantState:
        shape = tuple(dequantized_weight.shape)
        dtype = dequantized_weight.dtype

        @classmethod
        def from_dict(cls, quant_state, device):
            return cls()

    def dequantize_4bit(weight, quant_state):
        return dequantized_weight.clone()

    functional.QuantState = FakeQuantState
    functional.dequantize_4bit = dequantize_4bit

    bitsandbytes = types.ModuleType("bitsandbytes")
    bitsandbytes.__version__ = "99.0.0"
    bitsandbytes.functional = functional

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


def test_prequant_4bit_plain_param_is_dequantized(monkeypatch):
    packed_weight = torch.arange(3, dtype=torch.uint8).reshape(3, 1)
    dequantized_weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    target_param = torch.nn.Parameter(torch.empty(2, 3, dtype=torch.float16))
    _install_fake_bitsandbytes(monkeypatch, dequantized_weight)

    loader = _loader_for_entries(
        _bnb_4bit_entries("plain.weight", packed_weight),
        {"plain.weight": target_param},
    )
    quant_state_dict = {}

    loaded = list(loader._quantized_4bit_generator([], True, quant_state_dict))

    assert len(loaded) == 1
    assert loaded[0][0] == "plain.weight"
    assert loaded[0][1].shape == target_param.shape
    assert loaded[0][1].dtype == target_param.dtype
    torch.testing.assert_close(
        loaded[0][1], dequantized_weight.to(dtype=target_param.dtype)
    )
    assert quant_state_dict == {}


def test_prequant_4bit_bnb_param_keeps_packed_weight(monkeypatch):
    packed_weight = torch.arange(3, dtype=torch.uint8).reshape(3, 1)
    dequantized_weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    target_param = torch.nn.Parameter(
        torch.empty_like(packed_weight), requires_grad=False
    )
    target_param.use_bitsandbytes_4bit = True
    target_param.pack_factor = 2
    _install_fake_bitsandbytes(monkeypatch, dequantized_weight)

    loader = _loader_for_entries(
        _bnb_4bit_entries("bnb.weight", packed_weight),
        {"bnb.weight": target_param},
    )
    quant_state_dict = {}

    loaded = list(loader._quantized_4bit_generator([], True, quant_state_dict))

    assert len(loaded) == 1
    assert loaded[0] == ("bnb.weight", packed_weight)
    assert "bnb.weight" in quant_state_dict


def test_prequant_4bit_packed_bnb_param_keeps_packed_weight(monkeypatch):
    packed_weight = torch.arange(3, dtype=torch.uint8).reshape(3, 1)
    dequantized_weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    target_param = torch.nn.Parameter(
        torch.empty_like(packed_weight), requires_grad=False
    )
    target_param.use_bitsandbytes_4bit = True
    target_param.pack_factor = 2
    _install_fake_bitsandbytes(monkeypatch, dequantized_weight)

    loader = _loader_for_entries(
        _bnb_4bit_entries("layer.q_proj.weight", packed_weight),
        {"layer.qkv_proj.weight": target_param},
    )
    loader.modules_mapping = ParamMapping({"qkv_proj": ["q_proj", "k_proj", "v_proj"]})
    quant_state_dict = {}

    loaded = list(loader._quantized_4bit_generator([], True, quant_state_dict))

    assert len(loaded) == 1
    assert loaded[0] == ("layer.q_proj.weight", packed_weight)
    assert "layer.q_proj.weight" in quant_state_dict
