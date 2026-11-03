# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that GRT's config.json loads without a GRT-aware Transformers, and
verify INT8 / INT4 weight quantization on GRTForCausalLM."""

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch

import pytest
import torch
from torch import nn

from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.grt import GRTConfig

# matterwaveai/grt-30m's config.json, as Transformers' GRTConfig saves it.
_GRT_30M_CONFIG = {
    "angle_init_scale": 0.1,
    "architectures": ["GRTForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "coupling_init_mode": "diagonal",
    "coupling_init_scale": 0.01,
    "dtype": "float32",
    "eos_token_id": 151643,
    "ffn_n_rotations": 21,
    "hidden_size": 256,
    "initializer_range": 0.02,
    "intermediate_size": 1024,
    "learnable_residual_gates": True,
    "learnable_rope": True,
    "log_magnitude_init": 0.0,
    "max_position_embeddings": 2048,
    "model_type": "grt",
    "n_mixer_rotations": 1,
    "n_qkv_rotations": 11,
    "num_attention_heads": 4,
    "num_hidden_layers": 12,
    "output_scale_init": 30.0,
    "pad_token_id": 151643,
    "peri_ln": False,
    "phase_init_scale": 0.1,
    "pos_freq_base": 10000.0,
    "residual_gate_mode": "bounded_depth",
    "residual_norm_mode": "none",
    "tie_word_embeddings": True,
    "transformers_version": "5.21.0.dev0",
    "use_cache": True,
    "vocab_size": 151936,
}

# Sizes away from the defaults, so a value the config ignored would show.
_RESIZED = {
    "hidden_size": 512,
    "num_attention_heads": 8,
    "num_hidden_layers": 6,
    "max_position_embeddings": 4096,
    "intermediate_size": 1536,
    "vocab_size": 1000,
}


@pytest.mark.parametrize(
    "config_json",
    [_GRT_30M_CONFIG, {**_GRT_30M_CONFIG, **_RESIZED}],
    ids=["grt-30m", "resized"],
)
def test_config_loads(tmp_path, config_json):
    (tmp_path / "config.json").write_text(json.dumps(config_json))
    config = get_config(str(tmp_path), trust_remote_code=False)
    assert isinstance(config, GRTConfig)
    for key in (*_RESIZED, "n_qkv_rotations", "ffn_n_rotations", "n_mixer_rotations"):
        assert getattr(config, key) == config_json[key], key
    assert config.residual_gate_mode == "bounded_depth"
    assert config.residual_norm_mode == "none"
    assert not config.peri_ln
    assert config.learnable_residual_gates
    assert config.tie_word_embeddings


class _DummyAttention(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, scale: float, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        t = q.shape[0]
        qh = q.view(t, self.num_heads, self.head_dim).transpose(0, 1)
        kh = k.view(t, self.num_heads, self.head_dim).transpose(0, 1)
        vh = v.view(t, self.num_heads, self.head_dim).transpose(0, 1)
        scores = torch.matmul(qh, kh.transpose(-1, -2)) * self.scale
        mask = torch.triu(torch.full((t, t), float("-inf"), device=q.device), diagonal=1)
        probs = torch.softmax(scores + mask, dim=-1)
        out = torch.matmul(probs, vh).transpose(0, 1).contiguous()
        return out.view(t, self.num_heads * self.head_dim)


@dataclass
class _DummyModelConfig:
    hf_config: GRTConfig


@dataclass
class _DummyVllmConfig:
    model_config: _DummyModelConfig
    cache_config: object = None
    quant_config: object = None


def _weight_bytes(model: nn.Module) -> int:
    total = 0
    seen: set[int] = set()
    for t in (*model.parameters(), *model.buffers()):
        if t is not None and t.data_ptr() not in seen:
            seen.add(t.data_ptr())
            total += t.numel() * t.element_size()
    return total


def _load_grt_module():
    try:
        from vllm.model_executor.models import grt as grt_mod

        return grt_mod
    except ModuleNotFoundError:
        pkg_name = "vllm.model_executor.models"
        for parent in ("vllm.model_executor", pkg_name):
            if parent not in sys.modules:
                m = types.ModuleType(parent)
                m.__path__ = []
                sys.modules[parent] = m
        stubs = {
            "vllm.compilation.decorators": {"support_torch_compile": lambda cls: cls},
            "vllm.config": {"CacheConfig": object, "VllmConfig": object},
            "vllm.config.utils": {
                "getattr_iter": lambda obj, names, default=None, **kw: next(
                    (getattr(obj, n) for n in names if hasattr(obj, n)), default
                )
            },
            "vllm.distributed": {"get_tensor_model_parallel_world_size": lambda: 1},
            "vllm.model_executor.layers.attention": {"Attention": _DummyAttention},
            "vllm.model_executor.layers.quantization": {"QuantizationConfig": object},
            "vllm.sequence": {"IntermediateTensors": object},
            f"{pkg_name}.utils": {
                "AutoWeightsLoader": object,
                "maybe_prefix": lambda p, n: f"{p}.{n}" if p else n,
            },
        }
        for mod_name, attrs in stubs.items():
            if mod_name not in sys.modules:
                mod = types.ModuleType(mod_name)
                for k, v in attrs.items():
                    setattr(mod, k, v)
                sys.modules[mod_name] = mod
        grt_path = (
            Path(__file__).resolve().parents[2]
            / "vllm"
            / "model_executor"
            / "models"
            / "grt.py"
        )
        spec = importlib.util.spec_from_file_location(f"{pkg_name}.grt", grt_path)
        grt_mod = importlib.util.module_from_spec(spec)
        grt_mod.__package__ = pkg_name
        sys.modules[f"{pkg_name}.grt"] = grt_mod
        spec.loader.exec_module(grt_mod)
        return grt_mod


_GRT_MOD = _load_grt_module()


def _init_random_grt(
    config: GRTConfig, seed: int = 42, device: str | torch.device = "cpu"
):
    grt_mod = _GRT_MOD

    g = torch.Generator().manual_seed(seed)
    with (
        patch.object(grt_mod, "Attention", _DummyAttention),
        patch.object(grt_mod, "get_tensor_model_parallel_world_size", return_value=1),
    ):
        vllm_cfg = _DummyVllmConfig(model_config=_DummyModelConfig(hf_config=config))
        model = grt_mod.GRTForCausalLM(vllm_config=vllm_cfg)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "scales" in name or "norm" in name:
                p.copy_(torch.randn(p.shape, generator=g) * 0.1 + 0.5)
            elif "output_scale" in name:
                p.fill_(2.0)
            else:
                p.copy_(torch.randn(p.shape, generator=g) * 0.2)
    model = model.to(device)
    model.process_weights_after_loading()
    return model


@pytest.mark.parametrize("bits,atol", [(8, 0.15), (4, 0.6)])
def test_grt_weight_quantization(bits: int, atol: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = GRTConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        n_qkv_rotations=3,
        ffn_n_rotations=3,
        n_mixer_rotations=1,
    )
    ref_model = _init_random_grt(cfg, seed=7, device=device)
    q_model = _init_random_grt(cfg, seed=7, device=device)

    input_ids = torch.tensor([1, 5, 12, 33, 100], dtype=torch.long, device=device)
    positions = torch.arange(len(input_ids), dtype=torch.long, device=device)

    with torch.no_grad():
        ref_hidden = ref_model(input_ids, positions)
        ref_logits = ref_model.compute_logits(ref_hidden)
        fp32_bytes = _weight_bytes(ref_model)

        q_model.quantize_weights(bits=bits)
        q_bytes = _weight_bytes(q_model)
        q_hidden = q_model(input_ids, positions)
        q_logits = q_model.compute_logits(q_hidden)

    assert q_bytes < fp32_bytes
    # Angles and scales stay in fp32 so Givens rotations do not suffer phase drift
    for block in q_model.model.blocks:
        assert block.attention.qkv_angles.dtype == torch.float32
        assert block.attention.qkv_gains.dtype == torch.float32
        assert block.attention.o_angles.dtype == torch.float32
        assert block.attention.o_gains.dtype == torch.float32
        assert block.mixer.angles.dtype == torch.float32
        assert block.mixer.gains.dtype == torch.float32
        expected_dtype = torch.int8 if bits == 8 else torch.uint8
        assert block.attention.qkv_coupling.dtype == expected_dtype
        assert block.attention.o_coupling.dtype == expected_dtype
        assert block.mixer.coupling.dtype == expected_dtype
    assert q_model.model.embedding._embedding_table_q.dtype == (
        torch.int8 if bits == 8 else torch.uint8
    )
    assert q_model.head.dtype == (torch.int8 if bits == 8 else torch.uint8)
    torch.testing.assert_close(q_logits, ref_logits, atol=atol, rtol=0.1)

