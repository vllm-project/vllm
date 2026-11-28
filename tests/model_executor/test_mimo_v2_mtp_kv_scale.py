# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MiMoV2MTP.load_weights must consume FP8 KV cache scales.

MiMoV2MTP keeps a hand-written loader (Pro-format fused qkv chunking), so the
AutoWeightsLoader cache-scale mapping never reaches it and the model has to
apply quant_config.get_cache_scale_mapper() itself, like GlmOcrMTP and HYV3MTP.
The heavy submodules are stubbed with the checkpoint-side parameter names; the
loader and the mapper under test are the real ones.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
from transformers import PreTrainedConfig

from vllm.model_executor.layers.quantization.kv_cache import KVCacheScaleParameter
from vllm.model_executor.models import mimo_v2_mtp

K_SCALE = 12.5
V_SCALE = 0.03125
PREFIX = "model.mtp.layers.0"

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(scope="module")
def fp8_quant_config():
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    return Fp8Config.from_config(
        {"quant_method": "fp8", "activation_scheme": "dynamic"}
    )


class _StubAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_scale = KVCacheScaleParameter()
        self.v_scale = KVCacheScaleParameter()


class _StubSelfAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Module()
        self.qkv_proj.weight = nn.Parameter(torch.zeros(12, 8), requires_grad=False)
        self.attn = _StubAttention()


class _StubMTPLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _StubSelfAttn()


class _StubPredictor(nn.Module):
    """Checkpoint-side names of MiMoV2MultiTokenPredictor."""

    def __init__(self, *, vllm_config=None, prefix=""):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.mtp = nn.ModuleDict({"layers": nn.ModuleDict({"0": _StubMTPLayer()})})


class _StubLMHead(nn.Module):
    def __init__(self, vocab_size, hidden_size, prefix=None, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(vocab_size, hidden_size), requires_grad=False
        )


@pytest.fixture(autouse=True, scope="module")
def _single_tp():
    with (
        patch.object(mimo_v2_mtp, "get_tensor_model_parallel_rank", return_value=0),
        patch.object(
            mimo_v2_mtp, "get_tensor_model_parallel_world_size", return_value=1
        ),
    ):
        yield


def _make_model(quant_config):
    hf_config = Mock(spec=PreTrainedConfig)
    hf_config.vocab_size = 8
    hf_config.hidden_size = 4
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        quant_config=quant_config,
    )
    with (
        patch.object(mimo_v2_mtp, "MiMoV2MultiTokenPredictor", _StubPredictor),
        patch.object(mimo_v2_mtp, "ParallelLMHead", _StubLMHead),
    ):
        return mimo_v2_mtp.MiMoV2MTP(vllm_config=vllm_config, prefix="")


def _attn(model):
    return model.model.mtp["layers"]["0"].self_attn.attn


def _control_weights():
    return [
        ("model.embed_tokens.weight", torch.full((8, 4), 7.0)),
        (f"{PREFIX}.self_attn.qkv_proj.weight", torch.full((12, 8), 3.0)),
    ]


@pytest.mark.parametrize(
    "k_name,v_name",
    [
        # llm-compressor fused naming (Pro checkpoints)
        (
            f"{PREFIX}.self_attn.qkv_proj.k_scale",
            f"{PREFIX}.self_attn.qkv_proj.v_scale",
        ),
        # ModelOpt per-projection naming (Flash checkpoints)
        (
            f"{PREFIX}.self_attn.k_proj.k_scale",
            f"{PREFIX}.self_attn.v_proj.v_scale",
        ),
    ],
)
def test_load_weights_consumes_kv_scales(fp8_quant_config, k_name, v_name):
    model = _make_model(fp8_quant_config)
    loaded = model.load_weights(
        _control_weights()
        + [(k_name, torch.tensor(K_SCALE)), (v_name, torch.tensor(V_SCALE))]
    )

    attn = _attn(model)
    assert attn.k_scale.item() == K_SCALE
    assert attn.v_scale.item() == V_SCALE
    assert f"{PREFIX}.self_attn.attn.k_scale" in loaded
    assert f"{PREFIX}.self_attn.attn.v_scale" in loaded
    # control weights keep loading through the Pro-format and direct branches
    assert model.model.embed_tokens.weight.data.abs().sum() > 0
    assert model.model.mtp["layers"]["0"].self_attn.qkv_proj.weight.data.abs().sum() > 0


def test_load_weights_consumes_deprecated_kv_scale(fp8_quant_config):
    """The single kv_scale lands on k_scale; v_scale is duplicated from k later
    in process_weights_after_loading, as for every other model."""
    model = _make_model(fp8_quant_config)
    loaded = model.load_weights(
        _control_weights() + [(f"{PREFIX}.self_attn.kv_scale", torch.tensor(K_SCALE))]
    )

    attn = _attn(model)
    assert attn.k_scale.item() == K_SCALE
    assert attn.v_scale.item() == -1.0
    assert f"{PREFIX}.self_attn.attn.k_scale" in loaded


def test_load_weights_without_quant_config():
    """With quant_config=None the mapper is skipped and loading is unchanged."""
    model = _make_model(None)
    loaded = model.load_weights(
        _control_weights()
        + [(f"{PREFIX}.self_attn.qkv_proj.k_scale", torch.tensor(K_SCALE))]
    )

    assert model.model.embed_tokens.weight.data.abs().sum() > 0
    assert f"{PREFIX}.self_attn.attn.k_scale" not in loaded
    assert _attn(model).k_scale.item() == -1.0
