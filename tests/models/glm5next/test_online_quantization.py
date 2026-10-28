# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash loads the KDA and MLA attention projections as BF16 in every
checkpoint, so those layers never take the checkpoint's quantization method.
Online quantization requested explicitly with `--quantization-config` must
still reach them, while `kv_b_proj` (absorbed into the W_UK / W_UV bmm
weights) stays unquantized and the `--quantization` shorthands keep them
BF16."""

import pytest
import torch
from torch import nn
from transformers import Glm5NextTextConfig

from vllm.config.model import ModelConfig
from vllm.config.quantization import QuantizationConfigArgs
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.online.mxfp8 import (
    Mxfp8OnlineLinearMethod,
)
from vllm.models.glm5next.common import attention as glm_attention
from vllm.models.glm5next.common import model as glm_model
from vllm.models.glm5next.nvidia.specialized_decode.runtime import unsupported
from vllm.utils.torch_utils import set_default_torch_dtype

# Every attention projection, including those that must stay unquantized. The
# pattern must not match the attention module itself (``self_attn.attn``).
ATTENTION_TARGETS = {
    "re:.*[.]self_attn[.](in_proj_qkvbfg_a|f_b_proj|g_b_proj|conv1d|o_proj"
    "|fused_qkv_a_proj|q_b_proj|kv_b_proj|indexer[.](wq_b|wk_weights_proj))": "mxfp8"
}
# (projections online quantization reaches, projections that stay unquantized)
PROJECTIONS = {
    "deepseek_sparse_attention": (
        {"fused_qkv_a_proj", "q_b_proj", "o_proj", "indexer.wq_b"},
        {"kv_b_proj", "indexer.wk_weights_proj"},
    ),
    "linear_attention": (
        {"in_proj_qkvbfg_a", "f_b_proj", "g_b_proj", "o_proj"},
        {"conv1d"},
    ),
}


def _online(**args) -> OnlineQuantizationConfig:
    return OnlineQuantizationConfig(QuantizationConfigArgs(**args))


def _fp8_checkpoint(online: OnlineQuantizationConfig | None = None) -> Fp8Config:
    """A block-FP8 checkpoint config, optionally composed with online quant."""
    config = Fp8Config(weight_block_size=[128, 128])
    config.online_quantization_config = online
    return config


@pytest.fixture
def build_decoder_layer(default_vllm_config, dist_init, monkeypatch):
    # The sparse MLA attention backend depends on the GPU; every projection is
    # built before it.
    monkeypatch.setattr(
        glm_attention, "MultiHeadLatentAttentionWrapper", lambda *a, **k: nn.Identity()
    )
    default_vllm_config.model_config = ModelConfig(dtype="bfloat16")
    default_vllm_config.attention_config.sparse_mla_prefill_workspace_factor = 5

    def build(layer_type, quant_config):
        default_vllm_config.quant_config = quant_config
        config = Glm5NextTextConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=1,
            first_k_dense_replace=1,
            layer_types=[layer_type],
            mhc=False,
            mla_use_nope=True,
            indexer_rope_interleave=False,
        )
        with set_default_torch_dtype(torch.bfloat16), torch.device("cuda"):
            return glm_model.Glm5NextDecoderLayer(
                default_vllm_config, config, layer_idx=0, prefix="model.layers.0"
            )

    return build


@pytest.mark.parametrize("layer_type", list(PROJECTIONS))
@pytest.mark.parametrize(
    ("quant_config", "reaches_attention"),
    [
        (lambda: _fp8_checkpoint(), False),
        (lambda: _fp8_checkpoint(_online(targets=ATTENTION_TARGETS)), True),
        (lambda: _fp8_checkpoint(_online(linear="mxfp8")), True),
        (lambda: _online(targets=ATTENTION_TARGETS), True),
        # `--quantization mxfp8` on a BF16 checkpoint keeps attention BF16.
        (lambda: _online(linear="mxfp8"), False),
    ],
    ids=[
        "checkpoint",
        "checkpoint+targets",
        "checkpoint+linear",
        "online-targets",
        "online-shorthand",
    ],
)
def test_attention_projections_take_only_online_quantization(
    build_decoder_layer, layer_type, quant_config, reaches_attention
):
    layer = build_decoder_layer(layer_type, quant_config())

    methods = {
        name: type(module.quant_method)
        for name, module in layer.self_attn.named_modules()
        if hasattr(module, "quant_method")
    }
    online, unquantized = PROJECTIONS[layer_type]
    if not reaches_attention:
        online, unquantized = set(), online | unquantized
    assert methods == {
        **dict.fromkeys(online, Mxfp8OnlineLinearMethod),
        **dict.fromkeys(unquantized, UnquantizedLinearMethod),
    }


@pytest.mark.parametrize(
    ("quant_config", "rejected"),
    [
        (lambda: _fp8_checkpoint(), False),
        (lambda: _fp8_checkpoint(_online(targets=ATTENTION_TARGETS)), True),
    ],
    ids=["checkpoint", "checkpoint+targets"],
)
def test_specialized_decode_rejects_online_quantized_attention(
    default_vllm_config, quant_config, rejected
):
    """The specialized decode reads the KDA projection weights directly as BF16."""
    default_vllm_config.model_config = ModelConfig(dtype="bfloat16")
    default_vllm_config.quant_config = quant_config()

    reasons = unsupported(default_vllm_config)

    assert any("online quantization" in reason for reason in reasons) is rejected
