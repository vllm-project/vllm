# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash loads the KDA and MLA attention projections as BF16 in every
checkpoint, so those layers never take the checkpoint's quantization method.
Online quantization requested explicitly with ``--quantization-config`` must
still reach them, while ``kv_b_proj`` (absorbed into the W_UK / W_UV bmm
weights) stays unquantized and the ``--quantization`` shorthands keep them
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

    def build(layer_type, quant_config, layer_id=0):
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
                default_vllm_config,
                config,
                layer_idx=0,
                prefix=f"model.layers.{layer_id}",
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


def _fp8_block_weight(
    rows: int, cols: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A random block-FP8 weight, its scales and its BF16 dequantization."""
    weight = (torch.randn(rows, cols) * 0.1).to(torch.float8_e4m3fn)
    scale_inv = torch.rand(rows // 128, cols // 128) + 0.5
    blocks = weight.float().view(rows // 128, 128, cols // 128, 128)
    dequantized = (blocks * scale_inv[:, None, :, None]).view(rows, cols)
    return weight.cuda(), scale_inv.cuda(), dequantized.bfloat16().cuda()


@pytest.mark.parametrize(
    ("target", "checkpoint_shards"),
    [
        ("q_b_proj", {"q_b_proj": None}),
        ("fused_qkv_a_proj", {"q_a_proj": 0, "kv_a_proj_with_mqa": 1}),
    ],
)
def test_fp8_mla_weights_reach_online_target_as_bf16(
    build_decoder_layer, target, checkpoint_shards
):
    """FP8 checkpoints store the MLA projections as block FP8. The loader
    dequantizes them to BF16 for an online target, so the online-quantized
    result matches loading those BF16 weights directly."""
    quant_config = _fp8_checkpoint(_online(targets=ATTENTION_TARGETS))
    fp8_layer = build_decoder_layer("deepseek_sparse_attention", quant_config, 0)
    ref_layer = build_decoder_layer("deepseek_sparse_attention", quant_config, 1)
    fp8_module = getattr(fp8_layer.self_attn, target)
    ref_module = getattr(ref_layer.self_attn, target)
    # NoPE checkpoints omit the kv_a rope rows, which the loader zero-pads.
    kv_a_pad_size = fp8_layer.self_attn.qk_rope_head_dim

    params = {f"model.layers.0.{n}": p for n, p in fp8_layer.named_parameters()}
    buf: dict = {}
    loaded: set[str] = set()
    for name, shard_id in checkpoint_shards.items():
        if shard_id is None:
            rows = fp8_module.output_size
        elif shard_id == 0:
            rows = fp8_module.output_sizes[0]
        else:
            rows = fp8_module.output_sizes[1] - kv_a_pad_size
        weight, scale_inv, dequantized = _fp8_block_weight(rows, fp8_module.input_size)
        for suffix, tensor in (("weight", weight), ("weight_scale_inv", scale_inv)):
            assert glm_model._try_load_fp8_attn_proj(
                f"model.layers.0.self_attn.{name}.{suffix}",
                tensor,
                buf,
                params,
                loaded,
                kv_a_pad_size=kv_a_pad_size,
            )
        if shard_id == 1:
            padding = dequantized.new_zeros(kv_a_pad_size, dequantized.shape[1])
            dequantized = torch.cat([dequantized, padding])
        shard = () if shard_id is None else (shard_id,)
        ref_module.weight.weight_loader(ref_module.weight, dequantized, *shard)

    assert loaded == {f"model.layers.0.self_attn.{target}.weight"}
    got = dict(fp8_module.named_parameters())
    want = dict(ref_module.named_parameters())
    assert fp8_module.weight.dtype != torch.bfloat16
    assert got.keys() == want.keys()
    for name in got:
        assert torch.equal(got[name], want[name]), name
