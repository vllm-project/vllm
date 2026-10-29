# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash/DSpark context K/V projection with MXFP8 QKV weights."""

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    _mxfp8_e4m3_quantize_torch,
)
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100),
    reason="Native MXFP8 kernels require SM100-family CUDA",
)

HIDDEN, HEAD_DIM, NUM_HEADS, NUM_LAYERS, NUM_TOKENS = 256, 128, 2, 2, 7
Q_SIZE, KV_SIZE = NUM_HEADS * HEAD_DIM, HEAD_DIM  # one KV head


@pytest.fixture
def bf16_on_cuda(default_vllm_config, dist_init):
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        yield
    torch.set_default_dtype(old_dtype)


def _build_model(mode: str, has_bias: bool) -> tuple[DFlashQwen3Model, list]:
    # MiniMax-style MXFP8 config; "mixed" leaves layer 1 in BF16.
    quant_config = (
        None
        if mode == "bf16"
        else ModelOptMxFp8Config.from_config(
            {
                "quant_method": "mxfp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [1, 32],
                "ignored_layers": (
                    ["layers.1.self_attn.qkv_proj"] if mode == "mixed" else []
                ),
            }
        )
    )
    model = DFlashQwen3Model.__new__(DFlashQwen3Model)
    nn.Module.__init__(model)
    model.hidden_norm = RMSNorm(HIDDEN)
    model._rms_norm_eps = 1e-6
    attentions = []
    for i in range(NUM_LAYERS):
        attn = nn.Module()
        attn.q_size, attn.kv_size = Q_SIZE, KV_SIZE
        attn.k_norm = RMSNorm(HEAD_DIM)
        attn.qkv_proj = QKVParallelLinear(
            HIDDEN,
            HEAD_DIM,
            NUM_HEADS,
            1,
            bias=has_bias,
            quant_config=quant_config,
            prefix=f"layers.{i}.self_attn.qkv_proj",
            disable_tp=True,
        )
        weight = torch.randn(Q_SIZE + 2 * KV_SIZE, HIDDEN) / 16
        proj = attn.qkv_proj
        if isinstance(proj.quant_method, UnquantizedLinearMethod):
            proj.weight.data.copy_(weight)
        else:
            weight, scale = _mxfp8_e4m3_quantize_torch(weight)
            proj.weight.data.copy_(weight)
            proj.weight_scale.data.copy_(scale)
        if has_bias:
            proj.bias.data.normal_(std=0.1)
        attentions.append(attn)
    return model, attentions


@pytest.mark.parametrize("mode", ["bf16", "mxfp8", "mixed"])
@pytest.mark.parametrize("has_bias", [False, True])
@torch.inference_mode()
def test_context_kv_matches_qkv_projection(bf16_on_cuda, mode, has_bias):
    """The context K/V must equal the K/V slice of each layer's own QKV output
    after kernel repacking, eagerly and under CUDA graph replay."""
    torch.manual_seed(0)
    model, attentions = _build_model(mode, has_bias)
    # Load-time order: context buffers are built before kernels repack weights.
    model._build_context_kv_buffers(attentions, has_bias=has_bias)
    assert (model._fused_kv_weight is not None) == (mode == "bf16")
    assert (model._context_kv_proj is not None) == (mode == "mxfp8")
    for module in [*model.modules(), *(m for a in attentions for m in a.modules())]:
        method = getattr(module, "quant_method", None)
        if method is not None:
            method.process_weights_after_loading(module)

    x = torch.randn(NUM_TOKENS, HIDDEN)
    normed = model.hidden_norm(x)
    expected = torch.cat(
        [a.qkv_proj(normed)[0][:, Q_SIZE:] for a in attentions], dim=-1
    )

    def project() -> torch.Tensor:
        k, v = model._project_context_kv(x, NUM_TOKENS, NUM_LAYERS, 1, HEAD_DIM)
        return torch.stack([k, v]).permute(2, 1, 0, 3, 4).reshape(NUM_TOKENS, -1)

    torch.testing.assert_close(project(), expected, rtol=0.02, atol=0.02)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = project()
    x.normal_()
    graph.replay()
    torch.testing.assert_close(captured, project(), rtol=0, atol=0)
