# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KimiK3MergedQKVGateLinear with 128x128 block-FP8 weights.

The latents (q_a + kv_a) are not a multiple of the 128-row scale block, so the
gate must be moved onto a block boundary or half of each gate block is scaled
by a neighbouring block's scale. These tests load per-projection checkpoint
tensors the way the model loader does and compare against the dequantized
checkpoint.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.linear import (
    KimiK3MergedQKVGateLinear,
    _weight_block_rows,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype

BLOCK = 128

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="block-FP8 linear needs CUDA"
)

# (hidden, q_lora_rank, kv_lora_rank, qk_rope_head_dim, num_heads, v_head_dim)
SHAPES = [
    pytest.param((256, 128, 64, 16, 2, 128), id="small"),
    pytest.param((7168, 1536, 512, 64, 96, 128), id="kimi-k3"),
]


@pytest.fixture
def bf16_model(default_vllm_config):
    """The FP8 linear method and its kernel selection read the model config."""
    default_vllm_config.model_config = SimpleNamespace(
        dtype=torch.bfloat16, hf_text_config=SimpleNamespace(model_type="kimi_k3")
    )
    with set_default_torch_dtype(torch.bfloat16):
        yield default_vllm_config


def _fp8_checkpoint_tensor(rows: int, cols: int, gen: torch.Generator):
    """An fp8 weight and per-128x128-block scales that differ block to block,
    so applying a neighbour's scale is a large, detectable error."""
    w = torch.randn(rows, cols, generator=gen).clamp(-3, 3)
    w = (w * 100).to(current_platform.fp8_dtype())
    exps = torch.randint(-4, 5, (-(-rows // BLOCK), cols // BLOCK), generator=gen)
    scale = torch.pow(2.0, exps.float()) / 1024
    return w, scale


def _dequant(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    s = scale.repeat_interleave(BLOCK, 0)[: w.shape[0]]
    return w.float() * s.repeat_interleave(BLOCK, 1)


def _build(shape, quant_config):
    hidden, q_lora, kv_lora, rope, heads, v_head = shape
    return KimiK3MergedQKVGateLinear(
        hidden_size=hidden,
        q_lora_rank=q_lora,
        kv_lora_rank=kv_lora,
        qk_rope_head_dim=rope,
        total_num_heads=heads,
        v_head_dim=v_head,
        bias=False,
        quant_config=quant_config,
        prefix="model.layers.0.self_attn.fused_qkv_a_g_proj",
    ).cuda()


def _load(layer, checkpoint):
    for shard_id, (w, scale) in enumerate(checkpoint):
        for param, tensor in ((layer.weight, w), (layer.weight_scale_inv, scale)):
            param.weight_loader(param, tensor.cuda(), shard_id)


@requires_cuda
@pytest.mark.parametrize("shape", SHAPES)
@torch.inference_mode()
def test_block_fp8_gate_scales_aligned(dist_init, bf16_model, shape):
    hidden, q_lora, kv_lora, rope, heads, v_head = shape
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True, weight_block_size=[BLOCK, BLOCK]
    )
    layer = _build(shape, quant_config)
    latent = q_lora + kv_lora + rope
    assert layer.latent_rows == latent
    assert layer.gate_row_offset % BLOCK == 0
    assert layer.gate_row_offset - latent == layer.latent_padding < BLOCK

    gen = torch.Generator().manual_seed(0)
    rows = (q_lora, kv_lora + rope, heads * v_head)
    checkpoint = [_fp8_checkpoint_tensor(r, hidden, gen) for r in rows]
    _load(layer, checkpoint)

    # Every stored weight row must sit in the block holding its own scale.
    scale_per_row = layer.weight_scale_inv.repeat_interleave(BLOCK, 0).cpu()
    offsets = (0, q_lora, layer.gate_row_offset)
    for (w, scale), start in zip(checkpoint, offsets):
        n = w.shape[0]
        stored = layer.weight[start : start + n].cpu()
        assert torch.equal(stored.view(torch.uint8), w.view(torch.uint8))
        expected = scale.repeat_interleave(BLOCK, 0)[:n]
        assert torch.equal(scale_per_row[start : start + n], expected)
    padding = layer.weight[latent : layer.gate_row_offset].float()
    assert not padding.any()

    layer.quant_method.process_weights_after_loading(layer)

    x = torch.randn(16, hidden, generator=gen).to(torch.bfloat16).cuda()
    out, _ = layer(x)
    assert out.shape == (16, latent + heads * v_head)
    ref_w = torch.cat([_dequant(w, s) for w, s in checkpoint]).cuda()
    ref = x.float() @ ref_w.T
    # FP8 per-token-group activation quantization and BF16 output: a few
    # percent. A neighbour's scale is off by a factor 2^-8..2^8.
    for lo, hi in ((0, latent), (latent, out.shape[1])):
        err = (out[:, lo:hi].float() - ref[:, lo:hi]).norm() / ref[:, lo:hi].norm()
        assert err < 0.05, (lo, hi, err.item())


@requires_cuda
@torch.inference_mode()
def test_unquantized_layout_unchanged(dist_init, bf16_model):
    layer = _build(SHAPES[0].values[0], None)
    hidden, q_lora, kv_lora, rope, heads, v_head = SHAPES[0].values[0]
    assert layer.latent_padding == 0
    assert layer.gate_row_offset == layer.latent_rows == q_lora + kv_lora + rope
    assert layer.weight.shape[0] == layer.latent_rows + heads * v_head


@pytest.mark.parametrize(
    ("algo", "rows"), [("FP8_PB_WO", 128), ("NVFP4", 1), (None, 1)]
)
def test_block_rows_from_modelopt_mixed_precision(algo, rows):
    """A checkpoint quantizes q_a_proj / kv_a_proj_with_mqa / g_proj; the fused
    layer resolves their algo through the packed-module mapping."""
    layer = "model.layers.0.self_attn"
    shards = ["q_a_proj", "kv_a_proj_with_mqa", "g_proj"]
    quantized = {}
    if algo is not None:
        quantized = {
            f"{layer}.{name}": {
                "quant_algo": algo,
                "group_size": 128 if algo == "FP8_PB_WO" else 16,
            }
            for name in shards
        }
    quantized["model.layers.0.mlp.experts"] = {"quant_algo": "NVFP4", "group_size": 16}
    config = ModelOptMixedPrecisionConfig.from_config(
        {
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "kv_cache_quant_algo": None,
                "quantized_layers": quantized,
            }
        }
    )
    config.packed_modules_mapping = {"fused_qkv_a_g_proj": shards}
    assert _weight_block_rows(config, f"{layer}.fused_qkv_a_g_proj") == rows


def test_block_rows_from_fp8_config():
    blocked = Fp8Config(
        is_checkpoint_fp8_serialized=True, weight_block_size=[BLOCK, BLOCK]
    )
    assert _weight_block_rows(blocked, "x") == BLOCK
    assert _weight_block_rows(Fp8Config(is_checkpoint_fp8_serialized=True), "x") == 1
    assert _weight_block_rows(None, "x") == 1
