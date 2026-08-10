# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for DFlash fused-KV buffers with a quantized drafter.

The DFlash fused-KV path (``precompute_and_store_context_kv``) slices
``qkv_proj.weight`` and feeds it to a single ``F.linear``.  For a quantized
drafter this bypasses the quant method: an FP8 weight is stored as
``float8_e4m3fn``, so the GEMM raises a dtype mismatch, and other schemes
silently drop their scales.  ``_dequant_kv_weight`` dequantizes the KV rows to
the activation dtype (BF16) at buffer-build time.

See https://github.com/vllm-project/vllm/issues/51581.
"""

import json
from types import SimpleNamespace

import pytest
import torch

from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

HIDDEN = 256
HEADS = 8
KV_HEADS = 4
HEAD_DIM = 32
Q_SIZE = HEADS * HEAD_DIM  # 256
KV_SIZE = KV_HEADS * HEAD_DIM  # 128


@pytest.fixture
def fp8_vllm_config(tmp_path):
    """VllmConfig with a usable model_config dtype (the default fixture leaves
    model_config unset, which breaks Fp8LinearMethod construction)."""
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "vocab_size": 100,
        "hidden_size": HIDDEN,
        "intermediate_size": 512,
        "num_hidden_layers": 1,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "max_position_embeddings": 64,
    }))
    model_config = ModelConfig(
        model=str(tmp_path),
        tokenizer=str(tmp_path),
        dtype="bfloat16",
        seed=0,
        trust_remote_code=False,
    )
    config = VllmConfig(model_config=model_config)
    with set_current_vllm_config(config):
        yield config


def _make_fp8_qkv(scale_mode: str = "per_tensor"):
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
    )
    qkv = QKVParallelLinear(
        hidden_size=HIDDEN,
        head_size=HEAD_DIM,
        total_num_heads=HEADS,
        total_num_kv_heads=KV_HEADS,
        bias=False,
        quant_config=quant_config,
        prefix="test.qkv_proj",
    ).cuda()

    out = Q_SIZE + 2 * KV_SIZE
    w_bf16 = torch.randn(out, HIDDEN, dtype=torch.bfloat16, device="cuda")
    if scale_mode == "per_tensor":
        scale = w_bf16.abs().max() / 448.0
        qkv.weight.data.copy_((w_bf16 / scale).to(torch.float8_e4m3fn))
        # A fused QKV checkpoint carries one per-tensor scale per shard; all
        # entries are equal.
        qkv.weight_scale.data.copy_(scale.expand(3))
    elif scale_mode == "per_shard":
        scales = torch.tensor(
            [0.005, 0.02, 0.01], dtype=torch.float32, device="cuda")
        wq = (w_bf16[:Q_SIZE] / scales[0]).to(torch.float8_e4m3fn)
        wk = (w_bf16[Q_SIZE:Q_SIZE + KV_SIZE] / scales[1]).to(
            torch.float8_e4m3fn)
        wv = (w_bf16[Q_SIZE + KV_SIZE:] / scales[2]).to(torch.float8_e4m3fn)
        qkv.weight.data.copy_(torch.cat([wq, wk, wv], dim=0))
        qkv.weight_scale.data.copy_(scales)
    else:
        raise AssertionError(scale_mode)
    return qkv


def _dequant_kv_weight(attn):
    """Call the model's method (self is unused)."""
    return DFlashQwen3Model._dequant_kv_weight(None, attn)


@pytest.mark.parametrize("scale_mode", ["per_tensor", "per_shard"])
def test_dequant_kv_weight_fp8(dist_init, fp8_vllm_config, scale_mode):
    qkv = _make_fp8_qkv(scale_mode)
    attn = SimpleNamespace(qkv_proj=qkv, q_size=Q_SIZE, kv_size=KV_SIZE)

    kv = _dequant_kv_weight(attn)
    assert kv.dtype == torch.bfloat16
    assert kv.shape == (2 * KV_SIZE, HIDDEN)

    # FP8 quantization introduces a small rounding error, so compare with the
    # FP8 dequantized values rather than the original BF16 weights.
    qkv_weight_bf16 = qkv.weight[:, :].to(torch.bfloat16)
    scale = qkv.weight_scale.to(torch.bfloat16)
    if scale_mode == "per_tensor":
        ref_fp8 = qkv_weight_bf16[Q_SIZE:] * scale.max()
    else:
        per_row = torch.cat(
            [scale[1].expand(KV_SIZE), scale[2].expand(KV_SIZE)]).unsqueeze(1)
        ref_fp8 = qkv_weight_bf16[Q_SIZE:] * per_row
    torch.testing.assert_close(kv, ref_fp8, rtol=0, atol=0)


def test_dequant_kv_weight_unquantized(dist_init):
    qkv = QKVParallelLinear(
        hidden_size=HIDDEN,
        head_size=HEAD_DIM,
        total_num_heads=HEADS,
        total_num_kv_heads=KV_HEADS,
        bias=False,
        prefix="test.qkv_proj",
    ).cuda()
    qkv.weight.data.normal_()
    attn = SimpleNamespace(qkv_proj=qkv, q_size=Q_SIZE, kv_size=KV_SIZE)

    kv = _dequant_kv_weight(attn)
    assert kv.dtype == torch.bfloat16
    torch.testing.assert_close(kv, qkv.weight[Q_SIZE:], rtol=0, atol=0)


def test_dequant_kv_weight_fp8_post_process(dist_init, fp8_vllm_config):
    """The same helper must work when buffers are built after
    process_weights_after_loading (e.g. the lazy dummy-run path), where the
    FP8 weight is stored transposed with a single collapsed scale."""
    qkv = _make_fp8_qkv("per_tensor")
    qkv.quant_method.process_weights_after_loading(qkv)
    if qkv.weight.dtype != torch.float8_e4m3fn:
        # Some kernels (e.g. Marlin FP8 on A100-class GPUs) repack the weight
        # into a dense int32 format that is not a simple [in, out] transpose;
        # the dequant helper deliberately raises for those layouts.
        pytest.skip(
            f"FP8 weight was repacked into {qkv.weight.dtype} by the selected "
            "kernel; only the direct-fp8 layout is covered here")
    attn = SimpleNamespace(qkv_proj=qkv, q_size=Q_SIZE, kv_size=KV_SIZE)

    kv = _dequant_kv_weight(attn)
    assert kv.dtype == torch.bfloat16
    assert kv.shape == (2 * KV_SIZE, HIDDEN)

    # After processing the weight is [in, out] fp8 and the scale is scalar.
    ref = (qkv.weight.t()[Q_SIZE:].to(torch.bfloat16)
           * qkv.weight_scale.to(torch.bfloat16))
    torch.testing.assert_close(kv, ref, rtol=0, atol=0)
