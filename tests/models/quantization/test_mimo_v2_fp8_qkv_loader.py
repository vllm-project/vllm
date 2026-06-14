# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the MiMo-V2 fused-QKV FP8 loader's ckpt_tp detection.

The MiMo-V2.5 base checkpoint stores SWA layers' fused-QKV in 4 per-rank
chunks of [16Q | 2K | 2V] rows, but the SWA layer has num_kv_heads = 8.
Hardcoding ckpt_tp = num_kv_heads (the previous behavior) makes the
pre-sharded check reject the layout and the fallback slice the per-rank
shard at the wrong size, raising on shape mismatch. These tests pin the
data-driven detection so any regression resurfaces immediately.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.mimo_v2 import _mimo_v2_copy_paired_qkv_fp8

HIDDEN = 4096
BLOCK = [128, 128]


def _config() -> SimpleNamespace:
    # Mirrors XiaomiMiMo/MiMo-V2.5 base for fields the loader reads.
    return SimpleNamespace(
        head_dim=192,
        v_head_dim=128,
        num_attention_heads=64,
        num_key_value_heads=4,
        swa_head_dim=192,
        swa_v_head_dim=128,
        swa_num_attention_heads=64,
        swa_num_key_value_heads=8,
        hybrid_layer_pattern=[0, 1, 1, 1, 1, 0],
    )


def _build_chunked_qkv(
    n_chunks: int,
    q_heads_per_chunk: int,
    kv_heads_per_chunk: int,
    head_dim: int,
    v_head_dim: int,
    hidden: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    chunk_rows = (
        q_heads_per_chunk * head_dim
        + kv_heads_per_chunk * head_dim
        + kv_heads_per_chunk * v_head_dim
    )
    # Distinct per-chunk fingerprint so we can prove the right chunk landed.
    weight = torch.cat(
        [
            torch.full((chunk_rows, hidden), c, dtype=torch.float32)
            for c in range(n_chunks)
        ],
        dim=0,
    ).to(torch.float8_e4m3fn)
    scale_rows_per_chunk = -(-chunk_rows // BLOCK[0])
    scale = torch.ones(
        (n_chunks * scale_rows_per_chunk, hidden // BLOCK[1]),
        dtype=torch.float32,
    )
    return weight, scale, chunk_rows


@pytest.mark.parametrize("tp_rank", [0, 1, 2, 3])
def test_swa_layer_detects_ckpt_tp_from_scale_shape(tp_rank: int) -> None:
    config = _config()
    weight, scale, chunk_rows = _build_chunked_qkv(
        n_chunks=4,
        q_heads_per_chunk=16,
        kv_heads_per_chunk=2,
        head_dim=config.swa_head_dim,
        v_head_dim=config.swa_v_head_dim,
        hidden=HIDDEN,
    )
    assert weight.shape == (14848, HIDDEN)
    assert scale.shape == (116, HIDDEN // BLOCK[1])

    weight_param = torch.nn.Parameter(
        torch.empty((chunk_rows, HIDDEN), dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    scale_param = torch.nn.Parameter(
        torch.empty((chunk_rows // BLOCK[0], HIDDEN // BLOCK[1]), dtype=torch.float32),
        requires_grad=False,
    )

    _mimo_v2_copy_paired_qkv_fp8(
        config=config,
        weight_name="model.layers.1.self_attn.qkv_proj.weight",
        scale_name="model.layers.1.self_attn.qkv_proj.weight_scale_inv",
        weight_param=weight_param,
        scale_param=scale_param,
        loaded_weight=weight,
        loaded_scale=scale,
        tp_rank=tp_rank,
        tp_size=4,
        block_size=BLOCK,
    )

    expected = weight.narrow(0, tp_rank * chunk_rows, chunk_rows)
    assert torch.equal(weight_param.data, expected)


@pytest.mark.parametrize("tp_rank", [0, 1, 2, 3])
def test_ga_layer_unchanged_by_detection_refactor(tp_rank: int) -> None:
    config = _config()
    weight, scale, chunk_rows = _build_chunked_qkv(
        n_chunks=4,
        q_heads_per_chunk=16,
        kv_heads_per_chunk=1,
        head_dim=config.head_dim,
        v_head_dim=config.v_head_dim,
        hidden=HIDDEN,
    )
    assert weight.shape == (13568, HIDDEN)
    assert scale.shape == (108, HIDDEN // BLOCK[1])

    weight_param = torch.nn.Parameter(
        torch.empty((chunk_rows, HIDDEN), dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    scale_param = torch.nn.Parameter(
        torch.empty(
            (-(-chunk_rows // BLOCK[0]), HIDDEN // BLOCK[1]), dtype=torch.float32
        ),
        requires_grad=False,
    )

    _mimo_v2_copy_paired_qkv_fp8(
        config=config,
        weight_name="model.layers.0.self_attn.qkv_proj.weight",
        scale_name="model.layers.0.self_attn.qkv_proj.weight_scale_inv",
        weight_param=weight_param,
        scale_param=scale_param,
        loaded_weight=weight,
        loaded_scale=scale,
        tp_rank=tp_rank,
        tp_size=4,
        block_size=BLOCK,
    )

    expected = weight.narrow(0, tp_rank * chunk_rows, chunk_rows)
    assert torch.equal(weight_param.data, expected)
