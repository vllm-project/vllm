# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Indexer weights for the in-kernel (fused) indexer,
``Glm5MonoKernel(with_indexer=True)``.

Kernel contract (replicated on every TP rank, as vLLM's indexer is: ``wq_b``
ReplicatedLinear, ``wk_weights_proj`` disable_tp):

  w_index_k  FP8 E4M3 [128, 6144]  + s_index_k FP32 [1, 48]
             indexer.wk, 128x128 blocks
  w_index_q  FP8 E4M3 [4096, 2048] + s_index_q FP32 [32, 16]
             indexer.wq_b, 128x128 blocks, rows head-major
  w_index_w  BF16 [32, 6144]        indexer.weights_proj
  g_index_k, b_index_k FP32 [128]   indexer.k_norm LayerNorm weight / bias

The checkpoint keeps wk / wq_b / weights_proj in BF16; they are quantized with the same
offline block-128 recipe as FP8 attention (``fp8_attention.quant_fp8_block``), so the
fused indexer's projections differ from vLLM's BF16 GEMMs by that rounding.
"""

from __future__ import annotations

import torch

from vllm.models.deepseek_v32.amd.mono.fp8_attention import quant_fp8_block

INDEX_DIM, INDEX_HEADS, HIDDEN, Q_LORA = 128, 32, 6144, 2048


def pack_index_weights(
    wk: torch.Tensor,
    wq_b: torch.Tensor,
    weights_proj: torch.Tensor,
    k_norm_w: torch.Tensor,
    k_norm_b: torch.Tensor,
) -> dict[str, torch.Tensor]:
    assert tuple(wk.shape) == (INDEX_DIM, HIDDEN), wk.shape
    assert tuple(wq_b.shape) == (INDEX_HEADS * INDEX_DIM, Q_LORA), wq_b.shape
    assert tuple(weights_proj.shape) == (INDEX_HEADS, HIDDEN), weights_proj.shape
    t = {}
    t["w_index_k"], t["s_index_k"] = quant_fp8_block(wk, 128, 128)
    t["w_index_q"], t["s_index_q"] = quant_fp8_block(wq_b, 128, 128)
    t["w_index_w"] = weights_proj.to(torch.bfloat16).contiguous()
    t["g_index_k"] = k_norm_w.float().contiguous()
    t["b_index_k"] = k_norm_b.float().contiguous()
    return t


def index_weights_from_ckpt(
    ckpt_dir: str, layer: int, device
) -> dict[str, torch.Tensor]:
    from vllm.models.deepseek_v32.amd.mono.ckpt_weights import _Ckpt

    ck = _Ckpt(ckpt_dir)
    p = f"model.layers.{layer}.self_attn.indexer."
    g = lambda n: ck.get(p + n).to(device)  # noqa: E731
    return pack_index_weights(
        g("wk.weight"),
        g("wq_b.weight"),
        g("weights_proj.weight"),
        g("k_norm.weight"),
        g("k_norm.bias"),
    )
