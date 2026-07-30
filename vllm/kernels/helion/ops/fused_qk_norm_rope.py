# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import product
from typing import Any

import torch

from vllm.kernels.helion.case_key import CaseKey
from vllm.logger import init_logger
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )

import helion
import helion.language as hl

from vllm import ir
from vllm.kernels.helion.register import register_kernel
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

logger = init_logger(__name__)


def _compute_cos_sin_cache(
    max_position_embeddings, rotary_dim, device="cuda", dtype=torch.float
):
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, rotary_dim, 2, device=device, dtype=dtype) / rotary_dim)
    )

    t = torch.arange(max_position_embeddings, device=device, dtype=dtype)

    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1)
    return cache


def generate_inputs() -> dict[CaseKey, tuple[Any, ...]]:
    # TODO(xiaohongchen1991): it is difficult for kernel author to cover
    # all input property combination. Currently, dtypes are fixed. We need
    # optimization to bucket/skip some combinations
    num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    num_heads_pair = [
        (16, 8),
        (32, 8),
        (64, 8),
    ]
    head_dim_list = [64, 128, 256]
    in_dtype: torch.dtype = torch.bfloat16
    rotary_ratio = 1.0
    is_neox = True
    eps = 1e-6
    device = "cuda"
    inputs = {}

    for num_tokens, (num_q_heads, num_kv_heads), head_dim in product(
        num_tokens_list, num_heads_pair, head_dim_list
    ):
        total_dim = (num_q_heads + 2 * num_kv_heads) * head_dim
        qkv = torch.empty(
            num_tokens, total_dim, dtype=in_dtype, device=device
        ).uniform_(-0.1, 0.1)
        positions = torch.arange(num_tokens, dtype=torch.long, device=device)
        q_weight = torch.empty(head_dim, dtype=in_dtype, device=device).uniform_(
            0.8, 1.2
        )
        k_weight = torch.empty(head_dim, dtype=in_dtype, device=device).uniform_(
            0.8, 1.2
        )
        rotary_dim = int(head_dim * rotary_ratio)
        cos_sin_cache = _compute_cos_sin_cache(40960, rotary_dim)
        cos_sin_cache = cos_sin_cache.to(in_dtype)

        config_key = CaseKey(
            {
                "q_heads": num_q_heads,
                "kv_heads": num_kv_heads,
                "num_tokens": num_tokens,
                "head_dim": head_dim,
            }
        )
        inputs[config_key] = (
            qkv,
            num_q_heads,
            num_kv_heads,
            num_kv_heads,
            head_dim,
            eps,
            q_weight,
            k_weight,
            cos_sin_cache,
            is_neox,
            positions.view(-1),
        )

    return inputs


_pick_cache: dict[tuple[int, int, int, int], CaseKey | None] = {}


def pick_config(args: tuple[Any, ...], config_keys: list[CaseKey]) -> CaseKey | None:
    """Pick the best pre-tuned config for the given input shape.

    Selection strategy:
      1. Prefer configs tuned for the exact head_dim. If none exist, use
         configs without a head_dim key, then fall back to the closest head_dim.
      2. Find the closest q_heads among available configs
         (exact match preferred).
      3. Find the closest kv_heads among available configs
         (exact match preferred).
      4. Among the num_tokens values tuned for that q_heads and kv_heads, pick
         the smallest num_tokens >= the input's num_tokens. If the input is
         larger than all available num_tokens, fall back to the largest.
    """

    if not config_keys:
        return None

    qkv, q_heads, kv_heads, _, head_dim, *_ = args
    num_tokens = qkv.shape[0]

    cache_key = (num_tokens, q_heads, kv_heads, head_dim)
    cached = _pick_cache.get(cache_key)
    if cached is not None:
        return cached

    non_default_keys = [key for key in config_keys if not key.is_default()]
    candidate_keys = [
        key for key in non_default_keys if key.get("head_dim") == head_dim
    ]
    if not candidate_keys:
        candidate_keys = [key for key in non_default_keys if "head_dim" not in key]
    if not candidate_keys and non_default_keys:
        best_head_dim = min(
            {key["head_dim"] for key in non_default_keys},
            key=lambda value: abs(value - head_dim),
        )
        candidate_keys = [
            key for key in non_default_keys if key["head_dim"] == best_head_dim
        ]

    configs: dict[int, dict[int, list[int]]] = {}
    for key in candidate_keys:
        configs.setdefault(key["q_heads"], {}).setdefault(key["kv_heads"], []).append(
            key["num_tokens"]
        )

    if not configs:
        return None

    best_q_heads = min(configs, key=lambda s: abs(s - q_heads))
    best_kv_heads = min(configs[best_q_heads], key=lambda s: abs(s - kv_heads))
    available_num_tokens = sorted(configs[best_q_heads][best_kv_heads])
    best_num_tokens = next(
        (n for n in available_num_tokens if n >= num_tokens), available_num_tokens[-1]
    )

    result = next(
        key
        for key in candidate_keys
        if key["q_heads"] == best_q_heads
        and key["kv_heads"] == best_kv_heads
        and key["num_tokens"] == best_num_tokens
    )
    _pick_cache[cache_key] = result
    return result


def fake_impl(
    qkv: torch.Tensor,  # [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,  # [max_position, rotary_dim]
    is_neox: bool,
    position_ids: torch.Tensor,  # [num_tokens],
    forced_token_heads_per_warp: int = -1,  # dummy
) -> None:
    return


def baseline(
    qkv: torch.Tensor,  # [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,  # [max_position, rotary_dim]
    is_neox: bool,
    position_ids: torch.Tensor,  # [num_tokens],
    forced_token_heads_per_warp: int = -1,  # dummy
) -> None:
    q_size = num_heads_q * head_dim
    kv_size = num_heads_k * head_dim

    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

    q_by_head = q.view(*q.shape[:-1], q.shape[-1] // head_dim, head_dim)
    q_by_head = ir.ops.rms_norm(q_by_head, q_weight, eps)
    q = q_by_head.view(q.shape)

    k_by_head = k.view(*k.shape[:-1], k.shape[-1] // head_dim, head_dim)
    k_by_head = ir.ops.rms_norm(k_by_head, k_weight, eps)
    k = k_by_head.view(k.shape)

    q, k = RotaryEmbedding.forward_static(
        position_ids, q, k, head_dim, cos_sin_cache.shape[1], cos_sin_cache, is_neox
    )
    qkv[:, :q_size].copy_(q)
    qkv[:, q_size : q_size + kv_size].copy_(k)


# Overwrite autotune_baseline_atol and autotune_baseline_rtol
# if too many configs failed due to baseline check during autotuning
@register_kernel(
    mutates_args=["qkv"],
    config_picker=pick_config,
    input_generator=generate_inputs,
    fake_impl=fake_impl,
    helion_settings=helion.Settings(
        autotune_baseline_fn=baseline,
        autotune_baseline_atol=5e-2,
        autotune_baseline_rtol=5e-2,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    ),
)  # type: ignore[misc]
def fused_qk_norm_rope(
    qkv: torch.Tensor,  # [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,  # [max_position, rotary_dim]
    is_neox: bool,
    position_ids: torch.Tensor,  # [num_tokens],
    forced_token_heads_per_warp: int = -1,  # dummy
) -> None:
    assert qkv.ndim == 2
    num_tokens = qkv.shape[0]
    total_heads = num_heads_q + num_heads_k + num_heads_v
    assert qkv.shape[1] == total_heads * head_dim
    hl.specialize(qkv.shape[1])

    assert cos_sin_cache.ndim == 2
    max_position, rotary_dim = cos_sin_cache.shape
    hl.specialize(max_position)
    hl.specialize(rotary_dim)
    assert rotary_dim % 2 == 0
    assert rotary_dim <= head_dim
    embed_dim = rotary_dim // 2

    hl.specialize(num_heads_q)
    hl.specialize(num_heads_k)
    hl.specialize(num_heads_v)
    hl.specialize(head_dim)

    assert position_ids.ndim == 1 and position_ids.shape[0] == num_tokens
    hl.specialize(position_ids.shape[0])

    assert q_weight.ndim == 1 and q_weight.shape[0] == head_dim
    hl.specialize(q_weight.shape[0])
    assert k_weight.ndim == 1 and k_weight.shape[0] == head_dim
    hl.specialize(k_weight.shape[0])

    assert qkv.dtype == q_weight.dtype and q_weight.dtype == k_weight.dtype
    assert position_ids.dtype == torch.int64

    assert qkv.is_contiguous()
    assert position_ids.is_contiguous()
    assert q_weight.is_contiguous()
    assert k_weight.is_contiguous()
    assert cos_sin_cache.is_contiguous()

    qk_heads = num_heads_q + num_heads_k

    qkv = qkv.view(num_tokens, -1, head_dim)

    for tile_m, tile_gn, tile_n in hl.tile(
        [num_tokens, qk_heads, head_dim], block_size=[1, None, head_dim]
    ):
        use_q_weight = (tile_gn.index < num_heads_q)[None, :, None]

        if is_neox:
            x1_offset = hl.arange(embed_dim)
            x2_offset = x1_offset + embed_dim
        else:
            x1_offset = hl.arange(embed_dim) * 2
            x2_offset = x1_offset + 1

        # Partial RoPE still requires RMSNorm over the full head, including the
        # unrotated tail, so retain the full-head normalization path.
        if rotary_dim < head_dim:
            x_blk = qkv[tile_m, tile_gn, tile_n].to(dtype=torch.float32)
            rms = torch.rsqrt(x_blk.pow(2).sum(dim=-1) * (1.0 / head_dim) + eps)
            weight = torch.where(
                use_q_weight,
                q_weight[None, None, tile_n],
                k_weight[None, None, tile_n],
            )
            x_blk = (x_blk * rms[:, :, None]).to(qkv.dtype) * weight
            qkv[tile_m, tile_gn, tile_n] = x_blk
            x1_blk = qkv[tile_m, tile_gn, x1_offset]
            x2_blk = qkv[tile_m, tile_gn, x2_offset]
        else:
            # Full RoPE covers the entire head. Keep both rotary halves in
            # registers through RMSNorm and RoPE to avoid a store/reload round trip.
            x1_blk = qkv[tile_m, tile_gn, x1_offset].to(dtype=torch.float32)
            x2_blk = qkv[tile_m, tile_gn, x2_offset].to(dtype=torch.float32)
            rms = x1_blk.pow(2).sum(dim=-1) + x2_blk.pow(2).sum(dim=-1)
            rms = torch.rsqrt(rms * (1.0 / head_dim) + eps)
            x1_weight = torch.where(
                use_q_weight,
                q_weight[None, None, x1_offset],
                k_weight[None, None, x1_offset],
            )
            x2_weight = torch.where(
                use_q_weight,
                q_weight[None, None, x2_offset],
                k_weight[None, None, x2_offset],
            )
            x1_blk = (x1_blk * rms[:, :, None]).to(qkv.dtype) * x1_weight
            x2_blk = (x2_blk * rms[:, :, None]).to(qkv.dtype) * x2_weight

        pos_id = position_ids[tile_m]
        cos_blk = cos_sin_cache[pos_id, hl.arange(embed_dim)]
        sin_blk = cos_sin_cache[pos_id, hl.arange(embed_dim) + embed_dim]

        o1_blk = x1_blk * cos_blk[:, None, :] - x2_blk * sin_blk[:, None, :]
        o2_blk = x2_blk * cos_blk[:, None, :] + x1_blk * sin_blk[:, None, :]

        qkv[tile_m, tile_gn, x1_offset] = o1_blk
        qkv[tile_m, tile_gn, x2_offset] = o2_blk
