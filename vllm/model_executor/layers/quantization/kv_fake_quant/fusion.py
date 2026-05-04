# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SmoothKV-fused: fold s_K / s_V into projection weights at load time.

After fusion the runtime forward path is plain pertoken int4 -- zero per-step
cost beyond pertoken. Called once per worker from ``Worker.load_model``
post weight-load via ``maybe_run_post_load_fusion``.

Two architectural paths:

  (A) Models WITHOUT q_norm/k_norm  (Llama-3, Mistral, ...):
        qkv_proj rows -- Q × s_K (broadcast over GQA),  K ÷ s_K,  V ÷ s_V
        o_proj cols   -- × s_V_q (broadcast over GQA)

  (B) Models WITH q_norm/k_norm  (Qwen3, Qwen3-MoE, EXAONE-4):
        Pre-RoPE the path is qkv_proj -> reshape -> q_norm/k_norm -> RoPE.
        RMSNorm doesn't commute with per-channel scaling, so fold s_K into
        the post-norm gamma vectors instead:
            q_norm.weight *= s_K  ;  k_norm.weight /= s_K
        q/k_norm.weight has shape (head_dim,) shared across heads, so this
        requires HEAD-UNIFORM s_K (the `_huk_*` calib files).
        V/O fusion is unchanged from path A (V/O have neither RMSNorm nor RoPE).
"""

from __future__ import annotations

import re

import torch

from vllm.logger import init_logger

from .layer_hooks import (
    _load_calib_cached,
    _str_to_dtype,
    get_active_kv_quant_config,
)

logger = init_logger(__name__)


_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def _fuse_one_layer(
    module, sk_all, sv_all, layer_idx: int, tp_rank: int, tp_size: int,
) -> bool:
    """Fold one layer's s_K/s_V scales into its qkv_proj / o_proj weights.

    Returns True if the q_norm/k_norm path was used (used by the caller for
    the summary log).
    """
    total_kv = sk_all.shape[1]
    kv_per_rank = max(1, total_kv // tp_size)
    if total_kv >= tp_size:
        kv_start = tp_rank * kv_per_rank
    else:
        kv_start = 0
        kv_per_rank = total_kv
    kv_end = kv_start + kv_per_rank

    device = module.qkv_proj.weight.device
    dtype = module.qkv_proj.weight.dtype
    s_K = sk_all[layer_idx, kv_start:kv_end].to(device=device, dtype=torch.float32)
    s_V = sv_all[layer_idx, kv_start:kv_end].to(device=device, dtype=torch.float32)

    n_rep = module.num_heads // module.num_kv_heads
    q_size = module.q_size           # num_heads * head_dim (per rank)
    kv_size = module.kv_size         # num_kv_heads * head_dim (per rank)

    has_qknorm = hasattr(module, "q_norm") and hasattr(module, "k_norm")

    if has_qknorm:
        # Path B: fold s_K into q_norm / k_norm gammas. Requires head-uniform.
        head_diff = (s_K - s_K[:1]).abs().max().item()
        if head_diff > 1e-5:
            raise RuntimeError(
                f"layer {layer_idx}: smoothkv_fused needs head-uniform s_K for "
                f"models with q_norm/k_norm, but observed head-to-head diff = "
                f"{head_diff:.4g}. Build the calib with `--head_uniform_k` "
                f"(produces `_huk_*` files)."
            )
        s_K_per_channel = s_K[0]    # (head_dim,)
        with torch.no_grad():
            module.q_norm.weight.data.mul_(
                s_K_per_channel.to(module.q_norm.weight.dtype)
            )
            module.k_norm.weight.data.div_(
                s_K_per_channel.to(module.k_norm.weight.dtype)
            )
        # qkv_proj: only V rows get scaled (Q/K handled via norm gammas).
        s_V_flat = s_V.reshape(-1)
        ones_q = torch.ones(q_size, device=device, dtype=torch.float32)
        ones_k = torch.ones(kv_size, device=device, dtype=torch.float32)
        qkv_scale = torch.cat([ones_q, ones_k, 1.0 / s_V_flat])
    else:
        # Path A: fold s_K into qkv_proj rows directly.
        s_K_q = s_K.repeat_interleave(n_rep, dim=0).reshape(-1)
        s_K_flat = s_K.reshape(-1)
        s_V_flat = s_V.reshape(-1)
        qkv_scale = torch.cat([s_K_q, 1.0 / s_K_flat, 1.0 / s_V_flat])

    assert qkv_scale.shape[0] == q_size + 2 * kv_size, (
        f"layer {layer_idx}: qkv_scale len {qkv_scale.shape[0]} != "
        f"q_size+2*kv_size={q_size + 2 * kv_size}"
    )
    with torch.no_grad():
        module.qkv_proj.weight.data.mul_(qkv_scale.to(dtype).unsqueeze(-1))

    # o_proj input-channel scale: per q_head channel = s_V_q (both paths).
    s_V_q = s_V.repeat_interleave(n_rep, dim=0).reshape(-1)
    with torch.no_grad():
        module.o_proj.weight.data.mul_(s_V_q.to(dtype).unsqueeze(0))

    return has_qknorm


def fuse_smoothkv_into_model(
    model, sk_all: torch.Tensor, sv_all: torch.Tensor,
    tp_rank: int = 0, tp_size: int = 1,
) -> tuple[int, int]:
    """In-place fuse s_K / s_V into every attention layer of the model.

    Returns: (num_layers_fused, num_layers_using_qknorm_path).
    """
    fused = 0
    qknorm = 0
    for name, module in model.named_modules():
        m = _LAYER_RE.search(name)
        if m is None:
            continue
        if not (hasattr(module, "qkv_proj") and hasattr(module, "o_proj")):
            continue
        if not (name.endswith("self_attn") or name.endswith(".attn")):
            continue
        had_qknorm = _fuse_one_layer(
            module, sk_all, sv_all, int(m.group(1)), tp_rank, tp_size,
        )
        fused += 1
        qknorm += int(had_qknorm)
    return fused, qknorm


def maybe_run_post_load_fusion(model) -> None:
    """Called once per worker from ``Worker.load_model`` post weight-load.

    No-op unless ``LLM(kv_cache_quant_config=KVCacheQuantConfig(method=
    "smoothkv_fused", ...))`` was set. When active, folds s_K / s_V into
    the model's projection weights in place. After this returns, the runtime
    forward path uses plain pertoken int4 (no per-step rescaling).
    """
    cfg = get_active_kv_quant_config()
    if cfg is None or cfg.method != "smoothkv_fused":
        return
    sk, sv = _load_calib_cached(cfg.calib_path, _str_to_dtype(cfg.dtype))
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
    except Exception:
        tp_rank, tp_size = 0, 1

    fused, qknorm = fuse_smoothkv_into_model(model, sk, sv, tp_rank, tp_size)
    logger.info(
        "[kv_fake_quant] smoothkv_fused: folded scales into %d layer(s) "
        "(%d via q_norm/k_norm path, %d via qkv_proj-direct path), "
        "tp_rank=%d tp_size=%d",
        fused, qknorm, fused - qknorm, tp_rank, tp_size,
    )
