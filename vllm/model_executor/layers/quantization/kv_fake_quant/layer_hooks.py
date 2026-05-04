# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-Attention-layer KV-quant state + the two hot-path entry points.

``LayerKVQuantState`` is an ``nn.Module`` so the SmoothKV calib buffers
(``s_k``, ``s_v``) auto-migrate when ``module.to(device)`` is called on the
parent Attention layer.

  attach_kv_quant_to_layer(layer, prefix)        called from Attention.__init__
                                                 builds + attaches state
  apply_kv_quant(layer, key, value) -> (K, V)    called from Attention.forward
                                                 reads state + dispatches to kernels
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vllm.config import KVCacheQuantConfig, get_current_vllm_config_or_none
from vllm.logger import init_logger

from .kernels import fake_quantize_fp8, fake_quantize_pertoken

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Per-layer state object
# ---------------------------------------------------------------------------

class LayerKVQuantState(nn.Module):
    """Per-Attention-layer KV-quant state.

    Single source of truth: read by ``apply_kv_quant`` and by the inline
    check in ``Attention.forward``. Subclassing ``nn.Module`` (rather than
    using a plain dataclass) makes the smoothkv calib buffers auto-migrate
    with the parent's ``module.to(device)``.

    Attributes:
        method: "fp8" / "pertoken" / "smoothkv". Note: when the user picked
            "smoothkv_fused", this is "pertoken" because the s_K/s_V scaling
            was already folded into the projection weights at load time
            (see ``fusion.py``).
        group_size: per-group size for the quant kernels.
        bits: bit width (4 for pertoken/smoothkv, 8 for fp8 -- ignored on fp8).
        s_k / s_v: SmoothKV per-(kv_head, channel) scales, registered as
            non-persistent buffers. Only present when method == "smoothkv".
    """

    def __init__(
        self,
        method: str,
        group_size: int,
        bits: int,
        s_k: torch.Tensor | None = None,
        s_v: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.method = method
        self.group_size = group_size
        self.bits = bits
        if s_k is not None:
            assert s_v is not None, "s_k/s_v must be provided together"
            # Non-persistent buffers auto-migrate with module.to(device); won't
            # be saved with state_dict.
            self.register_buffer("s_k", s_k, persistent=False)
            self.register_buffer("s_v", s_v, persistent=False)


# ---------------------------------------------------------------------------
# Calib cache (one entry per calib_path, populated lazily inside the worker
# the first time a smoothkv layer is constructed)
# ---------------------------------------------------------------------------

_CALIB_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def _str_to_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        name, torch.bfloat16
    )


def _load_calib_cached(
    calib_path: str, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load (s_K, s_V) from the calib .pt and keep them on CPU; cache by path
    so workers only pay the IO cost once."""
    cached = _CALIB_CACHE.get(calib_path)
    if cached is not None:
        return cached
    calib = torch.load(calib_path, weights_only=True)
    sk = calib["s_K"].to(dtype)
    sv = calib["s_V"].to(dtype)
    _CALIB_CACHE[calib_path] = (sk, sv)
    return sk, sv


def get_active_kv_quant_config() -> KVCacheQuantConfig | None:
    """Read the active KV-quant config from the current VllmConfig, or None
    if no LLM is constructed / no kv_cache_quant_config was set."""
    vllm_cfg = get_current_vllm_config_or_none()
    if vllm_cfg is None:
        return None
    cfg = vllm_cfg.kv_cache_quant_config
    if cfg is None or not cfg.is_active():
        return None
    return cfg


# ---------------------------------------------------------------------------
# Per-layer registration (called from Attention.__init__)
# ---------------------------------------------------------------------------

def attach_kv_quant_to_layer(layer, prefix: str) -> None:
    """Build a ``LayerKVQuantState`` from the active config and attach it as
    ``layer.kv_quant_state``. No-op unless ``LLM(kv_cache_quant_config=...)``
    was set.

    For ``smoothkv_fused``: at runtime the layer behaves as plain ``pertoken``
    (the s_K / s_V scaling is already folded into qkv_proj / o_proj weights at
    load time by ``maybe_run_post_load_fusion``). So we set the per-layer
    method to ``"pertoken"`` here.
    """
    cfg = get_active_kv_quant_config()
    if cfg is None:
        return

    runtime_method = "pertoken" if cfg.method == "smoothkv_fused" else cfg.method
    s_k, s_v = None, None
    if cfg.method == "smoothkv":
        s_k, s_v = _resolve_smoothkv_scales(layer, prefix, cfg)

    layer.kv_quant_state = LayerKVQuantState(
        method=runtime_method,
        group_size=cfg.group_size,
        bits=cfg.bits,
        s_k=s_k,
        s_v=s_v,
    )


def _resolve_smoothkv_scales(
    layer, prefix: str, cfg: KVCacheQuantConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice the full calib (num_layers, full_num_kv_heads, head_dim) down to
    this layer + this TP worker's kv-head shard."""
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.model_executor.models.utils import extract_layer_index
    try:
        layer_idx = extract_layer_index(prefix)
    except Exception as e:
        raise RuntimeError(
            f"[kv_fake_quant] SmoothKV could not extract layer_idx "
            f"from prefix={prefix!r}"
        ) from e
    try:
        tp_rank = get_tensor_model_parallel_rank()
    except Exception:
        tp_rank = 0
    s_k_full, s_v_full = _load_calib_cached(
        cfg.calib_path, _str_to_dtype(cfg.dtype)
    )
    full_kv_heads = s_k_full.shape[1]
    per_worker = layer.num_kv_heads
    if per_worker == full_kv_heads:
        return s_k_full[layer_idx].clone(), s_v_full[layer_idx].clone()
    lo = tp_rank * per_worker
    hi = lo + per_worker
    return s_k_full[layer_idx, lo:hi].clone(), s_v_full[layer_idx, lo:hi].clone()


# ---------------------------------------------------------------------------
# Forward dispatch (called from Attention.forward)
# ---------------------------------------------------------------------------

def apply_kv_quant(
    layer, key: torch.Tensor, value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fake-quantized (K, V) per the layer's configured method.

    Reads ``layer.kv_quant_state``. torch.compile specializes per
    (method, group_size, bits) value at trace time.
    """
    state: LayerKVQuantState = layer.kv_quant_state
    method = state.method
    gs = state.group_size
    bits = state.bits
    nh = layer.num_kv_heads
    hd = layer.head_size

    if method == "fp8":
        key = fake_quantize_fp8(key, nh, hd, gs)
        value = fake_quantize_fp8(value, nh, hd, gs)
    elif method == "pertoken":
        key = fake_quantize_pertoken(key, nh, hd, gs, bits)
        value = fake_quantize_pertoken(value, nh, hd, gs, bits)
    elif method == "smoothkv":
        sk_flat = state.s_k.reshape(-1)
        sv_flat = state.s_v.reshape(-1)
        k_s = key / sk_flat
        k_s = fake_quantize_pertoken(k_s, nh, hd, gs, bits)
        key = k_s * sk_flat
        v_s = value / sv_flat
        v_s = fake_quantize_pertoken(v_s, nh, hd, gs, bits)
        value = v_s * sv_flat
    else:
        raise ValueError(f"Unknown method: {method!r}")
    return key, value
