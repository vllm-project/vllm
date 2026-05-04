# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
KV-cache fake-quantization for accuracy studies (BF16 / FP8 / per-token int4 /
SmoothKV / KIVI-2). Kernels run quant -> dequant in the same step, so KV
cache storage is unchanged but values are constrained to the chosen grid.

Wired into ``Attention`` via two inline reads (``__init__``: registration,
``forward``: dispatch). No ``Worker.load_model`` post-load hook, no class-level
monkey-patch -- mirrors the partial-sum pattern in #1.

Public API:
    configure_kv_quant(method, group_size, bits, calib_path)   # before LLM(...)
    attach_kv_quant_to_layer(layer, prefix)                    # called from Attention.__init__
    apply_kv_quant(layer, key, value) -> (key, value)          # called from Attention.forward
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration singleton
# ---------------------------------------------------------------------------

@dataclass
class _KvQuantConfig:
    method: str = "bf16"            # bf16 | fp16 | fp8 | pertoken | smoothkv | kivi2
    group_size: int = 128
    bits: int = 4
    # SmoothKV: full (num_layers, full_num_kv_heads, head_dim) tensors on CPU.
    # Sliced per-layer, per-TP-rank inside attach_kv_quant_to_layer().
    smoothkv_s_k_cpu: Optional[torch.Tensor] = None
    smoothkv_s_v_cpu: Optional[torch.Tensor] = None
    # Cached so we don't reload the calib file per layer.
    smoothkv_calib_path: Optional[str] = None


_CONFIG = _KvQuantConfig()


def configure_kv_quant(
    method: str,
    group_size: int = 128,
    bits: int = 4,
    calib_path: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Set the global KV fake-quant config. Must be called BEFORE ``LLM(...)``.

    Each ``Attention.__init__`` reads this singleton and copies the relevant
    fields onto the layer instance. SmoothKV calib is loaded once here (CPU)
    and sliced per-layer/per-TP-rank inside the layer ``__init__``.
    """
    if method not in ("bf16", "fp16", "fp8", "pertoken", "smoothkv",
                      "smoothkv_fused", "kivi2"):
        raise ValueError(
            f"Unknown method {method!r}; "
            "expected bf16/fp16/fp8/pertoken/smoothkv/smoothkv_fused/kivi2"
        )
    if method in ("smoothkv", "smoothkv_fused") and not calib_path:
        raise ValueError(f"method={method!r} requires calib_path")
    if method == "kivi2":
        bits = 2  # KIVI-2 is fixed at int2

    _CONFIG.method = method
    _CONFIG.group_size = group_size
    _CONFIG.bits = bits

    if method in ("smoothkv", "smoothkv_fused"):
        calib = torch.load(calib_path, weights_only=True)
        _CONFIG.smoothkv_s_k_cpu = calib["s_K"].to(dtype)
        _CONFIG.smoothkv_s_v_cpu = calib["s_V"].to(dtype)
        _CONFIG.smoothkv_calib_path = calib_path
    else:
        _CONFIG.smoothkv_s_k_cpu = None
        _CONFIG.smoothkv_s_v_cpu = None
        _CONFIG.smoothkv_calib_path = None

    logger.info(
        "[kv_fake_quant] configured method=%s group_size=%s bits=%s%s",
        method, group_size, bits,
        f" calib_path={calib_path}" if calib_path else "",
    )


# ---------------------------------------------------------------------------
# Kernels (registered as torch.library.custom_op for opaque torch.compile
# capture; names are visible in inductor's computation_graph.py dump).
# ---------------------------------------------------------------------------

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max  # 448.0


def _round_to_fp8e4m3(x_fp32: torch.Tensor) -> torch.Tensor:
    """sm_80-compatible E4M3 round (matches hardware cast for normal range).

    On A100, Triton's inductor codegen cannot lower torch.float8_e4m3fn casts
    inside cudagraphs; this fp32-arithmetic emulation does and is bit-identical
    for normals.
    """
    sign = torch.sign(x_fp32)
    abs_x = x_fp32.abs().clamp(max=_FP8_MAX)
    eps_floor = 2.0 ** -9
    abs_safe = abs_x.clamp(min=eps_floor)
    exp = torch.floor(torch.log2(abs_safe))
    exp_clamped = exp.clamp(min=-6.0, max=8.0)
    pow2_exp = torch.pow(2.0, exp_clamped)
    mantissa_q = torch.round(abs_x / pow2_exp * 8.0) / 8.0
    out = sign * mantissa_q * pow2_exp
    return torch.where(x_fp32 == 0, torch.zeros_like(out), out)


def _is_sm89_or_newer() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0) >= (8, 9)


@torch.library.custom_op(
    "vllm_kv_quant::fake_quantize_dequantize_fp8", mutates_args=()
)
def _fake_quantize_dequantize_fp8(
    data: torch.Tensor, group_size: int
) -> torch.Tensor:
    """FP8 E4M3 quant-dequant round-trip on (B, nh, T, D) input."""
    B, nh, T, D = data.shape
    num_groups = D // group_size
    grouped = data.view(B, nh, T, num_groups, group_size)

    amax = grouped.abs().amax(dim=-1).to(torch.float32)
    scale = (amax / _FP8_MAX).clamp(min=1e-4)
    scaled = grouped.to(torch.float32) / scale.unsqueeze(-1)

    if _is_sm89_or_newer():
        rounded = scaled.to(_FP8_DTYPE).to(torch.float32)
    else:
        rounded = _round_to_fp8e4m3(scaled)

    out = (rounded * scale.unsqueeze(-1)).view(B, nh, T, D)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.to(data.dtype)


@_fake_quantize_dequantize_fp8.register_fake
def _fake_quantize_dequantize_fp8_meta(
    data: torch.Tensor, group_size: int
) -> torch.Tensor:
    return torch.empty_like(data)


# ---- KIVI int{2,4} pack/unpack kernels (per-token along D, per-channel along T) ----

def _pack_tensor(data: torch.Tensor, bits: int, pack_dim: int) -> torch.Tensor:
    feat_per_int = 32 // bits
    shape = data.shape
    out_shape = (
        shape[:pack_dim] + (shape[pack_dim] // feat_per_int,) + shape[pack_dim + 1:]
    )
    code = torch.zeros(out_shape, dtype=torch.int32, device=data.device)
    unpacked_idx: list = [slice(None)] * len(shape)
    packed_idx: list = [slice(None)] * len(shape)
    row = 0
    i = 0
    while row < code.shape[pack_dim]:
        packed_idx[pack_dim] = row
        for j in range(i, i + feat_per_int):
            unpacked_idx[pack_dim] = j
            code[tuple(packed_idx)] |= data[tuple(unpacked_idx)] << (bits * (j - i))
        i += feat_per_int
        row += 1
    return code


def _unpack_tensor(code: torch.Tensor, bits: int, pack_dim: int) -> torch.Tensor:
    feat_per_int = 32 // bits
    shape = code.shape
    new_len = shape[pack_dim] * feat_per_int
    j = torch.arange(new_len, device=code.device) % feat_per_int
    i = torch.arange(new_len, device=code.device) // feat_per_int
    mask = 0xFF >> (8 - bits)
    packed_idx: list = [slice(None)] * len(shape)
    packed_idx[pack_dim] = i
    if pack_dim == 2:
        return ((code[tuple(packed_idx)] >> (j * bits)[None, None, :, None])
                .to(torch.int16)) & mask
    elif pack_dim == 3:
        return ((code[tuple(packed_idx)] >> (j * bits)).to(torch.int16)) & mask
    raise NotImplementedError(f"pack_dim={pack_dim} not supported")


@torch.library.custom_op(
    "vllm_kv_quant::quant_and_pack_vcache", mutates_args=()
)
def _quant_and_pack_vcache(
    v: torch.Tensor, group_size: int, bits: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token V quant: groups along last dim (head_dim). (B, nh, T, D)."""
    shape = v.shape
    num_groups = shape[-1] // group_size
    new_shape = shape[:-1] + (num_groups, group_size)
    max_int = 2 ** bits - 1
    data = v.view(new_shape)
    mn = torch.min(data, dim=-1, keepdim=True)[0]
    mx = torch.max(data, dim=-1, keepdim=True)[0]
    scale = (mx - mn) / max_int
    data = (data - mn) / scale
    data = data.clamp(0, max_int).round().to(torch.int32).view(shape)
    code = _pack_tensor(data, bits, pack_dim=3)
    return code, scale, mn


@_quant_and_pack_vcache.register_fake
def _quant_and_pack_vcache_meta(
    v: torch.Tensor, group_size: int, bits: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, nh, T, D = v.shape
    feat_per_int = 32 // bits
    num_groups = D // group_size
    code = torch.empty(B, nh, T, D // feat_per_int, dtype=torch.int32, device=v.device)
    scale = torch.empty(B, nh, T, num_groups, 1, dtype=v.dtype, device=v.device)
    mn = torch.empty_like(scale)
    return code, scale, mn


@torch.library.custom_op(
    "vllm_kv_quant::unpack_and_dequant_vcache", mutates_args=()
)
def _unpack_and_dequant_vcache(
    code: torch.Tensor, scale: torch.Tensor, mn: torch.Tensor,
    group_size: int, bits: int,
) -> torch.Tensor:
    data = _unpack_tensor(code, bits, pack_dim=3)
    shape = data.shape
    num_groups = shape[-1] // group_size
    data = data.view(shape[:-1] + (num_groups, group_size)).to(scale.dtype)
    data = data * scale + mn
    return data.view(shape)


@_unpack_and_dequant_vcache.register_fake
def _unpack_and_dequant_vcache_meta(
    code: torch.Tensor, scale: torch.Tensor, mn: torch.Tensor,
    group_size: int, bits: int,
) -> torch.Tensor:
    feat_per_int = 32 // bits
    B, nh, T, packed_D = code.shape
    return torch.empty(B, nh, T, packed_D * feat_per_int,
                       dtype=scale.dtype, device=code.device)


@torch.library.custom_op(
    "vllm_kv_quant::quant_and_pack_kcache", mutates_args=()
)
def _quant_and_pack_kcache(
    k: torch.Tensor, group_size: int, bits: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-channel K quant for KIVI-2: groups along T axis. (B, nh, T, D)."""
    shape = k.shape
    B, nh, T, D = shape
    num_groups = T // group_size
    new_shape = (B, nh, num_groups, group_size, D)
    max_int = 2 ** bits - 1
    data = k.view(new_shape)
    mn = torch.min(data, dim=-2, keepdim=True)[0]
    mx = torch.max(data, dim=-2, keepdim=True)[0]
    scale = (mx - mn) / max_int
    data = (data - mn) / scale
    data = data.clamp(0, max_int).round().to(torch.int32).view(shape)
    code = _pack_tensor(data, bits, pack_dim=2)
    return code, scale, mn


@_quant_and_pack_kcache.register_fake
def _quant_and_pack_kcache_meta(
    k: torch.Tensor, group_size: int, bits: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, nh, T, D = k.shape
    feat_per_int = 32 // bits
    num_groups = T // group_size
    code = torch.empty(B, nh, T // feat_per_int, D, dtype=torch.int32, device=k.device)
    scale = torch.empty(B, nh, num_groups, 1, D, dtype=k.dtype, device=k.device)
    mn = torch.empty_like(scale)
    return code, scale, mn


@torch.library.custom_op(
    "vllm_kv_quant::unpack_and_dequant_kcache", mutates_args=()
)
def _unpack_and_dequant_kcache(
    code: torch.Tensor, scale: torch.Tensor, mn: torch.Tensor,
    group_size: int, bits: int,
) -> torch.Tensor:
    data = _unpack_tensor(code, bits, pack_dim=2)
    shape = data.shape
    num_groups = shape[2] // group_size
    data = data.view(shape[:2] + (num_groups, group_size) + shape[3:]).to(scale.dtype)
    data = data * scale + mn
    return data.view(shape)


@_unpack_and_dequant_kcache.register_fake
def _unpack_and_dequant_kcache_meta(
    code: torch.Tensor, scale: torch.Tensor, mn: torch.Tensor,
    group_size: int, bits: int,
) -> torch.Tensor:
    feat_per_int = 32 // bits
    B, nh, packed_T, D = code.shape
    return torch.empty(B, nh, packed_T * feat_per_int, D,
                       dtype=scale.dtype, device=code.device)


# ---------------------------------------------------------------------------
# Shape helpers + high-level fake-quant entry points (compose ops above)
# ---------------------------------------------------------------------------

def _to_bnhtd(x: torch.Tensor, num_kv_heads: int, head_dim: int) -> torch.Tensor:
    """Reshape (T, num_kv_heads*head_dim) or (T, num_kv_heads, head_dim)
    -> (B=1, nh, T, D)."""
    if x.dim() == 2:
        T = x.shape[0]
        return x.view(1, T, num_kv_heads, head_dim).transpose(1, 2).contiguous()
    if x.dim() == 3:
        return x.unsqueeze(0).transpose(1, 2).contiguous()
    raise ValueError(f"unexpected KV shape {tuple(x.shape)}")


def _from_bnhtd(x4: torch.Tensor, orig_shape: torch.Size) -> torch.Tensor:
    return x4.transpose(1, 2).contiguous().view(*orig_shape)


def fake_quantize_fp8(
    x: torch.Tensor, num_kv_heads: int, head_dim: int, group_size: int
) -> torch.Tensor:
    orig_shape = x.shape
    orig_dtype = x.dtype
    x4 = _to_bnhtd(x, num_kv_heads, head_dim)
    out = torch.ops.vllm_kv_quant.fake_quantize_dequantize_fp8(x4, group_size)
    return _from_bnhtd(out, orig_shape).to(orig_dtype)


def fake_quantize_pertoken(
    x: torch.Tensor, num_kv_heads: int, head_dim: int,
    group_size: int, bits: int,
) -> torch.Tensor:
    """Per-token int{2,4} quant: groups along the last dim (head_dim).

    Used for both K and V in pertoken / smoothkv / kivi2 methods (the kernel
    is symmetric in K vs V; only KIVI-2's per-channel K uses a different
    kernel — see ``fake_quantize_k_perchannel``).
    """
    orig_shape = x.shape
    orig_dtype = x.dtype
    x4 = _to_bnhtd(x, num_kv_heads, head_dim)
    code, scale, mn = torch.ops.vllm_kv_quant.quant_and_pack_vcache(
        x4, group_size, bits
    )
    out = torch.ops.vllm_kv_quant.unpack_and_dequant_vcache(
        code, scale, mn, group_size, bits
    )
    return _from_bnhtd(out, orig_shape).to(orig_dtype)


def fake_quantize_k_perchannel(
    x: torch.Tensor, num_kv_heads: int, head_dim: int,
    group_size: int, bits: int,
) -> torch.Tensor:
    """Per-channel K (groups along T axis) -- only KIVI-2 uses this."""
    orig_shape = x.shape
    orig_dtype = x.dtype
    x4 = _to_bnhtd(x, num_kv_heads, head_dim)
    B, nh, T, D = x4.shape
    pad = (-T) % group_size
    if pad:
        x4 = torch.cat(
            [x4, torch.zeros(B, nh, pad, D, dtype=x4.dtype, device=x4.device)],
            dim=2,
        )
    code, scale, mn = torch.ops.vllm_kv_quant.quant_and_pack_kcache(
        x4, group_size, bits
    )
    out = torch.ops.vllm_kv_quant.unpack_and_dequant_kcache(
        code, scale, mn, group_size, bits
    )
    if pad:
        out = out[:, :, :T, :]
    return _from_bnhtd(out, orig_shape).to(orig_dtype)


# ---------------------------------------------------------------------------
# Per-layer registration (called from Attention.__init__)
# ---------------------------------------------------------------------------

def attach_kv_quant_to_layer(layer, prefix: str) -> None:
    """Read the global config; if a method is active, set per-instance attrs
    and (for SmoothKV) register CPU-side calib scales as non-persistent buffers
    so they auto-migrate to the device when ``module.to(device)`` is called.

    For ``smoothkv_fused``: at runtime the layer behaves as plain ``pertoken``
    (the s_K / s_V scaling is already folded into qkv_proj / o_proj weights at
    load time by ``maybe_run_post_load_fusion``). So we set the per-layer
    method to ``"pertoken"`` here.
    """
    if _CONFIG.method in ("bf16", "fp16"):
        return

    runtime_method = (
        "pertoken" if _CONFIG.method == "smoothkv_fused" else _CONFIG.method
    )
    layer._kv_quant_method = runtime_method
    layer._kv_quant_group_size = _CONFIG.group_size
    layer._kv_quant_bits = _CONFIG.bits

    if _CONFIG.method == "smoothkv":
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
        s_k_full = _CONFIG.smoothkv_s_k_cpu
        s_v_full = _CONFIG.smoothkv_s_v_cpu
        full_kv_heads = s_k_full.shape[1]
        per_worker = layer.num_kv_heads
        if per_worker == full_kv_heads:
            sk = s_k_full[layer_idx].clone()
            sv = s_v_full[layer_idx].clone()
        else:
            lo = tp_rank * per_worker
            hi = lo + per_worker
            sk = s_k_full[layer_idx, lo:hi].clone()
            sv = s_v_full[layer_idx, lo:hi].clone()
        # Non-persistent buffers auto-migrate with module.to(device); won't be
        # saved with state_dict.
        layer.register_buffer("_kv_quant_s_k", sk, persistent=False)
        layer.register_buffer("_kv_quant_s_v", sv, persistent=False)


# ---------------------------------------------------------------------------
# Forward dispatch (called from Attention.forward)
# ---------------------------------------------------------------------------

def apply_kv_quant(
    layer, key: torch.Tensor, value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fake-quantized (K, V) per the layer's configured method.

    Read at trace time; torch.compile specializes per (method, group_size, bits).
    """
    method = layer._kv_quant_method
    gs = layer._kv_quant_group_size
    bits = layer._kv_quant_bits
    nh = layer.num_kv_heads
    hd = layer.head_size

    if method == "fp8":
        key = fake_quantize_fp8(key, nh, hd, gs)
        value = fake_quantize_fp8(value, nh, hd, gs)
    elif method == "pertoken":
        key = fake_quantize_pertoken(key, nh, hd, gs, bits)
        value = fake_quantize_pertoken(value, nh, hd, gs, bits)
    elif method == "smoothkv":
        sk_flat = layer._kv_quant_s_k.reshape(-1)
        sv_flat = layer._kv_quant_s_v.reshape(-1)
        k_s = key / sk_flat
        k_s = fake_quantize_pertoken(k_s, nh, hd, gs, bits)
        key = k_s * sk_flat
        v_s = value / sv_flat
        v_s = fake_quantize_pertoken(v_s, nh, hd, gs, bits)
        value = v_s * sv_flat
    elif method == "kivi2":
        # KIVI-2: K per-channel int2, V per-token int2 (residual buffer not
        # faithfully simulated -- see KIVI paper §3.3).
        key = fake_quantize_pertoken(key, nh, hd, gs, 2)
        value = fake_quantize_pertoken(value, nh, hd, gs, 2)
    else:
        raise ValueError(f"Unknown _kv_quant_method: {method!r}")
    return key, value


# ---------------------------------------------------------------------------
# SmoothKV-fused: in-place fold s_K / s_V into projection weights at load time.
# Runtime path is then plain pertoken int4 -- zero per-step cost beyond
# pertoken. Called once per worker from Worker.load_model post weight-load.
# ---------------------------------------------------------------------------

import re as _re  # noqa: E402

_LAYER_RE = _re.compile(r"\.layers\.(\d+)\.")


def _fuse_one_layer(
    module, sk_all, sv_all, layer_idx: int, tp_rank: int, tp_size: int,
) -> bool:
    """Fold one layer's s_K/s_V scales into its qkv_proj / o_proj weights.

    Two paths depending on architecture:

    (A) Models WITHOUT q_norm/k_norm  (Llama-3, Mistral, ...):
          Q rows × s_K  (broadcast over GQA), K rows ÷ s_K, V rows ÷ s_V

    (B) Models WITH q_norm/k_norm  (Qwen3, Qwen3-MoE, ...):
          Pre-RoPE the path is qkv_proj -> reshape -> q_norm/k_norm -> RoPE.
          RMSNorm doesn't commute with per-channel scaling, so fold s_K into
          the post-norm gamma vectors instead:
              q_norm.weight *= s_K  ;  k_norm.weight /= s_K
          q/k_norm.weight has shape (head_dim,) shared across heads, so this
          requires HEAD-UNIFORM s_K (use the `_huk_` calib files).
          V/O fusion still uses per-(head, channel) s_V via qkv_proj V rows.

    Then o_proj input columns get s_V_q (broadcast over GQA) for both paths.
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

    No-op unless ``configure_kv_quant("smoothkv_fused", calib_path=...)`` was
    set before ``LLM(...)``. When active, folds s_K / s_V into the model's
    projection weights in place. After this returns, the runtime forward path
    uses plain pertoken int4 (no per-step rescaling).
    """
    if _CONFIG.method != "smoothkv_fused":
        return
    sk = _CONFIG.smoothkv_s_k_cpu
    sv = _CONFIG.smoothkv_s_v_cpu
    if sk is None or sv is None:
        raise RuntimeError(
            "[kv_fake_quant] smoothkv_fused configured but calib not loaded. "
            "configure_kv_quant() must run before LLM(...)."
        )
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
