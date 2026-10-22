# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-cache fake quantization (quantize-dequantize) for accuracy studies.

When ``VLLM_KV_CACHE_FAKE_QUANT`` is set, K and V of every attention layer are
QDQ'd right after RoPE and before they are written to a plain BF16/FP16 cache
(``--kv-cache-dtype auto``), so the cache and the current attention call both
see the quantized values.

- ``nvfp4``: vLLM's NVFP4 emulation (``ref_nvfp4_quant_dequant``): E2M1 values,
  one E4M3 scale per 16 values along head_dim, global scale 1.0.
- ``mxfp4``: Quark ``qdq_mxfp4`` (via ``mxfp4_utils``): block 32, E8M0 "even" scale.

``VLLM_KV_CACHE_FAKE_QUANT_K_HADAMARD=<b>`` additionally rotates K by a
normalized b x b Hadamard matrix (block-diagonal along head_dim) before the QDQ
and rotates it back after. Because the rotation is orthogonal, this equals
storing the rotated K and rotating Q by the same matrix, as a real kernel would.
"""

from functools import cache

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

KV_FAKE_QUANT_FORMATS = ("mxfp4", "nvfp4")
_NVFP4_BLOCK_SIZE = 16


@cache
def get_kv_fake_quant_format() -> str | None:
    """The KV fake-quant format for all attention layers, or None when off."""
    fmt = envs.VLLM_KV_CACHE_FAKE_QUANT
    if not fmt:
        return None
    if fmt not in KV_FAKE_QUANT_FORMATS:
        raise ValueError(
            f"Unsupported VLLM_KV_CACHE_FAKE_QUANT={fmt!r}, "
            f"expected one of {KV_FAKE_QUANT_FORMATS}"
        )
    logger.info("KV cache fake quant enabled: %s", fmt)
    return fmt


@cache
def get_kv_fake_quant_k_format() -> str | None:
    """The fake-quant format for K: the KV format, plus an optional rotation."""
    fmt = get_kv_fake_quant_format()
    block = envs.VLLM_KV_CACHE_FAKE_QUANT_K_HADAMARD
    if fmt is None or block == 0:
        return fmt
    if block < 2 or block & (block - 1):
        raise ValueError(
            f"VLLM_KV_CACHE_FAKE_QUANT_K_HADAMARD={block} must be a power of 2"
        )
    logger.info("KV cache fake quant: K Hadamard rotation, block %d", block)
    return f"{fmt}@had{block}"


@cache
def _hadamard(block: int, device: torch.device) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float32)
    while h.shape[0] < block:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / block**0.5).to(device)


def _kv_fake_quant(x: torch.Tensor, fmt: str) -> torch.Tensor:
    if "@had" in fmt:
        fmt, block_str = fmt.split("@had")
        block = int(block_str)
        shape = x.shape
        if shape[-1] % block:
            raise ValueError(f"head_dim {shape[-1]} not divisible by {block}")
        h = _hadamard(block, x.device)
        blocks = (*shape[:-1], shape[-1] // block, block)
        y = (x.float().reshape(blocks) @ h).reshape(shape).to(x.dtype)
        y = _kv_fake_quant(y, fmt).float().reshape(blocks) @ h.T
        return y.reshape(shape).to(x.dtype)
    if fmt == "nvfp4":
        from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
            ref_nvfp4_quant_dequant,
        )

        global_scale = torch.ones(1, dtype=torch.float32, device=x.device)
        x2d = x.reshape(-1, x.shape[-1]).contiguous()
        return ref_nvfp4_quant_dequant(x2d, global_scale, _NVFP4_BLOCK_SIZE).view(
            x.shape
        )
    if fmt == "mxfp4":
        from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
            _quant_dequant_mxfp4,
        )

        return _quant_dequant_mxfp4(x.contiguous(), "even")
    raise ValueError(f"Unsupported KV fake-quant format {fmt!r}")


def _kv_fake_quant_fake(x: torch.Tensor, fmt: str) -> torch.Tensor:
    return torch.empty_like(x)


# A custom op keeps the Triton / Quark kernels opaque to torch.compile.
direct_register_custom_op(
    op_name="kv_cache_fake_quant",
    op_func=_kv_fake_quant,
    fake_impl=_kv_fake_quant_fake,
)
kv_cache_fake_quant = torch.ops.vllm.kv_cache_fake_quant


# Persistent token-position buffer of the model runner (registered once; read by
# the K op to find sink positions, CUDA-graph safe since the buffer never moves).
_POSITIONS: torch.Tensor | None = None


def set_kv_fake_quant_positions(positions: torch.Tensor) -> None:
    global _POSITIONS
    _POSITIONS = positions


@cache
def get_kv_fake_quant_k_calib() -> dict | None:
    path = envs.VLLM_KV_CACHE_FAKE_QUANT_K_CALIB
    if not path or get_kv_fake_quant_format() is None:
        return None
    calib = torch.load(path, map_location="cpu")
    logger.info(
        "KV cache fake quant: K centering/balancing from %s (%d layers)",
        path,
        len(calib["layers"]),
    )
    return calib


@cache
def get_kv_fake_quant_k_sink() -> int:
    n = envs.VLLM_KV_CACHE_FAKE_QUANT_K_SINK
    if n and get_kv_fake_quant_format() is not None:
        logger.info("KV cache fake quant: K of the first %d positions kept", n)
        return n
    return 0


def _kv_fake_quant_k_calib(
    x: torch.Tensor, mean: torch.Tensor, lam: torch.Tensor, fmt: str, sink: int
) -> torch.Tensor:
    """x: [T, Hkv, D]; mean/lam: [Hkv, D]. lam * QDQ((x - mean) / lam) + mean."""
    xf = x.float()
    y = _kv_fake_quant(((xf - mean) / lam).to(x.dtype), fmt)
    out = (y.float() * lam + mean).to(x.dtype)
    if sink > 0:
        assert _POSITIONS is not None, "positions buffer not registered"
        keep = _POSITIONS[: x.shape[0]] < sink
        out = torch.where(keep[:, None, None], x, out)
    return out


direct_register_custom_op(
    op_name="kv_cache_fake_quant_k_calib",
    op_func=_kv_fake_quant_k_calib,
    fake_impl=lambda x, mean, lam, fmt, sink: torch.empty_like(x),
)
kv_cache_fake_quant_k_calib = torch.ops.vllm.kv_cache_fake_quant_k_calib


# --- Attention-aware MXFP4 encoding (format unchanged: E2M1 codes, E8M0 per 32) ---
_E2M1_LEVELS = (-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
                0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # fmt: skip
_AWARE_EXP_OFFSETS = (-1, 0, 1)
_AWARE_FLIPS = 4


@cache
def _e2m1_tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    lv = torch.tensor(_E2M1_LEVELS, dtype=torch.float32, device=device)
    return lv, (lv[1:] + lv[:-1]) / 2


@cache
def get_kv_fake_quant_k_sigma() -> dict | None:
    path = envs.VLLM_KV_CACHE_FAKE_QUANT_K_SIGMA
    if not path or get_kv_fake_quant_format() is None:
        return None
    if get_kv_fake_quant_format() != "mxfp4":
        raise ValueError("VLLM_KV_CACHE_FAKE_QUANT_K_SIGMA requires mxfp4")
    sigma = torch.load(path, map_location="cpu")
    logger.info(
        "KV cache fake quant: attention-aware MXFP4 K encoding from %s "
        "(%d layers, Hadamard-%d, exponent offsets %s, %d flips)",
        path, len(sigma["layers"]), sigma["block"], _AWARE_EXP_OFFSETS, _AWARE_FLIPS,
    )  # fmt: skip
    return sigma


@cache
def get_kv_fake_quant_v_mode() -> str:
    mode = envs.VLLM_KV_CACHE_FAKE_QUANT_V_MODE
    if mode and get_kv_fake_quant_format() is not None:
        if mode != "had32_mse3":
            raise ValueError(f"Unsupported VLLM_KV_CACHE_FAKE_QUANT_V_MODE={mode!r}")
        if get_kv_fake_quant_format() != "mxfp4":
            raise ValueError("VLLM_KV_CACHE_FAKE_QUANT_V_MODE requires mxfp4")
        logger.info("KV cache fake quant: V encoding %s", mode)
        return mode
    return ""


def _rotate32(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    h = _hadamard(32, x.device)
    return (x.float().reshape(*shape[:-1], shape[-1] // 32, 32) @ h).reshape(shape)


def _mx_encode(
    y: torch.Tensor, sigma: torch.Tensor | None, offsets: tuple[int, ...], flips: int
) -> torch.Tensor:
    """MXFP4 QDQ of y [T, Hkv, D] (fp32) choosing, per 32-group, the E8M0 exponent
    among OCP exponent + offsets and up to `flips` roundings to the other
    neighbouring E2M1 level, minimising d^T S d (S = sigma [Hkv, G, 32, 32], or the
    identity, i.e. MSE, when sigma is None)."""
    T, Hk, D = y.shape
    g = y.reshape(T, Hk, D // 32, 32)
    lv, md = _e2m1_tables(y.device)
    amax = g.abs().amax(-1, keepdim=True)
    e0 = torch.floor(torch.log2(amax.clamp_min(1e-30))) - 2
    S = None if sigma is None else sigma[None]
    best_q = best_e = None
    for off in offsets:
        sc = torch.exp2(e0 + off)
        x = g / sc
        idx = torch.bucketize(x, md)
        q = lv[idx]
        alt = torch.where(x > q, idx + 1, idx - 1).clamp(0, lv.numel() - 1)
        dq = (lv[alt] - q) * sc
        q = q * sc
        d = q - g
        if S is not None:
            s_diag = torch.diagonal(S, dim1=-2, dim2=-1)
            for _ in range(flips):
                gain = (
                    2 * dq * torch.einsum("...ij,...j->...i", S, d) + dq * dq * s_diag
                )
                gmin, imin = gain.min(-1, keepdim=True)
                onehot = torch.zeros_like(d).scatter_(-1, imin, 1.0) * (gmin < 0)
                step = dq * onehot
                q, d = q + step, d + step
                dq = torch.where(onehot.bool(), -dq, dq)
            err = torch.einsum("...i,...ij,...j->...", d, S, d)[..., None]
        else:
            err = (d * d).sum(-1, keepdim=True)
        if best_q is None:
            best_q, best_e = q, err
        else:
            take = err < best_e
            best_q = torch.where(take, q, best_q)
            best_e = torch.where(take, err, best_e)
    return torch.where(amax == 0, torch.zeros_like(best_q), best_q).reshape(T, Hk, D)


def _kv_fake_quant_k_aware(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """x: [T, Hkv, D]; Hadamard-32, attention-aware MXFP4 QDQ, rotate back."""
    y = _mx_encode(_rotate32(x), sigma, _AWARE_EXP_OFFSETS, _AWARE_FLIPS)
    return _rotate32(y).to(x.dtype)


def _kv_fake_quant_v_mode(x: torch.Tensor, mode: str) -> torch.Tensor:
    """x: [T, Hkv, D]; mode "had32_mse3"."""
    y = _mx_encode(_rotate32(x), None, _AWARE_EXP_OFFSETS, 0)
    return _rotate32(y).to(x.dtype)


direct_register_custom_op(
    op_name="kv_cache_fake_quant_k_aware",
    op_func=_kv_fake_quant_k_aware,
    fake_impl=lambda x, sigma: torch.empty_like(x),
)
kv_cache_fake_quant_k_aware = torch.ops.vllm.kv_cache_fake_quant_k_aware
direct_register_custom_op(
    op_name="kv_cache_fake_quant_v_mode",
    op_func=_kv_fake_quant_v_mode,
    fake_impl=lambda x, mode: torch.empty_like(x),
)
kv_cache_fake_quant_v_mode = torch.ops.vllm.kv_cache_fake_quant_v_mode


# --- Sigma_q calibration inside vLLM (for models too large for the HF path) ---
# With VLLM_KV_CACHE_SIGMA_CALIB_DIR set, every Attention layer accumulates the
# second moment of its post-RoPE queries (Hadamard-32 basis, per KV head and
# 32-group) while the file <dir>/ENABLE exists, and rewrites
# <dir>/rank<r>_layer<L>.pt after each step. Run that server with
# --enforce-eager and compilation mode 0 so this Python code runs every step.
def accumulate_sigma(layer, query: torch.Tensor) -> None:
    import os

    calib_dir = envs.VLLM_KV_CACHE_SIGMA_CALIB_DIR
    if not os.path.exists(os.path.join(calib_dir, "ENABLE")):
        return
    from vllm.distributed import get_tensor_model_parallel_rank

    T, Hq, D = query.shape
    Hk = layer.num_kv_heads
    qh = _rotate32(query).reshape(T, Hk, Hq // Hk, D // 32, 32)
    m = torch.einsum("tkgni,tkgnj->knij", qh, qh)
    if getattr(layer, "_sigma_sum", None) is None:
        layer._sigma_sum, layer._sigma_n = torch.zeros_like(m), 0
    layer._sigma_sum += m
    layer._sigma_n += T * (Hq // Hk)
    rank = get_tensor_model_parallel_rank()
    torch.save(
        {"sum": layer._sigma_sum.cpu(), "n": layer._sigma_n},
        os.path.join(calib_dir, f"rank{rank}_layer{layer._calib_layer_idx}.pt"),
    )
