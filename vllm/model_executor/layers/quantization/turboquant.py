# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant: online vector quantization for KV cache compression.

Implements the TurboQuant algorithm (https://arxiv.org/abs/2504.19874)
for sub-4-bit KV cache quantization with near-optimal distortion.

Two-stage approach:
  1. Random rotation + Lloyd-Max scalar quantization (b-1 bits)
  2. QJL 1-bit transform on residual for unbiased inner products

Adapted from:
  - https://github.com/tonbistudio/turboquant-pytorch
  - https://github.com/TheTom/turboquant_plus
  - https://github.com/Blaizzy/mlx-vlm/pull/858
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch import Tensor

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods


# ---------------------------------------------------------------------------
# Precomputed Lloyd-Max codebooks (Gaussian approximation N(0, 1/d)).
#
# Normalized centroids: multiply by 1/sqrt(d) to get actual values.
# Solved via Lloyd-Max algorithm on the Gaussian PDF.
# ---------------------------------------------------------------------------

CODEBOOKS_NORMALIZED: dict[int, list[float]] = {
    1: [-0.7979, 0.7979],  # +/-sqrt(2/pi)
    2: [-1.5104, -0.4528, 0.4528, 1.5104],
    3: [
        -2.1520, -1.3440, -0.7560, -0.2451,
        0.2451, 0.7560, 1.3440, 2.1520,
    ],
    4: [
        -2.7326, -2.0690, -1.6180, -1.2562,
        -0.9424, -0.6568, -0.3880, -0.1284,
        0.1284, 0.3880, 0.6568, 0.9424,
        1.2562, 1.6180, 2.0690, 2.7326,
    ],
}

EXPECTED_MSE_NORMALIZED: dict[int, float] = {
    1: 0.3634,
    2: 0.1175,
    3: 0.03454,
    4: 0.009497,
}


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV cache quantization.

    Supports fractional bit-widths (2.5, 3.5) via mixed-precision
    channel splitting:
      - 2.5-bit: 50% channels at 3-bit + 50% at 2-bit
      - 3.5-bit: 50% channels at 4-bit + 50% at 3-bit
    """

    bit_width: float = 3
    use_qjl: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        valid = {1, 2, 2.5, 3, 3.5, 4}
        if self.bit_width not in valid:
            raise ValueError(
                f"bit_width must be one of {sorted(valid)}, got {self.bit_width}"
            )
        if self.use_qjl and self.bit_width == 1:
            raise ValueError(
                "use_qjl=True requires bit_width >= 2 (1 bit leaves no "
                "capacity for MSE codebook after reserving 1 bit for QJL)"
            )

    @property
    def is_fractional(self) -> bool:
        return self.bit_width != int(self.bit_width)

    @property
    def channel_split(self) -> tuple[tuple[int, float], tuple[int, float]] | None:
        """Return ((hi_bits, hi_ratio), (lo_bits, lo_ratio)) for fractional.

        E.g. 3.5 -> ((4, 0.5), (3, 0.5)) meaning 50% at 4-bit, 50% at 3-bit.
             2.5 -> ((3, 0.5), (2, 0.5)) meaning 50% at 3-bit, 50% at 2-bit.
        """
        if not self.is_fractional:
            return None
        if self.bit_width == 2.5:
            return ((3, 0.5), (2, 0.5))
        if self.bit_width == 3.5:
            # 4*0.5 + 3*0.5 = 3.5  (correct weighted average)
            return ((4, 0.5), (3, 0.5))
        return None


# ---------------------------------------------------------------------------
# Random rotation via QR decomposition (Haar-distributed orthogonal matrix).
#
# All three reference implementations use this approach, NOT Hadamard.
# O(d^2) storage and compute per vector, but d=128 is small enough.
# ---------------------------------------------------------------------------


def _generate_rotation_matrix(
    d: int, seed: int, device: torch.device
) -> Tensor:
    """Generate a Haar-distributed random rotation matrix via QR."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen, device="cpu", dtype=torch.float32)
    Q, R = torch.linalg.qr(G)
    # Fix sign ambiguity in QR to get proper rotation
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device)


def _get_codebook(bit_width: int, dim: int, device: torch.device) -> Tensor:
    """Return Lloyd-Max codebook scaled to dimension d."""
    normalized = CODEBOOKS_NORMALIZED[bit_width]
    scale = 1.0 / math.sqrt(dim)
    return torch.tensor(
        [c * scale for c in normalized],
        device=device,
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# Public helpers (used by tests and external callers)
# ---------------------------------------------------------------------------


def random_rotate(x: Tensor, sign_flips: Tensor) -> Tensor:
    """Apply random sign flips followed by QR rotation.

    This is a simplified rotation for testing. For the full pipeline,
    use TurboQuantState which manages per-layer rotation matrices.
    """
    return x * sign_flips.unsqueeze(0) if sign_flips.dim() == 1 else x * sign_flips


def random_rotate_inverse(y: Tensor, sign_flips: Tensor) -> Tensor:
    """Inverse of random_rotate (sign flips are self-inverse)."""
    return random_rotate(y, sign_flips)


def scalar_quantize(x: Tensor, codebook: Tensor) -> Tensor:
    """Quantize values to nearest codebook entry, return indices."""
    boundaries = (codebook[:-1] + codebook[1:]) / 2.0
    return torch.bucketize(x.contiguous(), boundaries)


def scalar_dequantize(indices: Tensor, codebook: Tensor) -> Tensor:
    """Look up codebook entries by index."""
    return codebook[indices.long()]


# ---------------------------------------------------------------------------
# Core TurboQuant state -- one per attention layer
# ---------------------------------------------------------------------------


class TurboQuantState:
    """Stores per-layer random state for TurboQuant.

    Create one per attention layer. The rotation matrix is generated
    deterministically from seed + layer_idx.

    For fractional bit-widths (2.5, 3.5), channels are split into
    two groups quantized at different bit-widths.
    """

    def __init__(
        self,
        config: TurboQuantConfig,
        head_size: int,
        layer_idx: int,
        device: torch.device,
    ) -> None:
        self.config = config
        self.head_size = head_size
        self.device = device

        # Deterministic rotation matrix per layer (d x d)
        # Both Pi and PiT must be contiguous for Triton kernel access
        self.Pi = _generate_rotation_matrix(
            head_size, config.seed + layer_idx, device
        ).contiguous()
        self.PiT = self.Pi.T.contiguous()

        # QJL projection matrix
        if config.use_qjl:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(config.seed + layer_idx + 10000)
            self.S = torch.randn(
                head_size, head_size, generator=gen
            ).to(device)
        else:
            self.S = None

        # Setup codebooks for integer or fractional bit-widths
        if config.is_fractional:
            split = config.channel_split
            (hi_bits, hi_ratio), (lo_bits, lo_ratio) = split
            self.hi_bits = hi_bits
            self.lo_bits = lo_bits
            self.hi_channels = int(head_size * hi_ratio)
            self.lo_channels = head_size - self.hi_channels
            self.codebook_hi = _get_codebook(hi_bits, head_size, device)
            self.codebook_lo = _get_codebook(lo_bits, head_size, device)
            self.boundaries_hi = (self.codebook_hi[:-1] + self.codebook_hi[1:]) / 2.0
            self.boundaries_lo = (self.codebook_lo[:-1] + self.codebook_lo[1:]) / 2.0
            self.mse_bits = None  # not used for fractional
        else:
            mse_bits = int(config.bit_width) - 1 if config.use_qjl else int(config.bit_width)
            mse_bits = max(mse_bits, 1)
            self.mse_bits = mse_bits
            self.codebook = _get_codebook(mse_bits, head_size, device)
            self.boundaries = (self.codebook[:-1] + self.codebook[1:]) / 2.0
            self.hi_bits = None

    @torch.no_grad()
    def quantize(self, x: Tensor) -> dict[str, Tensor]:
        """Quantize KV head vectors."""
        orig_shape = x.shape
        flat = x.reshape(-1, self.head_size).float()

        # Extract norms, normalize to unit sphere
        norms = torch.norm(flat, dim=-1, keepdim=True)
        flat_norm = flat / (norms + 1e-8)

        # Rotate
        rotated = flat_norm @ self.PiT

        if self.config.is_fractional:
            return self._quantize_fractional(rotated, norms, orig_shape, x.device)
        else:
            return self._quantize_integer(rotated, norms, orig_shape, x.device)

    def _quantize_integer(self, rotated: Tensor, norms: Tensor,
                          orig_shape: tuple, device: torch.device) -> dict:
        indices = torch.bucketize(rotated.contiguous(), self.boundaries).to(torch.uint8)
        packed = pack_indices(indices.reshape(-1), self.mse_bits)

        result: dict[str, Tensor] = {
            "packed": packed,
            "n_elements": torch.tensor(indices.numel(), device=device),
            "orig_shape": torch.tensor(orig_shape, device=device),
            "norms": norms.squeeze(-1).to(torch.float16).reshape(orig_shape[:-1]),
            "indices": indices,
        }

        if self.config.use_qjl and self.S is not None:
            reconstructed = self.codebook[indices.long()]
            residual_rotated = rotated - reconstructed
            residual = residual_rotated @ self.Pi
            r_norm = torch.norm(residual, dim=-1)
            projected = residual @ self.S.T
            signs = (projected >= 0).to(torch.int8) * 2 - 1
            result["qjl_signs"] = signs.reshape(
                orig_shape[:-1] + (self.head_size,)
            )
            result["qjl_norms"] = r_norm.to(torch.float16).reshape(
                orig_shape[:-1]
            )
        return result

    def _quantize_fractional(self, rotated: Tensor, norms: Tensor,
                             orig_shape: tuple, device: torch.device) -> dict:
        """Split channels and quantize each group at different bit-widths."""
        hi_part = rotated[:, :self.hi_channels]
        lo_part = rotated[:, self.hi_channels:]

        hi_idx = torch.bucketize(hi_part.contiguous(), self.boundaries_hi).to(torch.uint8)
        lo_idx = torch.bucketize(lo_part.contiguous(), self.boundaries_lo).to(torch.uint8)

        hi_packed = pack_indices(hi_idx.reshape(-1), self.hi_bits)
        lo_packed = pack_indices(lo_idx.reshape(-1), self.lo_bits)

        result: dict[str, Tensor] = {
            "hi_packed": hi_packed,
            "lo_packed": lo_packed,
            "hi_n": torch.tensor(hi_idx.numel(), device=device),
            "lo_n": torch.tensor(lo_idx.numel(), device=device),
            "orig_shape": torch.tensor(orig_shape, device=device),
            "norms": norms.squeeze(-1).to(torch.float16).reshape(orig_shape[:-1]),
        }

        # QJL residual correction for fractional bit-widths
        if self.config.use_qjl and self.S is not None:
            hi_recon = self.codebook_hi[hi_idx.long()]
            lo_recon = self.codebook_lo[lo_idx.long()]
            rotated_recon = torch.cat([hi_recon, lo_recon], dim=-1)
            residual_rotated = rotated - rotated_recon
            residual = residual_rotated @ self.Pi
            r_norm = torch.norm(residual, dim=-1)
            projected = residual @ self.S.T
            signs = (projected >= 0).to(torch.int8) * 2 - 1
            result["qjl_signs"] = signs.reshape(
                orig_shape[:-1] + (self.head_size,)
            )
            result["qjl_norms"] = r_norm.to(torch.float16).reshape(
                orig_shape[:-1]
            )

        return result

    @torch.no_grad()
    def dequantize(self, state: dict[str, Tensor]) -> Tensor:
        """Dequantize back to full-precision vectors."""
        orig_shape = tuple(state["orig_shape"].tolist())
        norms = state["norms"].float()
        flat_norms = norms.reshape(-1)

        if self.config.is_fractional:
            x_hat = self._dequantize_fractional(state, flat_norms)
        else:
            x_hat = self._dequantize_integer(state, flat_norms)

        return x_hat.to(torch.bfloat16).reshape(orig_shape)

    def _dequantize_integer(self, state: dict, flat_norms: Tensor) -> Tensor:
        packed = state["packed"]
        n_elements = state["n_elements"].item()

        indices = unpack_indices(packed, self.mse_bits, n_elements)
        flat_indices = indices.reshape(-1, self.head_size).long()

        reconstructed = self.codebook[flat_indices]
        x_hat = reconstructed @ self.Pi  # unrotate

        if (self.config.use_qjl and self.S is not None
                and "qjl_signs" in state):
            signs = state["qjl_signs"].reshape(-1, self.head_size).float()
            r_norms = state["qjl_norms"].float().reshape(-1)
            scale = math.sqrt(math.pi / 2.0) / self.head_size
            qjl_recon = scale * r_norms.unsqueeze(-1) * (signs @ self.S)
            x_hat = x_hat + qjl_recon

        x_hat = x_hat * flat_norms.unsqueeze(-1)
        return x_hat

    def _dequantize_fractional(self, state: dict, flat_norms: Tensor) -> Tensor:
        hi_indices = unpack_indices(
            state["hi_packed"], self.hi_bits, state["hi_n"].item()
        ).reshape(-1, self.hi_channels).long()
        lo_indices = unpack_indices(
            state["lo_packed"], self.lo_bits, state["lo_n"].item()
        ).reshape(-1, self.lo_channels).long()

        hi_recon = self.codebook_hi[hi_indices]
        lo_recon = self.codebook_lo[lo_indices]

        # Concatenate channels and unrotate
        rotated_recon = torch.cat([hi_recon, lo_recon], dim=-1)
        x_hat = rotated_recon @ self.Pi

        # QJL residual correction for fractional bit-widths
        if (self.config.use_qjl and self.S is not None
                and "qjl_signs" in state):
            signs = state["qjl_signs"].reshape(-1, self.head_size).float()
            r_norms = state["qjl_norms"].float().reshape(-1)
            scale = math.sqrt(math.pi / 2.0) / self.head_size
            qjl_recon = scale * r_norms.unsqueeze(-1) * (signs @ self.S)
            x_hat = x_hat + qjl_recon

        x_hat = x_hat * flat_norms.unsqueeze(-1)
        return x_hat


# ---------------------------------------------------------------------------
# Asymmetric attention: compute scores from compressed K, decompress V only.
# Adapted from turboquant-pytorch V2 compressor.
# ---------------------------------------------------------------------------


@torch.no_grad()
def turboquant_asymmetric_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    k_state: TurboQuantState,
    v_state: TurboQuantState,
    scale: float,
) -> Tensor:
    """Compute attention with TurboQuant-compressed K/V.

    1. Compress K (store k_mse fp16 + QJL signs for unbiased scores)
    2. Compress V (store indices + norms, much smaller than bf16)
    3. Delete original K, V
    4. Compute attention scores from compressed K (asymmetric)
    5. Decompress V and apply attention weights

    Args:
        query: (B, S_q, H, D) -- queries
        key: (B, S_kv, H_kv, D) -- keys
        value: (B, S_kv, H_kv, D) -- values
        k_state: TurboQuantState for keys (should have use_qjl=True)
        v_state: TurboQuantState for values (MSE-only)
        scale: attention scale factor (1/sqrt(d))

    Returns:
        (B, S_q, H, D) attention output
    """
    B, S_q, H, D = query.shape
    _, S_kv, H_kv, _ = key.shape
    n_rep = H // H_kv  # GQA repeat factor

    # --- Compress K: store dequantized MSE + QJL metadata ---
    k_comp = k_state.quantize(key)
    k_mse = k_state.dequantize(k_comp).float()  # (B, S_kv, H_kv, D)

    # --- Compress V: only indices + norms (smaller than bf16) ---
    v_comp = v_state.quantize(value)

    # Free original K, V
    del key, value

    # --- Asymmetric attention scores ---
    # Expand KV heads for GQA
    if n_rep > 1:
        k_mse = k_mse.unsqueeze(3).expand(B, S_kv, H_kv, n_rep, D)
        k_mse = k_mse.reshape(B, S_kv, H, D)

    # score = Q @ K_mse^T  (standard part)
    q_f = query.float()  # (B, S_q, H, D)
    scores = torch.einsum("bqhd,bkhd->bhqk", q_f, k_mse) * scale

    # QJL correction for unbiased inner product
    if "qjl_signs" in k_comp and k_state.S is not None:
        signs = k_comp["qjl_signs"].float()  # (B, S_kv, H_kv, D)
        r_norms = k_comp["qjl_norms"].float()  # (B, S_kv, H_kv)
        m = D
        correction_scale = math.sqrt(math.pi / 2.0) / m * scale

        # Project queries through QJL matrix
        q_flat = q_f.reshape(-1, D)
        q_proj = (q_flat @ k_state.S.T).reshape(B, S_q, H, D)

        if n_rep > 1:
            signs = signs.unsqueeze(3).expand(B, S_kv, H_kv, n_rep, D)
            signs = signs.reshape(B, S_kv, H, D)
            r_norms = r_norms.unsqueeze(3).expand(B, S_kv, H_kv, n_rep)
            r_norms = r_norms.reshape(B, S_kv, H)

        # qjl_ip[b,h,q,k] = sum_d(q_proj[b,q,h,d] * signs[b,k,h,d])
        qjl_ip = torch.einsum("bqhd,bkhd->bhqk", q_proj, signs)
        scores = scores + correction_scale * qjl_ip * r_norms.permute(0, 2, 1).unsqueeze(2)

    del k_mse, k_comp

    # --- Softmax ---
    attn_weights = torch.softmax(scores, dim=-1).to(torch.bfloat16)
    del scores

    # --- Decompress V and apply weights ---
    v_deq = v_state.dequantize(v_comp)  # (B, S_kv, H_kv, D)
    del v_comp

    if n_rep > 1:
        v_deq = v_deq.unsqueeze(3).expand(B, S_kv, H_kv, n_rep, D)
        v_deq = v_deq.reshape(B, S_kv, H, D)

    # output[b,q,h,d] = sum_k(attn_weights[b,h,q,k] * v_deq[b,k,h,d])
    output = torch.einsum("bhqk,bkhd->bqhd", attn_weights.float(), v_deq.float())

    return output.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Bit packing: store sub-byte indices in packed uint8 tensors
# ---------------------------------------------------------------------------


def pack_indices(indices: Tensor, bits: int) -> Tensor:
    """Pack b-bit indices into uint8 tensor.

    For 4-bit: 2 values per byte.  For 2-bit: 4 values per byte.
    For 3-bit: pack into uint32 then view as uint8.
    """
    flat = indices.reshape(-1).to(torch.int32)
    n = flat.numel()

    if bits == 4:
        # Nibble packing: 2 values per byte
        pad = (2 - n % 2) % 2
        if pad:
            flat = torch.nn.functional.pad(flat, (0, pad))
        even = flat[0::2].to(torch.uint8)
        odd = flat[1::2].to(torch.uint8)
        packed = even | (odd << 4)
        return packed

    if bits == 2:
        # 4 values per byte
        pad = (4 - n % 4) % 4
        if pad:
            flat = torch.nn.functional.pad(flat, (0, pad))
        packed = (flat[0::4].to(torch.uint8)
                  | (flat[1::4].to(torch.uint8) << 2)
                  | (flat[2::4].to(torch.uint8) << 4)
                  | (flat[3::4].to(torch.uint8) << 6))
        return packed

    if bits == 3:
        # Pack into uint32: 10 values per 32 bits (30 used, 2 wasted)
        group = 10
        pad = (group - n % group) % group
        if pad:
            flat = torch.nn.functional.pad(flat, (0, pad))
        flat = flat.reshape(-1, group)
        packed32 = torch.zeros(flat.shape[0], dtype=torch.int32,
                               device=flat.device)
        for i in range(group):
            packed32 |= (flat[:, i] << (i * 3))
        return packed32.view(torch.uint8)

    if bits == 1:
        # 8 values per byte
        pad = (8 - n % 8) % 8
        if pad:
            flat = torch.nn.functional.pad(flat, (0, pad))
        packed = torch.zeros(flat.numel() // 8, dtype=torch.uint8,
                             device=flat.device)
        for i in range(8):
            packed |= (flat[i::8].to(torch.uint8) << i)
        return packed

    return indices.to(torch.uint8)


def unpack_indices(packed: Tensor, bits: int, n_elements: int) -> Tensor:
    """Unpack bit-packed tensor back to indices."""
    if bits == 4:
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        unpacked = torch.stack([low, high], dim=-1).reshape(-1)
        return unpacked[:n_elements]

    if bits == 2:
        b0 = packed & 0x03
        b1 = (packed >> 2) & 0x03
        b2 = (packed >> 4) & 0x03
        b3 = (packed >> 6) & 0x03
        unpacked = torch.stack([b0, b1, b2, b3], dim=-1).reshape(-1)
        return unpacked[:n_elements]

    if bits == 3:
        packed32 = packed.view(torch.int32)
        parts = []
        for i in range(10):
            parts.append((packed32 >> (i * 3)) & 0x07)
        unpacked = torch.stack(parts, dim=-1).reshape(-1)
        return unpacked[:n_elements]

    if bits == 1:
        parts = []
        for i in range(8):
            parts.append((packed >> i) & 0x01)
        unpacked = torch.stack(parts, dim=-1).reshape(-1)
        return unpacked[:n_elements]

    return packed.long()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def compute_distortion(
    original: Tensor,
    reconstructed: Tensor,
) -> dict[str, float]:
    """Compute MSE and inner-product distortion metrics."""
    mse = (original - reconstructed).pow(2).sum(dim=-1).mean().item()
    d = original.shape[-1]
    ip_orig = (original * original).sum(dim=-1)
    ip_recon = (original * reconstructed).sum(dim=-1)
    ip_distortion = (ip_orig - ip_recon).pow(2).mean().item()

    return {
        "mse": mse,
        "inner_product_distortion": ip_distortion,
        "mse_per_dim": mse / d,
    }


# ---------------------------------------------------------------------------
# vLLM integration: QuantizationConfig + KVCacheMethod
# ---------------------------------------------------------------------------


class TurboQuantVLLMConfig:
    """vLLM quantization config for TurboQuant KV cache compression.

    This is a KV-cache-only quantization method. It does not quantize
    model weights. In pre-dequant mode, K/V tensors are passed through
    a quantize->dequantize cycle before being stored in the standard
    bf16 KV cache, faithfully reproducing TurboQuant's quality impact.
    """

    def __init__(
        self,
        bit_width: float = 3,
        use_qjl: bool = False,
        seed: int = 42,
    ) -> None:
        self.tq_config = TurboQuantConfig(
            bit_width=bit_width,
            use_qjl=use_qjl,
            seed=seed,
        )
        self.packed_modules_mapping: dict[str, list[str]] = {}

    def get_name(self) -> "QuantizationMethods":
        return "turboquant"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TurboQuantVLLMConfig":
        bit_width = config.get("bit_width", 3)
        use_qjl = config.get("use_qjl", False)
        seed = config.get("seed", 42)
        return cls(bit_width=bit_width, use_qjl=use_qjl, seed=seed)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ):
        from vllm.model_executor.layers.attention.attention import Attention
        if isinstance(layer, Attention):
            return TurboQuantKVCacheMethod(self)
        return None

    def get_cache_scale(self, name: str) -> str | None:
        return None

    def apply_vllm_mapper(self, hf_to_vllm_mapper) -> None:
        pass

    def maybe_update_config(self, model_name: str) -> None:
        pass

    def is_mxfp4_quant(self, prefix: str, layer: torch.nn.Module) -> bool:
        return False


class TurboQuantKVCacheMethod:
    """KV cache method for TurboQuant pre-dequant mode.

    Attaches TurboQuantConfig to the Attention layer so that forward()
    can apply the quantize->dequantize cycle.
    """

    def __init__(self, quant_config: TurboQuantVLLMConfig):
        self.quant_config = quant_config
        self.tq_config = quant_config.tq_config

    def create_weights(self, layer: torch.nn.Module):
        from vllm.model_executor.layers.quantization.kv_cache import (
            BaseKVCacheMethod,
        )
        # Create standard k_scale/v_scale params (framework expects them)
        layer.q_scale = torch.nn.Parameter(
            torch.tensor(-1.0), requires_grad=False)
        layer.k_scale = torch.nn.Parameter(
            torch.tensor(-1.0), requires_grad=False)
        layer.v_scale = torch.nn.Parameter(
            torch.tensor(-1.0), requires_grad=False)
        layer.prob_scale = torch.nn.Parameter(
            torch.tensor(-1.0), requires_grad=False)
        # Attach turboquant config for forward() to use
        layer._turboquant_config = self.tq_config

    def apply(self, layer: torch.nn.Module) -> torch.Tensor:
        raise RuntimeError(
            "TurboQuantKVCacheMethod.apply should not be called.")

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not hasattr(layer, "q_scale"):
            return
        # Set all scales to 1.0 (turboquant doesn't use FP8 scales)
        layer._k_scale.fill_(1.0)
        layer._v_scale.fill_(1.0)
        layer._q_scale.fill_(1.0)
        layer._prob_scale.fill_(1.0)
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0
        layer._q_scale_float = 1.0
        # Clean up temporary Parameters
        for attr in ("k_scale", "v_scale", "q_scale", "prob_scale"):
            if hasattr(layer, attr):
                delattr(layer, attr)


def _use_triton_turboquant() -> bool:
    """Check if Triton TurboQuant kernels are available."""
    try:
        import triton  # noqa: F401
        return torch.cuda.is_available()
    except ImportError:
        return False


@torch.no_grad()
def turboquant_pre_dequant(
    key: Tensor,
    value: Tensor,
    tq_k_state: TurboQuantState,
    tq_v_state: TurboQuantState,
) -> tuple[Tensor, Tensor]:
    """Apply TurboQuant quantize->dequantize to K/V before cache storage.

    Uses Triton kernels when available (CUDA + triton installed),
    falls back to Python implementation otherwise.

    Args:
        key: (num_tokens, num_kv_heads, head_size)
        value: (num_tokens, num_kv_heads, head_size)
        tq_k_state: TurboQuantState for keys
        tq_v_state: TurboQuantState for values

    Returns:
        (key_lossy, value_lossy) in original dtype
    """
    orig_dtype = key.dtype

    # Use Triton kernels for non-fractional bit-widths on CUDA
    if (_use_triton_turboquant()
            and not tq_k_state.config.is_fractional
            and not tq_k_state.config.use_qjl
            and key.is_cuda):
        from vllm.v1.attention.ops.triton_turboquant import (
            turboquant_decode,
            turboquant_encode,
        )
        # Encode K
        k_indices, k_norms = turboquant_encode(
            key, tq_k_state.PiT, tq_k_state.codebook, tq_k_state.boundaries)
        k_lossy = turboquant_decode(
            k_indices, k_norms, tq_k_state.Pi, tq_k_state.codebook,
            output_dtype=orig_dtype)
        # Encode V
        v_indices, v_norms = turboquant_encode(
            value, tq_v_state.PiT, tq_v_state.codebook, tq_v_state.boundaries)
        v_lossy = turboquant_decode(
            v_indices, v_norms, tq_v_state.Pi, tq_v_state.codebook,
            output_dtype=orig_dtype)
        return k_lossy, v_lossy

    # Fallback: Python implementation (fractional, QJL, or CPU)
    k_lossy = tq_k_state.dequantize(tq_k_state.quantize(key))
    v_lossy = tq_v_state.dequantize(tq_v_state.quantize(value))
    return k_lossy.to(orig_dtype), v_lossy.to(orig_dtype)
