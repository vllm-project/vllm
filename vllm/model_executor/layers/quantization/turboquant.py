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
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.layers.quantization.kv_cache import (
    BaseKVCacheMethod,
)
from vllm.utils.math_utils import next_power_of_2

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
        -2.1520,
        -1.3440,
        -0.7560,
        -0.2451,
        0.2451,
        0.7560,
        1.3440,
        2.1520,
    ],
    4: [
        -2.7326,
        -2.0690,
        -1.6180,
        -1.2562,
        -0.9424,
        -0.6568,
        -0.3880,
        -0.1284,
        0.1284,
        0.3880,
        0.6568,
        0.9424,
        1.2562,
        1.6180,
        2.0690,
        2.7326,
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

    Outlier-aware mode: channels listed in ``outlier_channels`` are kept
    at full precision (bf16) and excluded from rotation/quantization.
    Alternatively, set ``outlier_fraction`` > 0 to auto-detect outliers
    via per-channel variance calibration during ``TurboQuantState`` init.
    """

    bit_width: float = 3
    value_bit_width: float | None = None
    use_qjl: bool = False
    seed: int = 42
    outlier_channels: list[int] | None = None
    outlier_fraction: float = 0.0
    lite_mode: bool = False

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
        if self.outlier_fraction < 0.0 or self.outlier_fraction >= 1.0:
            raise ValueError(
                f"outlier_fraction must be in [0, 1), got {self.outlier_fraction}"
            )
        if self.outlier_channels is not None and self.outlier_fraction > 0:
            raise ValueError(
                "Specify either outlier_channels or outlier_fraction, not both"
            )
        if self.lite_mode and self.use_qjl:
            raise ValueError(
                "lite_mode and use_qjl are mutually exclusive — "
                "TQ_LITE skips rotation and QJL entirely"
            )
        if self.value_bit_width is not None and self.value_bit_width not in valid:
            raise ValueError(
                f"value_bit_width must be one of {sorted(valid)}, "
                f"got {self.value_bit_width}"
            )

    @property
    def effective_value_bit_width(self) -> float:
        """Return the bit-width used for value quantization."""
        if self.value_bit_width is not None:
            return self.value_bit_width
        return self.bit_width

    @property
    def has_outliers(self) -> bool:
        return self.outlier_channels is not None or self.outlier_fraction > 0

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
# Fast Walsh-Hadamard Transform (FWHT) with random sign flips.
#
# Replaces the O(d²) random rotation matrix with O(d log d) FWHT.
# The paper proves Hadamard + random sign flips is a valid random
# rotation for TurboQuant (coordinates become approximately i.i.d.
# Gaussian after the transform).
# ---------------------------------------------------------------------------


def _generate_sign_flips(d: int, seed: int, device: torch.device) -> Tensor:
    """Generate random ±1 sign flips, deterministic from seed."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    signs = (
        torch.randint(0, 2, (d,), generator=gen, device="cpu", dtype=torch.float32) * 2
        - 1
    )
    return signs.to(device)


def _hadamard_transform(x: Tensor) -> Tensor:
    """Fast Walsh-Hadamard Transform (FWHT), PyTorch fallback.

    Operates on the last dimension. Requires last dim to be power of 2.
    O(d log d) compute. Fused Triton kernel used on GPU.
    """
    d = x.shape[-1]
    h = 1
    while h < d:
        # Butterfly: add/subtract pairs at distance h
        x_even = x[..., 0 :: 2 * h].clone()
        x_odd = x[..., h :: 2 * h].clone()
        x[..., 0 :: 2 * h] = x_even + x_odd
        x[..., h :: 2 * h] = x_even - x_odd
        h *= 2
    # Normalize to make it orthogonal
    x = x / math.sqrt(d)
    return x


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
        self.layer_idx = layer_idx
        self.device = device

        # --- Outlier channel setup ---
        # outlier_idx: sorted tensor of channel indices kept at bf16
        # normal_idx: sorted tensor of channel indices to quantize
        if config.outlier_channels is not None:
            self.outlier_idx = torch.tensor(
                sorted(config.outlier_channels),
                dtype=torch.long,
                device=device,
            )
        elif config.outlier_fraction > 0:
            n_outliers = max(1, int(head_size * config.outlier_fraction))
            # Placeholder — will be replaced by calibrate_outliers()
            self.outlier_idx = torch.arange(
                n_outliers,
                dtype=torch.long,
                device=device,
            )
        else:
            self.outlier_idx = None

        if self.outlier_idx is not None:
            all_idx = torch.arange(head_size, dtype=torch.long, device=device)
            outlier_mask = torch.zeros(head_size, dtype=torch.bool, device=device)
            outlier_mask[self.outlier_idx] = True
            self.normal_idx = all_idx[~outlier_mask]
            self.normal_size = self.normal_idx.shape[0]
        else:
            self.normal_idx = None
            self.normal_size = head_size

        # Lite mode: skip rotation (no Hadamard, no sign flips)
        if config.lite_mode:
            self.use_hadamard = False
            self._hadamard_d = self.normal_size
            self._pad_size = 0
            self.sign_flips = None
            self.Pi = None
            self.PiT = None
            self.S = None

            bw = int(config.bit_width)
            self.mse_bits = bw
            # Scale codebook by 1/sqrt(head_size) directly — no padding
            self.codebook = _get_codebook(bw, self.normal_size, device)
            self.boundaries = (self.codebook[:-1] + self.codebook[1:]) / 2.0
            self.hi_bits = None
        else:
            # Rotation: use Hadamard + sign flips (O(d log d)) when dimension
            # is power of 2, fall back to QR rotation matrix (O(d²)) otherwise.
            self._init_rotation(config.seed + layer_idx, device)

            # QJL projection matrix (also only for normal channels)
            if config.use_qjl:
                gen = torch.Generator(device="cpu")
                gen.manual_seed(config.seed + layer_idx + 10000)
                self.S = torch.randn(
                    self.normal_size,
                    self.normal_size,
                    generator=gen,
                    device="cpu",
                ).to(device)
            else:
                self.S = None

            # Setup codebooks for integer or fractional bit-widths
            # Codebook scaling uses normal_size (the dimension being quantized)
            if config.is_fractional:
                split = config.channel_split
                (hi_bits, hi_ratio), (lo_bits, lo_ratio) = split
                self.hi_bits = hi_bits
                self.lo_bits = lo_bits
                self.hi_channels = int(self.normal_size * hi_ratio)
                self.lo_channels = self.normal_size - self.hi_channels
                self.codebook_hi = _get_codebook(hi_bits, self.normal_size, device)
                self.codebook_lo = _get_codebook(lo_bits, self.normal_size, device)
                self.boundaries_hi = (
                    self.codebook_hi[:-1] + self.codebook_hi[1:]
                ) / 2.0
                self.boundaries_lo = (
                    self.codebook_lo[:-1] + self.codebook_lo[1:]
                ) / 2.0
                self.mse_bits = None  # not used for fractional
            else:
                bw = int(config.bit_width)
                mse_bits = bw - 1 if config.use_qjl else bw
                mse_bits = max(mse_bits, 1)
                self.mse_bits = mse_bits
                # Scale codebook by 1/sqrt(hadamard_d), not
                # 1/sqrt(normal_size), because the Hadamard pads to
                # hadamard_d and output has std ≈ 1/sqrt(hadamard_d).
                self.codebook = _get_codebook(mse_bits, self._hadamard_d, device)
                self.boundaries = (self.codebook[:-1] + self.codebook[1:]) / 2.0
                self.hi_bits = None

    def _init_rotation(self, seed: int, device: torch.device) -> None:
        """Initialize rotation: Hadamard (O(d log d)) with padding."""
        d = self.normal_size
        # Always use Hadamard — pad to next power of 2 if needed
        self.use_hadamard = True
        self._hadamard_d = next_power_of_2(d)
        self._pad_size = self._hadamard_d - d
        self.sign_flips = _generate_sign_flips(self._hadamard_d, seed, device)
        self.Pi = None
        self.PiT = None

    def rotate(self, x: Tensor) -> Tensor:
        """Apply rotation: sign flip → pad → Hadamard → trim."""
        if self._pad_size > 0:
            x = torch.nn.functional.pad(x, (0, self._pad_size), value=0.0)
        x = x * self.sign_flips
        x = _hadamard_transform(x)
        if self._pad_size > 0:
            x = x[..., : self.normal_size]
        return x

    def unrotate(self, x: Tensor) -> Tensor:
        """Inverse rotation: pad → Hadamard → sign flip → trim."""
        if self._pad_size > 0:
            x = torch.nn.functional.pad(x, (0, self._pad_size), value=0.0)
        x = _hadamard_transform(x)
        x = x * self.sign_flips
        if self._pad_size > 0:
            x = x[..., : self.normal_size]
        return x

    def calibrate_outliers(
        self,
        calibration_data: Tensor,
        n_outliers: int | None = None,
    ) -> None:
        """Identify outlier channels from calibration data.

        Args:
            calibration_data: (num_samples, head_size) — K or V vectors
                from a few forward passes.
            n_outliers: Number of outlier channels to select. If None,
                uses ``int(head_size * config.outlier_fraction)``.
        """
        if n_outliers is None:
            n_outliers = max(1, int(self.head_size * self.config.outlier_fraction))

        flat = calibration_data.reshape(-1, self.head_size).float()
        # Per-channel variance: high-variance channels are outliers
        var = flat.var(dim=0)  # (head_size,)
        _, top_idx = var.topk(n_outliers)
        self.outlier_idx = top_idx.sort().values.to(self.device)

        all_idx = torch.arange(self.head_size, dtype=torch.long, device=self.device)
        outlier_mask = torch.zeros(self.head_size, dtype=torch.bool, device=self.device)
        outlier_mask[self.outlier_idx] = True
        self.normal_idx = all_idx[~outlier_mask]
        new_normal_size = self.normal_idx.shape[0]

        # Regenerate rotation if normal_size changed
        if new_normal_size != self.normal_size:
            self.normal_size = new_normal_size
            self._init_rotation(self.config.seed + self.layer_idx, self.device)
            # Regenerate codebooks for new dimension
            if not self.config.is_fractional and self.mse_bits is not None:
                self.codebook = _get_codebook(
                    self.mse_bits, self._hadamard_d, self.device
                )
                self.boundaries = (self.codebook[:-1] + self.codebook[1:]) / 2.0

    @torch.no_grad()
    def quantize(self, x: Tensor) -> dict[str, Tensor]:
        """Quantize KV head vectors."""
        orig_shape = x.shape
        flat = x.reshape(-1, self.head_size).float()

        # Split outlier / normal channels
        if self.outlier_idx is not None:
            outlier_vals = flat[:, self.outlier_idx]  # kept as-is
            flat_normal = flat[:, self.normal_idx]
        else:
            outlier_vals = None
            flat_normal = flat

        # Extract norms of normal channels, normalize to unit sphere
        norms = torch.norm(flat_normal, dim=-1, keepdim=True)
        flat_norm = flat_normal / (norms + 1e-8)

        # Rotate (normal channels only)
        rotated = self.rotate(flat_norm)

        if self.config.is_fractional:
            result = self._quantize_fractional(rotated, norms, orig_shape, x.device)
        else:
            result = self._quantize_integer(rotated, norms, orig_shape, x.device)

        if outlier_vals is not None:
            result["outlier_vals"] = outlier_vals.to(torch.bfloat16)

        return result

    def _quantize_integer(
        self, rotated: Tensor, norms: Tensor, orig_shape: tuple, device: torch.device
    ) -> dict:
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
            residual = self.unrotate(residual_rotated)
            r_norm = torch.norm(residual, dim=-1)
            projected = residual @ self.S.T
            signs = (projected >= 0).to(torch.int8) * 2 - 1
            result["qjl_signs"] = signs.reshape(orig_shape[:-1] + (self.normal_size,))
            result["qjl_norms"] = r_norm.to(torch.float16).reshape(orig_shape[:-1])
        return result

    def _quantize_fractional(
        self, rotated: Tensor, norms: Tensor, orig_shape: tuple, device: torch.device
    ) -> dict:
        """Split channels and quantize each group at different bit-widths."""
        hi_part = rotated[:, : self.hi_channels]
        lo_part = rotated[:, self.hi_channels :]

        hi_idx = torch.bucketize(hi_part.contiguous(), self.boundaries_hi).to(
            torch.uint8
        )
        lo_idx = torch.bucketize(lo_part.contiguous(), self.boundaries_lo).to(
            torch.uint8
        )

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
            residual = self.unrotate(residual_rotated)
            r_norm = torch.norm(residual, dim=-1)
            projected = residual @ self.S.T
            signs = (projected >= 0).to(torch.int8) * 2 - 1
            result["qjl_signs"] = signs.reshape(orig_shape[:-1] + (self.normal_size,))
            result["qjl_norms"] = r_norm.to(torch.float16).reshape(orig_shape[:-1])

        return result

    @torch.no_grad()
    def dequantize(self, state: dict[str, Tensor]) -> Tensor:
        """Dequantize back to full-precision vectors."""
        orig_shape = tuple(state["orig_shape"].tolist())
        norms = state["norms"].float()
        flat_norms = norms.reshape(-1)

        if self.config.is_fractional:
            x_hat_normal = self._dequantize_fractional(state, flat_norms)
        else:
            x_hat_normal = self._dequantize_integer(state, flat_norms)

        # Reassemble outlier + normal channels
        if self.outlier_idx is not None and "outlier_vals" in state:
            n_vectors = x_hat_normal.shape[0]
            x_hat = torch.empty(
                n_vectors,
                self.head_size,
                dtype=torch.float32,
                device=x_hat_normal.device,
            )
            x_hat[:, self.normal_idx] = x_hat_normal
            x_hat[:, self.outlier_idx] = state["outlier_vals"].float()
        else:
            x_hat = x_hat_normal

        return x_hat.to(torch.bfloat16).reshape(orig_shape)

    def _dequantize_integer(self, state: dict, flat_norms: Tensor) -> Tensor:
        packed = state["packed"]
        n_elements = state["n_elements"].item()

        indices = unpack_indices(packed, self.mse_bits, n_elements)
        flat_indices = indices.reshape(-1, self.normal_size).long()

        reconstructed = self.codebook[flat_indices]
        x_hat = self.unrotate(reconstructed)

        if self.config.use_qjl and self.S is not None and "qjl_signs" in state:
            signs = state["qjl_signs"].reshape(-1, self.normal_size).float()
            r_norms = state["qjl_norms"].float().reshape(-1)
            scale = math.sqrt(math.pi / 2.0) / self.normal_size
            qjl_recon = scale * r_norms.unsqueeze(-1) * (signs @ self.S)
            x_hat = x_hat + qjl_recon

        x_hat = x_hat * flat_norms.unsqueeze(-1)
        return x_hat

    def _dequantize_fractional(self, state: dict, flat_norms: Tensor) -> Tensor:
        hi_indices = (
            unpack_indices(state["hi_packed"], self.hi_bits, state["hi_n"].item())
            .reshape(-1, self.hi_channels)
            .long()
        )
        lo_indices = (
            unpack_indices(state["lo_packed"], self.lo_bits, state["lo_n"].item())
            .reshape(-1, self.lo_channels)
            .long()
        )

        hi_recon = self.codebook_hi[hi_indices]
        lo_recon = self.codebook_lo[lo_indices]

        # Concatenate channels and unrotate
        rotated_recon = torch.cat([hi_recon, lo_recon], dim=-1)
        x_hat = self.unrotate(rotated_recon)

        # QJL residual correction for fractional bit-widths
        if self.config.use_qjl and self.S is not None and "qjl_signs" in state:
            signs = state["qjl_signs"].reshape(-1, self.normal_size).float()
            r_norms = state["qjl_norms"].float().reshape(-1)
            scale = math.sqrt(math.pi / 2.0) / self.normal_size
            qjl_recon = scale * r_norms.unsqueeze(-1) * (signs @ self.S)
            x_hat = x_hat + qjl_recon

        x_hat = x_hat * flat_norms.unsqueeze(-1)
        return x_hat


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


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
        packed = (
            flat[0::4].to(torch.uint8)
            | (flat[1::4].to(torch.uint8) << 2)
            | (flat[2::4].to(torch.uint8) << 4)
            | (flat[3::4].to(torch.uint8) << 6)
        )
        return packed

    if bits == 3:
        # Pack into uint32: 10 values per 32 bits (30 used, 2 wasted)
        group = 10
        pad = (group - n % group) % group
        if pad:
            flat = torch.nn.functional.pad(flat, (0, pad))
        flat = flat.reshape(-1, group)
        packed32 = torch.zeros(flat.shape[0], dtype=torch.int32, device=flat.device)
        for i in range(group):
            packed32 |= flat[:, i] << (i * 3)
        return packed32.view(torch.uint8)

    if bits == 1:
        # 8 values per byte
        pad = (8 - n % 8) % 8
        if pad:
            flat = torch.nn.functional.pad(flat, (0, pad))
        packed = torch.zeros(flat.numel() // 8, dtype=torch.uint8, device=flat.device)
        for i in range(8):
            packed |= flat[i::8].to(torch.uint8) << i
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


class TurboQuantVLLMConfig(QuantizationConfig):
    """vLLM quantization config for TurboQuant KV cache compression.

    This is a KV-cache-only quantization method. It does not quantize
    model weights.
    """

    def __init__(
        self,
        bit_width: float = 3,
        use_qjl: bool = False,
        seed: int = 42,
        outlier_channels: list[int] | None = None,
        outlier_fraction: float = 0.0,
        lite_mode: bool = False,
    ) -> None:
        super().__init__()
        self.tq_config = TurboQuantConfig(
            bit_width=bit_width,
            use_qjl=use_qjl,
            seed=seed,
            outlier_channels=outlier_channels,
            outlier_fraction=outlier_fraction,
            lite_mode=lite_mode,
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant):
        return None

    def get_name(self) -> QuantizationMethods:
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
    def from_config(cls, config: dict[str, Any]) -> TurboQuantVLLMConfig:
        import os

        bit_width = config.get("bit_width", 3)
        use_qjl = config.get("use_qjl", False)
        seed = config.get("seed", 42)
        outlier_channels = config.get("outlier_channels")
        outlier_fraction = config.get("outlier_fraction", 0.0)
        lite_mode = config.get(
            "lite_mode",
            os.environ.get("TQ_LITE", "0") in ("1", "true", "True"),
        )
        return cls(
            bit_width=bit_width,
            use_qjl=use_qjl,
            seed=seed,
            outlier_channels=outlier_channels,
            outlier_fraction=outlier_fraction,
            lite_mode=lite_mode,
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from vllm.model_executor.layers.attention.attention import Attention

        if isinstance(layer, Attention):
            return TurboQuantKVCacheMethod(self)
        return None


class TurboQuantKVCacheMethod(BaseKVCacheMethod):
    """KV cache method for TurboQuant.

    Attaches TurboQuantConfig to the Attention layer so that forward()
    can apply the quantize->dequantize cycle.
    """

    def __init__(self, quant_config: TurboQuantVLLMConfig):
        super().__init__(quant_config)
        self.tq_config = quant_config.tq_config

    def create_weights(self, layer: torch.nn.Module):
        super().create_weights(layer)
        # Attach turboquant config for forward() to use
        layer._turboquant_config = self.tq_config

    def apply(self, layer: torch.nn.Module) -> torch.Tensor:
        raise RuntimeError("TurboQuantKVCacheMethod.apply should not be called.")

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


def _triton_pre_dequant_one(
    x: Tensor,
    state: TurboQuantState,
    orig_dtype: torch.dtype,
) -> Tensor:
    """Encode->decode for a single K or V tensor, outlier-aware.

    Uses Triton kernels for QR rotation, PyTorch for Hadamard rotation.
    """
    if state.outlier_idx is not None:
        normal_x = x[..., state.normal_idx]
        outlier_x = x[..., state.outlier_idx]

        normal_lossy = _encode_decode_normal(normal_x.contiguous(), state, orig_dtype)

        out = torch.empty_like(x, dtype=orig_dtype)
        out[..., state.normal_idx] = normal_lossy
        out[..., state.outlier_idx] = outlier_x.to(orig_dtype)
        return out
    else:
        return _encode_decode_normal(x, state, orig_dtype)


def _encode_decode_normal(
    x: Tensor,
    state: TurboQuantState,
    orig_dtype: torch.dtype,
) -> Tensor:
    """Encode->decode normal channels using fused Hadamard Triton kernels."""
    if x.is_cuda:
        try:
            from vllm.v1.attention.ops.triton_hadamard_turboquant import (
                hadamard_turboquant_decode,
                hadamard_turboquant_encode,
            )

            indices, norms = hadamard_turboquant_encode(
                x.float() if x.dtype != torch.float32 else x,
                state.sign_flips,
                state.codebook,
                state.boundaries,
            )
            return hadamard_turboquant_decode(
                indices,
                norms,
                state.sign_flips,
                state.codebook,
                output_dtype=orig_dtype,
            )
        except Exception:
            pass  # Fall through to PyTorch path

    # PyTorch fallback
    orig_shape = x.shape
    flat = x.reshape(-1, state.normal_size).float()
    norms = torch.norm(flat, dim=-1, keepdim=True)
    flat_norm = flat / (norms + 1e-8)
    rotated = state.rotate(flat_norm)
    indices = torch.bucketize(rotated.contiguous(), state.boundaries).to(torch.uint8)
    reconstructed = state.codebook[indices.long()]
    x_hat = state.unrotate(reconstructed) * norms
    return x_hat.reshape(orig_shape).to(orig_dtype)


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

    When outlier channels are configured, those channels are kept at
    full precision while the remaining channels go through the
    rotate->quantize->dequantize->unrotate pipeline.

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
    if (
        _use_triton_turboquant()
        and not tq_k_state.config.is_fractional
        and not tq_k_state.config.use_qjl
        and key.is_cuda
    ):
        k_lossy = _triton_pre_dequant_one(key, tq_k_state, orig_dtype)
        v_lossy = _triton_pre_dequant_one(value, tq_v_state, orig_dtype)
        return k_lossy, v_lossy

    # Fallback: Python implementation (fractional, QJL, or CPU)
    k_lossy = tq_k_state.dequantize(tq_k_state.quantize(key))
    v_lossy = tq_v_state.dequantize(tq_v_state.quantize(value))
    return k_lossy.to(orig_dtype), v_lossy.to(orig_dtype)
