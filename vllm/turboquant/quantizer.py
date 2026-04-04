# SPDX-License-Identifier: Apache-2.0
"""TurboQuant quantizer: rotation + Lloyd-Max scalar quantization.

Pure PyTorch implementation for correctness validation and as fallback.
Triton/CUDA kernels replace this in production.
"""

import math

import torch
import torch.nn as nn

from vllm.turboquant.centroids import get_centroids
from vllm.turboquant.config import TurboQuantConfig


def generate_rotation_matrix(
    d: int, seed: int, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Generate Haar-distributed random orthogonal matrix via QR decomposition."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen, device="cpu", dtype=torch.float32)
    Q, R = torch.linalg.qr(G)
    # Fix sign ambiguity for determinism
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device)


class TurboQuantizer(nn.Module):
    """TurboQuant quantizer with PolarQuant MSE stage.

    For each KV vector x:
      1. Normalize: x_hat = x / ||x||
      2. Rotate: y = Pi @ x_hat
      3. Scalar quantize: idx[j] = nearest(y[j], centroids)
      4. Reconstruct: x_mse = Pi^T @ centroids[idx]
      5. Residual norm: gamma = ||x_hat - x_mse||

    Storage: indices (mse_bits*d bits) + vec_norm (fp16) + res_norm (fp16)
    """

    def __init__(self, config: TurboQuantConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        d = config.head_dim
        seed = config.seed + layer_idx * 1337

        # Rotation matrix Pi (orthogonal, d x d)
        self.register_buffer(
            "Pi", generate_rotation_matrix(d, seed=seed).float()
        )
        # Lloyd-Max centroids for the MSE stage
        centroids = get_centroids(d, config.mse_bits)
        self.register_buffer("centroids", centroids.float())

        # Precomputed transpose for efficiency
        self.register_buffer("PiT", self.Pi.T.contiguous())

    @torch.no_grad()
    def quantize(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Quantize vectors using TurboQuant (PolarQuant MSE).

        Args:
            x: Input vectors (..., head_dim) in any shape.

        Returns:
            Dict with:
                mse_indices: (..., head_dim) uint8 indices
                vec_norm: (...,) float16 vector norms
                res_norm: (...,) float16 residual norms
        """
        orig_shape = x.shape
        d = self.config.head_dim
        flat = x.reshape(-1, d).float()

        # 1. Normalize
        vec_norm = torch.norm(flat, dim=-1, keepdim=True)  # (N, 1)
        x_hat = flat / (vec_norm + 1e-8)

        # 2. Rotate
        rotated = x_hat @ self.Pi.T  # (N, d)

        # 3. Scalar quantize to nearest centroid
        diffs = rotated.unsqueeze(-1) - self.centroids  # (N, d, n_centroids)
        indices = diffs.abs().argmin(dim=-1).to(torch.uint8)  # (N, d)

        # 4. Reconstruct MSE approximation
        reconstructed_rotated = self.centroids[indices.long()]  # (N, d)
        x_mse = reconstructed_rotated @ self.Pi  # (N, d) back to original space

        # 5. Residual norm
        residual = x_hat - x_mse
        res_norm = torch.norm(residual, dim=-1, keepdim=True)  # (N, 1)

        # Reshape back
        batch_shape = orig_shape[:-1]
        return {
            "mse_indices": indices.reshape(*batch_shape, d),
            "vec_norm": vec_norm.squeeze(-1).half().reshape(batch_shape),
            "res_norm": res_norm.squeeze(-1).half().reshape(batch_shape),
        }

    @torch.no_grad()
    def dequantize(self, compressed: dict[str, torch.Tensor]) -> torch.Tensor:
        """Dequantize MSE component (for reconstruction/validation).

        Returns TurboQuant reconstruction (MSE only, no QJL).
        """
        d = self.config.head_dim
        indices = compressed["mse_indices"]
        vec_norm = compressed["vec_norm"]

        orig_shape = indices.shape
        flat_idx = indices.reshape(-1, d).long()
        flat_vec_norm = vec_norm.reshape(-1).float().unsqueeze(-1)

        # MSE reconstruction
        reconstructed_rotated = self.centroids[flat_idx]  # (N, d)
        if self.config.norm_correction:
            rot_norms = reconstructed_rotated.norm(
                dim=-1, keepdim=True).clamp(min=1e-8)
            reconstructed_rotated = reconstructed_rotated / rot_norms
        x_mse = reconstructed_rotated @ self.Pi  # (N, d)

        # Rescale by original norm
        x_recon = flat_vec_norm * x_mse
        return x_recon.reshape(orig_shape)

    # -- Standalone benchmark helpers (not used in vLLM inference path) --

    @torch.no_grad()
    def attention_scores(
        self,
        queries: torch.Tensor,
        compressed: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute attention scores <Q, K> from compressed K.

        Uses the asymmetric MSE estimator:
          <q, k> ~ norm * <q_rot, c[idx]>

        where q_rot = q @ Pi^T is precomputed per query.

        Args:
            queries: (batch, heads, seq_q, head_dim)
            compressed: Dict from quantize() with keys shaped
                        (batch, heads, seq_k, head_dim)

        Returns:
            scores: (batch, heads, seq_q, seq_k)
        """
        indices = compressed["mse_indices"]  # (B, H, S_k, D)
        vec_norm = compressed["vec_norm"]  # (B, H, S_k)

        # Precompute rotated queries (once per query token)
        q_float = queries.float()
        q_rot = q_float @ self.PiT     # = q @ Pi^T, shape (B, H, S_q, D)

        # Term 1: <q, Pi^T @ c[idx]> = sum_j q_rot[j] * centroids[idx[j]]
        k_mse_rotated = self.centroids[indices.long()]  # (B, H, S_k, D)
        term1 = torch.matmul(q_rot, k_mse_rotated.transpose(-2, -1))

        # Combine with vector norms
        scores = vec_norm.float().unsqueeze(-2) * term1

        return scores

    def pack_cache(
        self, compressed: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Pack compressed state into contiguous uint8 cache tensor.

        Layout per token per head (for tq3, head_dim=128):
          [0:48]   MSE indices (128 coords * 3 bits / 8 = 48 bytes)
          [48:50]  vec_norm    (float16 = 2 bytes)
          [50:52]  res_norm    (float16 = 2 bytes)
          Total: 52 bytes
        """
        d = self.config.head_dim
        mse_bits = self.config.mse_bits
        indices = compressed["mse_indices"]  # (..., d) uint8
        vec_norm = compressed["vec_norm"]     # (...) float16
        res_norm = compressed["res_norm"]     # (...) float16

        batch_shape = indices.shape[:-1]

        # Pack MSE indices (mse_bits per coordinate)
        flat_idx = indices.reshape(-1, d)
        if mse_bits == 1:
            packed_mse = self._pack_bits(flat_idx, 1)
        elif mse_bits == 2:
            packed_mse = self._pack_bits(flat_idx, 2)
        elif mse_bits == 3:
            packed_mse = self._pack_bits(flat_idx, 3)
        else:
            packed_mse = flat_idx

        # Pack norms as raw float16 bytes
        vec_norm_bytes = vec_norm.reshape(-1).half().view(torch.uint8).reshape(-1, 2)
        res_norm_bytes = res_norm.reshape(-1).half().view(torch.uint8).reshape(-1, 2)

        # Concatenate
        packed = torch.cat([packed_mse, vec_norm_bytes, res_norm_bytes],
                           dim=-1)
        return packed.reshape(*batch_shape, -1)

    def unpack_cache(self, packed: torch.Tensor) -> dict[str, torch.Tensor]:
        """Unpack uint8 cache tensor back to compressed state."""
        d = self.config.head_dim
        mse_bits = self.config.mse_bits

        batch_shape = packed.shape[:-1]
        flat = packed.reshape(-1, packed.shape[-1])

        mse_bytes = math.ceil(d * mse_bits / 8)

        # Split
        packed_mse = flat[:, :mse_bytes]
        vec_norm_bytes = flat[:, mse_bytes:mse_bytes + 2]
        res_norm_bytes = flat[:, mse_bytes + 2:mse_bytes + 4]

        # Unpack MSE indices
        indices = self._unpack_bits(packed_mse, mse_bits, d)

        # Unpack norms
        vec_norm = vec_norm_bytes.view(torch.float16).reshape(-1)
        res_norm = res_norm_bytes.view(torch.float16).reshape(-1)

        return {
            "mse_indices": indices.reshape(*batch_shape, d),
            "vec_norm": vec_norm.reshape(batch_shape),
            "res_norm": res_norm.reshape(batch_shape),
        }

    @staticmethod
    def _pack_bits(data: torch.Tensor, bits_per_elem: int) -> torch.Tensor:
        """Pack integer data into uint8 bytes."""
        N, D = data.shape
        if bits_per_elem == 1:
            padded_D = ((D + 7) // 8) * 8
            padded = torch.zeros(N, padded_D, dtype=torch.uint8, device=data.device)
            padded[:, :D] = data
            padded = padded.reshape(N, -1, 8)
            shifts = torch.arange(8, device=data.device, dtype=torch.uint8)
            return (padded << shifts).sum(dim=-1, dtype=torch.uint8)
        elif bits_per_elem == 2:
            padded_D = ((D + 3) // 4) * 4
            padded = torch.zeros(N, padded_D, dtype=torch.uint8, device=data.device)
            padded[:, :D] = data
            padded = padded.reshape(N, -1, 4)
            shifts = torch.arange(0, 8, 2, device=data.device, dtype=torch.uint8)
            return (padded << shifts).sum(dim=-1, dtype=torch.uint8)
        elif bits_per_elem == 3:
            total_bits = D * 3
            total_bytes = (total_bits + 7) // 8
            result = torch.zeros(N, total_bytes, dtype=torch.uint8, device=data.device)
            for i in range(D):
                bit_offset = i * 3
                byte_idx = bit_offset // 8
                bit_idx = bit_offset % 8
                val = data[:, i].to(torch.int32)
                result[:, byte_idx] |= ((val << bit_idx) & 0xFF).to(torch.uint8)
                if bit_idx > 5:  # overflow into next byte
                    result[:, byte_idx + 1] |= (
                        (val >> (8 - bit_idx)) & 0xFF
                    ).to(torch.uint8)
            return result
        else:
            return data  # bits >= 8, no packing needed

    @staticmethod
    def _unpack_bits(
        packed: torch.Tensor, bits_per_elem: int, n_elems: int
    ) -> torch.Tensor:
        """Unpack uint8 bytes into integer data."""
        N = packed.shape[0]
        mask = (1 << bits_per_elem) - 1

        if bits_per_elem == 1:
            expanded = packed.unsqueeze(-1).expand(-1, -1, 8)
            shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
            unpacked = ((expanded >> shifts) & 1).reshape(N, -1)
            return unpacked[:, :n_elems]
        elif bits_per_elem == 2:
            expanded = packed.unsqueeze(-1).expand(-1, -1, 4)
            shifts = torch.arange(0, 8, 2, device=packed.device, dtype=torch.uint8)
            unpacked = ((expanded >> shifts) & mask).reshape(N, -1)
            return unpacked[:, :n_elems].to(torch.uint8)
        elif bits_per_elem == 3:
            result = torch.zeros(N, n_elems, dtype=torch.uint8, device=packed.device)
            for i in range(n_elems):
                bit_offset = i * 3
                byte_idx = bit_offset // 8
                bit_idx = bit_offset % 8
                val = packed[:, byte_idx].to(torch.int32) >> bit_idx
                if bit_idx > 5 and byte_idx + 1 < packed.shape[1]:
                    val |= packed[:, byte_idx + 1].to(torch.int32) << (8 - bit_idx)
                result[:, i] = (val & mask).to(torch.uint8)
            return result
        else:
            return packed[:, :n_elems]
