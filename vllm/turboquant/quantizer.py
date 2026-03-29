# SPDX-License-Identifier: Apache-2.0
"""TurboQuant quantizer: rotation + Lloyd-Max + QJL residual correction.

Pure PyTorch implementation for correctness validation and as fallback.
Triton/CUDA kernels replace this in production.
"""

import math
from typing import Optional

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


def generate_qjl_matrix(
    d: int, seed: int, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Generate i.i.d. N(0,1) projection matrix for QJL."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    S = torch.randn(d, d, generator=gen, device="cpu", dtype=torch.float32)
    return S.to(device)


class TurboQuantizer(nn.Module):
    """TurboQuant quantizer with MSE stage + QJL residual correction.

    For each KV vector x:
      1. Normalize: x_hat = x / ||x||
      2. Rotate: y = Pi @ x_hat
      3. Scalar quantize: idx[j] = nearest(y[j], centroids)
      4. Reconstruct: x_mse = Pi^T @ centroids[idx]
      5. Residual: r = x_hat - x_mse, gamma = ||r||
      6. QJL: signs = sign(S @ r)

    Storage: indices ((b-1)*d bits) + signs (d bits) + norm (fp16) + gamma (fp16)
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
        # QJL projection matrix S (random Gaussian, d x d)
        self.register_buffer(
            "S", generate_qjl_matrix(d, seed=seed + 1).float()
        )
        # Lloyd-Max centroids for the MSE stage
        centroids = get_centroids(d, config.mse_bits)
        self.register_buffer("centroids", centroids.float())

        # Precomputed transposes for efficiency
        self.register_buffer("PiT", self.Pi.T.contiguous())
        self.register_buffer("ST", self.S.T.contiguous())

    @torch.no_grad()
    def quantize(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Quantize vectors using TurboQuant.

        Args:
            x: Input vectors (..., head_dim) in any shape.

        Returns:
            Dict with:
                mse_indices: (..., head_dim) uint8 indices
                qjl_signs: (..., head_dim) int8 signs {-1, +1}
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

        # 5. Residual
        residual = x_hat - x_mse
        res_norm = torch.norm(residual, dim=-1, keepdim=True)  # (N, 1)

        # 6. QJL: project residual and take sign
        projected = residual @ self.S.T  # (N, d)
        signs = torch.where(projected >= 0,
                            torch.ones_like(projected, dtype=torch.int8),
                            -torch.ones_like(projected, dtype=torch.int8))

        # Reshape back
        batch_shape = orig_shape[:-1]
        return {
            "mse_indices": indices.reshape(*batch_shape, d),
            "qjl_signs": signs.reshape(*batch_shape, d),
            "vec_norm": vec_norm.squeeze(-1).half().reshape(batch_shape),
            "res_norm": res_norm.squeeze(-1).half().reshape(batch_shape),
        }

    @torch.no_grad()
    def dequantize(self, compressed: dict[str, torch.Tensor]) -> torch.Tensor:
        """Dequantize MSE component (for reconstruction/validation).

        Returns full TurboQuant reconstruction including QJL correction.
        """
        d = self.config.head_dim
        indices = compressed["mse_indices"]
        signs = compressed["qjl_signs"]
        vec_norm = compressed["vec_norm"]
        res_norm = compressed["res_norm"]

        orig_shape = indices.shape
        flat_idx = indices.reshape(-1, d).long()
        flat_signs = signs.reshape(-1, d).float()
        flat_vec_norm = vec_norm.reshape(-1).float().unsqueeze(-1)
        flat_res_norm = res_norm.reshape(-1).float().unsqueeze(-1)

        # MSE reconstruction
        reconstructed_rotated = self.centroids[flat_idx]  # (N, d)
        x_mse = reconstructed_rotated @ self.Pi  # (N, d)

        # QJL correction
        m = self.S.shape[0]
        correction_scale = math.sqrt(math.pi / 2) / m
        x_qjl = correction_scale * flat_res_norm * (flat_signs @ self.S)  # (N, d)

        # Combine and rescale
        x_recon = flat_vec_norm * (x_mse + x_qjl)
        return x_recon.reshape(orig_shape)

    @torch.no_grad()
    def attention_scores(
        self,
        queries: torch.Tensor,
        compressed: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute unbiased attention scores <Q, K> from compressed K.

        Uses the asymmetric estimator:
          <q, k> ~ norm * (<q_rot, c[idx]> + gamma * sqrt(pi/2)/m * <q_proj, signs>)

        where q_rot = q @ Pi^T and q_proj = q @ S^T are precomputed per query.

        Args:
            queries: (batch, heads, seq_q, head_dim)
            compressed: Dict from quantize() with keys shaped
                        (batch, heads, seq_k, head_dim)

        Returns:
            scores: (batch, heads, seq_q, seq_k)
        """
        d = self.config.head_dim
        indices = compressed["mse_indices"]  # (B, H, S_k, D)
        signs = compressed["qjl_signs"]  # (B, H, S_k, D)
        vec_norm = compressed["vec_norm"]  # (B, H, S_k)
        res_norm = compressed["res_norm"]  # (B, H, S_k)

        # Precompute rotated/projected queries (once per query token)
        q_float = queries.float()
        q_rot = q_float @ self.PiT     # = q @ Pi^T, shape (B, H, S_q, D)
        q_proj = q_float @ self.ST      # = q @ S^T,  shape (B, H, S_q, D)

        # Term 1: <q, Pi^T @ c[idx]> = sum_j q_rot[j] * centroids[idx[j]]
        k_mse_rotated = self.centroids[indices.long()]  # (B, H, S_k, D)
        term1 = torch.matmul(q_rot, k_mse_rotated.transpose(-2, -1))  # (B,H,S_q,S_k)

        # Term 2: QJL correction
        signs_float = signs.float()
        qjl_ip = torch.matmul(q_proj, signs_float.transpose(-2, -1))  # (B,H,S_q,S_k)

        m = d  # QJL dimension = head_dim
        correction_scale = math.sqrt(math.pi / 2) / m
        term2 = correction_scale * qjl_ip * res_norm.float().unsqueeze(-2)

        # Combine with vector norms
        scores = vec_norm.float().unsqueeze(-2) * (term1 + term2)

        return scores

    def pack_cache(
        self, compressed: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Pack compressed state into contiguous uint8 cache tensor.

        Layout per token per head (for tq3, head_dim=128):
          [0:32]   MSE indices (128 coords * 2 bits / 8 = 32 bytes)
          [32:48]  QJL signs   (128 coords * 1 bit / 8 = 16 bytes)
          [48:50]  vec_norm    (float16 = 2 bytes)
          [50:52]  res_norm    (float16 = 2 bytes)
          Total: 52 bytes
        """
        d = self.config.head_dim
        mse_bits = self.config.mse_bits
        indices = compressed["mse_indices"]  # (..., d) uint8
        signs = compressed["qjl_signs"]      # (..., d) int8
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

        # Pack QJL signs (1 bit per coord, 8 per byte)
        flat_signs = ((signs.reshape(-1, d) + 1) // 2).to(torch.uint8)  # 0 or 1
        packed_signs = self._pack_bits(flat_signs, 1)

        # Pack norms as raw float16 bytes
        vec_norm_bytes = vec_norm.reshape(-1).half().view(torch.uint8).reshape(-1, 2)
        res_norm_bytes = res_norm.reshape(-1).half().view(torch.uint8).reshape(-1, 2)

        # Concatenate
        packed = torch.cat([packed_mse, packed_signs, vec_norm_bytes, res_norm_bytes],
                           dim=-1)
        return packed.reshape(*batch_shape, -1)

    def unpack_cache(self, packed: torch.Tensor) -> dict[str, torch.Tensor]:
        """Unpack uint8 cache tensor back to compressed state."""
        d = self.config.head_dim
        mse_bits = self.config.mse_bits

        batch_shape = packed.shape[:-1]
        flat = packed.reshape(-1, packed.shape[-1])

        mse_bytes = math.ceil(d * mse_bits / 8)
        qjl_bytes = math.ceil(d / 8)

        # Split
        packed_mse = flat[:, :mse_bytes]
        packed_signs = flat[:, mse_bytes:mse_bytes + qjl_bytes]
        vec_norm_bytes = flat[:, mse_bytes + qjl_bytes:mse_bytes + qjl_bytes + 2]
        res_norm_bytes = flat[:, mse_bytes + qjl_bytes + 2:mse_bytes + qjl_bytes + 4]

        # Unpack MSE indices
        indices = self._unpack_bits(packed_mse, mse_bits, d)

        # Unpack QJL signs
        signs_01 = self._unpack_bits(packed_signs, 1, d)
        signs = (signs_01.to(torch.int8) * 2 - 1)  # 0,1 -> -1,+1

        # Unpack norms
        vec_norm = vec_norm_bytes.view(torch.float16).reshape(-1)
        res_norm = res_norm_bytes.view(torch.float16).reshape(-1)

        return {
            "mse_indices": indices.reshape(*batch_shape, d),
            "qjl_signs": signs.reshape(*batch_shape, d),
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
