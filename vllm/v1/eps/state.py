# ABOUTME: JL sketch state tailored per request for EPS.
# ABOUTME: Holds per-head projections and Gram matrices for fast bounds.

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EpsJLState:
    """Per-request Johnson-Lindenstrauss sketch state."""

    Phi: torch.Tensor  # [H, d, m] projections (float32)
    G: torch.Tensor  # [L, H, P, m, m] Gram summaries (float32)
    frob2: torch.Tensor  # [L, H, P] Frobenius energy per group (float32)

    @classmethod
    def init(
        cls,
        *,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        num_groups: int,
        sketch_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> "EpsJLState":
        phi = torch.randint(
            0, 2, (num_heads, head_dim, sketch_dim), device=device, dtype=torch.int8
        )
        phi = (phi * 2 - 1).to(dtype)
        G = torch.zeros(
            (num_layers, num_heads, num_groups, sketch_dim, sketch_dim),
            device=device,
            dtype=dtype,
        )
        frob2 = torch.zeros(
            (num_layers, num_heads, num_groups),
            device=device,
            dtype=dtype,
        )
        return cls(Phi=phi, G=G, frob2=frob2)

    def group_id_from_block(self, block_id: int, group_blocks: int) -> int:
        return block_id // group_blocks

    def ensure_group_capacity(self, required_groups: int) -> None:
        """Grow Gram and Frobenius buffers for newly observed groups."""
        current = self.G.shape[2]
        if required_groups <= current:
            return

        device = self.G.device
        dtype = self.G.dtype
        sketch_dim = self.G.shape[3]
        num_layers, num_heads = self.G.shape[:2]

        new_shape = (num_layers, num_heads, required_groups, sketch_dim, sketch_dim)
        expanded_G = torch.zeros(new_shape, device=device, dtype=dtype)
        expanded_G[:, :, :current].copy_(self.G)
        self.G = expanded_G

        frob_shape = (num_layers, num_heads, required_groups)
        expanded_frob2 = torch.zeros(
            frob_shape, device=self.frob2.device, dtype=self.frob2.dtype
        )
        expanded_frob2[:, :, :current].copy_(self.frob2)
        self.frob2 = expanded_frob2
