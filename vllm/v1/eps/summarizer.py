# ABOUTME: JL sketch updates driven by KV writes for EPS.
# ABOUTME: Maintains Gram matrices per (layer, head, group).

from __future__ import annotations

import torch

from .state import EpsJLState


@torch.no_grad()
def jl_update_once(
    state: EpsJLState,
    *,
    layer: int,
    head: int,
    group: int,
    k_vec: torch.Tensor,
) -> None:
    """Update JL summary with a single key vector."""
    z = torch.matmul(state.Phi[head].transpose(0, 1), k_vec)
    state.G[layer, head, group].add_(torch.outer(z, z))
    state.frob2[layer, head, group].add_(torch.dot(k_vec, k_vec))


@torch.no_grad()
def jl_update_block(
    state: EpsJLState,
    *,
    layer: int,
    head: int,
    group: int,
    K_block: torch.Tensor,
) -> None:
    """Update JL summary with a block of key vectors."""
    Y = torch.matmul(K_block, state.Phi[head])
    state.G[layer, head, group].add_(Y.transpose(0, 1) @ Y)
    state.frob2[layer, head, group].add_((K_block * K_block).sum())
