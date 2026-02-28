# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright INL Dynamics / Complexity-ML
"""
INL Dynamics — Inertial Navigation Layer.

PID-like control with velocity tracking for numerical stability.
Core innovation of the Complexity / Pacific-Prime architecture.

    mu(h) = mu_base + mu_proj(h)         # contextual equilibrium
    error = h - mu(h)                     # deviation from equilibrium
    v_next = alpha * v - beta * error     # velocity update (damped)
    h_next = h + dt * gate * v_next       # position update (gated)

Parameters alpha, beta, gate are learned via a small controller MLP
and clamped to stable ranges via sigmoid / softplus.

TP compatibility: INL Dynamics operates on full hidden_size tensors
(attention output is all-reduced before reaching dynamics). Controller
weights are replicated across TP ranks — they are intentionally small
(hidden_size → 64 → hidden_size*3) so replication cost is negligible.

GitHub: https://github.com/Complexity-ML/complexity-deep
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class INLDynamics(nn.Module):
    """
    INL Dynamics — PID-like control with velocity tracking.

    Receives full hidden_size tensors (post all-reduce from attention).
    Controller weights are replicated across TP ranks.
    """

    def __init__(
        self,
        hidden_size: int,
        controller_hidden: int = 64,
        dt: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt

        # Learnable equilibrium
        self.mu = nn.Parameter(torch.zeros(hidden_size))
        self.mu_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.mu_proj.weight)

        # Controller MLP (small — replicated across TP ranks)
        self.controller_in = nn.Linear(hidden_size * 2, controller_hidden)
        self.controller_out = nn.Linear(controller_hidden, hidden_size * 3)

        # Initialize for stability: alpha ≈ 0.9, beta ≈ 0.1, gate ≈ 0.5
        with torch.no_grad():
            bias = self.controller_out.bias
            bias[:hidden_size].fill_(2.2)               # sigmoid(2.2) ≈ 0.9
            bias[hidden_size:hidden_size * 2].fill_(-2.2)  # softplus(-2.2) ≈ 0.1
            bias[hidden_size * 2:].fill_(0.0)            # sigmoid(0) = 0.5
            self.controller_out.weight.normal_(0, 0.01)

    def forward(
        self,
        h: torch.Tensor,
        v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            h: (num_tokens, hidden_size) — hidden states
            v: (num_tokens, hidden_size) — velocity states (None → zeros)

        Returns:
            h_next: updated hidden states
            v_next: updated velocity states
            mu_contextual: contextual equilibrium (for mu-guidance)
        """
        if v is None:
            v = torch.zeros_like(h)

        # Controller: [h, v] → [alpha, beta, gate]
        hv = torch.cat([h, v], dim=-1)
        ctrl = F.silu(self.controller_in(hv))
        ctrl_out = self.controller_out(ctrl)

        alpha_raw, beta_raw, gate_raw = torch.split(
            ctrl_out, self.hidden_size, dim=-1
        )
        alpha = torch.sigmoid(alpha_raw)
        beta = torch.clamp(F.softplus(beta_raw), max=2.0)
        gate = torch.sigmoid(gate_raw)

        # PID update
        mu_contextual = self.mu + self.mu_proj(h)
        error = h - mu_contextual
        v_next = alpha * v - beta * error
        v_next = torch.clamp(v_next, min=-10.0, max=10.0)
        h_next = h + self.dt * gate * v_next

        return h_next, v_next, mu_contextual
