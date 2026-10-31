# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-PyTorch references for a tree verify of linear-attention layers.

A verify request carries ``T`` rows and ``parents[t]``: the row index of row
``t``'s parent, -1 for row 0, every other row's parent an earlier row. A
chain is ``parents[t] = t - 1``. Each row's conv taps and recurrent state come
from its ancestors instead of the rows before it, so a row's output equals a
plain chain run over its root-to-row path. One request at a time, float32.
"""

import torch
import torch.nn.functional as F

from tests.utils_tree import path_to


def chain_parents(num_rows: int) -> torch.Tensor:
    return torch.arange(-1, num_rows - 1, dtype=torch.int32)


def conv_tree_ref(
    x: torch.Tensor,
    history: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    parents: torch.Tensor,
    silu: bool = True,
) -> torch.Tensor:
    """Causal depthwise conv of a tree.

    x: [T, D] the rows' inputs; history: [D, W - 1] the conv window before the
    rows, oldest first; weight: [D, W]. Row ``t``'s taps are ``t`` and its
    ``W - 1`` nearest ancestors, then the history's newest columns.
    """
    num_rows = x.shape[0]
    width = weight.shape[1]
    out = torch.empty(num_rows, x.shape[1], dtype=torch.float32)
    for row in range(num_rows):
        taps = path_to(parents, row)
        values = [history[:, h].float() for h in range(width - 1)]
        values += [x[r].float() for r in taps]
        acc = torch.zeros(x.shape[1]) if bias is None else bias.float().clone()
        for j, value in enumerate(values[-width:]):
            acc += weight[:, j].float() * value
        out[row] = F.silu(acc) if silu else acc
    return out


def gdn_step(
    h: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One gated-delta-rule step with the sigmoid gating and in-kernel qk
    L2 norm of ``fused_sigmoid_gating_delta_rule_update``.

    h: [HV, V, K]; q, k: [H, K]; v: [HV, V]; a, b: [HV]. Returns (out [HV, V],
    new h).
    """
    num_v_heads = v.shape[0]
    group = num_v_heads // q.shape[0]
    q = q.float().repeat_interleave(group, dim=0)
    k = k.float().repeat_interleave(group, dim=0)
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * scale
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    g = -torch.exp(A_log.float()) * F.softplus(a.float() + dt_bias.float())
    beta = torch.sigmoid(b.float())
    h = h * torch.exp(g)[:, None, None]
    v_new = (v.float() - torch.einsum("hvk,hk->hv", h, k)) * beta[:, None]
    h = h + v_new[:, :, None] * k[:, None, :]
    return torch.einsum("hvk,hk->hv", h, q), h


def gdn_tree_ref(q, k, v, a, b, A_log, dt_bias, h0, parents, scale):
    """Tree gated delta rule: row ``t`` starts from its parent's state (``h0``
    for row 0). Returns (out [T, HV, V], states [T, HV, V, K])."""
    outs: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    for row in range(q.shape[0]):
        parent = int(parents[row])
        h = h0.float() if parent < 0 else states[parent]
        out, h = gdn_step(
            h, q[row], k[row], v[row], a[row], b[row], A_log, dt_bias, scale
        )
        outs.append(out)
        states.append(h)
    return torch.stack(outs), torch.stack(states)


def ssu_step(state, x, dt, A, B, C, D, dt_bias):
    """One Mamba2 selective state update (softplus dt, no z).

    state: [nheads, dim, dstate]; x, dt: [nheads, dim]; A: [nheads, dim,
    dstate]; B, C: [ngroups, dstate]; D, dt_bias: [nheads, dim].
    """
    nheads = state.shape[0]
    group = nheads // B.shape[0]
    dt = F.softplus(dt.float() + dt_bias.float())
    B = B.float().repeat_interleave(group, dim=0)
    C = C.float().repeat_interleave(group, dim=0)
    state = state * torch.exp(dt[..., None] * A.float()) + (
        dt[..., None] * B[:, None, :] * x.float()[..., None]
    )
    out = torch.einsum("hdn,hn->hd", state, C) + x.float() * D.float()
    return out, state


def ssu_tree_ref(x, dt, A, B, C, D, dt_bias, state0, parents):
    """Tree selective state update: row ``t`` starts from its parent's state
    (``state0`` for row 0). Returns (out [T, nheads, dim], states)."""
    outs: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    for row in range(x.shape[0]):
        parent = int(parents[row])
        state = state0.float() if parent < 0 else states[parent]
        out, state = ssu_step(state, x[row], dt[row], A, B[row], C[row], D, dt_bias)
        outs.append(out)
        states.append(state)
    return torch.stack(outs), torch.stack(states)
