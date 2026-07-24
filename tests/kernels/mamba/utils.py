# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def selective_state_update_ref(
    state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False
):
    """
    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, dim) or (batch, nheads, dim)
        dt: (batch, dim) or (batch, nheads, dim)
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, dstate) or (batch, ngroups, dstate)
        C: (batch, dstate) or (batch, ngroups, dstate)
        D: (dim,) or (nheads, dim)
        z: (batch, dim) or (batch, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
    Return:
        out: (batch, dim) or (batch, nheads, dim)
    """
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    batch, nheads, dim, dstate = state.shape
    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
        dt = dt + dt_bias
    dt = F.softplus(dt) if dt_softplus else dt
    dA = torch.exp(
        rearrange(dt, "b h d -> b h d 1") * A
    )  # (batch, nheads, dim, dstate)
    B = repeat(B, "b g n -> b (g h) n", h=nheads // ngroups)  # (batch, nheads, dstate)
    C = repeat(C, "b g n -> b (g h) n", h=nheads // ngroups)  # (batch, nheads, dstate)
    dB = rearrange(dt, "b h d -> b h d 1") * rearrange(
        B, "b h n -> b h 1 n"
    )  # (batch, nheads, dim, dstate)
    state.copy_(
        state * dA + dB * rearrange(x, "b h d -> b h d 1")
    )  # (batch, dim, dstate
    out = torch.einsum("bhdn,bhn->bhd", state.to(C.dtype), C)
    if D is not None:
        out += (x * D).to(out.dtype)
    out = (out if z is None else out * F.silu(z)).to(x.dtype)
    if not has_heads:
        out = out.squeeze(1)
    return out


def selective_state_update_replayssm_state_and_output_ref(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    x_cache: torch.Tensor | None = None,
    dt_cache: torch.Tensor | None = None,
    B_cache: torch.Tensor | None = None,
    write_pos: torch.Tensor | None = None,
    max_cache_len: int = 16,
) -> torch.Tensor:
    """Pure-PyTorch cached-dot reference for validation."""
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)

    batch, nheads, dim, dstate = state.shape
    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)

    assert x_cache is not None
    assert dt_cache is not None
    assert B_cache is not None
    assert x_cache.shape == (batch, nheads, max_cache_len, dim)
    assert dt_cache.shape == (batch, nheads, max_cache_len)
    assert B_cache.shape == (batch, ngroups, max_cache_len, dstate)
    assert write_pos is not None
    assert write_pos.shape == (batch,) and write_pos.dtype == torch.int32

    ratio = nheads // ngroups

    dt_val = dt[:, :, 0].float()
    if dt_bias is not None:
        dt_val = dt_val + dt_bias[:, 0].float()
    if dt_softplus:
        dt_val = F.softplus(dt_val)
    A_val = A[:, 0, 0].float()
    C_heads = C.repeat_interleave(ratio, dim=1)
    out = torch.empty(batch, nheads, dim, device=x.device, dtype=torch.float32)

    for b in range(batch):
        cache_len = int(write_pos[b].item())
        is_flush = cache_len == max_cache_len - 1
        n_steps = cache_len + 1

        dt_all = torch.zeros(nheads, n_steps, device=x.device, dtype=torch.float32)
        if cache_len > 0:
            dt_all[:, :cache_len] = dt_cache[b, :, :cache_len]
        dt_all[:, cache_len] = dt_val[b]

        cumsum = torch.cumsum(dt_all, dim=-1)
        total = cumsum[:, -1]
        dA_cumsum = A_val[:, None] * cumsum
        dA_total = A_val * total
        total_decay = torch.exp(dA_total)
        scale = dt_all * torch.exp(dA_total[:, None] - dA_cumsum)

        x_all = torch.zeros(nheads, dim, n_steps, device=x.device, dtype=x.dtype)
        if cache_len > 0:
            x_all[..., :cache_len] = x_cache[b, :, :cache_len, :].permute(0, 2, 1)
        x_all[..., cache_len] = x[b]

        B_all = torch.zeros(ngroups, n_steps, dstate, device=B.device, dtype=B.dtype)
        if cache_len > 0:
            B_all[:, :cache_len, :] = B_cache[b, :, :cache_len, :]
        B_all[:, cache_len, :] = B[b]

        B_heads = B_all.repeat_interleave(ratio, dim=0)
        x_scaled = x_all.float() * scale[:, None, :]
        delta = torch.einsum("hdk,hkn->hdn", x_scaled, B_heads.float())
        state_new = state[b].float() * total_decay[:, None, None] + delta
        if is_flush:
            state[b].copy_(state_new.to(state.dtype))
        else:
            x_cache[b, :, cache_len, :] = x[b]
            dt_cache[b, :, cache_len] = dt_val[b]
            B_cache[b, :, cache_len, :] = B[b]

        out[b] = torch.einsum("hdn,hn->hd", state_new, C_heads[b].float())
    if D is not None:
        out = out + (x.float() * D[None]).to(out.dtype)
    if z is not None:
        out = out * F.silu(z.float())
    out = out.to(x.dtype)
    if not has_heads:
        out = out.squeeze(1)
    return out


def selective_state_update_replayssm_output_only_ref(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    x_cache: torch.Tensor | None = None,
    dt_cache: torch.Tensor | None = None,
    B_cache: torch.Tensor | None = None,
    write_pos: torch.Tensor | None = None,
    max_cache_len: int = 16,
) -> torch.Tensor:
    """Pure-PyTorch cached-bc reference for validation."""
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)

    batch, nheads, dim, dstate = state.shape
    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape

    ratio = nheads // ngroups

    dt_val = dt[:, :, 0].float()
    if dt_bias is not None:
        dt_val = dt_val + dt_bias[:, 0].float()
    if dt_softplus:
        dt_val = F.softplus(dt_val)
    A_val = A[:, 0, 0].float()
    C_heads = C.repeat_interleave(ratio, dim=1)
    out = torch.empty(batch, nheads, dim, device=x.device, dtype=torch.float32)

    assert x_cache is not None
    assert dt_cache is not None
    assert B_cache is not None
    assert write_pos is not None

    for b in range(batch):
        cache_len = int(write_pos[b].item())
        is_flush = cache_len == max_cache_len - 1
        n_steps = cache_len + 1

        dt_all = torch.zeros(nheads, n_steps, device=x.device, dtype=torch.float32)
        if cache_len > 0:
            dt_all[:, :cache_len] = dt_cache[b, :, :cache_len]
        dt_all[:, cache_len] = dt_val[b]

        cumsum = torch.cumsum(dt_all, dim=-1)
        total = cumsum[:, -1]
        dA_cumsum = A_val[:, None] * cumsum
        dA_total = A_val * total
        total_decay = torch.exp(dA_total)
        scale = dt_all * torch.exp(dA_total[:, None] - dA_cumsum)

        x_all = torch.zeros(nheads, dim, n_steps, device=x.device, dtype=x.dtype)
        if cache_len > 0:
            x_all[..., :cache_len] = x_cache[b, :, :cache_len, :].permute(0, 2, 1)
        x_all[..., cache_len] = x[b]

        B_all = torch.zeros(ngroups, n_steps, dstate, device=B.device, dtype=B.dtype)
        if cache_len > 0:
            B_all[:, :cache_len, :] = B_cache[b, :, :cache_len, :]
        B_all[:, cache_len, :] = B[b]

        B_heads = B_all.repeat_interleave(ratio, dim=0)
        C_heads_b = C_heads[b]

        if is_flush:
            x_scaled = x_all.float() * scale[:, None, :]
            delta = torch.einsum("hdk,hkn->hdn", x_scaled, B_heads.float())
            state_new = state[b].float() * total_decay[:, None, None] + delta
            state[b].copy_(state_new.to(state.dtype))
            out[b] = torch.einsum("hdn,hn->hd", state_new, C_heads_b.float())
        else:
            checkpoint_out = torch.einsum(
                "hdn,hn->hd", state[b].float(), C_heads_b.float()
            )
            checkpoint_out = checkpoint_out * total_decay[:, None]
            BC = torch.einsum("hkn,hn->hk", B_heads.float(), C_heads_b.float())
            cache_out = torch.einsum("hdk,hk->hd", x_all.float(), scale * BC)
            out[b] = checkpoint_out + cache_out
            x_cache[b, :, cache_len, :] = x[b]
            dt_cache[b, :, cache_len] = dt_val[b]
            B_cache[b, :, cache_len, :] = B[b]

    if D is not None:
        out = out + (x.float() * D[None]).to(out.dtype)
    if z is not None:
        out = out * F.silu(z.float())
    out = out.to(x.dtype)
    if not has_heads:
        out = out.squeeze(1)
    return out


def allocate_update_caches(
    batch: int,
    nheads: int,
    ngroups: int,
    dim: int,
    dstate: int,
    max_cache_len: int,
    device: torch.device,
    x_dtype: torch.dtype,
    B_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate dense reference caches for standalone validation."""
    x_cache = torch.zeros(
        batch,
        nheads,
        max_cache_len,
        dim,
        device=device,
        dtype=x_dtype,
    )
    dt_cache = torch.zeros(
        batch,
        nheads,
        max_cache_len,
        device=device,
        dtype=torch.float32,
    )
    B_cache = torch.zeros(
        batch,
        ngroups,
        max_cache_len,
        dstate,
        device=device,
        dtype=B_dtype,
    )
    write_pos = torch.zeros(batch, dtype=torch.int32, device=device)
    return x_cache, dt_cache, B_cache, write_pos
