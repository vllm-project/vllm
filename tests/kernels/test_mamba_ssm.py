import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_scan_fn, selective_state_update)


def selective_state_update_ref(state,
                               x,
                               dt,
                               A,
                               B,
                               C,
                               D=None,
                               z=None,
                               dt_bias=None,
                               dt_softplus=False):
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
    dA = torch.exp(rearrange(dt, "b h d -> b h d 1") *
                   A)  # (batch, nheads, dim, dstate)
    B = repeat(B, "b g n -> b (g h) n",
               h=nheads // ngroups)  # (batch, nheads, dstate)
    C = repeat(C, "b g n -> b (g h) n",
               h=nheads // ngroups)  # (batch, nheads, dstate)
    dB = rearrange(dt, "b h d -> b h d 1") * rearrange(
        B, "b h n -> b h 1 n")  # (batch, nheads, dim, dstate)
    state.copy_(state * dA +
                dB * rearrange(x, "b h d -> b h d 1"))  # (batch, dim, dstate
    out = torch.einsum("bhdn,bhn->bhd", state.to(C.dtype), C)
    if D is not None:
        out += (x * D).to(out.dtype)
    out = (out if z is None else out * F.silu(z)).to(x.dtype)
    if not has_heads:
        out = out.squeeze(1)
    return out


def selective_scan_ref(u,
                       delta,
                       A,
                       B,
                       C,
                       D=None,
                       z=None,
                       delta_bias=None,
                       delta_softplus=False,
                       return_last_state=False,
                       position_indices=None,
                       prev_state=None):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    prev_state: r(B D N), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    B = B.float()
    C = C.float()
    x = A.new_zeros((batch, dim, dstate)) if prev_state is None else prev_state
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        if position_indices is not None and position_indices[0, i] == 0:
            x = deltaB_u[:, :, i]
        else:
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        ys.append(y)
    y = torch.stack(ys, dim=2)  # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


@pytest.mark.parametrize('wtype', [torch.float32])
@pytest.mark.parametrize('itype', [torch.float32])
@pytest.mark.parametrize('seqlen', [128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("return_last_state", [True])
@pytest.mark.parametrize('has_delta_bias', [True])
@pytest.mark.parametrize('delta_softplus', [True])
@pytest.mark.parametrize('has_z', [True])
@pytest.mark.parametrize('has_D', [True])
@pytest.mark.parametrize("varBC_groups", [1, 2])
@pytest.mark.parametrize("is_variable_C", [True])
@pytest.mark.parametrize("is_variable_B", [True])
@pytest.mark.parametrize("scan_chunks", [1, 2, 3])
def test_selective_scan(is_variable_B, is_variable_C, varBC_groups, has_D,
                        has_z, has_delta_bias, delta_softplus,
                        return_last_state, seqlen, itype, wtype, scan_chunks):
    if varBC_groups > 1 and (not is_variable_B or not is_variable_C):
        pytest.skip()  # This config is not applicable
    device = 'cuda'
    rtol, atol = (6e-4, 2e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    if has_z:  # If we have z, the errors on the weights seem higher
        rtolw = max(rtolw, rtol)
        atolw = max(atolw, atol)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    dim = 4
    dstate = 8
    A = (-0.5 * torch.rand(dim, dstate, device=device, dtype=wtype))
    if not is_variable_B:
        B_shape = [dim, dstate]
    elif varBC_groups == 1:
        B_shape = [batch_size, dstate, seqlen]
    else:
        B_shape = [batch_size, varBC_groups, dstate, seqlen]
    B = torch.randn(B_shape,
                    device=device,
                    dtype=wtype if not is_variable_B else itype)
    if not is_variable_C:
        C_shape = [dim, dstate]
    elif varBC_groups == 1:
        C_shape = [batch_size, dstate, seqlen]
    else:
        C_shape = [batch_size, varBC_groups, dstate, seqlen]
    C = torch.randn(C_shape,
                    device=device,
                    dtype=wtype if not is_variable_C else itype)
    D = torch.randn(dim, device=device, dtype=torch.float32) if has_D else None
    z = torch.randn(batch_size, dim, seqlen, device=device,
                    dtype=itype) if has_z else None
    delta_bias = (0.5 * torch.rand(dim, device=device, dtype=torch.float32)
                  ) if has_delta_bias else None
    u = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype)
    delta = (0.5 *
             torch.rand(batch_size, dim, seqlen, device=device, dtype=itype))
    state = None
    state_ref = None
    out = None
    out_ref = None
    outs = []
    for c in range(scan_chunks):
        chunked_prompt_len = seqlen // scan_chunks
        chunk_start = chunked_prompt_len * c
        chunk_end = chunked_prompt_len * (c + 1)
        if c == scan_chunks - 1:
            chunk_end = seqlen
        _B = B
        if is_variable_B:
            _B = B[..., chunk_start:chunk_end]
        _C = C
        if is_variable_B:
            _C = C[..., chunk_start:chunk_end]
        _z = z
        if has_z:
            assert z is not None
            _z = z[..., chunk_start:chunk_end]
        out, *rest = selective_scan_fn(u[..., chunk_start:chunk_end],
                                       delta[..., chunk_start:chunk_end],
                                       A,
                                       _B,
                                       _C,
                                       D,
                                       z=_z,
                                       delta_bias=delta_bias,
                                       delta_softplus=delta_softplus,
                                       return_last_state=return_last_state,
                                       prev_state=state if c > 0 else None)
        outs.append(out)
        if return_last_state:
            state = rest[0]
    if len(outs) > 1:
        out = torch.cat(outs, dim=-1)
    out_ref, *rest = selective_scan_ref(u,
                                        delta,
                                        A,
                                        B,
                                        C,
                                        D,
                                        z=z,
                                        delta_bias=delta_bias,
                                        delta_softplus=delta_softplus,
                                        return_last_state=return_last_state)
    if return_last_state:
        state_ref = rest[0]

    assert out is not None and out_ref is not None
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)
    if return_last_state:
        assert state is not None and state_ref is not None
        assert torch.allclose(state, state_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype",
                         [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_z", [False, True])
@pytest.mark.parametrize("dstate", [16, 32, 64])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
def test_selective_state_update(dim, dstate, has_z, itype):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (5e-3, 1e-2)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
        if torch.version.hip:
            atol *= 2
    # set seed
    torch.random.manual_seed(0)
    batch_size = 1
    state = torch.randn(batch_size, dim, dstate, dtype=itype, device=device)
    x = torch.randn(batch_size, dim, device=device, dtype=itype)
    dt = torch.randn(batch_size, dim, device=device, dtype=itype)
    dt_bias = torch.rand(dim, device=device) - 4.0
    A = -torch.rand(dim, dstate, device=device) - 1.0
    B = torch.randn(batch_size, dstate, device=device)
    C = torch.randn(batch_size, dstate, device=device)
    D = torch.randn(dim, device=device)
    z = torch.randn_like(x) if has_z else None
    state_ref = state.detach().clone()
    out = selective_state_update(state,
                                 x,
                                 dt,
                                 A,
                                 B,
                                 C,
                                 D=D,
                                 z=z,
                                 dt_bias=dt_bias,
                                 dt_softplus=True)
    out_ref = selective_state_update_ref(state_ref,
                                         x,
                                         dt,
                                         A,
                                         B,
                                         C,
                                         D=D,
                                         z=z,
                                         dt_bias=dt_bias,
                                         dt_softplus=True)

    assert torch.allclose(state, state_ref, rtol=rtol, atol=atol)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)
