# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from tests.kernels.utils import opcheck
from vllm import _custom_ops as ops  # noqa: F401
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_scan_fn,
    selective_state_update,
)
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.utils import PAD_SLOT_ID


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


def selective_scan_ref(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
    prev_state=None,
    final_state_out=None,
):
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
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum("bdn,dn->bd", x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
            else:
                y = torch.einsum("bdn,bdn->bd", x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            if final_state_out is None:
                final_state_out = x
            else:
                final_state_out.copy_(x)
        ys.append(y)
    y = torch.stack(ys, dim=2)  # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, final_state_out)


def selective_scan_opcheck_fn(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    cu_seq_len=None,
    cache_indices=None,
    has_initial_state=None,
    ssm_states=None,
    pad_slot_id=PAD_SLOT_ID,
    block_size=2048,
    block_idx_first_scheduled_token=None,
    block_idx_last_scheduled_token=None,
    initial_state_idx=None,
):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate).
    """
    if u.stride(-1) != 1:
        u = u.contiguous()
    if delta.stride(-1) != 1:
        delta = delta.contiguous()
    if D is not None:
        D = D.contiguous()
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if z is not None and z.stride(-1) != 1:
        z = z.contiguous()
    if B.dim() == 3 and cu_seq_len is None:
        B = B.unsqueeze(1)
    if B.dim() == 2 and cu_seq_len is not None:
        B = B.unsqueeze(0)
    if C.dim() == 3 and cu_seq_len is None:
        C = C.unsqueeze(1)
    if C.dim() == 2 and cu_seq_len is not None:
        C = C.unsqueeze(0)

    # Disable test_autograd_registration for now as it seems to trigger
    # a bogus error.
    opcheck(
        torch.ops._C.selective_scan_fwd,
        (
            u,
            delta,
            A,
            B,
            C,
            D,
            z,
            delta_bias,
            delta_softplus,
            cu_seq_len,
            cache_indices,
            has_initial_state,
            ssm_states,
            pad_slot_id,
            block_size,
            block_idx_first_scheduled_token,
            block_idx_last_scheduled_token,
            initial_state_idx,
        ),
        test_utils=["test_schema", "test_faketensor"],
    )


@pytest.mark.parametrize("wtype", [torch.float32])
@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seqlen", [128, 1024, 4096])
@pytest.mark.parametrize("has_delta_bias", [True])
@pytest.mark.parametrize("delta_softplus", [True])
@pytest.mark.parametrize("has_z", [True])
@pytest.mark.parametrize("has_D", [True])
@pytest.mark.parametrize("varBC_groups", [1, 2])
@pytest.mark.parametrize("is_variable_C", [True])
@pytest.mark.parametrize("is_variable_B", [True])
@pytest.mark.parametrize("scan_chunks", [1, 3])
def test_selective_scan(
    is_variable_B,
    is_variable_C,
    varBC_groups,
    has_D,
    has_z,
    has_delta_bias,
    delta_softplus,
    seqlen,
    itype,
    wtype,
    scan_chunks,
):
    if varBC_groups > 1 and (not is_variable_B or not is_variable_C):
        pytest.skip()  # This config is not applicable
    device = "cuda"
    rtol, atol = (6e-4, 2e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    if has_z:  # If we have z, the errors on the weights seem higher
        rtolw = max(rtolw, rtol)
        atolw = max(atolw, atol)
    # set seed
    set_random_seed(0)
    batch_size = 1
    dim = 4
    dstate = 8
    A = -0.5 * torch.rand(dim, dstate, device=device, dtype=wtype)
    A_ref = A.clone()
    if not is_variable_B:
        B_shape = [dim, dstate]
    elif varBC_groups == 1:
        B_shape = [batch_size, dstate, seqlen]
    else:
        B_shape = [batch_size, varBC_groups, dstate, seqlen]
    B = torch.randn(B_shape, device=device, dtype=wtype if not is_variable_B else itype)
    B_ref = B.clone()
    if not is_variable_C:
        C_shape = [dim, dstate]
    elif varBC_groups == 1:
        C_shape = [batch_size, dstate, seqlen]
    else:
        C_shape = [batch_size, varBC_groups, dstate, seqlen]
    C = torch.randn(C_shape, device=device, dtype=wtype if not is_variable_C else itype)
    C_ref = C.clone()
    D = torch.randn(dim, device=device, dtype=torch.float32) if has_D else None
    D_ref = D.clone()
    z = (
        torch.randn(batch_size, dim, seqlen, device=device, dtype=itype)
        if has_z
        else None
    )
    z_ref = z.clone() if has_z else None
    delta_bias = (
        (0.5 * torch.rand(dim, device=device, dtype=torch.float32))
        if has_delta_bias
        else None
    )
    u = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype)
    u_ref = u.clone()
    delta = 0.5 * torch.rand(batch_size, dim, seqlen, device=device, dtype=itype)
    delta_ref = delta.clone()
    state_shape = (batch_size, u.shape[1], int(A.shape[1]))
    state = torch.randn(state_shape, device=u.device, dtype=itype, requires_grad=False)
    state_ref = state.clone()
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
        out = selective_scan_fn(
            u[..., chunk_start:chunk_end],
            state,
            delta[..., chunk_start:chunk_end],
            A,
            _B,
            _C,
            D,
            z=_z,
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
            has_initial_state=torch.ones(batch_size, device=u.device, dtype=torch.bool)
            if c > 0
            else None,
            pad_slot_id=PAD_SLOT_ID,
            block_size=2048,
            block_idx_first_scheduled_token=None,
            block_idx_last_scheduled_token=None,
            initial_state_idx=None,
        )
        outs.append(out)
    if len(outs) > 1:
        out = torch.cat(outs, dim=-1)

    out_ref, state_ref, *rest = selective_scan_ref(
        u_ref,
        delta_ref,
        A_ref,
        B_ref,
        C_ref,
        D_ref,
        z=z_ref,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
        return_last_state=True,
    )

    assert out is not None and out_ref is not None
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)
    assert state is not None and state_ref is not None
    assert torch.allclose(state, state_ref.to(itype), rtol=rtol, atol=atol)

    selective_scan_opcheck_fn(
        u,
        delta,
        A,
        B,
        C,
        D,
        z,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
        ssm_states=state,
        block_size=2048,
    )


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("has_z", [False, True])
@pytest.mark.parametrize("dstate", [16, 64])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
def test_selective_state_update(dim, dstate, has_z, itype):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (5e-3, 1e-2)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
        if torch.version.hip:
            atol *= 2
    # set seed
    set_random_seed(0)
    batch_size = 1
    state = torch.randn(batch_size, dim, dstate, dtype=itype, device=device)
    x = torch.randn(batch_size, dim, device=device, dtype=itype)
    out = torch.empty_like(x)
    dt = torch.randn(batch_size, dim, device=device, dtype=itype)
    dt_bias = torch.rand(dim, device=device) - 4.0
    A = -torch.rand(dim, dstate, device=device) - 1.0
    B = torch.randn(batch_size, dstate, device=device)
    C = torch.randn(batch_size, dstate, device=device)
    D = torch.randn(dim, device=device)
    z = torch.randn_like(x) if has_z else None
    state_ref = state.detach().clone()
    selective_state_update(
        state, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True, out=out
    )
    out_ref = selective_state_update_ref(
        state_ref, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True
    )

    assert torch.allclose(state, state_ref, rtol=rtol, atol=atol)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("has_z", [False, True])
@pytest.mark.parametrize("dstate", [16, 64])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
@pytest.mark.parametrize("max_seq_len", [1, 2, 4])
def test_selective_state_update_varlen(dim, dstate, has_z, itype, max_seq_len):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (5e-3, 1e-2)
    if itype == torch.bfloat16:
        rtol, atol = 5e-2, 1.5e-1
        if torch.version.hip:
            atol *= 2
    # set seed
    set_random_seed(0)
    batch_size = 4
    token_counts = torch.randint(1, max_seq_len + 1, (batch_size,), device=device)
    total_tokens = int(token_counts.sum().item())
    cu_seqlens = torch.tensor(
        [0] + torch.cumsum(token_counts, dim=0).tolist(),
        dtype=torch.int32,
        device=device,
    )
    state = torch.randn(batch_size, dim, dstate, dtype=itype, device=device)
    x = torch.randn(total_tokens, dim, device=device, dtype=itype)
    out = torch.empty_like(x)
    dt = torch.randn(total_tokens, dim, device=device, dtype=itype)
    dt_bias = torch.rand(dim, device=device) - 4.0
    A = -torch.rand(dim, dstate, device=device) - 1.0
    B = torch.randn(total_tokens, dstate, device=device)
    C = torch.randn(total_tokens, dstate, device=device)
    D = torch.randn(dim, device=device)
    z = torch.randn_like(x) if has_z else None
    state_ref = state.detach().clone()
    selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=True,
        out=out,
        cu_seqlens=cu_seqlens,
    )

    out_ref_list = []
    for seq_idx in range(batch_size):
        start_idx = cu_seqlens[seq_idx].item()
        end_idx = cu_seqlens[seq_idx + 1].item()
        num_tokens = end_idx - start_idx
        for token_idx in range(num_tokens):
            idx = start_idx + token_idx
            out_ref_list.append(
                selective_state_update_ref(
                    state_ref[seq_idx : seq_idx + 1],
                    x[idx : idx + 1],
                    dt[idx : idx + 1],
                    A,
                    B[idx : idx + 1],
                    C[idx : idx + 1],
                    D=D,
                    z=z[idx : idx + 1] if has_z else None,
                    dt_bias=dt_bias,
                    dt_softplus=True,
                )
            )
    out_ref = torch.cat(out_ref_list, dim=0)
    assert torch.allclose(state, state_ref, rtol=rtol, atol=atol)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("wtype", [torch.float32])
@pytest.mark.parametrize("itype", [torch.float32])
@pytest.mark.parametrize("seqlen", [1, 256, 1024, 4096])
@pytest.mark.parametrize("return_last_state", [True])
@pytest.mark.parametrize("has_delta_bias", [True])
@pytest.mark.parametrize("delta_softplus", [True])
@pytest.mark.parametrize("has_z", [True])
@pytest.mark.parametrize("has_D", [True])
@pytest.mark.parametrize("varBC_groups", [1, 2])
@pytest.mark.parametrize("is_variable_C", [True])
@pytest.mark.parametrize("is_variable_B", [True])
# tests correctness in case subset of the sequences are padded
@pytest.mark.parametrize("with_padding", [False, True])
def test_selective_scan_varlen(
    with_padding,
    is_variable_B,
    is_variable_C,
    varBC_groups,
    has_D,
    has_z,
    has_delta_bias,
    delta_softplus,
    return_last_state,
    seqlen,
    itype,
    wtype,
):
    if varBC_groups > 1 and (not is_variable_B or not is_variable_C):
        pytest.skip()  # This config is not applicable
    device = "cuda"
    rtol, atol = (6e-4, 2e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    if has_z:  # If we have z, the errors on the weights seem higher
        rtolw = max(rtolw, rtol)
        atolw = max(atolw, atol)
    # set seed
    torch.random.manual_seed(0)
    seqlens = []
    batch_size = 4
    if seqlen < 10:
        batch_size = 1
    padding = 3 if with_padding else 0
    padded_batch_size = batch_size + padding

    if with_padding and seqlen < padded_batch_size:
        pytest.skip()

    nsplits = padded_batch_size - 1
    eos_pos = torch.randperm(seqlen - 1)[:nsplits].sort().values
    seqlens.append(
        torch.diff(
            torch.cat([torch.tensor([-1]), eos_pos, torch.tensor([seqlen - 1])])
        ).tolist()
    )

    assert sum(seqlens[-1]) == seqlen
    assert all(s > 0 for s in seqlens[-1])

    total_entries = batch_size * 10
    cumsum = torch.cumsum(torch.tensor(seqlens[0]), dim=0).to(torch.int32)
    cumsum = torch.concat([torch.tensor([0], dtype=torch.int32), cumsum], dim=0).cuda()

    dim = 4
    dstate = 8
    A = -0.5 * torch.rand(dim, dstate, device=device, dtype=wtype)
    A_ref = A.clone()
    B_shape = [varBC_groups, dstate, seqlen]
    B = torch.randn(B_shape, device=device, dtype=wtype if not is_variable_B else itype)
    B_ref = B.clone()
    C_shape = [varBC_groups, dstate, seqlen]
    C = torch.randn(C_shape, device=device, dtype=wtype if not is_variable_C else itype)
    C_ref = C.clone()
    D = torch.randn(dim, device=device, dtype=torch.float32) if has_D else None
    D_ref = D.clone()
    z = torch.randn(dim, seqlen, device=device, dtype=itype)
    z_ref = z.clone()
    delta_bias = (
        (0.5 * torch.rand(dim, device=device, dtype=torch.float32))
        if has_delta_bias
        else None
    )
    u = torch.randn(dim, seqlen, device=device, dtype=itype)
    u_ref = u.clone()
    delta = 0.5 * torch.rand(dim, seqlen, device=device, dtype=itype)
    delta_ref = delta.clone()
    out = None
    out_ref = None

    prev_state_shape = (total_entries, u.shape[0], int(A.shape[1]))
    prev_state = torch.randn(
        prev_state_shape, device=u.device, dtype=itype, requires_grad=False
    )
    prev_state_ref = prev_state.clone()
    state_indices = torch.randperm(total_entries, dtype=torch.int32, device=u.device)[
        :batch_size
    ]
    unused_states_bool = torch.ones(total_entries, dtype=torch.bool, device=device)
    unused_states_bool[state_indices] = False
    padded_state_indices = torch.concat(
        [
            state_indices,
            torch.as_tensor([PAD_SLOT_ID] * padding, dtype=torch.int32, device=device),
        ],
        dim=-1,
    )

    has_initial_state = torch.randint(
        0, 2, (cumsum.shape[0] - 1,), dtype=torch.bool, device=u.device
    )
    out = selective_scan_fn(
        u,
        prev_state,
        delta,
        A,
        B,
        C,
        D,
        z,
        delta_bias,
        delta_softplus,
        cumsum,
        padded_state_indices,
        has_initial_state,
    )
    outs_ref = []
    splits = [
        torch.split(var, seqlens[0], dim=-1)
        for var in (u_ref, delta_ref, B_ref, C_ref, z_ref)
    ]
    for i in range(len(seqlens[0])):
        u_s, delta_s, B_s, C_s, z_s = (v[i].unsqueeze(0) for v in splits)
        if padded_state_indices[i] == PAD_SLOT_ID:
            continue
        out_ref_s, _ = selective_scan_ref(
            u_s,
            delta_s,
            A_ref,
            B_s,
            C_s,
            D_ref,
            z=z_s,
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
            return_last_state=return_last_state,
            prev_state=prev_state_ref[padded_state_indices[i]].unsqueeze(0)
            if has_initial_state[i]
            else None,
            final_state_out=prev_state_ref[padded_state_indices[i]].unsqueeze(0),
        )
        outs_ref.append(out_ref_s)
    out_ref = torch.cat(outs_ref, dim=-1)[0]

    unpadded_out = out[:, : out_ref[0].shape[-1]]
    print("Output diff max", (unpadded_out - out_ref).max())
    print("Output diff mean", (unpadded_out - out_ref).mean())
    print("Output state diff max", (prev_state - prev_state_ref).max())
    print("Output state diff mean", (prev_state - prev_state_ref).mean())
    assert torch.allclose(prev_state, prev_state_ref, rtol=rtol, atol=atol)
    assert torch.allclose(unpadded_out, out_ref, rtol=rtol, atol=atol)
    selective_scan_opcheck_fn(
        u,
        delta,
        A,
        B,
        C,
        D,
        z,
        delta_bias,
        delta_softplus,
        cumsum,
        padded_state_indices,
        has_initial_state,
        prev_state,
        block_size=2048,
    )


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("has_z", [True])
@pytest.mark.parametrize("dstate", [16, 64])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
# tests correctness in case subset of the sequences are padded
@pytest.mark.parametrize("with_padding", [True, False])
def test_selective_state_update_with_batch_indices(
    with_padding, dim, dstate, has_z, itype
):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (5e-3, 1e-2)
    if itype == torch.bfloat16:
        rtol, atol = 1e-1, 1e-1
        if torch.version.hip:
            atol *= 2
    # set seed
    torch.random.manual_seed(0)
    batch_size = 3
    padding = 5 if with_padding else 0
    padded_batch_size = batch_size + padding
    total_entries = 10 * batch_size
    state = torch.randn(total_entries, dim, dstate, dtype=itype, device=device)
    state_indices = torch.randperm(total_entries)[:batch_size].to(
        dtype=torch.int32, device=device
    )
    unused_states_bool = torch.ones(total_entries, dtype=torch.bool, device=device)
    unused_states_bool[state_indices] = False
    padded_state_indices = torch.concat(
        [
            state_indices,
            torch.as_tensor([PAD_SLOT_ID] * padding, dtype=torch.int32, device=device),
        ],
        dim=0,
    )
    x = torch.randn(padded_batch_size, dim, device=device, dtype=itype)
    out = torch.empty_like(x)
    dt = torch.randn(padded_batch_size, dim, device=device, dtype=itype)
    dt_bias = torch.rand(dim, device=device) - 4.0
    A = -torch.rand(dim, dstate, device=device) - 1.0
    B = torch.randn(padded_batch_size, dstate, device=device)
    C = torch.randn(padded_batch_size, dstate, device=device)
    D = torch.randn(dim, device=device)
    z = torch.randn_like(x) if has_z else None
    state_ref = state[state_indices, :].clone()
    state_before = state.clone()
    selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=padded_state_indices,
        pad_slot_id=PAD_SLOT_ID,
        out=out,
    )
    out_ref = selective_state_update_ref(
        state_ref,
        x[:batch_size],
        dt[:batch_size],
        A,
        B[:batch_size],
        C[:batch_size],
        D=D,
        z=z[:batch_size],
        dt_bias=dt_bias,
        dt_softplus=True,
    )

    print("Output diff max", (out[:batch_size] - out_ref).max())
    print("Output diff mean", (out[:batch_size] - out_ref).mean())
    print("Output state diff max", (state[state_indices, :] - state_ref).max())
    print("Output state diff mean", (state[state_indices, :] - state_ref).mean())
    # test padded entries stay the same
    if with_padding:
        assert torch.equal(state_before[unused_states_bool], state[unused_states_bool])
        assert torch.equal(x[batch_size + 1 :], x[batch_size + 1 :])
        assert torch.equal(dt[batch_size + 1 :], dt[batch_size + 1 :])
        assert torch.equal(B[batch_size + 1 :], B[batch_size + 1 :])
        assert torch.equal(C[batch_size + 1 :], C[batch_size + 1 :])

    # test "real" entries
    assert torch.allclose(state[state_indices, :], state_ref, rtol=rtol, atol=atol)
    assert torch.allclose(out[:batch_size], out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("has_z", [False, True])
@pytest.mark.parametrize("tie_hdim", [False, True])
@pytest.mark.parametrize("ngroups", [1, 4])
@pytest.mark.parametrize("dstate", [16, 64])
@pytest.mark.parametrize("dim", [2048, 4096])
def test_selective_state_update_with_heads_with_batch_indices(
    dim, dstate, ngroups, has_z, tie_hdim, itype
):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (5e-3, 3e-2)
    if itype == torch.bfloat16:
        rtol, atol = 1e-1, 1e-1
    # set seed
    torch.random.manual_seed(0)
    batch_size = 3
    headdim = 64
    nheads = dim // headdim

    total_entries = 10 * batch_size
    state = torch.randn(
        total_entries, nheads, headdim, dstate, dtype=itype, device=device
    )
    state_indices = torch.randperm(total_entries)[:batch_size].to(
        dtype=torch.int32, device=device
    )

    x = torch.randn(batch_size, nheads, headdim, device=device, dtype=itype)
    out = torch.empty_like(x)
    if not tie_hdim:
        dt = torch.randn(batch_size, nheads, headdim, device=device, dtype=itype)
        dt_bias = torch.rand(nheads, headdim, device=device) - 4.0
        A = -torch.rand(nheads, headdim, dstate, device=device) - 1.0
        D = torch.randn(nheads, headdim, device=device)
    else:
        dt = repeat(
            torch.randn(batch_size, nheads, device=device, dtype=itype),
            "b h -> b h p",
            p=headdim,
        )
        dt_bias = repeat(torch.rand(nheads, device=device) - 4.0, "h -> h p", p=headdim)
        A = repeat(
            -torch.rand(nheads, device=device) - 1.0, "h -> h p n", p=headdim, n=dstate
        )
        D = repeat(torch.randn(nheads, device=device), "h -> h p", p=headdim)
    B = torch.randn(batch_size, ngroups, dstate, device=device)
    C = torch.randn(batch_size, ngroups, dstate, device=device)
    z = torch.randn_like(x) if has_z else None
    state_ref = state[state_indices, :].detach().clone()
    selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=state_indices,
        pad_slot_id=PAD_SLOT_ID,
        out=out,
    )
    out_ref = selective_state_update_ref(
        state_ref, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True
    )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.allclose(state[state_indices, :], state_ref, rtol=rtol, atol=atol)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("has_z", [False, True])
@pytest.mark.parametrize("dstate", [16, 64])
@pytest.mark.parametrize("dim", [2048, 4096])
@pytest.mark.parametrize("max_seq_len", [2, 4])
def test_selective_state_update_with_num_accepted_tokens(
    dim, dstate, has_z, itype, max_seq_len
):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (5e-3, 1e-2)
    if itype == torch.bfloat16:
        rtol, atol = 5e-2, 1.5e-1
        if torch.version.hip:
            atol *= 2

    set_random_seed(0)
    batch_size = 4

    tokens_per_seq = torch.randint(1, max_seq_len + 1, (batch_size,), device=device)
    total_tokens = int(tokens_per_seq.sum().item())

    num_accepted_tokens = torch.randint(0, max_seq_len, (batch_size,), device=device)
    num_accepted_tokens[0] = 0  # Add edge-case of no accepted tokens
    num_accepted_tokens[1] = max_seq_len  # Add edge-case of all tokens accepted

    cu_seqlens = torch.tensor(
        [0] + torch.cumsum(tokens_per_seq, dim=0).tolist(),
        dtype=torch.int32,
        device=device,
    )

    total_state_slots = 50
    state = torch.randn(total_state_slots, dim, dstate, dtype=itype, device=device)

    state_batch_indices = torch.full(
        (batch_size, max_seq_len), PAD_SLOT_ID, dtype=torch.int32, device=device
    )
    initial_state_slots = torch.randint(
        0, 15, (batch_size,), device=device, dtype=torch.int32
    )
    for seq_idx in range(batch_size):
        token_pos = max(num_accepted_tokens[seq_idx].item() - 1, 0)
        state_batch_indices[seq_idx, token_pos] = initial_state_slots[seq_idx]

    dst_state_batch_indices = torch.full(
        (batch_size, max_seq_len), PAD_SLOT_ID, dtype=torch.int32, device=device
    )
    slot_offset = 15
    dst_slots_map = {}
    for seq_idx in range(batch_size):
        for token_idx in range(tokens_per_seq[seq_idx].item()):
            dst_state_batch_indices[seq_idx, token_idx] = slot_offset
            dst_slots_map[(seq_idx, token_idx)] = slot_offset
            slot_offset += 1

    x = torch.randn(total_tokens, dim, device=device, dtype=itype)
    out = torch.empty_like(x)
    dt = torch.randn(total_tokens, dim, device=device, dtype=itype)
    dt_bias = torch.rand(dim, device=device) - 4.0
    A = -torch.rand(dim, dstate, device=device) - 1.0
    B = torch.randn(total_tokens, dstate, device=device)
    C = torch.randn(total_tokens, dstate, device=device)
    D = torch.randn(dim, device=device)
    z = torch.randn_like(x) if has_z else None

    state_ref_intermediate = {}
    out_ref_list = []

    for seq_idx in range(batch_size):
        seq_start = cu_seqlens[seq_idx].item()
        seq_end = cu_seqlens[seq_idx + 1].item()
        num_tokens = seq_end - seq_start

        token_pos = max(num_accepted_tokens[seq_idx].item() - 1, 0)
        initial_slot = state_batch_indices[seq_idx, token_pos].item()
        state_seq = state[initial_slot : initial_slot + 1].clone()

        for token_idx in range(num_tokens):
            global_idx = seq_start + token_idx

            out_token = selective_state_update_ref(
                state_seq,
                x[global_idx : global_idx + 1],
                dt[global_idx : global_idx + 1],
                A,
                B[global_idx : global_idx + 1],
                C[global_idx : global_idx + 1],
                D=D,
                z=z[global_idx : global_idx + 1] if has_z else None,
                dt_bias=dt_bias,
                dt_softplus=True,
            )
            out_ref_list.append(out_token)
            state_ref_intermediate[(seq_idx, token_idx)] = state_seq.clone()

    out_ref = torch.cat(out_ref_list, dim=0)

    selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=True,
        out=out,
        cu_seqlens=cu_seqlens,
        state_batch_indices=state_batch_indices,
        dst_state_batch_indices=dst_state_batch_indices,
        num_accepted_tokens=num_accepted_tokens,
        pad_slot_id=PAD_SLOT_ID,
    )

    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)

    for seq_idx in range(batch_size):
        num_tokens = tokens_per_seq[seq_idx].item()
        for token_idx in range(num_tokens):
            dst_slot = dst_slots_map[(seq_idx, token_idx)]
            state_ref = state_ref_intermediate[(seq_idx, token_idx)].squeeze(0)
            assert torch.allclose(state[dst_slot], state_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("has_z", [False, True])
@pytest.mark.parametrize("dstate", [16, 64])
@pytest.mark.parametrize("dim", [2048, 4096])
@pytest.mark.parametrize("max_seq_len", [2, 4])
def test_selective_state_update_varlen_with_num_accepted(
    dim, dstate, has_z, itype, max_seq_len
):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (5e-3, 1e-2)
    if itype == torch.bfloat16:
        rtol, atol = 5e-2, 1.5e-1
        if torch.version.hip:
            atol *= 2

    set_random_seed(0)
    batch_size = 4

    tokens_per_seq = torch.randint(1, max_seq_len + 1, (batch_size,), device=device)
    total_tokens = int(tokens_per_seq.sum().item())

    num_accepted_tokens = torch.randint(0, max_seq_len, (batch_size,), device=device)
    num_accepted_tokens[0] = 0  # Add edge-case of no accepted tokens
    num_accepted_tokens[1] = max_seq_len  # Add edge-case of all tokens accepted

    cu_seqlens = torch.tensor(
        [0] + torch.cumsum(tokens_per_seq, dim=0).tolist(),
        dtype=torch.int32,
        device=device,
    )

    total_state_slots = 50
    state = torch.randn(total_state_slots, dim, dstate, dtype=itype, device=device)

    state_batch_indices = torch.full(
        (batch_size, max_seq_len), PAD_SLOT_ID, dtype=torch.int32, device=device
    )

    initial_state_slots = torch.randint(
        0, 15, (batch_size,), device=device, dtype=torch.int32
    )
    for seq_idx in range(batch_size):
        token_pos = max(num_accepted_tokens[seq_idx].item() - 1, 0)
        state_batch_indices[seq_idx, token_pos] = initial_state_slots[seq_idx]

    dst_state_batch_indices = torch.full(
        (batch_size, max_seq_len), PAD_SLOT_ID, dtype=torch.int32, device=device
    )

    slot_offset = 15
    dst_slots_map = {}
    for seq_idx in range(batch_size):
        for token_idx in range(tokens_per_seq[seq_idx].item()):
            dst_state_batch_indices[seq_idx, token_idx] = slot_offset
            dst_slots_map[(seq_idx, token_idx)] = slot_offset
            slot_offset += 1

    x = torch.randn(total_tokens, dim, device=device, dtype=itype)
    out = torch.empty_like(x)
    dt = torch.randn(total_tokens, dim, device=device, dtype=itype)
    dt_bias = torch.rand(dim, device=device) - 4.0
    A = -torch.rand(dim, dstate, device=device) - 1.0
    B = torch.randn(total_tokens, dstate, device=device)
    C = torch.randn(total_tokens, dstate, device=device)
    D = torch.randn(dim, device=device)
    z = torch.randn_like(x) if has_z else None

    state_ref_intermediate = {}

    for seq_idx in range(batch_size):
        seq_start = cu_seqlens[seq_idx].item()
        seq_end = cu_seqlens[seq_idx + 1].item()
        num_tokens = seq_end - seq_start

        token_pos = max(num_accepted_tokens[seq_idx].item() - 1, 0)
        initial_slot = state_batch_indices[seq_idx, token_pos].item()
        state_seq = state[initial_slot : initial_slot + 1].clone()

        for token_idx in range(num_tokens):
            global_idx = seq_start + token_idx

            selective_state_update_ref(
                state_seq,
                x[global_idx : global_idx + 1],
                dt[global_idx : global_idx + 1],
                A,
                B[global_idx : global_idx + 1],
                C[global_idx : global_idx + 1],
                D=D,
                z=z[global_idx : global_idx + 1] if has_z else None,
                dt_bias=dt_bias,
                dt_softplus=True,
            )

            state_ref_intermediate[(seq_idx, token_idx)] = state_seq.clone()

    selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=True,
        out=out,
        cu_seqlens=cu_seqlens,
        state_batch_indices=state_batch_indices,
        dst_state_batch_indices=dst_state_batch_indices,
        num_accepted_tokens=num_accepted_tokens,
        pad_slot_id=PAD_SLOT_ID,
    )

    for seq_idx in range(batch_size):
        num_tokens = tokens_per_seq[seq_idx].item()

        for token_idx in range(num_tokens):
            dst_slot = dst_slots_map[(seq_idx, token_idx)]
            state_ref = state_ref_intermediate[(seq_idx, token_idx)].squeeze(0)

            assert torch.allclose(state[dst_slot], state_ref, rtol=rtol, atol=atol)
