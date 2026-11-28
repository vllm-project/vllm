# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from einops import repeat

from tests.kernels.mamba.utils import (
    requires_triton_stochastic_rounding,
    selective_state_update_ref,
)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    convert_rs_fp16x2,
    selective_state_update,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

DEVICE = current_platform.device_type

pytestmark = pytest.mark.skipif(
    not (
        current_platform.is_cuda_alike()
        or current_platform.is_xpu()
        or current_platform.is_cpu()
    ),
    reason="mamba_ssm kernels require CUDA-alike, XPU, or CPU",
)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("has_z", [False, True])
@pytest.mark.parametrize("dstate", [16, 64])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
@pytest.mark.skipif(
    current_platform.is_cpu(),
    reason=(
        "CPU kernel for selective_state_update only supports "
        "Mamba 2 (scalar A/dt), not Mamba 1."
    ),
)
def test_selective_state_update(dim, dstate, has_z, itype):
    device = DEVICE
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (5e-3, 1e-2)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
        if (
            current_platform.is_rocm()
            or current_platform.is_xpu()
            or current_platform.is_device_capability_family(90)
        ):
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


@pytest.mark.parametrize("philox_rounds", [0, 4])
@pytest.mark.parametrize("has_z", [False, True])
@pytest.mark.parametrize("dstate", [16, 64])
@pytest.mark.parametrize("dim", [2048, 4096])
@requires_triton_stochastic_rounding
def test_selective_state_update_stochastic_rounding(dim, dstate, has_z, philox_rounds):
    device = DEVICE
    rtol, atol = 5e-3, 1e-1
    # set seed
    set_random_seed(0)
    batch_size = 1
    state = torch.randn(batch_size, dim, dstate, dtype=torch.float16, device=device)
    x = torch.randn(batch_size, dim, device=device, dtype=torch.bfloat16)
    out = torch.empty_like(x)
    dt = torch.randn(batch_size, dim, device=device, dtype=torch.bfloat16)
    dt_bias = torch.rand(dim, device=device) - 4.0
    A = -torch.rand(dim, dstate, device=device) - 1.0
    B = torch.randn(batch_size, dstate, device=device)
    C = torch.randn(batch_size, dstate, device=device)
    D = torch.randn(dim, device=device)
    z = torch.randn_like(x) if has_z else None
    # Reference uses fp32 state to get ground truth
    state_ref = state.float()
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
        enable_stochastic_rounding=True,
        cache_philox_rounds=philox_rounds,
    )
    out_ref = selective_state_update_ref(
        state_ref, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True
    )

    assert state.dtype == torch.float16
    assert torch.allclose(state, state_ref.to(torch.float16), rtol=rtol, atol=atol)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)


@triton.jit
def _stochastic_round_kernel(x_ptr, out_ptr, seed, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < N)
    rand_offsets = tl.program_id(0) * N + offs
    rand = tl.randint(seed, rand_offsets)
    y = convert_rs_fp16x2(x, rand)
    tl.store(out_ptr + rand_offsets, y, mask=offs < N)


@requires_triton_stochastic_rounding
def test_stochastic_rounding_is_unbiased():
    """Each draw must land on an fp16 neighbour of x and the mean over draws
    must converge to x; round-to-nearest or misused random bits are biased."""
    num_values, num_draws = 1024, 4096
    set_random_seed(0)
    x = torch.randn(num_values, device=DEVICE) * 4
    out = torch.empty(num_draws, num_values, dtype=torch.float16, device=DEVICE)
    _stochastic_round_kernel[(num_draws,)](x, out, 1234, num_values, BLOCK=1024)

    ulp = torch.ldexp(torch.ones_like(x), torch.frexp(x).exponent - 11).clamp(2**-24)
    err = (out.float() - x) / ulp
    assert err.abs().max() < 1
    assert (err.mean(dim=0).abs() < 1 / 16).all()
    assert (err > 0).any() and (err < 0).any()


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("has_z", [False, True])
@pytest.mark.parametrize("dstate", [16, 64])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
@pytest.mark.parametrize("max_seq_len", [1, 2, 4])
@pytest.mark.skipif(
    current_platform.is_cpu(),
    reason=(
        "CPU kernel for selective_state_update only supports "
        "Mamba 2 (scalar A/dt), not Mamba 1."
    ),
)
def test_selective_state_update_varlen(dim, dstate, has_z, itype, max_seq_len):
    device = DEVICE
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (5e-3, 1e-2)
    if itype == torch.bfloat16:
        rtol, atol = 5e-2, 1.5e-1
        if current_platform.is_rocm() or current_platform.is_xpu():
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
                    z=z[idx : idx + 1] if z is not None else None,
                    dt_bias=dt_bias,
                    dt_softplus=True,
                )
            )
    out_ref = torch.cat(out_ref_list, dim=0)
    assert torch.allclose(state, state_ref, rtol=rtol, atol=atol)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("has_z", [True])
@pytest.mark.parametrize("dstate", [16, 64])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
# tests correctness in case subset of the sequences are padded
@pytest.mark.parametrize("with_padding", [True, False])
@pytest.mark.skipif(
    current_platform.is_cpu(),
    reason=(
        "CPU kernel for selective_state_update only supports "
        "Mamba 2 (scalar A/dt), not Mamba 1."
    ),
)
def test_selective_state_update_with_batch_indices(
    with_padding, dim, dstate, has_z, itype
):
    device = DEVICE
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (5e-3, 1e-2)
    if itype == torch.bfloat16:
        rtol, atol = 1e-1, 1e-1
        if current_platform.is_rocm() or current_platform.is_xpu():
            atol *= 2
    # set seed
    torch.random.manual_seed(0)
    batch_size = 3
    padding = 5 if with_padding else 0
    padded_batch_size = batch_size + padding
    total_entries = 10 * batch_size
    state = torch.randn(total_entries, dim, dstate, dtype=itype, device=device)
    # +1 to exclude index 0 (null block)
    state_indices = (torch.randperm(total_entries - 1)[:batch_size] + 1).to(
        dtype=torch.int32, device=device
    )
    unused_states_bool = torch.ones(total_entries, dtype=torch.bool, device=device)
    unused_states_bool[state_indices] = False
    padded_state_indices = torch.concat(
        [
            state_indices,
            torch.as_tensor(
                [NULL_BLOCK_ID] * padding, dtype=torch.int32, device=device
            ),
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
        z=z[:batch_size] if z is not None else None,
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
@pytest.mark.skipif(
    current_platform.is_cpu(),
    reason=(
        "CPU kernel for selective_state_update only supports "
        "Mamba 2 (scalar A/dt), not Mamba 1."
    ),
)
def test_selective_state_update_with_heads_with_batch_indices(
    dim, dstate, ngroups, has_z, tie_hdim, itype
):
    device = DEVICE
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
    # +1 to exclude index 0 (null block)
    state_indices = (torch.randperm(total_entries - 1)[:batch_size] + 1).to(
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
@pytest.mark.skipif(
    current_platform.is_cpu(),
    reason=(
        "CPU kernel for selective_state_update only supports "
        "Mamba 2 (scalar A/dt), not Mamba 1."
    ),
)
def test_selective_state_update_with_num_accepted_tokens(
    dim, dstate, has_z, itype, max_seq_len
):
    device = DEVICE
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (5e-3, 1e-2)
    if itype == torch.bfloat16:
        rtol, atol = 5e-2, 1.5e-1
        if current_platform.is_rocm() or current_platform.is_xpu():
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
        (batch_size, max_seq_len), NULL_BLOCK_ID, dtype=torch.int32, device=device
    )
    # Start from 1 to exclude null block at index 0
    initial_state_slots = torch.randint(
        1, 15, (batch_size,), device=device, dtype=torch.int32
    )
    for seq_idx in range(batch_size):
        token_pos = max(num_accepted_tokens[seq_idx].item() - 1, 0)
        state_batch_indices[seq_idx, token_pos] = initial_state_slots[seq_idx]

    dst_state_batch_indices = torch.full(
        (batch_size, max_seq_len), NULL_BLOCK_ID, dtype=torch.int32, device=device
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
                z=z[global_idx : global_idx + 1] if z is not None else None,
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
@pytest.mark.skipif(
    current_platform.is_cpu(),
    reason=(
        "CPU kernel for selective_state_update only supports "
        "Mamba 2 (scalar A/dt), not Mamba 1."
    ),
)
def test_selective_state_update_varlen_with_num_accepted(
    dim, dstate, has_z, itype, max_seq_len
):
    device = DEVICE
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (5e-3, 1e-2)
    if itype == torch.bfloat16:
        rtol, atol = 5e-2, 1.5e-1
        if current_platform.is_rocm() or current_platform.is_xpu():
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
        (batch_size, max_seq_len), NULL_BLOCK_ID, dtype=torch.int32, device=device
    )

    # Start from 1 to exclude null block at index 0
    initial_state_slots = torch.randint(
        1, 15, (batch_size,), device=device, dtype=torch.int32
    )
    for seq_idx in range(batch_size):
        token_pos = max(num_accepted_tokens[seq_idx].item() - 1, 0)
        state_batch_indices[seq_idx, token_pos] = initial_state_slots[seq_idx]

    dst_state_batch_indices = torch.full(
        (batch_size, max_seq_len), NULL_BLOCK_ID, dtype=torch.int32, device=device
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
                z=z[global_idx : global_idx + 1] if z is not None else None,
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
    )

    for seq_idx in range(batch_size):
        num_tokens = tokens_per_seq[seq_idx].item()

        for token_idx in range(num_tokens):
            dst_slot = dst_slots_map[(seq_idx, token_idx)]
            state_ref = state_ref_intermediate[(seq_idx, token_idx)].squeeze(0)

            assert torch.allclose(state[dst_slot], state_ref, rtol=rtol, atol=atol)
