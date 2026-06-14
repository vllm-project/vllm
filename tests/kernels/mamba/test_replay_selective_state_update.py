# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.mamba.utils import selective_state_update_ref
from vllm.model_executor.layers.mamba.ops.replay_selective_state_update import (
    replay_selective_state_update,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DEVICE = current_platform.device_type

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="replay selective state update currently requires CUDA",
)


def _run_tokens(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a token sequence with the reference SSU and return outputs/state."""
    state = state.clone()
    outs = []
    for token_idx in range(x.shape[0]):
        outs.append(
            selective_state_update_ref(
                state,
                x[token_idx : token_idx + 1],
                dt[token_idx : token_idx + 1],
                A,
                B[token_idx : token_idx + 1],
                C[token_idx : token_idx + 1],
                D=D,
                dt_bias=dt_bias,
                dt_softplus=True,
            )
        )
    return torch.cat(outs, dim=0), state


@pytest.mark.parametrize(
    ("itype", "state_dtype"),
    [
        (torch.bfloat16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
    ],
)
def test_replay_selective_state_update_two_steps(itype, state_dtype):
    set_random_seed(0)
    device = DEVICE
    batch = 3
    cache_size = 8
    num_steps = 4
    nheads = 2
    ngroups = 1
    head_dim = 16
    dstate = 16
    rtol, atol = (6e-2, 2e-1)

    state_indices = torch.tensor([1, 3, 5], dtype=torch.int32, device=device)
    state = torch.randn(
        cache_size,
        nheads,
        head_dim,
        dstate,
        dtype=state_dtype,
        device=device,
    )
    state_initial = state.clone()

    old_x = torch.zeros(
        cache_size,
        num_steps,
        nheads,
        head_dim,
        dtype=itype,
        device=device,
    )
    old_B = torch.zeros(
        cache_size,
        2,
        num_steps,
        ngroups,
        dstate,
        dtype=itype,
        device=device,
    )
    old_dt = torch.zeros(
        cache_size,
        2,
        nheads,
        num_steps,
        dtype=torch.float32,
        device=device,
    )
    old_dA_cumsum = torch.zeros_like(old_dt)
    cache_buf_idx = torch.zeros(cache_size, dtype=torch.int32, device=device)
    replay_valid = torch.zeros(cache_size, dtype=torch.int32, device=device)

    A_scalar = -torch.rand(nheads, dtype=torch.float32, device=device) - 1.0
    A = A_scalar[:, None, None].expand(nheads, head_dim, dstate)
    D = torch.randn(nheads, head_dim, dtype=itype, device=device)
    dt_bias_scalar = torch.rand(nheads, dtype=torch.float32, device=device) - 4.0
    dt_bias = dt_bias_scalar[:, None].expand(nheads, head_dim)

    def make_inputs():
        x = torch.randn(
            batch,
            num_steps,
            nheads,
            head_dim,
            dtype=itype,
            device=device,
        )
        dt_scalar = torch.randn(
            batch,
            num_steps,
            nheads,
            dtype=itype,
            device=device,
        )
        dt = dt_scalar[..., None].expand(batch, num_steps, nheads, head_dim)
        B = torch.randn(
            batch,
            num_steps,
            ngroups,
            dstate,
            dtype=itype,
            device=device,
        )
        C = torch.randn(
            batch,
            num_steps,
            ngroups,
            dstate,
            dtype=itype,
            device=device,
        )
        out = torch.empty_like(x)
        return x, dt, B, C, out

    x0, dt0, B0, C0, out0 = make_inputs()
    accepted0 = torch.tensor([1, 3, 4], dtype=torch.int32, device=device)
    replay_selective_state_update(
        state,
        old_x,
        old_B,
        old_dt,
        old_dA_cumsum,
        cache_buf_idx,
        replay_valid,
        x0,
        dt0,
        A,
        B0,
        C0,
        out0,
        accepted0,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=state_indices,
    )

    for batch_idx, state_idx in enumerate(state_indices.tolist()):
        out_ref, _ = _run_tokens(
            state_initial[state_idx : state_idx + 1],
            x0[batch_idx],
            dt0[batch_idx],
            A,
            B0[batch_idx],
            C0[batch_idx],
            D,
            dt_bias,
        )
        torch.testing.assert_close(out0[batch_idx], out_ref, rtol=rtol, atol=atol)
        torch.testing.assert_close(
            state[state_idx],
            state_initial[state_idx],
            rtol=rtol,
            atol=atol,
        )

    replay_indices = state_indices.long()
    assert torch.equal(replay_valid[replay_indices], torch.ones_like(state_indices))
    assert torch.equal(cache_buf_idx[replay_indices], torch.ones_like(state_indices))

    x1, dt1, B1, C1, out1 = make_inputs()
    replay_selective_state_update(
        state,
        old_x,
        old_B,
        old_dt,
        old_dA_cumsum,
        cache_buf_idx,
        replay_valid,
        x1,
        dt1,
        A,
        B1,
        C1,
        out1,
        accepted0,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=state_indices,
    )

    for batch_idx, state_idx in enumerate(state_indices.tolist()):
        _, replayed_state = _run_tokens(
            state_initial[state_idx : state_idx + 1],
            x0[batch_idx, : accepted0[batch_idx].item()],
            dt0[batch_idx, : accepted0[batch_idx].item()],
            A,
            B0[batch_idx, : accepted0[batch_idx].item()],
            C0[batch_idx, : accepted0[batch_idx].item()],
            D,
            dt_bias,
        )
        out_ref, _ = _run_tokens(
            replayed_state,
            x1[batch_idx],
            dt1[batch_idx],
            A,
            B1[batch_idx],
            C1[batch_idx],
            D,
            dt_bias,
        )
        torch.testing.assert_close(out1[batch_idx], out_ref, rtol=rtol, atol=atol)
        torch.testing.assert_close(
            state[state_idx],
            replayed_state.squeeze(0),
            rtol=rtol,
            atol=atol,
        )
    assert torch.equal(cache_buf_idx[replay_indices], torch.zeros_like(state_indices))


@pytest.mark.parametrize("itype", [torch.bfloat16, torch.float16])
def test_replay_selective_state_update_multiple_groups(itype):
    set_random_seed(0)
    device = DEVICE
    batch = 3
    cache_size = 16
    num_steps = 6
    nheads = 4
    ngroups = 2
    head_dim = 8
    dstate = 16
    rtol, atol = (6e-2, 2e-1)

    state_indices_table = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 0, 0, 0],
        ],
        dtype=torch.int32,
        device=device,
    )
    state_indices = state_indices_table[:, 0]
    assert not state_indices.is_contiguous()
    state = torch.randn(
        cache_size,
        nheads,
        head_dim,
        dstate,
        dtype=torch.float16,
        device=device,
    )
    state_initial = state.clone()

    old_x = torch.zeros(
        cache_size,
        num_steps,
        nheads,
        head_dim,
        dtype=itype,
        device=device,
    )
    old_B = torch.zeros(
        cache_size,
        2,
        num_steps,
        ngroups,
        dstate,
        dtype=itype,
        device=device,
    )
    old_dt = torch.zeros(
        cache_size,
        2,
        nheads,
        num_steps,
        dtype=torch.float32,
        device=device,
    )
    old_dA_cumsum = torch.zeros_like(old_dt)
    cache_buf_idx = torch.zeros(cache_size, dtype=torch.int32, device=device)
    replay_valid = torch.zeros(cache_size, dtype=torch.int32, device=device)

    A_scalar = -torch.rand(nheads, dtype=torch.float32, device=device) - 1.0
    A = A_scalar[:, None, None].expand(nheads, head_dim, dstate)
    D = torch.randn(nheads, head_dim, dtype=itype, device=device)
    dt_bias_scalar = torch.rand(nheads, dtype=torch.float32, device=device) - 4.0
    dt_bias = dt_bias_scalar[:, None].expand(nheads, head_dim)

    x = torch.randn(
        batch,
        num_steps,
        nheads,
        head_dim,
        dtype=itype,
        device=device,
    )
    dt_scalar = torch.randn(
        batch,
        num_steps,
        nheads,
        dtype=itype,
        device=device,
    )
    dt = dt_scalar[..., None].expand(batch, num_steps, nheads, head_dim)
    B = torch.randn(batch, num_steps, ngroups, dstate, dtype=itype, device=device)
    C = torch.randn(batch, num_steps, ngroups, dstate, dtype=itype, device=device)
    out = torch.empty_like(x)
    accepted = torch.ones(batch, dtype=torch.int32, device=device)

    replay_selective_state_update(
        state,
        old_x,
        old_B,
        old_dt,
        old_dA_cumsum,
        cache_buf_idx,
        replay_valid,
        x,
        dt,
        A,
        B,
        C,
        out,
        accepted,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=state_indices,
    )

    for batch_idx, state_idx in enumerate(state_indices.tolist()):
        out_ref, _ = _run_tokens(
            state_initial[state_idx : state_idx + 1],
            x[batch_idx],
            dt[batch_idx],
            A,
            B[batch_idx],
            C[batch_idx],
            D,
            dt_bias,
        )
        torch.testing.assert_close(out[batch_idx], out_ref, rtol=rtol, atol=atol)
