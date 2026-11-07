# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculative decoding with the Triton Mamba2 ReplaySSM kernels.

The kernels follow FlashInfer's ``checkpointing_ssu`` ring contract and are
driven here by the tracker transition of ``_postprocess_replayssm_kernel``.
Over many verify steps with random acceptance, every verified output and every
materialized prefix state must match a plain recurrence over the accepted
tokens.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.ops.selective_state_update_replayssm_spec import (  # noqa: E501
    replayssm_materialize,
    selective_state_update_replayssm_spec,
)
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.worker.replayssm_utils import _cuda_i64_ptrs, _cuda_i64_slot_strides

NHEADS, HEADDIM, NGROUPS, DSTATE = 8, 64, 2, 128


def _step_ref(S, x, dt, A, B, C, D):
    """One token of the exact recurrence on fp32 ``S`` (nheads, dim, dstate)."""
    ratio = S.shape[0] // B.shape[0]
    B = B.float().repeat_interleave(ratio, 0)
    C = C.float().repeat_interleave(ratio, 0)
    dt = dt.float()[:, None, None]
    S = S * torch.exp(A[:, None, None] * dt) + dt * x.float()[:, :, None] * B[:, None]
    return S, torch.einsum("hdn,hn->hd", S, C) + D[:, None] * x.float()


@pytest.mark.parametrize(
    "state_dtype,act_dtype",
    [
        (torch.float32, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.float16, torch.bfloat16),
    ],
)
@pytest.mark.parametrize("num_spec,max_window", [(1, 4), (5, 16), (3, 4)])
def test_replayssm_spec_decode_matches_recurrence(
    state_dtype, act_dtype, num_spec, max_window
):
    set_random_seed(0)
    device = "cuda"
    T = 1 + num_spec
    ring_len = max_window + T
    batch, num_slots, num_steps = 4, 8, 24
    exact = state_dtype == act_dtype == torch.float32
    rtol, atol = (1e-4, 1e-3) if exact else (2e-2, 5e-2)

    A = -torch.rand(NHEADS, device=device) - 0.5
    dt_bias = torch.rand(NHEADS, device=device) - 2.0
    D = torch.randn(NHEADS, device=device)
    state = torch.randn(num_slots, NHEADS, HEADDIM, DSTATE, device=device).to(
        state_dtype
    )
    x_cache = torch.randn(num_slots, NHEADS, ring_len, HEADDIM, device=device).to(
        act_dtype
    )
    B_cache = torch.randn(num_slots, NGROUPS, ring_len, DSTATE, device=device).to(
        act_dtype
    )
    dt_cache = torch.rand(num_slots, NHEADS, ring_len, device=device)
    # Rows 1, 3, 4 and 6 hold requests; the last decode row is CUDA-graph padding.
    slots = torch.tensor([3, 1, 6, 4, NULL_BLOCK_ID], device=device, dtype=torch.int32)
    ring_start = torch.randint(0, ring_len, (num_slots,), device=device)
    ring_start = ring_start.to(torch.int32)
    num_committed = torch.zeros(num_slots, device=device, dtype=torch.int32)
    true_state = {s: state[s].float() for s in slots[:batch].tolist()}
    pad_x = torch.randn(NHEADS, HEADDIM, device=device).to(act_dtype)

    for _ in range(num_steps):
        seqlens = torch.randint(1, T + 1, (batch,)).tolist()
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(seqlens).cumsum(0).tolist(), sum(seqlens)],
            device=device,
            dtype=torch.int32,
        )
        total = sum(seqlens) + 1
        x = torch.randn(total, NHEADS, HEADDIM, device=device).to(act_dtype)
        x[-1] = pad_x
        dt = torch.randn(total, NHEADS, device=device).to(act_dtype)
        B = torch.randn(total, NGROUPS, DSTATE, device=device).to(act_dtype)
        C = torch.randn(total, NGROUPS, DSTATE, device=device).to(act_dtype)
        out = torch.zeros_like(x)
        prev_start, prev_committed = ring_start.clone(), num_committed.clone()
        selective_state_update_replayssm_spec(
            state,
            x_cache,
            B_cache,
            dt_cache,
            ring_start,
            num_committed,
            x,
            dt[..., None].expand(-1, -1, HEADDIM),
            A[:, None, None].expand(-1, HEADDIM, DSTATE),
            B,
            C,
            out,
            cu_seqlens=cu_seqlens,
            max_seqlen=T,
            state_batch_indices=slots,
            D=D[:, None].expand(-1, HEADDIM),
            dt_bias=dt_bias[:, None].expand(-1, HEADDIM),
            dt_softplus=True,
        )
        assert out[-1].eq(0).all(), "padded row must not be written"

        for b, slot in enumerate(slots[:batch].tolist()):
            bos, seqlen = int(cu_seqlens[b]), seqlens[b]
            accepted = int(torch.randint(1, seqlen + 1, ()))
            S = true_state[slot]
            for t in range(bos, bos + seqlen):
                S, y = _step_ref(
                    S, x[t], F.softplus(dt[t].float() + dt_bias), A, B[t], C[t], D
                )
                torch.testing.assert_close(out[t].float(), y, rtol=rtol, atol=atol)
                if t == bos + accepted - 1:
                    true_state[slot] = S
            # Tracker transition of _postprocess_replayssm_kernel.
            if int(prev_committed[slot]) + seqlen > max_window:
                ring_start[slot] = (prev_start[slot] + prev_committed[slot]) % ring_len
                num_committed[slot] = accepted
            else:
                num_committed[slot] = prev_committed[slot] + accepted

    # Publishing a prefix snapshot replays the committed ring entries onto the
    # checkpoint in another slot; layer 0 has no work and must be skipped.
    live = slots[:batch].long()
    dst_slots = [2, 5, 7, NULL_BLOCK_ID]
    src = torch.full((2, 8), NULL_BLOCK_ID, device=device, dtype=torch.int32)
    dst = src.clone()
    src[1, :batch] = slots[:batch]
    dst[1, :batch] = torch.tensor(dst_slots, device=device)
    active = torch.full((8,), -1, device=device, dtype=torch.int32)
    active[:batch] = torch.arange(batch, device=device).flip(0)
    before = state.clone()
    tables = [
        (_cuda_i64_ptrs([t, t]), _cuda_i64_slot_strides([t, t]))
        for t in (state, x_cache, B_cache, dt_cache)
    ]
    (s_ptrs, s_strides), (x_ptrs, x_strides), (B_ptrs, B_strides) = tables[:3]
    dt_ptrs, dt_strides = tables[3]
    replayssm_materialize(
        s_ptrs,
        s_strides,
        x_ptrs,
        x_strides,
        B_ptrs,
        B_strides,
        dt_ptrs,
        dt_strides,
        _cuda_i64_ptrs([A, A]),
        src,
        dst,
        ring_start[live].contiguous(),
        num_committed[live].contiguous(),
        active,
        num_reqs=batch,
        state_dtype=state_dtype,
        input_dtype=act_dtype,
        matrixA_dtype=torch.float32,
        num_heads=NHEADS,
        dim=HEADDIM,
        dstate=DSTATE,
        heads_per_group=NHEADS // NGROUPS,
        ring_buffer_len=ring_len,
    )
    for slot, dst_slot in zip(live.tolist(), dst_slots[:-1]):
        torch.testing.assert_close(
            state[dst_slot].float(), true_state[slot], rtol=rtol, atol=atol
        )
    untouched = [s for s in range(num_slots) if s not in dst_slots]
    assert state[untouched].equal(before[untouched])
