# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.gdn.replayssm import (
    pack_replayssm_rows,
    run_gdn_replayssm,
)


def test_pack_replayssm_rows_preserves_tokens_and_masks_padding():
    values = torch.tensor([[1.0], [2.0], [3.0]])
    packed = pack_replayssm_rows(
        values, torch.tensor([0, 1, 3]), 4, padding_value=-torch.inf
    )
    torch.testing.assert_close(
        packed.squeeze(-1),
        torch.tensor(
            [
                [1.0, -torch.inf, -torch.inf, -torch.inf],
                [2.0, 3.0, -torch.inf, -torch.inf],
            ]
        ),
    )


def _reference_gdn_step(
    state,
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
):
    """FP32 recurrence using FlashInfer's effective BF16 gate constants."""
    state = state.float().clone()
    A_log = A_log.to(torch.bfloat16).float()
    dt_bias = dt_bias.to(torch.bfloat16).float()
    group_size = v.shape[1] // k.shape[1]
    outputs = []
    for token in range(q.shape[0]):
        k_hv = F.normalize(k[token].float(), dim=-1).repeat_interleave(
            group_size, dim=0
        )
        q_hv = F.normalize(q[token].float(), dim=-1).repeat_interleave(
            group_size, dim=0
        )
        q_hv *= 1.0 / math.sqrt(q.shape[-1])
        decay = -torch.exp(A_log) * F.softplus(a[token].float() + dt_bias)
        beta = torch.sigmoid(b[token].float())
        state *= torch.exp(decay)[:, None, None]
        prediction = torch.einsum("hvk,hk->hv", state, k_hv)
        update = (v[token].float() - prediction) * beta[:, None]
        state += update[:, :, None] * k_hv[:, None, :]
        outputs.append(torch.einsum("hvk,hk->hv", state, q_hv))
    return torch.stack(outputs), state


def _reconstruct_replayssm_state(
    checkpoint,
    k_cache,
    u_cache,
    g_cache,
    committed,
    ring_start,
):
    state = checkpoint.float().clone()
    if committed == 0:
        return state
    rows = torch.tensor(
        [(ring_start + offset) % k_cache.size(1) for offset in range(committed)],
        dtype=torch.long,
        device=checkpoint.device,
    )
    k_logical = k_cache.index_select(1, rows).float().repeat_interleave(2, dim=0)
    u_logical = u_cache.index_select(1, rows).float()
    g_logical = g_cache.index_select(1, rows).float()
    final_g = g_logical[:, -1]
    weights = torch.exp(final_g[:, None] - g_logical)
    return torch.exp(final_g)[:, None, None] * state + torch.einsum(
        "hpv,hpk->hvk", weights[:, :, None] * u_logical, k_logical
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    reason="FlashInfer GDN ReplaySSM requires an SM90+ CUDA device",
)
@pytest.mark.parametrize("width,query_len,accepted", [(1, 1, 1), (4, 3, 2), (8, 6, 4)])
def test_gdn_replayssm_output_and_accepted_state(
    monkeypatch, width, query_len, accepted
):
    """Check nonzero outputs and accepted state through replay, wrap, and flush."""
    monkeypatch.setenv("SGLANG_GDN_WY_STRIDED_QKV", "1")
    module = (
        "gdn_decode_bf16_wy_ucache_stp"
        if width == 1
        else "gdn_decode_bf16_wy_ucache_flush"
    )
    pytest.importorskip(f"flashinfer.gdn_kernels.{module}")
    device, dtype = torch.device("cuda"), torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(20260909 + width)
    ring_slots = 16 if width == 1 else 32

    def random_tensor(*shape):
        return torch.randn(*shape, generator=generator, device=device).to(dtype)

    # Use one page stride for every state, as MambaBase.bind_kv_cache does.
    shapes = [
        (2, 128, 128),
        (2, ring_slots, 128),
        (1, ring_slots, 128),
        (2, ring_slots),
    ]
    dtypes = [dtype, dtype, dtype, torch.float32]
    sizes = [math.prod(shape) * dt.itemsize for shape, dt in zip(shapes, dtypes)]
    pages = torch.zeros(2, sum(sizes) + 16, dtype=torch.uint8, device=device)
    states = []
    offset = 0
    for shape, dt, size in zip(shapes, dtypes, sizes):
        states.append(pages[:, offset : offset + size].view(dt).view(2, *shape))
        offset += size
    checkpoint, u_cache, k_cache, g_cache = states
    checkpoint.copy_(random_tensor(2, 2, 128, 128) * 0.03)
    ring_start = torch.zeros(2, dtype=torch.int32, device=device)
    committed = torch.zeros_like(ring_start)
    A_log = torch.full((2,), -3.0, device=device)
    dt_bias = random_tensor(2).float() * 0.2
    accepted_state = checkpoint[1].float().clone()

    # Eighteen steps cross the native STP flush and the MTP ring boundary.
    for _ in range(18):
        q, k = random_tensor(query_len, 1, 128), random_tensor(query_len, 1, 128)
        v = random_tensor(query_len, 2, 128) * 0.1
        a, b = random_tensor(query_len, 2) * 0.2, random_tensor(query_len, 2)
        output = run_gdn_replayssm(
            mixed_qkv=torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=-1),
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            checkpoint_state=checkpoint,
            replayssm_cache=(u_cache, k_cache, g_cache),
            ring_start=ring_start,
            num_committed=committed,
            query_start_loc=torch.tensor(
                [0, query_len], dtype=torch.int32, device=device
            ),
            state_indices=torch.tensor([1], dtype=torch.int32, device=device),
            output_indices=torch.arange(query_len, dtype=torch.int32, device=device),
            executed_query_width=width,
            num_k_heads=1,
            num_v_heads=2,
            head_k_dim=128,
            head_v_dim=128,
        )
        expected_output, _ = _reference_gdn_step(
            accepted_state, q, k, v, a, b, A_log, dt_bias
        )
        torch.testing.assert_close(output.float(), expected_output, atol=8e-3, rtol=0)
        _, accepted_state = _reference_gdn_step(
            accepted_state,
            q[:accepted],
            k[:accepted],
            v[:accepted],
            a[:accepted],
            b[:accepted],
            A_log,
            dt_bias,
        )
        old_committed = int(committed[1].item())
        if width == 1:
            committed[1] = 0 if old_committed >= 15 else old_committed + 1
        elif old_committed + width > 16:
            ring_start[1] = (ring_start[1] + old_committed) % ring_slots
            committed[1] = accepted
        else:
            committed[1] = old_committed + accepted
        reconstructed = _reconstruct_replayssm_state(
            checkpoint[1],
            k_cache[1],
            u_cache[1],
            g_cache[1],
            int(committed[1].item()),
            int(ring_start[1].item()),
        )
        torch.testing.assert_close(reconstructed, accepted_state, atol=2e-2, rtol=0)
