# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.models.kimi_k3.amd.ops.third_party.kda import chunk_kda
from vllm.models.kimi_k3.amd.ops.third_party.kda.chunk_intra import (
    chunk_kda_fwd_intra,
)
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="AMD KDA requires ROCm",
)


def _naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = v.dtype
    scale = k.shape[-1] ** -0.5
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    state = initial_state.float().clone()
    output = torch.zeros_like(v)

    for i in range(q.shape[1]):
        q_i, k_i, v_i, g_i, beta_i = (
            q[:, i],
            k[:, i],
            v[:, i],
            g[:, i],
            beta[:, i],
        )
        state = state * g_i[..., None].exp()
        state += torch.einsum(
            "bhk,bhv->bhkv",
            beta_i[..., None] * k_i,
            v_i - (k_i[..., None] * state).sum(-2),
        )
        output[:, i] = torch.einsum("bhk,bhkv->bhv", q_i * scale, state)

    return output.to(dtype), state


def _assert_rmse_close(
    reference: torch.Tensor,
    actual: torch.Tensor,
    threshold: float = 0.005,
) -> None:
    difference = (reference.float() - actual.float()).flatten()
    relative_rmse = difference.square().mean().sqrt() / (
        reference.float().flatten().square().mean().sqrt() + 1e-8
    )
    assert not torch.isnan(actual).any()
    assert relative_rmse < threshold


@torch.inference_mode()
def test_chunk_kda_matches_recurrent_reference() -> None:
    torch.manual_seed(42)
    num_heads = 8
    head_dim = 128
    cu_seqlens = [0, 15, 100]
    total_tokens = cu_seqlens[-1]
    num_sequences = len(cu_seqlens) - 1
    dtype = torch.bfloat16

    q = torch.rand(1, total_tokens, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.rand_like(q)
    v = torch.rand_like(q)
    g = F.logsigmoid(
        torch.randn(
            1,
            total_tokens,
            num_heads,
            head_dim,
            device="cuda",
            dtype=torch.float32,
        )
    ).to(dtype)
    beta = torch.rand(1, total_tokens, num_heads, device="cuda", dtype=dtype).sigmoid()
    initial_state = torch.randn(
        num_sequences,
        num_heads,
        head_dim,
        head_dim,
        device="cuda",
        dtype=torch.float32,
    )

    reference_outputs = []
    reference_states = []
    for sequence, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        output, state = _naive_recurrent_kda(
            l2norm_fwd(q[:, start:end].contiguous()),
            l2norm_fwd(k[:, start:end].contiguous()),
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            initial_state[sequence].unsqueeze(0),
        )
        reference_outputs.append(output)
        reference_states.append(state)

    actual_output, actual_state = chunk_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        initial_state=initial_state.transpose(-1, -2).contiguous().clone(),
        output_final_state=True,
        cu_seqlens=torch.tensor(cu_seqlens, device="cuda", dtype=torch.int64),
        use_qk_l2norm_in_kernel=True,
    )

    reference_output = torch.cat(reference_outputs, dim=1)
    reference_state = torch.cat(reference_states).transpose(-1, -2).contiguous()
    _assert_rmse_close(reference_output, actual_output)
    _assert_rmse_close(reference_state, actual_state)


@pytest.mark.parametrize("safe_gate", [False, True])
@pytest.mark.parametrize(
    "cu_seqlens",
    [
        pytest.param(None, id="fixed"),
        pytest.param([0, 15, 100, 300], id="varlen"),
    ],
)
@torch.inference_mode()
def test_chunk_kda_intra_zeros_upper_triangle(
    safe_gate: bool,
    cu_seqlens: list[int] | None,
) -> None:
    torch.manual_seed(42)
    chunk_size = 64
    num_heads = 8
    head_dim = 128
    total_tokens = 300

    q = torch.randn(
        1,
        total_tokens,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn_like(q)
    g = torch.randn_like(q)
    beta = torch.rand(
        1,
        total_tokens,
        num_heads,
        device="cuda",
        dtype=torch.bfloat16,
    )

    cu_seqlens_t = None
    chunk_indices = None
    sequence_bounds = [(0, total_tokens)]
    if cu_seqlens is not None:
        cu_seqlens_t = torch.tensor(cu_seqlens, device="cuda", dtype=torch.int64)
        chunk_indices = prepare_chunk_indices(cu_seqlens_t, chunk_size)
        sequence_bounds = list(zip(cu_seqlens[:-1], cu_seqlens[1:]))

    _, A = chunk_kda_fwd_intra(
        q=q,
        k=k,
        gk=g,
        beta=beta,
        scale=head_dim**-0.5,
        cu_seqlens=cu_seqlens_t,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
    )

    columns = torch.arange(chunk_size, device="cuda")
    for start, end in sequence_bounds:
        rows = torch.arange(end - start, device="cuda") % chunk_size
        upper_triangle = columns[None, :] > rows[:, None]
        upper_values = A[0, start:end].masked_select(upper_triangle[:, None, :])
        assert torch.count_nonzero(upper_values) == 0
