# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools

import pytest
import torch
import torch.nn.functional as F

import vllm._custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

set_random_seed(12345)

NUM_HEADS = [
    (2, 4),
    (4, 4),
]
HEAD_DIMS = [
    (32, 32),
    (64, 32),
]
CHUNK_SIZE = 64
PREFILL_SEQ_LENS = [
    [1],
    [1, 2, 3],
    [CHUNK_SIZE - 1],
    [CHUNK_SIZE],
    [CHUNK_SIZE + 1],
    [CHUNK_SIZE - 1, CHUNK_SIZE, CHUNK_SIZE + 1],
    [2 * CHUNK_SIZE - 1, 2 * CHUNK_SIZE, 2 * CHUNK_SIZE + 1],
    [4 * CHUNK_SIZE + 17],
]
DECODE_BATCH_SIZES = [1, 3, 5]


@functools.lru_cache(maxsize=128, typed=False)
def tensor_cache(
    elem_num: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.rand(elem_num, dtype=dtype)
    return tensor


def ref_l2norm(
    x: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-5,
) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def ref_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    softplus_x = F.softplus(a.float() + dt_bias.float(), beta=1.0, threshold=20.0)
    g = -torch.exp(A_log.float()) * softplus_x
    beta = torch.sigmoid(b.float()).to(dtype=b.dtype)
    return g, beta


def ref_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    g, beta = ref_gdn_gating(A_log, a, b, dt_bias)
    out = torch.empty_like(value)
    final_state = torch.empty_like(initial_state)

    for seq_idx in range(cu_seqlens.numel() - 1):
        begin = int(cu_seqlens[seq_idx].item())
        end = int(cu_seqlens[seq_idx + 1].item())
        q_seq = query[:, begin:end]
        k_seq = key[:, begin:end]
        v_seq = value[:, begin:end]
        g_seq = g[begin:end].unsqueeze(0)
        beta_seq = beta[begin:end].unsqueeze(0)
        initial_dtype = q_seq.dtype

        if use_qk_l2norm_in_kernel:
            q_seq = ref_l2norm(q_seq, dim=-1)
            k_seq = ref_l2norm(k_seq, dim=-1)

        if q_seq.shape[2] != v_seq.shape[2]:
            repeat_factor = v_seq.shape[2] // q_seq.shape[2]
            q_seq = q_seq.repeat_interleave(repeat_factor, dim=2)
            k_seq = k_seq.repeat_interleave(repeat_factor, dim=2)

        q_seq, k_seq, v_seq, beta_seq, g_seq = [
            x.transpose(1, 2).contiguous().to(torch.float32)
            for x in (q_seq, k_seq, v_seq, beta_seq, g_seq)
        ]

        batch_size, num_heads, seq_len, head_dim = q_seq.shape
        v_head_dim = v_seq.shape[-1]
        q_seq = q_seq * (1 / (head_dim**0.5))
        out_seq = torch.empty(
            batch_size,
            num_heads,
            seq_len,
            v_head_dim,
            dtype=v_seq.dtype,
        )
        state = initial_state[seq_idx : seq_idx + 1].to(v_seq)

        for token_idx in range(seq_len):
            q_t = q_seq[:, :, token_idx]
            k_t = k_seq[:, :, token_idx]
            v_t = v_seq[:, :, token_idx]
            g_t = g_seq[:, :, token_idx].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta_seq[:, :, token_idx].unsqueeze(-1)

            state = state * g_t
            kv_mem = (state * k_t.unsqueeze(-2)).sum(dim=-1)
            delta = (v_t - kv_mem) * beta_t
            state = state + delta.unsqueeze(-1) * k_t.unsqueeze(-2)
            out_seq[:, :, token_idx] = (state * q_t.unsqueeze(-2)).sum(dim=-1)

        out[:, begin:end] = out_seq.transpose(1, 2).contiguous().to(initial_dtype)
        final_state[seq_idx] = state.squeeze(0)

    return out, final_state


def gdn_inputs(
    num_tokens: int,
    num_heads: tuple[int, int],
    head_dims: tuple[int, int],
) -> tuple[torch.Tensor, ...]:
    num_qk_heads, num_v_heads = num_heads
    head_dim, v_head_dim = head_dims
    q_shape = (1, num_tokens, num_qk_heads, head_dim)
    q_numel = num_tokens * num_qk_heads * head_dim
    q = tensor_cache(q_numel, torch.bfloat16).view(q_shape)
    k = tensor_cache(q_numel, torch.bfloat16).view(q_shape)

    v_shape = (1, num_tokens, num_v_heads, v_head_dim)
    v = tensor_cache(num_tokens * num_v_heads * v_head_dim, torch.bfloat16).view(
        v_shape
    )

    gate_shape = (num_tokens, num_v_heads)
    gate_numel = num_tokens * num_v_heads
    a = tensor_cache(gate_numel, torch.bfloat16).view(gate_shape)
    b = tensor_cache(gate_numel, torch.bfloat16).view(gate_shape)
    A_log = tensor_cache(num_v_heads, torch.float32)
    dt_bias = tensor_cache(num_v_heads, torch.bfloat16)
    return q, k, v, a, b, A_log, dt_bias


@pytest.mark.parametrize("num_tokens", [1, 9])
@pytest.mark.parametrize("num_v_heads", [4, 8])
@torch.inference_mode()
def test_fused_gdn_gating_cpu(
    num_tokens: int,
    num_v_heads: int,
) -> None:
    gate_shape = (num_tokens, num_v_heads)
    gate_numel = num_tokens * num_v_heads
    a = tensor_cache(gate_numel, torch.bfloat16).view(gate_shape)
    b = tensor_cache(gate_numel, torch.bfloat16).view(gate_shape)
    A_log = tensor_cache(num_v_heads, torch.float32)
    dt_bias = tensor_cache(num_v_heads, torch.bfloat16)

    g_ref, beta_ref = ref_gdn_gating(A_log, a, b, dt_bias)
    g, beta = ops.fused_gdn_gating_cpu(A_log, a, b, dt_bias)

    torch.testing.assert_close(g, g_ref.unsqueeze(0), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        beta.float(), beta_ref.unsqueeze(0).float(), atol=5e-3, rtol=5e-3
    )


# decode path
@pytest.mark.parametrize("batch_size", DECODE_BATCH_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_dims", HEAD_DIMS)
@torch.inference_mode()
def test_fused_sigmoid_gating_delta_rule_update_cpu(
    batch_size: int,
    num_heads: tuple[int, int],
    head_dims: tuple[int, int],
) -> None:
    q, k, v, a, b, A_log, dt_bias = gdn_inputs(
        num_tokens=batch_size,
        num_heads=num_heads,
        head_dims=head_dims,
    )
    _, num_v_heads = num_heads
    head_dim, v_head_dim = head_dims
    state_indices = torch.arange(batch_size, dtype=torch.int32)
    cu_seqlens = torch.arange(batch_size + 1, dtype=torch.int32)
    state_shape = (batch_size, num_v_heads, head_dim, v_head_dim)
    state = tensor_cache(
        batch_size * num_v_heads * head_dim * v_head_dim, torch.float32
    ).view(state_shape)
    state_ref = state[state_indices].transpose(-1, -2).contiguous()

    out_ref, final_state_ref = ref_gated_delta_rule(
        query=q,
        key=k,
        value=v,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=state_ref,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )
    out_ref = out_ref.transpose(0, 1).contiguous()

    state_out = state.clone()
    out = ops.fused_sigmoid_gating_delta_rule_update_cpu(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=state_out,
        initial_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        state_out[state_indices].transpose(-1, -2),
        final_state_ref,
        atol=1e-2,
        rtol=1e-2,
    )


# prefill path
@pytest.mark.parametrize("seq_lens", PREFILL_SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_dims", HEAD_DIMS)
@torch.inference_mode()
def test_chunk_gated_delta_rule_cpu(
    seq_lens: list[int],
    num_heads: tuple[int, int],
    head_dims: tuple[int, int],
) -> None:
    total_tokens = sum(seq_lens)
    q, k, v, a, b, A_log, dt_bias = gdn_inputs(
        num_tokens=total_tokens,
        num_heads=num_heads,
        head_dims=head_dims,
    )
    _, num_v_heads = num_heads
    head_dim, v_head_dim = head_dims
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int32
    )
    initial_state_shape = (len(seq_lens), num_v_heads, head_dim, v_head_dim)
    initial_state = tensor_cache(
        len(seq_lens) * num_v_heads * head_dim * v_head_dim, torch.float32
    ).view(initial_state_shape)
    initial_state_ref = initial_state.transpose(-1, -2).contiguous()

    out_ref, final_state_ref = ref_gated_delta_rule(
        query=q,
        key=k,
        value=v,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=initial_state_ref,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )

    g, beta = ref_gdn_gating(A_log, a, b, dt_bias)
    out, final_state = ops.chunk_gated_delta_rule_cpu(
        query=q,
        key=k,
        value=v,
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        final_state.transpose(-1, -2),
        final_state_ref,
        atol=1e-2,
        rtol=1e-2,
    )
