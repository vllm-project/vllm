# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
import torch.nn.functional as F


def l2norm(
    x: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    if query.shape[2] != value.shape[2]:
        repeat_factor = value.shape[2] // query.shape[2]
        query = query.repeat_interleave(repeat_factor, dim=2)
        key = key.repeat_interleave(repeat_factor, dim=2)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, _ = key.shape
    v_head_dim = value.shape[-1]
    if scale is None:
        scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.empty(
        batch_size,
        num_heads,
        sequence_length,
        v_head_dim,
        dtype=value.dtype,
    )
    last_recurrent_state = initial_state.to(value)

    for token_idx in range(sequence_length):
        q_t = query[:, :, token_idx]
        k_t = key[:, :, token_idx]
        v_t = value[:, :, token_idx]
        g_t = g[:, :, token_idx].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, token_idx].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-2)).sum(dim=-1)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + delta.unsqueeze(
            -1
        ) * k_t.unsqueeze(-2)
        core_attn_out[:, :, token_idx] = (last_recurrent_state * q_t.unsqueeze(-2)).sum(
            dim=-1
        )

    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    softplus_x = F.softplus(a.float() + dt_bias.float(), beta=beta, threshold=threshold)
    g = -torch.exp(A_log.float()) * softplus_x
    beta_output = torch.sigmoid(b.float()).to(dtype=b.dtype)
    return g, beta_output


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(v)
    state_dtype = initial_state.dtype
    chunk_size = 128
    sequence_bounds = [
        (
            seq_idx,
            int(cu_seqlens[seq_idx].item()),
            int(cu_seqlens[seq_idx + 1].item()),
        )
        for seq_idx in range(len(cu_seqlens) - 1)
    ]
    chunk_eye = torch.eye(chunk_size, dtype=torch.float32)
    num_sequences = len(sequence_bounds)
    num_value_heads = v.shape[2]
    value_head_dim = v.shape[3]
    key_head_dim = k.shape[3]
    final_state = torch.empty(
        (num_sequences, num_value_heads, value_head_dim, key_head_dim),
        dtype=state_dtype,
    )

    for seq_idx, begin, end in sequence_bounds:
        q_seq = q[:, begin:end]
        k_seq = k[:, begin:end]
        v_seq = v[:, begin:end]
        g_seq = g[:, begin:end]
        beta_seq = beta[:, begin:end]

        initial_dtype = q_seq.dtype
        if use_qk_l2norm_in_kernel:
            q_seq = l2norm(q_seq, dim=-1, eps=1e-6)
            k_seq = l2norm(k_seq, dim=-1, eps=1e-6)

        num_qk_heads = q_seq.shape[2]
        num_value_heads = v_seq.shape[2]
        if num_qk_heads != num_value_heads:
            repeat_factor = num_value_heads // num_qk_heads
            q_seq = q_seq.repeat_interleave(repeat_factor, dim=2)
            k_seq = k_seq.repeat_interleave(repeat_factor, dim=2)

        q_seq, k_seq, v_seq, beta_seq, g_seq = [
            x.transpose(1, 2).contiguous().to(torch.float32)
            for x in (q_seq, k_seq, v_seq, beta_seq, g_seq)
        ]
        seq_batch_size, num_heads, seq_len, qk_head_dim = q_seq.shape
        value_head_dim = v_seq.shape[-1]

        if scale is None:
            scale = 1 / (qk_head_dim**0.5)

        q_seq = q_seq * scale

        seq_state = initial_state[seq_idx : seq_idx + 1].to(v_seq)
        seq_output = torch.empty(
            seq_batch_size,
            num_heads,
            seq_len,
            value_head_dim,
            dtype=v_seq.dtype,
        )

        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            q_chunk = q_seq[:, :, chunk_start:chunk_end]
            k_chunk = k_seq[:, :, chunk_start:chunk_end]
            v_chunk = v_seq[:, :, chunk_start:chunk_end]
            beta_chunk = beta_seq[:, :, chunk_start:chunk_end]
            g_chunk = g_seq[:, :, chunk_start:chunk_end]
            chunk_len = chunk_end - chunk_start

            cum_g = g_chunk.cumsum(dim=-1)
            exp_cum_g = cum_g.exp()
            decay = (cum_g.unsqueeze(-1) - cum_g.unsqueeze(-2)).exp()

            interaction = (k_chunk * beta_chunk.unsqueeze(-1)) @ k_chunk.transpose(
                -1, -2
            )
            interaction = torch.tril(interaction * decay, diagonal=-1)
            system = interaction + chunk_eye[:chunk_len, :chunk_len]

            solved_values = torch.linalg.solve_triangular(
                system,
                v_chunk * beta_chunk.unsqueeze(-1),
                upper=False,
            )
            solved_keys = torch.linalg.solve_triangular(
                system,
                (k_chunk * beta_chunk.unsqueeze(-1)) * exp_cum_g.unsqueeze(-1),
                upper=False,
            )

            incoming_memory = torch.einsum("bhvk,bhck->bhcv", seq_state, solved_keys)
            transformed_values = solved_values - incoming_memory

            # Each chunk contributes both from the incoming recurrent state and
            # from its own in-chunk interactions.
            inter_chunk = torch.einsum(
                "bhvk,bhck->bhcv",
                seq_state,
                q_chunk * exp_cum_g.unsqueeze(-1),
            )
            intra_chunk = torch.tril((q_chunk @ k_chunk.transpose(-1, -2)) * decay)
            seq_output[:, :, chunk_start:chunk_end] = (
                inter_chunk + intra_chunk @ transformed_values
            )

            # Carry the recurrent state forward to the next chunk boundary.
            end_decay = (cum_g[:, :, -1:] - cum_g).exp().unsqueeze(-1)
            decayed_keys = k_chunk * end_decay
            seq_state = seq_state * exp_cum_g[:, :, -1, None, None] + torch.einsum(
                "bhcv,bhck->bhvk", transformed_values, decayed_keys
            )

        output[0, begin:end].copy_(
            seq_output.transpose(1, 2).contiguous().to(initial_dtype).squeeze(0)
        )
        final_state[seq_idx].copy_(seq_state.squeeze(0).to(state_dtype).contiguous())

    return output, final_state
