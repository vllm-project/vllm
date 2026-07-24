# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import types

import pytest
import torch
import torch.nn.functional as F

import vllm._custom_ops as ops
from vllm.model_executor.layers.mamba.ops.cpu import gdn_attention
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
CONV_DIM = 128
CONV_KERNEL = 4
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


# (total_tokens, split) pairs mimicking where chunked prefill breaks a sequence
# across two scheduler steps: chunk-aligned and non-aligned splits.
TWO_CALL_SPLITS = [
    (2 * CHUNK_SIZE, CHUNK_SIZE),
    (2 * CHUNK_SIZE + 17, CHUNK_SIZE),
    (2 * CHUNK_SIZE + 17, CHUNK_SIZE + 9),
    (4 * CHUNK_SIZE + 17, 2 * CHUNK_SIZE),
    (3 * CHUNK_SIZE, CHUNK_SIZE + 1),
]


@pytest.mark.parametrize("total_tokens, split", TWO_CALL_SPLITS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_dims", HEAD_DIMS)
@torch.inference_mode()
def test_chunk_gated_delta_rule_cpu_two_call_split(
    total_tokens: int,
    split: int,
    num_heads: tuple[int, int],
    head_dims: tuple[int, int],
) -> None:
    """A prefill split into two calls (the second seeded with the first's
    ``final_state`` and a rebased ``cu_seqlens``) must match the single-call
    result, mimicking the cross-scheduler-step handoff in
    ``cpu_gdn_attention_core``.
    """
    q, k, v, a, b, A_log, dt_bias = gdn_inputs(
        num_tokens=total_tokens,
        num_heads=num_heads,
        head_dims=head_dims,
    )
    _, num_v_heads = num_heads
    head_dim, v_head_dim = head_dims

    g, beta = ref_gdn_gating(A_log, a, b, dt_bias)
    g = g.unsqueeze(0)  # [1, T, HV]
    beta = beta.unsqueeze(0)

    zero_state = torch.zeros(1, num_v_heads, head_dim, v_head_dim, dtype=torch.float32)

    # Reference: whole sequence in one call, no initial state.
    out_full, final_full = ops.chunk_gated_delta_rule_cpu(
        query=q,
        key=k,
        value=v,
        g=g,
        beta=beta,
        initial_state=zero_state,
        output_final_state=True,
        cu_seqlens=torch.tensor([0, total_tokens], dtype=torch.int32),
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )

    # Call 1: tokens [0:split], no initial state, capture final state.
    out1, state1 = ops.chunk_gated_delta_rule_cpu(
        query=q[:, :split],
        key=k[:, :split],
        value=v[:, :split],
        g=g[:, :split],
        beta=beta[:, :split],
        initial_state=zero_state,
        output_final_state=True,
        cu_seqlens=torch.tensor([0, split], dtype=torch.int32),
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    # Call 2: tokens [split:T] seeded with call 1's final state and a cu_seqlens
    # rebased to start at 0, as cpu_gdn_attention_core continues a prefill chunk.
    tail = total_tokens - split
    out2, state2 = ops.chunk_gated_delta_rule_cpu(
        query=q[:, split:],
        key=k[:, split:],
        value=v[:, split:],
        g=g[:, split:],
        beta=beta[:, split:],
        initial_state=state1.to(torch.float32),
        output_final_state=True,
        cu_seqlens=torch.tensor([0, tail], dtype=torch.int32),
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )

    out_split = torch.cat([out1, out2], dim=1)

    # State must be near-exact; output allows a looser bound for the bf16 round-trip.
    torch.testing.assert_close(state2, final_full, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(out_split, out_full, atol=2e-2, rtol=2e-2)


def _conv_inputs(total_tokens: int):
    x = tensor_cache(total_tokens * CONV_DIM, torch.bfloat16).view(
        total_tokens, CONV_DIM
    )
    weight = tensor_cache(CONV_DIM * CONV_KERNEL, torch.bfloat16).view(
        CONV_DIM, CONV_KERNEL
    )
    bias = tensor_cache(CONV_DIM, torch.bfloat16)
    return x, weight, bias


def _sd_conv_states(
    num_slots: int, state_len: int, dim: int = CONV_DIM
) -> torch.Tensor:
    storage = torch.zeros(num_slots, state_len, dim, dtype=torch.bfloat16)
    return storage.transpose(1, 2)


def _maybe_pack_conv_weight(weight: torch.Tensor, is_vnni: bool) -> torch.Tensor:
    return ops.causal_conv1d_weight_pack(weight) if is_vnni else weight


@torch.inference_mode()
def test_spec_aware_mixed_routing_preserves_token_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_tokens = 4
    projection = torch.arange(num_tokens * 16, dtype=torch.float32).view(num_tokens, 16)
    mixed_qkv, b, a = projection[:, :4], projection[:, 4:6], projection[:, 6:8]
    assert all(not tensor.is_contiguous() for tensor in (mixed_qkv, b, a))

    spec_indices = torch.tensor([0, 2])
    nonspec_indices = torch.tensor([1, 3])
    metadata = types.SimpleNamespace(
        spec_sequence_masks=torch.ones(1, dtype=torch.bool),
        spec_token_indx=spec_indices,
        non_spec_token_indx=nonspec_indices,
        num_prefills=1,
        num_decodes=0,
    )
    routed = []

    def record(*args):
        routed.append(args[2:5])
        return args[2]

    monkeypatch.setattr(gdn_attention, "is_conv_state_dim_first", lambda: True)
    monkeypatch.setattr(gdn_attention, "_spec_forward", record)
    monkeypatch.setattr(gdn_attention, "_spec_aware_nonspec_subset", record)

    layer = types.SimpleNamespace(
        kv_cache=[torch.empty(1, 4, 6), torch.empty(1, 1, 1, 1)]
    )
    core_attn_out = torch.empty_like(mixed_qkv)
    gdn_attention._cpu_gdn_attention_spec_aware(
        layer=layer,
        attn_metadata_i=metadata,
        mixed_qkv=mixed_qkv,
        b=b,
        a=a,
        core_attn_out=core_attn_out,
        width=CONV_KERNEL,
        state_len=6,
    )

    expected_inputs = (mixed_qkv, b, a)
    assert len(routed) == 2
    for actual_inputs, indices in zip(routed, (spec_indices, nonspec_indices)):
        for actual, expected in zip(actual_inputs, expected_inputs):
            assert actual.is_contiguous()
            torch.testing.assert_close(actual, expected.index_select(0, indices))
    torch.testing.assert_close(core_attn_out, mixed_qkv)


@torch.inference_mode()
def test_spec_aware_nonspec_materializes_state_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block_table = torch.arange(8, dtype=torch.int32).view(2, 4)
    state_indices = block_table[:, 0]
    assert not state_indices.is_contiguous()

    metadata = types.SimpleNamespace(
        non_spec_state_indices_tensor=state_indices,
        non_spec_query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=2,
        num_prefill_tokens=4,
        has_initial_state=torch.tensor([False, False]),
    )

    recorded_indices = None

    def causal_conv1d_fwd_cpu(**kwargs):
        nonlocal recorded_indices
        recorded_indices = kwargs["cache_indices"]
        return kwargs["x"]

    def fused_gdn_gating_cpu(**kwargs):
        return kwargs["a"], kwargs["b"]

    def chunk_gated_delta_rule_cpu(**kwargs):
        out = torch.zeros(1, 4, 1, 1)
        return out, kwargs["initial_state"]

    monkeypatch.setattr(torch.cpu, "_is_amx_tile_supported", lambda: True)
    monkeypatch.setattr(gdn_attention, "is_conv_state_dim_first", lambda: False)
    monkeypatch.setattr(
        gdn_attention.ops, "causal_conv1d_fwd_cpu", causal_conv1d_fwd_cpu
    )
    monkeypatch.setattr(gdn_attention.ops, "fused_gdn_gating_cpu", fused_gdn_gating_cpu)
    monkeypatch.setattr(
        gdn_attention.ops,
        "chunk_gated_delta_rule_cpu",
        chunk_gated_delta_rule_cpu,
    )

    layer = types.SimpleNamespace(
        activation="silu",
        conv1d=types.SimpleNamespace(weight=torch.empty(0), bias=None),
        A_log=torch.empty(0),
        dt_bias=torch.empty(0),
        rearrange_mixed_qkv=lambda x: (
            x[:, :1].view(1, 4, 1, 1),
            x[:, :1].view(1, 4, 1, 1),
            x[:, :1].view(1, 4, 1, 1),
        ),
    )
    gdn_attention._spec_aware_nonspec(
        layer=layer,
        attn_metadata_i=metadata,
        mixed_qkv=torch.zeros(4, 4),
        b=torch.zeros(4, 1),
        a=torch.zeros(4, 1),
        core_attn_out=torch.zeros(4, 1, 1),
        conv_buf=torch.zeros(8, 1, 6),
        ssm_state=torch.zeros(8, 1, 1, 1),
        width=4,
    )

    assert recorded_indices is not None
    assert recorded_indices.is_contiguous()
    torch.testing.assert_close(
        recorded_indices, torch.tensor([0, 4], dtype=torch.int32)
    )


@torch.inference_mode()
def test_spec_forward_prepares_native_conv_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block_table = torch.tensor([[0, 1], [4, 5]], dtype=torch.int32)
    state_indices = block_table[:, 0]
    accepted_counts = torch.tensor([1, 4], dtype=torch.int32)
    assert not state_indices.is_contiguous()
    metadata = types.SimpleNamespace(
        num_spec_decodes=2,
        spec_state_indices_tensor=block_table,
        spec_query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        num_accepted_tokens=accepted_counts,
    )
    forwarded_indices = None
    forwarded_counts = None

    def causal_conv1d_update_cpu(**kwargs):
        nonlocal forwarded_counts, forwarded_indices
        forwarded_indices = kwargs["conv_state_indices"]
        forwarded_counts = kwargs["num_accepted_tokens"]
        return kwargs["x"]

    monkeypatch.setattr(torch.cpu, "_is_amx_tile_supported", lambda: True)
    monkeypatch.setattr(gdn_attention, "is_conv_state_dim_first", lambda: False)
    monkeypatch.setattr(
        gdn_attention.ops, "causal_conv1d_update_cpu", causal_conv1d_update_cpu
    )
    monkeypatch.setattr(
        gdn_attention.ops,
        "fused_sigmoid_gating_delta_rule_update_spec_cpu",
        lambda **kwargs: kwargs["q"],
    )

    layer = types.SimpleNamespace(
        activation="silu",
        conv1d=types.SimpleNamespace(weight=torch.empty(1, CONV_KERNEL), bias=None),
        A_log=None,
        dt_bias=None,
        rearrange_mixed_qkv=lambda x: (x.unsqueeze(0),) * 3,
    )
    gdn_attention._spec_forward(
        layer=layer,
        attn_metadata_i=metadata,
        mixed_qkv_spec=torch.zeros(8, 1, dtype=torch.bfloat16),
        b_spec=torch.empty(0),
        a_spec=torch.empty(0),
        conv_buf=torch.empty(0),
        ssm_state=torch.empty(0),
        width=CONV_KERNEL,
        state_len=0,
    )

    expected = (
        (forwarded_indices, torch.tensor([0, 4], dtype=torch.int32)),
        (forwarded_counts, torch.tensor([1, 4], dtype=torch.int32)),
    )
    for actual, reference in expected:
        assert actual is not None
        assert actual.is_contiguous()
        assert actual.dtype == torch.int32
        torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize("total_tokens, split", TWO_CALL_SPLITS)
@torch.inference_mode()
def test_causal_conv1d_torch_two_call_split(total_tokens: int, split: int) -> None:
    """Non-AMX conv-state handoff: a two-call split (the second seeded via
    ``has_initial_state=True`` from the conv_states the first wrote back) must
    match the single-call result.
    """
    from vllm.model_executor.layers.mamba.ops.cpu.causal_conv1d import (
        causal_conv1d_fn_cpu as causal_conv1d_torch,
    )

    x, weight, bias = _conv_inputs(total_tokens)
    state_len = CONV_KERNEL - 1
    # [num_slots, conv_dim, state_len]; slot 0 used here.
    conv_states_full = torch.zeros(1, CONV_DIM, state_len, dtype=x.dtype)
    conv_states_split = torch.zeros(1, CONV_DIM, state_len, dtype=x.dtype)

    # x is [conv_dim, T] for causal_conv1d_torch.
    xt = x.transpose(0, 1).contiguous()

    out_full = causal_conv1d_torch(
        x=xt,
        weight=weight,
        bias=bias,
        conv_states=conv_states_full,
        query_start_loc=torch.tensor([0, total_tokens], dtype=torch.int32),
        cache_indices=torch.tensor([0], dtype=torch.int32),
        has_initial_state=torch.tensor([False]),
        activation="silu",
    )

    out1 = causal_conv1d_torch(
        x=xt[:, :split],
        weight=weight,
        bias=bias,
        conv_states=conv_states_split,
        query_start_loc=torch.tensor([0, split], dtype=torch.int32),
        cache_indices=torch.tensor([0], dtype=torch.int32),
        has_initial_state=torch.tensor([False]),
        activation="silu",
    )
    out2 = causal_conv1d_torch(
        x=xt[:, split:],
        weight=weight,
        bias=bias,
        conv_states=conv_states_split,
        query_start_loc=torch.tensor([0, total_tokens - split], dtype=torch.int32),
        cache_indices=torch.tensor([0], dtype=torch.int32),
        has_initial_state=torch.tensor([True]),
        activation="silu",
    )
    out_split = torch.cat([out1, out2], dim=1)

    torch.testing.assert_close(out_split, out_full, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(
    not torch.cpu._is_amx_tile_supported(),
    reason="requires AMX support",
)
@torch.inference_mode()
def test_causal_conv1d_update_cpu_accepts_wide_state() -> None:
    state_len = CONV_KERNEL - 1
    wide_state_len = state_len + 5
    batch_size = 3
    is_vnni = True
    x, weight, bias = _conv_inputs(batch_size)
    conv_state_indices = torch.tensor([2, 0, 1], dtype=torch.int32)

    narrow_state = _sd_conv_states(batch_size, state_len)
    narrow_state.copy_(
        tensor_cache(narrow_state.numel(), torch.bfloat16).view_as(narrow_state)
    )
    wide_state = _sd_conv_states(batch_size, wide_state_len)
    wide_state[:, :, :state_len].copy_(narrow_state)
    wide_state[:, :, state_len:].fill_(7)
    wide_tail = wide_state[:, :, state_len:].clone()

    conv_weight = _maybe_pack_conv_weight(weight, is_vnni)
    out_narrow = ops.causal_conv1d_update_cpu(
        x=x,
        conv_states=narrow_state,
        weight=conv_weight,
        bias=bias,
        silu_activation=True,
        conv_state_indices=conv_state_indices,
        is_vnni=is_vnni,
    )
    out_wide = ops.causal_conv1d_update_cpu(
        x=x,
        conv_states=wide_state,
        weight=conv_weight,
        bias=bias,
        silu_activation=True,
        conv_state_indices=conv_state_indices,
        is_vnni=is_vnni,
    )

    torch.testing.assert_close(out_wide, out_narrow, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        wide_state[:, :, :state_len], narrow_state, atol=0, rtol=0
    )
    torch.testing.assert_close(wide_state[:, :, state_len:], wide_tail, atol=0, rtol=0)


def _ref_causal_conv1d_update_cpu_multi(
    x: torch.Tensor,
    conv_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    silu_activation: bool,
    conv_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len, dim = x.shape
    state_len = conv_states.size(2)
    conv_out = torch.empty_like(x)
    conv_weight = weight.unsqueeze(1)

    for i in range(batch_size):
        slot = int(conv_state_indices[i].item())
        offset = int(num_accepted_tokens[i].item()) - 1
        state = conv_states[slot]
        x_seq = x[i].transpose(0, 1).to(state.dtype)
        prior = state[:, offset : offset + CONV_KERNEL - 1]
        conv_in = torch.cat([prior, x_seq], dim=-1).unsqueeze(0)
        out = F.conv1d(conv_in, conv_weight, bias, groups=dim)[0]
        if silu_activation:
            out = F.silu(out)
        conv_out[i] = out.transpose(0, 1).to(conv_out.dtype)
        keep = state[:, offset + 1 : offset + 1 + (state_len - seq_len)]
        state.copy_(torch.cat([keep, x_seq], dim=-1))

    return conv_out


@pytest.mark.skipif(
    not torch.cpu._is_amx_tile_supported(),
    reason="requires AMX support",
)
@pytest.mark.parametrize(
    ("batch_size, seq_len, accepted_counts, has_bias, silu_activation, is_vnni"),
    [
        (1, 1, [1], False, False, False),
        (1, 1, [1], True, True, True),
        (4, 4, [1, 2, 3, 4], False, True, False),
        (4, 4, [4, 3, 2, 1], True, False, True),
        (4, 16, [1, 5, 10, 16], False, False, True),
        (4, 16, [16, 10, 5, 1], True, True, False),
    ],
)
@torch.inference_mode()
def test_causal_conv1d_update_cpu_multi_token_matches_python(
    batch_size: int,
    seq_len: int,
    accepted_counts: list[int],
    has_bias: bool,
    silu_activation: bool,
    is_vnni: bool,
) -> None:
    dim = 96
    state_len = seq_len + 2
    x = tensor_cache(batch_size * seq_len * dim, torch.bfloat16).view(
        batch_size, seq_len, dim
    )
    weight = tensor_cache(dim * CONV_KERNEL, torch.bfloat16).view(dim, CONV_KERNEL)
    bias = tensor_cache(dim, torch.bfloat16) if has_bias else None
    conv_state_indices = torch.arange(batch_size - 1, -1, -1, dtype=torch.int32)
    num_accepted_tokens = torch.tensor(accepted_counts, dtype=torch.int32)

    conv_states_ref = _sd_conv_states(batch_size, state_len, dim)
    conv_states_ref.copy_(
        tensor_cache(conv_states_ref.numel(), torch.bfloat16).view_as(conv_states_ref)
    )
    conv_states = conv_states_ref.clone()

    conv_weight = _maybe_pack_conv_weight(weight, is_vnni)
    out = ops.causal_conv1d_update_cpu(
        x=x,
        conv_states=conv_states,
        weight=conv_weight,
        bias=bias,
        silu_activation=silu_activation,
        conv_state_indices=conv_state_indices,
        is_vnni=is_vnni,
        num_accepted_tokens=num_accepted_tokens,
    )
    ref_out = _ref_causal_conv1d_update_cpu_multi(
        x=x,
        conv_states=conv_states_ref,
        weight=weight,
        bias=bias,
        silu_activation=silu_activation,
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
    )

    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(conv_states, conv_states_ref, atol=0, rtol=0)


@pytest.mark.skipif(
    not torch.cpu._is_amx_tile_supported(),
    reason="requires AMX support",
)
@pytest.mark.parametrize("num_accepted", [0, 17])
@torch.inference_mode()
def test_causal_conv1d_update_cpu_rejects_invalid_accepted_count(
    num_accepted: int,
) -> None:
    batch_size = 1
    seq_len = 16
    dim = 96
    state_len = seq_len + 2
    x = torch.zeros(batch_size, seq_len, dim, dtype=torch.bfloat16)
    weight = torch.zeros(dim, CONV_KERNEL, dtype=torch.bfloat16)
    conv_states = _sd_conv_states(batch_size, state_len, dim)

    with pytest.raises(RuntimeError, match="num_accepted_tokens must be in.*seqlen"):
        ops.causal_conv1d_update_cpu(
            x=x,
            conv_states=conv_states,
            weight=weight,
            bias=None,
            silu_activation=True,
            conv_state_indices=torch.tensor([0], dtype=torch.int32),
            is_vnni=False,
            num_accepted_tokens=torch.tensor([num_accepted], dtype=torch.int32),
        )


@pytest.mark.skipif(
    not torch.cpu._is_amx_tile_supported(),
    reason="requires AMX support",
)
@pytest.mark.parametrize("total_tokens, split", TWO_CALL_SPLITS)
@torch.inference_mode()
def test_causal_conv1d_fwd_cpu_two_call_split(total_tokens: int, split: int) -> None:
    """AMX prefill conv op must honor ``has_initial_state`` so a two-call split
    matches the single-call result.

    Regression test for ``causal_conv1d_fwd_varlen_kernel_impl`` (``conv.cpp``)
    ignoring the carried conv state on continued chunks.
    """
    state_len = CONV_KERNEL - 1
    x, weight, bias = _conv_inputs(total_tokens)

    def amx(x_seg, conv_states, has_init):
        seq = x_seg.shape[0]
        return ops.causal_conv1d_fwd_cpu(
            x=x_seg.transpose(0, 1),  # [dim, seq]; stride(-2)==1 (view of [seq,dim])
            weight=weight,
            bias=bias,
            conv_states=conv_states,
            query_start_loc=torch.tensor([0, seq], dtype=torch.int32),
            cache_indices=torch.tensor([0], dtype=torch.int32),
            has_initial_state=torch.tensor([has_init]),
            silu_activation=True,
            is_vnni=False,
        ).contiguous()

    # conv_state layout passed by the AMX branch: [num_slots, dim, state_len].
    cs_full = torch.zeros(1, CONV_DIM, state_len, dtype=x.dtype)
    out_full = amx(x, cs_full, False)

    cs_split = torch.zeros(1, CONV_DIM, state_len, dtype=x.dtype)
    out1 = amx(x[:split], cs_split, False)
    out2 = amx(x[split:], cs_split, True)
    out_split = torch.cat([out1, out2], dim=1)

    torch.testing.assert_close(out_split, out_full, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(
    not torch.cpu._is_amx_tile_supported(),
    reason="requires AMX support",
)
@torch.inference_mode()
def test_causal_conv1d_fwd_cpu_accepts_wide_state() -> None:
    state_len = CONV_KERNEL - 1
    wide_state_len = state_len + 5
    is_vnni = True
    seq_lens = [CHUNK_SIZE - 1, CHUNK_SIZE + 5]
    total_tokens = sum(seq_lens)
    x, weight, bias = _conv_inputs(total_tokens)
    query_start_loc = torch.tensor([0, seq_lens[0], total_tokens], dtype=torch.int32)
    cache_indices = torch.tensor([2, 0], dtype=torch.int32)
    has_initial_state = torch.tensor([True, False])

    narrow_state = _sd_conv_states(3, state_len)
    narrow_state.copy_(
        tensor_cache(narrow_state.numel(), torch.bfloat16).view_as(narrow_state)
    )
    wide_state = _sd_conv_states(3, wide_state_len)
    wide_state[:, :, :state_len].copy_(narrow_state)
    wide_state[:, :, state_len:].fill_(7)
    wide_tail = wide_state[:, :, state_len:].clone()

    conv_weight = _maybe_pack_conv_weight(weight, is_vnni)
    out_narrow = ops.causal_conv1d_fwd_cpu(
        x=x.transpose(0, 1),
        weight=conv_weight,
        bias=bias,
        conv_states=narrow_state,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        silu_activation=True,
        is_vnni=is_vnni,
    )
    out_wide = ops.causal_conv1d_fwd_cpu(
        x=x.transpose(0, 1),
        weight=conv_weight,
        bias=bias,
        conv_states=wide_state,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        silu_activation=True,
        is_vnni=is_vnni,
    )

    torch.testing.assert_close(out_wide, out_narrow, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        wide_state[:, :, :state_len], narrow_state, atol=0, rtol=0
    )
    torch.testing.assert_close(wide_state[:, :, state_len:], wide_tail, atol=0, rtol=0)


@torch.inference_mode()
def test_batch_memcpy_cpu_fallback() -> None:
    """The ctypes batch_memcpy fallback (used when triton-cpu is absent) must
    copy each src into its dst, validating the (src_ptrs, dst_ptrs, sizes)
    argument order against ctypes.memmove(dst, src, size).
    """
    from vllm.utils.cpu_triton_utils import batch_memcpy_kernel

    # Varied byte sizes, including a non-power-of-two run.
    sizes_bytes = [256, 1024, 17 * 4, 4096]
    srcs = [torch.rand(n // 4, dtype=torch.float32) for n in sizes_bytes]
    dsts = [torch.zeros_like(s) for s in srcs]

    src_ptrs = torch.tensor([s.data_ptr() for s in srcs], dtype=torch.uint64)
    dst_ptrs = torch.tensor([d.data_ptr() for d in dsts], dtype=torch.uint64)
    sizes = torch.tensor(sizes_bytes, dtype=torch.int32)

    batch_memcpy_kernel[(len(srcs),)](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=1024)

    for src, dst in zip(srcs, dsts):
        torch.testing.assert_close(dst, src)
