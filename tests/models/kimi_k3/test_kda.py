# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision tests for vllm's chunk_kda Triton operator.

Compares chunk_kda against a naive recurrent reference (float32).
Uses torch.rand for q/k/v to match FLA's test pattern.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
    gather_initial_states,
)
from vllm.models.kimi_k3.amd.ops.third_party.kda import (
    fused_recurrent_kda_packed_decode as fused_recurrent_kda_packed_decode_amd,
)
from vllm.models.kimi_k3.nvidia.kda import (
    is_flashkda_supported,
    is_fused_kda_decode_supported,
)
from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
    chunk_kda,
    chunk_kda_with_fused_gate,
    fused_kda_gate,
    fused_recurrent_kda,
    fused_recurrent_kda_fwd,
    fused_recurrent_kda_packed_decode,
)
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

DEVICE = current_platform.device_type

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="The KDA kernels require a CUDA-alike or XPU device.",
)

# The AMD and NVIDIA copies of the KDA kernels are vendored separately and are
# free to diverge, so the shared-semantics tests below run against both.
PACKED_DECODE_IMPLS = {
    "nvidia": fused_recurrent_kda_packed_decode,
    "amd": fused_recurrent_kda_packed_decode_amd,
}


@torch.inference_mode()
def test_gather_initial_states_correctness():
    row_size = 8 * 128 * 128
    storage = torch.randn(5, row_size + 256, dtype=torch.float32, device=DEVICE)
    state = storage[:, :row_size].view(5, 8, 128, 128)
    assert not state.is_contiguous()
    assert state[0].is_contiguous()
    indices = torch.tensor([4, 1, 3], dtype=torch.int32, device=DEVICE)
    has_initial_state = torch.tensor([True, False, True], device=DEVICE)

    expected = state[indices].clone()
    expected[~has_initial_state] = 0

    torch.testing.assert_close(
        gather_initial_states(state, indices, has_initial_state),
        expected,
    )


def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Naive recurrent KDA reference, ported from FLA's naive.py."""
    dtype = v.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K**-0.5

    q, k, v, g, beta = (x.to(torch.float) for x in [q, k, v, g, beta])
    q = q * scale

    S = k.new_zeros(B, H, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    for i in range(T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum(
            "bhk,bhv->bhkv",
            b_i[..., None] * k_i,
            v_i - (k_i[..., None] * S).sum(-2),
        )
        o[:, i] = torch.einsum("bhk,bhkv->bhv", q_i, S)
    if not output_final_state:
        S = None
    return o.to(dtype), S


def assert_close(
    name: str,
    ref: torch.Tensor,
    tri: torch.Tensor,
    ratio: float,
    err_atol: float = 1e-6,
):
    """RMSE-based relative error comparison."""
    abs_err = (ref.detach() - tri.detach()).flatten().abs().max().item()
    rmse_diff = (ref.detach() - tri.detach()).flatten().square().mean().sqrt().item()
    rmse_base = ref.detach().flatten().square().mean().sqrt().item()
    rel_err = rmse_diff / (rmse_base + 1e-8)
    print(f"{name:>4} | abs={abs_err:.6f} | rmse={rel_err:.6f} | thr={ratio}")
    if abs_err <= err_atol:
        return
    assert not torch.isnan(ref).any(), f"{name}: NaN detected in ref"
    assert not torch.isnan(tri).any(), f"{name}: NaN detected in tri"
    assert rel_err < ratio, (
        f"{name}: max abs err {abs_err:.6f}, rmse ratio {rel_err:.6f} >= {ratio}"
    )


@pytest.mark.parametrize(
    ("H", "D", "cu_seqlens", "dtype"),
    [
        pytest.param(
            *test,
            id="H{}-D{}-cu{}-{}".format(*test),
        )
        for test in [
            (32, 128, [0, 64], torch.float16),
            (32, 128, [0, 1024], torch.float16),
            (32, 128, [0, 15], torch.float16),
            (32, 128, [0, 256, 512, 768, 1024], torch.float16),
            (32, 128, [0, 15, 100, 300, 1200], torch.float16),
            (64, 128, [0, 256, 500, 1000], torch.float16),
            (32, 128, [0, 8192], torch.float16),
            (32, 128, [0, 256, 500, 1000], torch.bfloat16),
        ]
    ],
)
@torch.inference_mode()
def test_chunk_kda(
    H: int,
    D: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    T = cu_seqlens[-1]
    torch.manual_seed(42)
    B = 1
    cu_seqlens_t = torch.LongTensor(cu_seqlens).to(DEVICE)
    N = len(cu_seqlens) - 1

    q = torch.rand(B, T, H, D, dtype=dtype, device=DEVICE)
    k = torch.rand(B, T, H, D, dtype=dtype, device=DEVICE)
    v = torch.rand(B, T, H, D, dtype=dtype, device=DEVICE)
    g = F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float32, device=DEVICE)).to(
        dtype
    )
    beta = torch.rand(B, T, H, dtype=dtype, device=DEVICE).sigmoid()
    h0 = torch.randn(N, H, D, D, dtype=torch.float32, device=DEVICE)

    # Naive reference with l2norm_fwd (same kernel as chunk_kda)
    ref_outputs = []
    ref_states = []
    for i in range(N):
        s, e = cu_seqlens[i], cu_seqlens[i + 1]
        q_i = l2norm_fwd(q[:, s:e].contiguous())
        k_i = l2norm_fwd(k[:, s:e].contiguous())
        o_i, ht_i = naive_recurrent_kda(
            q_i,
            k_i,
            v[:, s:e],
            g[:, s:e],
            beta[:, s:e],
            initial_state=h0[i],
            output_final_state=True,
        )
        ref_outputs.append(o_i)
        ref_states.append(ht_i)
    ref_o = torch.cat(ref_outputs, dim=1)
    ref_ht = torch.cat(ref_states, dim=0)

    # h0 transposed to (V, K) layout for the kernel; naive uses (K, V)
    tri_o, tri_ht = chunk_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        initial_state=h0.transpose(-1, -2).contiguous().clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens_t,
        use_qk_l2norm_in_kernel=True,
    )

    assert not torch.isnan(tri_o).any(), "Triton output o contains NaN"
    assert not torch.isnan(tri_ht).any(), "Triton output ht contains NaN"
    assert_close("o", ref_o, tri_o, 0.005)
    assert_close("ht", ref_ht, tri_ht.transpose(-1, -2).contiguous(), 0.005)


@pytest.mark.parametrize(
    ("cu_seqlens", "dtype", "lower_bound"),
    [
        ([0, 64], torch.float16, None),
        ([0, 15, 100, 300], torch.bfloat16, None),
        ([0, 15, 100, 300], torch.bfloat16, -3.0),
    ],
)
@torch.inference_mode()
def test_chunk_kda_fused_gate_cumsum_matches_unfused(
    cu_seqlens: list[int],
    dtype: torch.dtype,
    lower_bound: float | None,
):
    H, D = 8, 64
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1
    torch.manual_seed(123)

    cu_seqlens_t = torch.tensor(cu_seqlens, dtype=torch.int32, device=DEVICE)
    q = torch.randn(1, T, H, D, dtype=dtype, device=DEVICE)
    k = torch.randn(1, T, H, D, dtype=dtype, device=DEVICE)
    v = torch.randn(1, T, H, D, dtype=dtype, device=DEVICE)
    raw_g = torch.randn(1, T, H, D, dtype=dtype, device=DEVICE)
    beta_storage = torch.randn(1, T, 2 * H + 3, dtype=dtype, device=DEVICE)
    raw_beta = beta_storage[..., 1 : 2 * H + 1 : 2]
    beta = raw_beta.float().sigmoid()
    A_log = (torch.randn(H, dtype=torch.float32, device=DEVICE) * 0.5).contiguous()
    dt_bias = (
        torch.randn(H * D, dtype=torch.float32, device=DEVICE) * 0.1
    ).contiguous()
    h0 = torch.randn(N, H, D, D, dtype=torch.float32, device=DEVICE)
    initial_state = h0.transpose(-1, -2).contiguous()

    gate = fused_kda_gate(
        raw_g.reshape(T, H * D),
        A_log,
        D,
        g_bias=dt_bias,
        lower_bound=lower_bound,
    )
    if lower_bound is not None:
        expected_gate = lower_bound * torch.sigmoid(
            A_log.exp()[None, :, None]
            * (raw_g.float().view(T, H, D) + dt_bias.view(H, D))
        )
        torch.testing.assert_close(gate, expected_gate)
    gate = gate.unsqueeze(0)
    old_o, old_ht = chunk_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=gate,
        beta=beta,
        initial_state=initial_state.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens_t,
        use_qk_l2norm_in_kernel=True,
    )
    new_o, new_ht = chunk_kda_with_fused_gate(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        g_bias=dt_bias,
        lower_bound=lower_bound,
        initial_state=initial_state.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens_t,
        use_qk_l2norm_in_kernel=True,
    )

    assert_close("o", old_o, new_o, 1e-3, err_atol=1e-3)
    assert_close("ht", old_ht, new_ht, 1e-3, err_atol=1e-3)


@pytest.mark.parametrize("num_seqs", [1, 8, 32])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
@pytest.mark.parametrize("state_indices_stride", [1, 8])
@pytest.mark.parametrize("impl", PACKED_DECODE_IMPLS.keys())
@torch.inference_mode()
def test_packed_kda_decode_correctness(
    num_seqs: int,
    lower_bound: float | None,
    state_indices_stride: int,
    impl: str,
):
    H, D = 8, 128
    torch.manual_seed(321)

    packed_storage = torch.randn(
        num_seqs,
        3 * H * D + 1,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    mixed_qkv = packed_storage[:, : 3 * H * D]
    assert mixed_qkv.stride(0) == 3 * H * D + 1
    q, k, v = (
        x.contiguous().view(1, num_seqs, H, D) for x in mixed_qkv.split(H * D, dim=-1)
    )
    raw_g = torch.randn(
        1,
        num_seqs,
        H,
        D,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    raw_beta = torch.randn(
        1,
        num_seqs,
        H,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    beta = raw_beta.float().sigmoid()
    A_log = torch.randn(H, dtype=torch.float32, device=DEVICE) * 0.5
    dt_bias = torch.randn(H, D, dtype=torch.float32, device=DEVICE) * 0.1
    state_storage = torch.randn(
        num_seqs + 1,
        H * D * D + 17,
        dtype=torch.float32,
        device=DEVICE,
    )
    state = state_storage[:, : H * D * D].view(num_seqs + 1, H, D, D)
    assert not state.is_contiguous()
    assert state.stride()[1:] == (D * D, D, 1)
    state_indices_storage = torch.zeros(
        num_seqs,
        state_indices_stride,
        dtype=torch.int32,
        device=DEVICE,
    )
    state_indices = state_indices_storage[:, 0]
    state_indices.copy_(
        torch.arange(
            1,
            num_seqs + 1,
            dtype=torch.int32,
            device=DEVICE,
        )
    )
    gate = fused_kda_gate(
        raw_g.reshape(num_seqs, H * D),
        A_log,
        D,
        g_bias=dt_bias,
        lower_bound=lower_bound,
    ).unsqueeze(0)
    dense_state = state.clone()
    dense_out, _ = fused_recurrent_kda_fwd(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        scale=D**-0.5,
        initial_state=dense_state,
        inplace_final_state=True,
        cu_seqlens=torch.arange(
            num_seqs + 1,
            dtype=torch.int32,
            device=DEVICE,
        ),
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
    )
    packed_state = state
    packed_out, _ = PACKED_DECODE_IMPLS[impl](
        mixed_qkv=mixed_qkv,
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        initial_state=packed_state,
        state_indices=state_indices,
    )

    assert_close("o", dense_out, packed_out, 1e-3, err_atol=1e-3)
    assert_close("ht", dense_state, packed_state, 1e-3, err_atol=1e-3)


@pytest.mark.parametrize(
    ("H", "fuse_gate"),
    [(12, True), (12, False), (12, None), (96, None)],
)
@pytest.mark.parametrize("lower_bound", [-5.0, None])
@torch.inference_mode()
def test_kda_spec_decode_correctness(
    H: int,
    fuse_gate: bool | None,
    lower_bound: float | None,
):
    num_seqs, query_len, D = 3, 3, 128
    T = num_seqs * query_len
    torch.manual_seed(1234)

    qkv_storage = torch.randn(
        1,
        T,
        3 * H * D + 7,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    packed_qkv = qkv_storage[..., : 3 * H * D]
    q, k, v = (x.view(1, T, H, D) for x in packed_qkv.split(H * D, dim=-1))
    gate_storage = torch.randn(
        1,
        T,
        H * D + 5,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    raw_g = gate_storage[..., : H * D].view(1, T, H, D)
    beta_storage = torch.randn(
        1,
        T,
        H + 1,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    raw_beta = beta_storage[..., :H]
    A_log = 0.5 * torch.randn(H, dtype=torch.float32, device=DEVICE)
    dt_bias = 0.1 * torch.randn(H, D, dtype=torch.float32, device=DEVICE)
    cu_seqlens = torch.arange(
        0,
        T + 1,
        query_len,
        dtype=torch.int32,
        device=DEVICE,
    )
    state_indices = torch.arange(
        1,
        T + 1,
        dtype=torch.int32,
        device=DEVICE,
    ).view(num_seqs, query_len)
    num_accepted_tokens = torch.tensor(
        [1, 2, 3],
        dtype=torch.int32,
        device=DEVICE,
    )
    state_storage = 0.01 * torch.randn(
        T + 1,
        H * D * D + 17,
        dtype=torch.float32,
        device=DEVICE,
    )
    state = state_storage[:, : H * D * D].view(T + 1, H, D, D)
    output_storage = torch.full(
        (1, T, H * D + 11),
        torch.nan,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    output = output_storage[..., : H * D].view(1, T, H, D)

    gate = fused_kda_gate(
        raw_g.contiguous().view(T, H * D),
        A_log,
        D,
        g_bias=dt_bias,
        lower_bound=lower_bound,
    ).unsqueeze(0)
    beta = raw_beta.float().sigmoid()
    q_norm = l2norm_fwd(q.contiguous())
    k_norm = l2norm_fwd(k.contiguous())
    expected_state = state.clone()
    expected_outputs = []
    for seq, accepted in enumerate(num_accepted_tokens.tolist()):
        recurrent_state = expected_state[state_indices[seq, accepted - 1]].transpose(
            -1, -2
        )
        start = seq * query_len
        for token in range(query_len):
            token_slice = slice(start + token, start + token + 1)
            token_output, recurrent_state = naive_recurrent_kda(
                q_norm[:, token_slice],
                k_norm[:, token_slice],
                v[:, token_slice],
                gate[:, token_slice],
                beta[:, token_slice],
                initial_state=recurrent_state,
                output_final_state=True,
            )
            assert recurrent_state is not None
            expected_outputs.append(token_output)
            expected_state[state_indices[seq, token]] = recurrent_state.transpose(
                -1, -2
            )
    expected = torch.cat(expected_outputs, dim=1)

    actual_state = state.clone()
    actual, _ = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        initial_state=actual_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        num_accepted_tokens=num_accepted_tokens,
        out=output,
        fuse_gate=fuse_gate,
    )

    assert actual.data_ptr() == output.data_ptr()
    assert_close("o", expected, actual, 1e-3, err_atol=1e-3)
    used_states = state_indices.flatten().long()
    assert_close(
        "ht",
        expected_state[used_states],
        actual_state[used_states],
        3e-3,
        err_atol=3e-3,
    )
    assert torch.isnan(output_storage[..., H * D :]).all()


@pytest.mark.parametrize(
    ("num_heads", "num_seqs", "lower_bound", "fuse_output_norm"),
    [
        (12, 1, -5.0, True),
        (12, 4, None, False),
        (24, 4, None, False),
        (48, 1, -5.0, True),
        (96, 1, -5.0, True),
    ],
)
@torch.inference_mode()
def test_fused_kda_decode_correctness(
    num_heads: int,
    num_seqs: int,
    lower_bound: float | None,
    fuse_output_norm: bool,
):
    D, W = 128, 4
    if not is_fused_kda_decode_supported(
        num_heads,
        D,
        W,
        num_spec=0,
        input_dtype=torch.bfloat16,
        conv_state_dtype=torch.bfloat16,
    ):
        pytest.skip("Fused KDA decode is not supported on this platform")
    torch.manual_seed(967 + num_heads + num_seqs)
    dim = num_heads * D
    slots = num_seqs + 2
    packed_x_storage = torch.randn(
        num_seqs, 3 * dim + 17, dtype=torch.bfloat16, device=DEVICE
    )
    packed_x = packed_x_storage[:, : 3 * dim]
    weight = 0.1 * torch.randn(3 * dim, W, dtype=torch.float32, device=DEVICE)
    conv_seed = 0.1 * torch.randn(
        slots,
        W - 1,
        3 * dim,
        dtype=torch.bfloat16,
        device=DEVICE,
    ).transpose(1, 2)
    raw_g = torch.randn(
        1,
        num_seqs,
        num_heads,
        D,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    raw_beta_storage = torch.randn(
        1,
        num_seqs,
        num_heads + 1,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    raw_beta = raw_beta_storage[:, :, :num_heads]
    output_gate_storage = torch.randn(
        num_seqs,
        dim + 7,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    output_gate = output_gate_storage[:, :dim].view(num_seqs, num_heads, D)
    norm_weight = torch.randn(D, dtype=torch.float32, device=DEVICE)
    norm_eps = 1e-5
    A_log = 0.5 * torch.randn(num_heads, dtype=torch.float32, device=DEVICE)
    dt_bias = 0.1 * torch.randn(dim, dtype=torch.float32, device=DEVICE)
    state_indices = torch.arange(
        num_seqs,
        0,
        -1,
        dtype=torch.int32,
        device=DEVICE,
    )
    state_seed = 0.01 * torch.randn(
        slots,
        num_heads,
        D,
        D,
        dtype=torch.float32,
        device=DEVICE,
    )

    conv_ref = conv_seed.clone()
    state_ref = state_seed.clone()
    mixed_qkv = causal_conv1d_update(
        packed_x,
        conv_ref,
        weight,
        activation="silu",
        conv_state_indices=state_indices,
        validate_data=True,
        out=torch.empty_like(packed_x),
    )
    expected, _ = fused_recurrent_kda_packed_decode(
        mixed_qkv=mixed_qkv,
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        initial_state=state_ref,
        state_indices=state_indices,
    )
    if fuse_output_norm:
        expected_float = expected.float()
        expected = (
            expected_float
            * torch.rsqrt(expected_float.square().mean(dim=-1, keepdim=True) + norm_eps)
            * norm_weight
            * output_gate.float().sigmoid().unsqueeze(0)
        ).to(expected.dtype)

    conv_slot_elements = 3 * dim * (W - 1)
    state_slot_elements = num_heads * D * D
    conv_slot_bytes = conv_slot_elements * torch.bfloat16.itemsize
    page_bytes = conv_slot_bytes + state_slot_elements * torch.float32.itemsize
    cache_storage = torch.empty(slots * page_bytes, dtype=torch.uint8, device=DEVICE)
    conv_actual = torch.as_strided(
        cache_storage.view(torch.bfloat16),
        size=(slots, 3 * dim, W - 1),
        stride=(page_bytes // torch.bfloat16.itemsize, 1, 3 * dim),
    )
    state_actual = torch.as_strided(
        cache_storage.view(torch.float32),
        size=(slots, num_heads, D, D),
        stride=(page_bytes // torch.float32.itemsize, D * D, D, 1),
        storage_offset=conv_slot_bytes // torch.float32.itemsize,
    )
    conv_actual.copy_(conv_seed)
    state_actual.copy_(state_seed)
    fused_weight = weight.reshape(3, dim, W).transpose(1, 2).contiguous()
    actual = ops.fused_kda_decode(
        x=packed_x,
        weight=fused_weight,
        bias=None,
        conv_state=conv_actual,
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        state_indices=state_indices,
        state=state_actual,
        lower_bound=lower_bound,
        output_gate=output_gate if fuse_output_norm else None,
        norm_weight=norm_weight if fuse_output_norm else None,
        norm_eps=norm_eps,
    )

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(conv_actual, conv_ref, atol=0, rtol=0)
    torch.testing.assert_close(state_actual, state_ref, atol=3e-2, rtol=3e-2)


def test_fused_kda_decode_rejects_speculative_conv_state():
    assert not is_fused_kda_decode_supported(
        num_heads=12,
        head_dim=128,
        conv_width=4,
        num_spec=2,
        input_dtype=torch.bfloat16,
        conv_state_dtype=torch.bfloat16,
    )


@torch.inference_mode()
def test_flashkda_correctness():
    if not is_flashkda_supported(128, torch.bfloat16, -3.0):
        pytest.skip("FlashKDA is not supported on this platform")

    import vllm._flashkda_C  # noqa: F401

    B, T, H, D = 1, 48, 2, 128
    torch.manual_seed(11)
    q, k, v, raw_g = [
        torch.randn(B, T, H, D, dtype=torch.bfloat16, device=DEVICE) for _ in range(4)
    ]
    beta_logits = torch.randn(B, T, H, dtype=torch.bfloat16, device=DEVICE)
    A_log = torch.randn(H, dtype=torch.float32, device=DEVICE) * 0.5
    dt_bias = torch.randn(H, D, dtype=torch.float32, device=DEVICE) * 0.1
    initial_state = torch.randn(2, H, D, D, dtype=torch.float32, device=DEVICE)
    cu_seqlens = torch.tensor([0, 17, T], dtype=torch.int32, device=DEVICE)
    lower_bound = -3.0

    gate = lower_bound * torch.sigmoid(
        A_log.exp()[None, None, :, None] * (raw_g.float() + dt_bias[None, None, :, :])
    )
    beta = beta_logits.float().sigmoid()
    q_norm = l2norm_fwd(q.contiguous())
    k_norm = l2norm_fwd(k.contiguous())

    expected_outputs = []
    expected_states = []
    for i, (start, end) in enumerate(
        zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist())
    ):
        output, final_state = naive_recurrent_kda(
            q_norm[:, start:end],
            k_norm[:, start:end],
            v[:, start:end],
            gate[:, start:end],
            beta[:, start:end],
            initial_state=initial_state[i].transpose(-1, -2),
            output_final_state=True,
        )
        expected_outputs.append(output)
        expected_states.append(final_state)
    expected_out = torch.cat(expected_outputs, dim=1)
    expected_state = torch.cat(expected_states).transpose(-1, -2).contiguous()

    actual_out = torch.empty_like(v)
    actual_state = torch.empty_like(initial_state)
    workspace = torch.empty(
        torch.ops._flashkda_C.get_workspace_size(T, H, cu_seqlens.numel() - 1),
        dtype=torch.uint8,
        device=DEVICE,
    )
    torch.ops._flashkda_C.fwd(
        q,
        k,
        v,
        raw_g,
        beta_logits,
        D**-0.5,
        actual_out,
        workspace,
        A_log,
        dt_bias,
        lower_bound,
        initial_state,
        actual_state,
        cu_seqlens,
    )

    assert_close("o", expected_out, actual_out, 0.01)
    assert_close("ht", expected_state, actual_state, 0.01)
