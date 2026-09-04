# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision tests for vllm's chunk_kda Triton operator.

Compares chunk_kda against a naive recurrent reference (float32).
Uses torch.rand for q/k/v to match FLA's test pattern.
"""

from types import SimpleNamespace

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
from vllm.models.kimi_k3.nvidia import kda as nvidia_kda
from vllm.models.kimi_k3.nvidia.kda import (
    _flashkda_prefill,
    _store_cache_checkpoints_kernel,
    is_flashkda_supported,
    is_fused_kda_decode_supported,
)
from vllm.models.kimi_k3.nvidia.model import KimiLinearForCausalLM
from vllm.models.kimi_k3.nvidia.ops import recoverssm as recoverssm_ops
from vllm.models.kimi_k3.nvidia.ops.recoverssm import (
    KDARecoverSSMCommitContext,
    kda_recoverssm_verify,
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
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

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


def test_kda_warmup_skips_missing_metadata(monkeypatch):
    monkeypatch.setattr(
        nvidia_kda,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={}),
    )
    layer = object.__new__(nvidia_kda.KimiK3DeltaAttention)
    object.__setattr__(layer, "prefix", "language_model.model.layers.0.self_attn")
    empty = torch.empty(0, device=DEVICE)

    assert layer._forward(empty, empty, empty, empty, empty) is None


def test_kda_recoverssm_config_state_layout():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(
                linear_attn_config={
                    "num_heads": 4,
                    "head_dim": 32,
                    "short_conv_kernel_size": 4,
                }
            ),
        ),
        cache_config=SimpleNamespace(
            mamba_cache_dtype="auto",
            use_kda_recoverssm=True,
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        speculative_config=SimpleNamespace(num_speculative_tokens=2),
    )

    assert KimiLinearForCausalLM.get_mamba_state_dtype_from_config(vllm_config) == (
        torch.bfloat16,
        torch.float32,
        torch.float32,
        torch.bfloat16,
    )
    assert KimiLinearForCausalLM.get_mamba_state_shape_from_config(vllm_config)[2:] == (
        (4, 3, 32),
        (4, 3, 64),
    )


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
    (
        "conv_state_dim_first",
        "use_request_indices",
        "lower_bound",
        "align_mode",
    ),
    [
        pytest.param(False, False, None, False, id="baseline"),
        pytest.param(True, True, -5.0, True, id="all-features"),
        pytest.param(False, True, -5.0, False, id="request-indexed"),
        pytest.param(True, False, None, True, id="aligned"),
    ],
)
@torch.inference_mode()
def test_kda_recoverssm_verify_and_group_commit(
    monkeypatch: pytest.MonkeyPatch,
    lower_bound: float | None,
    use_request_indices: bool,
    conv_state_dim_first: bool,
    align_mode: bool,
):
    monkeypatch.setattr(
        recoverssm_ops,
        "is_conv_state_dim_first",
        lambda: conv_state_dim_first,
    )
    num_layers, num_seqs, query_len = 2, 2, 8
    num_blocks, num_heads, dim = (7 if align_mode else 3), 4, 128
    total_tokens = num_seqs * query_len
    torch.manual_seed(20260808)

    q, k, v, raw_g = [
        torch.randn(
            1,
            total_tokens,
            num_heads,
            dim,
            dtype=torch.bfloat16,
            device=DEVICE,
        )
        for _ in range(4)
    ]
    raw_beta = torch.randn(
        1,
        total_tokens,
        num_heads,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    query_start_loc = torch.arange(
        0,
        total_tokens + 1,
        query_len,
        dtype=torch.int32,
        device=DEVICE,
    )
    state_indices = torch.tensor(
        [5, 6] if align_mode else [1, 2], dtype=torch.int32, device=DEVICE
    )
    accepted = [2, 8]
    if use_request_indices:
        global_num_accepted = torch.tensor(
            [0, accepted[0], 0, accepted[1]],
            dtype=torch.int32,
            device=DEVICE,
        )
        request_indices = torch.tensor([1, 3], dtype=torch.int32, device=DEVICE)
    else:
        global_num_accepted = torch.tensor(accepted, dtype=torch.int32, device=DEVICE)
        request_indices = None

    block_table = None
    num_computed_tokens = None
    mamba_block_size = None
    if align_mode:
        batch_size = 4 if use_request_indices else num_seqs
        block_table = torch.full((batch_size, 2), -1, dtype=torch.int32, device=DEVICE)
        rows = (
            request_indices
            if request_indices is not None
            else torch.arange(num_seqs, device=DEVICE)
        )
        block_table[rows] = torch.tensor(
            [[1, 5], [2, 6]],
            dtype=torch.int32,
            device=DEVICE,
        )
        num_computed_tokens = torch.zeros(batch_size, dtype=torch.int32, device=DEVICE)
        num_computed_tokens[rows] = 4
        mamba_block_size = 8

    layers = []
    expected_outputs = []
    expected_states = []
    initial_states = []
    initial_conv_states = []
    history_len, conv_dim = 3, 12
    for layer_idx in range(num_layers):
        A_log = (
            0.2 * torch.randn(num_heads, dtype=torch.float32, device=DEVICE)
            + layer_idx * 0.03
        ).contiguous()
        dt_bias = (
            0.1 * torch.randn(num_heads, dim, dtype=torch.float32, device=DEVICE)
        ).contiguous()
        checkpoint = 0.01 * torch.randn(
            num_blocks,
            num_heads,
            dim,
            dim,
            dtype=torch.float32,
            device=DEVICE,
        )
        conv_shape = (
            (num_blocks, conv_dim, history_len + query_len - 1)
            if conv_state_dim_first
            else (num_blocks, history_len + query_len - 1, conv_dim)
        )
        conv_state = torch.randn(conv_shape, dtype=torch.bfloat16, device=DEVICE)
        correction_cache = torch.empty(
            num_blocks,
            num_heads,
            query_len,
            dim,
            dtype=torch.float32,
            device=DEVICE,
        )
        kg_cache = torch.empty(
            num_blocks,
            num_heads,
            query_len,
            2 * dim,
            dtype=torch.bfloat16,
            device=DEVICE,
        )
        layer = SimpleNamespace(
            kv_cache=(
                conv_state,
                checkpoint,
                correction_cache,
                kg_cache,
            ),
            A_log=A_log,
            dt_bias=dt_bias,
            local_num_heads=num_heads,
            head_dim=dim,
            gate_lower_bound=lower_bound,
        )
        layers.append(layer)
        initial_states.append(checkpoint.clone())
        initial_conv_states.append(conv_state.clone())

        actual_output = kda_recoverssm_verify(
            q=q,
            k=k,
            v=v,
            raw_g=raw_g,
            raw_beta=raw_beta,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            checkpoint_state=checkpoint,
            correction_cache=correction_cache,
            kg_cache=kg_cache,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            spec_query_len=query_len,
        )

        normalized_q = q.float() * torch.rsqrt(
            q.float().square().sum(dim=-1, keepdim=True) + 1e-6
        )
        normalized_k = k.float() * torch.rsqrt(
            k.float().square().sum(dim=-1, keepdim=True) + 1e-6
        )
        gate_input = raw_g.float() + dt_bias.view(1, 1, num_heads, dim)
        if lower_bound is None:
            gate = -A_log.exp().view(1, 1, num_heads, 1) * F.softplus(gate_input)
        else:
            gate = lower_bound * torch.sigmoid(
                A_log.exp().view(1, 1, num_heads, 1) * gate_input
            )
        beta = raw_beta.float().sigmoid()

        reference_output = []
        committed_states = checkpoint.clone()
        for seq_idx, commit_len in enumerate(accepted):
            start = seq_idx * query_len
            end = start + query_len
            output, _ = naive_recurrent_kda(
                normalized_q[:, start:end],
                normalized_k[:, start:end],
                v[:, start:end],
                gate[:, start:end],
                beta[:, start:end],
                initial_state=checkpoint[state_indices[seq_idx]].transpose(-1, -2),
            )
            reference_output.append(output)
            _, committed_state = naive_recurrent_kda(
                normalized_q[:, start : start + commit_len],
                normalized_k[:, start : start + commit_len],
                v[:, start : start + commit_len],
                gate[:, start : start + commit_len],
                beta[:, start : start + commit_len],
                initial_state=checkpoint[state_indices[seq_idx]].transpose(-1, -2),
                output_final_state=True,
            )
            assert committed_state is not None
            final_block = state_indices[seq_idx]
            if align_mode:
                assert block_table is not None
                row = request_indices[seq_idx] if use_request_indices else seq_idx
                final_block = block_table[row, (4 + commit_len) // 8]
            committed_states[final_block] = committed_state.transpose(-1, -2)
            if align_mode and 4 + commit_len >= 8:
                _, boundary_state = naive_recurrent_kda(
                    normalized_q[:, start : start + 4],
                    normalized_k[:, start : start + 4],
                    v[:, start : start + 4],
                    gate[:, start : start + 4],
                    beta[:, start : start + 4],
                    initial_state=checkpoint[state_indices[seq_idx]].transpose(-1, -2),
                    output_final_state=True,
                )
                assert boundary_state is not None
                assert block_table is not None
                row = request_indices[seq_idx] if use_request_indices else seq_idx
                committed_states[block_table[row, 0]] = boundary_state.transpose(-1, -2)
        expected_outputs.append(torch.cat(reference_output, dim=1))
        expected_states.append(committed_states)
        torch.testing.assert_close(checkpoint, initial_states[-1])
        torch.testing.assert_close(
            actual_output,
            expected_outputs[-1],
            atol=3e-2,
            rtol=3e-2,
        )

    context = KDARecoverSSMCommitContext.create(
        layers,
        spec_query_len=query_len,
        max_num_reqs=global_num_accepted.shape[0],
    )
    context.commit(
        global_num_accepted,
        state_indices,
        query_start_loc,
        request_indices=request_indices,
        block_table=block_table,
        num_computed_tokens=num_computed_tokens,
        mamba_block_size=mamba_block_size,
    )

    for layer_idx, layer in enumerate(layers):
        torch.testing.assert_close(
            layer.kv_cache[1],
            expected_states[layer_idx],
            atol=3e-3,
            rtol=3e-3,
        )
        for seq_idx, commit_len in enumerate(accepted):
            block = state_indices[seq_idx]
            if align_mode:
                assert block_table is not None
                row = request_indices[seq_idx] if use_request_indices else seq_idx
                block = block_table[row, (4 + commit_len) // 8]
            source_block = state_indices[seq_idx] if align_mode else block
            if conv_state_dim_first:
                actual_conv = layer.kv_cache[0][block, :, :history_len]
                expected_conv = initial_conv_states[layer_idx][
                    source_block,
                    :,
                    commit_len - 1 : commit_len - 1 + history_len,
                ]
            else:
                actual_conv = layer.kv_cache[0][block, :history_len]
                expected_conv = initial_conv_states[layer_idx][
                    source_block,
                    commit_len - 1 : commit_len - 1 + history_len,
                ]
            torch.testing.assert_close(actual_conv, expected_conv)
            if align_mode and 4 + commit_len >= 8:
                assert block_table is not None
                boundary_block = block_table[row, 0]
                if conv_state_dim_first:
                    actual_boundary_conv = layer.kv_cache[0][
                        boundary_block, :, :history_len
                    ]
                    expected_boundary_conv = initial_conv_states[layer_idx][
                        state_indices[seq_idx], :, 3 : 3 + history_len
                    ]
                else:
                    actual_boundary_conv = layer.kv_cache[0][
                        boundary_block, :history_len
                    ]
                    expected_boundary_conv = initial_conv_states[layer_idx][
                        state_indices[seq_idx], 3 : 3 + history_len
                    ]
                torch.testing.assert_close(actual_boundary_conv, expected_boundary_conv)


@pytest.mark.parametrize(
    ("num_heads", "num_seqs", "lower_bound", "fuse_output_norm", "conv_layout"),
    [
        (12, 1, -5.0, True, "SD"),
        (12, 4, None, False, "SD"),
        (24, 4, None, False, "SD"),
        (48, 1, -5.0, True, "SD"),
        (96, 1, -5.0, True, "SD"),
        (12, 1, -5.0, True, "DS"),
        (12, 4, None, False, "DS"),
        (24, 4, None, False, "DS"),
        (48, 1, -5.0, True, "DS"),
        (96, 1, -5.0, True, "DS"),
    ],
)
@torch.inference_mode()
def test_fused_kda_decode_correctness(
    num_heads: int,
    num_seqs: int,
    lower_bound: float | None,
    fuse_output_norm: bool,
    conv_layout: str,
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
    torch.manual_seed(967 + num_heads + num_seqs + (conv_layout == "DS"))
    dim = num_heads * D
    slots = num_seqs + 2
    packed_x_storage = torch.randn(
        num_seqs, 3 * dim + 17, dtype=torch.bfloat16, device=DEVICE
    )
    packed_x = packed_x_storage[:, : 3 * dim]
    weight = 0.1 * torch.randn(3 * dim, W, dtype=torch.float32, device=DEVICE)
    if conv_layout == "DS":
        # DS cache layout: per slot the taps are innermost
        # (stride (W-1, 1)), matching VLLM_SSM_CONV_STATE_LAYOUT=DS.
        conv_seed = 0.1 * torch.randn(
            slots,
            3 * dim,
            W - 1,
            dtype=torch.bfloat16,
            device=DEVICE,
        )
    else:
        # SD cache layout: per slot the channels are innermost.
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
        stride=(
            page_bytes // torch.bfloat16.itemsize,
            (W - 1) if conv_layout == "DS" else 1,
            1 if conv_layout == "DS" else 3 * dim,
        ),
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
def test_flashkda_near_collinear_keys_remain_finite():
    """Guard against unstable inversion of near-collinear key blocks."""
    lower_bound = -5.0
    if not is_flashkda_supported(128, torch.bfloat16, lower_bound):
        pytest.skip("FlashKDA is not supported on this platform")

    import vllm._flashkda_C  # noqa: F401

    T, H, D = 16384, 1, 128
    torch.manual_seed(0)
    key = torch.randn(1, 1, H, D, dtype=torch.bfloat16, device=DEVICE)
    qk = key.expand(1, T, H, D).contiguous()
    value_block = torch.randn(1, 16, H, D, dtype=torch.bfloat16, device=DEVICE)
    value = value_block.repeat(1, T // 16, 1, 1)
    raw_gate = torch.full_like(qk, -12.0)
    raw_beta = torch.full((1, T, H), 8.0, dtype=qk.dtype, device=DEVICE)
    A_log = torch.zeros(H, dtype=torch.float32, device=DEVICE)
    dt_bias = torch.zeros(H, D, dtype=torch.float32, device=DEVICE)
    initial_state = torch.zeros(1, H, D, D, dtype=torch.float32, device=DEVICE)
    final_state = torch.empty_like(initial_state)
    output = torch.empty_like(value)
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=DEVICE)
    workspace = torch.empty(
        torch.ops._flashkda_C.get_workspace_size(T, H, 1),
        dtype=torch.uint8,
        device=DEVICE,
    )

    torch.ops._flashkda_C.fwd(
        qk,
        qk,
        value,
        raw_gate,
        raw_beta,
        D**-0.5,
        output,
        workspace,
        A_log,
        dt_bias,
        lower_bound,
        initial_state,
        final_state,
        cu_seqlens,
    )

    assert torch.isfinite(output).all()
    assert torch.isfinite(final_state).all()


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
    _, expected_checkpoint = naive_recurrent_kda(
        q_norm[:, :16],
        k_norm[:, :16],
        v[:, :16],
        gate[:, :16],
        beta[:, :16],
        initial_state=initial_state[0:1].transpose(-1, -2),
        output_final_state=True,
    )
    assert expected_checkpoint is not None
    expected_checkpoint = expected_checkpoint.transpose(-1, -2).contiguous()

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

    checkpoint_out = torch.empty_like(v)
    checkpoint_final_state = torch.empty_like(initial_state)
    checkpoint_state = torch.empty_like(initial_state)
    checkpoint_offsets = torch.tensor([16, 31], dtype=torch.int32, device=DEVICE)
    _flashkda_prefill(
        q=q,
        k=k,
        v=v,
        g=raw_g,
        beta=beta_logits,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        out=checkpoint_out,
        final_state=checkpoint_final_state,
        workspace=workspace,
        checkpoint_state=checkpoint_state,
        checkpoint_offsets=checkpoint_offsets,
    )

    assert_close("checkpoint_o", expected_out, checkpoint_out, 0.01)
    assert_close("checkpoint_ht", expected_state, checkpoint_final_state, 0.01)
    assert_close("checkpoint", expected_checkpoint, checkpoint_state[:1], 0.01)

    conv_state = torch.zeros(2, H * D, 3, dtype=q.dtype, device=DEVICE)
    recurrent_storage = torch.zeros(2, H * D * D + 8, device=DEVICE)
    recurrent_state = recurrent_storage[:, : H * D * D].view(2, H, D, D)
    conv_input = q[0].flatten(1)
    checkpoint_state_indices = torch.tensor(
        [1, NULL_BLOCK_ID], dtype=torch.int32, device=DEVICE
    )
    state_len = conv_state.shape[-1]
    width = H * D
    recurrent_row_size = checkpoint_state[0].numel()
    block_size = 256
    _store_cache_checkpoints_kernel[
        (
            checkpoint_state_indices.numel(),
            (max(width * state_len, recurrent_row_size) + block_size - 1) // block_size,
        )
    ](
        conv_input,
        conv_state,
        checkpoint_state,
        recurrent_state,
        cu_seqlens,
        checkpoint_offsets,
        checkpoint_state_indices,
        conv_input.stride(0),
        conv_input.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        checkpoint_state.stride(0),
        recurrent_state.stride(0),
        checkpoint_offsets.stride(0),
        state_len,
        width,
        recurrent_row_size,
        NULL_BLOCK_ID,
        block_size,
    )
    torch.testing.assert_close(conv_state[1], q[0, 13:16].flatten(1).transpose(0, 1))
    torch.testing.assert_close(recurrent_state[1], checkpoint_state[0])
