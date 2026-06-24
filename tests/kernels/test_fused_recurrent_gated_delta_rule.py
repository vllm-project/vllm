# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accuracy tests for the fused_recurrent_gated_delta_rule GDN/FLA Triton kernel."""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import fused_recurrent_gated_delta_rule
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="fused_recurrent_gated_delta_rule dispatches a Triton kernel that "
    "requires a CUDA-alike or XPU device.",
)

DEVICE = current_platform.device_type

# HV > H is grouped value attention; V > 32 splits the grid over V.
SHAPES = [(4, 4, 128, 128), (2, 8, 128, 128), (2, 8, 64, 32)]
SERVING = (torch.bfloat16, True, False)
SERVING_FP32 = (torch.float32, True, False)
NO_L2NORM = (torch.float32, False, False)
HEADWISE_BETA = (torch.float32, True, True)
MODES = [SERVING, SERVING_FP32, NO_L2NORM, HEADWISE_BETA]
TOL = {torch.bfloat16: 1e-2, torch.float32: 1e-4}


@pytest.fixture(autouse=True)
def setup_device():
    with torch.device(DEVICE):
        set_random_seed(0)
        yield


def ref_fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    inplace_final_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """float32 recurrence over the shared state pool, sequential only over time."""
    q, k, v, g, beta = (x.float()[0] for x in (q, k, v, g, beta))
    if beta.ndim == 2:  # scalar beta must broadcast over V
        beta = beta[..., None]
    if use_qk_l2norm_in_kernel:
        q, k = (x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + 1e-6) for x in (q, k))
    rep = v.shape[1] // q.shape[1]
    q = (q * k.shape[-1] ** -0.5).repeat_interleave(rep, dim=1)
    k = k.repeat_interleave(rep, dim=1)

    o = torch.zeros_like(v)
    pool = initial_state.float().clone()
    final_state = (
        pool.clone()
        if inplace_final_state
        else torch.zeros(v.shape[0], *pool.shape[1:])
    )
    indices = ssm_state_indices.view(cu_seqlens.numel() - 1, -1).long()

    for n in range(cu_seqlens.numel() - 1):
        bos, eos = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
        first = 0 if num_accepted_tokens is None else int(num_accepted_tokens[n]) - 1
        if indices[n, first] <= 0:
            continue
        h = pool[indices[n, first]]
        for t in range(bos, eos):
            k_t = k[t][:, None]
            h = h * g[t][:, None, None].exp()
            h = h + ((v[t] - (h * k_t).sum(-1)) * beta[t])[..., None] * k_t
            o[t] = (h * q[t][:, None]).sum(-1)
            final_state[indices[n, t - bos] if inplace_final_state else t] = h

    return o.unsqueeze(0), final_state


def make_gdn_inputs(
    num_tokens: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    dtype: torch.dtype,
    beta_headwise: bool = False,
) -> tuple[torch.Tensor, ...]:
    # Op docstring recipe: unit-norm k, g < 0 keep the recurrence stable.
    q = torch.randn(1, num_tokens, H, K, dtype=dtype)
    k = F.normalize(torch.randn(1, num_tokens, H, K, dtype=dtype), dim=-1)
    v = torch.randn(1, num_tokens, HV, V, dtype=dtype)
    g = F.logsigmoid(torch.rand(1, num_tokens, HV))
    beta_shape = (1, num_tokens, HV, V) if beta_headwise else (1, num_tokens, HV)
    beta = torch.rand(beta_shape, dtype=dtype).sigmoid()
    return q, k, v, g, beta


def make_batch(
    num_seqs: int, seq_len: int, num_pad: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform requests plus GDN's trailing zero-token NULL_BLOCK_ID padding."""
    num_tokens = num_seqs * seq_len
    cu_seqlens = torch.arange(0, num_tokens + 1, seq_len, dtype=torch.int32)
    indices = (torch.randperm(num_tokens, dtype=torch.int32) + 1).view(
        num_seqs, seq_len
    )
    pad = torch.full((num_pad, seq_len), NULL_BLOCK_ID, dtype=torch.int32)
    return torch.cat([cu_seqlens, cu_seqlens[-1].repeat(num_pad)]), torch.cat(
        [indices, pad]
    )


def run_pair(
    inputs: tuple[torch.Tensor, ...],
    state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    indices: torch.Tensor,
    **kwargs,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Reference first: it mirrors the op's kwargs, and the op overwrites `state`."""
    ref = ref_fused_recurrent_gated_delta_rule(
        *inputs, state, cu_seqlens, indices, **kwargs
    )
    out = fused_recurrent_gated_delta_rule(
        *inputs,
        initial_state=state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=indices,
        **kwargs,
    )
    return out, ref


@pytest.mark.parametrize("num_decodes,num_pad", [(1, 0), (3, 2)])
@pytest.mark.parametrize("H,HV,K,V", SHAPES)
@pytest.mark.parametrize("dtype,use_qk_l2norm,beta_headwise", MODES)
def test_fused_recurrent_gated_delta_rule_decode(
    num_decodes: int,
    num_pad: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    dtype: torch.dtype,
    use_qk_l2norm: bool,
    beta_headwise: bool,
) -> None:
    inputs = make_gdn_inputs(num_decodes, H, HV, K, V, dtype, beta_headwise)
    state = torch.randn(num_decodes + 1, HV, V, K)
    cu_seqlens, indices = make_batch(num_decodes, 1, num_pad)

    (o, _), (o_ref, state_ref) = run_pair(
        inputs,
        state,
        cu_seqlens,
        indices[:, 0],
        inplace_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
    )

    tol = TOL[dtype]
    torch.testing.assert_close(o.float(), o_ref, atol=tol, rtol=tol)
    torch.testing.assert_close(state, state_ref, atol=tol, rtol=tol)


@pytest.mark.parametrize(
    "num_spec_decodes,num_pad,num_spec_tokens", [(1, 0, 1), (2, 2, 3)]
)
@pytest.mark.parametrize("H,HV,K,V", SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_recurrent_gated_delta_rule_spec_decode(
    num_spec_decodes: int,
    num_pad: int,
    num_spec_tokens: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    dtype: torch.dtype,
) -> None:
    seq_len = num_spec_tokens + 1
    num_tokens = num_spec_decodes * seq_len
    inputs = make_gdn_inputs(num_tokens, H, HV, K, V, dtype)
    state = torch.randn(num_tokens + 1, HV, V, K)
    cu_seqlens, indices = make_batch(num_spec_decodes, seq_len, num_pad)
    num_accepted_tokens = torch.randint(
        1, seq_len + 1, (num_spec_decodes + num_pad,), dtype=torch.int32
    )

    (o, _), (o_ref, state_ref) = run_pair(
        inputs,
        state,
        cu_seqlens,
        indices,
        inplace_final_state=True,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=True,
    )

    tol = TOL[dtype]
    torch.testing.assert_close(o.float(), o_ref, atol=tol, rtol=tol)
    torch.testing.assert_close(state, state_ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("H,HV,K,V", SHAPES)
def test_fused_recurrent_gated_delta_rule_non_inplace_final_state(
    H: int, HV: int, K: int, V: int
) -> None:
    num_seqs, seq_len = 2, 4
    num_tokens = num_seqs * seq_len
    inputs = make_gdn_inputs(num_tokens, H, HV, K, V, torch.float32)
    state = torch.randn(num_tokens + 1, HV, V, K)
    state_before = state.clone()
    cu_seqlens, indices = make_batch(num_seqs, seq_len, 0)

    (o, per_token), (o_ref, per_token_ref) = run_pair(
        inputs,
        state,
        cu_seqlens,
        indices,
        inplace_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )

    tol = TOL[torch.float32]
    torch.testing.assert_close(o.float(), o_ref, atol=tol, rtol=tol)
    torch.testing.assert_close(per_token.float(), per_token_ref, atol=tol, rtol=tol)
    torch.testing.assert_close(state, state_before, atol=0, rtol=0)


@pytest.mark.parametrize("invalid_id", [NULL_BLOCK_ID, PAD_SLOT_ID])
def test_fused_recurrent_gated_delta_rule_skips_invalid_state_index(
    invalid_id: int,
) -> None:
    num_decodes, H, HV, K, V = 4, 4, 8, 128, 128
    inputs = make_gdn_inputs(num_decodes, H, HV, K, V, torch.bfloat16)
    state = torch.randn(num_decodes + 1, HV, V, K)
    cu_seqlens, indices = make_batch(num_decodes, 1, 0)
    indices = indices[:, 0]
    indices[1] = invalid_id

    (o, _), (o_ref, state_ref) = run_pair(
        inputs, state, cu_seqlens, indices, inplace_final_state=True
    )

    # skipped requests never write o
    valid = indices > 0
    tol = TOL[torch.bfloat16]
    torch.testing.assert_close(o[:, valid].float(), o_ref[:, valid], atol=tol, rtol=tol)
    torch.testing.assert_close(state, state_ref, atol=tol, rtol=tol)
