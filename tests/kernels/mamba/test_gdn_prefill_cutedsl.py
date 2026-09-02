# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(100)
):
    pytest.skip(
        reason="GDN CuteDSL prefill requires CUDA SM10x.",
        allow_module_level=True,
    )

from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (  # noqa: E402
    chunk_gated_delta_rule_cutedsl,
    prepare_metadata_cutedsl,
)
from vllm.third_party.flash_linear_attention.ops import (  # noqa: E402
    chunk_gated_delta_rule,
)
from vllm.third_party.flash_linear_attention.ops.index import (  # noqa: E402
    prepare_chunk_indices,
    prepare_chunk_offsets,
)


@pytest.mark.parametrize("num_seqs", [1, 5, 257])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_gdn_chunk_cutedsl_correctness(num_seqs: int, state_dtype: torch.dtype):
    rng_cpu = torch.Generator("cpu").manual_seed(1234)
    rng = torch.Generator("cuda").manual_seed(2345)

    seq_lens = torch.randint(1, 130, (num_seqs,), dtype=torch.int32, generator=rng_cpu)
    cu_seqlens = torch.zeros(num_seqs + 1, device="cuda", dtype=torch.int32)
    cu_seqlens[1:] = seq_lens.to(device="cuda").cumsum(0)
    total_tokens = int(cu_seqlens[-1].item())

    num_k_heads = 4
    num_v_heads = 8
    head_k_dim = 128
    head_v_dim = 128
    dtype = torch.bfloat16

    q = torch.randn(
        1,
        total_tokens,
        num_k_heads,
        head_k_dim,
        device="cuda",
        dtype=dtype,
        generator=rng,
    )
    k = torch.randn_like(q, generator=rng)
    v = torch.randn(
        1,
        total_tokens,
        num_v_heads,
        head_v_dim,
        device="cuda",
        dtype=dtype,
        generator=rng,
    )
    q = F.normalize(q.float(), p=2, dim=-1).to(dtype)
    k = F.normalize(k.float(), p=2, dim=-1).to(dtype)
    a = torch.randn(
        1, total_tokens, num_v_heads, device="cuda", dtype=dtype, generator=rng
    )
    b = torch.randn(
        1, total_tokens, num_v_heads, device="cuda", dtype=dtype, generator=rng
    )
    # Match upstream FLA GatedDeltaNet synthetic initialization:
    # https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py
    A = torch.empty(num_v_heads, device="cuda", dtype=torch.float32).uniform_(
        0, 16, generator=rng
    )
    A_log = torch.log(A)
    dt = torch.exp(
        torch.rand(num_v_heads, device="cuda", dtype=torch.float32, generator=rng)
        * (math.log(0.1) - math.log(0.001))
        + math.log(0.001)
    )
    dt = torch.clamp(dt, min=1e-4)
    dt_bias = dt + torch.log(-torch.expm1(-dt))
    g = -A_log.exp().view(1, 1, num_v_heads) * F.softplus(
        a.float() + dt_bias.view(1, 1, num_v_heads)
    )
    beta = torch.sigmoid(b.float())
    initial_state = (
        torch.randn(
            num_seqs,
            num_v_heads,
            head_v_dim,
            head_k_dim,
            device="cuda",
            dtype=state_dtype,
            generator=rng,
        )
        * 0.05
    )

    # check metadata kernel
    chunk_indices, chunk_offsets = prepare_metadata_cutedsl(cu_seqlens, total_tokens)
    torch.accelerator.synchronize()

    expected_indices = prepare_chunk_indices(cu_seqlens, 64)
    expected_offsets = prepare_chunk_offsets(cu_seqlens, 64)
    total_chunks = int(expected_offsets[-1].item())

    torch.testing.assert_close(chunk_offsets, expected_offsets.to(torch.int32))
    torch.testing.assert_close(
        chunk_indices[:total_chunks],
        expected_indices,
    )

    ref_o, ref_state = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )
    actual_core_attn_out = torch.empty(
        total_tokens,
        num_v_heads,
        head_v_dim,
        device="cuda",
        dtype=dtype,
    )
    actual_o, actual_state = chunk_gated_delta_rule_cutedsl(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        core_attn_out=actual_core_attn_out,
    )
    torch.accelerator.synchronize()

    # check main kernel
    o_error = (actual_o.float() - ref_o.float()).abs()
    state_error = (
        actual_state.float() - ref_state.to(actual_state.dtype).float()
    ).abs()
    assert o_error.max().item() < 2e-3
    assert o_error.mean().item() < 6e-5
    assert state_error.max().item() < 2e-2
    assert state_error.mean().item() < 6e-4
    core_attn_out_error = (
        actual_core_attn_out.float() - actual_o.squeeze(0).float()
    ).abs()
    assert core_attn_out_error.max().item() == 0

    # check main kernel when core_attn_out is not passed
    no_buffer_o, no_buffer_state = chunk_gated_delta_rule_cutedsl(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    torch.accelerator.synchronize()

    no_buffer_o_error = (no_buffer_o.float() - ref_o.float()).abs()
    no_buffer_state_error = (
        no_buffer_state.float() - ref_state.to(no_buffer_state.dtype).float()
    ).abs()
    buffer_o_error = (no_buffer_o.float() - actual_o.float()).abs()
    buffer_state_error = (
        no_buffer_state.float() - actual_state.to(no_buffer_state.dtype).float()
    ).abs()
    assert no_buffer_o_error.max().item() < 2e-3
    assert no_buffer_o_error.mean().item() < 6e-5
    assert no_buffer_state_error.max().item() < 2e-2
    assert no_buffer_state_error.mean().item() < 6e-4
    assert buffer_o_error.max().item() == 0
    assert buffer_state_error.max().item() == 0
