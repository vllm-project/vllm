# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-4 / K2 unit test: dual-anchor (separate read vs write) SSM state indices
for the GDN decode kernel ``fused_sigmoid_gating_delta_rule_update``.

The all-mode + MTP decode path must read the initial recurrent state from one set
of physical blocks (``ssm_state_indices``, the previous-step anchor) while writing
the updated state into a *different* set of blocks (``ssm_state_indices_output``,
the current-step anchor). This mirrors Mamba2's
``state_batch_indices`` (read) vs ``dst_state_batch_indices`` (write) split in
``selective_state_update``.

These tests use tiny dummy tensors and assert the exact read/write isolation
property plus backward compatibility (output index ``None`` -> in-place).
"""

import pytest
import torch

from vllm.model_executor.layers.fla.ops import (
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DEVICE = current_platform.device_type


def _make_decode_inputs(num_reqs, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
                        dtype, tp_size=1):
    """Build single-token-per-request decode inputs matching the existing
    ``test_fused_sigmoid_gating_delta_rule`` shapes."""
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    mixed_qkv_dim = (key_dim * 2 + value_dim) // tp_size
    num_tokens = num_reqs  # seq_len == 1 for decode

    mixed_qkv = torch.rand(num_tokens, mixed_qkv_dim, dtype=dtype)
    query, key, value = torch.split(
        mixed_qkv,
        [key_dim // tp_size, key_dim // tp_size, value_dim // tp_size],
        dim=-1,
    )
    query = query.view(1, num_tokens, num_k_heads, head_k_dim)
    key = key.view(1, num_tokens, num_k_heads, head_k_dim)
    value = value.view(1, num_tokens, num_v_heads, head_v_dim)

    A_log = torch.rand(num_v_heads // tp_size, dtype=dtype)
    dt_bias = torch.rand(num_v_heads // tp_size, dtype=dtype)
    a = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    b = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    cu_seqlens = torch.arange(0, num_tokens + 1, dtype=torch.int32)
    return dict(
        A_log=A_log, a=a, b=b, dt_bias=dt_bias,
        q=query, k=key, v=value, cu_seqlens=cu_seqlens,
        num_tokens=num_tokens, num_v_heads=num_v_heads,
        head_k_dim=head_k_dim, head_v_dim=head_v_dim,
    )


@pytest.mark.parametrize("num_reqs", [1, 2, 4])
@pytest.mark.parametrize("num_k_heads", [16])
@pytest.mark.parametrize("num_v_heads", [32])
@pytest.mark.parametrize("head_k_dim", [128])
@pytest.mark.parametrize("head_v_dim", [128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dual_index_read_write_isolation(
    num_reqs, num_k_heads, num_v_heads, head_k_dim, head_v_dim, dtype
):
    """Reading from slot set A while writing to a *disjoint* slot set B must:
    (1) leave the A slots unchanged, (2) land the computed final state in B, and
    (3) produce the same attention output as the in-place run."""
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    ins = _make_decode_inputs(
        num_reqs, num_k_heads, num_v_heads, head_k_dim, head_v_dim, dtype
    )
    num_tokens = ins["num_tokens"]
    HV, K, V = ins["num_v_heads"], ins["head_k_dim"], ins["head_v_dim"]

    total_entries = num_tokens * 4
    # disjoint read / write slot sets (avoid NULL_BLOCK_ID==0)
    perm = torch.randperm(total_entries - 1, dtype=torch.int32) + 1
    read_idx = perm[:num_tokens].contiguous()
    write_idx = perm[num_tokens:2 * num_tokens].contiguous()
    assert set(read_idx.tolist()).isdisjoint(set(write_idx.tolist()))

    base_state = torch.rand(total_entries, HV, V, K, dtype=dtype)

    def call(state, out_idx):
        return fused_sigmoid_gating_delta_rule_update(
            A_log=ins["A_log"], a=ins["a"], b=ins["b"], dt_bias=ins["dt_bias"],
            q=ins["q"], k=ins["k"], v=ins["v"],
            initial_state=state, inplace_final_state=True,
            ssm_state_indices=read_idx, ssm_state_indices_output=out_idx,
            cu_seqlens=ins["cu_seqlens"], use_qk_l2norm_in_kernel=True,
        )

    # Reference: in-place (output index defaults to read index) -> final lands in A.
    state_ref = base_state.clone()
    read_orig = state_ref[read_idx].clone()
    out_ref, _ = call(state_ref, None)
    final_at_read = state_ref[read_idx].clone()
    # in-place actually mutated the read slots (sanity: state changed)
    assert not torch.allclose(final_at_read, read_orig, atol=1e-3, rtol=1e-3)

    # Dual-anchor: read from A, write to B.
    state_dual = base_state.clone()
    a_before = state_dual[read_idx].clone()
    out_dual, _ = call(state_dual, write_idx)

    # (1) read slots untouched
    torch.testing.assert_close(state_dual[read_idx], a_before, atol=0, rtol=0)
    # (2) computed final state landed in the write slots (== the in-place result)
    torch.testing.assert_close(
        state_dual[write_idx], final_at_read, atol=1e-2, rtol=1e-2
    )
    # (3) attention output identical regardless of where the state is written
    torch.testing.assert_close(out_dual, out_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dual_index_none_is_inplace(dtype):
    """``ssm_state_indices_output=None`` must reproduce exactly the legacy
    in-place behavior (write index == read index)."""
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    ins = _make_decode_inputs(2, 16, 32, 128, 128, dtype)
    num_tokens = ins["num_tokens"]
    HV, K, V = ins["num_v_heads"], ins["head_k_dim"], ins["head_v_dim"]

    total_entries = num_tokens * 4
    perm = torch.randperm(total_entries - 1, dtype=torch.int32) + 1
    read_idx = perm[:num_tokens].contiguous()
    base_state = torch.rand(total_entries, HV, V, K, dtype=dtype)

    def call(state, out_idx):
        return fused_sigmoid_gating_delta_rule_update(
            A_log=ins["A_log"], a=ins["a"], b=ins["b"], dt_bias=ins["dt_bias"],
            q=ins["q"], k=ins["k"], v=ins["v"],
            initial_state=state, inplace_final_state=True,
            ssm_state_indices=read_idx, ssm_state_indices_output=out_idx,
            cu_seqlens=ins["cu_seqlens"], use_qk_l2norm_in_kernel=True,
        )

    # None (default) vs explicitly passing read_idx as the output index: identical.
    state_none = base_state.clone()
    out_none, _ = call(state_none, None)

    state_explicit = base_state.clone()
    out_explicit, _ = call(state_explicit, read_idx)

    torch.testing.assert_close(out_none, out_explicit, atol=0, rtol=0)
    torch.testing.assert_close(
        state_none[read_idx], state_explicit[read_idx], atol=0, rtol=0
    )


@pytest.mark.parametrize("num_speculative_tokens", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dual_index_spec_2d(num_speculative_tokens, dtype):
    """2-D index case (the MTP spec-decode shape): read anchor at the accepted
    position from one block row, write the per-token states into a different
    block row. Validates the read row stays intact and the write row receives
    the states for every speculative position."""
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    tp_size = 1
    num_k_heads, num_v_heads, head_k_dim, head_v_dim = 16, 32, 128, 128
    num_reqs = 2
    seq = num_speculative_tokens + 1
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    mixed_qkv_dim = (key_dim * 2 + value_dim) // tp_size
    num_tokens = num_reqs * seq

    mixed_qkv = torch.rand(num_tokens, mixed_qkv_dim, dtype=dtype)
    query, key, value = torch.split(
        mixed_qkv,
        [key_dim // tp_size, key_dim // tp_size, value_dim // tp_size],
        dim=-1,
    )
    query = query.view(1, num_tokens, num_k_heads, head_k_dim)
    key = key.view(1, num_tokens, num_k_heads, head_k_dim)
    value = value.view(1, num_tokens, num_v_heads, head_v_dim)
    A_log = torch.rand(num_v_heads // tp_size, dtype=dtype)
    dt_bias = torch.rand(num_v_heads // tp_size, dtype=dtype)
    a = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    b = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    num_accepted_tokens = torch.randint(
        1, seq + 1, (num_reqs,), dtype=torch.int32
    )
    cu_seqlens = torch.arange(0, num_tokens + 1, seq, dtype=torch.int32)

    HV, K, V = num_v_heads, head_k_dim, head_v_dim
    total_entries = num_tokens * 4
    perm = torch.randperm(total_entries - 1, dtype=torch.int32) + 1
    read_2d = perm[:num_reqs * seq].view(num_reqs, seq).contiguous()
    write_2d = perm[num_reqs * seq:2 * num_reqs * seq].view(num_reqs, seq).contiguous()
    base_state = torch.rand(total_entries, HV, V, K, dtype=dtype)

    def call(state, in_idx, out_idx):
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log, a=a, b=b, dt_bias=dt_bias,
            q=query, k=key, v=value,
            initial_state=state, inplace_final_state=True,
            ssm_state_indices=in_idx, ssm_state_indices_output=out_idx,
            num_accepted_tokens=num_accepted_tokens,
            cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=True,
        )

    # in-place reference: read AND write through read_2d.
    state_ref = base_state.clone()
    out_ref, _ = call(state_ref, read_2d, None)
    # in-place wrote the per-token states into the read_2d rows
    final_at_read = state_ref[read_2d.flatten()].clone()

    # dual-anchor: read read_2d, write write_2d
    state_dual = base_state.clone()
    read_before = state_dual[read_2d.flatten()].clone()
    out_dual, _ = call(state_dual, read_2d, write_2d)

    # read rows untouched
    torch.testing.assert_close(
        state_dual[read_2d.flatten()], read_before, atol=0, rtol=0
    )
    # write rows now hold the same per-token states the in-place run wrote to read rows
    torch.testing.assert_close(
        state_dual[write_2d.flatten()], final_at_read, atol=1e-2, rtol=1e-2
    )
    # output identical
    torch.testing.assert_close(out_dual, out_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("num_speculative_tokens", [3])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_dual_index_noncontiguous_2d(num_speculative_tokens, dtype):
    """P1 Fix 3: a NON-contiguous 2-D index tensor (transposed/strided view)
    must produce identical results to its .contiguous() copy. The kernel
    indexes the token dim with an implicit stride of 1, so before the wrapper
    normalization a strided view silently read wrong offsets."""
    torch.set_default_device(DEVICE)
    set_random_seed(0)
    tp_size = 1
    num_k_heads, num_v_heads, head_k_dim, head_v_dim = 16, 32, 128, 128
    num_reqs = 2
    seq = num_speculative_tokens + 1
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    mixed_qkv_dim = (key_dim * 2 + value_dim) // tp_size
    num_tokens = num_reqs * seq

    mixed_qkv = torch.rand(num_tokens, mixed_qkv_dim, dtype=dtype)
    query, key, value = torch.split(
        mixed_qkv,
        [key_dim // tp_size, key_dim // tp_size, value_dim // tp_size],
        dim=-1,
    )
    query = query.view(1, num_tokens, num_k_heads, head_k_dim)
    key = key.view(1, num_tokens, num_k_heads, head_k_dim)
    value = value.view(1, num_tokens, num_v_heads, head_v_dim)
    A_log = torch.rand(num_v_heads // tp_size, dtype=dtype)
    dt_bias = torch.rand(num_v_heads // tp_size, dtype=dtype)
    a = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    b = torch.rand(num_tokens, num_v_heads, dtype=dtype)
    num_accepted_tokens = torch.randint(
        1, seq + 1, (num_reqs,), dtype=torch.int32
    )
    cu_seqlens = torch.arange(0, num_tokens + 1, seq, dtype=torch.int32)

    HV, K, V = num_v_heads, head_k_dim, head_v_dim
    total_entries = num_tokens * 4
    perm = torch.randperm(total_entries - 1, dtype=torch.int32) + 1
    read_contig = perm[:num_reqs * seq].view(num_reqs, seq).contiguous()
    write_contig = (
        perm[num_reqs * seq:2 * num_reqs * seq].view(num_reqs, seq).contiguous()
    )
    # Same logical values, NON-contiguous layout: strides (1, num_reqs).
    read_strided = read_contig.t().contiguous().t()
    write_strided = write_contig.t().contiguous().t()
    assert read_strided.stride(-1) != 1  # genuinely non-contiguous view
    torch.testing.assert_close(read_strided, read_contig, atol=0, rtol=0)
    base_state = torch.rand(total_entries, HV, V, K, dtype=dtype)

    def call(state, in_idx, out_idx):
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log, a=a, b=b, dt_bias=dt_bias,
            q=query, k=key, v=value,
            initial_state=state, inplace_final_state=True,
            ssm_state_indices=in_idx, ssm_state_indices_output=out_idx,
            num_accepted_tokens=num_accepted_tokens,
            cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=True,
        )

    # REGRESSION: contiguous path (reference).
    state_ref = base_state.clone()
    out_ref, _ = call(state_ref, read_contig, write_contig)

    # NEW: strided views must give bit-identical behavior to the contiguous run.
    state_strided = base_state.clone()
    out_strided, _ = call(state_strided, read_strided, write_strided)

    torch.testing.assert_close(out_strided, out_ref, atol=0, rtol=0)
    torch.testing.assert_close(state_strided, state_ref, atol=0, rtol=0)
