# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-1 packed-qkv tests for ``fused_sigmoid_gating_delta_rule_update``.

The GDN spec-decode SSM kernel accepts the packed conv output (``mixed_qkv``)
directly, addressing q/k/v with per-tensor offsets in-kernel (same arithmetic
as ``fused_recurrent_gated_delta_rule_packed_decode``) instead of requiring
host-side ``rearrange_mixed_qkv`` (cat + 3 contiguous copies, ~375 us/step at
c32). Test groups (kernel-opt plan, Step 1 test plan):

- A: kernel-level packed-vs-tensor parity — decode (T=1), the packed x table
  4-combo matrix, and the T>1 pointer-advance case (non-standard row stride).
- B: golden parity — the tensor path is code-identical to the pre-change
  kernel when ``mixed_qkv=None`` (the PACKED branches are constexpr-pruned),
  so its result is the golden reference; packed must match bit-exactly.
- C: wrapper assertion tests (mutual exclusivity, last-dim contiguity,
  required head geometry).
- D: layer-level A/B parity through ``VLLM_GDN_PACKED_SPEC_QKV`` with a
  monkeypatched call counter on ``rearrange_mixed_qkv`` (reusing the
  test_gdn_all_mode_spec_decode / test_gdn_all_mode_prefill harnesses).

Groups E/F of the plan (GPU battery + nsys A/B) run outside pytest.
"""

from __future__ import annotations

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(
        reason="GDN packed-qkv tests require CUDA (Triton/FLA kernels).",
        allow_module_level=True,
    )

from tests.kernels.mamba import (  # noqa: E402
    test_gdn_all_mode_decode as decode_harness,
)
from tests.kernels.mamba import (  # noqa: E402
    test_gdn_all_mode_prefill as prefill_harness,
)
from tests.kernels.mamba import (  # noqa: E402
    test_gdn_all_mode_spec_decode as spec_harness,
)
from tests.v1.attention.utils import BatchSpec  # noqa: E402
from vllm.third_party.flash_linear_attention.ops import (  # noqa: E402
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (  # noqa: E402
    QwenGatedDeltaNetAttention,
)
from vllm.utils.torch_utils import set_random_seed  # noqa: E402

H = 4  # num qk heads
HV = 8  # num value heads
K = 128  # head_qk_dim
V = 128  # head_v_dim
PACKED_DIM = 2 * H * K + HV * V

GEOMETRY = dict(num_qk_heads=H, head_qk_dim=K, num_v_heads=HV, head_v_dim=V)


def _make_inputs(num_reqs, seq, dtype, device, row_pad=0, seed=0):
    """Packed buffer + the equivalent split q/k/v views (tensor-path input).

    ``row_pad`` widens the underlying rows so ``mixed_qkv.stride(0)`` differs
    from the packed width — the per-token advance must follow the stride.
    """
    set_random_seed(seed)
    num_tokens = num_reqs * seq
    buf = torch.rand(num_tokens, PACKED_DIM + row_pad, dtype=dtype, device=device)
    mixed_qkv = buf[:, :PACKED_DIM]
    assert mixed_qkv.stride(-1) == 1
    query, key, value = torch.split(mixed_qkv, [H * K, H * K, HV * V], dim=-1)
    query = query.reshape(1, num_tokens, H, K)
    key = key.reshape(1, num_tokens, H, K)
    value = value.reshape(1, num_tokens, HV, V)
    A_log = torch.rand(HV, dtype=dtype, device=device)
    dt_bias = torch.rand(HV, dtype=dtype, device=device)
    a = torch.rand(num_tokens, HV, dtype=dtype, device=device)
    b = torch.rand(num_tokens, HV, dtype=dtype, device=device)
    cu_seqlens = torch.arange(0, num_tokens + 1, seq, dtype=torch.int32, device=device)
    return dict(
        mixed_qkv=mixed_qkv,
        q=query,
        k=key,
        v=value,
        A_log=A_log,
        dt_bias=dt_bias,
        a=a,
        b=b,
        cu_seqlens=cu_seqlens,
        num_tokens=num_tokens,
    )


def _make_state(num_slots, dtype, device):
    """State pool with slot 0 reserved (NULL_BLOCK_ID)."""
    return torch.rand(num_slots, HV, V, K, dtype=dtype, device=device)


def _call(ins, state, packed, **index_kwargs):
    qkv = (
        dict(mixed_qkv=ins["mixed_qkv"], **GEOMETRY)
        if packed
        else dict(q=ins["q"], k=ins["k"], v=ins["v"])
    )
    return fused_sigmoid_gating_delta_rule_update(
        A_log=ins["A_log"],
        a=ins["a"],
        b=ins["b"],
        dt_bias=ins["dt_bias"],
        initial_state=state,
        inplace_final_state=True,
        cu_seqlens=ins["cu_seqlens"],
        use_qk_l2norm_in_kernel=True,
        **qkv,
        **index_kwargs,
    )


# --------------------------------------------------------------------------
# Group A: kernel-level packed-vs-tensor parity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_packed_matches_tensor_decode(dtype):
    """A1: single-token decode (T=1, 1D indices) — the packed path must
    reproduce the tensor path bit-exactly (same loads, same arithmetic)."""
    device = torch.device("cuda")
    num_reqs = 4
    ins = _make_inputs(num_reqs, 1, dtype, device)
    perm = torch.randperm(4 * num_reqs - 1, dtype=torch.int32, device=device) + 1
    idx = perm[:num_reqs].contiguous()
    base_state = _make_state(4 * num_reqs, torch.float32, device)

    state_ref = base_state.clone()
    out_ref, _ = _call(ins, state_ref, packed=False, ssm_state_indices=idx)
    state_packed = base_state.clone()
    out_packed, _ = _call(ins, state_packed, packed=True, ssm_state_indices=idx)

    torch.testing.assert_close(out_packed, out_ref, atol=0, rtol=0)
    torch.testing.assert_close(state_packed, state_ref, atol=0, rtol=0)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("use_table", [False, True])
def test_packed_table_matrix(packed, use_table):
    """A2-A5: {packed, tensor} x {2D indices, block_table + anchors} — all
    four combos of the spec-decode configuration must agree bit-exactly with
    the tensor + indices reference (outputs and the full state pool)."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_reqs, seq = 2, 3
    ins = _make_inputs(num_reqs, seq, dtype, device)
    accepted = torch.tensor([2, 1], dtype=torch.int32, device=device)

    # Per-request table row: read window [0, seq), write window [seq, 2*seq).
    width = 2 * seq
    perm = (
        torch.randperm(4 * num_reqs * width - 1, dtype=torch.int32, device=device) + 1
    )
    table = perm[: num_reqs * width].view(num_reqs, width).contiguous()
    read_anchor = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    write_anchor = torch.full((num_reqs,), seq, dtype=torch.int32, device=device)
    base_state = _make_state(int(table.max().item()) + 1, torch.float32, device)

    def index_kwargs(with_table):
        if with_table:
            return dict(
                block_table=table,
                read_anchor=read_anchor,
                write_anchor=write_anchor,
                num_accepted_tokens=accepted,
            )
        return dict(
            ssm_state_indices=table[:, :seq],
            ssm_state_indices_output=table[:, seq:],
            num_accepted_tokens=accepted,
        )

    state_ref = base_state.clone()
    out_ref, _ = _call(ins, state_ref, packed=False, **index_kwargs(False))

    state_run = base_state.clone()
    out_run, _ = _call(ins, state_run, packed=packed, **index_kwargs(use_table))

    torch.testing.assert_close(out_run, out_ref, atol=0, rtol=0)
    torch.testing.assert_close(state_run, state_ref, atol=0, rtol=0)


def test_packed_pointer_advance_multi_token():
    """A6: T>1 with a non-standard row stride — the packed per-token advance
    must follow ``mixed_qkv.stride(0)`` (a padded buffer breaks any H*K-based
    advance) and the per-sequence base must be ``bos * stride``."""
    device = torch.device("cuda")
    num_reqs, seq = 3, 4
    ins = _make_inputs(num_reqs, seq, torch.bfloat16, device, row_pad=32)
    assert ins["mixed_qkv"].stride(0) == PACKED_DIM + 32
    accepted = torch.tensor([3, 1, 4], dtype=torch.int32, device=device)
    perm = (
        torch.randperm(4 * num_reqs * seq - 1, dtype=torch.int32, device=device) + 1
    )
    idx = perm[: num_reqs * seq].view(num_reqs, seq).contiguous()
    base_state = _make_state(4 * num_reqs * seq, torch.float32, device)

    state_ref = base_state.clone()
    out_ref, _ = _call(
        ins,
        state_ref,
        packed=False,
        ssm_state_indices=idx,
        num_accepted_tokens=accepted,
    )
    state_packed = base_state.clone()
    out_packed, _ = _call(
        ins,
        state_packed,
        packed=True,
        ssm_state_indices=idx,
        num_accepted_tokens=accepted,
    )

    torch.testing.assert_close(out_packed, out_ref, atol=0, rtol=0)
    torch.testing.assert_close(state_packed, state_ref, atol=0, rtol=0)


# --------------------------------------------------------------------------
# Group B: golden parity against the (unchanged) tensor path
# --------------------------------------------------------------------------


def test_packed_matches_tensor_golden():
    """B7: golden tensors from the tensor path of the patched kernel — that
    path is code-identical to the pre-change kernel when ``mixed_qkv=None``
    (PACKED branches constexpr-pruned), so bit-exact equality here proves the
    packed path reproduces the pre-change numerics (torch.equal, not
    assert_close)."""
    device = torch.device("cuda")
    num_reqs, seq = 2, 3
    ins = _make_inputs(num_reqs, seq, torch.bfloat16, device, seed=1234)
    accepted = torch.tensor([2, 3], dtype=torch.int32, device=device)
    width = 2 * seq
    perm = (
        torch.randperm(4 * num_reqs * width - 1, dtype=torch.int32, device=device) + 1
    )
    table = perm[: num_reqs * width].view(num_reqs, width).contiguous()
    read_anchor = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    write_anchor = torch.full((num_reqs,), seq, dtype=torch.int32, device=device)
    base_state = _make_state(int(table.max().item()) + 1, torch.float32, device)

    kwargs = dict(
        block_table=table,
        read_anchor=read_anchor,
        write_anchor=write_anchor,
        num_accepted_tokens=accepted,
    )
    golden_state = base_state.clone()
    golden_out, _ = _call(ins, golden_state, packed=False, **kwargs)
    state = base_state.clone()
    out, _ = _call(ins, state, packed=True, **kwargs)

    assert torch.equal(out, golden_out)
    assert torch.equal(state, golden_state)


# --------------------------------------------------------------------------
# Group C: wrapper assertions
# --------------------------------------------------------------------------


def _assert_case_inputs():
    device = torch.device("cuda")
    ins = _make_inputs(2, 1, torch.bfloat16, device)
    idx = torch.tensor([1, 2], dtype=torch.int32, device=device)
    state = _make_state(4, torch.float32, device)
    return ins, idx, state


def test_packed_rejects_qkv_tensors():
    """C8: ``mixed_qkv`` and ``q``/``k``/``v`` are mutually exclusive."""
    ins, idx, state = _assert_case_inputs()
    with pytest.raises(AssertionError, match="mutually exclusive"):
        fused_sigmoid_gating_delta_rule_update(
            A_log=ins["A_log"],
            a=ins["a"],
            b=ins["b"],
            dt_bias=ins["dt_bias"],
            q=ins["q"],
            k=ins["k"],
            v=ins["v"],
            mixed_qkv=ins["mixed_qkv"],
            **GEOMETRY,
            initial_state=state,
            ssm_state_indices=idx,
            cu_seqlens=ins["cu_seqlens"],
        )


def test_packed_rejects_noncontiguous_last_dim():
    """C9: a ``mixed_qkv`` view with ``stride(-1) != 1`` must be rejected —
    the kernel's per-tensor offsets assume unit element stride."""
    ins, idx, state = _assert_case_inputs()
    strided = torch.rand(
        2, 2 * PACKED_DIM, dtype=torch.bfloat16, device=ins["mixed_qkv"].device
    )[:, ::2]
    assert strided.stride(-1) != 1
    with pytest.raises(AssertionError, match="contiguous in the last dim"):
        fused_sigmoid_gating_delta_rule_update(
            A_log=ins["A_log"],
            a=ins["a"],
            b=ins["b"],
            dt_bias=ins["dt_bias"],
            mixed_qkv=strided,
            **GEOMETRY,
            initial_state=state,
            ssm_state_indices=idx,
            cu_seqlens=ins["cu_seqlens"],
        )


def test_packed_requires_head_geometry():
    """C10: head geometry can't be inferred from a packed buffer, so the
    geometry kwargs are mandatory in packed mode."""
    ins, idx, state = _assert_case_inputs()
    with pytest.raises(AssertionError, match="head geometry"):
        fused_sigmoid_gating_delta_rule_update(
            A_log=ins["A_log"],
            a=ins["a"],
            b=ins["b"],
            dt_bias=ins["dt_bias"],
            mixed_qkv=ins["mixed_qkv"],
            initial_state=state,
            ssm_state_indices=idx,
            cu_seqlens=ins["cu_seqlens"],
        )


# --------------------------------------------------------------------------
# Group D: layer-level A/B via VLLM_GDN_PACKED_SPEC_QKV
# --------------------------------------------------------------------------


@pytest.fixture
def rearrange_counter(monkeypatch):
    """Count material ``rearrange_mixed_qkv`` calls (``None`` passthroughs on
    empty partitions are free and don't count)."""
    calls = {"n": 0}
    orig = QwenGatedDeltaNetAttention.rearrange_mixed_qkv

    def counting(self, mixed_qkv):
        if mixed_qkv is not None:
            calls["n"] += 1
        return orig(self, mixed_qkv)

    monkeypatch.setattr(QwenGatedDeltaNetAttention, "rearrange_mixed_qkv", counting)
    return calls


def test_layer_pure_spec_packed_matches_rearrange(monkeypatch, rearrange_counter):
    """D11: pure-spec batch through the real ``_forward_core`` — the packed
    path (flag on, default) must be bit-identical to the rearrange path (flag
    off), and the counter must show the rearrange was actually skipped."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    weights = prefill_harness._make_weights(device)
    batch = BatchSpec(seq_lens=[103, 231], query_lens=[3, 3])
    inputs = prefill_harness._make_inputs(6, device)
    st = spec_harness._rand_states(device, 8)
    seeds = {}
    for i, col in enumerate((1, 2, 3)):
        seeds[(0, col)] = (st[i][0] if col == 1 else None, st[i][1])
    for i, col in enumerate((3, 4, 5)):
        seeds[(1, col)] = (st[4 + i][0] if col == 3 else None, st[4 + i][1])
    drafts = [spec_harness.NUM_SPEC, spec_harness.NUM_SPEC]

    monkeypatch.setenv("VLLM_GDN_PACKED_SPEC_QKV", "0")
    out_rearr, _, ssm_rearr, _ = spec_harness._run_spec(
        "all", batch, inputs, weights, drafts, [2, 1], seeds
    )
    assert rearrange_counter["n"] == 1

    rearrange_counter["n"] = 0
    monkeypatch.setenv("VLLM_GDN_PACKED_SPEC_QKV", "1")
    out_packed, _, ssm_packed, _ = spec_harness._run_spec(
        "all", batch, inputs, weights, drafts, [2, 1], seeds
    )
    assert rearrange_counter["n"] == 0

    torch.testing.assert_close(out_packed, out_rearr, atol=0, rtol=0)
    torch.testing.assert_close(ssm_packed, ssm_rearr, atol=0, rtol=0)


def test_layer_peeled_decode_packed_matches_rearrange(monkeypatch, rearrange_counter):
    """D12: mixed decode+prefill batch (peeled decode rows) — flag on/off A/B
    in one process: identical outputs and final states, rearrange skipped on
    the packed run."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    weights = prefill_harness._make_weights(device)
    # Row 0: in-block decode (seq 101, block 1); row 1: fresh 192-token
    # prefill — the proven mixed-batch shape of test_gdn_all_mode_decode.
    batch = BatchSpec(seq_lens=[101, 192], query_lens=[1, 192])
    inputs = prefill_harness._make_inputs(193, device)
    conv_s = torch.randn_like(
        prefill_harness._make_pools(1, torch.float32, device)[0][0]
    )
    ssm_s = torch.randn_like(
        prefill_harness._make_pools(1, torch.float32, device)[1][0]
    )
    seeds = {(0, 1): (conv_s, ssm_s)}

    monkeypatch.setenv("VLLM_GDN_PACKED_SPEC_QKV", "0")
    out_rearr, conv_rearr, ssm_rearr, _ = decode_harness._run_mode(
        "align", batch, inputs, False, seeds, weights
    )
    assert rearrange_counter["n"] == 1

    rearrange_counter["n"] = 0
    monkeypatch.setenv("VLLM_GDN_PACKED_SPEC_QKV", "1")
    out_packed, conv_packed, ssm_packed, _ = decode_harness._run_mode(
        "align", batch, inputs, False, seeds, weights
    )
    assert rearrange_counter["n"] == 0

    torch.testing.assert_close(out_packed, out_rearr, atol=0, rtol=0)
    torch.testing.assert_close(conv_packed, conv_rearr, atol=0, rtol=0)
    torch.testing.assert_close(ssm_packed, ssm_rearr, atol=0, rtol=0)
