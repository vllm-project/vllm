# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU reference KDA (Kimi Delta Attention) ops for GLM-5.3-Flash.

Checks the pure-torch implementations in
``vllm/models/glm5next/cpu/ops/kda.py`` against an independent
transcription of the vendor Triton kernels' math. The reference below was
additionally validated against the vendor ``fused_recurrent_kda`` kernel
under ``TRITON_INTERPRET=1`` (outputs matched to ~1e-9 on fp32 inputs),
so agreement here pins the packaged ops to the kernel semantics.
"""

import pytest
import torch

from vllm.models.glm5next.cpu.ops.kda import (
    chunk_kda_with_fused_gate,
    fused_recurrent_kda,
)
from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

torch.manual_seed(12345)

H, K, V = 2, 64, 64
LOWER_BOUND = -5.0
SEQ_LENS = [17, 13, 10]  # varlen batch, B == 1 flattened
SLOTS = [1, 3, 5]
T = sum(SEQ_LENS)
CU = torch.tensor([0, 17, 30, 40], dtype=torch.int32)


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def _reference(
    q,
    k,
    v,
    g,
    beta,
    a_log,
    g_bias,
    safe,
    init_states,
    lens,
    precomputed_gate=False,
):
    """Independent transcription of the KDA recurrence.

    init_states: per-sequence [H, V, K] tensors or None. Returns
    (outputs [T, H, V], final states list).
    """
    qn = _l2norm(q[0]) * (K**-0.5)
    kn = _l2norm(k[0])
    if precomputed_gate:
        gate = g[0].float()
    else:
        a = torch.exp(a_log.float()).view(1, H, 1)
        x = g[0].float() + g_bias.float().view(H, K)
        if safe:
            gate = LOWER_BOUND / (1.0 + torch.exp(-(a * x)))
        else:
            sp = torch.where(x > 20.0, x, torch.log1p(torch.exp(x)))
            gate = -a * sp
    b = torch.sigmoid(beta[0].float())

    o = torch.empty(T, H, V)
    finals = []
    bos = 0
    for i, n in enumerate(lens):
        S = (
            torch.zeros(H, V, K)
            if init_states is None
            else init_states[i].float().clone()
        )
        for t in range(bos, bos + n):
            S = S * torch.exp(gate[t]).unsqueeze(1)
            kv = (S * kn[t].unsqueeze(1)).sum(-1)
            vd = (v[0, t].float() - kv) * b[t].unsqueeze(-1)
            S = S + vd.unsqueeze(-1) * kn[t].unsqueeze(-2)
            o[t] = (S * qn[t].unsqueeze(1)).sum(-1)
        finals.append(S.clone())
        bos += n
    return o, finals


def _inputs():
    return (
        torch.randn(1, T, H, K),
        torch.randn(1, T, H, K),
        torch.randn(1, T, H, V),
        torch.randn(1, T, H, K) * 0.5,
        torch.randn(1, T, H) * 0.5,
        torch.randn(H) * 0.3 - 1.0,
        torch.randn(H, K) * 0.1,
    )


def _check(name, a, b, tol=1e-5):
    d = (a.float() - b.float()).abs().max().item()
    assert d <= tol, f"{name}: maxdiff={d:.3e}"


@pytest.mark.parametrize("safe", [True, False])
def test_chunk_kda_matches_reference(safe):
    """The chunk (prefill) op must match the reference recurrence for both
    the bounded safe gate and the softplus gate."""
    q, k, v, g, beta, a_log, g_bias = _inputs()
    beta_sig = torch.sigmoid(beta)  # chunk contract: pre-sigmoided fp32
    init = torch.randn(len(SLOTS), H, V, K) * 0.3

    o_ref, finals_ref = _reference(
        q, k, v, g, beta, a_log, g_bias, safe, list(init), SEQ_LENS
    )
    o, fs = chunk_kda_with_fused_gate(
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        beta_sig.clone(),
        a_log,
        g_bias,
        initial_state=init.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=CU,
        safe_gate=safe,
        lower_bound=LOWER_BOUND if safe else None,
    )
    _check(f"chunk output (safe={safe})", o[0], o_ref)
    _check(f"chunk final state (safe={safe})", fs, torch.stack(finals_ref))


@pytest.mark.parametrize("safe", [True, False])
def test_chunk_kda_no_initial_state(safe):
    q, k, v, g, beta, a_log, g_bias = _inputs()
    beta_sig = torch.sigmoid(beta)

    o_ref, finals_ref = _reference(
        q, k, v, g, beta, a_log, g_bias, safe, None, SEQ_LENS
    )
    o, fs = chunk_kda_with_fused_gate(
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        beta_sig.clone(),
        a_log,
        g_bias,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=CU,
        safe_gate=safe,
        lower_bound=LOWER_BOUND if safe else None,
    )
    _check(f"chunk output no-init (safe={safe})", o[0], o_ref)
    _check(f"chunk final state no-init (safe={safe})", fs, torch.stack(finals_ref))


def test_recurrent_kda_paged_cache():
    """The recurrent (decode) op must read/write the paged state cache
    through ``ssm_state_indices`` exactly like the vendor kernel: slot 0 is
    the NULL block and is skipped, each sequence owns its slot."""
    q, k, v, g, beta, a_log, g_bias = _inputs()
    init = torch.randn(8, H, V, K) * 0.3

    o_ref, finals_ref = _reference(
        q, k, v, g, beta, a_log, g_bias, True, [init[s] for s in SLOTS], SEQ_LENS
    )

    cache = init.clone()
    o, _ = fused_recurrent_kda(
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        beta.clone(),
        initial_state=cache,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=CU,
        ssm_state_indices=torch.tensor(SLOTS, dtype=torch.int32),
        sigmoid_beta=True,
        a_log=a_log,
        g_bias=g_bias,
        compute_gate=True,
        lower_bound=LOWER_BOUND,
    )
    _check("recurrent output", o[0], o_ref)
    for i, s in enumerate(SLOTS):
        _check(f"recurrent slot {s} state", cache[s], finals_ref[i])

    # untouched slots must stay untouched
    untouched = [s for s in range(8) if s not in SLOTS]
    _check(
        "untouched cache slots",
        cache[untouched],
        init[untouched],
        tol=0.0,
    )


def test_recurrent_kda_null_slot_skipped():
    """Slot 0 (NULL block) sequences are skipped entirely, mirroring the
    kernel's early return."""
    q, k, v, g, beta, a_log, g_bias = _inputs()
    init = torch.randn(4, H, V, K)

    slots = torch.tensor([0, 2, 0], dtype=torch.int32)
    cu = torch.tensor([0, 5, 18, T], dtype=torch.int32)
    bos, eos = 5, 18
    o_ref, finals_ref = _reference(
        q[:, bos:eos].contiguous(),
        k[:, bos:eos].contiguous(),
        v[:, bos:eos].contiguous(),
        g[:, bos:eos].contiguous(),
        beta[:, bos:eos].contiguous(),
        a_log,
        g_bias,
        True,
        [init[2]],
        [eos - bos],
    )

    cache = init.clone()
    o, _ = fused_recurrent_kda(
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        beta.clone(),
        initial_state=cache,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
        ssm_state_indices=slots,
        sigmoid_beta=True,
        a_log=a_log,
        g_bias=g_bias,
        compute_gate=True,
        lower_bound=LOWER_BOUND,
    )
    # only sequence 1 (slot 2) produced output; the reference runs that
    # sequence alone, so its rows are 0..n-1
    n = 18 - 5
    _check("null-slot output rows", o[0, 5:18], o_ref[:n])
    _check("null-slot state", cache[2], finals_ref[0])
    _check("null slot untouched", cache[0], init[0], tol=0.0)


def test_recurrent_kda_precomputed_gate():
    """compute_gate=False consumes g as the final log-decay (the vendor
    wrapper's contract; softplus gates reach decode pre-computed)."""
    q, k, v, g, beta, a_log, g_bias = _inputs()
    init = torch.randn(8, H, V, K) * 0.3

    a = torch.exp(a_log.float()).view(1, H, 1)
    x = g[0].float() + g_bias.float().view(H, K)
    sp = torch.where(x > 20.0, x, torch.log1p(torch.exp(x)))
    gate_final = (-a * sp).unsqueeze(0).contiguous()

    o_ref, _ = _reference(
        q,
        k,
        v,
        gate_final,
        beta,
        None,
        None,
        True,
        [init[s] for s in SLOTS],
        SEQ_LENS,
        precomputed_gate=True,
    )
    cache = init.clone()
    o, _ = fused_recurrent_kda(
        q.clone(),
        k.clone(),
        v.clone(),
        gate_final,
        beta.clone(),
        initial_state=cache,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=CU,
        ssm_state_indices=torch.tensor(SLOTS, dtype=torch.int32),
        sigmoid_beta=True,
        compute_gate=False,
    )
    _check("pre-computed gate output", o[0], o_ref)


@pytest.mark.parametrize("safe", [True, False])
def test_recurrent_and_chunk_agree(safe):
    """Recurrent (decode-form) and chunk (prefill-form) ops implement the
    same math and must agree on identical inputs."""
    q, k, v, g, beta, a_log, g_bias = _inputs()
    beta_sig = torch.sigmoid(beta)
    init = torch.randn(len(SLOTS), H, V, K) * 0.3

    o_chunk, fs_chunk = chunk_kda_with_fused_gate(
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        beta_sig.clone(),
        a_log,
        g_bias,
        initial_state=init.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=CU,
        safe_gate=safe,
        lower_bound=LOWER_BOUND if safe else None,
    )

    cache = torch.zeros(8, H, V, K)
    cache[SLOTS] = init
    if safe:
        g_rec, compute_gate = g.clone(), True
    else:
        # the recurrent op computes gates from raw logits only via the
        # bounded safe branch; the softplus gate arrives pre-computed
        a = torch.exp(a_log.float()).view(1, H, 1)
        x = g[0].float() + g_bias.float().view(H, K)
        sp = torch.where(x > 20.0, x, torch.log1p(torch.exp(x)))
        g_rec = (-a * sp).unsqueeze(0).contiguous()
        compute_gate = False
    o_rec, _ = fused_recurrent_kda(
        q.clone(),
        k.clone(),
        v.clone(),
        g_rec,
        beta_sig.clone(),
        initial_state=cache,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=CU,
        ssm_state_indices=torch.tensor(SLOTS, dtype=torch.int32),
        sigmoid_beta=False,  # beta already sigmoided, like the chunk path
        a_log=a_log,
        g_bias=g_bias,
        compute_gate=compute_gate,
        lower_bound=LOWER_BOUND if compute_gate else None,
    )
    _check(f"chunk vs recurrent output (safe={safe})", o_chunk[0], o_rec[0])
    _check(f"chunk vs recurrent state (safe={safe})", fs_chunk, cache[SLOTS])
