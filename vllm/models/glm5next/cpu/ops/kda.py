# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-PyTorch reference implementations of the KDA (Kimi Delta Attention)
ops for the CPU platform.

These mirror the vendor Triton kernels
(``glm5next/nvidia/ops/third_party/kda``) math-for-math, in fp32, using the
recurrent formulation for both prefill and decode. Performance is not the
goal — same philosophy as the CPU MLA reference backend — the point is that
GLM-5.3-Flash can execute end-to-end on CPU with correct numerics.

Semantics replicated exactly from the kernels:

* q/k l2-normalization divides by ``sqrt(sum(x*x) + 1e-6)`` (epsilon inside
  the sqrt), then q is scaled by ``K**-0.5``.
* safe-gate:     ``y = lower_bound / (1 + exp(-exp(A_log) * (g + g_bias)))``
  non-safe gate: ``y = -exp(A_log) * softplus(g + g_bias, beta=1, threshold=20)``
  applied as a per-key-dim log-decay on the state: ``S *= exp(y)``.
* delta rule:    ``v' = (v - S @ k) * sigmoid(beta_raw)``; ``S += v' k^T``.
* output:        ``o = S @ q``.
* states live in a paged cache ``[slots, H, V, K]`` addressed through
  ``ssm_state_indices``; slot 0 is the NULL block and is never read from or
  written to (the kernel skips those programs entirely).
"""

from __future__ import annotations

import torch

_NULL_STATE_SLOT = 0


def _kda_gate(
    raw_g: torch.Tensor,
    a_log: torch.Tensor,
    g_bias: torch.Tensor | None,
    lower_bound: float | None,
    head_k_dim: int,
) -> torch.Tensor:
    """Compute the KDA gate in fp32.

    ``raw_g``: [..., H, K] head-shaped raw gate logits. Returns [..., H, K].
    ``a_log``: [H]. ``g_bias``: [H, K] or None (broadcast over the leading
    dims). ``lower_bound`` set selects the bounded safe-gate branch; None
    selects the -exp(A) * softplus branch.
    """
    g = raw_g.float()
    H = a_log.numel()
    assert g.shape[-2] == H and g.shape[-1] == head_k_dim, (
        f"gate shape mismatch: {tuple(g.shape)} vs H={H}, K={head_k_dim}"
    )

    a = torch.exp(a_log.float())  # [H]
    # broadcast a over [..., H, K]
    a = a.reshape(*([1] * (g.ndim - 2)), H, 1)

    if g_bias is not None:
        bias = g_bias.float()
        # the model parameter is flat [H * K]; accept both layouts
        if bias.ndim == 1:
            bias = bias.view(H, head_k_dim)
        g = g + bias  # [H, K] broadcasts over leading dims

    if lower_bound is not None:
        # safe-gate branch: bounded to (lower_bound, 0)
        y = lower_bound / (1.0 + torch.exp(-(a * g)))
    else:
        # non-safe branch: y = -exp(A) * softplus(g + bias)
        # softplus with beta=1, threshold=20 (linear above the threshold)
        sp = torch.where(
            g > 20.0,
            g,
            torch.log1p(torch.exp(g)),
        )
        y = -a * sp

    return y


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    """Kernel-exact l2norm: x / sqrt(sum(x*x) + 1e-6), fp32."""
    x = x.float()
    return x / torch.sqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)


def _recurrent_sequence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the KDA recurrence over one sequence.

    q, k: [T, H, K] (raw; normalized here when the caller requests it — the
    caller pre-normalizes instead, see wrappers). v: [T, H, V].
    gate: [T, H, K] fp32 log-decay. beta: [T, H] fp32 (already sigmoided).
    state: [H, V, K] fp32 initial state or None for zeros.

    Returns (o [T, H, V], final_state [H, V, K] fp32).
    """
    T, H, K = q.shape
    V = v.shape[-1]

    q_n = _l2norm(q) * scale
    k_n = _l2norm(k)

    decay = torch.exp(gate)  # [T, H, K]

    S = torch.zeros(H, V, K, dtype=torch.float32, device=v.device)
    if state is not None:
        S = S + state.float()

    o = torch.empty(T, H, V, dtype=torch.float32, device=v.device)
    for t in range(T):
        # column-wise decay (per key dim), broadcast over V
        S = S * decay[t].unsqueeze(1)  # [H, V, K] * [H, 1, K]
        # delta rule: v' = (v - S @ k) * beta ; S += v' k^T
        kv = (S * k_n[t].unsqueeze(1)).sum(dim=-1)  # [H, V]
        v_delta = (v[t].float() - kv) * beta[t].unsqueeze(-1)  # [H, V]
        S = S + v_delta.unsqueeze(-1) * k_n[t].unsqueeze(-2)  # [H, V, K]
        # output: o = S @ q
        o[t] = (S * q_n[t].unsqueeze(1)).sum(dim=-1)  # [H, V]

    return o, S


def _normalize_gate_and_beta(
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    g_bias: torch.Tensor | None,
    lower_bound: float | None,
    head_k_dim: int,
    sigmoid_beta: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gate logits [1, T, H, K] -> fp32 gate; beta [1, T, H] -> fp32."""
    gate = _kda_gate(raw_g, a_log, g_bias, lower_bound, head_k_dim)
    beta = beta.float()
    if sigmoid_beta:
        beta = torch.sigmoid(beta)
    return gate, beta


def fused_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    sigmoid_beta: bool = False,
    a_log: torch.Tensor | None = None,
    g_bias: torch.Tensor | None = None,
    compute_gate: bool = False,
    lower_bound: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """CPU reference for the fused recurrent KDA op (decode path).

    Shapes (varlen, B == 1): q/k [1, T, H, K], v [1, T, H, V], g
    [1, T, H, K] raw gate logits, beta [1, T, H] raw logits. Reads and
    writes the paged state cache in place through ``ssm_state_indices``
    when provided (slot 0 is the NULL block and is skipped), matching the
    kernel's continuous-batching semantics.
    """
    if num_accepted_tokens is not None:
        raise NotImplementedError(
            "Speculative decoding is not yet supported by the CPU KDA "
            "reference implementation."
        )
    if not use_qk_l2norm_in_kernel:
        raise NotImplementedError(
            "CPU KDA reference always l2-normalizes q/k in the op "
            "(use_qk_l2norm_in_kernel=False is not supported)."
        )

    assert q.shape[0] == 1, "varlen layout expected (B == 1)"
    assert beta is not None, "beta is required"
    T, K = q.shape[1], q.shape[3]
    if scale is None:
        scale = K**-0.5

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()

    if compute_gate:
        assert a_log is not None and g_bias is not None, (
            "compute_gate requires a_log and g_bias"
        )
        # compute_gate implements the bounded (safe_gate) branch only,
        # mirroring the vendor kernel.
        assert lower_bound is not None, (
            "compute_gate implements the bounded (safe_gate) branch only"
        )
        gate, beta_n = _normalize_gate_and_beta(
            g, beta, a_log, g_bias, lower_bound, K, sigmoid_beta
        )
    else:
        gate = g.float()
        beta_n = beta.float()
        if sigmoid_beta:
            beta_n = torch.sigmoid(beta_n)

    if out is None:
        out = torch.empty_like(v)
    else:
        assert out.shape == v.shape

    if cu_seqlens is None:
        cu_seqlens = torch.tensor([0, T], dtype=torch.long, device=q.device)
        lens = [T]
    else:
        cu_seqlens = cu_seqlens.to(torch.long).cpu()
        lens = torch.diff(cu_seqlens).tolist()

    N = len(lens)
    gate = gate[0]  # [T, H, K]
    beta_n = beta_n[0]  # [T, H]
    q_f, k_f, v_f = q[0], k[0], v[0]

    for i in range(N):
        bos, eos = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
        if eos <= bos:
            continue

        state_i = None
        writeback_index = None
        if ssm_state_indices is not None:
            slot = int(ssm_state_indices[i])
            if slot <= _NULL_STATE_SLOT:
                # NULL block: the kernel skips these programs entirely.
                continue
            if initial_state is not None:
                state_i = initial_state[slot]
            writeback_index = slot
        elif initial_state is not None:
            # dense [B, H, V, K] layout, one state per batch element
            state_i = initial_state[i]
            writeback_index = i

        o_i, S = _recurrent_sequence(
            q_f[bos:eos],
            k_f[bos:eos],
            v_f[bos:eos],
            gate[bos:eos],
            beta_n[bos:eos],
            state_i,
            scale,
        )

        out[0, bos:eos] = o_i.to(out.dtype)
        if (
            writeback_index is not None
            and inplace_final_state
            and initial_state is not None
        ):
            initial_state[writeback_index] = S.to(initial_state.dtype)

    return out, initial_state


def chunk_kda_with_fused_gate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """CPU reference for the chunked KDA prefill op.

    Uses the recurrent formulation (mathematically identical to the chunked
    algorithm) — correctness over speed, per the CPU platform's reference
    philosophy. ``beta`` arrives pre-sigmoided (fp32), matching the chunk
    path's contract in ``common/kda.py``. ``initial_state`` is the gathered
    per-sequence state ``[N, H, V, K]``; returns ``(o, final_state)`` with
    ``final_state`` shaped ``[N, H, V, K]``.
    """
    if not use_qk_l2norm_in_kernel:
        raise NotImplementedError(
            "CPU KDA reference always l2-normalizes q/k in the op."
        )

    assert q.shape[0] == 1, "varlen layout expected (B == 1)"
    T, H, K = q.shape[1], q.shape[2], q.shape[3]
    V = v.shape[-1]
    scale = K**-0.5

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    raw_g = raw_g.contiguous()
    beta = beta.contiguous()

    # Gate: safe_gate selects the bounded branch (needs lower_bound);
    # otherwise the -exp(A) * softplus branch.
    gate = _kda_gate(
        raw_g,
        A_log,
        g_bias,
        lower_bound if safe_gate else None,
        K,
    )
    beta_n = beta.float()  # already sigmoided by the caller

    if cu_seqlens is None:
        cu_seqlens = torch.tensor([0, T], dtype=torch.long, device=q.device)
        lens = [T]
    else:
        cu_seqlens = cu_seqlens.to(torch.long).cpu()
        lens = torch.diff(cu_seqlens).tolist()

    N = len(lens)
    o = torch.empty_like(v)
    final_state = (
        torch.zeros(N, H, V, K, dtype=v.dtype, device=v.device)
        if output_final_state
        else None
    )

    gate = gate[0]  # [T, H, K]
    beta_n = beta_n[0]  # [T, H]
    q_f, k_f, v_f = q[0], k[0], v[0]

    for i in range(N):
        bos, eos = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
        if eos <= bos:
            continue

        state_i = initial_state[i] if initial_state is not None else None
        o_i, S = _recurrent_sequence(
            q_f[bos:eos],
            k_f[bos:eos],
            v_f[bos:eos],
            gate[bos:eos],
            beta_n[bos:eos],
            state_i,
            scale,
        )
        o[0, bos:eos] = o_i.to(o.dtype)
        if final_state is not None:
            final_state[i] = S.to(final_state.dtype)

    return o, final_state
