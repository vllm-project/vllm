# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU backend for the GLM-5.3-Flash KDA operator.

The reference implementation used here is deliberately written out longhand in
``_reference_kda`` and shares no code with the backend under test, so a shared
misunderstanding cannot cancel out. The reference mirrors
``tests/models/glm5next/test_kda_recurrent.py::naive_recurrent_kda`` -- the
upstream CUDA-side oracle -- including its V-major ``[H, V, K]`` state.
"""

import pytest
import torch

from vllm.models.glm5next.cpu import (
    chunk_kda_with_fused_gate,
    fused_recurrent_kda,
)
from vllm.platforms import current_platform

# Production-shaped dims: a simplified [H] / [H, D] layout would hide the real
# broadcasting of A_log ([1, 1, H, 1]) and g_bias ([H * D]).
H, D = 8, 128
LOWER_BOUND = -5.0
NULL_BLOCK_ID = 0


def _chunk_beta(raw_beta: torch.Tensor) -> torch.Tensor:
    """Pre-sigmoid beta the way the common chunk path does.

    ``common/kda.py`` sigmoids beta in fp32 before calling the chunk operator,
    so the backend receives it already gated and the reference must be fed the
    same value with ``sigmoid_beta=False``. Feeding the raw logits instead is a
    silent, large error (the two betas differ by far more than any tolerance).
    """
    return torch.sigmoid(raw_beta.float())


def _reference_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    g_bias: torch.Tensor,
    state: torch.Tensor,
    *,
    sigmoid_beta: bool,
    l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 reference for one sequence.

    ``[T, H, D]`` inputs and a V-major ``[H, D, D]`` state. ``beta`` is
    ``[T, H]`` (per head, broadcast along V).
    """
    q, k, v, raw_g, beta = (x.float() for x in (q, k, v, raw_g, beta))
    if l2norm:
        q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    # The query scale is a separate step from L2 normalization: the backend
    # applies it unconditionally (``q * scale``), with ``scale`` defaulting to
    # ``K**-0.5``. Keeping it here means ``l2norm`` toggles only the
    # normalization, so callers can compare either configuration.
    q = q * D**-0.5
    # A_log is a per-head scalar stored as [1, 1, H, 1]; it multiplies along the
    # head axis only, while g_bias adds along the key axis.
    a = a_log.reshape(-1)
    gate = LOWER_BOUND * torch.sigmoid(a.exp()[:, None] * (raw_g + g_bias))
    b = torch.sigmoid(beta) if sigmoid_beta else beta
    s = state.clone()
    out = torch.empty(q.shape[0], v.shape[1], v.shape[2], dtype=torch.float32)
    for t in range(q.shape[0]):
        # The gate is per (head, key-dim) and decays the state along its K
        # axis, so it must be broadcast on axis 1 of the V-major [H, V, K]
        # state -- matching the kernel's `b_h *= exp(b_gk[None, :])`.
        s = s * gate[t].exp().unsqueeze(1)
        u = b[t][:, None] * (v[t] - torch.einsum("hvk,hk->hv", s, k[t]))
        s = s + u[:, :, None] * k[t][:, None, :]
        out[t] = torch.einsum("hvk,hk->hv", s, q[t])
    return out, s


def _make_inputs(
    seq_lens: list[int],
    *,
    seed: int = 0,
    with_initial_state: bool = True,
):
    """Build token-flattened packed inputs plus a paged state pool."""
    g = torch.Generator().manual_seed(seed)
    T = sum(seq_lens)
    proj = H * D

    def rnd(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=g, dtype=torch.float32).to(dtype)

    q = rnd(1, T, H, D)
    k = rnd(1, T, H, D)
    v = rnd(1, T, H, D)
    raw_g = rnd(1, T, H, D)
    beta = rnd(1, T, H)
    a_log = torch.randn(1, 1, H, 1, generator=g, dtype=torch.float32)
    g_bias = 0.1 * torch.randn(H * D, generator=g, dtype=torch.float32)

    cu = [0]
    for length in seq_lens:
        cu.append(cu[-1] + length)
    cu_seqlens = torch.tensor(cu, dtype=torch.int32)

    num_slots = len(seq_lens) + 8
    state = torch.randn(num_slots, H, D, D, generator=g, dtype=torch.float32)
    return {
        "q": q,
        "k": k,
        "v": v,
        "raw_g": raw_g,
        "beta": beta,
        "a_log": a_log,
        "g_bias": g_bias,
        "cu_seqlens": cu_seqlens,
        "state": state,
        "proj": proj,
    }


def _slots_for(seq_lens: list[int], order: list[int]):
    """Map sequence -> slot, honouring ``order`` (which may be shuffled)."""
    return torch.tensor([order[i] + 1 for i in range(len(seq_lens))], dtype=torch.int32)


# ---------------------------------------------------------------------------
# oracle self-checks
# ---------------------------------------------------------------------------


def test_reference_is_sensitive_to_the_decay_axis():
    """The state is V-major, so the per-key decay must scale its K axis.

    A transposed decay still compiles and runs (KDA is square, so the shapes
    match), which means the backend and its reference could share the same
    wrong axis and cancel out. This test proves the reference's decay lands on
    the K axis by checking it against a hand-written first step.
    """
    d = _make_inputs([1], seed=100)
    a = d["a_log"].reshape(-1)
    g = d["g_bias"].view(H, D)
    gate = LOWER_BOUND * torch.sigmoid(
        a.exp()[:, None] * (d["raw_g"][0, 0].float() + g)
    )

    # A state whose only non-zero V row is the last one makes the axis choice
    # observable: decaying along K scales each entry by gate[:, o_k], so the
    # decayed row is not constant across K.
    state = torch.zeros(H, D, D)
    state[:, -1, :] = 1.0

    k_axis = state * gate.exp().unsqueeze(1)  # [H, V, K] * [H, 1, K]
    decayed_row = k_axis[:, -1, :]
    assert not torch.allclose(
        decayed_row, decayed_row[:, :1].expand(H, D), rtol=1e-3, atol=1e-3
    ), "the decay must vary along K, not collapse to one value per head"

    _, ref_state = _reference_kda(
        d["q"][0, :1],
        d["k"][0, :1],
        d["v"][0, :1],
        d["raw_g"][0, :1],
        d["beta"][0, :1],
        d["a_log"],
        g,
        state,
        sigmoid_beta=False,
        l2norm=False,
    )

    # Reconstruct the reference's first step: decay, then the delta update.
    k_ = d["k"][0, 0].float()
    v_ = d["v"][0, 0].float()
    b_ = d["beta"][0, 0].float()
    scaled = k_axis
    delta = b_[:, None] * (v_ - torch.einsum("hvk,hk->hv", scaled, k_))
    expected = scaled + delta[:, :, None] * k_[:, None, :]
    torch.testing.assert_close(ref_state, expected, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# packed prefill
# ---------------------------------------------------------------------------


def test_packed_prefill_sequence_isolation():
    """cu_seqlens=[0,3,5]: the second sequence must not inherit the first's
    state. A backend that ignores ``cu_seqlens`` treats all tokens as one
    sequence and fails here."""
    seq_lens = [3, 2]
    d = _make_inputs(seq_lens, seed=1)

    out, final = chunk_kda_with_fused_gate(
        q=d["q"],
        k=d["k"],
        v=d["v"],
        raw_g=d["raw_g"],
        beta=torch.sigmoid(d["beta"].float()),
        A_log=d["a_log"],
        g_bias=d["g_bias"],
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=d["cu_seqlens"],
        safe_gate=True,
        lower_bound=LOWER_BOUND,
    )

    assert final.shape == (2, H, D, D), final.shape

    # Sequence 0 starts from zeros (no initial state) and consumes only [0:3];
    # sequence 1 starts from zeros and consumes only [3:5].
    for n, (bos, eos) in enumerate([(0, 3), (3, 5)]):
        ref_out, ref_state = _reference_kda(
            d["q"][0, bos:eos],
            d["k"][0, bos:eos],
            d["v"][0, bos:eos],
            d["raw_g"][0, bos:eos],
            _chunk_beta(d["beta"][0, bos:eos]),
            d["a_log"],
            d["g_bias"].view(H, D),
            torch.zeros(H, D, D),
            sigmoid_beta=False,
        )
        torch.testing.assert_close(
            out[0, bos:eos].float(), ref_out, rtol=1e-2, atol=1e-3
        )
        torch.testing.assert_close(final[n], ref_state, rtol=1e-4, atol=1e-4)

    # The isolation invariant, stated directly: sequence 1's final state must
    # NOT equal what a single merged sequence would produce.
    _, merged = _reference_kda(
        d["q"][0],
        d["k"][0],
        d["v"][0],
        d["raw_g"][0],
        _chunk_beta(d["beta"][0]),
        d["a_log"],
        d["g_bias"].view(H, D),
        torch.zeros(H, D, D),
        sigmoid_beta=False,
    )
    assert not torch.allclose(final[1], merged, rtol=1e-3, atol=1e-3)


def test_packed_prefill_segments_match_independent_runs():
    """Running the packed batch must equal running each sequence alone."""
    seq_lens = [3, 2]
    d = _make_inputs(seq_lens, seed=2)

    _, final = chunk_kda_with_fused_gate(
        q=d["q"],
        k=d["k"],
        v=d["v"],
        raw_g=d["raw_g"],
        beta=torch.sigmoid(d["beta"].float()),
        A_log=d["a_log"],
        g_bias=d["g_bias"],
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=d["cu_seqlens"],
        safe_gate=True,
        lower_bound=LOWER_BOUND,
    )

    for n, (bos, eos) in enumerate([(0, 3), (3, 5)]):
        _, alone = chunk_kda_with_fused_gate(
            q=d["q"][:, bos:eos],
            k=d["k"][:, bos:eos],
            v=d["v"][:, bos:eos],
            raw_g=d["raw_g"][:, bos:eos],
            beta=torch.sigmoid(d["beta"][:, bos:eos].float()),
            A_log=d["a_log"],
            g_bias=d["g_bias"],
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=torch.tensor([0, eos - bos], dtype=torch.int32),
            safe_gate=True,
            lower_bound=LOWER_BOUND,
        )
        torch.testing.assert_close(final[n], alone[0], rtol=1e-4, atol=1e-4)


def test_ignoring_cu_seqlens_is_caught():
    """Counter-example guard: an implementation that ignores ``cu_seqlens``
    must fail the isolation check, otherwise the check is vacuous."""
    seq_lens = [3, 2]
    d = _make_inputs(seq_lens, seed=3)

    _, final = chunk_kda_with_fused_gate(
        q=d["q"],
        k=d["k"],
        v=d["v"],
        raw_g=d["raw_g"],
        beta=torch.sigmoid(d["beta"].float()),
        A_log=d["a_log"],
        g_bias=d["g_bias"],
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=d["cu_seqlens"],
        safe_gate=True,
        lower_bound=LOWER_BOUND,
    )

    # Simulate "cu_seqlens ignored" by running everything as one sequence and
    # taking its final state as sequence 1's.
    _, wrong = _reference_kda(
        d["q"][0],
        d["k"][0],
        d["v"][0],
        d["raw_g"][0],
        _chunk_beta(d["beta"][0]),
        d["a_log"],
        d["g_bias"].view(H, D),
        torch.zeros(H, D, D),
        sigmoid_beta=False,
    )
    with pytest.raises(AssertionError):
        torch.testing.assert_close(final[1], wrong, rtol=1e-4, atol=1e-4)


def test_packed_prefill_respects_distinct_initial_states():
    """Each sequence must start from its own ``[N, H, V, K]`` initial state."""
    seq_lens = [3, 2]
    d = _make_inputs(seq_lens, seed=4)
    init = d["state"][:2].clone()

    _, final = chunk_kda_with_fused_gate(
        q=d["q"],
        k=d["k"],
        v=d["v"],
        raw_g=d["raw_g"],
        beta=torch.sigmoid(d["beta"].float()),
        A_log=d["a_log"],
        g_bias=d["g_bias"],
        initial_state=init,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=d["cu_seqlens"],
        safe_gate=True,
        lower_bound=LOWER_BOUND,
    )

    for n, (bos, eos) in enumerate([(0, 3), (3, 5)]):
        _, ref_state = _reference_kda(
            d["q"][0, bos:eos],
            d["k"][0, bos:eos],
            d["v"][0, bos:eos],
            d["raw_g"][0, bos:eos],
            _chunk_beta(d["beta"][0, bos:eos]),
            d["a_log"],
            d["g_bias"].view(H, D),
            init[n],
            sigmoid_beta=False,
        )
        torch.testing.assert_close(final[n], ref_state, rtol=1e-4, atol=1e-4)

    assert not torch.allclose(init[0], init[1])


# ---------------------------------------------------------------------------
# plain decode
# ---------------------------------------------------------------------------


def test_decode_shuffled_slots():
    """slots=[7,2,5]: the pool must be indexed by the given slots, not by
    position."""
    seq_lens = [1, 1, 1]
    d = _make_inputs(seq_lens, seed=5)
    slots = torch.tensor([7, 2, 5], dtype=torch.int32)
    pool = d["state"].clone()
    before = pool.clone()

    out, final = fused_recurrent_kda(
        q=d["q"],
        k=d["k"],
        v=d["v"],
        g=d["raw_g"],
        beta=d["beta"],
        initial_state=pool,
        cu_seqlens=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        ssm_state_indices=slots,
        sigmoid_beta=True,
        a_log=d["a_log"],
        g_bias=d["g_bias"],
        compute_gate=True,
        lower_bound=LOWER_BOUND,
        inplace_final_state=True,
    )

    assert final is pool, "inplace_final_state must return the same object"

    for n, slot in enumerate(slots.tolist()):
        ref_out, ref_state = _reference_kda(
            d["q"][0, n : n + 1],
            d["k"][0, n : n + 1],
            d["v"][0, n : n + 1],
            d["raw_g"][0, n : n + 1],
            d["beta"][0, n : n + 1],
            d["a_log"],
            d["g_bias"].view(H, D),
            before[slot],
            sigmoid_beta=True,
        )
        torch.testing.assert_close(out[0, n].float(), ref_out[0], rtol=1e-2, atol=1e-3)
        torch.testing.assert_close(pool[slot], ref_state, rtol=1e-4, atol=1e-4)


def test_decode_only_touches_its_own_slot():
    """State pool entries outside the given slots must be byte-identical."""
    seq_lens = [1, 1]
    d = _make_inputs(seq_lens, seed=6)
    slots = torch.tensor([7, 2], dtype=torch.int32)
    pool = d["state"].clone()
    before = pool.clone()

    fused_recurrent_kda(
        q=d["q"],
        k=d["k"],
        v=d["v"],
        g=d["raw_g"],
        beta=d["beta"],
        initial_state=pool,
        cu_seqlens=torch.tensor([0, 1, 2], dtype=torch.int32),
        ssm_state_indices=slots,
        sigmoid_beta=True,
        a_log=d["a_log"],
        g_bias=d["g_bias"],
        compute_gate=True,
        lower_bound=LOWER_BOUND,
    )

    touched = set(slots.tolist())
    for slot in range(pool.shape[0]):
        if slot not in touched:
            torch.testing.assert_close(pool[slot], before[slot], rtol=0, atol=0)


def test_decode_null_slot_is_skipped():
    """A NULL_BLOCK_ID (<= 0) slot must not be read or written."""
    seq_lens = [1, 1]
    d = _make_inputs(seq_lens, seed=7)
    slots = torch.tensor([NULL_BLOCK_ID, 4], dtype=torch.int32)
    pool = d["state"].clone()
    before = pool.clone()

    fused_recurrent_kda(
        q=d["q"],
        k=d["k"],
        v=d["v"],
        g=d["raw_g"],
        beta=d["beta"],
        initial_state=pool,
        cu_seqlens=torch.tensor([0, 1, 2], dtype=torch.int32),
        ssm_state_indices=slots,
        sigmoid_beta=True,
        a_log=d["a_log"],
        g_bias=d["g_bias"],
        compute_gate=True,
        lower_bound=LOWER_BOUND,
    )

    torch.testing.assert_close(
        pool[NULL_BLOCK_ID], before[NULL_BLOCK_ID], rtol=0, atol=0
    )
    # The valid slot must still have been advanced.
    assert not torch.allclose(pool[4], before[4])


def test_null_slot_does_not_contaminate_its_sibling():
    """A skipped NULL sequence must not shift or leak into the live one."""
    d = _make_inputs([1, 1], seed=15)
    slots = [NULL_BLOCK_ID, 4]
    pool = d["state"].clone()
    before = pool.clone()

    out, _ = fused_recurrent_kda(
        q=d["q"],
        k=d["k"],
        v=d["v"],
        g=d["raw_g"],
        beta=d["beta"],
        initial_state=pool,
        cu_seqlens=torch.tensor([0, 1, 2], dtype=torch.int32),
        ssm_state_indices=torch.tensor(slots, dtype=torch.int32),
        sigmoid_beta=True,
        a_log=d["a_log"],
        g_bias=d["g_bias"],
        compute_gate=True,
        lower_bound=LOWER_BOUND,
    )

    # The live sequence consumed token 1 only, seeded from slot 4.
    ref_out, ref_state = _reference_kda(
        d["q"][0, 1:2],
        d["k"][0, 1:2],
        d["v"][0, 1:2],
        d["raw_g"][0, 1:2],
        d["beta"][0, 1:2],
        d["a_log"],
        d["g_bias"].view(H, D),
        before[4],
        sigmoid_beta=True,
    )
    torch.testing.assert_close(out[0, 1].float(), ref_out[0], rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(pool[4], ref_state, rtol=1e-4, atol=1e-4)


def test_zero_length_segment_consumes_no_tokens():
    """An empty ``cu_seqlens`` segment must consume nothing.

    The following segment has to see exactly its own token slice, not a
    shifted one. Segment lengths differ (``[0, 2]``) so a backend that ignores
    empty segments and starts consuming at the wrong offset produces visibly
    different output.

    Empty segments are a packed-prefill concept, so this goes through
    ``chunk_kda_with_fused_gate``. The recurrent entry point is a one-token-per
    -sequence decode kernel and rejects multi-token segments (see
    ``test_recurrent_rejects_multi_token_segments``); the NVIDIA kernel it
    mirrors likewise early-returns on ``T == 0`` rather than accepting them.
    """
    d = _make_inputs([0, 2], seed=16)

    out, final = chunk_kda_with_fused_gate(
        q=d["q"],
        k=d["k"],
        v=d["v"],
        raw_g=d["raw_g"],
        beta=torch.sigmoid(d["beta"].float()),
        A_log=d["a_log"],
        g_bias=d["g_bias"],
        initial_state=None,
        output_final_state=True,
        cu_seqlens=torch.tensor([0, 0, 2], dtype=torch.int32),
        safe_gate=True,
        lower_bound=LOWER_BOUND,
    )

    # Zero initial state, since the empty segment gets none.
    zeros = torch.zeros_like(final[1])
    torch.testing.assert_close(final[0], zeros, rtol=0, atol=0)

    # The live segment consumed exactly tokens [0:2], so it must match a
    # reference run over that slice alone.
    ref_out, ref_state = _reference_kda(
        d["q"][0, 0:2],
        d["k"][0, 0:2],
        d["v"][0, 0:2],
        d["raw_g"][0, 0:2],
        torch.sigmoid(d["beta"][0, 0:2].float()),
        d["a_log"],
        d["g_bias"].view(H, D),
        zeros,
        sigmoid_beta=False,
        l2norm=False,
    )
    torch.testing.assert_close(final[1], ref_state, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out[0, 0:2].float(), ref_out, rtol=1e-2, atol=2e-3)


# ---------------------------------------------------------------------------
# prefill -> decode
# ---------------------------------------------------------------------------


def test_prefill_output_feeds_decode_as_initial_state():
    """The final state of a packed prefill must be a valid decode seed.

    When ``ssm_state_indices`` is given the returned state *is* the pool and is
    addressed by slot, so the decode result is read back as
    ``final[slots[n]]`` -- not ``final[n]``. Slots must be non-NULL (0 is
    ``NULL_BLOCK_ID``), which is itself asserted by the neighbouring test.
    """
    seq_lens = [3, 2]
    d = _make_inputs(seq_lens, seed=8)

    _, prefill_final = chunk_kda_with_fused_gate(
        q=d["q"],
        k=d["k"],
        v=d["v"],
        raw_g=d["raw_g"],
        beta=torch.sigmoid(d["beta"].float()),
        A_log=d["a_log"],
        g_bias=d["g_bias"],
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=d["cu_seqlens"],
        safe_gate=True,
        lower_bound=LOWER_BOUND,
    )

    # Continue both sequences with one more decode token, handing the prefill
    # states over through non-NULL slots.
    nxt = _make_inputs([1, 1], seed=9)
    slots = [1, 2]
    pool = torch.zeros(4, H, D, D, dtype=torch.float32)
    pool[slots[0]] = prefill_final[0]
    pool[slots[1]] = prefill_final[1]
    pool_before = pool.clone()

    out, final = fused_recurrent_kda(
        q=nxt["q"],
        k=nxt["k"],
        v=nxt["v"],
        g=nxt["raw_g"],
        beta=nxt["beta"],
        initial_state=pool,
        cu_seqlens=torch.tensor([0, 1, 2], dtype=torch.int32),
        ssm_state_indices=torch.tensor(slots, dtype=torch.int32),
        sigmoid_beta=True,
        a_log=nxt["a_log"],
        g_bias=nxt["g_bias"],
        compute_gate=True,
        lower_bound=LOWER_BOUND,
    )

    assert final is pool
    for n, slot in enumerate(slots):
        ref_out, ref_state = _reference_kda(
            nxt["q"][0, n : n + 1],
            nxt["k"][0, n : n + 1],
            nxt["v"][0, n : n + 1],
            nxt["raw_g"][0, n : n + 1],
            nxt["beta"][0, n : n + 1],
            nxt["a_log"],
            nxt["g_bias"].view(H, D),
            pool_before[slot],
            sigmoid_beta=True,
        )
        torch.testing.assert_close(out[0, n].float(), ref_out[0], rtol=1e-2, atol=1e-3)
        torch.testing.assert_close(final[slot], ref_state, rtol=1e-4, atol=1e-4)


def test_decode_state_is_addressed_by_slot_not_by_sequence():
    """``final`` is the pool, so ``final[n]`` is NOT sequence ``n``.

    Reading the result by sequence position silently returns an unrelated (or
    untouched) slot whenever the slots are shuffled. Pin the difference down
    so this cannot regress into a passing-but-wrong test.
    """
    d = _make_inputs([1, 1], seed=14)
    slots = [3, 1]
    pool = d["state"].clone()
    before = pool.clone()

    _, final = fused_recurrent_kda(
        q=d["q"],
        k=d["k"],
        v=d["v"],
        g=d["raw_g"],
        beta=d["beta"],
        initial_state=pool,
        cu_seqlens=torch.tensor([0, 1, 2], dtype=torch.int32),
        ssm_state_indices=torch.tensor(slots, dtype=torch.int32),
        sigmoid_beta=True,
        a_log=d["a_log"],
        g_bias=d["g_bias"],
        compute_gate=True,
        lower_bound=LOWER_BOUND,
    )

    # Sequence 0 lives in slot 3 and sequence 1 in slot 1, so the
    # sequence-indexed read `final[0]` is slot 0 -- which was never written.
    torch.testing.assert_close(final[0], before[0], rtol=0, atol=0)

    for n, slot in enumerate(slots):
        _, ref_state = _reference_kda(
            d["q"][0, n : n + 1],
            d["k"][0, n : n + 1],
            d["v"][0, n : n + 1],
            d["raw_g"][0, n : n + 1],
            d["beta"][0, n : n + 1],
            d["a_log"],
            d["g_bias"].view(H, D),
            before[slot],
            sigmoid_beta=True,
        )
        torch.testing.assert_close(final[slot], ref_state, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# speculative recurrent state routing
# ---------------------------------------------------------------------------


def test_speculative_state_slots_match_reference(monkeypatch: pytest.MonkeyPatch):
    """Accepted state is read from accepted-1 and every token writes its slot."""
    from vllm.models.glm5next.cpu import kda as cpu_kda

    native_calls = 0
    native_recurrent = cpu_kda._native_recurrent_kda

    def count_native_calls(*args, **kwargs):
        nonlocal native_calls
        native_calls += 1
        return native_recurrent(*args, **kwargs)

    monkeypatch.setattr(cpu_kda, "_native_recurrent_kda", count_native_calls)
    seq_lens = [3, 3]
    d = _make_inputs(seq_lens, seed=10)
    slots = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    accepted = torch.tensor([2, 1], dtype=torch.int32)
    pool = d["state"].clone()
    before = pool.clone()

    out, final = fused_recurrent_kda(
        q=d["q"],
        k=d["k"],
        v=d["v"],
        g=d["raw_g"],
        beta=d["beta"],
        initial_state=pool,
        cu_seqlens=torch.tensor([0, 3, 6], dtype=torch.int32),
        ssm_state_indices=slots,
        num_accepted_tokens=accepted,
        sigmoid_beta=True,
        a_log=d["a_log"],
        g_bias=d["g_bias"],
        compute_gate=True,
        lower_bound=LOWER_BOUND,
    )

    expected_out = torch.empty_like(out[0], dtype=torch.float32)
    expected_pool = before.clone()
    for n in range(2):
        start = int(accepted[n].item()) - 1
        state = before[int(slots[n, start].item())]
        ref_out, state = _reference_kda(
            d["q"][0, n * 3 : (n + 1) * 3],
            d["k"][0, n * 3 : (n + 1) * 3],
            d["v"][0, n * 3 : (n + 1) * 3],
            d["raw_g"][0, n * 3 : (n + 1) * 3],
            d["beta"][0, n * 3 : (n + 1) * 3],
            d["a_log"],
            d["g_bias"].view(H, D),
            state,
            sigmoid_beta=True,
        )
        expected_out[n * 3 : (n + 1) * 3] = ref_out
        for t in range(3):
            _, token_state = _reference_kda(
                d["q"][0, n * 3 + t : n * 3 + t + 1],
                d["k"][0, n * 3 + t : n * 3 + t + 1],
                d["v"][0, n * 3 + t : n * 3 + t + 1],
                d["raw_g"][0, n * 3 + t : n * 3 + t + 1],
                d["beta"][0, n * 3 + t : n * 3 + t + 1],
                d["a_log"],
                d["g_bias"].view(H, D),
                before[int(slots[n, start].item())]
                if t == 0
                else expected_pool[int(slots[n, t - 1].item())],
                sigmoid_beta=True,
            )
            expected_pool[int(slots[n, t].item())] = token_state

    torch.testing.assert_close(out[0].float(), expected_out, rtol=1e-2, atol=1e-3)
    for n in range(2):
        for t in range(3):
            torch.testing.assert_close(
                final[int(slots[n, t].item())],
                expected_pool[int(slots[n, t].item())],
                rtol=1e-4,
                atol=1e-4,
            )
    if hasattr(torch.ops._C, "glm5next_kda_recurrent"):
        assert native_calls == sum(seq_lens)


def test_recurrent_non_inplace_returns_per_token_states():
    """The non-inplace recurrent contract stores one state per token."""
    d = _make_inputs([1, 1], seed=17)
    initial = d["state"].clone()
    before = initial.clone()
    _out, final = fused_recurrent_kda(
        q=d["q"],
        k=d["k"],
        v=d["v"],
        g=d["raw_g"],
        beta=d["beta"],
        initial_state=initial,
        cu_seqlens=torch.tensor([0, 1, 2], dtype=torch.int32),
        ssm_state_indices=torch.tensor([3, 4], dtype=torch.int32),
        sigmoid_beta=True,
        a_log=d["a_log"],
        g_bias=d["g_bias"],
        compute_gate=True,
        lower_bound=LOWER_BOUND,
        inplace_final_state=False,
    )

    assert final is not initial
    assert final.shape == (2, H, D, D)
    torch.testing.assert_close(initial, before, rtol=0, atol=0)
    for n, slot in enumerate((3, 4)):
        _, expected = _reference_kda(
            d["q"][0, n : n + 1],
            d["k"][0, n : n + 1],
            d["v"][0, n : n + 1],
            d["raw_g"][0, n : n + 1],
            d["beta"][0, n : n + 1],
            d["a_log"],
            d["g_bias"].view(H, D),
            before[slot],
            sigmoid_beta=True,
        )
        torch.testing.assert_close(final[n], expected, rtol=1e-4, atol=1e-4)


def test_state_helpers_match_cuda_helper_contract():
    """CPU gather/scatter preserve initialized rows and cache slot order."""
    from vllm.models.glm5next.cpu import gather_initial_states_cpu, scatter_states_cpu

    state = torch.arange(4 * 2 * 2 * 2, dtype=torch.float32).reshape(4, 2, 2, 2)
    indices = torch.tensor([3, 1], dtype=torch.int32)
    has_initial = torch.tensor([True, False])
    gathered = gather_initial_states_cpu(state, indices, has_initial)
    torch.testing.assert_close(gathered[0], state[3])
    torch.testing.assert_close(gathered[1], torch.zeros_like(state[0]))

    source = torch.full((2, 2, 2, 2), -1.0)
    scatter_states_cpu(state, source, indices)
    torch.testing.assert_close(state[3], source[0])
    torch.testing.assert_close(state[1], source[1])


def test_cpu_state_helpers_preserve_null_slot():
    """Slot zero is padding and must never be read from or overwritten."""
    from vllm.models.glm5next.cpu import gather_initial_states_cpu, scatter_states_cpu

    state = torch.arange(4 * 2 * 2 * 2, dtype=torch.float32).reshape(4, 2, 2, 2)
    before = state.clone()
    indices = torch.tensor([0, 2], dtype=torch.int32)
    gathered = gather_initial_states_cpu(
        state,
        indices,
        torch.tensor([True, True]),
    )

    torch.testing.assert_close(gathered[0], torch.zeros_like(state[0]))
    torch.testing.assert_close(gathered[1], state[2])

    source = torch.full((2, 2, 2, 2), -1.0)
    scatter_states_cpu(state, source, indices)
    torch.testing.assert_close(state[0], before[0], rtol=0, atol=0)
    torch.testing.assert_close(state[2], source[1], rtol=0, atol=0)


def test_cpu_backend_does_not_import_triton():
    """The CPU backend must resolve to pure PyTorch, with no Triton or CUDA.

    Checking the *imports* rather than the source text matters: the module
    docstring legitimately refers to Triton when explaining which kernel it
    mirrors, so a text scan would fail for a cosmetic reason while a missing
    guard would pass.
    """
    import ast
    from pathlib import Path

    from vllm.models.glm5next.cpu import kda as cpu_kda

    src = Path(cpu_kda.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)

    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    for name in imported:
        assert "triton" not in name.lower(), f"unexpected import: {name}"
        assert "cuda" not in name.lower(), f"unexpected import: {name}"
    # The module must at minimum import torch and nothing device-specific.
    assert "torch" in imported


@pytest.mark.skipif(
    not current_platform.is_cpu(), reason="the common-layer dispatch is CPU-only"
)
def test_common_layer_uses_cpu_kda_and_state_helpers():
    """CPU dispatch covers the KDA kernels and prefill state movement."""
    import vllm.models.glm5next.common.kda as common_kda

    assert common_kda.chunk_kda_with_fused_gate.__module__ == (
        "vllm.models.glm5next.cpu.kda"
    )
    assert common_kda.fused_recurrent_kda.__module__ == ("vllm.models.glm5next.cpu.kda")
    assert common_kda.gather_initial_states.__module__ == (
        "vllm.models.glm5next.cpu.kda"
    )
    assert common_kda.scatter_states.__module__ == "vllm.models.glm5next.cpu.kda"


@pytest.mark.skipif(
    not current_platform.is_cpu(), reason="the CPU convolution path is CPU-only"
)
def test_cpu_speculative_conv_matches_rolling_reference():
    """The CPU speculative conv mirrors the CUDA rolling-window contract."""
    from vllm.models.glm5next.cpu import causal_conv1d_update_cpu

    torch.manual_seed(41)
    width, num_spec, dim = 4, 3, 5
    state_len = width - 1 + num_spec
    x = torch.randn(5, dim)
    weight = torch.randn(dim, width)
    bias = torch.randn(dim)
    state = torch.randn(2, dim, state_len)
    before = state.clone()
    query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int32)
    slots = torch.tensor([1, 0], dtype=torch.int32)
    accepted = torch.tensor([2, 1], dtype=torch.int32)

    output = causal_conv1d_update_cpu(
        x,
        state,
        weight,
        bias,
        activation="silu",
        conv_state_indices=slots,
        num_accepted_tokens=accepted,
        query_start_loc=query_start_loc,
        max_query_len=num_spec + 1,
    )
    expected = torch.zeros_like(x)
    expected_state = before.clone()
    for seq, (bos, eos) in enumerate(((0, 3), (3, 5))):
        slot = int(slots[seq].item())
        if slot == NULL_BLOCK_ID:
            continue
        offset = int(accepted[seq].item()) - 1
        local_state_len = state_len - (num_spec + 1 - (eos - bos))
        window = before[slot, :, offset : offset + width - 1].clone()
        for token in range(bos, eos):
            value = x[token]
            y = (window * weight[:, :-1]).sum(-1) + value * weight[:, -1] + bias
            expected[token] = torch.nn.functional.silu(y)
            window = torch.cat((window[:, 1:], value[:, None]), dim=-1)
        for idx in range(local_state_len):
            if idx + eos - bos < local_state_len:
                expected_state[slot, :, idx] = before[slot, :, offset + idx + 1]
            else:
                expected_state[slot, :, idx] = x[
                    bos + idx - (local_state_len - (eos - bos))
                ]
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(state, expected_state)


@pytest.mark.skipif(
    not current_platform.is_cpu(), reason="the common-layer dispatch is CPU-only"
)
def test_common_layer_packed_prefill_updates_cpu_state(monkeypatch):
    """Packed prefill crosses conv, KDA, gather, and scatter CPU boundaries."""
    from types import SimpleNamespace

    import vllm.models.glm5next.common.kda as common_kda
    from vllm.models.glm5next.common.kda import Glm5NextLinearAttention
    from vllm.models.glm5next.cpu import kda as cpu_kda
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    native_calls = 0
    native_recurrent = cpu_kda._native_recurrent_kda

    def count_native_calls(*args, **kwargs):
        nonlocal native_calls
        native_calls += 1
        return native_recurrent(*args, **kwargs)

    monkeypatch.setattr(cpu_kda, "_native_recurrent_kda", count_native_calls)
    monkeypatch.setattr(common_kda, "_cast_sigmoid", lambda x: x.float().sigmoid())

    layer = object.__new__(Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = "layer"
    layer.kda_safe_gate = True
    layer.kda_lower_bound = LOWER_BOUND
    layer.kda_prefill_backend = "cpu"
    layer._conv_state_dim_first = True
    layer.local_projection_size = H * D
    layer.local_num_heads = H
    layer.head_dim = D
    layer.A_log = torch.nn.Parameter(torch.zeros(1, 1, H, 1))
    layer.dt_bias = torch.nn.Parameter(torch.zeros(H * D))
    layer._merged_conv_weight = torch.randn(3 * H * D, 2)
    layer.q_conv1d = SimpleNamespace(bias=torch.randn(3 * H * D))
    conv_state = torch.zeros(5, 3 * H * D, 1)
    recurrent_state = torch.randn(5, H, D, D)
    before = recurrent_state.clone()
    layer.kv_cache = [conv_state, recurrent_state]

    metadata = GDNAttentionMetadata(
        num_prefills=2,
        num_prefill_tokens=5,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=5,
        has_initial_state=torch.tensor([False, True]),
        non_spec_query_start_loc=torch.tensor([0, 3, 5], dtype=torch.int32),
        non_spec_state_indices_tensor=torch.tensor([2, 3], dtype=torch.int32),
    )
    monkeypatch.setattr(
        common_kda,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"layer": metadata}),
    )

    qkv = torch.randn(5, 3 * H * D)
    raw_gate = torch.randn(1, 5, H, D)
    beta = torch.randn(1, 5, H)
    output = torch.empty(1, 5, H, D)
    layer._forward(qkv, raw_gate, beta, output)

    assert torch.isfinite(output).all()
    assert not torch.equal(recurrent_state[2], before[2])
    assert not torch.equal(recurrent_state[3], before[3])
    torch.testing.assert_close(recurrent_state[NULL_BLOCK_ID], before[NULL_BLOCK_ID])
    if hasattr(torch.ops._C, "glm5next_kda_recurrent"):
        assert native_calls == 2


@pytest.mark.skipif(
    not current_platform.is_cpu(), reason="the common-layer dispatch is CPU-only"
)
def test_common_layer_prefill_state_feeds_decode(monkeypatch):
    """The cache produced by layer prefill is consumed by ordinary decode."""
    from types import SimpleNamespace

    import vllm.models.glm5next.common.kda as common_kda
    from vllm.models.glm5next.common.kda import Glm5NextLinearAttention
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    monkeypatch.setattr(common_kda, "_cast_sigmoid", lambda x: x.float().sigmoid())
    layer = object.__new__(Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = "layer"
    layer.kda_safe_gate = True
    layer.kda_lower_bound = LOWER_BOUND
    layer.kda_prefill_backend = "cpu"
    layer._conv_state_dim_first = True
    layer.local_projection_size = H * D
    layer.local_num_heads = H
    layer.head_dim = D
    layer.A_log = torch.nn.Parameter(torch.zeros(1, 1, H, 1))
    layer.dt_bias = torch.nn.Parameter(torch.zeros(H * D))
    layer._merged_conv_weight = torch.randn(3 * H * D, 2)
    layer.q_conv1d = SimpleNamespace(bias=torch.randn(3 * H * D))
    layer.kv_cache = [
        torch.zeros(4, 3 * H * D, 1),
        torch.zeros(4, H, D, D),
    ]

    metadata = GDNAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=3,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=3,
        has_initial_state=torch.tensor([False]),
        non_spec_query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        non_spec_state_indices_tensor=torch.tensor([2], dtype=torch.int32),
    )
    monkeypatch.setattr(
        common_kda,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"layer": metadata}),
    )
    layer._forward(
        torch.randn(3, 3 * H * D),
        torch.randn(1, 3, H, D),
        torch.randn(1, 3, H),
        torch.empty(1, 3, H, D),
    )
    prefill_state = layer.kv_cache[1][2].clone()

    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=1,
        num_decode_tokens=1,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=1,
        non_spec_query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        non_spec_state_indices_tensor=torch.tensor([2], dtype=torch.int32),
    )
    decode_output = torch.empty(1, 1, H, D)
    layer._forward(
        torch.randn(1, 3 * H * D),
        torch.randn(1, 1, H, D),
        torch.randn(1, 1, H),
        decode_output,
    )

    assert torch.isfinite(decode_output).all()
    assert not torch.equal(layer.kv_cache[1][2], prefill_state)


@pytest.mark.skipif(
    not current_platform.is_cpu(), reason="the common-layer dispatch is CPU-only"
)
def test_common_layer_speculative_decode_uses_cpu_conv_and_kda(monkeypatch):
    """Speculative layer execution uses the CPU rolling conv and KDA routes."""
    from types import SimpleNamespace

    import vllm.models.glm5next.common.kda as common_kda
    from vllm.models.glm5next.common.kda import Glm5NextLinearAttention
    from vllm.models.glm5next.cpu import kda as cpu_kda
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    monkeypatch.setattr(common_kda, "_cast_sigmoid", lambda x: x.float().sigmoid())
    native_calls = 0
    native_recurrent = cpu_kda._native_recurrent_kda

    def count_native_calls(*args, **kwargs):
        nonlocal native_calls
        native_calls += 1
        return native_recurrent(*args, **kwargs)

    monkeypatch.setattr(cpu_kda, "_native_recurrent_kda", count_native_calls)
    layer = object.__new__(Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = "layer"
    layer.kda_safe_gate = True
    layer.kda_lower_bound = LOWER_BOUND
    layer.kda_prefill_backend = "cpu"
    layer._conv_state_dim_first = True
    layer.local_projection_size = H * D
    layer.local_num_heads = H
    layer.head_dim = D
    layer.A_log = torch.nn.Parameter(torch.zeros(1, 1, H, 1))
    layer.dt_bias = torch.nn.Parameter(torch.zeros(H * D))
    layer._merged_conv_weight = torch.randn(3 * H * D, 4)
    layer.q_conv1d = SimpleNamespace(bias=torch.randn(3 * H * D))
    conv_state = torch.randn(8, 3 * H * D, 6)
    recurrent_state = torch.randn(8, H, D, D)
    before_conv = conv_state.clone()
    before_recurrent = recurrent_state.clone()
    layer.kv_cache = [conv_state, recurrent_state]

    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=1,
        num_spec_decode_tokens=3,
        num_actual_tokens=3,
        spec_query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        spec_state_indices_tensor=torch.tensor([[1, 2, 3]], dtype=torch.int32),
        spec_sequence_masks=torch.tensor([True]),
        num_accepted_tokens=torch.tensor([2], dtype=torch.int32),
    )
    monkeypatch.setattr(
        common_kda,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"layer": metadata}),
    )

    output = torch.empty(1, 3, H, D)
    layer._forward(
        torch.randn(3, 3 * H * D),
        torch.randn(1, 3, H, D),
        torch.randn(1, 3, H),
        output,
    )

    assert torch.isfinite(output).all()
    assert native_calls == 3
    assert not torch.equal(conv_state[1], before_conv[1])
    for slot in (1, 2, 3):
        assert not torch.equal(recurrent_state[slot], before_recurrent[slot])


@pytest.mark.skipif(
    not current_platform.is_cpu(), reason="the rebinding is CPU-platform only"
)
def test_causal_conv_is_rebound_to_the_cpu_implementation():
    """On CPU, ``causal_conv1d`` must already resolve to the CPU functions.

    ``ops/causal_conv1d.py`` rebinds the names at import time when the platform
    is CPU, so GLM5Next's ordinary prefill/decode convolution needs no change.
    This pins that assumption down: if the rebinding ever moves or is dropped,
    the GLM KDA common layer would silently call the Triton path on CPU.
    """
    from vllm.model_executor.layers.mamba.ops import causal_conv1d as cc
    from vllm.model_executor.layers.mamba.ops.cpu.causal_conv1d import (
        causal_conv1d_fn_cpu,
        causal_conv1d_update_cpu,
    )

    assert cc.causal_conv1d_fn is causal_conv1d_fn_cpu
    assert cc.causal_conv1d_update is causal_conv1d_update_cpu


def test_cpu_conv_update_ignores_num_accepted_tokens():
    """The CPU conv wrapper accepts ``**kwargs`` and drops
    ``num_accepted_tokens``.

    This is why speculative convolution is a separate increment rather than
    something the KDA layer may assume works: the call succeeds but does not
    implement speculative semantics. Asserting the drop keeps a future change
    from silently making the wrapper look speculative-capable.

    Requires the real conv ops, so it is CPU-platform gated like the binding
    test above; ``pytestmark``-style local arithmetic runs skip it.
    """
    if not current_platform.is_cpu():
        pytest.skip("the CPU conv ops are only importable on the CPU platform")

    import inspect

    from vllm.model_executor.layers.mamba.ops.cpu.causal_conv1d import (
        causal_conv1d_update_cpu,
    )

    params = inspect.signature(causal_conv1d_update_cpu).parameters
    assert "num_accepted_tokens" not in params
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()), (
        "the wrapper is expected to swallow extra kwargs"
    )


# ---------------------------------------------------------------------------
# recurrent entry point: decode-shaped, one token per sequence
# ---------------------------------------------------------------------------


def test_recurrent_rejects_multi_token_segments():
    """The recurrent entry point is a decode kernel: one token per sequence.

    ``fused_recurrent_kda`` mirrors the NVIDIA ``fused_recurrent_kda`` kernel,
    which computes ``bos, eos`` from ``cu_seqlens`` and strides through tokens
    one at a time. A 1-D ``ssm_state_indices`` therefore addresses exactly one
    state slot per sequence, and a multi-token segment cannot be expressed.
    Rejecting it (rather than silently consuming only the first token) keeps
    the packed-prefill concept on ``chunk_kda_with_fused_gate``.
    """
    d = _make_inputs([3], seed=21)

    with pytest.raises(ValueError, match="one token per sequence"):
        fused_recurrent_kda(
            q=d["q"],
            k=d["k"],
            v=d["v"],
            g=d["raw_g"],
            beta=d["beta"],
            initial_state=d["state"][:2].clone(),
            cu_seqlens=torch.tensor([0, 3], dtype=torch.int32),
            ssm_state_indices=torch.tensor([1], dtype=torch.int32),
            sigmoid_beta=True,
            a_log=d["a_log"],
            g_bias=d["g_bias"],
            compute_gate=True,
            lower_bound=LOWER_BOUND,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_recurrent_matches_reference(dtype: torch.dtype):
    """The AVX-512 dense-state kernel matches the independent recurrence."""
    if not hasattr(torch.ops._C, "glm5next_kda_recurrent"):
        pytest.skip("GLM5Next KDA AVX-512 operator is not built")

    d = _make_inputs([4], seed=31)
    q = d["q"].to(dtype)
    k = d["k"].to(dtype)
    v = d["v"].to(dtype)
    raw_g = d["raw_g"].to(dtype)
    beta = d["beta"].to(dtype)
    initial_state = d["state"][1].clone()
    expected_out, expected_state = _reference_kda(
        q[0],
        k[0],
        v[0],
        raw_g[0],
        beta[0],
        d["a_log"],
        d["g_bias"].reshape(H, D),
        initial_state,
        sigmoid_beta=True,
    )

    actual_out, actual_state = torch.ops._C.glm5next_kda_recurrent(
        q,
        k,
        v,
        raw_g,
        beta,
        initial_state,
        D**-0.5,
        True,
        d["a_log"],
        d["g_bias"],
        True,
        LOWER_BOUND,
    )

    tolerance = 2e-2 if dtype is torch.bfloat16 else 2e-5
    torch.testing.assert_close(
        actual_out.float(), expected_out.unsqueeze(0), rtol=tolerance, atol=tolerance
    )
    torch.testing.assert_close(
        actual_state.float(), expected_state, rtol=2e-5, atol=2e-5
    )
