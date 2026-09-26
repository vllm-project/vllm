# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN RecoverSSM: verify against the native fused kernel, commit against the native
per-token states, and the boundary checks (malformed metadata must raise ValueError
before any kernel reads or writes out of bounds)."""

from typing import Any, cast

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn.recoverssm_gdn import (
    GDNRecoverSSMCommitContext,
    gdn_recoverssm_verify,
)
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
H, HV, K, V, T = 16, 48, 128, 128, 4
QLENS = (4, 4, 3)


def _inputs(nb: int = 10) -> dict:
    torch.manual_seed(0)
    dev = "cuda"
    tot = sum(QLENS)
    qsl = torch.tensor([0, 4, 8, 11], dtype=torch.int32, device=dev)
    return dict(
        A_log=(torch.rand(HV, device=dev) * 2 - 1).float(),
        a=torch.randn(tot, HV, device=dev, dtype=torch.bfloat16),
        b=torch.randn(tot, HV, device=dev, dtype=torch.bfloat16),
        dt_bias=(torch.randn(HV, device=dev) * 0.5).bfloat16(),
        q=torch.randn(1, tot, H, K, device=dev, dtype=torch.bfloat16),
        k=torch.randn(1, tot, H, K, device=dev, dtype=torch.bfloat16),
        v=torch.randn(1, tot, HV, V, device=dev, dtype=torch.bfloat16),
        checkpoint_state=(torch.randn(nb, HV, V, K, device=dev) * 0.05).float(),
        replay_cache=torch.zeros(nb, HV, T, V + K + 1, device=dev),
        query_start_loc=qsl,
        state_indices=torch.tensor([2, 5, 7], dtype=torch.int32, device=dev),
        spec_query_len=T,
    )


def _verify(inp: dict, **override):
    d = {**inp, **override}
    return gdn_recoverssm_verify(
        d["A_log"],
        d["a"],
        d["b"],
        d["dt_bias"],
        d["q"],
        d["k"],
        d["v"],
        checkpoint_state=d["checkpoint_state"],
        replay_cache=d["replay_cache"],
        query_start_loc=d["query_start_loc"],
        state_indices=d["state_indices"],
        spec_query_len=d["spec_query_len"],
        out=d.get("out"),
    )


def _native(inp: dict):
    return fused_sigmoid_gating_delta_rule_update(
        A_log=inp["A_log"],
        a=inp["a"],
        b=inp["b"],
        dt_bias=inp["dt_bias"],
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        initial_state=inp["checkpoint_state"].clone(),
        inplace_final_state=False,
        cu_seqlens=inp["query_start_loc"].long(),
        ssm_state_indices=inp["state_indices"].long(),
        use_qk_l2norm_in_kernel=True,
    )


def _context(inp: dict, ckpt: torch.Tensor) -> GDNRecoverSSMCommitContext:
    nb = ckpt.shape[0]
    conv = [torch.zeros(nb, 8, 3 + T - 1, device="cuda", dtype=torch.bfloat16)]
    return GDNRecoverSSMCommitContext.from_tensors(
        conv, [ckpt], [inp["replay_cache"]], spec_query_len=T, max_num_reqs=8
    )


def test_verify_matches_native_and_keeps_the_checkpoint():
    inp = _inputs()
    before = inp["checkpoint_state"].clone()
    out = _verify(inp)
    ref, _ = _native(inp)
    torch.testing.assert_close(
        out.float(), ref.reshape(out.shape).float(), rtol=0, atol=0
    )
    assert torch.equal(inp["checkpoint_state"], before)


@pytest.mark.parametrize("accepted", [1, 2, 3, 4])
def test_commit_matches_native_per_token_state(accepted):
    inp = _inputs()
    _verify(inp)
    _, states = _native(inp)
    ckpt = inp["checkpoint_state"].clone()
    ctx = _context(inp, ckpt)
    acc = torch.tensor(
        [min(accepted, q) for q in QLENS], dtype=torch.int32, device="cuda"
    )
    ctx.commit(acc, inp["state_indices"], inp["query_start_loc"])
    for i, blk in enumerate(inp["state_indices"].tolist()):
        tok = int(inp["query_start_loc"][i]) + int(acc[i]) - 1
        torch.testing.assert_close(ckpt[blk], states[tok], rtol=1e-5, atol=1e-6)
    others = [b for b in range(ckpt.shape[0]) if b not in inp["state_indices"].tolist()]
    assert torch.equal(ckpt[others], inp["checkpoint_state"][others])


@pytest.mark.parametrize(
    "case", ["short_qsl", "capacity", "hv_lt_h", "a_log", "out", "replay", "device"]
)
def test_verify_rejects_malformed_inputs(case):
    inp = _inputs()
    over, match = {
        "short_qsl": (
            dict(query_start_loc=inp["query_start_loc"][:-1]),
            "query metadata",
        ),
        "capacity": (
            dict(
                state_indices=inp["state_indices"][:2],
                query_start_loc=inp["query_start_loc"][:3],
            ),
            "activation capacity",
        ),
        "hv_lt_h": (
            dict(
                v=inp["v"][:, :, :8].contiguous(),
                a=inp["a"][:, :8].contiguous(),
                b=inp["b"][:, :8].contiguous(),
            ),
            "positive multiple",
        ),
        "a_log": (dict(A_log=torch.zeros(HV + 1, device="cuda")), "A_log"),
        "out": (
            dict(
                out=torch.empty(1, 11, HV, V + 1, device="cuda", dtype=torch.bfloat16)
            ),
            "output shape",
        ),
        "replay": (
            dict(replay_cache=torch.zeros(10, HV, T, V + K, device="cuda")),
            "replay buffer",
        ),
        "device": (dict(A_log=inp["A_log"].cpu()), "same device"),
    }[case]
    with pytest.raises(ValueError, match=match):
        _verify(inp, **over)


def test_commit_rejects_partial_or_short_metadata():
    inp = _inputs()
    ctx = _context(inp, inp["checkpoint_state"].clone())
    acc = torch.ones(3, dtype=torch.int32, device="cuda")
    bt = torch.zeros(3, 4, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="align metadata is incomplete"):
        ctx.commit(
            acc,
            inp["state_indices"],
            inp["query_start_loc"],
            block_table=bt,
            mamba_block_size=16,
        )
    with pytest.raises(ValueError, match="commit metadata"):
        ctx.commit(acc, inp["state_indices"], inp["query_start_loc"][:-1])
    with pytest.raises(ValueError, match="request mapping"):
        ctx.commit(
            acc,
            inp["state_indices"],
            inp["query_start_loc"],
            request_indices=torch.zeros(2, dtype=torch.int32, device="cuda"),
        )


def test_context_rejects_inconsistent_layers():
    inp = _inputs()
    ck = inp["checkpoint_state"]
    conv = torch.zeros(ck.shape[0], 8, 3 + T - 1, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="at least one layer"):
        GDNRecoverSSMCommitContext.from_tensors(
            [], [], [], spec_query_len=T, max_num_reqs=8
        )
    with pytest.raises(ValueError, match="differ"):
        GDNRecoverSSMCommitContext.from_tensors(
            [conv], [ck, ck], [inp["replay_cache"]], spec_query_len=T, max_num_reqs=8
        )
    with pytest.raises(ValueError, match="block count"):
        GDNRecoverSSMCommitContext.from_tensors(
            [conv[:5]], [ck], [inp["replay_cache"]], spec_query_len=T, max_num_reqs=8
        )


# ------------------------------------------------------------------ review round 2:
# mapping, align, extreme gates
from vllm.v1.attention.backends.recoverssm_metadata import (  # noqa: E402
    recoverssm_request_indices,
    recoverssm_spec_rows,
)

# (A_log range, a scale, b offset): drive g = -exp(A_log) * softplus(a + dt_bias)
# and beta = sigmoid(b)
GATES = {
    "typical": (0.0, 1.0, 0.0),
    "g_near_0": (-12.0, 1.0, 0.0),
    "g_very_negative": (2.5, 4.0, 0.0),
    "beta_near_0": (0.0, 1.0, -12.0),
    "beta_near_1": (0.0, 1.0, 12.0),
}


def _make(qlens, nb=16, seed=0, gate="typical"):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rnd = lambda *s: torch.randn(*s, device="cuda", generator=g)  # noqa: E731
    a_log_shift, ascale, boff = GATES[gate]
    tot = sum(qlens)
    qsl = torch.tensor(
        [0, *torch.tensor(qlens).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
    )
    return dict(
        A_log=(
            torch.rand(HV, device="cuda", generator=g) * 2 - 1 + a_log_shift
        ).float(),
        a=(rnd(tot, HV) * ascale).bfloat16(),
        b=(rnd(tot, HV) + boff).bfloat16(),
        dt_bias=(rnd(HV) * 0.5).bfloat16(),
        q=rnd(1, tot, H, K).bfloat16(),
        k=rnd(1, tot, H, K).bfloat16(),
        v=rnd(1, tot, HV, V).bfloat16(),
        checkpoint_state=(rnd(nb, HV, V, K) * 0.05).float(),
        replay_cache=torch.zeros(nb, HV, T, V + K + 1, device="cuda"),
        query_start_loc=qsl,
        spec_query_len=T,
    )


def test_spec_rows_mixed_batch():
    # rows: 0 spec K=3 | 1 prefill | 2 draft-less decode | 3 spec K=1 | 4 padded
    # zero-query | 5,6 beyond num_reqs
    is_prefilling = torch.tensor([False, True, False, False, False, False, False])
    qsl = torch.tensor([0, 4, 104, 105, 107, 107], dtype=torch.int32)
    drafts_in = torch.tensor(
        [3, -1, -1, 1, -1, 2, 0], dtype=torch.int32
    )  # 5, 6: garbage padding
    drafts, rows = recoverssm_spec_rows(is_prefilling, qsl, drafts_in, num_reqs=5)
    assert rows.tolist() == [0, 2, 3]
    assert drafts.tolist() == [1, -1, 1, 1, -1, -1, -1]
    idx = recoverssm_request_indices(rows, 3, torch.device("cuda"))
    assert idx is not None and idx.tolist() == [0, 2, 3]
    assert (
        recoverssm_request_indices(torch.tensor([0, 1, 2]), 3, torch.device("cuda"))
        is None
    )
    with pytest.raises(ValueError, match="spec rows classified"):
        recoverssm_request_indices(rows, 2, torch.device("cuda"))


def test_commit_maps_noncontiguous_requests_in_align_mode():
    """Spec rows 0, 2, 3 of a 5-request batch; accepted counts, block table and
    num_computed are indexed by request row. Every spec request must commit its own
    state into its own block."""
    qlens = [4, 1, 2]  # the spec rows' windows (K=3, draft-less, K=1)
    inp = _make(qlens, nb=40)
    bs = 8
    bt = torch.arange(1, 1 + 5 * 6, dtype=torch.int32, device="cuda").reshape(
        5, 6
    )  # request row -> blocks
    nc = torch.tensor(
        [5, 0, 16, 9, 0], dtype=torch.int32, device="cuda"
    )  # per request row
    req_rows = torch.tensor([0, 2, 3], dtype=torch.int32, device="cuda")
    src = torch.stack([bt[r, int(nc[r]) // bs] for r in req_rows.tolist()]).contiguous()
    inp["state_indices"] = src
    _verify(inp)
    _, states = _native(inp)
    ckpt = inp["checkpoint_state"].clone()
    ctx = _context(inp, ckpt)
    acc = torch.tensor(
        [3, 9, 1, 2, 9], dtype=torch.int32, device="cuda"
    )  # rows 1, 4 must be ignored
    ctx.commit(
        acc,
        src,
        inp["query_start_loc"],
        request_indices=req_rows,
        block_table=bt,
        num_computed_tokens=nc,
        mamba_block_size=bs,
    )
    for i, r in enumerate(req_rows.tolist()):
        n = min(int(acc[r]), qlens[i])
        c0 = int(nc[r])
        final_blk = int(bt[r, (c0 + n) // bs])
        tok = int(inp["query_start_loc"][i]) + n - 1
        torch.testing.assert_close(ckpt[final_blk], states[tok], rtol=1e-5, atol=1e-6)


# (num_computed, accepted, block size): boundary after 1 / after 2 / ending exactly on
# it / crossing and continuing / accepted = 1 without crossing / the full window
# crossing after 3
ALIGN_CASES = [
    (3, 2, 4),
    (2, 3, 4),
    (0, 4, 4),
    (6, 4, 4),
    (1, 1, 4),
    (5, 4, 8),
    (7, 4, 8),
    (4, 4, 8),
]


@pytest.mark.parametrize("nc,accepted,bs", ALIGN_CASES)
def test_align_boundary_and_final_state_match_native(nc, accepted, bs):
    inp = _make([T], nb=12)
    bt = torch.arange(2, 8, dtype=torch.int32, device="cuda").reshape(1, 6)
    src = bt[0, nc // bs].reshape(1).contiguous()
    inp["state_indices"] = src
    _verify(inp)
    _, states = _native(inp)
    ckpt = inp["checkpoint_state"].clone()
    ctx = _context(inp, ckpt)
    ctx.commit(
        torch.tensor([accepted], dtype=torch.int32, device="cuda"),
        src,
        inp["query_start_loc"],
        block_table=bt,
        num_computed_tokens=torch.tensor([nc], dtype=torch.int32, device="cuda"),
        mamba_block_size=bs,
    )
    next_boundary = (nc // bs + 1) * bs
    final_blk = int(bt[0, (nc + accepted) // bs])
    if (
        nc + accepted >= next_boundary
    ):  # crosses: the boundary state goes into the source block
        torch.testing.assert_close(
            ckpt[int(src)], states[next_boundary - nc - 1], rtol=1e-5, atol=1e-6
        )
    torch.testing.assert_close(
        ckpt[final_blk], states[accepted - 1], rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("gate", list(GATES))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_commit_matches_native_under_extreme_gates(gate, seed):
    inp = _make(list(QLENS), nb=10, seed=seed, gate=gate)
    inp["state_indices"] = torch.tensor([2, 5, 7], dtype=torch.int32, device="cuda")
    _verify(inp)
    _, states = _native(inp)
    for accepted in (1, 2, 3, 4):
        ckpt = inp["checkpoint_state"].clone()
        ctx = _context(inp, ckpt)
        acc = torch.tensor(
            [min(accepted, q) for q in QLENS], dtype=torch.int32, device="cuda"
        )
        ctx.commit(acc, inp["state_indices"], inp["query_start_loc"])
        for i, blk in enumerate(inp["state_indices"].tolist()):
            tok = int(inp["query_start_loc"][i]) + int(acc[i]) - 1
            torch.testing.assert_close(ckpt[blk], states[tok], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_verify_on_strided_views_writes_in_place(batch):
    """The layer passes views of the packed conv output (token stride = the packed
    width) and the layer's output buffer; result and replay record must equal the
    contiguous-copy path."""
    torch.manual_seed(batch)
    H, HV, K, V, SQ, NB = 16, 48, 128, 128, 4, 6
    T = batch * SQ
    qd, vd = H * K, HV * V
    packed = torch.randn(T, 2 * qd + 2 * vd, device="cuda", dtype=torch.bfloat16)
    mixed = packed[:, : 2 * qd + vd]
    a = torch.randn(T, HV, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(T, HV, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(HV, device="cuda")
    dt_bias = torch.randn(HV, device="cuda")
    checkpoint = torch.randn(NB, HV, V, K, device="cuda") * 0.05
    qsl = torch.arange(0, T + 1, SQ, device="cuda", dtype=torch.int32)
    si = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)

    def run(q, k, v, out=None):
        replay = torch.zeros(NB, HV, SQ, V + K + 1, device="cuda")
        o = gdn_recoverssm_verify(
            A_log,
            a,
            b,
            dt_bias,
            q,
            k,
            v,
            checkpoint_state=checkpoint,
            replay_cache=replay,
            query_start_loc=qsl,
            state_indices=si,
            spec_query_len=SQ,
            out=out,
        )
        return o, replay

    q, k, v = (
        x.contiguous().view(1, T, -1, d)
        for x, d in zip(torch.split(mixed, [qd, qd, vd], dim=-1), (K, K, V))
    )
    want_out, want_replay = run(q, k, v)

    core = torch.zeros(T + 3, HV, V, device="cuda", dtype=torch.bfloat16)
    views = (
        mixed[:, :qd].view(1, T, -1, K),
        mixed[:, qd : 2 * qd].view(1, T, -1, K),
        mixed[:, 2 * qd :].view(1, T, -1, V),
    )
    got_out, got_replay = run(*views, out=core[:T].unsqueeze(0))
    assert got_out.data_ptr() == core.data_ptr()
    assert torch.equal(core[:T].unsqueeze(0), want_out)
    assert torch.equal(got_replay, want_replay)


def test_verify_on_strided_views_with_variable_query_lengths():
    """Token-strided views with ragged windows (query lengths 4, 1, 3), as MTP
    scheduling produces them."""
    torch.manual_seed(7)
    H, HV, K, V, SQ, NB = 16, 48, 128, 128, 4, 6
    qsl_list = [0, 4, 5, 8]
    T = qsl_list[-1]
    qd, vd = H * K, HV * V
    packed = torch.randn(T, 2 * qd + 2 * vd, device="cuda", dtype=torch.bfloat16)
    mixed = packed[:, : 2 * qd + vd]
    a = torch.randn(T, HV, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(T, HV, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(HV, device="cuda")
    dt_bias = torch.randn(HV, device="cuda")
    checkpoint = torch.randn(NB, HV, V, K, device="cuda") * 0.05
    qsl = torch.tensor(qsl_list, device="cuda", dtype=torch.int32)
    si = torch.tensor([1, 3, 5], device="cuda", dtype=torch.int32)

    def run(q, k, v, out=None):
        replay = torch.zeros(NB, HV, SQ, V + K + 1, device="cuda")
        o = gdn_recoverssm_verify(
            A_log,
            a,
            b,
            dt_bias,
            q,
            k,
            v,
            checkpoint_state=checkpoint,
            replay_cache=replay,
            query_start_loc=qsl,
            state_indices=si,
            spec_query_len=SQ,
            out=out,
        )
        return o, replay

    q, k, v = (
        x.contiguous().view(1, T, -1, d)
        for x, d in zip(torch.split(mixed, [qd, qd, vd], dim=-1), (K, K, V))
    )
    want_out, want_replay = run(q, k, v)
    core = torch.zeros(T + 2, HV, V, device="cuda", dtype=torch.bfloat16)
    views = (
        mixed[:, :qd].view(1, T, -1, K),
        mixed[:, qd : 2 * qd].view(1, T, -1, K),
        mixed[:, 2 * qd :].view(1, T, -1, V),
    )
    _, got_replay = run(*views, out=core[:T].unsqueeze(0))
    assert torch.equal(core[:T].unsqueeze(0), want_out)
    assert torch.equal(got_replay, want_replay)


def test_layer_verify_qkv_views_and_fallback():
    """The layer's guard: views for a [tokens, q|k|v] conv output with a contiguous
    last dimension, the contiguous-copy fallback for any other layout; same values
    either way."""
    from types import SimpleNamespace

    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention as Layer,
    )

    H, HV, K, V, T = 4, 8, 16, 16, 5
    layer = SimpleNamespace(
        key_dim=H * K, value_dim=HV * V, tp_size=1, head_k_dim=K, head_v_dim=V
    )
    layer.rearrange_mixed_qkv = lambda x: Layer.rearrange_mixed_qkv(cast(Any, layer), x)
    width = 2 * H * K + HV * V
    packed = torch.randn(T, width + HV * V, dtype=torch.bfloat16)  # q|k|v|z, like qkvz
    strided = packed[:, :width]
    q, k, v, is_view = Layer._recoverssm_verify_qkv(cast(Any, layer), strided)
    assert is_view
    assert q.data_ptr() == packed.data_ptr()
    rq, rk, rv = Layer.rearrange_mixed_qkv(cast(Any, layer), strided)
    assert torch.equal(q, rq) and torch.equal(k, rk) and torch.equal(v, rv)

    # Column-major storage: last dimension not contiguous -> copies, same values.
    col_major = strided.t().contiguous().t()
    assert col_major.stride(-1) != 1
    q2, k2, v2, is_view2 = Layer._recoverssm_verify_qkv(cast(Any, layer), col_major)
    assert not is_view2
    assert torch.equal(q2, rq) and torch.equal(k2, rk) and torch.equal(v2, rv)


# ------------------------------------------------------------- BF16 recurrent state
# RecoverSSM upcasts the BF16 checkpoint, carries the accepted window in FP32 and rounds
# once when it writes the committed state. The reference is the FP32 trajectory from the
# same (already BF16-rounded) checkpoint; the committed state must equal its BF16
# rounding up to one BF16 ulp, and be no further from it than the native path is.
BF16_ULP = 2**-7


def _bf16_case(qlens, nb=12, seed=0):
    inp = _make(qlens, nb=nb, seed=seed)
    inp["checkpoint_state"] = inp["checkpoint_state"].bfloat16()
    ref = dict(inp)
    ref["checkpoint_state"] = inp["checkpoint_state"].float()
    return inp, ref


def test_verify_bf16_state_matches_native():
    inp, _ = _bf16_case(list(QLENS))
    inp["state_indices"] = torch.tensor([2, 5, 7], dtype=torch.int32, device="cuda")
    before = inp["checkpoint_state"].clone()
    out = _verify(inp)
    ref, _ = _native(inp)
    torch.testing.assert_close(
        out.float(), ref.reshape(out.shape).float(), rtol=0, atol=0
    )
    assert torch.equal(inp["checkpoint_state"], before)


@pytest.mark.parametrize("accepted", [1, 2, 3, 4])
@pytest.mark.parametrize("seed", [0, 1])
def test_commit_bf16_state_rounds_once(accepted, seed):
    inp, ref_inp = _bf16_case(list(QLENS), seed=seed)
    idx = torch.tensor([2, 5, 7], dtype=torch.int32, device="cuda")
    inp["state_indices"] = ref_inp["state_indices"] = idx
    _verify(inp)
    _, native_states = _native(inp)
    _, ref_states = _native(ref_inp)
    ckpt = inp["checkpoint_state"].clone()
    _context(inp, ckpt).commit(
        torch.tensor(
            [min(accepted, q) for q in QLENS], dtype=torch.int32, device="cuda"
        ),
        idx,
        inp["query_start_loc"],
    )
    for i, blk in enumerate(idx.tolist()):
        tok = int(inp["query_start_loc"][i]) + min(accepted, QLENS[i]) - 1
        ref = ref_states[tok].float()
        got = ckpt[blk].float()
        torch.testing.assert_close(
            got, ref.bfloat16().float(), rtol=BF16_ULP, atol=1e-5
        )
        err_rssm = (got - ref).abs().mean()
        err_native = (native_states[tok].bfloat16().float() - ref).abs().mean()
        assert err_rssm <= err_native * 1.05 + 1e-7, (err_rssm, err_native)


@pytest.mark.parametrize("nc,accepted,bs", ALIGN_CASES)
def test_align_boundary_bf16_state(nc, accepted, bs):
    inp, ref_inp = _bf16_case([T])
    bt = torch.arange(2, 8, dtype=torch.int32, device="cuda").reshape(1, 6)
    src = bt[0, nc // bs].reshape(1).contiguous()
    inp["state_indices"] = ref_inp["state_indices"] = src
    _verify(inp)
    _, ref_states = _native(ref_inp)
    ckpt = inp["checkpoint_state"].clone()
    _context(inp, ckpt).commit(
        torch.tensor([accepted], dtype=torch.int32, device="cuda"),
        src,
        inp["query_start_loc"],
        block_table=bt,
        num_computed_tokens=torch.tensor([nc], dtype=torch.int32, device="cuda"),
        mamba_block_size=bs,
    )
    next_boundary = (nc // bs + 1) * bs
    final_blk = int(bt[0, (nc + accepted) // bs])
    if nc + accepted >= next_boundary:
        torch.testing.assert_close(
            ckpt[int(src)].float(),
            ref_states[next_boundary - nc - 1].float().bfloat16().float(),
            rtol=BF16_ULP,
            atol=1e-5,
        )
    torch.testing.assert_close(
        ckpt[final_blk].float(),
        ref_states[accepted - 1].float().bfloat16().float(),
        rtol=BF16_ULP,
        atol=1e-5,
    )
