# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred RecoverSSM: the fused replay verify (KdaFused) against the stock eager
pipeline (Triton RecoverSSM commit after sampling, then the stock RecoverSSM verify)
and an FP64 reference of the recurrence.

KdaFused replays each request's pending accepted tokens from its request-indexed
records into the checkpoint and verifies the step in chunked form. Expected: the
checkpoint equals the stock commit's to FP32 rounding, the conv window and the key /
gate records exactly, the outputs to BF16 rounding, the corrections to FP32
rounding; the error against FP64 is on par with stock's."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(100)
):
    pytest.skip("The specialized decode needs SM10x", allow_module_level=True)
pytest.importorskip("cutlass")

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (  # noqa: E402
    causal_conv1d_update,
)
from vllm.models.kimi_k3.nvidia.ops.recoverssm import (  # noqa: E402
    KDARecoverSSMCommitContext,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import (  # noqa: E402
    prepared,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda.prepared import (  # noqa: E402
    PreparedKda,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.recoverssm.context import (  # noqa: E402
    KDADeferredCommitContext,
    KDARecoverSSMDeferred,
    allocate_records,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.recoverssm.kernels import (  # noqa: E402
    kda_recoverssm_verify,
)
from vllm.third_party.flash_linear_attention.ops.kda import (  # noqa: E402
    rms_norm_gated,
)

D, LOWER_BOUND, EPS = 128, -5.0, 1e-5
DEVICE = torch.device("cuda")


def _pitch(heads):
    return -(-(4 * heads * D + D + heads) // 16) * 16


def _layer(heads, seed, records, caches):
    dim = heads * D
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, device=DEVICE, generator=g)
    return SimpleNamespace(
        conv1d=SimpleNamespace(weight=randn(3 * dim, 1, 4) * 0.5, bias=None),
        dt_bias=randn(dim) * 0.5,
        A_log=randn(heads) * 0.5,
        f_b_proj=SimpleNamespace(
            weight=(randn(dim, D) * 3).round().clamp(-8, 8).bfloat16() / 64
        ),
        o_norm=SimpleNamespace(
            weight=(torch.rand(D, device=DEVICE, generator=g) + 0.5).bfloat16(),
            eps=EPS,
        ),
        gate_lower_bound=LOWER_BOUND,
        recoverssm_records=records,
        recoverssm_backend="triton",
        kv_cache=caches,
        local_num_heads=heads,
        head_dim=D,
    )


def _step_inputs(heads, seed, rows):
    dim = heads * D
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    projected = torch.randn(rows, _pitch(heads), device=DEVICE, generator=g)
    projected = projected.bfloat16()
    f_a = (torch.randn(rows, D, device=DEVICE, generator=g) * 1.5).round().clamp(-3, 3)
    projected[:, 4 * dim : 4 * dim + D] = f_a
    return projected


def _stock_verify(layer, heads, projected, caches, table, q, record_indices=None):
    """The stock RecoverSSM verify (Triton conv, F_B GEMM, Triton verify, gated
    RMSNorm); returns (output, q, k, v, raw gate, beta) of the step."""
    dim, n = heads * D, projected.shape[0]
    conv, state = caches
    weight = layer.conv1d.weight.view(3 * dim, 4)
    g2 = projected[:, 3 * dim : 4 * dim].view(1, n, heads, D)
    f_a = projected[:, 4 * dim : 4 * dim + D]
    beta = projected[:, 4 * dim + D : 4 * dim + D + heads].unsqueeze(0)
    g1 = (f_a @ layer.f_b_proj.weight.t()).view(1, n, heads, D)
    starts = torch.arange(0, n + 1, q, dtype=torch.int32, device=DEVICE)
    ones = torch.ones(n // q, dtype=torch.int32, device=DEVICE)
    mixed = causal_conv1d_update(
        projected[:, : 3 * dim], conv, weight, None, activation="silu",
        conv_state_indices=table, num_accepted_tokens=ones,
        query_start_loc=starts, max_query_len=q, validate_data=False,
        out=torch.empty_like(projected[:, : 3 * dim]),
    )  # fmt: skip
    xq, xk, xv = (x.reshape(1, n, heads, D) for x in mixed.split(dim, dim=-1))
    qkv = xq, xk, xv
    core = kda_recoverssm_verify(
        *qkv, g1, beta, layer.A_log, layer.dt_bias, LOWER_BOUND, state,
        layer.recoverssm_records, starts, table, q, record_indices=record_indices,
    )  # fmt: skip
    out = rms_norm_gated(
        core, g2, layer.o_norm.weight, None, "sigmoid", eps=layer.o_norm.eps
    )
    return out.reshape(n, dim), (*qkv, g1, beta, core)


def _reference(layer, state, inputs, q, n_prev, prev):
    """FP64: the replay of `n_prev` accepted tokens of the previous step's
    operands `prev` (normalized keys, decays, corrections) and this step's verify
    from it; returns (replayed checkpoint, corrections, core outputs)."""
    qs, ks, vs, g1, beta, _ = inputs
    heads = state.shape[0]
    a = layer.A_log.double().exp()
    dt = layer.dt_bias.double().view(heads, D)
    s = state.double()
    if prev is not None:
        keys, decays, us = prev
        for t in range(n_prev):
            s = s * decays[t][:, None, :] + us[t][:, :, None] * keys[t][:, None, :]
    checkpoint = s.clone()
    us_new, outs = [], []
    for t in range(q):
        qt, kt, vt = (x[t].double() for x in (qs, ks, vs))
        kt = kt / (kt.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
        qt = qt / (qt.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt() * D**-0.5
        decay = torch.exp(
            LOWER_BOUND * torch.sigmoid(a[:, None] * (g1[t].double() + dt))
        )
        s = s * decay[:, None, :]
        u = (vt - (s * kt[:, None, :]).sum(-1)) * torch.sigmoid(beta[t].double())[
            :, None
        ]
        s = s + u[:, :, None] * kt[:, None, :]
        us_new.append(u)
        outs.append((s * qt[:, None, :]).sum(-1))
    return checkpoint, torch.stack(us_new), torch.stack(outs)


def _record_operands(layer, records, row, heads, n):
    """FP64 normalized keys, decays and corrections of record slot `row`."""
    a = layer.A_log.double().exp()
    dt = layer.dt_bias.double().view(heads, D)
    kg = records.key_gate[row].double()
    keys = kg[:, :n, :D].transpose(0, 1)
    keys = keys / (keys.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
    gates = kg[:, :n, D:].transpose(0, 1)
    decays = torch.exp(LOWER_BOUND * torch.sigmoid(a[None, :, None] * (gates + dt)))
    us = records.correction[row].double()[:, :n].transpose(0, 1)
    return keys, decays, us


@pytest.fixture
def split(request, monkeypatch):
    """Force KdaFused's CTAs per item."""
    monkeypatch.setattr(prepared, "FUSED_SPLIT", ((1 << 30, request.param),))
    prepared.executable.cache_clear()
    yield request.param
    prepared.executable.cache_clear()


@pytest.fixture(params=["ws", "fused", "flush"])
def path(request, monkeypatch):
    """Deferred steps through KdaFusedWS or KdaFused (the replay in the verify), or
    the policy's small-batch path: the deferred commit (flush), then the regular
    core."""
    items = 1 << 30 if request.param == "flush" else 0
    monkeypatch.setattr(prepared, "FUSED_ITEMS", items)
    ws_items = 0 if request.param == "ws" else 1 << 30
    monkeypatch.setattr(prepared, "FUSED_WS_ITEMS", ws_items)
    prepared.executable.cache_clear()
    yield request.param
    prepared.executable.cache_clear()


CASES = [
    # (heads, requests, q, split)
    (12, 1, 5, 8),
    (12, 3, 5, 4),
    (12, 6, 8, 2),
    (12, 20, 5, 1),
    (96, 2, 5, 2),
    (96, 3, 2, 1),
    (96, 4, 8, 4),
]


@pytest.mark.parametrize("conv_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "heads,requests,q,split", CASES, indirect=["split"],
    ids=[f"h{h}-r{r}-q{q}-s{s}" for h, r, q, s in CASES],
)  # fmt: skip
@torch.inference_mode()
def test_fused_steps_match_stock(heads, requests, q, split, path, conv_dtype):
    """Three steps of one batch whose rows map to permuted record slots, with
    pending counts 0 (first step), then random in [1, q] including 1 and q: each
    fused step against the stock eager commit + verify of the same inputs."""
    seed = heads + requests + q + split
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    slots = 1 + requests
    max_reqs = requests + 3
    dim = heads * D
    conv = torch.randn(slots, 3 * dim, q + 2, device=DEVICE, generator=gen)
    conv = conv.bfloat16().to(conv_dtype)
    state = torch.randn(slots, heads, D, D, device=DEVICE, generator=gen) * 0.1
    table = torch.arange(1, slots, dtype=torch.int32, device=DEVICE)
    # Record slots: a permutation of request-state slots (as idx_mapping gives).
    record_rows = torch.randperm(max_reqs, device=DEVICE, generator=gen)[:requests]
    record_rows = record_rows.to(torch.int32)
    pending = torch.zeros(max_reqs, dtype=torch.int32, device=DEVICE)

    ours_records = allocate_records(max_reqs, heads, q, D, torch.bfloat16, DEVICE)
    stock_records = allocate_records(requests, heads, q, D, torch.bfloat16, DEVICE)
    ours_caches = (conv.clone(), state.clone())
    stock_caches = (conv.clone(), state.clone())
    ours_layer = _layer(heads, seed, ours_records, ours_caches)
    stock_layer = _layer(heads, seed, stock_records, stock_caches)
    kda = PreparedKda(ours_layer, {(requests, q): (_pitch(heads), 0)}, True)
    kda.prepare_caches(ours_caches)
    assert kda.fused
    stock_ctx = KDARecoverSSMCommitContext.create(
        [stock_layer], spec_query_len=q, max_num_reqs=requests, backend="triton"
    )
    starts = torch.arange(0, requests * q + 1, q, dtype=torch.int32, device=DEVICE)
    accepted = torch.ones(requests, dtype=torch.int32, device=DEVICE)
    ours_ctx = KDADeferredCommitContext.create(
        [ours_layer], spec_query_len=q, max_num_reqs=max_reqs
    )
    assert ours_ctx.backend == "fp32_deferred"
    prev = None
    errors, u_errors = [], []  # (ours, stock) max relative error vs FP64
    for step in range(3):
        projected = _step_inputs(heads, seed * 7 + step, requests * q)
        if step:
            sampled = torch.randint(
                1, q + 1, (requests,), dtype=torch.int32, device=DEVICE, generator=gen
            )
            sampled[0], sampled[-1] = 1, q
            prev_ops = [
                _record_operands(ours_layer, ours_records, int(r), heads, q)
                for r in record_rows.tolist()
            ]
            # After sampling: the deferred commit (step 2: row 0 committed now).
            eager = None
            if step == 2:
                eager = torch.zeros(requests, dtype=torch.int32, device=DEVICE)
                eager[0] = 1
            ours_ctx.commit(
                sampled, table, starts, record_rows=record_rows, pending=pending,
                eager=eager,
            )  # fmt: skip
            expected = sampled.clone()
            if eager is not None:
                expected[0] = 0
            torch.testing.assert_close(pending[record_rows.long()], expected)
            stock_ctx.commit(sampled, table, starts)  # the eager stock commit
        else:
            sampled, prev_ops = None, None
        ref_states = stock_caches[1].clone() if step == 0 else prev
        deferred = KDARecoverSSMDeferred(
            record_rows, pending, None, context=ours_ctx, state_indices=table
        )
        ours = torch.empty(requests * q, dim, device=DEVICE, dtype=torch.bfloat16)
        kda.run(
            projected, ours, caches=ours_caches, slots=table.view(-1, 1),
            accepted=accepted, q=q, deferred=deferred,
        )  # fmt: skip
        stock, inputs = _stock_verify(
            stock_layer, heads, projected, stock_caches, table, q
        )
        # Checkpoints: the stock commit's to FP32 rounding.
        written = table.long()
        ours_state, stock_state = ours_caches[1][written], stock_caches[1][written]
        err = (ours_state - stock_state).abs().max() / stock_state.abs().max()
        assert err < 1e-5, (step, err.item())
        torch.testing.assert_close(
            ours_caches[0][written], stock_caches[0][written], atol=0, rtol=0
        )
        # Records: keys and gates exact, corrections to FP32 rounding.
        rows = record_rows.long()
        torch.testing.assert_close(
            ours_records.key_gate[rows], stock_records.key_gate, atol=0, rtol=0
        )
        corr, stock_corr = ours_records.correction[rows], stock_records.correction
        err = (corr - stock_corr).abs().max() / stock_corr.abs().max()
        assert err < 1e-4, (step, err.item())
        # Outputs: BF16 roundings of the same values.
        same = (ours.view(torch.int16) == stock.view(torch.int16)).float().mean()
        rms = stock.float().pow(2).mean(-1, keepdim=True).sqrt()
        out_err = ((ours.float() - stock.float()).abs() / rms).max().item()
        assert same.item() > 0.97, same.item()
        assert out_err < 6e-2, out_err
        # FP64: ours vs stock error on the corrections and checkpoints.
        for b in range(requests):
            n_prev = 0 if sampled is None else int(sampled[b])
            q_b, k_b, v_b, g_b = (
                x[0].view(requests, q, heads, D)[b] for x in inputs[:4]
            )
            beta_b = inputs[4][0].view(requests, q, heads)[b]
            base = state[1 + b] if step == 0 else None
            if step:
                base = ref_states[b]
            checkpoint, us, _ = _reference(
                ours_layer, base, (q_b, k_b, v_b, g_b, beta_b, None), q,
                n_prev, prev_ops[b] if prev_ops else None,
            )  # fmt: skip
            scale = checkpoint.abs().max()
            ours_err = (ours_caches[1][1 + b].double() - checkpoint).abs().max() / scale
            stock_err = (
                stock_caches[1][1 + b].double() - checkpoint
            ).abs().max() / scale
            assert ours_err < 4 * stock_err + 1e-6, (ours_err.item(), stock_err.item())
            errors.append((ours_err.item(), stock_err.item()))
            u_scale = us.abs().max()
            ours_u = ours_records.correction[int(record_rows[b])].double()
            ours_u = (ours_u.transpose(0, 1) - us).abs().max() / u_scale
            stock_u = stock_records.correction[b].double()
            stock_u = (stock_u.transpose(0, 1) - us).abs().max() / u_scale
            assert ours_u < 4 * stock_u + 1e-6, (ours_u.item(), stock_u.item())
            u_errors.append((ours_u.item(), stock_u.item()))
        prev = ours_caches[1][written].clone()
    worst = lambda e: tuple(f"{max(x[i] for x in e):.2e}" for i in (0, 1))
    print(f"FP64 error (ours, stock): checkpoint {worst(errors)}, u {worst(u_errors)}")


@pytest.mark.parametrize(
    "heads,requests,ctas", [(12, 20, 7), (12, 3, 36), (96, 3, 5), (12, 2, 23)]
)
def test_fused_ws_uneven_ctas(heads, requests, ctas, monkeypatch):
    """KdaFusedWS's strided item order (CTA c: items c, c + ctas, ...) with CTA
    counts that leave the CTAs unequal item counts (240 items on 7 CTAs: 35 / 34;
    24 on 23: one CTA takes two) or one item each: the steps still match stock."""
    monkeypatch.setattr(prepared, "FUSED_ITEMS", 0)
    monkeypatch.setattr(prepared, "FUSED_WS_ITEMS", 0)
    ws = prepared.KdaFusedWS
    monkeypatch.setattr(
        prepared,
        "KdaFusedWS",
        lambda *args: ws(*args[:-1], ctas),  # type: ignore[call-arg]
    )
    prepared.executable.cache_clear()
    try:
        test_fused_steps_match_stock(heads, requests, 5, 1, "ws", torch.float32)
    finally:
        prepared.executable.cache_clear()


@pytest.mark.parametrize("path", ["ws", "fused"], indirect=True)
@pytest.mark.parametrize("split", [1, 4], indirect=True)
@torch.inference_mode()
def test_padded_rows_and_graph(split, path):
    """Padded rows (slot 0): zero output, no cache, record or pending effect; the
    step replays from a CUDA graph with new pending counts and record slots."""
    heads, requests, q, pad = 12, 5, 5, 2
    seed = 11 + split
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    dim, slots, max_reqs = heads * D, 1 + requests, requests + 2
    conv = torch.randn(slots, 3 * dim, q + 2, device=DEVICE, generator=gen)
    state = torch.randn(slots, heads, D, D, device=DEVICE, generator=gen) * 0.1
    records = allocate_records(max_reqs, heads, q, D, torch.bfloat16, DEVICE)
    records.correction.normal_(generator=gen)
    records.key_gate.copy_(
        torch.randn(records.key_gate.shape, device=DEVICE, generator=gen)
    )
    table = torch.arange(1, slots, dtype=torch.int32, device=DEVICE).view(-1, 1)
    projected = _step_inputs(heads, seed, requests * q)
    accepted = torch.ones(requests, dtype=torch.int32, device=DEVICE)

    def run(table_now, rows, pending_now, use_graph=False):
        caches = (conv.clone(), state.clone())
        recs = (records.correction.clone(), records.key_gate.clone())
        layer = _layer(
            heads, seed, SimpleNamespace(correction=recs[0], key_gate=recs[1]), caches
        )
        kda = PreparedKda(layer, {(requests, q): (_pitch(heads), 0)}, True)
        kda.prepare_caches(caches)
        out = torch.empty(requests * q, dim, device=DEVICE, dtype=torch.bfloat16)
        rows_buf, pending_buf = rows.clone(), pending_now.clone()
        deferred = SimpleNamespace(record_rows=rows_buf, pending=pending_buf)
        fn = lambda: kda.run(
            projected, out, caches=caches, slots=table_now, accepted=accepted, q=q,
            deferred=deferred,
        )  # fmt: skip
        if use_graph:
            # Capture on zero pending / -1 record rows, replay with the real ones.
            rows_buf.fill_(-1)
            pending_buf.zero_()
            saved = (
                caches[0].clone(),
                caches[1].clone(),
                recs[0].clone(),
                recs[1].clone(),
            )
            fn()
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            for dst, src in zip((*caches, *recs), saved):
                dst.copy_(src)
            rows_buf.copy_(rows)
            pending_buf.copy_(pending_now)
            graph.replay()
        else:
            fn()
        torch.accelerator.synchronize()
        return out, caches, recs, pending_buf

    rows = torch.tensor([4, 0, 6, 2, 3], dtype=torch.int32, device=DEVICE)
    pending = torch.tensor([0, 3, 5, 0, 2, 0, 4], dtype=torch.int32, device=DEVICE)
    full = run(table, rows, pending)
    graphed = run(table, rows, pending, use_graph=True)
    for a, b in zip(
        (full[0], *full[1], *full[2]), (graphed[0], *graphed[1], *graphed[2])
    ):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    padded_table = table.clone()
    padded_table[pad] = 0
    out, caches, recs, pending_after = run(padded_table, rows, pending)
    view = out.view(requests, q, -1)
    assert (view[pad] == 0).all()
    keep = [r for r in range(requests) if r != pad]
    torch.testing.assert_close(
        view[keep], full[0].view(requests, q, -1)[keep], atol=0, rtol=0
    )
    own = int(table[pad])
    torch.testing.assert_close(caches[0][own], conv[own], atol=0, rtol=0)
    torch.testing.assert_close(caches[1][own], state[own], atol=0, rtol=0)
    rec = int(rows[pad])
    torch.testing.assert_close(recs[0][rec], records.correction[rec], atol=0, rtol=0)
    torch.testing.assert_close(recs[1][rec], records.key_gate[rec], atol=0, rtol=0)
    torch.testing.assert_close(pending_after, pending, atol=0, rtol=0)


@torch.inference_mode()
def test_stock_verify_record_indices():
    """The deferred RecoverSSM's Triton verify (the stock one, indexed) writes
    request b's records to slot `record_indices[b]` (deferred steps the stock path
    runs), else to row b."""
    heads, requests, q = 12, 4, 5
    dim, slots = heads * D, 1 + requests
    gen = torch.Generator(device=DEVICE).manual_seed(5)
    conv = torch.randn(slots, 3 * dim, q + 2, device=DEVICE, generator=gen)
    state = torch.randn(slots, heads, D, D, device=DEVICE, generator=gen) * 0.1
    table = torch.arange(1, slots, dtype=torch.int32, device=DEVICE)
    projected = _step_inputs(heads, 9, requests * q)
    rows = torch.tensor([5, 1, 3, 0], dtype=torch.int32, device=DEVICE)
    results = []
    for indices in (None, rows):
        records = allocate_records(6, heads, q, D, torch.bfloat16, DEVICE)
        records.correction.zero_()
        records.key_gate.zero_()
        caches = (conv.clone(), state.clone())
        layer = _layer(heads, 5, records, caches)
        out, _ = _stock_verify(layer, heads, projected, caches, table, q, indices)
        results.append((out, records))
    (out, plain), (out_indexed, indexed) = results
    torch.testing.assert_close(out_indexed, out, atol=0, rtol=0)
    for b, r in enumerate(rows.tolist()):
        torch.testing.assert_close(indexed.correction[r], plain.correction[b])
        torch.testing.assert_close(indexed.key_gate[r], plain.key_gate[b])
