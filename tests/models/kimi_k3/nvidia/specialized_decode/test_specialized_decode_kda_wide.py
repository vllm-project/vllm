# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized KDA core at every head count and batch regime against the stock
composed decode (DS conv layout): TP8 / DCP8 (12 heads) and DEP's TP1 attention
(96 heads), BF16 and FP32 conv state (`--mamba-cache-dtype float32`), plain and
RecoverSSM verify steps and q = 1 steps, from one request to the wide batches the
per-batch policy hands to the wide kernel.

Stock: Triton `causal_conv1d_update` -> `fused_recurrent_kda` (verify),
`kda_recoverssm_verify` (RecoverSSM verify) or
`fused_recurrent_kda_packed_decode` (q = 1) -> gated RMSNorm. The specialized core
runs every stage at stock precision; only FP32 reduction order differs."""

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
    KDARecoverSSMRecords,
    kda_recoverssm_verify,
)
from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (  # noqa: E402
    fused_recurrent_kda,
    fused_recurrent_kda_packed_decode,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import (  # noqa: E402
    prepared,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda.prepared import (  # noqa: E402
    PreparedKda,
)
from vllm.third_party.flash_linear_attention.ops.kda import (  # noqa: E402
    rms_norm_gated,
)

D, LOWER_BOUND, EPS = 128, -5.0, 1e-5
DEVICE = torch.device("cuda")


@pytest.fixture(params=["split", "wide", "ws"], ids=["KdaSplit", "KdaWide", "KdaWS"])
def kernel(request, monkeypatch):
    """The KDA cores at every shape: KdaSplit (cluster-split heads), KdaWide (a head
    per CTA) and KdaWS (warp-specialized, persistent), whichever the per-batch
    policy would pick. KdaStream (persistent KdaWide) is bit-identical to KdaWide
    (test_stream_matches_wide_bitwise)."""
    items = 0 if request.param == "wide" else 1 << 30
    streamed = 0 if request.param == "stream" else 1 << 30
    specialized = 0 if request.param == "ws" else 1 << 30
    kinds = ("decode", "verify", "record")
    monkeypatch.setattr(prepared, "WIDE_ITEMS", dict.fromkeys(kinds, items))
    monkeypatch.setattr(prepared, "STREAM_ITEMS", dict.fromkeys(kinds, streamed))
    monkeypatch.setattr(prepared, "WS_ITEMS", dict.fromkeys(kinds, specialized))
    monkeypatch.setattr(prepared, "STREAM_HEADS", streamed)
    monkeypatch.setattr(prepared, "WS_SMALL", specialized)
    prepared.executable.cache_clear()
    yield request.param
    prepared.executable.cache_clear()


def _pitch(heads):
    return -(-(4 * heads * D + D + heads) // 16) * 16


def _attention(heads, seed, records=None):
    dim = heads * D
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, device=DEVICE, generator=g)
    return SimpleNamespace(
        conv1d=SimpleNamespace(weight=randn(3 * dim, 1, 4) * 0.5, bias=None),
        dt_bias=randn(dim) * 0.5,
        A_log=randn(heads) * 0.5,
        # F_B weights and F_A rows on a 2^-6 / integer grid: the gate projection is
        # exact in any order, so the raw gates (BF16) agree bit for bit.
        f_b_proj=SimpleNamespace(
            weight=(randn(dim, D) * 3).round().clamp(-8, 8).bfloat16() / 64
        ),
        o_norm=SimpleNamespace(
            weight=(torch.rand(D, device=DEVICE, generator=g) + 0.5).bfloat16(),
            eps=EPS,
        ),
        gate_lower_bound=LOWER_BOUND,
        recoverssm_records=records,
    )


def _inputs(heads, seed, rows, slots, window, conv_dtype):
    dim = heads * D
    g = torch.Generator(device=DEVICE).manual_seed(seed + 1)
    randn = lambda *shape: torch.randn(*shape, device=DEVICE, generator=g)
    projected = randn(rows, _pitch(heads)).bfloat16()
    projected[:, 4 * dim : 4 * dim + D] = (randn(rows, D) * 1.5).round().clamp(-3, 3)
    # Conv taps are BF16 projection values in either cache dtype.
    conv = randn(slots, 3 * dim, window).bfloat16().to(conv_dtype)
    state = randn(slots, heads, D, D) * 0.1
    return projected, conv, state


def _stock(attention, heads, projected, conv, state, table, accepted, q):
    """`table`: [B, q] (plain verify), [B, 1] (RecoverSSM: the checkpoints) or [B]
    (q = 1)."""
    dim, n = heads * D, projected.shape[0]
    weight = attention.conv1d.weight.view(3 * dim, 4)
    mixed = projected[:, : 3 * dim]
    g2 = projected[:, 3 * dim : 4 * dim].view(1, n, heads, D)
    f_a = projected[:, 4 * dim : 4 * dim + D]
    beta = projected[:, 4 * dim + D : 4 * dim + D + heads].unsqueeze(0)
    g1 = (f_a @ attention.f_b_proj.weight.t()).view(1, n, heads, D)
    common = dict(A_log=attention.A_log, dt_bias=attention.dt_bias)
    if q == 1:
        mixed = causal_conv1d_update(
            mixed, conv, weight, None, activation="silu",
            conv_state_indices=table, validate_data=True,
            out=torch.empty_like(mixed),
        )  # fmt: skip
        core, _ = fused_recurrent_kda_packed_decode(
            mixed_qkv=mixed, raw_g=g1, raw_beta=beta, lower_bound=LOWER_BOUND,
            initial_state=state, state_indices=table, **common,
        )  # fmt: skip
    else:
        starts = torch.arange(0, n + 1, q, dtype=torch.int32, device=DEVICE)
        mixed = causal_conv1d_update(
            mixed, conv, weight, None, activation="silu",
            conv_state_indices=table[:, 0], num_accepted_tokens=accepted,
            query_start_loc=starts, max_query_len=q, validate_data=False,
            out=torch.empty_like(mixed),
        )  # fmt: skip
        xq, xk, xv = (x.reshape(1, n, heads, D) for x in mixed.split(dim, dim=-1))
        qkv = xq, xk, xv
        if attention.recoverssm_records is not None:
            core = kda_recoverssm_verify(
                *qkv, g1, beta, attention.A_log, attention.dt_bias, LOWER_BOUND,
                state, attention.recoverssm_records, starts, table[:, 0], q,
            )  # fmt: skip
        else:
            core, _ = fused_recurrent_kda(
                *qkv, raw_g=g1, raw_beta=beta, lower_bound=LOWER_BOUND,
                initial_state=state, cu_seqlens=starts, ssm_state_indices=table,
                num_accepted_tokens=accepted, **common,
            )  # fmt: skip
    out = rms_norm_gated(
        core, g2, attention.o_norm.weight, None, "sigmoid", eps=attention.o_norm.eps
    )
    return out.reshape(n, dim)


def _compare(ours, stock, states, stock_states):
    same = (ours.view(torch.int16) == stock.view(torch.int16)).float().mean().item()
    ours, stock = ours.float(), stock.float()
    rms = stock.pow(2).mean(-1, keepdim=True).sqrt()
    out_err = ((ours - stock).abs() / rms).max().item()
    state_err = (
        (states - stock_states).abs().mean() / stock_states.abs().mean()
    ).item()
    # BF16 Q/K/V and output roundings match stock: only FP32 reduction order is
    # left (state error ~1e-7; without the matched roundings ~1e-3).
    assert same > 0.99, same
    assert out_err < 6e-2, out_err  # one or two BF16 ulps of a large element
    assert state_err < 2e-5, state_err


CASES = [
    # (heads, requests, q, record)
    (12, 1, 5, False),
    (12, 48, 5, False),
    (12, 64, 5, True),
    (12, 33, 8, True),
    (96, 1, 5, True),
    (96, 4, 5, False),
    (96, 24, 5, True),
    (96, 7, 2, False),
]


@pytest.mark.parametrize("conv_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("heads,requests,q,record", CASES)
@torch.inference_mode()
def test_verify_matches_stock(kernel, heads, requests, q, record, conv_dtype):
    seed = heads + requests * 10 + q
    records = None
    if record:
        records = KDARecoverSSMRecords.allocate(
            requests, heads, q, D, torch.bfloat16, DEVICE
        )
    attention = _attention(heads, seed, records)
    columns = 1 if record else q
    slots = 1 + requests * columns
    projected, conv, state = _inputs(
        heads, seed, requests * q, slots, q + 2, conv_dtype
    )
    table = torch.arange(1, slots, dtype=torch.int32, device=DEVICE)
    table = table.view(requests, columns)
    accepted = torch.randint(
        1, q + 1, (requests,), dtype=torch.int32, device=DEVICE,
        generator=torch.Generator(device=DEVICE).manual_seed(seed),
    )  # fmt: skip
    if record:
        accepted.fill_(1)  # the commit rolled the window and the checkpoint

    caches = (conv.clone(), state.clone())
    pitch = _pitch(heads)
    kda = PreparedKda(attention, {(requests, q): (pitch, int(heads == 12))}, True)
    ours = torch.empty(requests * q, heads * D, device=DEVICE, dtype=torch.bfloat16)
    kda.prepare_caches(caches).run(
        projected, ours, caches=caches, slots=table, accepted=accepted, q=q
    )
    if not record:
        stock_conv, stock_state = conv.clone(), state.clone()
        stock = _stock(
            attention, heads, projected, stock_conv, stock_state, table, accepted, q
        )
        torch.testing.assert_close(caches[0], stock_conv, atol=0, rtol=0)
        written = table.flatten().long()
        _compare(ours, stock, caches[1][written], stock_state[written])
        return

    # RecoverSSM: the outputs and window of the plain stock verify from the
    # checkpoints (the states it writes are the commit's job), the checkpoints left,
    # and the records of the stock RecoverSSM verify (its Triton kernel normalizes
    # keys with rsqrt: corrections agree to FP32 rounding, keys and gates exactly).
    torch.testing.assert_close(caches[1], state, atol=0, rtol=0)
    assert records is not None
    ours_records = (records.correction.clone(), records.key_gate.clone())
    plain = SimpleNamespace(**{**vars(attention), "recoverssm_records": None})
    rows = torch.arange(1, 1 + requests * q, dtype=torch.int32, device=DEVICE)
    rows = rows.view(requests, q)
    plain_conv = conv.new_zeros(1 + requests * q, *conv.shape[1:])
    plain_state = state.new_zeros(1 + requests * q, *state.shape[1:])
    plain_conv[rows[:, 0].long()] = conv[table[:, 0].long()]
    plain_state[rows[:, 0].long()] = state[table[:, 0].long()]
    stock = _stock(plain, heads, projected, plain_conv, plain_state, rows, accepted, q)
    torch.testing.assert_close(
        caches[0][table[:, 0].long()], plain_conv[rows[:, 0].long()], atol=0, rtol=0
    )
    last = rows[:, -1].long()
    _compare(ours, stock, plain_state[last], plain_state[last])
    _stock(attention, heads, projected, conv.clone(), state.clone(), table, accepted, q)
    torch.testing.assert_close(ours_records[1], records.key_gate, atol=0, rtol=0)
    correction, stock_correction = ours_records[0], records.correction
    err = (correction - stock_correction).abs().max() / stock_correction.abs().max()
    assert err < 1e-4, err.item()


@pytest.mark.parametrize("conv_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("heads,batch", [(12, 1), (12, 96), (96, 1), (96, 40)])
@torch.inference_mode()
def test_decode_matches_stock(kernel, heads, batch, conv_dtype):
    """Q = 1 steps (no speculative tokens in the step)."""
    attention = _attention(heads, batch)
    slots = 1 + batch
    projected, conv, state = _inputs(heads, batch, batch, slots, 3, conv_dtype)
    rows = torch.arange(1, slots, dtype=torch.int32, device=DEVICE)
    mode = (1 if batch == 1 else 2 if batch <= 8 else 0) if heads == 12 else 0
    kda = PreparedKda(attention, {batch: (_pitch(heads), mode)}, True)
    caches = (conv.clone(), state.clone())
    ours = torch.empty(batch, heads * D, device=DEVICE, dtype=torch.bfloat16)
    kda.prepare_caches(caches).run(projected, ours, caches=caches, slots=rows)
    stock_conv, stock_state = conv.clone(), state.clone()
    stock = _stock(attention, heads, projected, stock_conv, stock_state, rows, None, 1)
    torch.testing.assert_close(caches[0], stock_conv, atol=0, rtol=0)
    _compare(ours, stock, caches[1][rows.long()], stock_state[rows.long()])


@pytest.mark.parametrize(
    "kernel", ["split", "wide", "ws", "stream"], indirect=True,
    ids=["KdaSplit", "KdaWide", "KdaWS", "KdaStream"],
)  # fmt: skip
@pytest.mark.parametrize("q,record", [(1, False), (5, True), (5, False)])
@torch.inference_mode()
def test_padded_rows(kernel, q, record):
    """CUDA graph padding rows (slot 0): zero output, no cache writes, and the other
    rows bit-identical to a run without padding."""
    heads, requests, pad = 96, 6, 3
    records = None
    if record:
        records = KDARecoverSSMRecords.allocate(
            requests, heads, q, D, torch.bfloat16, DEVICE
        )
    attention = _attention(heads, 7, records)
    columns = 1 if record or q == 1 else q
    slots = 1 + requests * columns
    projected, conv, state = _inputs(
        heads, 7, requests * q, slots, q + 2 if q > 1 else 3, torch.float32
    )
    table = torch.arange(1, slots, dtype=torch.int32, device=DEVICE)
    table = table.view(requests, columns) if q > 1 else table
    accepted = torch.ones(requests, dtype=torch.int32, device=DEVICE)
    key = (requests, q) if q > 1 else requests
    mode = int(heads == 12)
    kw = dict(accepted=accepted, q=q) if q > 1 else {}
    results = []
    for padded in (False, True):
        slots_now = table.clone()
        if padded:
            slots_now[pad] = 0
        caches = (conv.clone(), state.clone())
        kda = PreparedKda(attention, {key: (_pitch(heads), mode)}, True)
        out = torch.empty(requests * q, heads * D, device=DEVICE, dtype=torch.bfloat16)
        kda.prepare_caches(caches).run(
            projected, out, caches=caches, slots=slots_now, **kw
        )
        results.append((out, caches))
    (full, full_caches), (out, caches) = results
    rows = out.view(requests, q, -1)
    assert (rows[pad] == 0).all()
    keep = [r for r in range(requests) if r != pad]
    expected = full.view(requests, q, -1)[keep]
    torch.testing.assert_close(rows[keep], expected, atol=0, rtol=0)
    own = table[pad].flatten().long() if q > 1 else table[pad : pad + 1].long()
    torch.testing.assert_close(caches[0][own], conv[own], atol=0, rtol=0)
    torch.testing.assert_close(caches[1][own], state[own], atol=0, rtol=0)


@pytest.mark.parametrize("dim_first", [True, False], ids=["dim_first", "window_first"])
@pytest.mark.parametrize(
    "heads,requests,q,record", [(12, 40, 5, True), (96, 5, 8, False), (96, 9, 1, False)]
)
@torch.inference_mode()
def test_stream_matches_wide_bitwise(
    monkeypatch, heads, requests, q, record, dim_first
):
    """KdaStream runs KdaWide's arithmetic per item: outputs, caches and records are
    bit-identical, on both conv cache layouts (dim-first and window-first)."""
    seed = heads + requests + q
    columns = 1 if record or q == 1 else q
    slots = 1 + requests * columns
    window = q + 2 if q > 1 else 3
    projected, conv, state = _inputs(
        heads, seed, requests * q, slots, window, torch.float32
    )
    if not dim_first:  # [slot, window, channel] storage, viewed [slot, channel, window]
        conv = conv.transpose(1, 2).contiguous()
    table = torch.arange(1, slots, dtype=torch.int32, device=DEVICE)
    table = table.view(requests, columns) if q > 1 else table
    accepted = torch.randint(
        1, q + 1, (requests,), dtype=torch.int32, device=DEVICE,
        generator=torch.Generator(device=DEVICE).manual_seed(seed),
    )  # fmt: skip
    if record:
        accepted.fill_(1)
    key = (requests, q) if q > 1 else requests * q
    kw = dict(accepted=accepted, q=q) if q > 1 else {}
    kinds = ("decode", "verify", "record")
    results = []
    for arm in ("wide", "stream"):
        monkeypatch.setattr(prepared, "WIDE_ITEMS", dict.fromkeys(kinds, 0))
        streamed = 0 if arm == "stream" else 1 << 30
        monkeypatch.setattr(prepared, "STREAM_ITEMS", dict.fromkeys(kinds, streamed))
        monkeypatch.setattr(prepared, "STREAM_HEADS", streamed)
        monkeypatch.setattr(prepared, "WS_SMALL", 1 << 30)
        prepared.executable.cache_clear()
        records = None
        if record:
            records = KDARecoverSSMRecords.allocate(
                requests, heads, q, D, torch.bfloat16, DEVICE
            )
        attention = _attention(heads, seed, records)
        caches = (conv.clone(), state.clone())
        kda = PreparedKda(
            attention, {key: (_pitch(heads), int(heads == 12))}, dim_first
        )
        out = torch.empty(requests * q, heads * D, device=DEVICE, dtype=torch.bfloat16)
        kda.prepare_caches(caches).run(projected, out, caches=caches, slots=table, **kw)
        kept = (records.correction, records.key_gate) if records is not None else ()
        results.append((out, *caches, *kept))
    prepared.executable.cache_clear()
    for wide, stream in zip(*results):
        torch.testing.assert_close(stream, wide, atol=0, rtol=0)


@pytest.mark.parametrize(
    "heads,requests,q,record,expected",
    [
        (12, 1, 5, True, "KdaSplit"),
        (12, 16, 5, True, "KdaWS"),
        (12, 28, 5, True, "KdaWS"),
        (12, 32, 5, True, "KdaWide"),
        (12, 64, 5, True, "KdaStream"),
        (12, 72, 5, True, "KdaWS"),
        (96, 8, 5, True, "KdaStream"),
        (96, 9, 5, True, "KdaWS"),
        (12, 128, 5, True, "KdaWS"),
        (96, 1, 5, True, "KdaStream"),
        (96, 16, 5, True, "KdaWS"),
        (96, 128, 8, True, "KdaWS"),
        (96, 128, 5, False, "KdaWide"),
        (96, 64, 1, False, "KdaWS"),
        (12, 64, 1, False, "KdaSplit"),
    ],
)
def test_policy_picks_core(monkeypatch, heads, requests, q, record, expected):
    """The per-batch policy (no compile): which core a step shape gets."""
    picked = lambda specs, h, mode, bias, device, tokens=1, record=False, fused=False: (
        type(
            prepared.core(specs, h, mode, bias, device, tokens, record, fused)
        ).__name__
    )
    monkeypatch.setattr(prepared, "executable", picked)
    meta = dict(device="meta")
    records = None
    if record:
        records = SimpleNamespace(
            correction=torch.empty(requests, heads, q, D, **meta),
            key_gate=torch.empty(
                requests, heads, q, 2 * D, dtype=torch.bfloat16, **meta
            ),
        )
    attention = _attention(heads, 0, records)
    slots = 1 + requests * (1 if record or q == 1 else q)
    window = q + 2 if q > 1 else 3
    caches = (
        torch.empty(slots, 3 * heads * D, window, **meta),
        torch.empty(slots, heads, D, D, **meta),
    )
    key = (requests, q) if q > 1 else requests
    kda = PreparedKda(attention, {key: (_pitch(heads), int(heads == 12))}, True)
    assert kda.prepare_caches(caches).kernels[key] == expected


@pytest.mark.parametrize(
    "heads,requests,q,record", [(96, 6, 5, True), (96, 40, 1, False)]
)
@torch.inference_mode()
def test_ws_conv_layouts_bitwise(monkeypatch, heads, requests, q, record):
    """KdaWS on the window-first conv cache ([slot, window, channel]: per-channel
    cp.async taps and scalar window stores) equals KdaWS on the dim-first cache
    (tested against stock above) bit for bit: outputs, caches and records."""
    seed = heads + requests + q
    columns = 1 if record or q == 1 else q
    slots = 1 + requests * columns
    window = q + 2 if q > 1 else 3
    projected, conv, state = _inputs(
        heads, seed, requests * q, slots, window, torch.float32
    )
    table = torch.arange(1, slots, dtype=torch.int32, device=DEVICE)
    table = table.view(requests, columns) if q > 1 else table
    accepted = torch.ones(requests, dtype=torch.int32, device=DEVICE)
    key = (requests, q) if q > 1 else requests * q
    kw = dict(accepted=accepted, q=q) if q > 1 else {}
    kinds = ("decode", "verify", "record")
    monkeypatch.setattr(prepared, "WS_ITEMS", dict.fromkeys(kinds, 0))
    built, ws = [], prepared.KdaWS

    def spy(*a):
        built.append(a[5])  # conv_dim_first
        return ws(*a)

    monkeypatch.setattr(prepared, "KdaWS", spy)
    results = []
    for dim_first in (True, False):
        prepared.executable.cache_clear()
        records = None
        if record:
            records = KDARecoverSSMRecords.allocate(
                requests, heads, q, D, torch.bfloat16, DEVICE
            )
        attention = _attention(heads, seed, records)
        stored = conv.clone() if dim_first else conv.transpose(1, 2).contiguous()
        caches = (stored, state.clone())
        kda = PreparedKda(attention, {key: (_pitch(heads), 0)}, dim_first)
        built.clear()
        kda.prepare_caches(caches)
        assert built == [dim_first], built  # KdaWS on this layout
        out = torch.empty(requests * q, heads * D, device=DEVICE, dtype=torch.bfloat16)
        kda.run(projected, out, caches=caches, slots=table, **kw)
        conv_out = caches[0] if dim_first else caches[0].transpose(1, 2)
        kept = (records.correction, records.key_gate) if records is not None else ()
        results.append((out, conv_out, caches[1], *kept))
    prepared.executable.cache_clear()
    for first, window_first in zip(*results):
        torch.testing.assert_close(window_first, first, atol=0, rtol=0)
