# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 KDA ReplaySSM on ROCm (`--use-replayssm`).

The FlyDSL verify in ring mode plus the commit after sampling must reproduce
the speculative path the AMD layer runs without the flag on gfx942:
`causal_conv1d_update` with `num_accepted_tokens`, the Triton
`fused_recurrent_kda` writing one state slot per draft position, and the gated
RMSNorm. The page then holds one state per request instead of 1 + k.

Tolerances (`TOL`). Both paths take the same BF16 inputs but round in
different places: FlyDSL's f_b projection and SiLU, the order of the K
reductions. Measured over two steps with an FP32 state, the gated-norm outputs
differ by at most 2e-3 (one BF16 ulp) and the states by at most 8e-7; with a
BF16 state, by 1.6e-2 and 3.9e-3 (one BF16 ulp of either). Committing one step
too many moves them by 3.9 and 1.4. The fold replays FlyDSL's own recurrence
and is within 8e-8 of float64 (`FOLD_TOL`). The conv state is a copy of BF16
inputs on both paths, so it must match exactly.
"""

from math import prod
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm Kimi-K3 KDA ReplaySSM"
)

H, D, W = 12, 128, 4
C = 3 * H * D
LOWER_BOUND = -5.0
NORM_EPS = 1e-5
NUM_SPEC = 3
T = NUM_SPEC + 1
BF16 = torch.bfloat16
F32 = torch.float32

# (gated-norm output, recurrent state) per recurrent state dtype.
TOL = {
    F32: (dict(atol=1e-2, rtol=1e-2), dict(atol=1e-5, rtol=1e-5)),
    BF16: (dict(atol=3e-2, rtol=1e-2), dict(atol=2e-2, rtol=1e-2)),
}
FOLD_TOL = dict(atol=1e-6, rtol=1e-5)


def _verify():
    from vllm.models.kimi_k3.amd.ops.kda_replayssm import get_flydsl_kda_verify

    return get_flydsl_kda_verify() if current_platform.is_rocm() else None


requires_flydsl_verify = pytest.mark.skipif(
    _verify() is None,
    reason="needs AITER's flydsl_kda_verify_with_f_b (ROCm/aiter#7529) on "
    "gfx942 / gfx950",
)
requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _ring_shapes(num_spec: int = NUM_SPEC):
    from vllm.models.kimi_k3.amd.ops.kda_replayssm import (
        kda_replayssm_state_dtype,
        kda_replayssm_state_shape,
    )

    shapes = kda_replayssm_state_shape(8, 8 * H, D, W, num_spec)
    dtypes = kda_replayssm_state_dtype((BF16, F32))
    return shapes, dtypes


def _bind_pages(num_slots: int, shapes, dtypes, device, pad_bytes: int = 48):
    """KV pages bound the way the layer binds them, with a pitch that is not
    a whole number of state rows (as in the packed DSA block row)."""
    from vllm.model_executor.layers.mamba.abstract import MambaBase

    page = sum(prod(s) * d.itemsize for s, d in zip(shapes, dtypes)) + pad_bytes
    pages = torch.zeros(num_slots, page, dtype=torch.int8, device=device)
    holder = SimpleNamespace(
        get_state_shape=lambda: shapes,
        get_state_dtype=lambda: dtypes,
        state_alignment=1,
    )
    MambaBase.bind_kv_cache(holder, pages.view(num_slots, 1, 1, page))
    return pages, holder.kv_cache


def _conv_dim_major(conv_state: torch.Tensor) -> torch.Tensor:
    from vllm.models.kimi_k3.amd.ops.kda_replayssm import conv_state_dim_major

    return conv_state_dim_major(conv_state)


def _ring_layer(num_slots: int, lines: int, device, state_dtype=F32, name="l"):
    shapes, dtypes = _ring_shapes()
    dtypes = (dtypes[0], state_dtype, *dtypes[2:])
    pages, caches = _bind_pages(num_slots, shapes, dtypes, device)
    window = torch.zeros(lines, T, C, W - 1, dtype=BF16, device=device)
    return SimpleNamespace(
        prefix=name, kv_cache=caches, kda_conv_window=window, pages=pages
    )


def _fold_reference(state, k, v, g, beta):
    """Float64 replay of FlyDSL's recurrence over `n` ring steps."""
    h = state.double()
    for t in range(k.shape[0]):
        kt = k[t].double()
        kt = kt / torch.sqrt((kt * kt).sum(-1, keepdim=True) + 1e-6)
        h = h * torch.exp(g[t].double())[:, None, :]
        v_new = (v[t].double() - torch.einsum("hvk,hk->hv", h, kt)) * beta[t].double()[
            :, None
        ]
        h = h + v_new[:, :, None] * kt[:, None, :]
    return h


@requires_gpu
@pytest.mark.parametrize("state_dtype", [F32, BF16])
def test_commit_fold_matches_float64(state_dtype):
    """All layers of a group in one launch, CUDA graph padding and zero-accept
    rows, request rows that differ from spec rows, and slot 0 live."""
    from vllm.models.kimi_k3.amd.ops.kda_replayssm import KDAReplaySSMCommit

    torch.manual_seed(0)
    device = torch.device("cuda")
    num_slots, lines = 16, 8
    layers = [
        _ring_layer(num_slots, lines, device, state_dtype, f"l{i}") for i in range(3)
    ]
    for layer in layers:
        conv, state, raw_v, raw_k, g, beta = layer.kv_cache
        conv.copy_(torch.randn_like(conv, dtype=F32))
        state.copy_(0.1 * torch.randn_like(state, dtype=F32))
        raw_v.copy_(torch.randn_like(raw_v, dtype=F32))
        raw_k.copy_(torch.randn_like(raw_k, dtype=F32))
        g.uniform_(LOWER_BOUND, 0.0)
        beta.uniform_(0.0, 1.0)
        layer.kda_conv_window.copy_(torch.randn_like(layer.kda_conv_window, dtype=F32))

    # Spec row b commits batch row request_rows[b]; row 2 is padding.
    cache_indices = torch.tensor([3, 7, -1, 0, 12, 5], dtype=torch.int32)
    line_indices = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)
    request_rows = torch.tensor([2, 0, 5, 1, 3, 4], dtype=torch.int32)
    num_sampled = torch.tensor([4, 1, 3, 0, 2, 4], dtype=torch.int32)
    before = [[t.clone() for t in layer.kv_cache] for layer in layers]

    KDAReplaySSMCommit(layers)(
        cache_indices.to(device),
        line_indices.to(device),
        request_rows.to(device),
        num_sampled.to(device),
        num_rows=6,
        num_tokens=T,
    )

    for layer, old in zip(layers, before):
        conv, state = layer.kv_cache[:2]
        old_conv, old_state, raw_v, raw_k, g, beta = old
        touched = set()
        for b in range(6):
            slot, line = int(cache_indices[b]), int(line_indices[b])
            n = int(num_sampled[request_rows[b]])
            if slot < 0 or n == 0:
                continue
            touched.add(slot)
            expected = _fold_reference(
                old_state[slot],
                raw_k[slot][:, :n].transpose(0, 1),
                raw_v[slot][:, :n].transpose(0, 1),
                g[slot][:, :n].transpose(0, 1),
                beta[slot][:, :n].transpose(0, 1),
            )
            if state_dtype == F32:
                torch.testing.assert_close(state[slot].double(), expected, **FOLD_TOL)
            else:
                # Rounded once, after the last step.
                torch.testing.assert_close(
                    state[slot], expected.to(BF16), atol=1e-2, rtol=1e-2
                )
            window = layer.kda_conv_window[line, n - 1]
            assert torch.equal(_conv_dim_major(conv)[slot], window)
        for slot in set(range(num_slots)) - touched:
            assert torch.equal(state[slot], old_state[slot])
            assert torch.equal(conv[slot], old_conv[slot])


def _weights(device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def randn(*shape, scale=1.0):
        return (scale * torch.randn(*shape, generator=g)).to(device)

    return SimpleNamespace(
        f_b=randn(H * D, D, scale=D**-0.5).to(BF16),
        conv=randn(C, W, scale=0.5),
        A_log=(torch.rand(H, generator=g) * 2 - 1).to(device),
        dt_bias=randn(H * D, scale=0.5),
        norm=(1 + randn(D, scale=0.1)).to(BF16),
    )


def _inputs(num_tokens: int, device, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=g).to(device=device, dtype=BF16)

    return SimpleNamespace(
        f_a=randn(num_tokens, D),
        x=randn(num_tokens, C),
        raw_beta=randn(1, num_tokens, H),
        gate=randn(num_tokens, H, D),
    )


def _gated_norm(o, gate, weight):
    x = o.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + NORM_EPS)
    return (x * weight.float() * torch.sigmoid(gate.float())).to(BF16)


def _baseline(w, inp, conv_state, state, spec_indices, query_start_loc, accepted):
    """The AMD layer's gfx942 speculative path without `--use-replayssm`."""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )
    from vllm.models.kimi_k3.amd.ops.third_party.kda import fused_recurrent_kda

    n = inp.x.shape[0]
    lens = query_start_loc.diff()
    conv = causal_conv1d_update(
        inp.x,
        conv_state,
        w.conv,
        None,
        activation="silu",
        conv_state_indices=spec_indices[:, 0],
        num_accepted_tokens=accepted,
        query_start_loc=query_start_loc,
        max_query_len=int(lens.max()),
        validate_data=False,
        out=torch.empty_like(inp.x),
    )
    q, k, v = (t.view(1, n, H, D) for t in conv.split(H * D, dim=-1))
    o, _ = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        raw_g=F.linear(inp.f_a, w.f_b).view(1, n, H, D),
        raw_beta=inp.raw_beta,
        A_log=w.A_log,
        dt_bias=w.dt_bias,
        lower_bound=LOWER_BOUND,
        initial_state=state,
        cu_seqlens=query_start_loc,
        ssm_state_indices=spec_indices,
        num_accepted_tokens=accepted,
        uniform_sequence_length=(
            int(lens[0]) if bool((lens == lens[0]).all()) else None
        ),
    )
    return _gated_norm(o, inp.gate, w.norm)


def _ring_verify(w, inp, layer, cache_indices, line_indices, **kwargs):
    conv_state, state = layer.kv_cache[:2]
    return _verify()(
        f_a=inp.f_a,
        f_b_weight=w.f_b.view(H, D, D),
        x=inp.x,
        conv_weight=w.conv,
        conv_bias=None,
        conv_state=_conv_dim_major(conv_state),
        raw_beta=inp.raw_beta,
        A_log=w.A_log,
        dt_bias=w.dt_bias,
        lower_bound=LOWER_BOUND,
        state=state,
        cache_indices=cache_indices,
        intermediate_indices=line_indices,
        intermediate_conv_window=layer.kda_conv_window,
        output_gate=inp.gate,
        norm_weight=w.norm,
        norm_eps=NORM_EPS,
        num_tokens=T,
        replay_ring=layer.kv_cache[2:6],
        **kwargs,
    )


def _baseline_caches(num_rows: int, device, state_dtype=F32):
    """Per-position state slots: row b owns slots 1 + b*T .. b*T + T."""
    num_slots = 1 + num_rows * T
    conv = torch.zeros(num_slots, C, W - 1 + NUM_SPEC, dtype=BF16, device=device)
    state = torch.zeros(num_slots, H, D, D, dtype=state_dtype, device=device)
    spec_indices = (
        1 + torch.arange(num_rows * T, dtype=torch.int32, device=device)
    ).view(num_rows, T)
    return conv, state, spec_indices


def _seed_caches(layer, slots, conv, state, spec_indices, seed=1):
    """The same checkpoint in both layouts."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ring_conv = _conv_dim_major(layer.kv_cache[0])
    for b, slot in enumerate(slots):
        c0 = torch.randn(C, W - 1, generator=g).to(conv.device, BF16)
        s0 = (0.1 * torch.randn(H, D, D, generator=g)).to(conv.device)
        conv[spec_indices[b, 0], :, : W - 1] = c0
        state[spec_indices[b, 0]] = s0
        if slot >= 0:
            ring_conv[slot] = c0
            layer.kv_cache[1][slot] = s0


@requires_flydsl_verify
@pytest.mark.parametrize("state_dtype", [F32, BF16])
@pytest.mark.parametrize("conv_layout", ["SD", "DS"])
def test_verify_and_commit_match_spec_path(monkeypatch, conv_layout, state_dtype):
    """Two verify steps, committing a different accepted count per row in
    between (1..T and a zero-accept row), with a CUDA graph padding row."""
    from vllm.model_executor.layers.mamba import mamba_utils
    from vllm.models.kimi_k3.amd.ops.kda_replayssm import KDAReplaySSMCommit

    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", conv_layout)
    mamba_utils.get_conv_state_layout.cache_clear()
    try:
        device = torch.device("cuda")
        w = _weights(device)
        # Row 6 is padding; row 5 accepts nothing in the first commit.
        slots = [5, 0, 9, 3, 11, 7, 2, -1]
        first = [1, 2, 3, 4, 4, 0, 2, 1]
        second = [4, 3, 2, 1, 2, 3, 1, 1]
        rows = len(slots)
        out_tol, state_tol = TOL[state_dtype]
        layer = _ring_layer(16, rows, device, state_dtype)
        conv, state, spec_indices = _baseline_caches(rows, device, state_dtype)
        _seed_caches(layer, slots, conv, state, spec_indices)
        cache_indices = torch.tensor(slots, dtype=torch.int32, device=device)
        line_indices = torch.arange(rows, dtype=torch.int32, device=device)
        qsl = torch.arange(0, rows * T + 1, T, dtype=torch.int32, device=device)
        commit = KDAReplaySSMCommit([layer])
        live = [b for b in range(rows) if slots[b] >= 0]
        zero_slot = slots[5]
        zero_state = layer.kv_cache[1][zero_slot].clone()
        zero_conv = _conv_dim_major(layer.kv_cache[0])[zero_slot].clone()

        inp = _inputs(rows * T, device, seed=2)
        ones = torch.ones(rows, dtype=torch.int32, device=device)
        expected = _baseline(w, inp, conv, state, spec_indices, qsl, ones)
        actual = _ring_verify(w, inp, layer, cache_indices, line_indices)
        for b in live:
            rows_b = slice(b * T, (b + 1) * T)
            torch.testing.assert_close(
                actual[0, rows_b], expected[0, rows_b], **out_tol
            )
        assert not actual[0, 7 * T :].any()

        num_sampled = torch.tensor(first, dtype=torch.int32, device=device)
        commit(cache_indices, line_indices, None, num_sampled, rows, T)
        assert torch.equal(layer.kv_cache[1][zero_slot], zero_state)
        assert torch.equal(_conv_dim_major(layer.kv_cache[0])[zero_slot], zero_conv)

        inp = _inputs(rows * T, device, seed=3)
        accepted = num_sampled.clamp_min(1)
        expected = _baseline(w, inp, conv, state, spec_indices, qsl, accepted)
        actual = _ring_verify(w, inp, layer, cache_indices, line_indices)
        for b in live:
            if first[b] == 0:
                continue
            rows_b = slice(b * T, (b + 1) * T)
            torch.testing.assert_close(
                actual[0, rows_b], expected[0, rows_b], **out_tol
            )

        num_sampled = torch.tensor(second, dtype=torch.int32, device=device)
        commit(cache_indices, line_indices, None, num_sampled, rows, T)
        ring_conv = _conv_dim_major(layer.kv_cache[0])
        for b in live:
            if first[b] == 0:
                continue
            n, slot = second[b], slots[b]
            torch.testing.assert_close(
                layer.kv_cache[1][slot], state[spec_indices[b, n - 1]], **state_tol
            )
            window = conv[spec_indices[b, 0], :, n - 1 : n - 1 + W - 1]
            assert torch.equal(ring_conv[slot], window)
    finally:
        mamba_utils.get_conv_state_layout.cache_clear()


def _layer_stub(w, layer):
    """The attributes `KimiK3DeltaAttention._replayssm_verify` reads."""
    return SimpleNamespace(
        f_b_proj=SimpleNamespace(weight=w.f_b),
        conv1d=SimpleNamespace(weight=w.conv.view(C, 1, W)),
        local_num_heads=H,
        head_dim=D,
        conv_size=W,
        A_log=w.A_log,
        dt_bias=w.dt_bias,
        gate_lower_bound=LOWER_BOUND,
        o_norm=SimpleNamespace(weight=w.norm, eps=NORM_EPS),
        kda_conv_window=layer.kda_conv_window,
        kv_cache=layer.kv_cache,
    )


@requires_flydsl_verify
def test_mixed_ragged_batch_matches_spec_path():
    """A batch mixing prefills with spec rows shorter than T, as the layer
    runs it: gather the spec tokens dense, verify, scatter them back, then
    commit by batch row."""
    from vllm.models.kimi_k3.amd.kda import KimiK3DeltaAttention
    from vllm.models.kimi_k3.amd.kda_metadata import (
        KDAReplaySSMVerifyMetadata,
        _replayssm_dense_tokens,
    )
    from vllm.models.kimi_k3.amd.ops.kda_replayssm import KDAReplaySSMCommit

    device = torch.device("cuda")
    w = _weights(device, seed=4)
    query_lens = [4, 5, 2, 1, 3, 4]
    spec = [True, False, True, True, False, True]
    query_start_loc_cpu = torch.tensor([0, *query_lens]).cumsum(0).to(torch.int32)
    masks = torch.tensor(spec)
    spec_rows = masks.nonzero().flatten()
    rows = len(spec_rows)
    slots = [6, 1, 10, 3]
    layer = _ring_layer(12, rows, device)
    conv, state, spec_indices = _baseline_caches(rows, device)
    _seed_caches(layer, slots, conv, state, spec_indices, seed=5)

    inp = _inputs(int(query_start_loc_cpu[-1]), device, seed=6)
    spec_token_indx = torch.cat(
        [
            torch.arange(query_start_loc_cpu[r], query_start_loc_cpu[r + 1])
            for r in spec_rows
        ]
    ).to(device)
    spec_inp = SimpleNamespace(
        **{
            k: getattr(inp, k).index_select(
                1 if k == "raw_beta" else 0, spec_token_indx
            )
            for k in ("f_a", "x", "raw_beta", "gate")
        }
    )
    spec_lens = torch.tensor(query_lens)[masks]
    spec_qsl = torch.tensor([0, *spec_lens]).cumsum(0).to(torch.int32).to(device)
    expected = _baseline(
        w,
        spec_inp,
        conv,
        state,
        spec_indices,
        spec_qsl,
        torch.ones(rows, dtype=torch.int32, device=device),
    )

    gather, scatter = _replayssm_dense_tokens(query_start_loc_cpu, masks, rows, T)
    commit = KDAReplaySSMCommit([layer])
    r = KDAReplaySSMVerifyMetadata(
        num_tokens=T,
        cache_indices=torch.tensor(slots, dtype=torch.int32, device=device),
        line_indices=torch.arange(rows, dtype=torch.int32, device=device),
        num_spec_decodes=rows,
        request_rows=spec_rows.to(torch.int32).to(device),
        commit=commit,
        gather_tokens=gather.to(device),
        scatter_tokens=scatter.to(device),
    )
    core_attn_out = torch.zeros(
        1, int(query_start_loc_cpu[-1]), H, D, dtype=BF16, device=device
    )
    out = KimiK3DeltaAttention._replayssm_verify(
        _layer_stub(w, layer),
        r,
        inp.x,
        inp.f_a,
        inp.gate,
        inp.raw_beta,
        _conv_dim_major(layer.kv_cache[0]),
        layer.kv_cache[1],
        core_attn_out,
    )
    assert out is not None
    torch.testing.assert_close(out, expected[0], **TOL[F32][0])

    # Accept every token of each spec row; the batch rows index num_sampled.
    num_sampled = torch.zeros(len(query_lens), dtype=torch.int32)
    num_sampled[masks] = spec_lens.to(torch.int32)
    commit(
        r.cache_indices, r.line_indices, r.request_rows, num_sampled.to(device), rows, T
    )
    ring_conv = _conv_dim_major(layer.kv_cache[0])
    for b, slot in enumerate(slots):
        n = int(spec_lens[b])
        torch.testing.assert_close(
            layer.kv_cache[1][slot], state[spec_indices[b, n - 1]], **TOL[F32][1]
        )
        window = conv[spec_indices[b, 0], :, n - 1 : n - 1 + W - 1]
        assert torch.equal(ring_conv[slot], window)


@requires_flydsl_verify
def test_forward_ragged_spec_only_batch(monkeypatch):
    """A spec-only batch whose rows drafted different counts, through the
    layer's `_forward` with the builder's metadata (the gather path)."""
    from vllm.models.kimi_k3.amd import kda

    device = torch.device("cuda")
    w = _weights(device, seed=11)
    query_lens = [T, 1, 3]
    builder = _builder(use_replayssm=True)
    common = _common([20, 30, 40], query_lens, [False] * 3)
    md = builder.build(
        0,
        common,
        num_accepted_tokens=torch.ones(3, dtype=torch.int32, device=device),
        num_decode_draft_tokens_cpu=torch.tensor(query_lens, dtype=torch.int32) - 1,
    )
    assert md.replayssm.gather_tokens is not None
    slots = common.block_table_tensor[:, 0].tolist()
    layer = _ring_layer(8, 8, device)
    conv, state, spec_indices = _baseline_caches(3, device)
    _seed_caches(layer, slots, conv, state, spec_indices, seed=12)

    num_tokens = sum(query_lens)
    inp = _inputs(num_tokens, device, seed=13)
    qsl = torch.tensor([0, *query_lens]).cumsum(0).to(torch.int32).to(device)
    ones = torch.ones(3, dtype=torch.int32, device=device)
    expected = _baseline(w, inp, conv, state, spec_indices, qsl, ones)

    stub = _layer_stub(w, layer)
    stub.prefix = "layer.0"
    stub._replayssm_verify = kda.KimiK3DeltaAttention._replayssm_verify.__get__(stub)
    monkeypatch.setattr(
        kda,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"layer.0": md}),
    )
    core_attn_out = torch.zeros(1, num_tokens, H, D, dtype=BF16, device=device)
    kda.KimiK3DeltaAttention._forward(
        stub,
        mixed_qkv=inp.x,
        f_a=inp.f_a,
        g1=F.linear(inp.f_a, w.f_b).view(1, num_tokens, H, D),
        g2=inp.gate,
        beta=inp.raw_beta,
        core_attn_out=core_attn_out,
    )
    torch.testing.assert_close(core_attn_out, expected, **TOL[F32][0])


@requires_flydsl_verify
def test_verify_cuda_graph_replay_matches_eager():
    from vllm.models.kimi_k3.amd.ops.kda_replayssm import (
        precompile_flydsl_kda_verify,
    )

    device = torch.device("cuda")
    precompile_flydsl_kda_verify(
        _verify(),
        num_tokens=T,
        state_dtype=F32,
        norm_eps=NORM_EPS,
        lower_bound=LOWER_BOUND,
        device=device,
    )
    w = _weights(device)
    rows = 6
    layer = _ring_layer(10, rows, device)
    conv, state, spec_indices = _baseline_caches(rows, device)
    _seed_caches(layer, [4, 1, 8, 2, 6, -1], conv, state, spec_indices)
    cache_indices = torch.tensor([4, 1, 8, 2, 6, -1], dtype=torch.int32, device=device)
    line_indices = torch.arange(rows, dtype=torch.int32, device=device)
    static = _inputs(rows * T, device, seed=7)
    out = torch.empty(1, rows * T, H, D, dtype=BF16, device=device)
    written = (out, layer.kda_conv_window, *layer.kv_cache[2:6])

    def run():
        _ring_verify(w, static, layer, cache_indices, line_indices, out=out)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    fresh = _inputs(rows * T, device, seed=8)
    for name in ("f_a", "x", "raw_beta", "gate"):
        getattr(static, name).copy_(getattr(fresh, name))
    cache_indices.copy_(torch.tensor([2, 8, -1, 4, 1, 6], dtype=torch.int32))
    # A padding row leaves its window line as it was.
    for t in written:
        t.zero_()
    run()
    eager = [t.clone() for t in written]
    for t in written:
        t.zero_()
    graph.replay()
    for e, r in zip(eager, written):
        assert torch.equal(e, r)


@requires_flydsl_verify
def test_ring_slots_past_4gib():
    """The real pool spans more than 4 GiB: slot rebasing is 64-bit, and the
    results match a small pool bit for bit."""
    from vllm.models.kimi_k3.amd.ops.kda_replayssm import KDAReplaySSMCommit

    device = torch.device("cuda")
    shapes, dtypes = _ring_shapes()
    page = sum(prod(s) * d.itemsize for s, d in zip(shapes, dtypes)) + 48
    big_slots = (5 << 30) // page
    w = _weights(device)
    inp = _inputs(2 * T, device, seed=9)
    num_sampled = torch.tensor([3, 4], dtype=torch.int32, device=device)
    line_indices = torch.arange(2, dtype=torch.int32, device=device)
    results = []
    for num_slots, slots in ((4, [1, 3]), (big_slots, [big_slots - 1, big_slots - 7])):
        layer = _ring_layer(num_slots, 2, device)
        assert (layer.kv_cache[1][slots[0]].data_ptr() - layer.pages.data_ptr()) > (
            4 << 30
        ) or num_slots == 4
        conv, state, spec_indices = _baseline_caches(2, device)
        _seed_caches(layer, slots, conv, state, spec_indices, seed=10)
        cache_indices = torch.tensor(slots, dtype=torch.int32, device=device)
        out = _ring_verify(w, inp, layer, cache_indices, line_indices)
        KDAReplaySSMCommit([layer])(
            cache_indices, line_indices, None, num_sampled, 2, T
        )
        results.append([out] + [t[s].clone() for t in layer.kv_cache for s in slots])
        del layer
        torch.cuda.empty_cache()
    for small, big in zip(*results):
        assert torch.equal(small, big)


def test_dense_token_maps():
    from vllm.models.kimi_k3.amd.kda_metadata import _replayssm_dense_tokens

    # Rows: spec(4), prefill(5), spec(2), prefill(3), spec(1).
    query_start_loc = torch.tensor([0, 4, 9, 11, 14, 15], dtype=torch.int32)
    masks = torch.tensor([True, False, True, False, True])
    gather, scatter = _replayssm_dense_tokens(query_start_loc, masks, 4, T)
    assert gather.tolist() == [0, 1, 2, 3, 9, 10, 10, 10, 14, 14, 14, 14, 0, 0, 0, 0]
    assert scatter.tolist() == [0, 1, 2, 3, 4, 5, 8]


def test_page_holds_one_state_and_its_ring():
    """With the ring the KDA page needs no per-draft state slots: one block
    per request, the ring inside the page, and no two states overlapping."""
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateShapeCalculator,
    )
    from vllm.v1.kv_cache_interface import MambaSpec

    shapes, dtypes = _ring_shapes()
    assert shapes[2:] == ((H, T, D), (H, T, D), (H, T, D), (H, T))
    num_slots = 3
    pages, caches = _bind_pages(num_slots, shapes, dtypes, "cpu", pad_bytes=0)
    page = pages.shape[1]
    spans = []
    for state in caches:
        start = state.data_ptr() - pages.data_ptr()
        span = state[0].numel() * state.element_size()
        assert state.stride(0) * state.element_size() == page
        spans.append((start, start + span))
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert end <= start
    assert spans[-1][1] <= page

    baseline_shapes = MambaStateShapeCalculator.kda_state_shape(
        8, 8 * H, D, conv_kernel_size=W, num_spec=NUM_SPEC
    )
    baseline = MambaSpec(
        shapes=baseline_shapes,
        dtypes=dtypes[:2],
        block_size=-1,
        num_speculative_blocks=NUM_SPEC,
    )
    ring = MambaSpec(
        shapes=shapes, dtypes=dtypes, block_size=-1, num_speculative_blocks=0
    )
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode="none"))
    assert baseline.max_num_blocks_per_req(vllm_config, 1 << 20) == 1 + NUM_SPEC
    assert ring.max_num_blocks_per_req(vllm_config, 1 << 20) == 1
    ring_bytes = 2 * H * T * D * 2 + H * T * D * 4 + H * T * 4
    conv_column_bytes = C * 2
    assert ring.page_size_bytes == (
        baseline.page_size_bytes - NUM_SPEC * conv_column_bytes + ring_bytes
    )


def _model_config(use_replayssm: bool, num_spec: int = NUM_SPEC):
    text_config = SimpleNamespace(
        linear_attn_config={
            "num_heads": 8 * H,
            "head_dim": D,
            "short_conv_kernel_size": W,
        }
    )
    model_config = SimpleNamespace(
        dtype=BF16, hf_config=SimpleNamespace(text_config=text_config)
    )
    config = SimpleNamespace(
        model_config=model_config,
        cache_config=SimpleNamespace(
            use_replayssm=use_replayssm,
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype="auto",
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
        speculative_config=SimpleNamespace(num_speculative_tokens=num_spec),
        num_speculative_tokens=num_spec,
    )

    def with_hf_config(hf_config):
        return SimpleNamespace(
            **{
                **vars(config),
                "model_config": SimpleNamespace(dtype=BF16, hf_config=hf_config),
            }
        )

    config.with_hf_config = with_hf_config
    return config


@pytest.mark.parametrize("use_replayssm", [False, True])
def test_model_page_matches_layer_page(use_replayssm):
    """The platform sizes the packed block row from the model's hooks before
    any layer exists; they must describe the layer's page. Without the flag
    they stay the baseline's."""
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateShapeCalculator,
    )
    from vllm.models.kimi_k3.amd.model import KimiK3ForConditionalGeneration

    config = _model_config(use_replayssm)
    shapes = KimiK3ForConditionalGeneration.get_mamba_state_shape_from_config(config)
    dtypes = KimiK3ForConditionalGeneration.get_mamba_state_dtype_from_config(config)
    if use_replayssm:
        assert (shapes, dtypes) == _ring_shapes()
    else:
        assert shapes == MambaStateShapeCalculator.kda_state_shape(
            8, 8 * H, D, conv_kernel_size=W, num_spec=NUM_SPEC
        )
        assert dtypes == (BF16, F32)
    # Without speculative tokens there is no ring either.
    config = _model_config(use_replayssm, num_spec=0)
    assert (
        len(KimiK3ForConditionalGeneration.get_mamba_state_shape_from_config(config))
        == 2
    )


def test_replayssm_is_declared_on_kimi_k3_only():
    from vllm.model_executor.models.interfaces import supports_replayssm
    from vllm.models.kimi_k3.amd.linear import KimiLinearForCausalLM
    from vllm.models.kimi_k3.amd.model import KimiK3ForConditionalGeneration

    assert supports_replayssm(KimiK3ForConditionalGeneration)
    # Kimi-Linear on ROCm has no ring path and keeps refusing the flag.
    assert not supports_replayssm(KimiLinearForCausalLM)


def _check_stub(**overrides):
    stub = SimpleNamespace(
        num_spec=NUM_SPEC,
        local_num_heads=H,
        head_dim=D,
        conv_size=W,
        gate_lower_bound=LOWER_BOUND,
        conv1d=SimpleNamespace(bias=None),
        f_b_proj=SimpleNamespace(weight=torch.empty(H * D, D, dtype=BF16)),
        o_norm=SimpleNamespace(weight=torch.empty(D, dtype=BF16)),
        get_state_dtype=lambda: (BF16, F32),
    )
    for name, value in overrides.items():
        setattr(stub, name, value)
    return stub


def _check_config():
    return SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
        model_config=SimpleNamespace(dtype=BF16),
        speculative_config=SimpleNamespace(enable_adaptive_verification=False),
        additional_config={},
        kv_transfer_config=None,
    )


@pytest.mark.parametrize(
    "stub_overrides,config_overrides,verify_present,reason",
    [
        ({}, {}, False, "flydsl_kda_verify_with_f_b"),
        ({"num_spec": 8}, {}, True, "at most 7 speculative tokens"),
        ({"local_num_heads": 24}, {}, True, "(TP8)"),
        ({"gate_lower_bound": None}, {}, True, "lower bound"),
        (
            {
                "f_b_proj": SimpleNamespace(
                    weight=torch.empty(1, dtype=torch.float8_e4m3fn)
                )
            },
            {},
            True,
            "unquantized bfloat16 f_b_proj",
        ),
        (
            {},
            {"cache_config": SimpleNamespace(mamba_cache_mode="align")},
            True,
            "--no-enable-prefix-caching",
        ),
        (
            {},
            {"additional_config": {"decode_backend": "latency_specialized"}},
            True,
            "decode_backend=auto",
        ),
        (
            {},
            {"kv_transfer_config": SimpleNamespace(is_kv_transfer_instance=True)},
            True,
            "no KV connector",
        ),
    ],
)
def test_replayssm_fails_closed(
    monkeypatch, stub_overrides, config_overrides, verify_present, reason
):
    from vllm.models.kimi_k3.amd import kda

    verify = SimpleNamespace(max_tokens=8, state_dtypes=(F32, BF16))
    monkeypatch.setattr(
        kda, "get_flydsl_kda_verify", lambda: verify if verify_present else None
    )
    monkeypatch.setattr(kda.rocm_aiter_ops, "is_enabled", lambda: True)
    config = _check_config()
    for name, value in config_overrides.items():
        setattr(config, name, value)
    with pytest.raises(ValueError, match="--use-replayssm") as excinfo:
        kda.KimiK3DeltaAttention._check_replayssm_ring(
            _check_stub(**stub_overrides), config
        )
    assert reason in str(excinfo.value)
    # The supported deployment passes.
    monkeypatch.setattr(kda, "get_flydsl_kda_verify", lambda: verify)
    kda.KimiK3DeltaAttention._check_replayssm_ring(_check_stub(), _check_config())


def _builder(use_replayssm: bool, full_cuda_graph: bool = True):
    from tests.v1.attention.utils import create_vllm_config
    from vllm.config import SpeculativeConfig
    from vllm.config.compilation import CUDAGraphMode
    from vllm.models.kimi_k3.amd import kda_metadata
    from vllm.v1.kv_cache_interface import MambaSpec

    vllm_config = create_vllm_config(model_name="Qwen/Qwen3.5-0.8B", block_size=16)
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=NUM_SPEC
    )
    vllm_config.compilation_config.cudagraph_mode = (
        CUDAGraphMode.FULL_AND_PIECEWISE if full_cuda_graph else CUDAGraphMode.NONE
    )
    vllm_config.cache_config.mamba_cache_mode = "none"
    vllm_config.cache_config.use_replayssm = use_replayssm
    vllm_config.compilation_config.static_forward_context["layer.0"] = SimpleNamespace()
    return kda_metadata.KimiK3ROCmKDAMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=16,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            num_speculative_blocks=0 if use_replayssm else NUM_SPEC,
        ),
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=torch.device("cuda"),
    )


def _common(seq_lens, query_lens, is_prefilling, blocks_per_req=1):
    """Without prefix caching a KDA group's block table holds
    1 + num_speculative_blocks blocks per request; block 0 is the null block,
    never a request's."""
    from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata

    m = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        16,
        torch.device("cuda"),
    )
    num_reqs = len(seq_lens)
    block_table = torch.arange(
        1, 1 + num_reqs * blocks_per_req, dtype=torch.int32, device="cuda"
    ).view(num_reqs, blocks_per_req)
    return m.replace(
        is_prefilling=torch.tensor(is_prefilling), block_table_tensor=block_table
    )


@requires_gpu
def test_builder_spec_only_and_mixed_rows():
    from vllm.models.kimi_k3.amd.kda_metadata import KimiK3ROCmKDAMetadata

    builder = _builder(use_replayssm=True)
    # Uniform spec-only batch: the spec tokens are already dense.
    m = _common([20, 30], [T, T], [False, False])
    md = builder.build(
        0,
        m,
        num_accepted_tokens=torch.ones(2, dtype=torch.int32, device="cuda"),
        num_decode_draft_tokens_cpu=torch.tensor([NUM_SPEC] * 2, dtype=torch.int32),
    )
    assert isinstance(md, KimiK3ROCmKDAMetadata)
    assert md.spec_state_indices_tensor.shape == (2, 1)
    r = md.replayssm
    assert r is not None and r.gather_tokens is None and r.request_rows is None
    assert r.cache_indices.tolist() == m.block_table_tensor[:2, 0].tolist()
    assert r.line_indices.tolist() == [0, 1]
    # The cache indices live in a static buffer for FULL graph replay.
    assert r.cache_indices.data_ptr() == builder.replayssm_cache_indices.data_ptr()

    # A trailing FULL graph padding row: its null block becomes -1.
    m = _common([20, 30, 1], [T, T, 0], [False, False, False])
    md = builder.build(
        0,
        m,
        num_accepted_tokens=torch.ones(3, dtype=torch.int32, device="cuda"),
        num_decode_draft_tokens_cpu=torch.tensor(
            [NUM_SPEC, NUM_SPEC, -1], dtype=torch.int32
        ),
    )
    r = md.replayssm
    assert r.num_spec_decodes == 2
    assert r.cache_indices.tolist()[2:] == [-1] * (len(r.cache_indices) - 2)

    # Mixed: a prefill between two spec rows, the second one short.
    m = _common([20, 50, 30], [T, 50, 2], [False, True, False])
    md = builder.build(
        0,
        m,
        num_accepted_tokens=torch.ones(3, dtype=torch.int32, device="cuda"),
        num_decode_draft_tokens_cpu=torch.tensor([NUM_SPEC, -1, 1], dtype=torch.int32),
    )
    r = md.replayssm
    assert md.num_prefills == 1 and r.num_spec_decodes == 2
    assert r.request_rows.tolist() == [0, 2]
    assert r.cache_indices.tolist() == m.block_table_tensor[[0, 2], 0].tolist()
    assert r.gather_tokens.tolist() == [0, 1, 2, 3, 54, 55, 55, 55]
    assert r.scatter_tokens.tolist() == [0, 1, 2, 3, 4, 5]
    assert torch.equal(
        md.spec_token_indx, torch.tensor([0, 1, 2, 3, 54, 55]).to(md.spec_token_indx)
    )

    # A capture dummy wider than T is not a verify: it runs as a non-spec row.
    m = _common([20, 30], [T, T + 2], [False, False])
    md = builder.build(
        0,
        m,
        num_accepted_tokens=torch.ones(2, dtype=torch.int32, device="cuda"),
        num_decode_draft_tokens_cpu=torch.tensor(
            [NUM_SPEC, NUM_SPEC + 2], dtype=torch.int32
        ),
    )
    assert md.num_spec_decodes == 1 and md.num_prefills == 1
    assert md.replayssm.request_rows.tolist() == [0]


@requires_gpu
def test_builder_unchanged_without_flag():
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    builder = _builder(use_replayssm=False)
    m = _common([20, 30], [T, T], [False, False], blocks_per_req=T)
    md = builder.build(
        0,
        m,
        num_accepted_tokens=torch.ones(2, dtype=torch.int32, device="cuda"),
        num_decode_draft_tokens_cpu=torch.tensor([NUM_SPEC] * 2, dtype=torch.int32),
    )
    assert type(md) is GDNAttentionMetadata
    assert md.spec_state_indices_tensor.tolist() == m.block_table_tensor.tolist()


@requires_gpu
def test_commit_runs_after_sampling():
    """MRV2's RecoverSSM hook commits each KDA group once, with the step's
    sampled counts, and asks for no align postprocess."""
    from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState

    builder = _builder(use_replayssm=True)
    m = _common([20, 30], [T, T], [False, False])
    md = builder.build(
        0,
        m,
        num_accepted_tokens=torch.ones(2, dtype=torch.int32, device="cuda"),
        num_decode_draft_tokens_cpu=torch.tensor([NUM_SPEC] * 2, dtype=torch.int32),
    )
    calls = []
    md.replayssm.commit = lambda *args: calls.append(args)
    state = RecoverSSMState()
    groups = [[SimpleNamespace(layer_names=["layer.0"])]]
    state.record_step({"layer.0": md}, groups, for_capture=False)
    num_sampled = torch.tensor([2, 4], dtype=torch.int32)
    state.commit_step(
        num_sampled,
        torch.arange(2),
        state_indices=None,
        num_accepted_tokens=torch.ones(2, dtype=torch.int32, device="cuda"),
    )
    assert len(calls) == 1
    args = calls[0]
    assert args[3] is num_sampled and args[4] == 2 and args[5] == T
