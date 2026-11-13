# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Hopper specialized decode's KDA layer (`HopperKdaLayer`: warp split-K
in_proj, GLM `KdaSplit` with g_b_proj folded in, warp split-K o_proj) at one TP4
rank's shapes, plain decode and MTP verify, against an FP64 reference: its output
and recurrent-state errors must not exceed the stock kernels' (cuBLAS projections,
conv update, F_B and g_b GEMMs, fused recurrence, gated RMSNorm); it must write the
slots stock writes (per-token states in verify, the conv window) and nothing else,
padded requests (slot 0) included; a captured step replayed on new inputs, on
weights written in place after the capture, or on one layer object holding a graph
per capture size must match the step run eagerly; and a request at the top of an
8200-slot pool (state offsets past 2^31 elements) must give the bits it gives at the
bottom. Needs one SM90 GPU.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

HEADS, HEAD_DIM, HIDDEN = 16, 128, 4096
DIM = HEADS * HEAD_DIM
PITCH = 3 * DIM + HEADS + 2 * HEAD_DIM
BETA, F_A, G_A = 3 * DIM, 3 * DIM + HEADS, 3 * DIM + HEADS + HEAD_DIM
LOWER_BOUND, EPS = -5.0, 1e-5
WEIGHTS = ("w_in", "w_o", "f_b", "g_b", "conv_weight", "a_log", "dt_bias", "norm")

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.is_device_capability(90)),
    reason="Requires SM90",
)


def _metadata(slots, accepted):
    return SimpleNamespace(
        non_spec_state_indices_tensor=slots.view(-1),
        spec_state_indices_tensor=slots,
        num_accepted_tokens=accepted,
    )


class _Case:
    """One layer's weights and caches and a step of `requests` x `q` tokens whose
    last `padded` requests are CUDA graph padding (slot 0, one accepted token)."""

    def __init__(self, requests, q, padded=0, seed=0):
        g = torch.Generator(device="cuda").manual_seed(requests * 8 + q + 100 * seed)
        rand = lambda *shape, scale=1.0: (
            torch.randn(*shape, generator=g, device="cuda") * scale
        )
        self.requests, self.q, self.tokens = requests, q, requests * q
        self.w_in = rand(PITCH, HIDDEN, scale=0.02).bfloat16()
        self.w_o = rand(HIDDEN, DIM, scale=0.02).bfloat16()
        self.f_b = rand(DIM, HEAD_DIM, scale=HEAD_DIM**-0.5).bfloat16()
        self.g_b = rand(DIM, HEAD_DIM, scale=HEAD_DIM**-0.5).bfloat16()
        self.conv_weight = rand(3 * DIM, 1, 4, scale=0.5)
        self.a_log = (torch.rand(HEADS, generator=g, device="cuda") * 15 + 1).log()
        self.dt_bias = rand(DIM, scale=0.5)
        self.norm = (1 + rand(HEAD_DIM, scale=0.1)).bfloat16()
        slots = self.tokens + 8  # slot 0 is the null block
        state = rand(slots, HEADS, HEAD_DIM, HEAD_DIM, scale=0.05)
        conv = rand(slots, 2 + q, 3 * DIM).bfloat16()  # SD layout, width 3 + q - 1
        self.caches = (conv, state)
        order = torch.randperm(slots - 1, generator=g, device="cuda").int() + 1
        self.slots = order[: self.tokens].view(requests, q).contiguous()
        self.accepted = torch.randint(
            1, q + 1, (requests,), generator=g, device="cuda", dtype=torch.int32
        )
        if padded:
            self.slots[requests - padded :] = 0
            self.accepted[requests - padded :] = 1
        self.x = rand(self.tokens, HIDDEN).bfloat16()

    def clone_caches(self):
        return tuple(c.clone() for c in self.caches)

    def live(self):
        return self.slots[:, 0].repeat_interleave(self.q) > 0

    def metadata(self):
        return _metadata(self.slots, self.accepted)

    def stage(self, caches, batches=None):
        from vllm.models.glm5next.nvidia.specialized_decode.hopper.kda import (
            HopperKdaLayer,
        )

        weight = lambda w: SimpleNamespace(weight=w)
        attention = SimpleNamespace(
            local_num_heads=HEADS,
            in_proj_qkvbfg_a=weight(self.w_in),
            f_b_proj=weight(self.f_b),
            g_b_proj=weight(self.g_b),
            o_proj=weight(self.w_o),
            conv1d=weight(self.conv_weight),
            A_log=self.a_log.view(1, 1, HEADS, 1),
            dt_bias=self.dt_bias,
            o_norm=SimpleNamespace(weight=self.norm, eps=EPS),
            kda_lower_bound=LOWER_BOUND,
            kv_cache=caches,
        )
        stage = HopperKdaLayer(attention, batches or [self.tokens], self.q)
        stage.prepare_caches(caches)
        return stage

    def stock_core(self, caches, projected):
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
            causal_conv1d_update,
        )
        from vllm.models.glm5next.nvidia.ops.third_party.kda import (
            fused_recurrent_kda,
        )
        from vllm.third_party.flash_linear_attention.ops.kda import rms_norm_gated

        conv, state = caches
        tokens, q = self.tokens, self.q
        weight = self.conv_weight.view(3 * DIM, 4)
        qkv = projected[:, : 3 * DIM].clone()  # the conv update runs in place
        starts = torch.arange(0, tokens + 1, q, device="cuda", dtype=torch.int32)
        if q == 1:
            slots, accepted = self.slots.view(-1), None
            extra = dict(conv_state_indices=slots)
        else:
            slots, accepted = self.slots, self.accepted
            extra = dict(
                conv_state_indices=slots[:, 0],
                num_accepted_tokens=accepted,
                query_start_loc=starts,
                max_query_len=q,
            )
        qkv = causal_conv1d_update(
            qkv, conv.transpose(-1, -2), weight, None, activation="silu", **extra
        )
        shape = (1, tokens, HEADS, HEAD_DIM)
        query, key, value = (t.reshape(shape) for t in qkv.split(DIM, dim=-1))
        out = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
        fused_recurrent_kda(
            q=query,
            k=key,
            v=value,
            g=F.linear(projected[:, F_A:G_A], self.f_b).reshape(shape),
            beta=projected[:, BETA:F_A].unsqueeze(0),
            initial_state=state,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=starts,
            ssm_state_indices=slots,
            num_accepted_tokens=accepted,
            out=out,
            sigmoid_beta=True,
            a_log=self.a_log,
            g_bias=self.dt_bias,
            compute_gate=True,
            lower_bound=LOWER_BOUND,
        )
        heads = (tokens, HEADS, HEAD_DIM)
        gate = F.linear(projected[:, G_A:PITCH], self.g_b)
        return rms_norm_gated(
            out.view(heads),
            gate.view(heads),
            self.norm,
            None,
            activation="sigmoid",
            eps=EPS,
        ).view(tokens, DIM)

    def reference(self, projected):
        """FP64 core outputs, the state written per slot and the conv rows per
        slot, from `projected` (exact gates, no intermediate rounding)."""
        conv0, state0 = (c.double() for c in self.caches)
        x = projected.double()
        weight = self.conv_weight.double().view(3 * DIM, 4)
        f_b, g_b = self.f_b.double(), self.g_b.double()
        decay_rate = self.a_log.double().exp()[:, None]
        core = torch.zeros(self.tokens, DIM, dtype=torch.float64, device="cuda")
        states, convs = {}, {}
        unit = lambda t: t / ((t * t).sum(-1, keepdim=True) + 1e-6).sqrt()
        for r in range(self.requests):
            offset = int(self.accepted[r]) - 1 if self.q > 1 else 0
            slots = self.slots[r].tolist()
            if slots[offset] <= 0:
                continue
            state = state0[slots[offset]].clone()
            window = conv0[slots[0], offset : offset + 3]
            for t in range(self.q):
                n = r * self.q + t
                window = torch.cat([window, x[n, None, : 3 * DIM]])[-4:]
                y = (window.T * weight).sum(-1)
                y = y * torch.sigmoid(y)
                query, key, value = (
                    y[i * DIM : (i + 1) * DIM].view(HEADS, -1) for i in range(3)
                )
                query, key = unit(query) * HEAD_DIM**-0.5, unit(key)
                beta = torch.sigmoid(x[n, BETA:F_A])[:, None]
                raw = (x[n, F_A:G_A] @ f_b.T).view(HEADS, -1)
                gate = decay_rate * (raw + self.dt_bias.double().view(HEADS, -1))
                state = state * (LOWER_BOUND * torch.sigmoid(gate)).exp()[:, None]
                delta = value - torch.einsum("hvk,hk->hv", state, key)
                state = state + (delta * beta)[..., None] * key[:, None]
                o = torch.einsum("hvk,hk->hv", state, query)
                o = o / ((o * o).mean(-1, keepdim=True) + EPS).sqrt()
                gate_out = torch.sigmoid(x[n, G_A:PITCH] @ g_b.T)
                core[n] = (o * self.norm.double()).view(-1) * gate_out
                states[slots[t]] = state
            rows = projected[r * self.q : (r + 1) * self.q, : 3 * DIM]
            taps = self.caches[0][slots[0], offset + 1 : offset + 3]
            convs[slots[0]] = torch.cat([taps.to(rows.dtype), rows])
        return core, states, convs


def _errors(out, state, reference, live):
    ref, states = reference[:2]
    diff = (out.double() - ref)[live]
    error = diff.abs().flatten()
    rel_l2 = float(diff.norm() / ref[live].norm())
    state_error = max(
        float((state[s].double() - expected).abs().max())
        for s, expected in states.items()
    )
    return [rel_l2, float(error.quantile(0.99)), float(error.max()), state_error]


def _untouched(case, caches, reference):
    """Every state and conv slot outside the reference's writes keeps its bits;
    the written conv slots hold the exact window."""
    states, convs = reference[1:]
    for cache, written in ((1, states), (0, convs)):
        keep = torch.ones(case.caches[cache].shape[0], dtype=torch.bool, device="cuda")
        keep[list(written)] = False
        assert torch.equal(caches[cache][keep], case.caches[cache][keep])
    for slot, rows in convs.items():
        assert torch.equal(caches[0][slot, : rows.shape[0]], rows), slot


# (requests, q, padded requests): MTP5 verify at 1-5 requests (6-30 tokens),
# MTP3 at 32 tokens, plain decode at 1-32 tokens.
CASES = [
    (1, 6, 0),
    (2, 6, 0),
    (3, 6, 0),
    (4, 6, 0),
    (5, 6, 0),
    (4, 6, 2),
    (8, 4, 0),
    (1, 1, 0),
    (4, 1, 0),
    (8, 1, 3),
    (32, 1, 0),
]


@pytest.mark.parametrize(("requests", "q", "padded"), CASES)
def test_core_no_worse_than_stock(requests, q, padded):
    """The fused core on the same BF16 projections as stock."""
    case = _Case(requests, q, padded)
    ours_caches, stock_caches = case.clone_caches(), case.clone_caches()
    stage = case.stage(ours_caches)
    projected = F.linear(case.x, case.w_in)
    # Feed stock's projections and read the core output back unprojected.
    stage.in_proj = SimpleNamespace(run=lambda x, out: out.copy_(projected))
    stage.o_proj = SimpleNamespace(run=lambda core, out: out[:, :DIM].copy_(core))
    ours = stage.run(case.x, case.metadata(), q)[:, :DIM]
    stock = case.stock_core(stock_caches, projected)
    reference = case.reference(projected)
    live = case.live()
    ours_errors = _errors(ours, ours_caches[1], reference, live)
    stock_errors = _errors(stock, stock_caches[1], reference, live)
    # Output relative L2, p99 and max abs error, and the max state error.
    assert all(a <= b for a, b in zip(ours_errors, stock_errors)), (
        ours_errors,
        stock_errors,
    )
    _untouched(case, ours_caches, reference)
    assert torch.equal(ours[~live], torch.zeros_like(ours[~live]))


@pytest.mark.parametrize(("requests", "q", "padded"), CASES)
def test_layer_no_worse_than_stock(requests, q, padded):
    """The whole layer (projections included) against FP64 projections, with 10%
    slack for the GEMMs' different (equally exact) FP32 reduction orders."""
    case = _Case(requests, q, padded)
    ours_caches, stock_caches = case.clone_caches(), case.clone_caches()
    stage = case.stage(ours_caches)
    ours = stage.run(case.x, case.metadata(), q)
    stock = F.linear(
        case.stock_core(stock_caches, F.linear(case.x, case.w_in)), case.w_o
    )
    exact = case.x.double() @ case.w_in.double().T
    core, states, _ = case.reference(exact)
    reference = (core @ case.w_o.double().T, states)
    live = case.live()
    ours_errors = _errors(ours, ours_caches[1], reference, live)
    stock_errors = _errors(stock, stock_caches[1], reference, live)
    assert all(a <= 1.1 * b for a, b in zip(ours_errors, stock_errors)), (
        ours_errors,
        stock_errors,
    )
    # The conv window holds this layer's own BF16 projections.
    projected = torch.empty(case.tokens, PITCH, dtype=torch.bfloat16, device="cuda")
    stage.in_proj.run(case.x, projected)
    _untouched(case, ours_caches, case.reference(projected))


def _capture(stage, x, metadata, q):
    """Warms `stage` up outside a capture, then captures one step in a graph."""
    stage.run(x, metadata, q)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = stage.run(x, metadata, q)
    return graph, out


def _replay_matches_eager(graph, out, stage, caches, x, metadata, q):
    """Replays `graph` from the current caches and checks its output and cache
    writes against `stage` run eagerly from the same caches."""
    start = tuple(c.clone() for c in caches)
    graph.replay()
    replayed, after = out.clone(), tuple(c.clone() for c in caches)
    for cache, saved in zip(caches, start):
        cache.copy_(saved)
    expected = stage.run(x, metadata, q)
    assert torch.equal(replayed, expected)
    assert all(torch.equal(a, b) for a, b in zip(after, caches))


@pytest.mark.parametrize(("requests", "q"), [(1, 6), (4, 6), (2, 1)])
def test_cuda_graph_replay(requests, q):
    """A captured step replayed on new hidden states, slots and accepted counts
    (written into the captured buffers) matches the same step run eagerly."""
    case = _Case(requests, q)
    caches = case.clone_caches()
    stage = case.stage(caches)
    x, metadata = case.x.clone(), case.metadata()
    graph, out = _capture(stage, x, metadata, q)
    for seed in (1, 2):  # the second replay starts from the first one's states
        fresh = _Case(requests, q, seed=seed)
        x.copy_(fresh.x)
        case.slots.copy_(fresh.slots)
        case.accepted.copy_(fresh.accepted)
        _replay_matches_eager(graph, out, stage, caches, x, metadata, q)


@pytest.mark.parametrize(("requests", "q"), [(2, 6), (4, 1)])
def test_weight_refit(requests, q):
    """Weights written in place after the capture (load, refit, offload) reach the
    replay: the layer keeps the parameters or views of them, nothing derived."""
    case = _Case(requests, q)
    caches = case.clone_caches()
    stage = case.stage(caches)
    x, metadata = case.x.clone(), case.metadata()
    graph, out = _capture(stage, x, metadata, q)
    before = out.clone()
    fresh = _Case(requests, q, seed=3)
    for name in WEIGHTS:
        getattr(case, name).copy_(getattr(fresh, name))
    # A layer built after the write reads the new weights from the start.
    _replay_matches_eager(graph, out, case.stage(caches), caches, x, metadata, q)
    assert not torch.equal(out, before)


@pytest.mark.parametrize("q", [1, 6])
def test_one_layer_many_graphs(q):
    """One layer object built for every capture size of `q` holds a graph per size,
    as the runner captures them; replays in mixed order on new inputs match eager
    runs of the same object."""
    sizes = {1: [1, 2, 4, 8, 16, 24, 32], 6: [1, 2, 3, 4, 5]}[q]
    case = _Case(max(sizes), q)
    caches = case.clone_caches()
    stage = case.stage(caches, [r * q for r in sizes])
    graphs = {}
    for r in sizes:
        x = case.x[: r * q].clone()
        slots, accepted = case.slots[:r].clone(), case.accepted[:r].clone()
        metadata = _metadata(slots, accepted)
        graph, out = _capture(stage, x, metadata, q)
        graphs[r] = (graph, out, x, slots, accepted, metadata)
    order = sizes[::-1] + sizes[::2] + sizes[1::2]
    for seed, r in enumerate(order, 1):
        graph, out, x, slots, accepted, metadata = graphs[r]
        fresh = _Case(max(sizes), q, seed=seed)  # the same pool, new slots
        x.copy_(fresh.x[: r * q])
        slots.copy_(fresh.slots[:r])
        accepted.copy_(fresh.accepted[:r])
        _replay_matches_eager(graph, out, stage, caches, x, metadata, q)


@pytest.mark.parametrize("q", [1, 6])
def test_high_slot_index(q):
    """A request at the top of an 8200-slot pool, where a slot's state starts past
    2^31 FP32 elements (262144 per slot), gives the bits it gives at the bottom."""
    pool = 8200
    case = _Case(1, q)
    state = torch.empty(pool, HEADS, HEAD_DIM, HEAD_DIM, device="cuda")
    conv = torch.empty(pool, 2 + q, 3 * DIM, dtype=torch.bfloat16, device="cuda")
    caches = (conv, state)
    stage = case.stage(caches)
    accepted = case.accepted.clone()
    start = int(accepted[0]) - 1 if q > 1 else 0
    results = []
    for first in (1, pool - q):
        slots = torch.arange(first, first + q, dtype=torch.int32, device="cuda")
        slots = slots.view(1, q)
        state[slots[0, start]] = case.caches[1][case.slots[0, start]]
        conv[slots[0, 0]] = case.caches[0][case.slots[0, 0]]
        out = stage.run(case.x, _metadata(slots, accepted), q).clone()
        results.append((out, state[slots[0]].clone(), conv[slots[0, 0]].clone()))
    assert all(torch.equal(a, b) for a, b in zip(*results))
    assert int(slots[0, -1]) * HEADS * HEAD_DIM * HEAD_DIM >= 2**31
