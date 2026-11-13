# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode: AttnRes saved-block statistics computed ahead by the
previous layer's fused o-projection (`cutedsl.attn_res.prepared` ATTN_STATS and
ATTN_STATS_BLOCKS, `runtime.decode.SpecializedDecode.bind_statistics_ahead`).

- Policy and wiring (no GPU): the tables name TP16 + DCP16 only; the decode binds a
  KDA layer that follows a KDA layer, reads one of the listed saved-block counts and
  reads the same saved blocks as that layer's o-projection, at the listed rows only;
  `PreparedAttnRes` and `PreparedOproj` run the precomputed variants for the
  bound rows and the default kernels for every other row.
- Bits (SM100/SM103, one GPU): the fused o-projection that also computes the next
  layer's statistics, then the AttnRes that reads them, give the same output, prefix,
  saved-block and aux rows bit for bit as the pair without them, eager and in CUDA
  graph replays, on the ordinary and the Lamport-ring AttnRes. The TP16 all-reduce
  runs on one GPU: the producer's multicast store becomes plain stores into all 16
  rank slots of a local mailbox, in both arms.
"""

import math
from types import SimpleNamespace

import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass import Float32

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.attn_res import (
    prepared as attn_res_prepared,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    add_rn,
    at,
    store_words,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.oproj import (
    prepared as oproj_prepared,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.oproj import producer
from vllm.platforms import current_platform

requires_sm100 = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100),
    reason="the specialized decode kernels need SM100/SM103",
)

TP, CAP, HID, KO = 16, 64, 7168, 768  # TP16 o-projection: [B, 768] -> [B, 7168]
SENT = -0x80000000  # mailbox sentinel word
BLOCK = 12  # layers per AttnRes block
ROWS = (5, 8)  # one request at DSpark k = 4 and k = 7
BATCHES = tuple(range(1, 65))
STATS_BLOCKS = {2, 3, 4, 5, 7, 8}


def prev_valid(layer):
    return math.ceil(layer / BLOCK)


def schedule_layer(layer, kind="kda"):
    """Decoder `layer` as the decode wiring reads it."""
    attn = SimpleNamespace(in_proj_qkvgfab=None)
    if kind != "kda":
        attn = SimpleNamespace(dsa_indexer=None)
    return SimpleNamespace(
        self_attn=attn,
        layer_idx=layer,
        prev_valid_blocks=prev_valid(layer),
        is_block_write_layer=layer % BLOCK == 0,
    )


# --------------------------------------------------------------------------
# Policy and wiring (no GPU)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dcp", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("tp", [8, 16])
def test_statistics_tables(tp, dcp):
    on = (tp, dcp) == (16, 16)
    rows = attn_res_prepared.statistics_rows(tp, dcp)
    blocks = attn_res_prepared.statistics_blocks(tp, dcp)
    assert rows == (frozenset(range(5, 9)) if on else frozenset())
    assert blocks == (frozenset(STATS_BLOCKS) if on else frozenset())


class _Output:
    """PreparedOproj stand-in: records bind_ahead; `refuse` binds no row."""

    def __init__(self, refuse=False):
        self.calls, self.refuse = [], refuse

    def bind_ahead(self, target, mixture, scal, rows):
        self.calls.append((target.layer_idx, mixture, scal, frozenset(rows)))
        return () if self.refuse else tuple(sorted(rows))


class _Before:
    """PreparedAttnRes stand-in: records bind_statistics."""

    def __init__(self):
        self.calls = []

    def bind_statistics(self, mixture, scal, rows):
        self.calls.append((mixture, scal, tuple(rows)))


def _decode(layers, tp=16, dcp=16, batches=BATCHES, refuse=()):
    return SimpleNamespace(
        tp=tp,
        dcp=dcp,
        batches=batches,
        buffers={},
        device=torch.device("cpu"),
        attention={
            d.layer_idx: SimpleNamespace(output=_Output(d.layer_idx in refuse))
            for d in layers
        },
        before={d.layer_idx: _Before() for d in layers},
    )


def test_bind_statistics_ahead_layers():
    """The wiring binds exactly the KDA -> KDA pairs whose target reads a listed
    saved-block count and the host's saved blocks, at rows 5-8, each pair on the
    statistics rows of the host's parity; a host that binds no row leaves its
    target on the default kernel."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.decode import (
        SpecializedDecode,
    )

    # Every fourth layer DSA; layer 36 (a block write) left out, so 35 -> 37 reads
    # one saved block more than layer 35's o-projection.
    kinds = {i: "dsa" if i % 4 == 2 else "kda" for i in range(1, 93)}
    layers = [schedule_layer(i, kinds[i]) for i in range(1, 93) if i != 36]
    decode = _decode(layers, refuse={48})
    SpecializedDecode.bind_statistics_ahead(decode, layers)

    mixture, scal = decode.buffers["attn_res.statistics"]
    assert mixture.shape == (2, 8, HID) and mixture.dtype == torch.float32
    assert scal.shape == (2, 8, 2) and scal.dtype == torch.float32
    expected = set()
    for host, target in zip(layers, layers[1:]):
        kda = kinds[host.layer_idx] == kinds[target.layer_idx] == "kda"
        nb = target.prev_valid_blocks
        same = nb == host.prev_valid_blocks + host.is_block_write_layer
        if kda and nb in STATS_BLOCKS and same:
            expected.add((host.layer_idx, target.layer_idx))
    assert (35, 37) not in expected and (48, 49) in expected
    asked = set()
    for host in layers:
        for target, m, s, rows in decode.attention[host.layer_idx].output.calls:
            asked.add((host.layer_idx, target))
            parity = host.layer_idx % 2
            assert m.data_ptr() == mixture[parity].data_ptr() and m.shape == (8, HID)
            assert s.data_ptr() == scal[parity].data_ptr() and s.shape == (8, 2)
            assert rows == frozenset(range(5, 9))
    assert asked == expected
    bound = {i for i, before in decode.before.items() if before.calls}
    assert bound == {t for _, t in expected} - {49}
    for target in bound:
        ((m, s, rows),) = decode.before[target].calls
        assert rows == (5, 6, 7, 8)
        assert m.data_ptr() == mixture[(target - 1) % 2].data_ptr()


@pytest.mark.parametrize(
    "tp,dcp,batches",
    [(16, 8, BATCHES), (8, 8, BATCHES), (16, 16, (1, 2, 4))],
)
def test_bind_statistics_ahead_off(tp, dcp, batches):
    """Other (TP, DCP) pairs, and decodes without rows 5-8, bind nothing."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.decode import (
        SpecializedDecode,
    )

    layers = [schedule_layer(i) for i in range(1, 30)]
    decode = _decode(layers, tp, dcp, batches)
    SpecializedDecode.bind_statistics_ahead(decode, layers)
    assert decode.buffers == {}
    assert not any(a.output.calls for a in decode.attention.values())
    assert not any(b.calls for b in decode.before.values())


def test_attn_res_runs_the_bound_variants(monkeypatch):
    """PreparedAttnRes compiles a precomputed variant of each bound code and runs it,
    with the statistics rows, for the bound rows only."""
    prepared = attn_res_prepared.PreparedAttnRes.__new__(
        attn_res_prepared.PreparedAttnRes
    )
    prepared.code, prepared.views, prepared.eps = "code", ("v",), ("e0", "e1")
    prepared.ring = ("ring", "ring_view", (2, CAP))
    prepared.aux = ("aux", "aux_ring")
    prepared.stats = None
    compiled = []

    def compile_variant(slots, cap, aux=False, stats=0):
        compiled.append((slots, cap, aux, stats))
        return f"pre{len(compiled) - 1}"

    monkeypatch.setattr(prepared, "_compile", compile_variant)
    monkeypatch.setattr(attn_res_prepared, "to_cute", lambda t: ("cute", t.shape))
    monkeypatch.setattr(attn_res_prepared, "dynamic_view", lambda t: t)
    mixture, scal = torch.zeros(8, HID), torch.zeros(8, 2)
    prepared.bind_statistics(mixture, scal, (5, 8))
    assert compiled == [(0, 0, False, 8), (2, CAP, False, 8), (0, 0, True, 8)] + [
        (2, CAP, True, 8)
    ]
    launched = []

    def launch(code, source, prefix, blocks, out, aux=None, stats=()):
        launched.append((code, stats))
        return out

    monkeypatch.setattr(prepared, "_launch", launch)
    views = (("cute", (8, HID)), ("cute", (8, 2)))
    for rows in (4, 5, 8):
        bound = rows in (5, 8)
        prefix = torch.empty(rows, 1)
        launched.clear()
        prepared.run(prefix, prefix, prefix, prefix)
        prepared.run(prefix, prefix, prefix, prefix, aux=prefix)
        prepared.run_ring(prefix, prefix, prefix)
        prepared.run_ring(prefix, prefix, prefix, aux=prefix)
        if bound:
            assert launched == [
                ("pre0", views),
                ("pre2", views),
                ("pre1", views),
                ("pre3", views),
            ]
        else:
            codes = ("code", "aux", "ring", "aux_ring")
            assert launched == [(code, ()) for code in codes]


def test_attn_res_unbound_variants_stay_empty(monkeypatch):
    """Without a ring or an aux binding only the ordinary precomputed variant
    compiles."""
    prepared = attn_res_prepared.PreparedAttnRes.__new__(
        attn_res_prepared.PreparedAttnRes
    )
    prepared.ring = prepared.aux = None
    monkeypatch.setattr(prepared, "_compile", lambda *a: a)
    monkeypatch.setattr(attn_res_prepared, "to_cute", lambda t: t)
    prepared.bind_statistics(torch.zeros(8, HID), torch.zeros(8, 2), (5,))
    _, codes, _ = prepared.stats
    assert codes == [(0, 0, False, 8), None, None, None]


@pytest.mark.parametrize("nb", [0, 3])
def test_oproj_bind_ahead(nb, monkeypatch):
    """PreparedOproj compiles the fused kernel with the statistics for the fused rows
    it serves (up to the statistics rows' capacity, saved blocks present) and runs it
    with the next layer's weights, statistics rows and eps after its own arguments."""
    op = oproj_prepared.PreparedOproj.__new__(oproj_prepared.PreparedOproj)
    shape = lambda b: (KO, b, TP, nb, True, False, False)  # noqa: E731
    op.shapes = {b: shape(b) for b in range(1, 9)}
    op.ahead, op.ahead_views = {}, ()
    compiled = []

    def compile_oproj(*args):
        compiled.append(args)
        return lambda *call: launched.append(("ahead", call))

    monkeypatch.setattr(oproj_prepared, "compile_oproj", compile_oproj)
    monkeypatch.setattr(oproj_prepared, "to_cute", lambda t: ("cute", t.shape))
    norm = SimpleNamespace(weight=torch.ones(HID), variance_epsilon=1e-6)
    target = SimpleNamespace(
        self_attention_res_norm=norm,
        self_attention_res_proj=SimpleNamespace(weight=torch.ones(1, HID)),
    )
    launched: list = []
    bound = op.bind_ahead(target, torch.zeros(6, HID), torch.zeros(6, 2), {5, 6, 8, 9})
    if nb == 0:
        assert bound == () and compiled == []
        return
    assert bound == (5, 6)  # 8: past the 6 statistics rows; 9: no fused kernel
    assert compiled == [("fused", *shape(b), 64, False, 6) for b in (5, 6)]
    assert len(op.ahead_views) == 5
    assert op.ahead_views[:4] == (
        ("cute", (HID,)),
        ("cute", (HID,)),
        ("cute", (6, HID)),
        ("cute", (6, 2)),
    )
    # run: the bound row takes the statistics kernel, another row its own code.
    op.device, op.front, op.twoshots = torch.device("cpu"), None, set()
    op.views, op.eps, op.head, op.wview = ("w",) * 3, ("e0", "e1"), ("h",) * 3, "wv"
    op.codes = {
        b: (lambda *call, b=b: launched.append(("default", b, call)),) for b in (4, 5)
    }
    monkeypatch.setattr(oproj_prepared, "stream", lambda device: "stream")
    monkeypatch.setattr(oproj_prepared, "dynamic_view", lambda t: "row")
    rows = torch.empty(5, KO)
    op.run(rows, rows, rows, rows)
    ((kind, call),) = launched
    assert kind == "ahead"
    assert call[-1] == "stream" and call[-6:-1] == op.ahead_views
    assert call[:5] == (("cute", (5, KO)), "wv", "h", "h", "h")
    launched.clear()
    op.run(torch.empty(4, KO), rows, rows, rows)
    assert launched[0][:2] == ("default", 4)
    assert launched[0][2][-1] == "stream"


# --------------------------------------------------------------------------
# Bits (one GPU)
# --------------------------------------------------------------------------
@cute.jit
def _local_publish(self, args, red, cta, first: cutlass.Constexpr[int]):
    """OprojProducer.publish (single pass, no token groups) with plain stores into
    every rank slot of a local [TP, CAP, HID] mailbox instead of the multicast."""
    x, _, mailbox, mc, slot = args[:5]
    _, gt, _ = self.schedule(x.shape[0])
    count: cutlass.Constexpr = min(x.shape[0] - first, gt * 8)
    tid, _, _ = cute.arch.thread_idx()
    if tid < 8 * count:
        reduced = cute.make_rmem_tensor(8, Float32)
        out_half, row_tile, token = tid % 2, (tid // 2) % 4, tid // 8
        tile, row = token // 8, token % 8
        for v in cutlass.range_constexpr(8):
            value = red[tile, row_tile, 0, row, out_half * 8 + v]
            for q in cutlass.range_constexpr(1, self.parts):
                part = red[tile, row_tile, q, row, out_half * 8 + v]
                value = add_rn(value, part)
            reduced[v] = value
        words = producer.pack8(reduced, "latent")
        out_col = (cta * 4 + row_tile) * 16 + out_half * 8
        for r in cutlass.range_constexpr(mailbox.shape[0]):
            store_words(at(mailbox, r, first + token, out_col), words, scope="gpu")


class _LocalMailbox:
    """[parity, TP, CAP, HID] BF16 mailbox on this GPU, sentinel-armed."""

    def __init__(self):
        self.tp, self.rank = TP, 0
        self.buffer = torch.empty(
            (2, TP, CAP, HID), dtype=torch.bfloat16, device="cuda"
        )
        self.buffer.view(torch.int32).fill_(SENT)

    def slot(self, parity):
        return self.buffer[parity], 0

    def armed(self):
        return bool((self.buffer.view(torch.int32) == SENT).all())


def _decoder(layer, seed):
    """Decoder `layer` (KDA) with the weights the o-projection and the AttnRes
    read, deterministic per `seed`."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, device="cuda", generator=gen)  # noqa: E731

    def norm(scale=0.1, eps=1e-6):
        weight = (r(HID) * scale + 1).bfloat16()
        return SimpleNamespace(weight=weight, variance_epsilon=eps)

    decoder = schedule_layer(layer)
    decoder.self_attn.o_proj = SimpleNamespace(weight=(r(HID, KO) * 0.03).bfloat16())
    decoder.block_write_idx = layer // BLOCK
    decoder.is_moe_layer = True
    for name in ("mlp_res", "self_attention_res"):
        setattr(decoder, f"{name}_norm", norm())
        proj = SimpleNamespace(weight=(r(1, HID) * 0.05).bfloat16())
        setattr(decoder, f"{name}_proj", proj)
    decoder.post_attention_layernorm = norm()
    decoder.input_layernorm = norm(eps=1e-5)
    return decoder


class _Arm:
    """One o-projection -> AttnRes pair (host -> target layer), with or without the
    statistics ahead, and its row buffers."""

    def __init__(self, host, target, aux, ahead):
        from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.decode import (
            SpecializedDecode,
        )

        self.box = _LocalMailbox()
        self.oproj = oproj_prepared.PreparedOproj(host, self.box, ROWS, False)
        self.attn = attn_res_prepared.PreparedAttnRes.for_target(target, has_delta=True)
        self.ring = torch.empty((1, CAP, HID), dtype=torch.bfloat16, device="cuda")
        self.ring.view(torch.int32).fill_(SENT)
        self.attn.bind_ring(self.ring)
        if aux:
            self.attn.bind_aux()
        if ahead:
            decode = SimpleNamespace(
                tp=TP,
                dcp=16,
                batches=ROWS,
                buffers={},
                device=torch.device("cuda"),
                attention={host.layer_idx: SimpleNamespace(output=self.oproj)},
                before={target.layer_idx: self.attn},
            )
            SpecializedDecode.bind_statistics_ahead(decode, [host, target])
            assert tuple(sorted(self.oproj.ahead)) == ROWS
            assert tuple(sorted(self.attn.stats[0])) == ROWS
        else:
            assert self.oproj.ahead == {} and self.attn.stats is None
        empty = lambda *s: torch.zeros(*s, dtype=torch.bfloat16, device="cuda")  # noqa: E731
        self.prefix, self.blocks = empty(CAP, HID), empty(CAP, 8, HID)
        self.out, self.x = empty(CAP, HID), empty(CAP, HID)
        self.aux = empty(CAP, HID) if aux else None
        self.sep = torch.zeros(1024, device="cuda")

    def reset(self, b, variant, inputs):
        prefix, blocks, _, delta = inputs
        self.prefix.copy_(prefix)
        self.blocks.copy_(blocks)
        for t in (self.out, self.x) + ((self.aux,) if self.aux is not None else ()):
            t.zero_()
        self.ring.view(torch.int32).fill_(SENT)
        if variant == "ring":
            self.ring[0, :b].copy_(delta[:b])

    def layer(self, b, variant, inputs):
        local, delta = inputs[2], inputs[3]
        prefix, blocks = self.prefix[:b], self.blocks[:b]
        self.oproj.run(local[:b], prefix, blocks, self.out[:b])
        # The AttnRes reads the prefix and the statistics at launch: a plain kernel
        # in between (in the decode step the MoE runs there).
        self.sep.add_(1.0)
        aux = None if self.aux is None else self.aux[:b]
        if variant == "ring":
            self.attn.run_ring(prefix, blocks, self.x[:b], aux)
        else:
            self.attn.run(delta[:b], prefix, blocks, self.x[:b], aux)

    def snapshot(self, b):
        rows = [self.out[:b], self.x[:b], self.prefix[:b], self.blocks[:b]]
        if self.aux is not None:
            rows.append(self.aux[:b])
        state = (self.box.armed(), bool((self.ring.view(torch.int32) == SENT).all()))
        return [t.view(torch.int16).clone() for t in rows], state


def _check_bits(host_layer, aux):
    host = _decoder(host_layer, 100 + host_layer)
    target = _decoder(host_layer + 1, 1100 + host_layer)
    nb = host.prev_valid_blocks + host.is_block_write_layer
    assert nb == target.prev_valid_blocks and nb in STATS_BLOCKS
    gen = torch.Generator(device="cuda").manual_seed(7 + host_layer)
    r = lambda *s: torch.randn(*s, device="cuda", generator=gen)  # noqa: E731
    delta = (r(CAP, HID) * 0.3).bfloat16()
    words = delta.view(torch.int32)
    words[words == SENT] = 0  # no sentinel word in the Lamport delta rows
    inputs = (r(CAP, HID).bfloat16(), (r(CAP, 8, HID) * 0.5).bfloat16())
    inputs += (r(CAP, KO).bfloat16(), delta)
    arms = [_Arm(host, target, aux, ahead) for ahead in (False, True)]
    for b in ROWS:
        for variant in ("ordinary", "ring"):
            eager = []
            for arm in arms:
                arm.reset(b, variant, inputs)
                arm.layer(b, variant, inputs)
                torch.accelerator.synchronize()
                eager.append(arm.snapshot(b))
            for got, want in zip(eager[1][0], eager[0][0]):
                assert torch.equal(got, want), (host_layer, b, variant, "eager")
            assert eager[0][1] == eager[1][1], (host_layer, b, variant)
            assert eager[0][1][1], "the ring is re-armed"
            replays = []
            for arm in arms:
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):  # warm up outside the capture
                    arm.reset(b, variant, inputs)
                    arm.layer(b, variant, inputs)
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    arm.reset(b, variant, inputs)
                    arm.layer(b, variant, inputs)
                shots = []
                for _ in range(2):
                    arm.out.fill_(1.0)
                    arm.x.fill_(1.0)
                    graph.replay()
                    torch.accelerator.synchronize()
                    shots.append(arm.snapshot(b))
                replays.append(shots)
            for shot in range(2):
                pairs = zip(replays[1][shot][0], replays[0][shot][0])
                for got, want in pairs:
                    assert torch.equal(got, want), (host_layer, b, variant, shot)
                for got, want in zip(replays[0][shot][0], eager[0][0]):
                    assert torch.equal(got, want), (host_layer, b, variant, shot)


@requires_sm100
@pytest.mark.parametrize(
    "host_layer,aux",
    [(12, False), (25, True), (89, False)],
    ids=["nb2-block-write-host", "nb3-aux", "nb8"],
)
def test_statistics_ahead_bitwise(host_layer, aux, monkeypatch):
    monkeypatch.setattr(producer.OprojProducer, "publish", _local_publish)
    oproj_prepared.compile_oproj.cache_clear()  # compile with the local publish
    try:
        _check_bits(host_layer, aux)
    finally:
        oproj_prepared.compile_oproj.cache_clear()
