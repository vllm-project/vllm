# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K1 (``mono/k1.py``, Phase B1, DESIGN_BC 5, WP-B4) in a kernel layer's chain
on MI355X: K1 -> the stock attention core -> K2(gated), against the ops vLLM
runs for the same decode step, random weights at TP8 shapes (9.3).

The reference is composed from ops, as Phase A's (``test_numerics``):
``fused_norm_residual_norm_fwd`` (the B_ffn boundary) -> the unquantized GEMM
dispatch (``dispatch_unquantized_gemm``: wvSplitK / hipBLASLt) -> the fused
QK-norm / RoPE / KV insert (``fused_muse_spark_qknorm_rope_kv_insert``) ->
``paged_attention_v1`` (a sliding layer on ``_build_swa_decode_tables``'
window-trimmed table) -> ``rms_norm_gated`` -> the stock span from o_proj to
the unpooler (``test_numerics._stock_span``). The mono side runs vLLM's
core on K1's own rows and cache, as a B1 deployment does.

One sliding (RoPE) and one full (NoPE) layer at widths 1 and 8. Each K1
stage on the kernel's own inputs: ``stream_out`` within the magnitude-scaled
bound of DESIGN_BC 4, the ``arow`` rows <= 1 bf16 ulp from the fp64 pre-qkv
norm of the kernel's own ``stream_out`` and bit-equal to stock on >= 99.9 %,
``qkvo`` and the cache against the GEMM dispatch and the stock insert on the
kernel's own ``arow`` rows (cos; no other cache entry written). The chain:
K2's ``stream_out`` cos > 0.99999 against the stock chain, ``branch_out`` cos
> 0.9999 against vLLM's experts and unpooler on the kernel's own ``ffn_in``
rows and picks (Phase A's reference, ``test_numerics._check_step``), live
rows; the ranks agree bit for bit; NaN pad rows (slot -1) leave the live rows
and the cache unchanged and come out as zeros; one CUDA graph of the chain
(K1, the stock core, K2) replayed 300 times on two input sets written in
place in turn, every replay (the cache included) bit for bit the eager chain
on its own set; no wait gives up.
"""

from types import SimpleNamespace

import pytest
import ray
import torch

from tests.models.muse_spark.amd_specialized_decode import reference as ref
from tests.models.muse_spark.amd_specialized_decode import test_numerics as tn
from tests.utils import multi_gpu_test, multi_process_parallel
from vllm.models.muse_spark.amd.specialized_decode.mono import spec
from vllm.models.muse_spark.amd.specialized_decode.mono.abi import (
    STARTUP_WAIT_SECONDS,
    WAIT_SECONDS,
)

bf = torch.bfloat16
f32 = torch.float32
H, CORE, QKV, MT = spec.HIDDEN, spec.CORE, spec.QKV_ROWS, spec.MAX_TOKENS
WIDTHS = (1, 8)
KINDS = ("sliding", "full")
# the kernel layer each kind runs as: layer 1 is full, 2 sliding (DESIGN_BC
# 3.1), so each has its own K1 / K2 tag slots
LAYER = {"full": 1, "sliding": 2}
MAX_POS = 1 << 20  # the served RoPE table's rows
BLOCKS = 1024  # a rank's cache: 8 MiB
# each row's context length, this step's token included: the window's edges
# (a sliding layer's keys k >= L - 2048) and pages' edges
LENS = {
    1: (2100,),
    8: (1, 16, 17, 255, 300, 2048, 2049, 2500),
}
PAD_ROWS = (5, 6, 7)  # the pad case's rows (S = 8)
REPLAYS = 300  # the chain graph's replays
COS_STREAM = 0.99999
# branch_out against vLLM's experts on the kernel's own ffn_in rows and picks,
# as Phase A (test_numerics._check_step, COS_CHAIN): not against the stock
# chain, which is itself order-dependent there. vLLM's GEMM dispatch gives the
# same rows other bits at another width (S = 8 batched vs one row at a time),
# and the MXFP4 experts turn 1-ulp ffn_in flips into whole-stock branch cos
# 0.9986 at the qkv GEMM and 0.9994 at the o_proj GEMM alone (rows down to
# 0.9968) while stream_out stays > 0.9999999. K1's
# GEMV is held to the dispatch in _check_front, the seam's ffn_in by Phase A.
COS_BRANCH = 0.9999
# K1's arguments besides the step's tensors and the launch state
K1_WEIGHTS = (
    "w_postffn",
    "ffn_alpha",
    "ffn_beta",
    "w_preqkv",
    "w_qkv",
    "cos_sin",
    "max_pos",
)

needs_gfx950 = tn.needs_gfx950


# ---------------------------------------------------------------- the cases


def _case(S: int, pad_rows=(), seed: int = 0, device="cpu") -> SimpleNamespace:
    """One step's per-row metadata: ``lens`` (host ints, this token included),
    ``positions`` = L - 1, distinct pages a row (``reference.assign_pages``,
    never block 0), the block table int32 [S, pages], ``slots`` = the page of
    the row's position x 16 + its offset; pad rows as MRV2 leaves them (slot
    -1, position 0, length 0). The tensors on ``device`` (a CUDA graph
    captures no host copy)."""
    g = torch.Generator().manual_seed(41 + 8 * S + seed)
    lens = list(LENS[S])
    pages = ref.assign_pages(lens, BLOCKS, generator=g)
    bt = torch.zeros(S, max(len(p) for p in pages), dtype=torch.int32)
    for t, p in enumerate(pages):
        bt[t, : len(p)] = torch.tensor(p, dtype=torch.int32)
    pos = torch.tensor([L - 1 for L in lens], dtype=torch.int64)
    slots = bt[torch.arange(S), pos // spec.KV_BLOCK].long() * spec.KV_BLOCK
    slots += pos % spec.KV_BLOCK
    for t in pad_rows:
        lens[t], pos[t], slots[t] = 0, 0, ref.PAD_SLOT
    return SimpleNamespace(
        S=S,
        lens=lens,
        seq_lens=torch.tensor(lens, dtype=torch.int32, device=device),
        bt=bt.to(device),
        positions=pos.to(device),
        slots=slots.to(device),
        pad_rows=tuple(pad_rows),
    )


def _front_weights(rank: int, device, seed: int = 0) -> dict:
    """One rank's K1 weights: the previous layer's ``post_ffn_norm`` weight
    and ``add_ffn`` coefficients and this layer's ``pre_qkv_proj_norm``
    weight (replicated, seed 17 + seed), this rank's qkv_proj shard bf16
    [2304, 8192] (seed 100 + rank + 8 seed: DESIGN_BC 9.3)."""
    from vllm.models.muse_spark.common.layers import (
        MuseSpark1xNormPreservingResidualConnection,
    )

    gc = torch.Generator(device=device).manual_seed(17 + seed)
    gr = torch.Generator(device=device).manual_seed(100 + rank + 8 * seed)

    def rn(g, *shape, s=1.0, dtype=bf):
        return (torch.randn(*shape, generator=g, device=device) * s).to(dtype)

    conn = MuseSpark1xNormPreservingResidualConnection(H).to(device)
    with torch.no_grad():
        conn.gate.copy_(rn(gc, H, dtype=f32))
    conn.materialize_coefficients()
    return {
        "w_postffn": rn(gc, H, s=0.1),
        "ffn_alpha": conn.alpha.contiguous(),
        "ffn_beta": conn.beta.contiguous(),
        "w_preqkv": rn(gc, H, s=0.1),
        "w_qkv": rn(gr, QKV, H, s=H**-0.5),
    }


def _with_table(fw: dict, table: torch.Tensor | None) -> dict:
    """``fw`` with a layer's RoPE table (None: a NoPE layer, 0 and 0 rows)."""
    return {
        **fw,
        "cos_sin": 0 if table is None else table,
        "max_pos": 0 if table is None else int(table.shape[0]),
    }


def _step_inputs(
    device, S: int, pad_rows=(), seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """(branch bf16, stream fp32) [S, H], replicated (the previous layer's
    deferred boundary); pad rows NaN; ``seed`` picks another set of the same
    width."""
    g = torch.Generator(device=device).manual_seed(11 + S + 1000 * seed)
    branch = torch.randn(S, H, generator=g, device=device) * 0.7
    stream = torch.randn(S, H, generator=g, device=device) * 2.0
    for t in pad_rows:
        branch[t] = float("nan")
        stream[t] = float("nan")
    return branch.to(bf), stream


def _k1_args(runner, fw, kv, case, dev, branch, stream, slot):
    """K1's arguments (``spec.K1_ARGS``): the layer's weights, the cache's
    (``runtime.layers.kv_args``: byte strides), the step's tensors, the
    launch state; -> (args, stream_out, qkvo)."""
    from vllm.models.muse_spark.amd.specialized_decode.runtime.layers import kv_args

    S = case.S
    stream_out = torch.empty_like(stream)
    qkvo = torch.empty(S, QKV, dtype=bf, device=dev)
    args = {
        **{n: fw[n] for n in K1_WEIGHTS},
        **kv_args(kv),
        "branch": branch,
        "stream": stream,
        "pad": tn._pad(dev, case.pad_rows),
        "positions": case.positions.to(dev),
        "slots": case.slots.to(dev),
        "stream_out": stream_out,
        "qkvo": qkvo,
        "slot": slot,
        "rank": runner.rank,
        **runner.launch_args(),
    }
    return args, stream_out, qkvo


# ---------------------------------------------------------------- [TP8]


def _kv(rank: int, device) -> torch.Tensor:
    """A rank's cache with every context key in place (its KV heads)."""
    g = torch.Generator().manual_seed(300 + rank)
    return ref.kv_cache(BLOCKS, null=0.0, generator=g).to(device)


def _k1(runner, fw, kv, case, branch, stream, slot, key):
    """One K1 launch: -> (stream_out, qkvo)."""
    from vllm.models.muse_spark.amd.specialized_decode.mono import k1

    args, stream_out, qkvo = _k1_args(
        runner, fw, kv, case, stream.device, branch, stream, slot
    )
    k1.k1_launch(key, args)
    return stream_out, qkvo


def _k2g(runner, w, attn, qkvo, stream, pad, slot):
    """One K2(gated) launch on (attn, qkvo): -> (stream_out, branch_out)."""
    from vllm.models.muse_spark.amd.specialized_decode.mono import layer

    S = attn.shape[0]
    stream_out = torch.empty_like(stream)
    branch_out = torch.empty(S, H, dtype=bf, device=stream.device)
    layer.k2_launch(
        spec.K2Build(tokens=S, core="gated"),
        {
            **{n: w[n] for n in tn.KERNEL_WEIGHTS},
            "attn": attn,
            "qkvo": qkvo,
            "stream": stream,
            "pad": pad,
            "stream_out": stream_out,
            "branch_out": branch_out,
            "slot": slot,
            **runner.launch_args(),
            **runner.peer_args(),
        },
    )
    return stream_out, branch_out


def _stock_core(qkvo, kv, bt, seq_lens, max_len: int, window: bool):
    """VLLM's decode core on a qkv block: ``paged_attention_v1`` on the
    cache's K / V views, a sliding layer on ``_build_swa_decode_tables``'
    window-trimmed table (window 2048), a full layer on the whole table.
    bf16 [S, CORE]."""
    import aiter  # noqa: F401

    from vllm.v1.attention.backends.rocm_aiter_fa import (
        _PARTITION_SIZE_ROCM,
        AiterFlashAttentionMetadataBuilder,
    )

    S, dev = qkvo.shape[0], kv.device
    key_cache, value_cache = kv.transpose(1, 2).split(spec.V_OFF, dim=-1)
    table, lens, sliding = bt, seq_lens, 0
    if window:
        nwb = spec.WINDOW // spec.KV_BLOCK + 1
        builder = SimpleNamespace(
            aot_sliding_window=(spec.WINDOW - 1, 0),
            block_size=spec.KV_BLOCK,
            _swa_block_table=torch.zeros(S, nwb, dtype=torch.int32, device=dev),
            _swa_seq_lens=torch.zeros(S, dtype=torch.int32, device=dev),
            _swa_block_offsets=torch.arange(nwb, dtype=torch.int64, device=dev),
        )
        table, lens, max_len = (
            AiterFlashAttentionMetadataBuilder._build_swa_decode_tables(
                builder, bt, seq_lens, S, max_len
            )
        )
        sliding = spec.WINDOW
    parts = -(-max(max_len, 1) // _PARTITION_SIZE_ROCM)
    work = torch.empty(
        S * spec.HEADS * parts * 64 * 2 + 2 * S * spec.HEADS * parts * 4,
        dtype=torch.uint8,
        device=dev,
    )
    q = qkvo[:, :CORE].reshape(S, spec.HEADS, 64).contiguous()
    out = torch.empty_like(q)
    one = torch.ones(1, dtype=f32, device=dev)
    # the kernel reads cu_query_lens[t + 1] (query_len = its difference): decode
    # passes ``query_start_loc[:num_decodes]``, a view of the [S + 1] offsets,
    # so the row past the end is real; a bare arange(S) reads past its end and
    # a stale value > 1 there skips the row (output never written)
    qsl = torch.arange(S + 1, dtype=torch.int32, device=dev)[:S]
    torch.ops.aiter.paged_attention_v1(
        out, work, q, key_cache, value_cache, spec.QK_SCALE, table,
        qsl, lens, max_len, None,
        "auto", "NHD", 0.0, one, one, None, _PARTITION_SIZE_ROCM, 1, sliding,
    )  # fmt: skip
    return out.view(S, CORE)


def _gate(attn, qkvo):
    """VLLM's gated core (o_proj's input): ``rms_norm_gated(attn, og)``."""
    from vllm.third_party.flash_linear_attention.ops.kda import (
        rms_norm_gated,
    )

    S = attn.shape[0]
    return rms_norm_gated(
        attn.view(S, spec.HEADS, 64),
        qkvo[:, spec.ROW_G :].view(S, spec.HEADS, 64),
        None,
        None,
        "sigmoid",
        eps=spec.RMS_EPS,
    ).reshape(S, CORE)


def _stock_insert(rows, positions, table, slots, kv) -> None:
    """The stock fused QK-norm / RoPE / KV insert in place on ``rows`` (bf16
    [S, QKV], contiguous) and ``kv``: the decode instance (HPG 1)."""
    from vllm import _custom_ops as ops

    ops.fused_muse_spark_qknorm_rope_kv_insert(
        rows,
        positions,
        spec.HEADS,
        spec.KV_HEADS,
        spec.HEAD_DIM,
        spec.RMS_EPS,
        table,
        spec.HEAD_DIM if table is not None else 0,
        slots,
        kv,
        spec.KV_BLOCK,
    )


def _stock_boundary(fw, branch, stream):
    """The stock B_ffn boundary into the layer: (attn_in bf16, stream fp32)."""
    from vllm.models.muse_spark.common.ops.fused_rmsnorm import (
        fused_norm_residual_norm_fwd,
    )

    return fused_norm_residual_norm_fwd(
        branch,
        stream,
        fw["w_postffn"],
        fw["w_preqkv"],
        spec.POST_FFN_EPS,
        spec.RMS_EPS,
        alpha=fw["ffn_alpha"],
        beta=fw["ffn_beta"],
        post_gain_center=spec.GAIN,
        pre_gain_center=spec.GAIN,
        post_input_scale=True,
        out_dtype=bf,
    )


def _stock_front(fw, table, kv, case, branch, stream):
    """The stock front: (qkv rows after the insert, stream); ``kv`` in place."""
    dev = stream.device
    attn_in, stream_out = _stock_boundary(fw, branch, stream)
    qkv = tn._gemm(attn_in, fw["w_qkv"])
    _stock_insert(qkv, case.positions.to(dev), table, case.slots.to(dev), kv)
    return qkv, stream_out


def _scratch_rows(runner, name: str, cols: int, S: int) -> torch.Tensor:
    """Rows 0 .. S-1 of a plain bf16 scratch region (``arow``)."""
    off, nbytes = spec.scratch_layout()[name]
    return runner.scratch[off : off + nbytes].view(bf).view(MT, cols)[:S].cpu()


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int16)


def _check_front(runner, fw, table, kv0, kv, case, branch, stream, stream_k1, qkvo):
    """Each K1 stage on the kernel's own inputs (module docstring). Live rows;
    the pad rows' outputs are zeros. Rank-local checks (``_tp_checks``)."""
    S = case.S
    dev = stream.device
    live = torch.tensor([t not in case.pad_rows for t in range(S)])
    attn_in, ref_stream = _stock_boundary(fw, branch, stream)
    got_stream = stream_k1.cpu()
    ref_stream = ref_stream.cpu()
    # the boundary: stream_out within DESIGN_BC 4's magnitude-scaled bound
    tol = ref.stream_tolerance(
        branch.cpu(),
        stream.cpu(),
        fw["w_postffn"].cpu(),
        fw["ffn_alpha"].cpu(),
        fw["ffn_beta"].cpu(),
    )
    err = (got_stream - ref_stream).abs()
    assert (err[live] <= tol[live] + 1e-30).all(), err[live].max()
    # arow: <= 1 ulp from the fp64 pre-qkv norm of the kernel's own stream,
    # bit-equal to stock on >= 99.9 %
    arow = _scratch_rows(runner, spec.AROW, H, S)
    n = got_stream.double()
    rs = torch.rsqrt(n.square().mean(-1, keepdim=True) + spec.RMS_EPS)
    own = (n * rs * (fw["w_preqkv"].cpu().double() + spec.GAIN)).float().to(bf)
    assert ref.bf16_ulps(arow[live], own[live]).max().item() <= 1
    frac = (ref.bf16_ulps(arow[live], attn_in.cpu()[live]) == 0).float().mean()
    assert frac.item() >= 0.999, frac.item()
    # the GEMV and the epilogue: the GEMM dispatch and the stock insert on the
    # kernel's own arow rows (the two GEMVs' fp32 orders differ: cos)
    stock = tn._gemm(arow.to(dev), fw["w_qkv"]).contiguous()
    kv_stock = kv0.clone()
    _stock_insert(stock, case.positions.to(dev), table, case.slots.to(dev), kv_stock)
    got = qkvo.cpu()
    assert ref.cos(got[live], stock.cpu()[live]) > COS_STREAM
    assert (_bits(got[~live]) == 0).all()
    assert (got_stream[~live] == 0).all()
    # the cache: the stock insert's entries (cos), nothing else written
    put = _bits(kv_stock) != _bits(kv0)
    assert ref.cos(kv[put], kv_stock[put]) > COS_STREAM
    assert torch.equal(_bits(kv)[~put], _bits(kv0)[~put])


def _chain(runner, w, fw, table, kv, case, window, branch, stream):
    """The mono chain of one kernel layer, the step's epoch bumped first:
    K1 -> vLLM's core on K1's rows and cache -> K2(gated). -> (stream_k1,
    qkvo, attn, stream_out, branch_out)."""
    S, dev = case.S, stream.device
    layer_idx = LAYER["sliding" if window else "full"]
    key = spec.K1Build(tokens=S, rope=window)
    runner.step_begin()
    stream_k1, qkvo = _k1(
        runner, fw, kv, case, branch, stream, spec.k1_slot(layer_idx), key
    )
    attn = _stock_core(
        qkvo, kv, case.bt.to(dev), case.seq_lens.to(dev), max(case.lens), window
    )
    pad = tn._pad(dev, case.pad_rows)
    out = _k2g(runner, w, attn, qkvo, stream_k1, pad, spec.k2_slot(layer_idx))
    return (stream_k1, qkvo, attn, *out)


def _own_branch(runner, w, S: int) -> torch.Tensor:
    """The branch vLLM's experts and unpooler give on the last K2 launch's own
    ``ffn_in`` rows (``xrow``) and picks (``route``): Phase A's ``branch_s``
    (``test_numerics._check_step``). Read before any later launch; collective
    (the experts all-reduce)."""
    rb = tn.readback(runner, S)
    dev = runner.scratch.device
    _, moe = tn._stock_experts(
        rb["xrow"].to(dev), tn._route_ids(rb["ids"], dev), rb["tw"].to(dev), w
    )
    return tn._linear(moe, w["w_unpool"])


def _stock_chain(w, fw, table, kv, case, window, branch, stream) -> dict:
    """The stock chain from the same inputs (``kv`` in place)."""
    S, dev = case.S, stream.device
    qkv, stream1 = _stock_front(fw, table, kv, case, branch, stream)
    attn = _stock_core(
        qkv, kv, case.bt.to(dev), case.seq_lens.to(dev), max(case.lens), window
    )
    core = _gate(attn, qkv)
    out: dict = {}
    unit = torch.ones(S * spec.TOPK, 1, dtype=f32, device=dev)
    tn._stock_span(w, core, stream1, tn._pad(dev, case.pad_rows), out, unit)
    return out


@ray.remote(num_gpus=1, max_calls=1)
def _chain_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    device = tn._init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.models.muse_spark.amd.specialized_decode.runtime.runner import MonoRunner

    runner = MonoRunner(device)
    # the first launch of each build compiles it: the startup budget
    runner.set_wait_budget(STARTUP_WAIT_SECONDS)
    w = tn._weights(rank, device)
    fw0 = _front_weights(rank, device)
    rope = ref.rope_table(MAX_POS, device=device)
    for kind in KINDS:
        window = kind == "sliding"
        table = rope if window else None
        fw = _with_table(fw0, table)
        for S in WIDTHS:
            case = _case(S, device=device)
            branch, stream = _step_inputs(device, S)
            kv0 = _kv(rank, device)
            kv = kv0.clone()
            stream_k1, qkvo, _, s_out, b_out = _chain(
                runner, w, fw, table, kv, case, window, branch, stream
            )
            torch.accelerator.synchronize()
            own = _own_branch(runner, w, S)
            tn._assert_healthy(runner)
            with tn._tp_checks(f"the K1 stages ({kind}, S={S})"):
                _check_front(
                    runner, fw, table, kv0, kv, case, branch, stream, stream_k1, qkvo
                )
            kv_stock = kv0.clone()
            want = _stock_chain(w, fw, table, kv_stock, case, window, branch, stream)
            torch.accelerator.synchronize()
            with tn._tp_checks(f"the chain ({kind}, S={S})"):
                c_s = ref.cos(s_out, want["stream"])
                c_b = ref.cos(b_out, own)
                c_e = ref.cos(b_out, want["branch"])  # printed: COS_BRANCH's note
                print(
                    f"{kind} S={S}: stream_out cos {c_s:.8f}, branch cos {c_b:.7f}"
                    f" (vs the whole stock chain {c_e:.7f})"
                )
                assert c_s > COS_STREAM and c_b > COS_BRANCH, (c_s, c_b)
            for x, what in ((stream_k1, "K1 stream_out"), (s_out, "stream_out")):
                tn._all_ranks_equal(x, f"{what} ({kind}, S={S})")
            tn._all_ranks_equal(b_out, f"branch_out ({kind}, S={S})")

    # pad rows (S = 8, slot -1, length 0) with NaN inputs: the live rows and
    # the cache as with clean pad rows, the pad rows' outputs exact zeros
    window = True
    fw = _with_table(fw0, rope)
    case = _case(8, PAD_ROWS, device=device)
    clean_b, clean_s = _step_inputs(device, 8)
    dirty_b, dirty_s = _step_inputs(device, 8, PAD_ROWS)
    kv0 = _kv(rank, device)
    kv_c, kv_d = kv0.clone(), kv0.clone()
    clean = _chain(runner, w, fw, rope, kv_c, case, window, clean_b, clean_s)
    dirty = _chain(runner, w, fw, rope, kv_d, case, window, dirty_b, dirty_s)
    own = _own_branch(runner, w, 8)  # the dirty launch's
    # the stock chain all-reduces: outside the verdict block (no collective in
    # a ``_tp_checks``: a rank that failed before it would skip it)
    want = _stock_chain(w, fw, rope, kv0.clone(), case, window, clean_b, clean_s)
    torch.accelerator.synchronize()
    tn._assert_healthy(runner)
    live = [t for t in range(8) if t not in PAD_ROWS]
    with tn._tp_checks("the NaN pad rows"):
        assert torch.equal(_bits(kv_c), _bits(kv_d))
        for i in (0, 1, 3, 4):  # stream_k1, qkvo, stream_out, branch_out
            g, c = dirty[i], clean[i]
            assert torch.equal(g[live], c[live]), i
            assert (g[list(PAD_ROWS)].float() == 0).all(), i
        assert ref.cos(dirty[3][live], want["stream"][live]) > COS_STREAM
        assert ref.cos(dirty[4][live], own[live]) > COS_BRANCH

    # one CUDA graph of the chain (epoch bump, K1, the stock core, K2) at two
    # widths, replayed on two input sets written in place in turn: every
    # replay, the cache included, matches the eager chain on its own set. A
    # K1 or K2 wait that took a stale tag of the same parity (two epochs back:
    # the previous replay's) would read the other set's rows, which identical
    # inputs would hide. The step's slots are fixed, so the cache after a
    # replay is the pristine one with that set's K / V inserted.
    sets = [{S: _step_inputs(device, S, seed=k) for S in (1, 8)} for k in (0, 1)]
    ins = {}
    for S, kind in ((1, "full"), (8, "sliding")):
        window = kind == "sliding"
        table = rope if window else None
        ins[S] = (
            _with_table(fw0, table),
            table,
            _kv(rank, device),
            _case(S, device=device),
            window,
            *(t.clone() for t in sets[0][S]),
        )

    def step(S, k=None):
        """The chain on the graph's buffers, or (``k``) eagerly on set k and
        a copy of the pristine cache. -> (outputs, cache)."""
        fw, table, kv, case, window, branch, stream = ins[S]
        if k is not None:
            kv = kv.clone()
            branch, stream = sets[k][S]
        out = _chain(runner, w, fw, table, kv, case, window, branch, stream)
        return out, kv

    # the eager references, also the warm-ups (the S = 1 NoPE K1 build
    # compiles here: a cross-rank compile skew, so the startup budget), then
    # the serving budget for the graph
    want = [{S: step(S, k) for S in ins} for k in (0, 1)]
    torch.accelerator.synchronize()
    tn._assert_healthy(runner)
    runner.set_wait_budget(WAIT_SECONDS)
    graph = torch.cuda.CUDAGraph()
    outs = {}
    with torch.cuda.graph(graph):
        for S in ins:
            outs[S] = step(S)[0]
    # (replay, width index, output: stream_k1, qkvo, attn, stream_out,
    # branch_out, then the cache)
    bad = torch.zeros(REPLAYS, len(ins), 6, dtype=torch.bool, device=device)
    for i in range(REPLAYS):
        k = i % 2
        for S in ins:
            for dst, src in zip(ins[S][5:], sets[k][S]):
                dst.copy_(src)
        graph.replay()
        for j, S in enumerate(ins):
            w_out, w_kv = want[k][S]
            for o, (got, ref_out) in enumerate(zip(outs[S], w_out)):
                bad[i, j, o] = (got != ref_out).any()
            bad[i, j, 5] = (_bits(ins[S][2]) != _bits(w_kv)).any()
    torch.accelerator.synchronize()
    with tn._tp_checks("the chain's graph replays"):
        assert not bad.any(), bad.nonzero()[:8].tolist()
    tn._assert_healthy(runner)


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_layer_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _chain_worker)
