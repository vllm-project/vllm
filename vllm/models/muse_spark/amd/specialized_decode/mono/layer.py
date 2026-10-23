# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2: one Muse Spark decoder layer from the attention's gated core (o_proj's
input) to the layer's deferred FFN boundary, in one launch a layer and rank.

    o_proj        every CTA: tasks b and b + 256 (16 rows each), the partials
                  pushed to every rank (``ar``)
    seam          CTAs 0 .. 63: the AR reduce (rank order, fp32) and the
                  attention boundary (post_attn_norm, gated blend, pre_ffn_norm)
                  -> ``stream_out``, the ffn_in rows (``xrow``), XRDY
    route         CTAs 0 .. S-1, after their seam slice: row t's top-8
    router        CTAs 64 .. 95: one fp32 router row each -> every rank's logits
    pooler        CTAs 96 .. 223: 32 row groups x 4 K quarters -> every rank's
                  pooled rows
    xq            every CTA: the pooled rows -> pre-expert norm -> MXFP4 (LDS)
    union, ug     every CTA: the step's union of experts, gate/up tasks reversed
                  over the grid (CTA b: 255 - b, + 256, ...)
    down          every CTA b: pooled rows 16 b .. of every union expert ->
                  the column owner's ``rsx`` (always, zeros included)
    owner         CTAs 32 rank .. + 32: the rank-order reduce, the post-expert
                  norm (sum of squares exchanged), route weights, top-k sum
                  -> every rank's ``moe``
    unpool        every CTA: its row group's K quarter -> every rank's branch
    gather        CTAs 0 .. 63: the branch rows -> ``branch_out``

The attention core K2 starts from is the build's ``K2Build.core`` (one kernel
template, a variant a core; ``ABI_BY_CORE``): "tensor" (Phase A) reads
vLLM's gated core ``core``; "gated" (Phase B2, DESIGN_BC 6) reads vLLM's
attention output ``attn`` and the qkv rows ``qkvo`` and computes the output
norm and gate in its core load (``attention/gate.py``), those loads issued
before the o_proj weights; "split" (Phase C, the q = 1 widths, DESIGN_BC 7)
reads the qkv rows ``qkvo``, the layer's KV cache and its KV group's block
table and sequence lengths and runs the attention core itself:

    split         CTAs 0 .. n_units - 1: one split-KV unit each
                  (``attention/split.py``) -> partials, ``srdy``
    combine       CTAs C0 = 256 - 16 S ..: one (token, q head) each
                  (``attention/combine.py``): the units' partials merged, the
                  output norm and gate -> ``xo``, ``crdy``; a CTA in both
                  ranges runs its unit first

Every CTA issues its o_proj weights after its last role stage's publish
(``split_roles``, DESIGN_BC 7.2, rule 8: a publish waits for every load in
flight): a CTA with no role at entry, a split unit after its ``srdy``, a
combine after its ``crdy``; its core load then polls ``crdy`` with only its own
weights in flight. Only phase 1 differs between the cores; every stage from
``stage_o`` on is the same code.

Every hand-off is a tagged pair (5.10/5.11 of the design); the tag carries the
step epoch the runner bumps once a step and the launch's slot, so nothing is
zeroed between launches. Every poll waits on a stage earlier in this order on
some CTA of some rank, and every CTA is resident (one a CU), so the launch
cannot deadlock; a wait that gives up records itself and fails the rank
(``sync._spin``), and the launch still completes (every address is clamped).

LDS: the fixed ``red`` and ``RouteLds``, then one union of the stages'
structs (``lds_structs``); ``xq_ug`` is the sequential composite of
``load_xq``'s a1 rows and the ug epilogue's buffer, live together from
``load_xq`` to the last ug task. Every stage that touches the union or ``red``
ends with a barrier, so the next stage may reuse the bytes.

Adapted from the Kimi-K3 AMD specialized decode (vLLM #60683,
``kimi_k3/amd/specialized_decode/mono/layer.py``: ``build``, the kernel /
launcher pair, ``k2_launch``) with Muse's stages and static placement, the
context from ``common.context.make_context`` and the arguments packed by name.
"""

import os
from collections.abc import Mapping
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr
from flydsl.expr.typing import Int32, Int64

from vllm.models.muse_spark.amd.specialized_decode.mono import spec
from vllm.models.muse_spark.amd.specialized_decode.mono.attention import (
    back,
    combine,
    split,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.context import (
    lds_total_bytes,
    make_context,
    settle,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import traced
from vllm.models.muse_spark.amd.specialized_decode.mono.common.plan import (
    LDS_BYTES,
    THREADS,
    KernelAbi,
    key_tuple,
    source_digest,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.spec import (
    BLOCKS,
    G_TASKS,
    NA,
    P0,
    P_TASKS,
    R0,
    R_TASKS,
    RED_TASKS,
    K2Build,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.stages import (
    experts,
    pool,
    reduce,
    router,
    seam,
    unpool,
)

# The kernel's parameters: spec.K2_ARGS, but the residual stream is
# ``stream_in`` -- every ``@flyc.jit`` launcher takes the HIP stream as its last
# parameter ``stream`` (``KernelAbi.check``), and Python refuses a duplicate
# parameter name. ``k2_launch`` takes the K2_ARGS names; ``c["args"]`` keys them
# by K2_ARGS too (the stages read ``c["args"]["stream"]``).
KERNEL_NAMES = {"stream": "stream_in"}
# The kernel's parameters by K2Build.core (spec.K2_ARGS_BY_CORE, renamed as
# above): the cores ``build`` has a variant for.
ABI_BY_CORE = {
    core: KernelAbi(tuple(KERNEL_NAMES.get(n, n) for n in spec.K2_ARGS_BY_CORE[core]))
    for core in back.CORES
}
ABI = ABI_BY_CORE["tensor"]  # Phase A's
# Every .py under mono/: joins every build key (an edit to any stage or helper
# rebuilds, wherever FlyDSL's own source walk does not reach)
SOURCES = source_digest(".")
REGIONS = spec.REGIONS  # debug.region_ids order of the wait records

_STREAM = fx.Stream(None)


def lds_structs(s: int, core: str = "tensor") -> dict:
    """The stage union's members at ``s`` rows (design 5.13): stage key ->
    ``fx.struct`` (or the ``xq_ug`` composite of two). A split build adds its
    core's stages (DESIGN_BC 7.7); the others' union is Phase A's."""
    lds = {
        "back": back.lds_struct(s),
        "seam": seam.lds_struct(s),
        "pool": pool.lds_struct(s),
        "xq_ug": {"xq": pool.xq_lds_struct(s), "ug": experts.ug_lds_struct(s)},
        "down": experts.down_lds_struct(s),
        "reduce": reduce.lds_struct(s),
        "unpool": unpool.lds_struct(s),
    }
    if core == "split":
        lds[split.STAGE] = split.lds_struct(s)
        lds[combine.STAGE] = combine.lds_struct(s)
    return lds


def lds_bytes(s: int, core: str = "tensor") -> int:
    """K2's static LDS at ``s`` rows: ``red``, ``RouteLds``, the stage union
    (an upper bound of the ISA's group segment: fields no stage touches are
    dropped)."""
    return lds_total_bytes(lds_structs(s, core))


@traced
def split_roles(c) -> None:
    """A split build's core stages before any o_proj weight (DESIGN_BC 7.2):
    CTA b < n_units runs split unit b, CTA b >= C0 combine b - C0 (unit
    first when both). Neither polls nor publishes with a weight in flight
    (rule 8); a CTA with no role passes straight on to its weights. Every
    branch is CTA-uniform."""
    b = c["bid"]
    key = split.key_of(c)
    n, c0 = key.n_units(), key.c0()
    if const_expr(n < BLOCKS):
        if b < n:
            split.stage_split(c, b)
    else:  # every CTA holds a unit
        split.stage_split(c, b)
    if b >= c0:
        combine.stage_combine(c, fx.Int32(b) - c0)


@traced
def program(c, core: str = "tensor") -> None:
    """K2's stages on CTA ``c["bid"]`` in the design's program order (5.2).
    Every call is CTA-uniform (predicated on the CTA, never on the thread), so
    every thread reaches every barrier. ``core``: the build's (static)."""
    s = c["S"]
    b = c["bid"]
    b32 = fx.Int32(b)
    # -- phase 1: o_proj, both tasks' weights in flight before the first MFMA;
    # a split core's stages before them (DESIGN_BC 7.2), a gated core's own
    # loads (none for "tensor" and "split": DESIGN_BC 6.2)
    if const_expr(core == "split"):
        settle(c)  # the split and combine stages publish and poll
        split_roles(c)
    # (not ``pre``: the seam's, assigned under a traced ``if`` below, must not
    # become that branch's state)
    core_pre = back.core_loads(c, core)
    ops0 = back.o_loads(c, b)
    ops1 = back.o_loads(c, b + BLOCKS)
    if const_expr(core != "split"):
        settle(c)  # the context words' wait behind the core's and o_proj's loads
    back.load_core(c, core, core_pre)
    back.stage_o(c, b, ops0)
    back.stage_o(c, b + BLOCKS, ops1)
    # -- phase 2: the seam (and routing), the router rows, the pooler
    if b < NA:
        pre = seam.seam_loads(c, b)
        seam.stage_seam(c, b, pre)
        if b < s:
            router.stage_route(c, b)
    if (b >= R0) & (b < R0 + R_TASKS):
        rops = router.router_loads(c, b - R0)
        router.stage_router(c, b - R0, rops)
    if (b >= P0) & (b < P0 + P_TASKS):
        pops = pool.pool_loads(c, b - P0)
        pool.stage_pool(c, b - P0, pops)
    # -- phase 3: the a1 rows (every CTA), the union, ug, down
    pool.load_xq(c)
    nu = experts.load_route(c)
    experts.run_ug(c, nu)
    experts.run_down(c, b32, nu)
    # -- phase 4: owner reduce, unpooler (weights before the reduce's waits),
    # gather
    uops = unpool.unpool_loads(c, b)
    j = b32 - RED_TASKS * c["rank"]
    if (j >= 0) & (j < RED_TASKS):
        reduce.stage_owner(c, j)
    unpool.stage_unpool(c, b, uops)
    if b < G_TASKS:
        unpool.stage_gather(c, b)


_BUILDS: dict = {}
# FlyDSL's cache root, read when a launcher binds its cache (its first call)
CACHE_ENV = "FLYDSL_RUNTIME_CACHE_DIR"


def k2_cache_dir() -> Path:
    """Where the K2 builds are cached: FlyDSL's own default. Not the process's
    ``FLYDSL_RUNTIME_CACHE_DIR``, which importing aiter points at aiter's bundled
    kernels (read-only, or as root a directory of the image layer that a
    ``--rm`` container throws away)."""
    return Path.home() / ".flydsl" / "cache"


def _bind_cache(launch) -> None:
    """Binds ``launch``'s FlyDSL cache to ``k2_cache_dir`` now (FlyDSL binds it
    once, from the env, at a launcher's first call) and restores the env: aiter's
    FlyDSL kernels keep their bundled cache. Unbound (the env's choice) if the
    directory cannot be created."""
    bind = getattr(launch, "_ensure_cache_manager", None)
    path = k2_cache_dir()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    if bind is None:
        return
    old = os.environ.get(CACHE_ENV)
    os.environ[CACHE_ENV] = str(path)
    try:
        bind()
    finally:
        if old is None:
            del os.environ[CACHE_ENV]
        else:
            os.environ[CACHE_ENV] = old


def build(key: K2Build):
    """The ``@flyc.jit`` launcher of build ``key`` (``ABI_BY_CORE[key.core]``
    order, then ``stream``), cached. Every field of ``key`` and the mono
    sources are in the JIT key (``key_tuple``), referenced by both the kernel
    and the launcher. A core with no variant yet raises NotImplementedError
    (never another core's kernel under this key's name and ABI)."""
    if key in _BUILDS:
        return _BUILDS[key]
    if not isinstance(key, K2Build):
        raise TypeError(f"a K2 build key is a K2Build, not {type(key).__name__}")
    if key.core not in ABI_BY_CORE:
        raise NotImplementedError(
            f"K2 core {key.core!r}: no kernel variant yet (built: {tuple(ABI_BY_CORE)})"
        )
    s = key.tokens
    lds = lds_structs(s, key.core)
    nbytes = lds_bytes(s, key.core)
    if nbytes > LDS_BYTES:
        raise ValueError(f"K2 at {s} rows: {nbytes} B of LDS > {LDS_BYTES}")
    keyed = key_tuple(key, SOURCES)
    kernel, launch = _VARIANTS[key.core](key, s, lds, keyed)
    ABI_BY_CORE[key.core].check(kernel, launch)
    _bind_cache(launch)
    _BUILDS[key] = launch
    return launch


def _tensor(key: K2Build, s: int, lds: dict, keyed: tuple):
    """The "tensor" variant (Phase A): (kernel, launcher)."""

    @flyc.kernel(name=key.kernel_name(), known_block_size=[THREADS, 1, 1])
    def k2(
        core: Int64,
        stream_in: Int64,
        pad: Int64,
        w_o: Int64,
        attn_alpha: Int64,
        attn_beta: Int64,
        w_preffn: Int64,
        w_router: Int64,
        bias: Int64,
        w_pool: Int64,
        w_preexp: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        w_postexp: Int64,
        w_unpool: Int64,
        stream_out: Int64,
        branch_out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        slot: Int32,
        health: Int64,
        alarm: Int64,
    ):
        _ = keyed
        args = {
            "core": core,
            "stream": stream_in,
            "pad": pad,
            "w_o": w_o,
            "attn_alpha": attn_alpha,
            "attn_beta": attn_beta,
            "w_preffn": w_preffn,
            "w_router": w_router,
            "bias": bias,
            "w_pool": w_pool,
            "w_preexp": w_preexp,
            "w13": w13,
            "w13s": w13s,
            "w2": w2,
            "w2s": w2s,
            "w_postexp": w_postexp,
            "w_unpool": w_unpool,
            "stream_out": stream_out,
            "branch_out": branch_out,
            "scratch": scratch,
            "peers": peers,
            "rank": rank,
            "epoch": epoch,
            "slot": slot,
            "health": health,
            "alarm": alarm,
        }
        # a fault build's sync.stall is make_context's prologue
        c = make_context(s, args, lds, key=key)
        program(c)

    @flyc.jit
    def launch(
        core: Int64,
        stream_in: Int64,
        pad: Int64,
        w_o: Int64,
        attn_alpha: Int64,
        attn_beta: Int64,
        w_preffn: Int64,
        w_router: Int64,
        bias: Int64,
        w_pool: Int64,
        w_preexp: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        w_postexp: Int64,
        w_unpool: Int64,
        stream_out: Int64,
        branch_out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        slot: Int32,
        health: Int64,
        alarm: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k2(
            core,
            stream_in,
            pad,
            w_o,
            attn_alpha,
            attn_beta,
            w_preffn,
            w_router,
            bias,
            w_pool,
            w_preexp,
            w13,
            w13s,
            w2,
            w2s,
            w_postexp,
            w_unpool,
            stream_out,
            branch_out,
            scratch,
            peers,
            rank,
            epoch,
            slot,
            health,
            alarm,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return k2, launch


def _gated(key: K2Build, s: int, lds: dict, keyed: tuple):
    """The "gated" variant (Phase B2): ``attn`` and ``qkvo`` in place of
    ``core``, the gate in the core load. (kernel, launcher)."""

    @flyc.kernel(name=key.kernel_name(), known_block_size=[THREADS, 1, 1])
    def k2g(
        attn: Int64,
        qkvo: Int64,
        stream_in: Int64,
        pad: Int64,
        w_o: Int64,
        attn_alpha: Int64,
        attn_beta: Int64,
        w_preffn: Int64,
        w_router: Int64,
        bias: Int64,
        w_pool: Int64,
        w_preexp: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        w_postexp: Int64,
        w_unpool: Int64,
        stream_out: Int64,
        branch_out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        slot: Int32,
        health: Int64,
        alarm: Int64,
    ):
        _ = keyed
        args = {
            "attn": attn,
            "qkvo": qkvo,
            "stream": stream_in,
            "pad": pad,
            "w_o": w_o,
            "attn_alpha": attn_alpha,
            "attn_beta": attn_beta,
            "w_preffn": w_preffn,
            "w_router": w_router,
            "bias": bias,
            "w_pool": w_pool,
            "w_preexp": w_preexp,
            "w13": w13,
            "w13s": w13s,
            "w2": w2,
            "w2s": w2s,
            "w_postexp": w_postexp,
            "w_unpool": w_unpool,
            "stream_out": stream_out,
            "branch_out": branch_out,
            "scratch": scratch,
            "peers": peers,
            "rank": rank,
            "epoch": epoch,
            "slot": slot,
            "health": health,
            "alarm": alarm,
        }
        c = make_context(s, args, lds, key=key)
        program(c, "gated")

    @flyc.jit
    def launch(
        attn: Int64,
        qkvo: Int64,
        stream_in: Int64,
        pad: Int64,
        w_o: Int64,
        attn_alpha: Int64,
        attn_beta: Int64,
        w_preffn: Int64,
        w_router: Int64,
        bias: Int64,
        w_pool: Int64,
        w_preexp: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        w_postexp: Int64,
        w_unpool: Int64,
        stream_out: Int64,
        branch_out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        slot: Int32,
        health: Int64,
        alarm: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k2g(
            attn,
            qkvo,
            stream_in,
            pad,
            w_o,
            attn_alpha,
            attn_beta,
            w_preffn,
            w_router,
            bias,
            w_pool,
            w_preexp,
            w13,
            w13s,
            w2,
            w2s,
            w_postexp,
            w_unpool,
            stream_out,
            branch_out,
            scratch,
            peers,
            rank,
            epoch,
            slot,
            health,
            alarm,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return k2g, launch


def _split(key: K2Build, s: int, lds: dict, keyed: tuple):
    """The "split" variant (Phase C): ``qkvo``, the layer's KV cache and its
    KV group's block table and sequence lengths in place of ``core``, the
    split-KV core in-kernel (``split_roles``). (kernel, launcher)."""

    @flyc.kernel(name=key.kernel_name(), known_block_size=[THREADS, 1, 1])
    def k2s(
        qkvo: Int64,
        kv: Int64,
        kv_sb: Int64,
        kv_sh: Int64,
        kv_st: Int64,
        kv_blocks: Int32,
        block_table: Int64,
        bt_rows: Int32,
        bt_stride: Int32,
        seq_lens: Int64,
        stream_in: Int64,
        pad: Int64,
        w_o: Int64,
        attn_alpha: Int64,
        attn_beta: Int64,
        w_preffn: Int64,
        w_router: Int64,
        bias: Int64,
        w_pool: Int64,
        w_preexp: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        w_postexp: Int64,
        w_unpool: Int64,
        stream_out: Int64,
        branch_out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        slot: Int32,
        health: Int64,
        alarm: Int64,
    ):
        _ = keyed
        args = {
            "qkvo": qkvo,
            "kv": kv,
            "kv_sb": kv_sb,
            "kv_sh": kv_sh,
            "kv_st": kv_st,
            "kv_blocks": kv_blocks,
            "block_table": block_table,
            "bt_rows": bt_rows,
            "bt_stride": bt_stride,
            "seq_lens": seq_lens,
            "stream": stream_in,
            "pad": pad,
            "w_o": w_o,
            "attn_alpha": attn_alpha,
            "attn_beta": attn_beta,
            "w_preffn": w_preffn,
            "w_router": w_router,
            "bias": bias,
            "w_pool": w_pool,
            "w_preexp": w_preexp,
            "w13": w13,
            "w13s": w13s,
            "w2": w2,
            "w2s": w2s,
            "w_postexp": w_postexp,
            "w_unpool": w_unpool,
            "stream_out": stream_out,
            "branch_out": branch_out,
            "scratch": scratch,
            "peers": peers,
            "rank": rank,
            "epoch": epoch,
            "slot": slot,
            "health": health,
            "alarm": alarm,
        }
        c = make_context(s, args, lds, key=key)
        program(c, "split")

    @flyc.jit
    def launch(
        qkvo: Int64,
        kv: Int64,
        kv_sb: Int64,
        kv_sh: Int64,
        kv_st: Int64,
        kv_blocks: Int32,
        block_table: Int64,
        bt_rows: Int32,
        bt_stride: Int32,
        seq_lens: Int64,
        stream_in: Int64,
        pad: Int64,
        w_o: Int64,
        attn_alpha: Int64,
        attn_beta: Int64,
        w_preffn: Int64,
        w_router: Int64,
        bias: Int64,
        w_pool: Int64,
        w_preexp: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        w_postexp: Int64,
        w_unpool: Int64,
        stream_out: Int64,
        branch_out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        slot: Int32,
        health: Int64,
        alarm: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k2s(
            qkvo,
            kv,
            kv_sb,
            kv_sh,
            kv_st,
            kv_blocks,
            block_table,
            bt_rows,
            bt_stride,
            seq_lens,
            stream_in,
            pad,
            w_o,
            attn_alpha,
            attn_beta,
            w_preffn,
            w_router,
            bias,
            w_pool,
            w_preexp,
            w13,
            w13s,
            w2,
            w2s,
            w_postexp,
            w_unpool,
            stream_out,
            branch_out,
            scratch,
            peers,
            rank,
            epoch,
            slot,
            health,
            alarm,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return k2s, launch


# the kernel variant of each K2Build.core (``build``)
_VARIANTS = {"tensor": _tensor, "gated": _gated, "split": _split}
assert set(_VARIANTS) == set(ABI_BY_CORE)


def k2_launch(key: K2Build, args: Mapping[str, torch.Tensor | int]) -> None:
    """One K2 launch of build ``key`` on the current stream. ``args``: every
    ``spec.K2_ARGS_BY_CORE[key.core]`` name, nothing else
    (``ABI_BY_CORE[key.core].pack``); a tensor passes its address, an int
    (``rank``, ``slot``, ``alarm``) itself."""
    f = build(key)
    vals = {
        KERNEL_NAMES.get(n, n): v.data_ptr() if isinstance(v, torch.Tensor) else int(v)
        for n, v in args.items()
    }
    f(*ABI_BY_CORE[key.core].pack(vals), stream=torch.cuda.current_stream())
