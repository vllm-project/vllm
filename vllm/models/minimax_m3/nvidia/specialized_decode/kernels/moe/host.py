# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the fused MoE engine (kernel.py, 1..8 tokens): router, NVFP4 routed
experts, MXFP8 shared expert on
extra CTAs, combine and the push into the next layer, in one launch."""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
from typing import Any

import torch

HIDDEN = 6144
INTER = 768  # routed intermediate per TP rank at TP4 (MiniMax-M3: 3072 / 4)
NUM_EXPERTS = 128
SHARED_INTER = INTER  # the shared expert's intermediate per TP rank
SHARED_CTAS = 24  # the kernel's default shared-expert CTA count
TOP_K = 4
U1 = 2 * INTER // 32  # FC1 units per expert
# the router GEMV takes at most 2 experts per CTA (router weights held in registers)
MIN_CTAS = 64

_UNIT_BYTES = 32 * (HIDDEN // 2 + HIDDEN // 16)  # one FC1 unit's weight + scale bytes
_ROW2_BYTES = INTER // 2 + INTER // 16  # one FC2 weight row's bytes (values + scales)
_S1 = 6  # FC1 stages (items) per unit
_ITEM_BYTES = _UNIT_BYTES // _S1
_modules: dict[tuple, object] = {}
_compiled: dict[tuple, object] = {}
# trace-time options by token count 1..8 (index num_tokens - 1). ``sfa_cp``: 18 KB ring
# stages, the A scales copied by cp.async into the instruction layout (every token
# count).
SFA_CP_BY_M: tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1, 1)
# ``b2img``: FC1 -> FC2 hand-off through per-group global B images + release counters (7
# and 8 tokens).
B2IMG_BY_M: tuple[int, ...] = (0, 0, 0, 0, 0, 0, 1, 1)
# ``ar_pre``: the finalize's all-reduce push with its call parities / descriptor
# preloaded (all but 2 tokens).
AR_PRE_BY_M: tuple[int, ...] = (1, 0, 1, 1, 1, 1, 1, 1)
# Ring ``stages`` (0 = the kernel's default, the most that fit).
STAGES_BY_M: tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0, 6)
# ``hb_repl``: copies of the tagged FC1 -> FC2 records (applied when > 1); CTA bx polls
# copy bx % hb_repl.
HB_REPL_BY_M: tuple[int, ...] = (32, 32, 8, 8, 1, 1, 1, 1)
# ``lg_repl``: replicas of the router logit exchange at up to LG_REPL_MAX_TOKENS tokens.
LG_REPL = 32
LG_REPL_MAX_TOKENS = 4
# Byte-balanced FC1 ranges below one unit per CTA (:func:`_fine_ranges`).
FINE_PLAN = True
# phase lag: phase p streams FC1 of group p and FC2 of group p - M3_MOE_LAG (the
# kernel's default lag).
M3_MOE_LAG = 2

_ENGINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel.py")

# register cap of the b1_x4 builds (ptxas otherwise spills at M = 4 / 8; M = 1 needs
# 120)
B1X4_MAXNREG = 88


def _kernel_module(config: dict, path: str):
    """kernel.py executed once per configuration with `M3_CONFIG` injected (its
    constants are trace-time)."""
    key: tuple = (os.path.basename(path),) + tuple(sorted(config.items()))
    mod = _modules.get(key)
    if mod is None:
        name = (
            f"{__name__}_"
            + os.path.splitext(os.path.basename(path))[0]
            + "_"
            + "_".join(f"{k}{v}" for k, v in key[1:])
        )
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        mod.M3_CONFIG = dict(config)  # type: ignore[attr-defined]
        sys.modules[name] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        _modules[key] = mod
    return mod


def _view(t: torch.Tensor, align: int):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t.detach(), assumed_align=align)


@contextlib.contextmanager
def _maxnreg(n: int):
    """.maxnreg n on the kernel compiled inside (CuTe DSL has no launch knob for it).
    Patches a private CuTe DSL
    hook (`CuTeDSL._generate_kernel_attrs`); verify it still exists in the pinned
    nvidia-cutlass-dsl."""
    if not n:
        yield
        return
    from cutlass._mlir import ir
    from cutlass.cutlass_dsl import cutlass as _cd

    real = _cd.CuTeDSL._generate_kernel_attrs

    def gen(self, config):
        ret = real(self, config)
        ret["nvvm.maxnreg"] = ir.Attribute.parse(f"{int(n)} : i32")
        return ret

    _cd.CuTeDSL._generate_kernel_attrs = gen
    try:
        yield
    finally:
        _cd.CuTeDSL._generate_kernel_attrs = real


def is_supported(device: torch.device | None = None) -> tuple:
    if not torch.cuda.is_available():
        return False, "no CUDA"
    from .._arch import UnsupportedArch, device_spec

    idx = torch.device(device).index if device is not None else None
    try:
        device_spec(idx)
    except UnsupportedArch as err:
        return False, str(err)
    try:
        import cutlass  # noqa: F401
    except ImportError:
        return False, "CuTe DSL (nvidia-cutlass-dsl) is not installed"
    return True, ""


def push_arp(ws, device) -> tuple[torch.Tensor, torch.Tensor]:
    """The engine's `arp` descriptor for a push into a boundary exchange
    (`_exchange.PushWorkspace`):
    int64 [1, &peers[0..ndst-1], flags, rank, world, maxt, 0, 0], peers = every rank's
    exchange (unicast) or the
    exchange's multicast address (`ws.multicast`, ndst 1). Returns (arp, peers); keep
    `peers` alive as long
    as `arp`."""
    peers = torch.tensor(ws.push_ptrs(), dtype=torch.int64, device=device)
    arp = torch.tensor(
        [
            1,
            peers.data_ptr(),
            int(ws.flags.data_ptr()),
            ws.rank,
            ws.world,
            ws.maxt,
            0,
            0,
        ],
        dtype=torch.int64,
        device=device,
    )
    return arp, peers


def _shared_views(shared_weights, who):
    """(gate_up, its scales, down, its scales) of the shared expert as flat uint8 views,
    checked."""
    if shared_weights is None:
        raise ValueError(f"{who} (shared): shared_weights are required")
    gu_w, gu_sf, dn_w, dn_sf = shared_weights
    want = {
        "gate_up": (gu_w, (2 * SHARED_INTER, HIDDEN)),
        "down": (dn_w, (HIDDEN, SHARED_INTER)),
    }
    for name, (t, shape) in want.items():
        if (
            t.dtype != torch.float8_e4m3fn
            or tuple(t.shape) != shape
            or not t.is_contiguous()
        ):
            raise ValueError(
                f"{who} (shared): {name} weight must be contiguous e4m3 {shape}"
            )
    if (
        gu_sf.numel() != 2 * SHARED_INTER * HIDDEN // 32
        or dn_sf.numel() != HIDDEN * SHARED_INTER // 32
    ):
        raise ValueError(
            f"{who} (shared): the UE8M0 scales must be the swizzled [N, K / 32] buffers"
        )
    return (
        gu_w.view(torch.uint8).view(-1),
        gu_sf.view(torch.uint8).reshape(-1),
        dn_w.view(torch.uint8).view(-1),
        dn_sf.view(torch.uint8).reshape(-1),
    )


def _pretile_shared(gu8, gusf8, dn8, dnsf8, n_cta: int = 24) -> torch.Tensor:
    """The shared expert's weights as m3_moe's shared CTAs stream them (SH_PRETILE): per
    CTA s, per stage g, the 16 KB
    shared-memory image (128-byte swizzle: 16-byte chunk j of row r at j ^ (r % 8)) of
    the stage's 128-row MMA tile and
    its 512-byte instruction scale atom (row R's 4 codes at word 4 (R % 32) + R / 32).
    CTA s owns P = SHARED_INTER / 24
    = 16 positions in SPLIT = 128 / (2 P) = 4 K slices of GT = 48 / SPLIT = 12 k-tiles. Stages 0..GT-1: MMA row
    R = P O + i (O = 2 q + u: slice q, u 0 gate / 1 up) = weight row u SHARED_INTER + P
    s + i at k-tile GT q + g; then
    down tile j (rows 256 s + 128 j ..) at k-tile kt, j-major (SHARED_INTER / 128
    k-tiles each). Built once per
    layer."""  # noqa: E501
    dev = gu8.device
    H_, I_ = HIDDEN, SHARED_INTER
    P = I_ // n_cta
    split = 128 // (2 * P)
    gt = H_ // 128 // split
    dkt = I_ // 128
    gu = gu8.view(2 * I_, H_)
    dn = dn8.view(H_, I_)
    gkb, dkb = H_ // 32, I_ // 32

    def sf_bytes(sf, n, b, kbn):  # code of (row n, K block b), broadcast
        return sf[
            ((n >> 7) * (kbn // 4) + (b >> 2)) * 512
            + (n & 31) * 16
            + ((n >> 5) & 3) * 4
            + (b & 3)
        ]

    r = torch.arange(128, device=dev)
    q = r >> 5
    ri = r & 31
    octant = r // P
    chunk = torch.arange(8, device=dev)
    swz = chunk[None, :] ^ (
        r[:, None] & 7
    )  # source chunk of image chunk j: j ^ (r % 8)
    dt = H_ // n_cta // 128  # down tiles per shared CTA
    out = torch.empty(n_cta, gt + dt * dkt, 16384 + 512, dtype=torch.uint8, device=dev)
    for s in range(n_cta):
        # gate_up: MMA row R -> weight row, slice
        nrow = (octant & 1) * I_ + P * s + r % P
        sl = octant >> 1
        for g in range(gt):
            kt = sl * gt + g  # per row
            tile = gu[
                nrow[:, None],
                (kt[:, None] * 128 + torch.arange(128, device=dev)[None, :]),
            ]  # [128, 128]
            img = tile.view(128, 8, 16)[
                r[:, None], swz
            ]  # image chunk j = tile chunk j ^ (r % 8)
            out[s, g, :16384] = img.reshape(-1)
            kb = torch.arange(4, device=dev)
            codes = sf_bytes(
                gusf8, nrow[:, None], kt[:, None] * 4 + kb[None, :], gkb
            )  # [128, 4]
            atom = torch.empty(512, dtype=torch.uint8, device=dev)
            atom[(ri[:, None] * 16 + q[:, None] * 4 + kb[None, :]).reshape(-1)] = (
                codes.reshape(-1)
            )
            out[s, g, 16384:] = atom
        for gd in range(dt * dkt):
            jt, kt = gd // dkt, gd % dkt
            row0 = (H_ // n_cta) * s + 128 * jt
            tile = dn[row0 : row0 + 128, kt * 128 : (kt + 1) * 128]
            img = tile.reshape(128, 8, 16)[r[:, None], swz]
            out[s, gt + gd, :16384] = img.reshape(-1)
            out[s, gt + gd, 16384:] = dnsf8[
                ((row0 >> 7) * (dkb // 4) + kt) * 512 : ((row0 >> 7) * (dkb // 4) + kt)
                * 512
                + 512
            ]
    return out.view(-1)


# --------------------------------------------------------------------------------------
# m3_moe (kernel.py): 1..8 tokens per call, group-phased weight stream.
# --------------------------------------------------------------------------------------
# Pre-tiled shared weights per layer, keyed by the source buffers (which the engines
# read in place): every token-count state of a layer streams the same copy. An entry
# holds its sources, so their addresses are not reused while cached.
_PRETILED: dict[tuple, tuple] = {}


def _pretiled(gu8, gusf8, dn8, dnsf8, n_cta: int = SHARED_CTAS) -> torch.Tensor:
    key = (
        gu8.device,
        gu8.data_ptr(),
        gusf8.data_ptr(),
        dn8.data_ptr(),
        dnsf8.data_ptr(),
        n_cta,
    )
    hit = _PRETILED.get(key)
    if hit is None:
        hit = _PRETILED[key] = (
            (gu8, gusf8, dn8, dnsf8),
            _pretile_shared(gu8, gusf8, dn8, dnsf8, n_cta),
        )
    return hit[1]


def _fine_ranges(weights, total: int):
    """Contiguous FC1 item ranges [s_c, e_c) over the CTAs, sized as close to
    `weights` (ideal real-valued item
    counts summing to `total`) as the split-unit protocol allows: every unit (S1
    items) has at most one CTA boundary
    strictly inside it (its head and tail parts, summed by the tail's holder), so a
    range that starts inside a unit
    runs at least to that unit's end. Returns the item counts."""
    n = len(weights)
    items = [0] * n
    s = 0
    acc = 0.0
    for c in range(n):
        acc += weights[c]
        e = total if c == n - 1 else min(total, max(s, int(acc + 0.5)))
        if s % _S1 and e < total:  # started inside a unit: run to (at least) its end
            e = max(e, min(total, (s // _S1 + 1) * _S1))
        items[c] = e - s
        s = e
    assert s == total
    return items


def make_phase_plan(
    num_ctas: int,
    m_max: int,
    lag: int = 2,
    tail_w: float = 1.0,
    early0: bool = False,
    fine: bool = FINE_PLAN,
    nb2: int | None = None,
    units: int = U1,
    ioff: int = 0,
    boff: int = 0,
    zero_ok: bool = False,
):
    """The work split of `m3_moe` for each distinct-expert count G (4 .. 4 m_max): FC2
    32-row output blocks in
    contiguous per-CTA ranges (the same for every group), and per group p (slots 4 p ..)
    a range of its unit-major FC1
    stages per CTA, balancing the bytes each CTA streams in phase p (FC1 of group p plus
    FC2 of group p - lag, and the
    trailing FC2s with the last group). Ranges are empty or >= 6 stages (a unit is split
    over at most two CTAs); a
    group too small for every CTA gets whole units on the CTAs with the least FC2 work,
    or (`fine`) byte-balanced
    ranges of any length with at most one CTA boundary inside a unit
    (`_fine_ranges`). Returns int32
    [4 m_max + 1, num_ctas, m_max + 1] (per group i0 | i1 << 16, then b0 | nb << 16) and
    the largest block count."""
    gcap = TOP_K * m_max
    nb2 = nb2 or HIDDEN // 32
    plan = torch.zeros(gcap + 1, num_ctas, m_max + 1, dtype=torch.int32)
    nbs = [nb2 // num_ctas + (1 if c < nb2 % num_ctas else 0) for c in range(num_ctas)]
    b0s = [sum(nbs[:c]) for c in range(num_ctas)]
    blk_bytes = 32 * _ROW2_BYTES  # one block, one slot
    for g in range(TOP_K, gcap + 1):
        ng = (g + 3) // 4
        for c in range(num_ctas):
            plan[g, c, m_max] = (boff + b0s[c]) | (nbs[c] << 16)
        for p in range(ng):
            n_p = min(4, g - 4 * p)
            total = n_p * units * _S1
            fc2 = [0] * num_ctas
            for c in range(num_ctas):
                if p >= lag:
                    fc2[c] += nbs[c] * min(4, g - 4 * (p - lag)) * blk_bytes
                if p == ng - 1 and not (
                    early0 and p == 0 and m_max >= 2
                ):  # early0: group 0's ranges the same for every G
                    for q in range(max(0, ng - lag), ng):
                        fc2[c] += int(tail_w * nbs[c] * min(4, g - 4 * q) * blk_bytes)
            items = [0] * num_ctas
            if total >= _S1 * num_ctas:

                def counts_for(limit, fc2=fc2):
                    # zero_ok: a CTA whose FC2 share already exceeds the level takes no
                    # FC1 stages this phase (an empty range, as small groups use)
                    # instead of the 6-stage floor.
                    if zero_ok:
                        return [
                            v if v >= _S1 else 0
                            for v in ((limit - f) // _ITEM_BYTES for f in fc2)
                        ]
                    return [max(_S1, (limit - f) // _ITEM_BYTES) for f in fc2]

                lo, hi = 0, max(fc2) + total * _ITEM_BYTES
                while lo < hi:
                    mid = (lo + hi) // 2
                    if sum(counts_for(mid)) >= total:
                        hi = mid
                    else:
                        lo = mid + 1
                items = counts_for(lo)
                extra = sum(items) - total
                while extra > 0:
                    c = min(
                        (c for c in range(num_ctas) if items[c] > _S1),
                        key=lambda c: (-(fc2[c] + items[c] * _ITEM_BYTES), c),
                    )
                    items[c] -= 1
                    extra -= 1
            elif fine:
                # water-fill: CTA c streams fc2[c] + w_c item bytes, equal where w_c > 0
                lo_f, hi_f = 0.0, float(max(fc2) + total * _ITEM_BYTES)
                for _ in range(100):
                    mid = 0.5 * (lo_f + hi_f)  # type: ignore[assignment]
                    if sum(max(0.0, (mid - f) / _ITEM_BYTES) for f in fc2) >= total:
                        hi_f = mid
                    else:
                        lo_f = mid
                wts = [max(0.0, (hi_f - f) / _ITEM_BYTES) for f in fc2]
                sw = sum(wts)
                items = _fine_ranges([w * total / sw for w in wts], total)
            else:
                chosen = sorted(
                    sorted(range(num_ctas), key=lambda c: (fc2[c], c))[: total // _S1]
                )
                for c in chosen:
                    items[c] = _S1
            i0 = 0
            for c in range(num_ctas):
                if items[c]:
                    plan[g, c, p] = (ioff + i0) | ((ioff + i0 + items[c]) << 16)
                i0 += items[c]
            assert i0 == total
    return plan, max(nbs)


def build_config(
    num_tokens: int,
    num_ctas: int,
    nb_cap: int,
    shared: bool,
    n_shared: int,
    extra_cfg: dict,
    push_world: int = 0,
) -> dict:
    """The engine's trace-time config (`M3_CONFIG` of kernel.py) for one state;
    pure (no device), so the configurations can be traced and checked without a
    GPU."""
    if num_ctas < 1:
        raise ValueError(
            f"m3_moe: num_ctas {num_ctas} (live SMs - shared CTAs) must be >= 1"
        )
    config = {
        "m_max": num_tokens,
        "num_ctas": num_ctas,
        "nb2_cap": nb_cap,
        "fused_router": 1,
        "i_tp": INTER,
    }
    if HB_REPL_BY_M[num_tokens - 1] > 1:
        config["hb_repl"] = HB_REPL_BY_M[num_tokens - 1]
    if num_tokens <= LG_REPL_MAX_TOKENS:
        config["lg_repl"] = LG_REPL
    if shared:
        config["sh_i"] = SHARED_INTER
        config["shared"] = 1
        # the shared expert's SiLU exactly as vLLM's silu_and_mul_with_clamp (bit-exact
        # h)
        config["silu_vllm"] = 1
        if n_shared != SHARED_CTAS:
            config["nsh"] = n_shared
    config["sfa_cp"] = SFA_CP_BY_M[num_tokens - 1]
    config["b2img"] = B2IMG_BY_M[num_tokens - 1]
    config["ar_pre"] = AR_PRE_BY_M[num_tokens - 1]
    if STAGES_BY_M[num_tokens - 1] > 0:
        config["stages"] = STAGES_BY_M[num_tokens - 1]
    # router logits bit-identical to vLLM's ops.fp32_router_gemm (its K partition and
    # reduction order)
    config["router_vllm"] = 1
    if push_world:
        from .._exchange import CLUSTER

        # ar_nd: the push destinations (world unicast, or 1 = the multicast mapping)
        config.update(
            {"ar_v3": 1, "ar_cl": CLUSTER, "ar_nd": int(push_world), "ar_pre": 1}
        )
    config.update(extra_cfg)
    for k in ("ar_g3", "ar_in", "pf_next", "pf_fc2"):
        if config.get(k):
            raise ValueError(f"m3_moe: config {k} is not supported")
    return config


def nb2_cap(num_ctas: int) -> int:
    """FC2 output blocks per routed CTA (what make_phase_plan reports): ceil(H / 32 /
    num_ctas)."""
    return -(-(HIDDEN // 32) // num_ctas)


class M3MoeState:
    """Workspace + compiled kernel of `m3_moe` (1..8 tokens per call) on one device,
    shared by the layers that run
    on it in one stream order. `shared`: the shared expert runs on `n_shared`
    dedicated CTAs beside the
    `num_ctas` routed ones and the call returns bf16(shared + routed). `push_ws` (an
    `_exchange.PushWorkspace`):
    build the boundary push in (the call then pushes when given an `arp`). Build
    before CUDA-graph capture; the
    first call compiles."""

    fused_router = True

    def __init__(
        self,
        device: torch.device,
        num_tokens: int,
        num_ctas: int,
        shared: bool = True,
        n_shared: int = SHARED_CTAS,
        extra_cfg: dict | None = None,
        push_ws=None,
    ):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "M3MoeState allocates and compiles: build it before CUDA-graph capture"
            )
        if not 1 <= num_tokens <= 8:
            raise ValueError(f"m3_moe takes 1..8 tokens per call, not {num_tokens}")
        self.device = torch.device(device)
        ok, why = is_supported(self.device)
        if not ok:
            raise RuntimeError(f"m3_moe: {why}")
        from .._arch import device_spec

        sms = device_spec(self.device.index).sm_count
        self.n_shared = int(n_shared) if shared else 0
        self.num_ctas = int(num_ctas)
        if not self.num_ctas >= MIN_CTAS or self.num_ctas + self.n_shared > sms:
            raise ValueError(
                f"m3_moe: {self.num_ctas} routed + {self.n_shared} shared CTAs must be "
                f">= {MIN_CTAS} routed and fit "
                f"the {sms} SMs (one CTA per SM, all resident at once)"
            )
        self.m = num_tokens
        extra_cfg = dict(extra_cfg or {})
        plan, nb_cap = make_phase_plan(
            self.num_ctas,
            num_tokens,
            M3_MOE_LAG,
            0.0 if num_tokens == 1 else 1.0,
            True,
            zero_ok=bool(extra_cfg.get("bal")),
        )
        self.shared = shared
        self.push_ws = push_ws
        self.config = build_config(
            num_tokens,
            self.num_ctas,
            nb_cap,
            shared,
            self.n_shared,
            extra_cfg,
            push_world=int(push_ws.ndst) if push_ws is not None else 0,
        )
        # ext_route: routes from outside (1 = the post's tagged route lines, 2 = a
        # static table)
        self.ext_route = int(self.config.get("ext_route", 0))
        if self.ext_route and (
            int(self.config.get("xstage", 1)) or not int(self.config.get("pad_x", 0))
        ):
            raise ValueError("m3_moe ext_route: needs xstage=0 and pad_x=1")
        # b1_x4: FC1 input from the post's NVFP4 bytes (x4 / x4sf), read after the grid
        # wait
        self.b1_x4 = int(self.config.get("b1_x4", 0))
        if self.b1_x4 and not self.ext_route:
            raise ValueError(
                "m3_moe b1_x4: needs ext_route (the post that writes x4 also publishes "
                "the routes)"
            )
        if (
            self.b1_x4
        ):  # ptxas otherwise picks 80 registers and spills at M = 4; M = 1 needs 114
            self.config.setdefault("maxnreg", 120 if num_tokens == 1 else B1X4_MAXNREG)
        try:
            # maxnreg is a compile option, not a trace-time constant of kernel.py
            self.mod = mod = _kernel_module(
                {k: v for k, v in self.config.items() if k != "maxnreg"}, _ENGINE_PATH
            )
        except AssertionError as err:
            raise ValueError(
                f"m3_moe: num_tokens={num_tokens}, num_ctas={self.num_ctas} does not "
                "fit the shared "
                "memory / TMEM budget (fewer CTAs means more FC2 blocks per CTA): "
                f"{err}"
            ) from err
        kw = dict(device=self.device)
        self.plan = plan.to(self.device)
        self.partials = torch.full(
            (mod.G_CAP * mod.U1 * num_tokens * 32,), -1, dtype=torch.int32, **kw
        )
        hb_words = mod.G_CAP * num_tokens * mod.U1 * 4
        if mod.HB_REPL > 1:
            hb_words = mod.HB_REPL * mod.HB_STRIDE
        self.hbuf32 = torch.zeros(hb_words, dtype=torch.int32, **kw)
        self.epochs = torch.zeros(self.num_ctas + mod.NSH, dtype=torch.int32, **kw)
        self.route_dbg = torch.zeros(num_tokens, 8, dtype=torch.int32, **kw)
        self.tdbg = torch.zeros(
            self.num_ctas + mod.NSH, mod.TD_STRIDE, dtype=torch.int64, **kw
        )
        self.lgbuf = torch.zeros(mod.LG_REPL * mod.LG_STRIDE, dtype=torch.int64, **kw)
        self._dummy = torch.zeros(4, dtype=torch.int32, **kw)
        self._ar_off = torch.zeros(8, dtype=torch.int64, **kw)
        self.img32 = torch.zeros(
            max(1, mod.NG_CAP * mod.IMG_BYTES // 4), dtype=torch.int32, **kw
        )
        self.cnt = torch.zeros(mod.B2CNT_WORDS, dtype=torch.int32, **kw)
        # shared expert: h and y_sh as (call tag << 16 | bf16), tag 0 = never written
        self.hx = torch.zeros(num_tokens * SHARED_INTER, dtype=torch.int32, **kw)
        self.dnx = torch.zeros(num_tokens * HIDDEN, dtype=torch.int32, **kw)
        self._fn: Any = None

    @property
    def grid_ctas(self) -> int:
        return self.num_ctas + int(self.mod.NSH)

    @property
    def smem_bytes_upper_bound(self) -> int:
        """Upper bound of the kernel's shared memory per CTA (the ring + fixed operands
        + the small-array reserve)."""
        m = self.mod
        return int(m.STAGES * m._STAGE_SMEM + m._FIXED_SMEM + (128 if m.SHARED else 0))

    @property
    def threads(self) -> int:
        return int(self.mod.THREADS)

    @property
    def maxnreg(self) -> int:
        """The register cap per thread the kernel is compiled with (0: ptxas
        chooses)."""
        return int(self.config.get("maxnreg", 0))

    def layer(
        self,
        w3_w1_weight,
        w3_w1_weight_scale,
        w2_weight,
        w2_weight_scale,
        fc31_alpha,
        fc31_scale_c,
        act_alpha,
        act_beta,
        act_clamp,
        fc2_alpha,
        e_score_correction_bias,
        router_weight=None,
        fc31_input_scale=None,
        shared_weights=None,
        route_lines=None,
        rt_tab=None,
        rctr=None,
        x4=None,
        x4sf=None,
    ) -> M3MoeLayer:
        """One MoE layer's operands (the TRT-LLM-layout NVFP4 expert buffers, read in
        place, never copied).
        `shared_weights` (a `shared` state): (gate_up e4m3 [2 I, 6144], its swizzled
        UE8M0 scales, down e4m3
        [6144, I], its swizzled UE8M0 scales). ext_route=1: `route_lines` int64
        `[route_repl][route_maxt][16]` and
        `rctr` int32 [>= M * 32]; ext_route=2: `rt_tab` int32 `[M][4]`. b1_x4=1:
        `x4` uint8 `[>= M][3072]` /
        `x4sf` uint8 `[max(M, 16) + 1][384]` (fc1_per_token: row max(M, 16) = fp32
        s_t)."""
        layer = M3MoeLayer(
            self,
            w3_w1_weight,
            w3_w1_weight_scale,
            w2_weight,
            w2_weight_scale,
            fc31_alpha,
            fc31_scale_c,
            act_alpha,
            act_beta,
            act_clamp,
            fc2_alpha,
            e_score_correction_bias,
            router_weight,
            fc31_input_scale,
        )
        layer.route_lines, layer.rt_tab, layer.rctr = route_lines, rt_tab, rctr
        if route_lines is not None or rt_tab is not None or rctr is not None:
            self._ext_args(route_lines, rt_tab, rctr)
        layer.x4, layer.x4sf = x4, x4sf
        if x4 is not None or x4sf is not None:
            self._x4_args(x4, x4sf)
        if self.shared:
            layer.shared_weights = _shared_views(shared_weights, "m3_moe")
            layer.shared_mxfp8 = tuple(shared_weights)
            if self.mod.SH_PRETILE:
                gu8, _, dn8, _ = layer.shared_weights
                layer.shared_weights = (
                    gu8,
                    _pretiled(*layer.shared_weights, self.n_shared),  # type: ignore[call-arg]
                    dn8,
                    self._dummy.view(torch.uint8),
                )
        return layer

    def _ext_args(self, route_lines, rt_tab, rctr=None):
        """(lgbuf slot, x32 slot) for ext_route: route_lines ride in lgbuf's slot
        (unused without the router GEMV),
        rt_tab (ext_route=2) or the post's row counters rctr (ext_route=1) in x32's
        (unused with the fused router)."""
        mod = self.mod
        if not self.ext_route:
            if route_lines is not None or rt_tab is not None or rctr is not None:
                raise ValueError(
                    "m3_moe: route_lines / rt_tab / rctr need a state built with "
                    "ext_route=1 / 2"
                )
            return self.lgbuf, self._dummy
        if self.ext_route == 1:
            want = (mod.ROUTE_REPL, mod.ROUTE_MAXT, 16)
            if (
                route_lines is None
                or route_lines.dtype != torch.int64
                or not route_lines.is_contiguous()
                or route_lines.numel() != want[0] * want[1] * want[2]
                or route_lines.device != self.device
            ):
                got = (
                    None
                    if route_lines is None
                    else (route_lines.dtype, tuple(route_lines.shape))
                )
                raise ValueError(
                    "m3_moe ext_route=1: route_lines must be contiguous int64 "
                    f"{list(want)} "
                    f"(cfg route_repl / route_maxt) on {self.device}, got {got}"
                )
            if route_lines.data_ptr() % 16:
                raise ValueError(
                    "m3_moe ext_route=1: route_lines must be 16-byte aligned"
                )
            if (
                rctr is None
                or rctr.dtype != torch.int32
                or not rctr.is_contiguous()
                or rctr.numel() < self.m * 32
                or rctr.device != self.device
                or rctr.data_ptr() % 16
            ):
                raise ValueError(
                    "m3_moe ext_route=1: rctr (the post's row counters) must be "
                    "contiguous "
                    f"16-byte aligned int32 with >= {self.m * 32} words on "
                    f"{self.device}"
                )
            return route_lines.view(-1), rctr.view(-1)[: self.m * 32]
        if rctr is not None:
            raise ValueError(
                "m3_moe: rctr is the ext_route=1 post check (no route lines at "
                "ext_route=2)"
            )
        if (
            rt_tab is None
            or rt_tab.dtype != torch.int32
            or not rt_tab.is_contiguous()
            or rt_tab.numel() < self.m * TOP_K
            or rt_tab.device != self.device
            or (rt_tab.dim() == 2 and rt_tab.shape[1] != TOP_K)
            or rt_tab.dim() > 2
        ):
            raise ValueError(
                f"m3_moe ext_route=2: rt_tab must be contiguous int32 [>= {self.m}][4] "
                f"on {self.device}"
            )
        if rt_tab.data_ptr() % 16:
            raise ValueError("m3_moe ext_route=2: rt_tab must be 16-byte aligned")
        return self.lgbuf, rt_tab.view(-1)[: self.m * TOP_K]

    def _x4_args(self, x4, x4sf):
        """(logits slot, xsf32 slot): b1_x4 builds read the post's x4 values / scales
        there."""
        if not self.b1_x4:
            if x4 is not None or x4sf is not None:
                raise ValueError("m3_moe: x4 / x4sf need a state built with b1_x4=1")
            return self._dummy.view(torch.float32), self._dummy
        for nm, t, w in (("x4", x4, HIDDEN // 2), ("x4sf", x4sf, HIDDEN // 16)):
            if (
                t is None
                or t.dtype != torch.uint8
                or t.dim() != 2
                or t.shape[1] != w
                or t.shape[0] < self.m
                or not t.is_contiguous()
                or t.device != self.device
                or t.data_ptr() % 16
            ):
                got = (
                    None if t is None else (t.dtype, tuple(t.shape), t.data_ptr() % 16)
                )
                raise ValueError(
                    f"m3_moe b1_x4: {nm} must be contiguous 16-byte aligned uint8 [>= "
                    f"{self.m}][{w}] "
                    f"on {self.device}, got {got}"
                )
        if self.mod.FC1_PER_TOKEN:  # s_t rides in row max(m, 16) of the scale buffer
            x4r = max(self.m, 16)
            if x4sf.shape[0] != x4r + 1:
                raise ValueError(
                    f"m3_moe fc1_per_token: x4sf must be [{x4r + 1}][{HIDDEN // 16}] "
                    f"(row {x4r} = fp32 "
                    f"s_t), got {tuple(x4sf.shape)}"
                )
            return x4.view(-1)[: self.m * (HIDDEN // 2)].view(torch.int32), x4sf.view(
                -1
            ).view(torch.int32)
        return (
            x4.view(-1)[: self.m * (HIDDEN // 2)].view(torch.int32),
            x4sf.view(-1)[: self.m * (HIDDEN // 16)].view(torch.int32),
        )

    def run(
        self,
        layer,
        hidden,
        out=None,
        arp=None,
        route_lines=None,
        rt_tab=None,
        rctr=None,
        x4=None,
        x4sf=None,
    ) -> torch.Tensor:
        m = self.m
        if (
            hidden is None
            or hidden.dtype != torch.bfloat16
            or tuple(hidden.shape) != (m, HIDDEN)
            or not hidden.is_contiguous()
        ):
            raise ValueError(
                f"m3_moe (fused router): expects contiguous bf16 hidden states [{m}, "
                f"{HIDDEN}]"
            )
        if arp is not None and self.push_ws is None:
            raise ValueError(
                "m3_moe: arp given to a state built without a push workspace"
            )
        lgb, xq = self.lgbuf, self._dummy
        if self.ext_route:
            lgb, xq = self._ext_args(
                layer.route_lines if route_lines is None else route_lines,
                layer.rt_tab if rt_tab is None else rt_tab,
                layer.rctr if rctr is None else rctr,
            )
        elif route_lines is not None or rt_tab is not None or rctr is not None:
            self._ext_args(route_lines, rt_tab, rctr)
        fz = (
            hidden.view(-1).view(torch.int32),
            layer.router_weight.view(-1),
            layer.fc31_input_scale.view(-1),
            lgb,
        )
        lgv, xs = self._x4_args(
            layer.x4 if x4 is None else x4, layer.x4sf if x4sf is None else x4sf
        )
        if out is None:
            out = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device=self.device)
        sh = (
            layer.shared_weights
            if self.shared
            else (self._dummy.view(torch.uint8),) * 4
        )
        args = (
            layer.w3_w1_weight,
            layer.w3_w1_weight_scale.view(-1),
            layer.w2_weight,
            layer.w2_weight_scale.view(-1),
            lgv,
            layer.bias,
            xq,
            xs,
            layer.escal,
            self.plan.view(-1),
            self.partials,
            self.hbuf32,
            self.epochs,
            out.view(-1),
            self.route_dbg.view(-1),
            self.tdbg.view(-1),
            *fz,
            *sh,
            self.hx,
            self.dnx,
            self._ar_off if arp is None else arp,
            layer.sfa1_img
            if layer.sfa1_img is not None
            else layer.w3_w1_weight_scale.reshape(-1).view(torch.int32),
            layer.w2_weight_scale.reshape(-1).view(torch.int32),
            self.img32,
            self.cnt,
        )
        if self._fn is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "m3_moe compiles on its first call: call it once before capture"
                )
            import cutlass.cute as cute

            # 16-byte aligned views except epochs (index 12) and the FC1 input scale
            # (index 18)
            aligns = [4 if i in (12, 18) else 16 for i in range(len(args))]
            sig = [_view(t, a) for t, a in zip(args, aligns)]
            key = (_ENGINE_PATH,) + tuple(sorted(self.config.items()))
            fn = _compiled.get(key)
            if fn is None:
                with _maxnreg(int(self.config.get("maxnreg", 0))):
                    fn = _compiled[key] = cute.compile(
                        self.mod.m3_moe,
                        *sig,
                        cute.runtime.make_fake_stream(),
                        options="--enable-tvm-ffi",
                    )
            self._fn = fn
        self._fn(*args, torch.cuda.current_stream(self.device).cuda_stream)
        return out


class M3MoeLayer:
    """One MoE layer on a `M3MoeState`: `layer(hidden)`. The expert buffers are
    read in place, never copied;
    the per-expert scalars are packed once (layer constants: rebuild the layer after
    reloading weights)."""

    shared_weights: Any = None
    shared_mxfp8: Any = None
    route_lines: Any = None
    rt_tab: Any = None
    rctr: Any = None
    x4: Any = None
    x4sf: Any = None

    def __init__(
        self,
        state,
        w3_w1_weight,
        w3_w1_weight_scale,
        w2_weight,
        w2_weight_scale,
        fc31_alpha,
        fc31_scale_c,
        act_alpha,
        act_beta,
        act_clamp,
        fc2_alpha,
        bias,
        router_weight=None,
        fc31_input_scale=None,
    ):
        e = NUM_EXPERTS
        expected = {
            "w3_w1_weight": (w3_w1_weight, (e, 2 * INTER, HIDDEN // 2)),
            "w3_w1_weight_scale": (w3_w1_weight_scale, (e, 2 * INTER, HIDDEN // 16)),
            "w2_weight": (w2_weight, (e, HIDDEN, INTER // 2)),
            "w2_weight_scale": (w2_weight_scale, (e, HIDDEN, INTER // 16)),
        }
        for name, (t, shape) in expected.items():
            if (
                t.element_size() != 1
                or tuple(t.shape) != shape
                or not t.is_contiguous()
            ):
                raise ValueError(
                    f"m3_moe: {name} is {t.dtype} {tuple(t.shape)}, expected "
                    f"contiguous 1-byte {shape}"
                )
        self.state = state
        self.w3_w1_weight = w3_w1_weight.view(torch.uint8)
        self.w3_w1_weight_scale = w3_w1_weight_scale.view(torch.uint8)
        self.w2_weight = w2_weight.view(torch.uint8)
        self.w2_weight_scale = w2_weight_scale.view(torch.uint8)
        self.escal = torch.stack(
            [fc31_alpha, fc31_scale_c, act_alpha, act_beta, act_clamp, fc2_alpha]
        ).float()
        self.escal = self.escal.reshape(6, e).contiguous()
        self.bias = bias.float().contiguous()
        if (
            router_weight is None
            or fc31_input_scale is None
            or router_weight.dtype != torch.float32
            or tuple(router_weight.shape) != (e, HIDDEN)
            or not router_weight.is_contiguous()
            or fc31_input_scale.dtype != torch.float32
            or fc31_input_scale.numel() != 1
        ):
            raise ValueError(
                "m3_moe (fused router): needs router_weight f32 [128, 6144] and "
                "fc31_input_scale f32 [1]"
            )
        self.router_weight = router_weight
        self.fc31_input_scale = fc31_input_scale
        # SFA_BULK: the FC1 scale words pre-spliced into the instruction-atom order, 2
        # KB per (16-atom raw group,
        # unit % 4), so a stage's scales are one bulk copy
        self.sfa1_img = None
        if state.config.get("sfa_bulk"):
            r = (
                self.w3_w1_weight_scale.reshape(-1)
                .view(torch.int32)
                .view(-1, 4, 4, 32, 4)
            )  # [g, q, kb, lane, cm]
            self.sfa1_img = (
                r.permute(0, 4, 2, 3, 1).contiguous().view(-1)
            )  # [g, cm, kb, lane, q]

    def __call__(
        self,
        hidden,
        out=None,
        arp=None,
        route_lines=None,
        rt_tab=None,
        rctr=None,
        x4=None,
        x4sf=None,
    ):
        return self.state.run(
            self,
            hidden,
            out,
            arp=arp,
            route_lines=route_lines,
            rt_tab=rt_tab,
            rctr=rctr,
            x4=x4,
            x4sf=x4sf,
        )
