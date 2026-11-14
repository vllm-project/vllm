# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The `moe` stage: the fused MoE engine (TP4, 1..8 rows, mode Q) on the post's
hand-off.

- one explicit `FusedMoeConfig`;
- CTA roles are blockIdx; the engine reads the full expert buffers vLLM holds;
- grid sizes from the live SM count: routed CTAs = SMs - shared CTAs, shared CTAs per
  (rows, SMs) from
  `geometry.moe_n_shared`; host-side residency checks fail closed;
- the boundary push goes into `_exchange.push_workspace(transport, bucket, "g3")`
  (peer pointers from
  `_comm`), never into an extension-private IPC allocation.

Call contract (all on the current stream, PDL-chained after the post):

    stage = FusedMoeStage(FusedMoeConfig(rows=8), weights, transport=tp,
    handoff=post.handoff,
                          co_resident=(post.grid_use(),))
    stage.warmup()                       # compiles; eager, before CUDA-graph capture
    stage(out=ffn_out, push=True)        # push=True when the next layer's front consumes the push
"""  # noqa: E501

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from .. import _arch, geometry
from .._exchange import bucket, push_workspace
from . import host as H
from .weights import FC2_SCALE_Q, MoeWeights, prep_mode_q

logger = logging.getLogger(__name__)

ROUTE_REPL = (
    16  # route-line replicas the post writes (engine CTA bx polls replica bx % 16)
)
ROUTE_LINE_U64 = 16
RCTR_STRIDE = 32
SEED_WEIGHT_BF16 = 0x3E80  # bf16(0.25): weight of the seed routes of the compile call
VALID_N_SHARED = (
    geometry.MOE_VALID_N_SHARED
)  # shared-expert CTA counts the kernel's shared tiling admits at I = 768


def default_n_shared(rows: int, sm_count: int) -> int:
    """Shared-expert CTAs for `rows` tokens on the live `sm_count`
    (`geometry.moe_n_shared`: a bytes-per-CTA
    model, or a measured override)."""
    return geometry.moe_n_shared(rows, sm_count)


@dataclass(frozen=True)
class FusedMoeConfig:
    rows: int  # tokens per call, 1..8 (= the post's MoeHandoff.rows)
    n_shared: int | None = None  # None: default_n_shared(rows, live SM count)
    ext_route: int = (
        1  # 1: the post's route lines (production); 2: a static route table (tests)
    )
    fc1_per_token: bool = (
        True  # mode Q (post's per-token s_t); False = static FC1 scale (tests only)
    )
    push: bool = True  # build the boundary push (the call still decides per launch)
    extra: tuple = field(
        default_factory=tuple
    )  # ((key, value), ...) engine-config overrides (experiments)

    def __post_init__(self):
        if not 1 <= self.rows <= 8:
            raise ValueError(f"moe: rows {self.rows} outside 1..8")
        if self.ext_route not in (1, 2):
            raise ValueError("moe: ext_route 1 (post route lines) or 2 (static table)")
        if self.n_shared is not None and self.n_shared not in VALID_N_SHARED:
            raise ValueError(f"moe: n_shared {self.n_shared} not in {VALID_N_SHARED}")


def engine_config(cfg: FusedMoeConfig) -> dict:
    """The engine's trace-time config for the TP4 mode-Q build."""
    t = cfg.rows
    c = {"sf2_w6": 1, "b2buf": 1, "stages": 7} if t > 4 else {}
    if (
        t > 2
        # sf2_w6 moves the FC2 scale gather to warp 8, which exists only above 2 rows
        # (more builder warps)
    ):
        c["sf2_w6"] = 1
    c.update(
        early_trig=1, sfa_bulk=1, sh_nst=4, sh_pf=0, roll_fc2=1, b2img=1, sh_trig=1
    )
    c.update(ext_route=cfg.ext_route, xstage=0, pad_x=1, b1_x4=1)
    if cfg.ext_route == 1:
        c.update(route_repl=ROUTE_REPL, route_poll_ns=64, route_maxt=t)
    if cfg.fc1_per_token:
        c["pt5"] = 1
    c.update(dict(cfg.extra))
    return c


class FusedMoeStage:
    """See the module docstring."""

    stage = "moe"

    def __init__(
        self,
        cfg: FusedMoeConfig,
        weights: MoeWeights,
        *,
        transport=None,
        handoff=None,
        rt_tab: torch.Tensor | None = None,
        x4: torch.Tensor | None = None,
        x4sf: torch.Tensor | None = None,
        co_resident: Sequence[_arch.GridUse] = (),
        device: torch.device | None = None,
    ):
        """`handoff`: the post's `allreduce_norm.MoeHandoff` (routes, x4, normed
        rows). Tests without a post pass
        `rt_tab` (ext_route=2) and `x4` / `x4sf` (uint8 `[16][3072]` / `[17][384]`,
        row 16 = fp32 s_t) instead."""
        self.cfg = cfg
        dev = torch.device(device) if device is not None else torch.device("cuda")
        if (
            dev.index is None
        ):  # "cuda" -> "cuda:<current>": the operand checks compare devices exactly
            dev = torch.device("cuda", torch.accelerator.current_device_index())
        self.device = dev
        self.spec = _arch.device_spec(dev.index)
        if cfg.fc1_per_token:
            weights = prep_mode_q(weights)
            if weights.fc2_enc != FC2_SCALE_Q:
                raise ValueError("moe: mode Q needs the FC2 scale-16 weight prep")
        weights.validate()
        self.weights = weights
        nsh = (
            cfg.n_shared
            if cfg.n_shared is not None
            else default_n_shared(cfg.rows, self.spec.sm_count)
        )
        self.n_shared = nsh
        self.num_ctas = self.spec.sm_count - nsh
        if self.num_ctas < H.MIN_CTAS:
            raise _arch.ResidencyError(
                f"moe: {self.spec.sm_count} SMs leave {self.num_ctas} routed CTAs "
                f"(< {H.MIN_CTAS}) beside {nsh} shared CTAs"
            )
        self.push_ws = None
        if cfg.push:
            if transport is None:
                raise ValueError("moe: push=True needs a PeerTransport")
            self.push_ws = push_workspace(transport, bucket(cfg.rows), "g3")
        self.state = H.M3MoeState(
            dev,
            cfg.rows,
            self.num_ctas,
            shared=True,
            n_shared=nsh,
            extra_cfg=engine_config(cfg),
            push_ws=self.push_ws,
        )
        self._check_residency(co_resident)
        self.handoff = handoff
        route_lines = rctr = None
        if cfg.ext_route == 1:
            if handoff is None:
                raise ValueError("moe: ext_route=1 needs the post's MoeHandoff")
            if handoff.rows != cfg.rows:
                raise ValueError(
                    f"moe: handoff rows {handoff.rows} != config rows {cfg.rows}"
                )
            route_lines, rctr = handoff.route, handoff.rctr
        elif rt_tab is None:
            raise ValueError("moe: ext_route=2 needs rt_tab")
        if handoff is not None:
            x4, x4sf = (
                handoff.x4_q,
                (handoff.x4_sf_full if cfg.fc1_per_token else handoff.x4_sf),
            )
        if x4 is None or x4sf is None:
            raise ValueError(
                "moe: the engine reads the post's NVFP4 FC1 input: pass handoff or x4 "
                "/ x4sf"
            )
        self._rt_tab = rt_tab
        w = weights
        self.layer = self.state.layer(  # type: ignore[misc]
            *w.expert_args(),
            router_weight=w.router,
            fc31_input_scale=w.a1_gscale,
            shared_weights=w.shared(),
            route_lines=route_lines,
            rt_tab=rt_tab,
            rctr=rctr,
            x4=x4,
            x4sf=x4sf,
        )
        self.arp = self._peers = None
        if self.push_ws is not None:
            self.arp, self._peers = H.push_arp(self.push_ws, dev)
        self._warm = False

    # ------------------------------------------------------------------ residency (fail
    # closed)
    def grid_use(self) -> _arch.GridUse:
        st = self.state
        # maxnreg (the b1_x4 builds' register cap) bounds the engine's registers; 0 =
        # uncapped, not counted
        return _arch.GridUse(
            "moe.engine",
            st.grid_ctas,
            st.smem_bytes_upper_bound,
            st.threads,
            tmem_columns=_arch.TMEM_COLUMNS,
            regs_per_thread=st.maxnreg,
        )

    def _check_residency(self, co_resident):
        st, spec = self.state, self.spec
        if st.grid_ctas != spec.sm_count:
            # one persistent CTA per SM: more would deadlock, fewer leaves SMs to a
            # co-running kernel's CTAs
            raise _arch.ResidencyError(
                f"moe: grid {st.grid_ctas} != {spec.sm_count} SMs (one CTA per SM)"
            )
        # The CuTe DSL build exposes no function handle for
        # cudaOccupancyMaxActiveBlocksPerMultiprocessor, so the occupancy is the
        # resource bound (smem, threads, all 512 TMEM columns, the maxnreg register
        # cap).
        occ = _arch.ctas_per_sm(
            spec,
            smem_bytes=st.smem_bytes_upper_bound,
            threads=st.threads,
            tmem_columns=_arch.TMEM_COLUMNS,
            regs_per_thread=st.maxnreg,
        )
        # every engine CTA polls the others (FC1 -> FC2 records, B images, shared-expert
        # h / y_sh)
        _arch.check_persistent_grid("moe.engine", st.grid_ctas, occ, spec)
        if self.cfg.ext_route == 1 and not co_resident:
            raise _arch.ResidencyError(
                "moe: ext_route=1 polls the post's route lines before its grid wait; "
                "pass "
                "the post's GridUse as co_resident so co-residency can be checked"
            )
        if co_resident:
            _arch.check_co_resident([self.grid_use(), *co_resident], spec)

    # ------------------------------------------------------------------ compile / run
    def _seed_publish(self):
        """The compile call consumes one route publish: align the post's row counters
        with the engine's epochs and
        publish experts 0..3 at weight 0.25 with tag = epoch + 1 (what the post's next
        publish would carry)."""
        st, hd = self.state, self.handoff
        ep = st.epochs[: st.num_ctas].cpu()
        if ep.numel() == 0 or not bool((ep == ep[0]).all()):
            raise RuntimeError(
                "moe: engine epochs not uniform before the compile call "
                f"{ep.unique().tolist()}"
            )
        tag = int(ep[0]) + 1
        t = self.cfg.rows
        ids = torch.arange(4, dtype=torch.int64, device=self.device)
        words = (tag << 32) | (ids << 16) | SEED_WEIGHT_BF16
        hd.route.zero_()
        hd.route[:, :t, :4] = words.view(1, 1, 4)
        hd.rctr.view(-1, RCTR_STRIDE)[:, 0] = tag
        return tag

    def warmup(self):
        """Compile with one eager call (no push). Must run before CUDA-graph capture,
        outside any PDL chain."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("moe: warmup during graph capture")
        tag = None
        if self.cfg.ext_route == 1:
            tag = self._seed_publish()
        hidden = (
            self.handoff.normed
            if self.handoff is not None
            else torch.zeros(
                self.cfg.rows, H.HIDDEN, dtype=torch.bfloat16, device=self.device
            )
        )
        out = torch.empty(
            self.cfg.rows, H.HIDDEN, dtype=torch.bfloat16, device=self.device
        )
        self.layer(hidden, out=out)
        torch.accelerator.synchronize(self.device)
        if tag is not None:
            st = self.state
            ep = st.epochs[: st.num_ctas].cpu()
            if not bool((ep == tag).all()):
                raise RuntimeError(
                    "moe: after the compile call the engine epochs "
                    f"{ep.unique().tolist()} != route "
                    f"tag {tag}; the next post publish would never be consumed"
                )
            rd = st.route_dbg[: self.cfg.rows].cpu()
            if not bool(rd[:, 4:8].any(dim=1).all()):
                raise RuntimeError(
                    "moe: the compile call fell back to the give-up route (no publish "
                    "seen)"
                )
        self._warm = True

    def __call__(
        self,
        hidden: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        push: bool = False,
        rt_tab: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One engine launch. `hidden`: bf16 [rows, 6144] (default: the hand-off's
        normed rows, the shared-expert
        input). `push`: also push the output into the boundary exchange (only when the
        next consumer polls it,
        e.g. the next layer's kernel front; an unconsumed push would collide with the
        next step's)."""
        if not self._warm:
            raise RuntimeError("moe: call warmup() first (it compiles)")
        if hidden is None:
            hidden = self.handoff.normed
        if push and self.arp is None:
            raise ValueError(
                "moe: push requested but the stage was built without a push workspace"
            )
        return self.layer(
            hidden, out=out, arp=self.arp if push else None, rt_tab=rt_tab
        )


__all__ = ["FusedMoeConfig", "FusedMoeStage", "default_n_shared", "engine_config"]
