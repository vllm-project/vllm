# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode glue: one persistent launch from the unreduced o_proj partial to the MoE
inputs, then AITER's A16W4 gemm1 / gemm2 on its sort, the shared expert on its
MXFP8 rows and the final TP all-reduce. The launch is AITER's fused op
(fused_all_reduce_rmsnorm_moe_route, Gluon) when the installed AITER has it, else
the FlyDSL kernel in kernel.py.

Replaces, per MoE layer of a pure-decode step: the post-attention all-reduce, the
residual add, the Gemma RMSNorm (+ MXFP8 copy), the router, the top-k (x routed
scale), moe_sorting and the moe_buf zeroing.

Host state (built once, collectively, outside graph capture): one peer buffer and
one device epoch shared by every width and layer, and per width a compiled kernel,
its scratch and persistent outputs (graph replays bake their addresses). Layers
differ only in the weight pointers passed per launch. The epoch moves on the
device; every rank runs the same launches, so all ranks agree on it."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm.logger import init_logger

from . import kernel as K
from .carry import Mxfp8Carry
from .shared import SharedWeights, shared_decode

logger = init_logger(__name__)

# Row counts of the compiled FlyDSL kernels (multiples of 4 in [17, 192]); a
# pure-decode step of another width takes the stock path. Measured on MI355X (TP4):
# the FlyDSL glue beats the stock all-reduce .. sort chain at 24-192 rows (DSpark
# 7: 3-24 requests).
WIDTHS = tuple(range(24, 193, 8))
# AITER's fused op runs every pure-decode width from 1 row up to this many: measured
# on MI355X (TP4) it beats the stock chain at every width in that range and the
# FlyDSL kernel at every width of WIDTHS.
AITER_MAX_ROWS = 256
# The glue's cross-GPU transport: plain payloads into sentinel-armed inboxes.
TRANSPORT = "lamport"

# shared-expert stream policy (measured, TP4 MI355X): forked to the side stream
# up to this many rows; in-stream ahead of gemm1 above (the side stream loses
# 15-17 us at 96-128 rows)
SIDE_STREAM_MAX_ROWS = 64
FAULT_MESSAGE = "MiniMax-M3 decode glue: error word set (a peer wait timed out)"
POST_FAULT_MESSAGE = (
    "MiniMax-M3 decode glue: the post-MoE all-reduce + norm error record is set (a "
    "peer wait timed out)"
)
# The all-reduce after the MoE fused with the next layer's residual add, Gemma
# RMSNorm and MXFP8 quant (AITER's fused_all_reduce_rmsnorm_mxfp8, one launch)
# for decode steps of up to this many rows, the op's limit.
POST_MOE_MAX_ROWS = 256


def _writable_flydsl_cache() -> None:
    """AITER points FlyDSL at its bundled read-only cache: as the upstream mono
    runner, switch to a writable one for the process (here under ~/.cache, which
    run_agentx.sh mounts, so compiles persist across runs)."""
    cur = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if cur and os.access(cur, os.W_OK):
        return
    path = Path.home() / ".cache" / "flydsl"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(path)


@dataclass
class _Width:
    s: int
    compiled: Any
    scratch: torch.Tensor
    res_out: torch.Tensor  # [S, H] bf16: the residual after attention
    xn: torch.Tensor  # [S, H] bf16: normed rows (gemm1 input)
    xq: torch.Tensor  # [S, H] fp8: normed rows, MXFP8 (shared expert input)
    xsc: torch.Tensor  # [S, H / 32] uint8 E8M0
    ids: torch.Tensor  # [S, 4] int32 top-k ids
    w: torch.Tensor  # [S, 4] fp32 top-k weights, x routed scale
    sids: torch.Tensor  # moe_sorting outputs (block 16)
    sw: torch.Tensor
    seids: torch.Tensor
    nv: torch.Tensor
    moe_buf: torch.Tensor  # [S, H] bf16, zeroed by the kernel; gemm2 adds into it
    # [S, I/TP] bf16: the fused shared expert's stage-1 rows
    sh_inter: torch.Tensor | None = None
    # the AITER op's outputs (views of the above), by its names
    out: dict | None = None


class DecodeGlue:
    def __init__(
        self,
        widths: tuple[int, ...],
        group,
        device: torch.device,
        shared_inter: int = 0,
    ) -> None:
        self.widths = tuple(widths)
        # > 0: the shared expert runs fused in the MoE kernels
        self.shared_inter = shared_inter
        self.group = group
        self.rank = group.rank_in_group
        self.device = device
        self.bufs: dict[int, _Width] = {}
        self.peer: Any = None  # PeerBuffer, once set up
        self.fault_word: torch.Tensor | None = None
        self.transport = TRANSPORT
        # AITER's fused all-reduce + norm + MXFP8 buffers (setup_post_moe)
        self.post_ar: Any = None
        self._post_err: torch.Tensor | None = None
        # AITER's fused all-reduce .. moe_sorting op (setup), when installed
        self.fused: Any = None

    # ------------------------------------------------------------ setup
    def setup(self) -> None:
        """Collective (the peer-buffer handshake): every TP rank, same point."""
        if self._setup_fused():
            return
        self._setup_flydsl()

    def _setup_fused(self) -> bool:
        """AITER's fused op for every width 1 .. AITER_MAX_ROWS, if every rank has
        it; else False (the FlyDSL kernel runs)."""
        from vllm.models.deepseek_v41.amd.mono.common.peer_memory import tp_agree

        cpu = self.group.cpu_group
        ok, err = True, None
        try:
            from aiter.ops.triton.comms.fused.all_reduce_rmsnorm_moe_route import (
                AllReduceRmsnormMoeRoute,
                fused_all_reduce_rmsnorm_moe_route,  # noqa: F401
            )
        except Exception as e:  # noqa: BLE001 - an AITER without the op
            ok, err = False, e
        if not tp_agree(ok, cpu):
            logger.info(
                "MiniMax-M3 decode glue: AITER fused op unavailable (%s); FlyDSL "
                "kernel",
                err or "refused on another rank",
            )
            return False
        comm: Any = None
        try:  # collective
            comm = AllReduceRmsnormMoeRoute(
                cpu, AITER_MAX_ROWS, K.HIDDEN, K.EXPERTS, K.TOPK, router_fp32=True
            )
        except Exception as e:  # noqa: BLE001
            ok, err = False, e
        if not tp_agree(ok, cpu):
            logger.warning(
                "MiniMax-M3 decode glue: AITER fused op setup failed on %s rank: %s",
                "this" if err else "another",
                err,
            )
            if comm is not None:
                comm.close()
            return False
        self.fused = comm
        self.widths = tuple(range(1, AITER_MAX_ROWS + 1))
        full = self._alloc(AITER_MAX_ROWS)
        for s in self.widths:
            self.bufs[s] = self._view(full, s)
        self.fault_word = _error_record_view(comm)
        self._streams()
        # compile every variant now, outside graph capture (collective launches)
        dev = self.device
        nw = torch.zeros(K.HIDDEN, dtype=torch.bfloat16, device=dev)
        gw = torch.zeros(K.EXPERTS, K.HIDDEN, dtype=torch.float32, device=dev)
        bias = torch.zeros(K.EXPERTS, dtype=torch.float32, device=dev)
        for s in comm.warmup_tokens():
            x = torch.zeros(s, K.HIDDEN, dtype=torch.bfloat16, device=dev)
            self._fused_call(self.bufs[s], x, x, nw, gw, bias)
        torch.accelerator.synchronize()
        logger.info(
            "MiniMax-M3 decode glue: AITER fused op, widths 1-%d", AITER_MAX_ROWS
        )
        return True

    def _view(self, full: _Width, s: int) -> _Width:
        """Width s on the first rows of the largest width's buffers (widths never
        run at the same time; graphs keep the views' addresses)."""
        mp = K.max_pad(s)
        b = _Width(
            s=s,
            compiled=None,
            scratch=full.scratch,
            res_out=full.res_out[:s],
            xn=full.xn[:s],
            xq=full.xq[:s],
            xsc=full.xsc[:s],
            ids=full.ids[:s],
            w=full.w[:s],
            sids=full.sids[:mp],
            sw=full.sw[:mp],
            seids=full.seids[: mp // 16],
            nv=full.nv,
            moe_buf=full.moe_buf[:s],
            sh_inter=full.sh_inter[:s] if full.sh_inter is not None else None,
        )
        b.out = {
            "residual_out": b.res_out,
            "normed": b.xn,
            "x_q": b.xq,
            "x_scale": b.xsc,
            "topk_ids": b.ids,
            "topk_weights": b.w,
            "sorted_ids": b.sids,
            "sorted_weights": b.sw,
            "sorted_expert_ids": b.seids,
            "num_valid_ids": b.nv,
            "moe_buf": b.moe_buf,
        }
        return b

    def _fused_call(self, b: _Width, part, residual, norm_w, router_w, bias) -> None:
        from aiter.ops.triton.comms.fused.all_reduce_rmsnorm_moe_route import (
            fused_all_reduce_rmsnorm_moe_route,
        )

        fused_all_reduce_rmsnorm_moe_route(
            part,
            residual,
            norm_w,
            K.EPS,
            router_w,
            self.fused,
            correction_bias=bias,
            scoring="sigmoid",
            renormalize=True,
            routed_scaling_factor=K.ROUTED_SCALE,
            gemma=True,
            quant=True,
            block_size=K.SORT_BLOCK,
            out=b.out,
        )

    def _streams(self) -> None:
        self.stream = None
        if not _side_stream_disabled():
            from vllm.utils.torch_utils import aux_stream

            self.stream = aux_stream()
        self.ev_fork, self.ev_join = torch.cuda.Event(), torch.cuda.Event()

    def _setup_flydsl(self) -> None:
        import flydsl.compiler as flyc

        from vllm.models.deepseek_v41.amd.mono.common.peer_memory import PeerBuffer
        from vllm.platforms import current_platform

        dev = self.device
        # first: the only collective (nothing may fail on one rank before it)
        peer = PeerBuffer(K.peer_bytes(), self.group.cpu_group, self.rank, K.TP, dev)
        self.peer = peer
        peer.bytes.zero_()
        # the Lamport inboxes := the sentinel (unused by tagged builds)
        K.arm_peer(peer.bytes)
        self.sym = int(peer.local)
        self.peers_ptr = peer.addresses.data_ptr()
        # [failed, rank, epoch, site]
        self.fault_word = peer.bytes[:16].view(torch.int32)
        self.epoch = torch.zeros(K.EPOCH_WORDS, dtype=torch.int32, device=dev)
        self.tl = torch.zeros(K.BLOCKS * K.TL_POINTS, dtype=torch.int64, device=dev)
        self._streams()
        cus = current_platform.num_compute_units(dev.index)
        dummy_h = torch.zeros(1, K.HIDDEN, dtype=torch.bfloat16, device=dev)
        dummy_f = torch.zeros(K.EXPERTS, K.HIDDEN, dtype=torch.float32, device=dev)
        _writable_flydsl_cache()
        for s in self.widths:  # compile and load (a mode-0 launch returns at once)
            b = self._alloc(s)
            key = K.GlueBuild(rows=s, transport=self.transport)
            args = self._args(
                b, dummy_h, dummy_h, dummy_h.view(-1), dummy_f, dummy_f[0], 0
            )
            b.compiled = flyc.compile(K.build_glue(key), *args)
            self._check_residency(b.compiled, key, cus)
            self.bufs[s] = b
        logger.info(
            "MiniMax-M3 decode glue transport %s: %s",
            self.transport,
            {
                s: K.kernel_name(K.GlueBuild(rows=s, transport=self.transport))
                for s in self.widths
            },
        )
        torch.accelerator.synchronize()

    def _alloc(self, s: int) -> _Width:
        dev, h = self.device, K.HIDDEN
        mp = K.max_pad(s)

        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=dev)

        return _Width(
            s=s,
            compiled=None,
            scratch=torch.zeros(K.scratch_bytes(s), dtype=torch.uint8, device=dev),
            res_out=empty((s, h), torch.bfloat16),
            xn=empty((s, h), torch.bfloat16),
            xq=empty((s, h), torch.float8_e4m3fn),
            xsc=empty((s, h // 32), torch.uint8),
            ids=empty((s, K.TOPK), torch.int32),
            w=empty((s, K.TOPK), torch.float32),
            sids=torch.zeros(mp, dtype=torch.int32, device=dev),
            sw=torch.zeros(mp, dtype=torch.float32, device=dev),
            seids=torch.zeros(mp // 16, dtype=torch.int32, device=dev),
            nv=torch.zeros(2, dtype=torch.int32, device=dev),
            moe_buf=empty((s, h), torch.bfloat16),
            sh_inter=empty((s, self.shared_inter), torch.bfloat16)
            if self.shared_inter
            else None,
        )

    @staticmethod
    def _check_residency(
        compiled,
        key,
        cus: int,
        name: str | None = None,
        threads: int = K.THREADS,
        blocks: int = K.BLOCKS,
    ) -> None:
        """A persistent grid: every CTA must be resident at once."""
        from vllm.models.minimax_m3.amd.specialized_decode.mono.residency import (
            active_workgroups_per_cu,
        )

        name = name or K.kernel_name(key)
        per_cu = active_workgroups_per_cu(compiled, name, threads)
        if per_cu * cus < blocks:
            raise RuntimeError(
                f"{name}: {per_cu} resident CTA(s) per CU x {cus} CUs < {blocks}"
            )

    def _args(self, b: _Width, part, res_in, norm_w, router_w, bias, mode: int):
        return (
            part.data_ptr(),
            res_in.data_ptr(),
            b.res_out.data_ptr(),
            b.xn.data_ptr(),
            b.xq.data_ptr(),
            b.xsc.data_ptr(),
            b.ids.data_ptr(),
            b.w.data_ptr(),
            norm_w.data_ptr(),
            router_w.data_ptr(),
            bias.data_ptr(),
            b.sids.data_ptr(),
            b.sw.data_ptr(),
            b.seids.data_ptr(),
            b.nv.data_ptr(),
            b.moe_buf.data_ptr(),
            self.tl.data_ptr(),
            b.scratch.data_ptr(),
            self.sym,
            self.peers_ptr,
            self.rank,
            self.epoch.data_ptr(),
            mode,
            torch.cuda.current_stream(),
        )

    # ------------------------------------------------------------ per layer
    def launch(self, layer, part: torch.Tensor, residual: torch.Tensor) -> _Width:
        """The glue kernel alone (collective: every rank, same layer order), on
        a replicated residual (mode 1)."""
        b = self.bufs[part.shape[0]]
        moe, norm = layer.block_sparse_moe, layer.post_attention_layernorm
        if self.fused is not None:
            self._fused_call(
                b,
                part.contiguous(),
                residual.contiguous(),
                norm.weight,
                moe.gate.weight,
                moe.e_score_correction_bias,
            )
            return b
        args = self._args(
            b,
            part.contiguous(),
            residual.contiguous(),
            norm.weight,
            moe.gate.weight,
            moe.e_score_correction_bias,
            1,
        )
        b.compiled(*args)
        return b

    def run(
        self,
        layer,
        part,
        residual,
        sw: SharedWeights,
        aiter_moe,
        md,
        q_dtype_a,
        nxt_norm=None,
    ):
        """The layer's MoE half: (all-reduced MoE output, residual after
        attention); with `nxt_norm` (the next layer's input norm) and the fused
        post-MoE all-reduce set up: (Mxfp8Carry of the next layer's MXFP8
        attention input and residual, that residual)."""
        from vllm.utils.torch_utils import current_stream

        b = self.launch(layer, part, residual)
        if sw.gu_bf16 is not None:
            # the shared expert runs inside the routed MoE launches (its tiles first):
            # moe_buf = routed (x2.0 in the top-k weights) + shared
            out = aiter_moe.run_sorted(
                md,
                q_dtype_a,
                b.xn,
                (b.sids, b.sw, b.seids, b.nv, b.moe_buf, None, None),
                b.ids,
                b.w,
                shared=(sw.gu_bf16, sw.dn_bf16, b.sh_inter),
            )
            return self._post_moe(out, b, nxt_norm)
        mlp = layer.block_sparse_moe.shared_experts
        side = self.stream is not None and b.s <= SIDE_STREAM_MAX_ROWS
        if side:
            main = current_stream()
            self.ev_fork.record(main)
            with torch.cuda.stream(self.stream):
                self.ev_fork.wait(self.stream)
                sh = shared_decode(sw, mlp, b.xq, b.xsc)
                self.ev_join.record(self.stream)
        else:
            sh = shared_decode(sw, mlp, b.xq, b.xsc)
        # routed scale x2.0 is in the kernel's top-k weights: no output scaling
        routed = aiter_moe.run_sorted(
            md,
            q_dtype_a,
            b.xn,
            (b.sids, b.sw, b.seids, b.nv, b.moe_buf, None, None),
            b.ids,
            b.w,
        )
        if side:
            self.ev_join.wait(main)
        return self._post_moe(routed + sh, b, nxt_norm)

    # ------------------------------------------------------------ post-MoE
    def setup_post_moe(self) -> None:
        """Collective (every TP rank, after `setup`): AITER's fused all-reduce +
        Gemma RMSNorm + MXFP8 quant (ROCm/aiter#7792) when the installed AITER
        has it; else the post-MoE step stays the stock all-reduce."""
        from vllm.models.deepseek_v41.amd.mono.common.peer_memory import tp_agree

        cpu = self.group.cpu_group
        rows = min(POST_MOE_MAX_ROWS, max(self.widths))
        ok, err = True, None
        try:
            from aiter.ops.triton.comms.fused.all_reduce_rmsnorm_mxfp8 import (
                AllReduceRmsnormMxfp8,
                fused_all_reduce_rmsnorm_mxfp8,  # noqa: F401
            )
        except Exception as e:  # noqa: BLE001 - an AITER without the op
            ok, err = False, e
        if not tp_agree(ok, cpu):
            logger.info(
                "MiniMax-M3 decode glue: post-MoE all-reduce unfused (%s)",
                err or "refused on another rank",
            )
            return
        comm: Any = None
        try:
            comm = AllReduceRmsnormMxfp8(cpu, rows, K.HIDDEN)  # collective
        except Exception as e:  # noqa: BLE001
            ok, err = False, e
        if not tp_agree(ok, cpu):
            logger.warning(
                "MiniMax-M3 decode glue: fused post-MoE all-reduce setup failed on "
                "%s rank: %s",
                "this" if err else "another",
                err,
            )
            if comm is not None:
                comm.close()
            return
        self.post_ar = comm
        self._post_err = _error_record_view(comm)
        if self.group.rank_in_group == 0:
            logger.info(
                "MiniMax-M3 decode glue: post-MoE all-reduce + next input norm + "
                "MXFP8 quant fused (AITER) up to %d rows",
                rows,
            )

    def _post_moe(self, moe_out: torch.Tensor, b: _Width, nxt_norm):
        """The all-reduce after the MoE. With the next layer's input norm and
        AITER's fused op: one launch to that layer's MXFP8 attention input and
        residual (every row on every rank); else the stock all-reduce (the next
        layer's input norm adds the residual and quantizes)."""
        if (
            nxt_norm is not None
            and self.post_ar is not None
            and b.s <= self.post_ar.max_tokens
        ):
            from aiter.ops.triton.comms.fused.all_reduce_rmsnorm_mxfp8 import (
                fused_all_reduce_rmsnorm_mxfp8,
            )

            x_q, x_s, res = fused_all_reduce_rmsnorm_mxfp8(
                moe_out.contiguous(),
                b.res_out,
                nxt_norm.weight,
                nxt_norm.variance_epsilon,
                self.post_ar,
                gemma=True,
            )
            return Mxfp8Carry(x_q, x_s, res, b.s), res
        from vllm.distributed import tensor_model_parallel_all_reduce

        return tensor_model_parallel_all_reduce(moe_out), b.res_out

    # ------------------------------------------------------------ epoch
    def reset(self) -> None:
        """No launch in flight on any rank: mailboxes, scratch and the epoch
        cleared, the Lamport inboxes re-armed (the sticky error word is kept;
        AITER's fused op clears its error record too)."""
        if self.fused is not None:
            self.fused.reset()  # collective
            return
        assert self.peer is not None
        self.peer.bytes[16:].zero_()
        K.arm_peer(self.peer.bytes)
        for b in self.bufs.values():
            b.scratch.zero_()
        self.epoch.zero_()
        torch.accelerator.synchronize()

    def reset_post_moe(self) -> None:
        """Collective, every TP rank with no launch in flight: empties the fused
        post-MoE all-reduce's inboxes and clears its colours and error record."""
        if self.post_ar is not None:
            self.post_ar.reset()

    def assert_healthy(self) -> None:
        """Device-side (no host sync, so also under graph capture): traps the
        queue when any rank set the glue's error word or the fused post-MoE
        all-reduce's error record (what `post_ar.error()` reads)."""
        assert self.fault_word is not None
        torch._assert_async(self.fault_word[0] == 0, FAULT_MESSAGE)
        if self._post_err is not None:
            torch._assert_async(self._post_err[0] == 0, POST_FAULT_MESSAGE)

    def close(self) -> None:
        if self.fused is not None:
            self.fused.close()
            self.fused = None
        if self.post_ar is not None:
            self.post_ar.close()
            self.post_ar = None
            self._post_err = None
        if self.peer is not None:
            self.peer.close()
            self.peer = None


def _side_stream_disabled() -> bool:
    import vllm.envs as envs

    return bool(envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM)


def _error_record_view(comm) -> torch.Tensor | None:
    """The first words of the error record in `comm`'s own IPC buffer ([flag,
    rank, colour, site], what its `error()` reads), as a device view for a check
    without a host sync; None if the buffer is not where AITER documents it."""
    from vllm.models.deepseek_v41.amd.mono.common.peer_memory import _DeviceBytes

    try:
        base = int(comm.peers[0])
    except (AttributeError, IndexError, TypeError, ValueError):
        return None
    view = torch.as_tensor(_DeviceBytes(base, 16), device=comm.device)
    return view.view(torch.int32)
