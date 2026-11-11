# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SP (sequence-parallel) MoE half for large eager steps, TP4.

Per MoE layer, rows padded to a multiple of TP, R = S_pad / TP rows a rank:

1. reduce-scatter of the attention partial (AITER custom RS, else RCCL);
2. fused add + Gemma RMSNorm (bf16 + MXFP8) of the rank's R rows with its
   residual rows;
3. router (the layer's GateLinear) and the shared expert (full replicated MXFP8
   weights, AITER preshuffled GEMM) on the R rows;
4. biased top-k (x routed scale) and MXFP4 quant of the R normed rows;
5. all-gather of the FP4 codes and of (scales | ids | weights) (AITER custom AG,
   else RCCL);
6. the stock AITER A4W4 row for M = S on the gathered, prequantized rows
   (aiter_moe.AiterMoe.run_prequant);
7. shared output + the new residual rows added into the rank's own rows of the
   routed partial, then one TP all-reduce.

With the next layer's input norm (`_post_moe`), step 7 is a reduce-scatter of
the routed partial instead, and the next layer gets its MXFP8 attention input
(`Mxfp8Carry`). When the installed AITER has the fused MoE top-k combine +
reduce-scatter (`fused_moe_combine_reduce_scatter`, ROCm/aiter#7771) and the
stock row's stage 2 ends in the bf16 scatter epilogue (padded M >= 4096), the
routed MoE stops before its top-k combine (`scatter_intermediate`) and that one
launch does the combine and the reduce-scatter together: no [M, 6144] combined
output, no padding copy. Otherwise the combine and the reduce-scatter run as
before.

The all-reduced result already holds the new residual, so the layer returns
(result, zero residual): the next layer's fused add (or the final norm) adds a
zero. The zero residual is a stride-0 view of one zero row (the fused-add Triton
kernels take the residual's row stride)."""

from __future__ import annotations

import functools
import inspect
from typing import Any

import torch
import torch.nn.functional as F

from vllm.logger import init_logger

from .carry import Mxfp8Carry
from .shared import SharedWeights, shared_full

TOPK_GROUPS = 4  # biased_grouped_topk(.., 4, 4, ..): every group kept, a flat top-k
logger = init_logger(__name__)

HIDDEN, TOPK = 6144, 4
FAULT_MESSAGE = (
    "MiniMax-M3 SP: fused combine + reduce-scatter error record set (a peer wait "
    "timed out)"
)
# Rows of a step the fused combine + reduce-scatter takes at most (its receive
# buffer: 3 x ceil(rows / 4) x 12 KB a rank, 151 MB at 16384).
SCATTER_RS_MAX_ROWS = 16384


class SpPrefill:
    def __init__(self, group, min_rows: int, max_rows: int) -> None:
        self.group = group
        self.rank = group.rank_in_group
        self.world = group.world_size
        self.min_rows = max(min_rows, self.world)
        self.max_rows = max_rows
        self._zero: torch.Tensor | None = None
        # AITER's fused top-k combine + reduce-scatter buffers, once set up
        self.combine_rs = None
        self._err_view: torch.Tensor | None = None
        self._capture = None  # tests: a dict that receives each run's top-k
        self._setup_combine_rs()

    def _setup_combine_rs(self) -> None:
        """Collective, every TP rank at the same point (SpPrefill is built at
        the first eager forward, vLLM's profile run, so the memory profile sees
        the buffer). Off, with the old combine + reduce-scatter, when the
        installed AITER lacks the op or the `scatter_intermediate` stage-2
        keyword (ROCm/aiter#7771)."""
        from vllm.models.deepseek_v41.amd.mono.common.peer_memory import tp_agree

        cpu = self.group.cpu_group
        ok, err = True, None
        try:
            if not aiter_combine_rs_available():
                raise RuntimeError("AITER has no fused MoE combine + reduce-scatter")
            from aiter.ops.triton.comms.fused.moe_combine_reduce_scatter import (
                MoeCombineReduceScatter,
            )
        except Exception as e:  # noqa: BLE001
            ok, err = False, e
        if not tp_agree(ok, cpu):
            logger.info(
                "MiniMax-M3 SP: top-k combine + reduce-scatter unfused (%s)",
                err or "refused on another rank",
            )
            return
        comm: Any = None
        try:
            comm = MoeCombineReduceScatter(
                cpu, SCATTER_RS_MAX_ROWS, HIDDEN, TOPK
            )  # collective handshake
        except Exception as e:  # noqa: BLE001
            ok, err = False, e
        if not tp_agree(ok, cpu):
            logger.warning(
                "MiniMax-M3 SP: fused combine + reduce-scatter setup failed on %s "
                "rank: %s",
                "this" if err else "another",
                err,
            )
            if comm is not None:
                comm.close()
            return
        self.combine_rs = comm
        self._err_view = _error_record_view(comm)
        if self.rank == 0:
            logger.info(
                "MiniMax-M3 SP: top-k combine fused into the post-MoE "
                "reduce-scatter (AITER), up to %d rows, %.1f MB a rank",
                SCATTER_RS_MAX_ROWS,
                comm.nbytes / 1e6,
            )

    def assert_healthy(self) -> None:
        """Device-side: traps the queue once a wait of the fused combine +
        reduce-scatter timed out on any rank (its error record)."""
        if self._err_view is not None:
            torch._assert_async(self._err_view[0] == 0, FAULT_MESSAGE)

    def close(self) -> None:
        if self.combine_rs is not None:
            self.combine_rs.close()
            self.combine_rs = None
            self._err_view = None

    def zero_residual(self, rows: int, like: torch.Tensor) -> torch.Tensor:
        if (
            self._zero is None
            or self._zero.shape[1] != like.shape[1]
            or self._zero.dtype != like.dtype
        ):
            self._zero = torch.zeros(
                1, like.shape[1], dtype=like.dtype, device=like.device
            )
        return self._zero.expand(rows, like.shape[1])

    # ------------------------------------------------------------ collectives
    def _aiter_comm(self):
        ca = getattr(self.group.device_communicator, "aiter_ar_comm", None)
        ca = getattr(ca, "_impl", ca)
        return None if ca is None or getattr(ca, "disabled", True) else ca

    def _reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        from vllm.distributed import tensor_model_parallel_reduce_scatter

        ca = self._aiter_comm()
        if ca is not None and ca.should_custom_rs(x, 0):
            out = torch.empty(
                (x.shape[0] // self.world, x.shape[1]), dtype=x.dtype, device=x.device
            )
            ca.custom_reduce_scatter(x, out, dim=0)
            return out
        return tensor_model_parallel_reduce_scatter(x, dim=0)

    def _all_gather(self, x: torch.Tensor) -> torch.Tensor:
        """Rows of a byte tensor from every rank, in rank order (bf16 carrier:
        AITER's IPC gather takes 16/32-bit floats only)."""
        from vllm.distributed import tensor_model_parallel_all_gather

        x16 = x.contiguous().view(torch.bfloat16)
        ca = self._aiter_comm()
        if ca is not None and ca.should_custom_ag(x16):
            return ca.custom_all_gather(x16, dim=0).view(torch.uint8)
        return tensor_model_parallel_all_gather(x16, dim=0).view(torch.uint8)

    # ------------------------------------------------------------ per layer
    def run(
        self,
        layer,
        part: torch.Tensor,
        residual: torch.Tensor,
        sw: SharedWeights,
        aiter_moe,
        nxt_norm=None,
    ):
        """`part`: this rank's unreduced o_proj output [S, H]; `residual`: [S, H], or
        this rank's padded rows [R, H] (an Mxfp8Carry's). Without `nxt_norm`:
        (all-reduced MoE output + new residual, zero residual). With the next layer's
        input norm: the MoE partial reduce-scattered, + residual and shared rows, the
        next layer's fused add + Gemma RMSNorm + MXFP8 quant on this rank's rows, the
        codes and scales all-gathered -> (Mxfp8Carry, this rank's new residual rows)."""
        import aiter
        from aiter import dtypes
        from aiter.ops.quant import per_1x32_mx_quant_hip

        from vllm.distributed import tensor_model_parallel_all_reduce
        from vllm.models.minimax_m3.amd.ops.gemma_rmsnorm import (
            gemma_fused_add_rmsnorm_mxfp8_and_bf16,
        )

        moe, norm = layer.block_sparse_moe, layer.post_attention_layernorm
        s, h = part.shape
        rows = -(-s // self.world)
        if rows * self.world != s:
            part = F.pad(part, (0, 0, 0, rows * self.world - s))
        hs = self._reduce_scatter(part.contiguous())
        lo = self.rank * rows
        n = max(
            0, min(rows, s - lo)
        )  # this rank's real rows (the last may hold padding)
        if (
            residual.shape[0] == rows and s != rows
        ):  # this rank's rows already (sharded residual)
            res = residual
        else:
            res = residual[lo : lo + n]
            if n < rows:
                res = F.pad(res, (0, 0, 0, rows - n))
        x_q, x_s, normed, res_new = gemma_fused_add_rmsnorm_mxfp8_and_bf16(
            hs, res, norm.weight, norm.variance_epsilon
        )
        logits, _ = moe.gate(normed)
        sh = shared_full(sw, moe.shared_experts, x_q, x_s)
        topk = aiter_moe.topk
        w = torch.empty((rows, topk), dtype=torch.float32, device=part.device)
        ids = torch.empty((rows, topk), dtype=torch.int32, device=part.device)
        aiter.biased_grouped_topk(
            logits,
            moe.e_score_correction_bias,
            w,
            ids,
            TOPK_GROUPS,
            TOPK_GROUPS,
            True,
            float(moe.routed_scaling_factor),
        )
        fp4, sc = per_1x32_mx_quant_hip(
            normed, scale=None, quant_dtype=dtypes.fp4x2, shuffle=False
        )
        small = torch.cat(
            [sc.view(torch.uint8), ids.view(torch.uint8), w.view(torch.uint8)], dim=1
        )
        fp4_all = self._all_gather(fp4.view(torch.uint8))  # [S_pad, H / 2]
        small_all = self._all_gather(small)  # [S_pad, H / 32 + 8 topk]
        g = h // 32
        sc_all = small_all[:s, :g].contiguous()
        ids_all = small_all[:s, g : g + 4 * topk].contiguous().view(torch.int32)
        w_all = small_all[:s, g + 4 * topk :].contiguous().view(torch.float32)
        if self._capture is not None:
            self._capture.update(ids=ids_all, w=w_all)
        # stage 2 stops before its top-k combine: the intermediate goes to the
        # fused combine + reduce-scatter
        defer = None
        if (
            nxt_norm is not None
            and self.combine_rs is not None
            and s <= SCATTER_RS_MAX_ROWS
            and aiter_moe.scatter_deferrable(s)
        ):
            defer = {}
        # routed partial, x routed scale (not written when `defer` comes back
        # filled)
        out = aiter_moe.run_prequant(fp4_all[:s], sc_all, ids_all, w_all, scatter=defer)
        if nxt_norm is not None:
            return self._post_moe(out, s, rows, res_new, sh, nxt_norm, defer)
        if n:
            mine = out[lo : lo + n]
            mine += sh[:n]
            mine += res_new[:n]
        return tensor_model_parallel_all_reduce(out), self.zero_residual(s, out)

    def _post_moe(self, out, s, rows, res_new, sh, nxt_norm, defer=None):
        """The MoE partial reduce-scattered (bf16), + this rank's residual and
        shared rows, then the next layer's input norm with its MXFP8 quant on
        these rows; codes and scales all-gathered (~0.52x a bf16 gather).
        `defer` (filled by stage 2): the uncombined GEMM2 intermediate, whose
        top-k combine then runs fused into the reduce-scatter."""
        from aiter.ops.triton.comms.fused.moe_combine_reduce_scatter import (
            fused_moe_combine_reduce_scatter,
        )

        from vllm.models.minimax_m3.amd.ops.gemma_rmsnorm import (
            gemma_fused_add_rmsnorm_mxfp8,
        )

        if defer:
            # [R, H]: this rank's rows of the combined, reduce-scattered MoE
            # output (x routed scale)
            rs = fused_moe_combine_reduce_scatter(
                defer["inter"],
                defer["reverse_sorted"],
                defer["sorted_weights"],
                s,
                self.combine_rs,
            )
        else:
            if rows * self.world != s:
                out = F.pad(out, (0, 0, 0, rows * self.world - s))
            # [R, H]: this rank's rows of the MoE output
            rs = self._reduce_scatter(out.contiguous())
        res_in = res_new + sh[:rows]
        x_q, x_s, res_out = gemma_fused_add_rmsnorm_mxfp8(
            rs, res_in, nxt_norm.weight, nxt_norm.variance_epsilon
        )
        q_all = self._all_gather(x_q.view(torch.uint8))[:s].view(x_q.dtype)
        s_all = self._all_gather(x_s.view(torch.uint8))[:s].view(x_s.dtype)
        return Mxfp8Carry(q_all, s_all, res_out, s), res_out


@functools.cache
def aiter_combine_rs_available() -> bool:
    """Whether AITER has the fused MoE top-k combine + reduce-scatter and the
    stage-2 `scatter_intermediate` keyword it consumes (ROCm/aiter#7771)."""
    try:
        from aiter import fused_moe as fm
        from aiter.ops.triton.comms.fused.moe_combine_reduce_scatter import (  # noqa: F401
            MoeCombineReduceScatter,
            fused_moe_combine_reduce_scatter,
        )

        return all(
            "scatter_intermediate" in inspect.signature(getattr(fm, f)).parameters
            for f in ("_flydsl_v2_stage2_wrapper", "_mxfp4_a4w4_stage2_fw")
        )
    except (ImportError, AttributeError, TypeError, ValueError):
        return False


def _error_record_view(comm) -> torch.Tensor | None:
    """A device view of the first word of `comm`'s error record (its own IPC
    buffer starts with [flag, rank, colour, site]), for a check without a host
    sync; None if the buffer layout is not the documented one."""
    from vllm.models.deepseek_v41.amd.mono.common.peer_memory import _DeviceBytes

    try:
        base = int(comm.peers[0])
    except (AttributeError, IndexError, TypeError, ValueError):
        return None
    view = torch.as_tensor(_DeviceBytes(base, 16), device=comm.device)
    return view.view(torch.int32)
