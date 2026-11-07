# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram (layers 1, 14) with the asm decode layers: the wkv projection of the
looked-up rows depends only on the hash ids, so the prefetch stream runs it
right after the lookup (eager, as vLLM's prefetch) and the layer's forward
keeps only the gate on the critical path (_EngramSide); the layer's attention
seam (ours) applies gate · value as its post-mix. M ≤ 256 only; anything else
is vLLM's path."""

import torch
import torch.distributed as dist

from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphCapture,
    eager_break_during_capture,
)
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.models.deepseek_v41.amd.asm.decode import MONO_ROWS
from vllm.models.deepseek_v41.amd.engram import RocmEngram
from vllm.models.deepseek_v41.common.engram import Engram

logger = init_logger(__name__)

_eg: dict = {"seen": set(), "deferred": []}
EG_MAX_M = 256


def eg_flush(main) -> None:
    """Issues the deferred rows pushes on ``main`` and the waits + side GEMMs on
    each Engram layer's prefetch stream."""
    while _eg["deferred"]:
        side, engram, n, src = _eg["deferred"].pop(0)
        side.issue(engram, n, src, main, engram._prefetch_stream)


def _unshard(t: torch.Tensor, world: int, blocks: int) -> torch.Tensor:
    """[world, blocks, rows, ...] (all-gathered per-rank slices of each block)
    → [blocks, world, rows, ...] flattened, the rows in block order."""
    return (
        t.view(world, blocks, -1, *t.shape[1:])
        .transpose(0, 1)
        .reshape(t.shape)
        .contiguous()
    )


class _EngramSide:
    """One Engram layer's side chain. wkv resharded once (collective, at
    creation): rank r computes [key of hc copy r | the whole value] (10240
    columns) instead of 6400 columns of kv, so no kv all-gather. Side chain:
    MXFP8 quant of the rank's own rows (vLLM's op; the 32-blocks sit inside the
    256-wide hash columns, so the codes are those of vLLM's quant of the
    gathered rows), all-gather of [codes | scales], GEMM into a persistent
    buffer. The rows all-gather is split: the main stream pushes (at layer 0's
    attention eager break, when the lookups are long done), the prefetch stream
    waits. Critical path: gate of copy r, all-gather of the 4 gates (16 B per
    token); the next attention seam applies copy + gate · value."""

    def __init__(self, engram):
        from vllm.models.deepseek_v41.amd.asm.eg_runner import RANK_ROW, IpcAllGather

        tp = get_tp_group()
        r, world = tp.rank_in_group, tp.world_size
        D, hc = engram.dim, engram.hc_mult
        w, ws = engram.wkv.weight.data, engram.wkv.weight_scale.data
        assert w.dtype == torch.float8_e4m3fn and ws.shape[0] * 32 == w.shape[0], (
            w.dtype,
            w.shape,
            ws.shape,
        )
        assert world * w.shape[0] == (hc + 1) * D and D % 32 == 0
        full_w = torch.empty(
            world * w.shape[0], w.shape[1], dtype=torch.uint8, device=w.device
        )
        full_ws = torch.empty(
            world * ws.shape[0], ws.shape[1], dtype=torch.uint8, device=w.device
        )
        dist.all_gather_into_tensor(
            full_w, w.view(torch.uint8).contiguous(), group=tp.device_group
        )
        dist.all_gather_into_tensor(full_ws, ws.contiguous(), group=tp.device_group)
        if engram.kv_shards > 1:
            # RocmEngram's sharded gate: rank s holds the D / world slice s of
            # each of the hc + 1 blocks → wkv's row order
            assert engram.kv_shards == world
            full_w = _unshard(full_w, world, hc + 1)
            full_ws = _unshard(full_ws, world, hc + 1)
        self.w = (
            torch.cat([full_w[r * D : (r + 1) * D], full_w[hc * D :]])
            .view(torch.float8_e4m3fn)
            .contiguous()
        )
        b = D // 32
        self.ws = torch.cat(
            [full_ws[r * b : (r + 1) * b], full_ws[hc * b :]]
        ).contiguous()
        del full_w, full_ws
        self.r, self.D = r, D
        self.ag_rows = IpcAllGather(
            r, tp.cpu_group, RANK_ROW, EG_MAX_M, world, split=True
        )
        self.ag_gate = IpcAllGather(r, tp.cpu_group, 16, EG_MAX_M, world)
        # [key | value] planes
        self.kv2 = torch.zeros(2, EG_MAX_M, D, dtype=torch.bfloat16, device=w.device)
        self.gate = torch.zeros(EG_MAX_M, 4, dtype=torch.float32, device=w.device)

    def run(self, engram, rows, main):
        """On the prefetch stream (current): the quant, then the push / wait +
        GEMM (issue), deferred to layer 0's attention eager break above the mono
        decode layer's rows."""
        n = rows.shape[0]
        xq, xs = mxfp8_e4m3_quantize(rows.reshape(n, -1))
        if not hasattr(self, "src_buf"):
            self.src_buf = torch.empty(
                EG_MAX_M, xq.shape[1] + xs.shape[1], dtype=torch.uint8, device=xq.device
            )
            self.ev_q, self.ev_ag = torch.cuda.Event(), torch.cuda.Event()
        src = self.src_buf[:n]
        torch.cat([xq.view(torch.uint8), xs], dim=1, out=src)
        prefetch = torch.cuda.current_stream()
        self.ev_q.record(prefetch)
        if n > MONO_ROWS:
            _eg["deferred"].append((self, engram, n, src))
            return "deferred"
        self.issue(engram, n, src, main, prefetch, record=False)
        return None

    def issue(self, engram, n, src, main, prefetch, record=True):
        """The rows push on ``main`` after the quant (ev_q), then on the
        prefetch stream the push's wait and the side GEMM; record: then
        _prefetch_done."""
        from vllm.models.deepseek_v41.amd.asm.eg_runner import side_gemm_packed

        main.wait_event(self.ev_q)
        with torch.cuda.stream(main):
            self.ag_rows.push(src)
            self.ev_ag.record(main)
        prefetch.wait_event(self.ev_ag)
        with torch.cuda.stream(prefetch):
            g = self.ag_rows.wait(n, src)
            side_gemm_packed(g, self.w, self.ws, self.kv2, 0)
            if record:
                engram._prefetch_done.record(prefetch)

    def forward(self, engram, hidden_states, token_mask):
        """→ r' unchanged, with (r', gates, value) left on the Engram module for
        the layer's seam; outside the seam's row range → r'' (eg_out)."""
        from vllm.models.deepseek_v41.amd.asm.eg_runner import eg_gate, eg_out

        n = hidden_states.shape[0]
        assert hidden_states.is_contiguous()
        eg_gate(
            hidden_states,
            self.kv2[0, :n],
            engram.q_weight,
            engram.k_weight,
            token_mask,
            self.gate[:n],
            self.r,
            engram.eps,
            engram.clamp_value,
        )
        gates = self.ag_gate(self.gate[:n])
        lo, hi = engram.__dict__.get("_eg_seam_range", (1, 0))
        if lo <= n <= hi:
            engram._eg_pending = (hidden_states, gates, self.kv2[1, :n])
            return hidden_states
        return eg_out(hidden_states, self.kv2[1, :n], gates)


class AsmEngram(RocmEngram):
    """Engram with its wkv on the prefetch stream (_EngramSide) for decode
    steps of up to EG_MAX_M rows."""

    def _eligible(self, n: int) -> bool:
        return (
            1 <= n <= EG_MAX_M
            and self._batch is None
            and not self.use_sequence_parallel
            and self.embed_tokens.dp_size == 1
            and self._prefetch_stream is not None
            and get_tensor_model_parallel_world_size() == 4
        )

    @eager_break_during_capture
    def _start_prefetch(self, hash_ids: torch.Tensor, rows: torch.Tensor) -> None:
        n = hash_ids.shape[0]
        side = self.__dict__.get("_eg_side")
        if side is None and self._eligible(1):
            # collective: on the first prefetch of any size (every rank makes
            # the same calls)
            side = self._eg_side = _EngramSide(self)
            logger.debug("asm decode: Engram side chain on (created at M = %d)", n)
        if side is None or not self._eligible(n):
            Engram._start_prefetch.__wrapped__(self, hash_ids, rows)  # type: ignore[attr-defined]
            return
        current = torch.cuda.current_stream()
        stream = self._prefetch_stream
        assert stream is not None and self._prefetch_done is not None
        stream.wait_stream(current)
        hash_ids.record_stream(stream)
        with torch.cuda.stream(stream):
            self.embed_tokens.lookup(hash_ids, rows, background=True)
            ran = side.run(self, rows, main=current)
        if ran != "deferred":
            self._prefetch_done.record(stream)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hash_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = hash_ids.shape[0]
        side = self.__dict__.get("_eg_side")
        if side is None or not self._eligible(n):
            return super().forward(hidden_states, hash_ids, token_mask)
        assert hidden_states.shape == (n, self.hc_mult, self.dim) and (
            token_mask is None or token_mask.shape == (n,)
        )
        if n not in _eg["seen"]:
            _eg["seen"].add(n)
            logger.debug("asm decode: Engram at M = %d", n)
        if _eg["deferred"]:
            if (
                BreakableCUDAGraphCapture.current() is not None
                or torch.cuda.is_current_stream_capturing()
            ):
                raise RuntimeError(
                    "Engram: deferred all-gathers not flushed by layer 0 in a capture"
                )
            if not _eg.get("flush_fallback"):
                _eg["flush_fallback"] = True
                logger.debug(
                    "asm decode: Engram's deferred all-gathers flushed at the "
                    "Engram layer (layer 0 not reached, M = %d)",
                    n,
                )
            eg_flush(torch.cuda.current_stream())
        self._finish_prefetch(self._prefetch_done)
        return side.forward(self, hidden_states, token_mask)
