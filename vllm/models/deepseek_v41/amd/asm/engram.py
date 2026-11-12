# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram (layers 1, 14) on the asm decode layers' steps: the wkv projection of
the looked-up rows depends only on the hash ids, so it runs on the prefetch
stream once layer 0's attention is done (the lookups are by then; decode.py
`_eg_start`), and the layer keeps only the gate on the critical path; its
attention seam (ours) applies gate · value as its post-mix. The Engram module
stays vLLM's `RocmEngram`: its prefetch looks the rows up, and every other
step (prefill, the mono decode layer's) runs its own forward."""

import torch
import torch.distributed as dist

from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)

logger = init_logger(__name__)

EG_MAX_M = 256


def _unshard(t: torch.Tensor, world: int, blocks: int) -> torch.Tensor:
    """[world, blocks, rows, ...] (all-gathered per-rank slices of each block)
    → [blocks, world, rows, ...] flattened, the rows in block order."""
    return (
        t.view(world, blocks, -1, *t.shape[1:])
        .transpose(0, 1)
        .reshape(t.shape)
        .contiguous()
    )


class EngramSide:
    """One Engram layer's side chain. wkv resharded once (collective, at
    creation): rank r computes [key of hc copy r | the whole value] (10240
    columns) instead of 6400 columns of kv, so no kv all-gather. Side chain
    (`start`): MXFP8 quant of the rank's own rows (vLLM's op; the 32-blocks
    sit inside the 256-wide hash columns, so the codes are those of vLLM's
    quant of the gathered rows), all-gather of [codes | scales] (the push on
    the main stream, the wait on the prefetch stream), GEMM into a persistent
    buffer. Critical path (`gate`): gate of copy r, all-gather of the 4 gates
    (16 B per token); the next attention seam applies copy + gate · value."""

    @staticmethod
    def create(engram) -> "EngramSide | None":
        """The side chain where vLLM's Engram prefetches on a stream of its own
        (host-offloaded, TP-only tables) at TP4; collective: every rank makes the
        same calls."""
        from vllm.models.deepseek_v41.amd.asm import decode

        if (
            engram._prefetch_stream is None
            or engram._prefetch_done is None
            or engram._batch is not None
            or engram.use_sequence_parallel
            or engram.embed_tokens.dp_size != 1
            or get_tensor_model_parallel_world_size() != 4
        ):
            return None
        side = EngramSide(engram)
        decode._eg["sides"].append(side)
        logger.debug("asm decode: Engram side chain on")
        return side

    def __init__(self, engram):
        from vllm.models.deepseek_v41.amd.asm.eg_runner import RANK_ROW, IpcAllGather

        self.engram = engram
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
        self.gate_buf = torch.zeros(EG_MAX_M, 4, dtype=torch.float32, device=w.device)
        self.src_buf = torch.empty(
            EG_MAX_M, RANK_ROW, dtype=torch.uint8, device=w.device
        )
        self.ev_q, self.ev_ag = torch.cuda.Event(), torch.cuda.Event()
        self.done = torch.cuda.Event()
        # set by start(), consumed by the layer's Engram update in the same step
        self.started = False

    def start(self, n: int, main: torch.cuda.Stream) -> None:
        """This step's chain for n rows: the quant, the rows push on `main`,
        the push's wait and the side GEMM on the prefetch stream; then
        `done`."""
        from vllm.models.deepseek_v41.amd.asm.eg_runner import side_gemm_packed

        engram = self.engram
        prefetch = engram._prefetch_stream
        # after the lookup: vLLM's prefetch issued it on this stream, or inline
        # on main (not engram._prefetch_done: a captured graph whose stream
        # waits on its own event kills the HIP worker at replay)
        prefetch.wait_stream(main)
        with torch.cuda.stream(prefetch):
            rows = engram.staged_rows[:n]
            xq, xs = mxfp8_e4m3_quantize(rows.reshape(n, -1))
            src = self.src_buf[:n]
            torch.cat([xq.view(torch.uint8), xs], dim=1, out=src)
            self.ev_q.record(prefetch)
        main.wait_event(self.ev_q)
        with torch.cuda.stream(main):
            self.ag_rows.push(src)
            self.ev_ag.record(main)
        prefetch.wait_event(self.ev_ag)
        with torch.cuda.stream(prefetch):
            g = self.ag_rows.wait(n, src)
            side_gemm_packed(g, self.w, self.ws, self.kv2, 0)
            self.done.record(prefetch)
        self.started = True

    def gate(self, hidden_states: torch.Tensor, token_mask) -> tuple:
        """The gates of r' (`hidden_states` [n, 4, 5120]) → (all-gathered gates
        [n, 16] f32, value [n, 5120] bf16); after `done`."""
        from vllm.models.deepseek_v41.amd.asm.decode import _eg
        from vllm.models.deepseek_v41.amd.asm.eg_runner import eg_gate

        engram = self.engram
        n = hidden_states.shape[0]
        assert hidden_states.is_contiguous() and n <= EG_MAX_M
        if n not in _eg["seen"]:
            _eg["seen"].add(n)
            logger.debug("asm decode: Engram at M = %d", n)
        eg_gate(
            hidden_states,
            self.kv2[0, :n],
            engram.q_weight,
            engram.k_weight,
            token_mask,
            self.gate_buf[:n],
            self.r,
            engram.eps,
            engram.clamp_value,
        )
        return self.ag_gate(self.gate_buf[:n]), self.kv2[1, :n]
