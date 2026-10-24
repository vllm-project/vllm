# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram WKV ahead of the decoder layers on ROCm.

An Engram layer's WKV chain (head all-gather, WKV GEMM, output all-gather) reads
only the step's hash rows, so it can run on a side stream behind the decoder
layers before it. Its gathers must not be collectives there: a collective kernel
waits on the GPU for its peers, and one on a side stream next to the main
stream's TP collectives can deadlock the ranks. Each rank instead pushes its
shard into every peer's buffer with plain stores, and the main stream waits for
the push before a decoder layer whose TP collectives then double as a barrier:
once they complete on a rank, every rank has pushed.

For the Engram layer at index L (``wkv_layer = L - 1``):

- ``start`` (before the layers): push the heads, once the lookup is done.
- before layer ``HEADS_BARRIER_LAYER``: the main stream waits for the push.
- before the next layer: WKV GEMM on the heads, push the output.
- before ``wkv_layer``: the main stream waits for that push.
- layer L reads the gathered output from its own buffer.

A rank only pushes again in the next step after passing that step's barriers,
so no peer still reads the buffers it overwrites.
"""

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tp_group
from vllm.models.deepseek_v41.amd.mono.common.peer_memory import PeerBuffer
from vllm.models.deepseek_v41.common.engram import Engram
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

# The main stream waits for the head push before this layer, so the lookups and
# the push hide behind the layers before it.
HEADS_BARRIER_LAYER = 3
# Smaller eager steps (e.g. decode without CUDA graphs) were not measured.
ENGRAM_WKV_AHEAD_MIN_TOKENS = 513
# Enough for full peer bandwidth on MI355X TP4; a push grid sized to the rows
# instead occupies every CU and slows the main stream's kernels.
PUSH_WORKGROUPS = 128


@triton.jit
def _push_rows_kernel(
    src,
    src_stride,
    num_rows,
    peer_bases,
    dst_offset,
    dst_stride,
    NUM_COLS: tl.constexpr,
    NUM_PEERS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Store every row of ``src`` at ``dst_offset + row * dst_stride`` elements
    into the bf16 buffer of every peer in ``peer_bases``. Persistent over rows so
    the push occupies only its grid's CUs next to the main stream."""
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < NUM_COLS
    for row in range(tl.program_id(0), num_rows, tl.num_programs(0)):
        row = row.to(tl.int64)
        vals = tl.load(src + row * src_stride + cols, mask=mask)
        dst = dst_offset + row * dst_stride + cols
        for peer in tl.static_range(NUM_PEERS):
            base = tl.load(peer_bases + peer).to(tl.pointer_type(tl.bfloat16))
            tl.store(base + dst, vals, mask=mask)


def engram_wkv_ahead_supported(
    vllm_config: VllmConfig, engram: Engram, layer_idx: int, replay_start: int
) -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx942, on_gfx950

    table = engram.embed_tokens
    return (
        (on_gfx942() or on_gfx950())
        and table.tp_size > 1
        and table.dp_size == 1
        and not engram.use_sequence_parallel
        and engram._batch is None
        and get_pp_group().world_size == 1
        and layer_idx - 1 > HEADS_BARRIER_LAYER + 2
        and layer_idx < replay_start
        and vllm_config.lora_config is None
        and not vllm_config.parallel_config.use_ubatching
    )


class EngramWkvAhead:
    """Runs one Engram layer's WKV chain ahead of the decoder layers.

    The constructor is collective over the TP group (the peer buffer handle
    exchange).
    """

    def __init__(
        self, engram: Engram, layer_idx: int, max_tokens: int, min_tokens: int
    ) -> None:
        tp_group = get_tp_group()
        table = engram.embed_tokens
        self.engram = engram
        self.layer_idx = layer_idx
        self.wkv_layer = layer_idx - 1
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.rank = tp_group.rank_in_group
        self.tp_size = tp_group.world_size
        self.head_cols = table.part_n_hash_cols * table.dim
        self.cols = engram.wkv.output_size_per_partition
        device = torch.device(f"cuda:{torch.accelerator.current_device_index()}")
        heads_numel = max_tokens * self.tp_size * self.head_cols
        kv_numel = self.tp_size * max_tokens * self.cols
        # Uncached, so reads see the peers' stores without cache maintenance.
        self.peer = PeerBuffer(
            2 * (heads_numel + kv_numel),
            tp_group.cpu_group,
            self.rank,
            self.tp_size,
            device,
        )
        buf = self.peer.bytes.view(torch.bfloat16)
        self.heads = buf[:heads_numel].view(max_tokens, self.tp_size * self.head_cols)
        self.kv = buf[heads_numel:].view(self.tp_size, max_tokens, self.cols)
        self.kv_offset = heads_numel + self.rank * max_tokens * self.cols
        self.stream = torch.cuda.Stream(device=device)
        self._heads_pushed = torch.cuda.Event()
        self._kv_pushed = torch.cuda.Event()
        self._num_tokens = 0

    def _push(self, src: torch.Tensor, dst_offset: int, dst_stride: int) -> None:
        num_rows, num_cols = src.shape
        block = 1024
        col_blocks = triton.cdiv(num_cols, block)
        grid_rows = min(num_rows, max(1, PUSH_WORKGROUPS // col_blocks))
        _push_rows_kernel[(grid_rows, col_blocks)](
            src,
            src.stride(0),
            num_rows,
            self.peer.addresses,
            dst_offset,
            dst_stride,
            NUM_COLS=num_cols,
            NUM_PEERS=self.tp_size,
            BLOCK=block,
        )

    def start(self, num_tokens: int) -> None:
        """Push this rank's heads once the step's lookup is done. Every TP rank
        sees the same ``num_tokens``, so all of them take the same branch."""
        self._num_tokens = 0
        if (
            not self.min_tokens <= num_tokens <= self.max_tokens
            or torch.cuda.is_current_stream_capturing()
        ):
            return
        engram = self.engram
        self.stream.wait_stream(torch.cuda.current_stream())
        if engram._prefetch_done is not None:
            self.stream.wait_event(engram._prefetch_done)
        with torch.cuda.stream(self.stream):
            rows = engram.staged_rows[:num_tokens].flatten(1)
            self._push(rows, self.rank * self.head_cols, self.tp_size * self.head_cols)
        self._heads_pushed.record(self.stream)
        self._num_tokens = num_tokens

    def before_layer(self, idx: int) -> None:
        num_tokens = self._num_tokens
        if not num_tokens:
            return
        main = torch.cuda.current_stream()
        if idx == HEADS_BARRIER_LAYER:
            main.wait_event(self._heads_pushed)
        elif idx == HEADS_BARRIER_LAYER + 1:
            # The previous layer's TP collectives passed: every rank pushed.
            self.stream.wait_stream(main)
            engram = self.engram
            table = engram.embed_tokens
            with torch.cuda.stream(self.stream):
                heads = self.heads[:num_tokens].view(num_tokens, -1, table.dim)
                x = heads[:, : table.n_hash_cols].flatten(1)
                out = engram.wkv.quant_method.apply(engram.wkv, x, None)
                self._push(out, self.kv_offset, self.cols)
            self._kv_pushed.record(self.stream)
        elif idx == self.wkv_layer:
            main.wait_event(self._kv_pushed)
        elif idx == self.layer_idx:
            self.engram._ahead_kv = self.kv[:, :num_tokens]
            self._num_tokens = 0
