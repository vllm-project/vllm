# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSv4.1 Engram tables in an Engram pool (`vllm.distributed.engram_pool`): pool
rows hold values then E8M0 scales; layers unpack MXFP8 WKV input as rows land."""

from functools import partial

import torch
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.engram_pool import (
    EngramFetch,
    EngramPool,
    EngramPrefetch,
    get_engram_pool,
    landed_row,
)
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v41.common.engram import (
    Engram,
    EngramBatch,
    EngramLayout,
    ParallelEngramEmbedding,
    _engram_head_shard_weight_loader,
)
from vllm.models.deepseek_v41.common.ops.query_quant import (
    can_fuse_query_quant,
    mxfp8_scale_bytes,
    mxfp8_scale_offset,
)
from vllm.triton_utils import tl, triton

# Values per E8M0 scale.
SCALE_BLOCK = 32
_SCALE: tl.constexpr = tl.constexpr(SCALE_BLOCK)


@triton.jit(do_not_specialize=["num_rows"])
def _dequant_rows_kernel(
    packed,
    out,
    num_rows,
    ROW_STRIDE: tl.constexpr,
    DIM: tl.constexpr,
    PACKED_FP4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Dequantize pool rows (FP8 or MXFP4 values, then E8M0 scales) into BF16."""
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = rows < num_rows
    ROW_BYTES: tl.constexpr = DIM // 2 if PACKED_FP4 else DIM
    src = packed + rows * ROW_STRIDE
    cols = tl.arange(0, DIM)
    if PACKED_FP4:
        byte = tl.load(
            src[:, None] + (cols // 2)[None, :], mask=valid[:, None], other=0
        ).to(tl.int32)
        nib = tl.where(cols[None, :] % 2 == 0, byte & 0xF, (byte >> 4) & 0xF)
        e = (nib & 0x7) >> 1
        m = (nib & 0x1).to(tl.float32)
        pow2 = ((e - 2 + 127) << 23).to(tl.float32, bitcast=True)
        mag = tl.where(e == 0, m * 0.5, (2.0 + m) * pow2)
        values = tl.where((nib & 0x8) != 0, -mag, mag)
    else:
        values = tl.load(
            src.to(tl.pointer_type(tl.float8e4nv))[:, None] + cols[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
    scale = tl.load(
        src[:, None] + ROW_BYTES + (cols // _SCALE)[None, :],
        mask=valid[:, None],
        other=0,
    )
    scale = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    tl.store(
        out + rows[:, None] * DIM + cols[None, :],
        (values * scale).to(tl.bfloat16),
        mask=valid[:, None],
    )


@triton.jit(do_not_specialize=["num_tokens", "table"])
def _unpack_fp8_kernel(
    rows,
    slots,
    ready,
    values,
    scales,
    num_tokens,
    table,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    PLACED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Split table `table`'s fetched `[token][head]` rows into MXFP8 values and
    F8_128x4 scales; scales past `num_tokens` zero-pad the 128-row tile."""
    row = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    token, head = row // HEADS, row % HEADS
    valid = token < num_tokens
    src, live = landed_row(rows, slots, ready, table, row, valid, ROW_STRIDE, PLACED)
    cols = tl.arange(0, DIM)
    data = tl.load(src[:, None] + cols[None, :], mask=live[:, None], other=0)
    tl.store(values + row[:, None] * DIM + cols[None, :], data, mask=valid[:, None])
    scale_cols = tl.arange(0, DIM // _SCALE)
    scale = tl.load(
        src[:, None] + DIM + scale_cols[None, :], mask=live[:, None], other=0
    )
    group = head[:, None] * (DIM // _SCALE) + scale_cols[None, :]
    offset = mxfp8_scale_offset(token[:, None], group, HEADS * DIM // _SCALE)
    tl.store(scales + offset, scale)


class PoolEngramEmbedding(ParallelEngramEmbedding):
    """A table whose rows an Engram pool splits over GPUs of several engines;
    its parameters are views of this GPU's shard, so loaders fill it in place."""

    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        head_sizes: tuple[int, ...],
        packed_fp4: bool,
        pool: EngramPool,
        name: str,
    ) -> None:
        nn.Module.__init__(self)
        assert get_tensor_model_parallel_world_size() == 1
        self.num_embeddings, self.dim = num_embeddings, dim
        self.block_size, self.packed_fp4 = SCALE_BLOCK, packed_fp4
        self.row_bytes = dim // 2 if packed_fp4 else dim
        self.n_hash_cols = self.part_n_hash_cols = len(head_sizes)
        self.tp_size = self.dp_size = 1
        self.cpu_offload = False
        self.pool = pool
        weight_dtype = torch.uint8 if packed_fp4 else torch.float8_e4m3fn
        scale_dim = dim // SCALE_BLOCK
        if torch.get_default_device().type == "meta":
            self.table = None
            row_start = 0
            weight = torch.empty(sum(head_sizes), self.row_bytes, dtype=weight_dtype)
            scales = torch.empty(sum(head_sizes), scale_dim, dtype=torch.uint8)
        else:
            self.table = pool.table(sum(head_sizes), self.row_bytes + scale_dim, name)
            row_start, shard = self.table.row_start, self.table.shard
            weight = shard[:, : self.row_bytes].view(weight_dtype)
            scales = shard[:, self.row_bytes :]
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale_inv = nn.Parameter(scales, requires_grad=False)
        weight_loader = self._weight_loader
        if pool.host_shards and weight.is_cuda:
            weight_loader = partial(
                _engram_head_shard_weight_loader, max_copy_bytes=64 << 20
            )
        for param in (self.weight, self.weight_scale_inv):
            set_weight_attrs(
                param,
                {"weight_loader": weight_loader, "engram_vocab_start": row_start},
            )
        # Constant dummy values; the ue8m0 encoding of scale 1.0 is 127.
        set_weight_attrs(self.weight, {"dummy_weight_value": 1.0})
        set_weight_attrs(self.weight_scale_inv, {"dummy_weight_value": 127})

    def process_weights_after_loading(self) -> None:
        """Serve once every GPU of the group has loaded its rows."""
        assert self.table is not None
        shard = self.table.shard.data_ptr()
        if (self.weight.data_ptr(), self.weight_scale_inv.data_ptr()) != (
            shard,
            shard + self.row_bytes,
        ):
            raise RuntimeError("Engram pool table parameters must not be replaced")
        self.pool.ready(self.table)

    def lookup(
        self, indices: torch.Tensor, out: torch.Tensor, background: bool = False
    ) -> None:
        """Look up every head of `indices` into BF16 `out`."""
        assert self.table is not None and out.dtype == torch.bfloat16
        rows = indices.shape[0] * self.n_hash_cols
        if not rows:
            return
        packed = self.pool.lookup([self.table], indices[:, None])
        _dequant_rows_kernel[(triton.cdiv(rows, 16),)](
            packed,
            out,
            rows,
            ROW_STRIDE=self.table.row_stride,
            DIM=self.dim,
            PACKED_FP4=self.packed_fp4,
            BLOCK=16,
        )


class PoolEngram(Engram):
    """An Engram layer whose table is split over an Engram pool."""

    def _create_embedding(
        self, layout: EngramLayout, layer_hash_index: int
    ) -> ParallelEngramEmbedding:
        # One fetch reads every Engram layer's table.
        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        max_tokens = get_current_vllm_config().scheduler_config.max_num_batched_tokens
        pool = get_engram_pool(
            max_tokens * n_hash_cols,
            layout.row_bytes + layout.head_dim // SCALE_BLOCK,
            len(layout.num_embeddings),
        )
        return PoolEngramEmbedding(
            layout.num_embeddings[layer_hash_index],
            layout.head_dim,
            tuple(size for order in layout.primes[layer_hash_index] for size in order),
            layout.packed_fp4,
            pool,
            f"layer{layout.layer_ids[layer_hash_index]}",
        )


class PoolEngramBatch(EngramBatch):
    """Stage every layer's MXFP8 WKV input from one pool fetch per step: each
    layer's rows are unpacked as they land (`EngramPrefetch`)."""

    def __init__(self, layers: list[Engram], max_tokens: int) -> None:
        first = layers[0].layer_hash_index
        assert [layer.layer_hash_index for layer in layers] == list(
            range(first, first + len(layers))
        )
        self.layers, self.first = layers, first
        table = layers[0].embed_tokens
        assert isinstance(table, PoolEngramEmbedding)
        self.peer_exchange = None
        device = layers[0].staged_rows.device
        width = table.n_hash_cols * table.dim
        self.fp8_values = torch.empty(
            len(layers), max_tokens, width, dtype=torch.uint8, device=device
        )
        self.fp8_scales = torch.empty(
            len(layers),
            mxfp8_scale_bytes(max_tokens, width),
            dtype=torch.uint8,
            device=device,
        )
        tables = [layer.embed_tokens.table for layer in layers]
        self.prefetch = EngramPrefetch(table.pool, tables, self._unpack)
        self._num_tokens = 0
        for layer in layers:
            layer._batch = self
            layer.staged_rows = layer.staged_rows.new_empty(
                (0, *layer.staged_rows.shape[1:])
            )

    @classmethod
    def create(cls, layers: list[Engram], max_tokens: int) -> "EngramBatch | None":
        if (
            not layers
            or any(layer.embed_tokens.packed_fp4 for layer in layers)
            or layers[0].embed_tokens.dim % 128
            or get_current_vllm_config().lora_config is not None
            or not can_fuse_query_quant([layer.wkv for layer in layers])
        ):
            return None
        return cls(layers, max_tokens)

    def prepare_embeddings(self, hashes: torch.Tensor) -> None:
        self._num_tokens = hashes.shape[0]
        self.prefetch.start(hashes[:, self.first : self.first + len(self.layers)])

    def _unpack(self, fetch: EngramFetch, index: int) -> None:
        if not self._num_tokens:
            return
        rows, slots, ready, placed = fetch.landed(index)
        table = self.layers[0].embed_tokens
        heads = table.n_hash_cols
        _unpack_fp8_kernel[(triton.cdiv(self._num_tokens, 128) * 128 * heads // 16,)](
            rows,
            slots,
            ready,
            self.fp8_values[index],
            self.fp8_scales[index],
            self._num_tokens,
            index,
            HEADS=heads,
            DIM=table.dim,
            ROW_STRIDE=fetch.row_stride,
            PLACED=placed,
            BLOCK=16,
        )

    def wkv_input(self, layer: Engram, num_tokens: int) -> QuantizedActivation:
        self.prefetch.wait(layer.layer_hash_index - self.first)
        return super().wkv_input(layer, num_tokens)
