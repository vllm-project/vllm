# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Symmetric-memory Engram exchange for slots below ENGRAM_A2A_MIN_SLOT."""

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.distributed import get_engram_dp_group
from vllm.models.deepseek_v41.common.engram import (
    ENGRAM_A2A_MIN_SLOT,
    _engram_unpack_fp8_kernel,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.models.deepseek_v41.common.engram import Engram


@triton.jit(do_not_specialize=["num_tokens"])
def _peer_lookup_kernel(
    metadata,
    id_ptrs,
    output,
    num_tokens,
    HEADS: tl.constexpr,
    LAYERS: tl.constexpr,
    DP: tl.constexpr,
    RANK: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Look up this rank's heads for every peer's tokens into packed rows."""
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    local_heads: tl.constexpr = HEADS // DP
    valid = row < num_tokens * LAYERS * HEADS
    token = row // (LAYERS * local_heads)
    layer, head = row // local_heads % LAYERS, row % local_heads
    ids = tl.load(id_ptrs + token // num_tokens, valid, other=0).to(
        tl.pointer_type(tl.int32)
    )
    index = tl.load(
        ids + (token % num_tokens * LAYERS + layer) * HEADS + RANK * local_heads + head,
        valid,
        other=-1,
    ).to(tl.int64)
    # Per-layer metadata: weight pointer, scale pointer, vocab start, vocab end.
    table = metadata + layer * 4
    start, end = tl.load(table + 2), tl.load(table + 3)
    owned = valid & (index >= start) & (index < end)
    local = tl.where(owned, index - start, 0)
    weight = tl.multiple_of(tl.load(table), DIM).to(tl.pointer_type(tl.uint8))
    scales = tl.load(table + 1)
    scales = tl.multiple_of(scales, DIM // 32).to(tl.pointer_type(tl.uint8))
    col = tl.arange(0, DIM)
    value = tl.load(
        weight[:, None] + local[:, None] * DIM + col[None, :], owned[:, None], other=0
    )
    dst = row.to(tl.int64) * (DIM + DIM // 32)
    tl.store(output + dst[:, None] + col[None, :], value, valid[:, None])
    sc = tl.arange(0, DIM // 32)
    scale = tl.load(
        scales[:, None] + local[:, None] * (DIM // 32) + sc[None, :],
        owned[:, None],
        other=127,
    )
    tl.store(output + dst[:, None] + DIM + sc[None, :], scale, valid[:, None])


class EngramPeerExchange(nn.Module):
    """Two phase barriers fence reuse of each other's buffers across forwards.

    Pointer tables are buffers so level-2 sleep restores them on wake-up.
    """

    def __init__(
        self,
        layers: list["Engram"],
        values: torch.Tensor,
        scales: torch.Tensor,
        max_tokens: int,
    ):
        import torch.distributed._symmetric_memory as symm_mem

        super().__init__()
        group = get_engram_dp_group()
        assert group is not None
        self.dp_size, self.dp_rank = group.world_size, group.rank_in_group
        self.values, self.scales = values, scales
        self.num_layers = len(layers)
        self.hash_start = layers[0].layer_hash_index
        table = layers[0].embed_tokens
        self.heads, self.dim = table.n_hash_cols, table.dim
        self.ids = symm_mem.empty(
            (max_tokens, self.num_layers, self.heads),
            dtype=torch.int32,
            device=values.device,
        )
        self.rows = symm_mem.empty(
            (
                max_tokens * self.dp_size,
                self.num_layers,
                self.heads // self.dp_size,
                self.dim + self.dim // 32,
            ),
            dtype=torch.uint8,
            device=values.device,
        )
        self.id_handle = symm_mem.rendezvous(self.ids, group.device_group)
        self.row_handle = symm_mem.rendezvous(self.rows, group.device_group)
        metadata = []
        for layer in layers:
            table = layer.embed_tokens
            weight, scale = table._storage()
            metadata.append(
                (
                    weight.data_ptr(),
                    scale.data_ptr(),
                    table.vocab_start_idx,
                    table.vocab_end_idx,
                )
            )
        tables = {
            "metadata": metadata,
            "id_ptrs": self.id_handle.buffer_ptrs,
            "row_ptrs": self.row_handle.buffer_ptrs,
        }
        for name, data in tables.items():
            self.register_buffer(
                name,
                torch.tensor(data, dtype=torch.int64, device=values.device),
                persistent=False,
            )

    @classmethod
    def create(
        cls,
        layers: list["Engram"],
        values: torch.Tensor,
        scales: torch.Tensor,
        max_tokens: int,
    ) -> "EngramPeerExchange | None":
        # The Engram DP group is node-local and every rank agrees on these.
        group = get_engram_dp_group()
        assert group is not None
        if not (
            current_platform.is_device_capability_family(100)
            and group.world_size in (2, 4, 8)
            and layers[0].embed_tokens.n_hash_cols % group.world_size == 0
        ):
            return None
        physical_id = current_platform.visible_device_id_to_physical_device_id(
            torch.accelerator.current_device_index()
        )
        physical_ids = [None] * group.world_size
        dist.all_gather_object(physical_ids, physical_id, group=group.cpu_group)
        if not current_platform.is_fully_connected(physical_ids):
            return None
        return cls(layers, values, scales, max_tokens)

    @eager_break_during_capture
    def prepare(self, hashes: torch.Tensor, slot: int) -> None:
        """Stage every layer's MXFP8 WKV input for this rank's tokens."""
        tokens = hashes.shape[0]
        assert 0 <= tokens <= slot <= self.ids.shape[0] < ENGRAM_A2A_MIN_SLOT
        if slot == 0:
            return
        hash_end = self.hash_start + self.num_layers
        self.ids[:tokens].copy_(hashes[:, self.hash_start : hash_end])
        if tokens < slot:
            self.ids[tokens:slot].fill_(-1)
        # The preceding rows barrier completed every peer's hash reads. This
        # barrier completes their previous output reads before rows are reused.
        self.id_handle.barrier()
        _peer_lookup_kernel[(triton.cdiv(slot * self.num_layers * self.heads, 16),)](
            self.metadata,
            self.id_ptrs,
            self.rows,
            slot,
            HEADS=self.heads,
            LAYERS=self.num_layers,
            DP=self.dp_size,
            RANK=self.dp_rank,
            DIM=self.dim,
            BLOCK=16,
        )
        self.row_handle.barrier()
        _engram_unpack_fp8_kernel[(128 * self.heads // 16, self.num_layers)](
            self.row_ptrs,
            self.values,
            self.scales,
            slot,
            self.values.stride(0),
            self.scales.stride(0),
            HEADS=self.heads,
            LOCAL_HEADS=self.heads // self.dp_size,
            LAYERS=self.num_layers,
            RANK=self.dp_rank,
            DIM=self.dim,
            BLOCK=16,
            PEER=True,
        )
