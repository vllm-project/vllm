# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp PLE tables split by rows over the HBM of NVLink-connected DP ranks."""

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_engram_dp_group, get_tp_group
from vllm.distributed.device_communicators.nvlink_fabric import has_cross_node_nvlink
from vllm.distributed.parallel_state import GroupCoordinator, in_the_same_node_as
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    _e2m1_inline,
    _e4m3_inline,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from ..common.ngram_embedding import (
    Qwen4ExpPLEEmbedding,
    Qwen4ExpPLENvFp4EmbeddingMethod,
)

logger = init_logger(__name__)


def get_ple_peer_group(vllm_config: VllmConfig) -> GroupCoordinator | None:
    """The Engram DP group if its GPUs can split a GPU PLE table's rows: TP=1
    and NVLink between them."""
    engram_config = vllm_config.engram_config
    group = get_engram_dp_group()
    if (
        engram_config is None
        or engram_config.cpu_offload
        or engram_config.disk_offload
        or group is None
        or group.world_size == 1
        or get_tp_group().world_size > 1
        # Weight offloading moves the slice out of symmetric memory, and
        # sharded-state checkpoints store one table per TP rank, not DP rank.
        or vllm_config.offload_config.uva.cpu_offload_gb > 0
        or vllm_config.offload_config.prefetch.offload_group_size > 0
        or vllm_config.load_config.load_format
        in ("sharded_state", "runai_streamer_sharded")
    ):
        return None
    # Cross-node groups come from embedding_across_dp and need one NVLink fabric.
    if not all(in_the_same_node_as(group.cpu_group, source_rank=0)):
        return group if has_cross_node_nvlink(group.cpu_group) else None
    physical_id = current_platform.visible_device_id_to_physical_device_id(
        torch.accelerator.current_device_index()
    )
    physical_ids = [None] * group.world_size
    torch.distributed.all_gather_object(
        physical_ids, physical_id, group=group.cpu_group
    )
    return group if current_platform.is_fully_connected(physical_ids) else None


@triton.jit
def _copy_peer_rows_kernel(
    slice_ptrs,
    ids_ptr,
    output_ptr,
    row_bytes,
    num_rows,
    PEER_ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Copy the rows of `ids` as raw bytes from the slices that hold them.

    Rank r's slice, at `slice_ptrs[r]`, holds rows `[r * PEER_ROWS, (r + 1) *
    PEER_ROWS)`. IDs past `num_rows` read as zeros.
    """
    row = tl.program_id(0).to(tl.int64)
    idx = tl.load(ids_ptr + row).to(tl.int64)
    owned = idx < num_rows
    owner = tl.where(owned, idx // PEER_ROWS, 0)
    rows = tl.load(slice_ptrs + owner).to(tl.pointer_type(tl.uint8))
    offsets = tl.arange(0, BLOCK_D)
    in_row = offsets < row_bytes
    values = tl.load(
        rows + (idx - owner * PEER_ROWS) * row_bytes + offsets,
        mask=owned & in_row,
        other=0,
    )
    tl.store(output_ptr + row * row_bytes + offsets, values, mask=in_row)


@triton.jit
def _decode_nvfp4_peer_rows_kernel(
    weight_slice_ptrs,
    scale_slice_ptrs,
    global_scale_ptr,
    ids_ptr,
    output_ptr,
    embedding_dim,
    num_rows,
    PEER_ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Decode the NVFP4 rows of `ids` from the slices that hold them, as
    `_copy_peer_rows_kernel` copies plain rows."""
    row = tl.program_id(0).to(tl.int64)
    idx = tl.load(ids_ptr + row).to(tl.int64)
    owned = idx < num_rows
    owner = tl.where(owned, idx // PEER_ROWS, 0)
    local_idx = idx - owner * PEER_ROWS
    weights = tl.load(weight_slice_ptrs + owner).to(tl.pointer_type(tl.uint8))
    block_scales = tl.load(scale_slice_ptrs + owner).to(tl.pointer_type(tl.uint8))
    offsets = tl.arange(0, BLOCK_D)
    in_row = offsets < embedding_dim
    mask = owned & in_row
    packed = tl.load(
        weights + local_idx * (embedding_dim // 2) + offsets // 2,
        mask=mask,
        other=0,
    )
    # The low nibble holds the even element and the high nibble the odd one.
    codes = (packed >> ((offsets % 2) * 4)) & 0xF
    scales = _e4m3_inline(
        tl.load(
            block_scales + local_idx * (embedding_dim // 16) + offsets // 16,
            mask=mask,
            other=0,
        )
    )
    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    values = _e2m1_inline(codes) * scales * global_scale
    tl.store(output_ptr + row * embedding_dim + offsets, values, mask=in_row)


class Qwen4ExpPLEPeerEmbedding(Qwen4ExpPLEEmbedding):
    """PLE table whose rows are split over the HBM of NVLink-connected DP ranks.

    Each rank holds one row slice of every table parameter and reads its own
    tokens' rows from the ranks that hold them, so lookups need no collective.
    """

    def __init__(self, *args, peer_group: GroupCoordinator, **kwargs) -> None:
        self.peer_group = peer_group
        # Each table parameter's slice addresses and the handle keeping them
        # mapped, keyed by the address of this rank's slice.
        self.slices: dict[int, tuple[torch.Tensor, object]] = {}
        # Every rank looks up the whole vocabulary, so it is not ETP-sharded.
        super().__init__(*args, parallel_group=get_tp_group(), **kwargs)
        logger.info(
            "PLE table %s split over %d GPUs: %d rows, %.2f GiB on this GPU",
            kwargs["prefix"],
            peer_group.world_size,
            self.local_rows,
            sum(p.nbytes for p in self.parameters() if p.data_ptr() in self.slices)
            / 2**30,
        )

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate this rank's row slice of a table parameter and map the others'."""
        import torch.distributed._symmetric_memory as symm_mem

        group = self.peer_group
        self.peer_rows = triton.cdiv(num_embeddings, group.world_size)
        self.row_start = min(group.rank_in_group * self.peer_rows, num_embeddings)
        self.local_rows = min(self.peer_rows, num_embeddings - self.row_start)
        row_bytes = embedding_dim * dtype.itemsize
        buffer = symm_mem.empty(
            self.peer_rows * row_bytes,
            dtype=torch.uint8,
            device=torch.device("cuda", torch.accelerator.current_device_index()),
        )
        handle = symm_mem.rendezvous(buffer, group.device_group)
        rows = buffer[: self.local_rows * row_bytes].view(dtype)
        rows = rows.view(self.local_rows, embedding_dim)
        slice_ptrs = torch.tensor(
            handle.buffer_ptrs, dtype=torch.int64, device=buffer.device
        )
        self.slices[rows.data_ptr()] = (slice_ptrs, handle)
        return rows

    def slice_ptrs(self, param: torch.Tensor) -> torch.Tensor:
        """The address of every rank's slice of a table parameter."""
        return self.slices[param.data_ptr()][0]

    def _stored_vocab_range(self) -> tuple[int, int]:
        return self.row_start, self.row_start + self.local_rows

    def weight_loader(
        self,
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        checkpoint_start: int | None = None,
    ) -> None:
        if param.ndim != 2:
            super().weight_loader(param, loaded_weight, checkpoint_start)
            return
        # Table rows, also when a reload loads them into a fresh tensor first.
        # A table saved whole starts at row 0.
        self._copy_embedding_shard_(
            param, loaded_weight, checkpoint_start=checkpoint_start or 0
        )

    def finish_loading(self) -> None:
        """Peers read this table's rows directly, so serve only once every rank
        has loaded its own."""
        torch.accelerator.synchronize()
        self.peer_group.barrier()

    def start_prefetch(self, ngram_ids: torch.Tensor) -> None:
        """Peer lookups run in the forward."""
        return None

    def finalize_prefetch(self, total_tokens: int) -> torch.Tensor:
        raise RuntimeError("Peer-split PLE embedding does not support prefetch")

    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Look up every row of `ngram_ids` from the GPUs that hold it."""
        method = self.embedding_method
        ids = ngram_ids.reshape(-1)
        output = torch.empty(
            (*ngram_ids.shape, self.embedding_dim),
            dtype=method.lookup_dtype(self),
            device=ngram_ids.device,
        )
        if not ids.numel():
            return output
        if isinstance(method, Qwen4ExpPLENvFp4EmbeddingMethod):
            _decode_nvfp4_peer_rows_kernel[(ids.numel(),)](
                self.slice_ptrs(self.weight),
                self.slice_ptrs(self.weight_scale),
                self.weight_scale_2,
                ids,
                output,
                self.embedding_dim,
                self.org_vocab_size,
                PEER_ROWS=self.peer_rows,
                BLOCK_D=triton.next_power_of_2(self.embedding_dim),
            )
        else:
            row_bytes = self.embedding_dim * self.weight.element_size()
            _copy_peer_rows_kernel[(ids.numel(),)](
                self.slice_ptrs(self.weight),
                ids,
                output.view(torch.uint8),
                row_bytes,
                self.org_vocab_size,
                PEER_ROWS=self.peer_rows,
                BLOCK_D=triton.next_power_of_2(row_bytes),
            )
        return output
