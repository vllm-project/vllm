# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 indexer's prefill rows split over the TP ranks, on SM80
and SM90.

Under tensor parallelism every rank holds the whole indexer: it scores each
prefill row of a step against the row's compressed keys and keeps the row's
top-k. The rows are independent, so the ranks can split them instead: each
rank scores a contiguous share, balanced by keys, and the ranks exchange the
int32 top-k rows. vLLM's shared indexer no longer splits its rows (prefill
is token-sharded in the sparse MLA instead, which DeepSeek-V4.1's attention
layers do not use), so the SM80 and SM90 indexers keep the split here; their
metadata builders decide when a step splits.

* `RowShardedPrefillMetadata` is the indexer's prefill metadata with the
  ranks' shares (`row_shard_sizes`), which `balanced_prefill_row_shard`
  computes;
* `PrefillRowShard` narrows a prefill chunk, or a whole step's chunks, to
  one rank's rows, and gathers the ranks' top-k rows back into step order;
* `prefill_row_sharding_supported` says where the split may run, and
  `reserve_exchange_memory` reserves the exchange's transient memory in the
  memory profile, whose dummy runs never reach it.
"""

import dataclasses

import torch

import vllm.envs as envs
from vllm.config import CUDAGraphMode, VllmConfig, get_current_vllm_config_or_none
from vllm.distributed import get_tensor_model_parallel_rank, get_tp_group
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerPrefillChunkMetadata,
    DeepseekV32IndexerPrefillMetadata,
)


@dataclasses.dataclass
class RowShardedPrefillMetadata(DeepseekV32IndexerPrefillMetadata):
    """The indexer's prefill metadata with the TP ranks' contiguous shares of
    its rows; None when every rank scores every row."""

    row_shard_sizes: list[int] | None = None

    @classmethod
    def of(
        cls, prefill: DeepseekV32IndexerPrefillMetadata, sizes: list[int] | None
    ) -> "RowShardedPrefillMetadata":
        """`prefill`, as built by the stock builder, with the shares
        `sizes`."""
        fields = {f.name: getattr(prefill, f.name) for f in dataclasses.fields(prefill)}
        return cls(**fields, row_shard_sizes=sizes)


def balanced_prefill_row_shard(
    seq_lens_cpu: torch.Tensor,
    query_lens_cpu: torch.Tensor,
    compress_ratio: int,
    tp_size: int,
) -> list[int] | None:
    """The `tp_size` ranks' contiguous shares of a step's prefill rows,
    balanced by their compressed keys, from the prefill requests' sequence
    lengths and new tokens; None with fewer rows than ranks."""
    num_rows = int(query_lens_cpu.sum())
    if tp_size < 2 or num_rows < tp_size:
        return None

    query_lens = query_lens_cpu.to(torch.int64)
    first_key = torch.repeat_interleave(
        seq_lens_cpu.to(torch.int64) - query_lens + 1, query_lens
    )
    row_in_request = torch.arange(num_rows) - torch.repeat_interleave(
        torch.cumsum(query_lens, 0) - query_lens, query_lens
    )
    cost = torch.cumsum((first_key + row_in_request) // compress_ratio, 0)
    total = int(cost[-1]) if cost.numel() else 0
    if total <= 0:
        return None

    targets = torch.arange(1, tp_size) * total // tp_size
    bounds = [0, *torch.searchsorted(cost, targets).tolist(), num_rows]
    # Keep at least one row per rank; this also handles repeated costs from
    # the compression.
    for i in range(1, tp_size):
        bounds[i] = max(bounds[i], bounds[i - 1] + 1)
    for i in range(tp_size - 1, 0, -1):
        bounds[i] = min(bounds[i], bounds[i + 1] - 1)
    return [bounds[i + 1] - bounds[i] for i in range(tp_size)]


def prefill_row_sharding_supported(
    vllm_config: VllmConfig, dcp_world_size: int, use_pcp: bool, tp_size: int
) -> bool:
    """Whether a step's prefill rows may split over the TP ranks: the exchange
    runs on PyNCCL outside FULL CUDA graphs, without context parallelism."""
    cudagraph_mode = vllm_config.compilation_config.cudagraph_mode or CUDAGraphMode.NONE
    return (
        current_platform.is_cuda()
        and dcp_world_size == 1
        and not use_pcp
        and tp_size > 1
        and not envs.VLLM_DISABLE_PYNCCL
        and not envs.VLLM_USE_NCCL_SYMM_MEM
        and not envs.VLLM_BATCH_INVARIANT
        and cudagraph_mode.mixed_mode() != CUDAGraphMode.FULL
    )


def reserve_exchange_memory(
    topk_indices_buffer: torch.Tensor, device: torch.device
) -> None:
    """Allocate the exchange's worst-case transient memory for the memory
    profile: this rank's staged rows and the gathered tensor, each bounded by
    the top-k buffer."""
    cfg = get_current_vllm_config_or_none()
    if cfg is None or not prefill_row_sharding_supported(
        cfg, 1, False, get_tp_group().world_size
    ):
        return
    rows, width = topk_indices_buffer.shape[:2]
    _ = torch.empty(
        2 * rows * width * topk_indices_buffer.element_size(),
        dtype=torch.uint8,
        device=device,
    )


class PrefillRowShard:
    """One TP rank's contiguous share of a step's prefill rows.

    The rank scores only its share, and the group then exchanges the final
    int32 top-k rows instead of the logits."""

    def __init__(self, num_decode_tokens: int, sizes: list[int], rank: int):
        self.num_decode_tokens = num_decode_tokens
        self.sizes = sizes
        self.start = num_decode_tokens + sum(sizes[:rank])
        self.stop = self.start + sizes[rank]

    @classmethod
    def from_metadata(
        cls, prefill: DeepseekV32IndexerPrefillMetadata, num_decode_tokens: int
    ) -> "PrefillRowShard | None":
        """This rank's share of the step whose prefill metadata is
        `prefill`; None when the step does not split."""
        sizes = getattr(prefill, "row_shard_sizes", None)
        if sizes is None:
            return None
        return cls(num_decode_tokens, sizes, get_tensor_model_parallel_rank())

    def narrow(
        self, chunk: DeepseekV32IndexerPrefillChunkMetadata
    ) -> tuple[int, int, torch.Tensor, torch.Tensor] | None:
        """This rank's row window of `chunk` and the window's causal key
        bounds; None when the chunk holds none of its rows."""
        start = max(chunk.token_start, self.start)
        end = min(chunk.token_end, self.stop)
        if start >= end:
            return None
        lo, hi = start - chunk.token_start, end - chunk.token_start
        return start, end, chunk.cu_seqlen_ks[lo:hi], chunk.cu_seqlen_ke[lo:hi]

    def narrow_prefill(
        self, prefill: DeepseekV32IndexerPrefillMetadata
    ) -> DeepseekV32IndexerPrefillMetadata:
        """A copy of `prefill` whose chunks hold only this rank's rows. A
        chunk without any keeps an empty row window rather than going away:
        it still gathers its keys, which a continuation chunk reuses."""
        chunks = []
        for chunk in prefill.chunks:
            narrowed = self.narrow(chunk)
            if narrowed is None:
                start = end = chunk.token_start
                ks, ke = chunk.cu_seqlen_ks[:0], chunk.cu_seqlen_ke[:0]
            else:
                start, end, ks, ke = narrowed
            chunks.append(
                dataclasses.replace(
                    chunk,
                    token_start=start,
                    token_end=end,
                    cu_seqlen_ks=ks,
                    cu_seqlen_ke=ke,
                )
            )
        return dataclasses.replace(prefill, chunks=chunks)

    def exchange_topk(self, topk_buffer: torch.Tensor, width: int) -> None:
        """Gather every rank's top-k rows back into step order."""
        # The local input must outlive the async PyNCCL gather's enqueued copy.
        local = topk_buffer[self.start : self.stop, :width].contiguous()
        end = self.num_decode_tokens + sum(self.sizes)
        topk_buffer[self.num_decode_tokens : end, :width] = get_tp_group().all_gatherv(
            local, dim=0, sizes=self.sizes
        )
