# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep a long PCP + DCP prefill's whole indexer K on every rank.

Enabled by `ParallelConfig.pcp_dcp_indexer_k_replica`. Every indexer layer keeps
the FP8 K values and scales of the batch's single request in token order, in a
buffer of `max_model_len + 4096` rows that outlives the step. Each chunk's new K
rows reach every rank through the PCP gather of the new rows that precedes the
cache write, so every rank quantizes them into its replica with the cache-write
kernel, and the indexer reads its K from the replica: no DCP K all-gather,
de-interleave or host sync. A request whose earlier rows are not in the replica
(a prefix cache hit, a resumed request, rows computed while the replica was
invalid) builds it once from the sharded cache with the serial gather.

Cost: `(max_model_len + 4096) * (head_dim + 4)` bytes per indexer layer on every rank,
whatever the DCP width. It is allocated while the model is built and held for the life
of the process, so it comes out of the KV cache capacity.

The decision depends only on the global schedule and on replica state that every rank
updates the same way, so all ranks issue the same collectives. Anything not eligible
invalidates the layer's replica and runs the existing path.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from vllm import _custom_ops as ops
from vllm.distributed.parallel_state import get_dcp_group, get_world_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import current_stream
from vllm.v1.attention.ops.cp_common import scheduled_step
from vllm.v1.attention.ops.cuda_direct import (
    DirectKernel,
    KernelLaunch,
    cuda_driver,
    direct_launch,
)

logger = init_logger(__name__)

# FP32 (or packed UE8M0) scale bytes per K row.
SCALE_BYTES = 4
SLOT_BLOCK = 1024
# Rows of slack over max_model_len for the shard padding of the interleave.
K_CAPACITY_SLACK_ROWS = 4096


@triton.jit(
    do_not_specialize=["start", "rows"],
    do_not_specialize_on_alignment=["restore_ptr"],
)
def _token_slots_kernel(
    restore_ptr,
    slots_ptr,
    start: tl.int64,
    rows: tl.int32,
    BLOCK: tl.constexpr,
):
    """`slots[restore[g]] = start + g` for `g < rows`: the token position of
    every gathered row. Rows no token maps to keep their value."""
    g = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = g < rows
    row = tl.load(restore_ptr + g, mask=live, other=0)
    tl.store(slots_ptr + row, start + g.to(tl.int64), mask=live)


def k_replica_chunk(
    prefill: Any,
    num_decode_tokens: int,
    local_rows: int,
    pcp_world_size: int,
    dcp_world_size: int,
    capacity_rows: int,
    global_new_rows: int | None,
    global_end: int | None,
) -> tuple[Any, str]:
    """Return the chunk whose K a replica can serve, or None and why not.

    Args:
        prefill: The indexer prefill metadata of the layer.
        num_decode_tokens: Decode tokens of the layer's batch.
        local_rows: This rank's rows of the batch.
        pcp_world_size: PCP ranks.
        dcp_world_size: DCP ranks.
        capacity_rows: K rows a replica holds.
        global_new_rows: Rows of the global batch before PCP partitioning.
        global_end: Context end of the global batch's request, None unless the
            batch has exactly one. PCP splits that request into several local
            segments, so the local metadata cannot tell.

    Returns:
        `(chunk, "eligible")`, or `(None, reason)` for the serial path.

    """
    if prefill is None or not prefill.chunks:
        return None, "no_prefill"
    if num_decode_tokens != 0:
        return None, "decode_tokens"
    if len(prefill.chunks) != 1:
        return None, "many_chunks"
    chunk = prefill.chunks[0]
    if chunk.pcp_deinterleave_idx is None or chunk.skip_kv_gather:
        return None, "no_gather"
    if global_end is None:
        return None, "many_requests"
    if chunk.local_total_seq_lens * dcp_world_size > capacity_rows:
        return None, "over_capacity"
    if global_new_rows is None or global_new_rows % (2 * pcp_world_size) != 0:
        return None, "uneven_partition"
    if global_new_rows != pcp_world_size * local_rows:
        return None, "replicated_rows"
    if chunk.token_end - chunk.token_start != local_rows:
        return None, "partial_query"
    if global_new_rows > global_end or global_end > chunk.total_seq_lens:
        return None, "short_context"
    # Compressed K rows are not one per token.
    if prefill.max_prefill_seq_len != chunk.max_seqlen_k:
        return None, "compressed_k"
    return chunk, "eligible"


def replica_bytes(rows: int, head_dim: int, layers: int) -> int:
    """Bytes the replicas of `layers` indexer layers hold on each rank."""
    return rows * (head_dim + SCALE_BYTES) * layers


def replica_mode(
    req_id: str, start: int, held_req_id: str | None, held_rows: int
) -> str:
    """`append` the chunk to the replica, or `build` it from the cache first."""
    if start == 0 or (req_id == held_req_id and held_rows == start):
        return "append"
    return "build"


@dataclass
class LayerReplica:
    """One indexer layer's K in token order, laid out as a single cache block.

    `indexer_k_quant_and_cache` writes row `i` of a `[1, capacity, D + 4]` cache
    to values `[i * D, (i + 1) * D)` and its scale to `capacity * D + 4 * i`, so the
    values and the scales are each contiguous, as the top-k kernels read them.
    """

    buffer: torch.Tensor
    capacity: int
    head_dim: int
    req_id: str | None = None
    # Rows [0, rows) hold req_id's K; -1 when nothing is held.
    rows: int = -1

    def __post_init__(self) -> None:
        from vllm.platforms import current_platform

        values_bytes = self.capacity * self.head_dim
        self.cache = self.buffer.view(1, self.capacity, self.head_dim + SCALE_BYTES)
        self.values = (
            self.buffer[:values_bytes]
            .view(current_platform.fp8_dtype())
            .view(self.capacity, self.head_dim)
        )
        self.scales = self.buffer[values_bytes:].view(self.capacity, SCALE_BYTES)

    def invalidate(self) -> None:
        self.req_id, self.rows = None, -1


@dataclass
class ReplicaDecision:
    chunk: Any
    mode: str
    req_id: str
    start: int
    end: int
    restore_idx: torch.Tensor


class KReplica:
    """The per-layer replicas of one worker process."""

    def __init__(
        self, pcp_world_size: int, dcp_world_size: int, capacity_rows: int
    ) -> None:
        self.pcp_world_size = pcp_world_size
        self.dcp_world_size = dcp_world_size
        self.capacity_rows = capacity_rows
        self.layers: dict[int, LayerReplica] = {}
        self._indexers: dict[int, Any] = {}
        self._step: torch.Tensor | None = None
        self._decisions: dict[int, ReplicaDecision | None] = {}
        self._slots: torch.Tensor | None = None
        self._slot_buffer: torch.Tensor | None = None
        self._slots_launch: KernelLaunch | None = None
        self._slots_kernel: DirectKernel | None = None
        self._slots_loaded = False

    def register_layer(self, layer_id: int, indexer: Any, device: torch.device) -> None:
        """Allocate `layer_id`'s replica; a rank that cannot leaves it to every rank.

        Allocated while the model is built, so the KV cache is sized around it.
        """
        head_dim = indexer.head_dim
        nbytes = replica_bytes(self.capacity_rows, head_dim, 1)
        if not self._slots_loaded:
            # Compiled and loaded once while the model is built, so no step pays
            # for it.
            self._slots_loaded = True
            self._slots_kernel = direct_launch(
                _token_slots_kernel,
                {"restore_ptr": torch.int64, "slots_ptr": torch.int64},
                BLOCK=SLOT_BLOCK,
            )
        buffer = None
        if self._slots_kernel is None:
            logger.warning_once(
                "The indexer K replica's kernel cannot be launched directly; "
                "every layer gathers its K."
            )
        else:
            try:
                buffer = torch.zeros(nbytes, dtype=torch.uint8, device=device)
            except (RuntimeError, ValueError) as error:
                logger.warning(
                    "Indexer K replica of layer %d failed: %s", layer_id, error
                )
        if not self._all_ranks_agree(buffer is not None):
            logger.warning_once(
                "A rank could not allocate an indexer K replica; those layers "
                "gather their K."
            )
            return
        assert buffer is not None and self._slots_kernel is not None
        self._indexers[layer_id] = indexer
        self.layers[layer_id] = LayerReplica(buffer, self.capacity_rows, head_dim)
        if self._slots_launch is None:
            # Launched once (for zero rows) while the model is built.
            anchor = buffer.data_ptr()
            self._slots_launch = self._slots_kernel.bind(
                restore_ptr=anchor, slots_ptr=anchor
            )
            self._slots_launch(current_stream().cuda_stream, 1)
        logger.info_once(
            "Indexer K replica: %.1f MiB per indexer layer on every rank "
            "(%d rows x %d B), taken from the KV cache capacity.",
            nbytes / 2**20,
            self.capacity_rows,
            head_dim + SCALE_BYTES,
        )

    @staticmethod
    def _all_ranks_agree(ok: bool) -> bool:
        group = get_world_group()
        flags = [torch.zeros(1, dtype=torch.int64) for _ in range(group.world_size)]
        dist.all_gather(
            flags, torch.tensor([int(ok)], dtype=torch.int64), group=group.cpu_group
        )
        return all(int(f.item()) == 1 for f in flags)

    def decide(self, layer: int, local_rows: int) -> ReplicaDecision | None:
        """Whether `layer`'s replica serves this step, decided once per step.

        A step without a decision invalidates the layer's replica: its new rows are
        not appended.
        """
        from vllm.v1.worker.gpu.pcp_manager import active_pcp_manager

        replica = self.layers.get(layer)
        if replica is None:
            return None
        manager = active_pcp_manager()
        restore_idx = None if manager is None else manager.hidden_restore_idx
        if (
            restore_idx is not None
            and restore_idx is self._step
            and layer in self._decisions
        ):
            # Decided earlier in this step: the manager makes a new restore
            # index for every batch, and a batch runs one forward.
            return self._decisions[layer]
        metadata = get_forward_context().attn_metadata
        if (
            not isinstance(metadata, dict)
            or restore_idx is None
            or not scheduled_step()
        ):
            # A capture or warmup forward runs a dummy batch: its rows must not
            # reach the replica.
            replica.invalidate()
            return None
        assert manager is not None
        new_rows = restore_idx.shape[0]
        # The slot kernel reads the restore index as contiguous int64.
        readable = restore_idx.dtype == torch.int64 and restore_idx.is_contiguous()
        if restore_idx is not self._step:
            self._step, self._decisions, self._slots = restore_idx, {}, None
        if layer in self._decisions:
            return self._decisions[layer]
        decision = None
        layer_metadata: Any = metadata.get(self._indexers[layer].k_cache.prefix)
        end = manager.single_request_end
        if layer_metadata is not None:
            chunk, _ = k_replica_chunk(
                layer_metadata.prefill,
                layer_metadata.num_decode_tokens,
                local_rows,
                self.pcp_world_size,
                self.dcp_world_size,
                self.capacity_rows,
                new_rows,
                end,
            )
            if chunk is not None and readable:
                assert end is not None
                req_id = manager.global_batch.req_ids[0]
                start = end - new_rows
                mode = replica_mode(req_id, start, replica.req_id, replica.rows)
                decision = ReplicaDecision(chunk, mode, req_id, start, end, restore_idx)
        # Valid again only once this step's rows are in.
        replica.invalidate()
        self._decisions[layer] = decision
        return decision

    def take(
        self,
        layer: int,
        chunk: Any,
        gathered_k: torch.Tensor,
        quant_block_size: int,
        scale_fmt: str,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Bring `layer`'s replica up to this chunk and return its K, or None."""
        if layer not in self.layers or gathered_k.shape[0] % self.pcp_world_size:
            return None
        decision = self.decide(layer, gathered_k.shape[0] // self.pcp_world_size)
        if decision is None:
            return None
        if (
            decision.chunk is not chunk
            or gathered_k.shape[0] != decision.restore_idx.shape[0]
        ):
            return None
        replica = self.layers[layer]
        if decision.mode == "append":
            ops.indexer_k_quant_and_cache(
                gathered_k,
                replica.cache,
                self._slots_of(decision),
                quant_block_size,
                scale_fmt,
            )
        else:
            self._build(layer, replica, chunk)
        replica.req_id, replica.rows = decision.req_id, decision.end
        rows = chunk.pcp_deinterleave_idx.shape[0]
        return replica.values[:rows], replica.scales[:rows]

    def _slots_of(self, decision: ReplicaDecision) -> torch.Tensor:
        """Each gathered row's token position, shared by the step's layers: -1
        everywhere, then the kernel's positions, on the current stream."""
        if self._slots is None:
            idx = decision.restore_idx
            rows = idx.shape[0]
            buffer = self._slot_buffer
            if buffer is None or buffer.shape[0] < rows:
                buffer = self._slot_buffer = torch.empty_like(idx)
            slots = buffer[:rows]
            launch = self._slots_launch
            assert launch is not None and rows > 0
            stream = current_stream().cuda_stream
            cuda_driver().fill(buffer.data_ptr(), 0xFF, 8 * rows, stream)
            launch.update(
                restore_ptr=idx.data_ptr(),
                slots_ptr=buffer.data_ptr(),
                start=decision.start,
                rows=rows,
            )(stream, -(-rows // SLOT_BLOCK))
            self._slots = slots
        return self._slots

    def _build(self, layer: int, replica: LayerReplica, chunk: Any) -> None:
        """The serial gather of the cache, de-interleaved into the replica."""
        kv_cache = self._indexers[layer].k_cache.kv_cache
        rows = chunk.max_local_total_seq_lens
        values = torch.empty(
            (rows, replica.head_dim), dtype=replica.values.dtype, device=kv_cache.device
        )
        scales = torch.empty(
            (rows, SCALE_BYTES), dtype=torch.uint8, device=kv_cache.device
        )
        ops.cp_gather_indexer_k_quant_cache(
            kv_cache, values, scales, chunk.block_table, chunk.local_cu_seq_lens
        )
        idx = chunk.pcp_deinterleave_idx
        group = get_dcp_group()
        for shard, out in ((values, replica.values), (scales, replica.scales)):
            shard = shard[: chunk.local_total_seq_lens]
            gathered = shard.new_empty(
                (group.world_size * shard.shape[0], *shard.shape[1:])
            )
            dist.all_gather_into_tensor(gathered, shard, group=group.device_group)
            torch.index_select(gathered, 0, idx, out=out[: idx.shape[0]])


def k_replica_supported() -> bool:
    """A KV connector writes the cache outside the forward, behind the replica."""
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    if config is not None and config.kv_transfer_config is not None:
        logger.warning_once(
            "pcp_dcp_indexer_k_replica keeps no indexer K replica with a KV connector."
        )
        return False
    return True


def is_mtp_layer(layer_id: int) -> bool:
    """MTP layers follow the target model's layers and get no replica."""
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    num_layers = getattr(
        getattr(getattr(config, "model_config", None), "hf_config", None),
        "num_hidden_layers",
        None,
    )
    return num_layers is not None and layer_id >= num_layers


@dataclass
class IndexerKReplica:
    """Per-call handle handed to `sparse_attn_indexer` for one indexer layer."""

    replica: KReplica
    layer: int

    def take_k(
        self,
        chunk: Any,
        gathered_k: torch.Tensor,
        quant_block_size: int,
        scale_fmt: str,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return the layer's K replica brought up to this chunk, or None for
        the serial gather."""
        return self.replica.take(
            self.layer, chunk, gathered_k, quant_block_size, scale_fmt
        )


_K_REPLICA: KReplica | None = None
# sparse_attn_indexer is a registered custom op: non-tensor state cannot be an argument.
_INDEXER_K_REPLICA: IndexerKReplica | None = None


def set_indexer_k_replica(replica: IndexerKReplica) -> None:
    """Hand `replica` to the next `sparse_attn_indexer` call on this thread."""
    global _INDEXER_K_REPLICA
    _INDEXER_K_REPLICA = replica


def pop_indexer_k_replica() -> IndexerKReplica | None:
    """Take what `set_indexer_k_replica` handed over, if anything."""
    global _INDEXER_K_REPLICA
    replica, _INDEXER_K_REPLICA = _INDEXER_K_REPLICA, None
    return replica


def k_replica_for_layer(
    enabled: bool,
    layer_id: int,
    indexer: Any,
    use_pcp: bool,
    pcp_world_size: int,
    dcp_world_size: int,
    max_model_len: int,
) -> KReplica | None:
    """Register an indexer layer with the process's K replica, when enabled."""
    global _K_REPLICA
    if not enabled or not use_pcp or dcp_world_size <= 1:
        return None
    if _K_REPLICA is None:
        if not k_replica_supported():
            return None
        _K_REPLICA = KReplica(
            pcp_world_size,
            dcp_world_size,
            max_model_len + K_CAPACITY_SLACK_ROWS,
        )
    if indexer is not None and not is_mtp_layer(layer_id):
        device = torch.device("cuda", torch.accelerator.current_device_index())
        _K_REPLICA.register_layer(layer_id, indexer, device)
    return _K_REPLICA
