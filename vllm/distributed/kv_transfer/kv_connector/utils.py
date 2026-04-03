# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
KV cache helper for store.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

import torch

from vllm.config import VllmConfig, get_current_vllm_config, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
    from vllm.v1.kv_cache_interface import KVCacheSpec

logger = init_logger(__name__)

EngineId = str
# block ids as returned by the hybrid KV cache manager. list[list[int]] are allow
# mutability and are for connector internal use only.
BlockIds = tuple[list[int], ...] | list[list[int]]


def get_kv_connector_cache_layout():
    # NOTE (NickLucche) When running disaggregated PD with NIXL, HND layout is
    # used for faster transfer.
    vllm_config = get_current_vllm_config()
    kv_config = vllm_config.kv_transfer_config
    if kv_config is not None:
        connector_cls = KVConnectorFactory.get_connector_class(kv_config)
        required_kvcache_layout = connector_cls.get_required_kvcache_layout(vllm_config)
        if required_kvcache_layout is not None:
            return required_kvcache_layout
        logger.info_once(
            "Connectors do not specify a kv cache layout, defaulting to NHD."
        )
    return "NHD"


class KVOutputAggregator:
    """Utility class to aggregate the output of all workers into a single
    output corresponding to Rank 0 for scheduler."""

    def __init__(self, expected_finished_count: int):
        # Complete transfer tracker. Used to track finished requests
        # [req_id -> n_remaining_workers]
        self._recv_remaining_count = dict[str, int]()
        self._send_remaining_count = dict[str, int]()
        self._expected_finished_count = expected_finished_count

    @classmethod
    def from_connector(cls, connector: "KVConnectorBase", world_size: int):
        return cls(connector.get_finished_count() or world_size)

    def aggregate(
        self, outputs: list[ModelRunnerOutput | None], output_rank: int = 0
    ) -> ModelRunnerOutput | None:
        if not outputs[output_rank]:
            return None

        # Aggregate kv_connector_output from all workers

        def update_finished_set(
            req_ids: set[str] | None,
            remaining_count_dict: dict[str, int],
            finished_set: set[str],
        ) -> None:
            for req_id in req_ids or ():
                remaining_count = remaining_count_dict.get(
                    req_id, self._expected_finished_count
                )
                remaining_count_dict[req_id] = remaining_count - 1
                if remaining_count_dict[req_id] == 0:
                    finished_set.add(req_id)
                    del remaining_count_dict[req_id]

        finished_sending = set[str]()
        finished_recving = set[str]()
        aggregated_kv_connector_stats = None
        aggregated_kv_connector_worker_meta = None
        combined_kv_cache_events = None
        invalid_block_ids = set[int]()
        for model_runner_output in outputs:
            assert model_runner_output is not None
            kv_output = model_runner_output.kv_connector_output
            if not kv_output:
                continue
            # Allow the worker to dynamically update the expected number of
            # finished sending/recving for new requests.
            if (
                kv_output.expected_finished_count > 0
                and kv_output.expected_finished_count != self._expected_finished_count
            ):
                logger.debug(
                    "Expected finished requests updated from %d to %d",
                    self._expected_finished_count,
                    kv_output.expected_finished_count,
                )
                self._expected_finished_count = kv_output.expected_finished_count

            update_finished_set(
                kv_output.finished_sending, self._send_remaining_count, finished_sending
            )
            update_finished_set(
                kv_output.finished_recving, self._recv_remaining_count, finished_recving
            )

            # Aggregate kv_connector_stats from all workers.
            if aggregated_kv_connector_stats is None:
                # Use the first worker's kv_connector_stats as accumulator.
                aggregated_kv_connector_stats = kv_output.kv_connector_stats
            elif kv_connector_stats := kv_output.kv_connector_stats:
                if aggregated_kv_connector_stats is None:
                    aggregated_kv_connector_stats = kv_connector_stats
                else:
                    assert isinstance(
                        aggregated_kv_connector_stats, type(kv_connector_stats)
                    )
                    aggregated_kv_connector_stats = (
                        aggregated_kv_connector_stats.aggregate(kv_connector_stats)
                    )

            # Aggregate kv_connector_worker_meta from all workers.
            if aggregated_kv_connector_worker_meta is None:
                # Use the first worker's kv_connector_worker_meta as accumulator.
                aggregated_kv_connector_worker_meta = kv_output.kv_connector_worker_meta
            elif kv_connector_worker_meta := kv_output.kv_connector_worker_meta:
                aggregated_kv_connector_worker_meta = (
                    aggregated_kv_connector_worker_meta.aggregate(
                        kv_connector_worker_meta
                    )
                )

            # Combine kv_cache_events from all workers.
            if combined_kv_cache_events is None:
                # Use the first worker's kv_cache events as start event list.
                combined_kv_cache_events = kv_output.kv_cache_events
            elif kv_cache_events := kv_output.kv_cache_events:
                assert isinstance(
                    combined_kv_cache_events,
                    type(kv_cache_events),
                )
                worker_kv_cache_events = kv_cache_events.get_all_events()
                combined_kv_cache_events.add_events(worker_kv_cache_events)
                combined_kv_cache_events.increment_workers(1)

            invalid_block_ids |= kv_output.invalid_block_ids

        # select output of the worker specified by output_rank
        output = outputs[output_rank]

        assert output is not None
        output.kv_connector_output = KVConnectorOutput(
            finished_sending=finished_sending or None,
            finished_recving=finished_recving or None,
            kv_connector_stats=aggregated_kv_connector_stats or None,
            kv_cache_events=combined_kv_cache_events or None,
            kv_connector_worker_meta=aggregated_kv_connector_worker_meta or None,
            invalid_block_ids=invalid_block_ids,
            expected_finished_count=self._expected_finished_count,
        )

        return output


def _make_src_and_dst_indices(
    src_block_ids: list[int],
    dst_block_ids: list[int],
    src_device: torch.device | str,
    dst_device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    src_indices = torch.tensor(src_block_ids, device=src_device, dtype=torch.int64)
    dst_indices = torch.tensor(dst_block_ids, device=dst_device, dtype=torch.int64)
    return src_indices, dst_indices


def copy_kv_blocks(
    src_kv_caches: dict[str, torch.Tensor],
    dst_kv_caches: dict[str, torch.Tensor],
    src_block_ids: list[int],
    dst_block_ids: list[int],
    direction: Literal["h2d", "d2h"],
) -> None:
    """Copy kv blocks between different buffers."""
    if (
        not src_kv_caches
        or not dst_kv_caches
        or not src_block_ids
        or not dst_block_ids
        or len(src_block_ids) != len(dst_block_ids)
    ):
        return

    src_device = next(iter(src_kv_caches.values())).device
    dst_device = next(iter(dst_kv_caches.values())).device

    src_indices, dst_indices = _make_src_and_dst_indices(
        src_block_ids=src_block_ids,
        dst_block_ids=dst_block_ids,
        src_device=src_device,
        dst_device=dst_device,
    )

    if direction == "h2d":
        copy_fn = current_platform.insert_blocks_to_device
    else:
        copy_fn = current_platform.swap_out_blocks_to_host
    for layer_name in src_kv_caches:
        src_tensor = src_kv_caches[layer_name]
        dst_tensor = dst_kv_caches[layer_name]
        copy_fn(src_tensor, dst_tensor, src_indices, dst_indices)


def kv_postprocess_blksize_on_receive(cache, indices, block_size_ratio):
    """
    Transforms the layout of received KV cache blocks to the local block_size.
    (Only works for local blocksize > remote blocksize)

    example:
    local blocksize = 16 tokens, remote blocksize = 4 tokens
    local block[0] = remote block[0, 1, 2, 3]
    remote is |h0-b0|h1-b0|h2-b0|h3-b0|h0-b1|h1-b1|h2-b1|h3-b1|...
    local is  |h0-b0..................|h1-b0..................|...
    permute is to:
    1. view => view remote as n_blocks * remote_shape(H,remoteN,D)
    2. permute => (H, nblocks, remoteN, D)
    3. flatten => (H, localN, D)
    """
    blocks_to_update = cache.index_select(0, indices)
    # use physical order
    blocks_to_update = blocks_to_update.permute(0, 2, 1, 3)
    n_kv_heads, block_size, head_size = blocks_to_update.shape[1:]
    remote_block_size = block_size // block_size_ratio
    n_blocks = block_size_ratio

    permuted_blocks = (
        blocks_to_update.reshape(-1, n_blocks, n_kv_heads, remote_block_size, head_size)
        .permute(0, 2, 1, 3, 4)
        .flatten(2, 3)
    )
    permuted_blocks = permuted_blocks.permute(0, 2, 1, 3)
    cache.index_copy_(0, indices, permuted_blocks)


def kv_postprocess_layout_on_receive(cache, indices):
    """Transforms the layout of received KV cache blocks to the local format.

    This method corrects layout mismatches from direct memory copies by
    permuting the tensor dimensions.

    - **Source Layout:** `[num_blocks, n_kv_head, block_size, head_dim]`
    - **Target Layout:** `[num_blocks, block_size, n_kv_head, head_dim]`

    Implementation:
    - x = blocks_to_update.reshape(src_shape) # view local kv with sender layout
    - permuted_blocks = x.permute(*inv_order) # transpose n_kv_heads, block_size
    - cache.index_copy_(0, indices, permuted_blocks) # copy permuted kv back

    """
    blocks_to_update = cache.index_select(0, indices)
    target_shape = list(blocks_to_update.shape)
    target_shape[0] = -1
    inv_order = [0, 2, 1, 3]
    src_shape = tuple(target_shape[i] for i in inv_order)
    blocks_to_update = cache.index_select(0, indices)
    permuted_blocks = blocks_to_update.reshape(src_shape).permute(*inv_order)
    cache.index_copy_(0, indices, permuted_blocks)


def kv_postprocess_blksize_and_layout_on_receive(cache, indices, block_size_ratio):
    """
    Transforms the layout of received KV cache to the local block_size and HND.
    (Only works for local blocksize > remote blocksize)

    prefill is HND, smaller block_size
    decode(local) is NHD, larger block_size
    """
    blocks_to_update = cache.index_select(0, indices)

    block_size, n_kv_heads, head_size = blocks_to_update.shape[1:]
    remote_block_size = block_size // block_size_ratio
    n_blocks = block_size_ratio

    permuted_blocks = (
        blocks_to_update.reshape(-1, n_blocks, n_kv_heads, remote_block_size, head_size)
        .permute(0, 1, 3, 2, 4)
        .flatten(1, 2)
    )
    cache.index_copy_(0, indices, permuted_blocks)


def yield_req_data(
    scheduler_output,
) -> Iterator[tuple[str, tuple[list[int], ...], bool]]:
    """
    Yields:
        (req_id, new_block_id_groups, preempted)
    """
    # new requests
    for req_data in scheduler_output.scheduled_new_reqs:
        yield req_data.req_id, req_data.block_ids, False

    # cached requests
    cached_reqs = scheduler_output.scheduled_cached_reqs
    yield from zip(
        cached_reqs.req_ids,
        cached_reqs.new_block_ids,
        (req_id in cached_reqs.resumed_req_ids for req_id in cached_reqs.req_ids),
    )


@dataclass
class TpKVTopology:
    """
    Helper class for tensor parallel and KV topology information for
    mapping between local and remote TP workers.
    """

    tp_rank: int
    remote_tp_size: dict[EngineId, int]
    is_mla: bool
    total_num_kv_heads: int
    attn_backends: list[type[AttentionBackend]]
    engine_id: EngineId
    remote_block_size: dict[EngineId, int]
    tensor_shape: torch.Size | None = None
    is_mamba: bool = False

    def __post_init__(self):
        # Figure out whether the first dimension of the cache is K/V
        # or num_blocks. This is used to register the memory regions correctly.
        attn_backend = self.attn_backends[0]
        if not self.is_mamba:
            _MOCK_BLOCK_SIZE = 16
            kv_cache_shape: tuple[int, ...] = attn_backend.get_kv_cache_shape(
                num_blocks=1, block_size=_MOCK_BLOCK_SIZE, num_kv_heads=1, head_size=1
            )
            logger.debug("Test kv_cache_shape: %s", kv_cache_shape)
        # Non-MLA backends caches have 5 dims [2, num_blocks, H,N,D],
        # we just mock num_blocks to 1 for the dimension check below.
        # Hybrid SSM models assume a single blocks_first layout
        self._is_kv_layout_blocks_first = self.is_mamba or (
            len(kv_cache_shape) == 5 and kv_cache_shape[0] == 1
        )

        self._cross_layers_blocks = False
        if self.tensor_shape is not None:
            self._cross_layers_blocks = (
                len(self.tensor_shape) == len(kv_cache_shape) + 1
            )
            self.tensor_shape: torch.Size

        if self._cross_layers_blocks:
            logger.debug("Using cross-layer KV cache")
            # prepend layers dimension
            _MOCK_NUM_LAYERS = 80
            kv_cache_shape = (_MOCK_NUM_LAYERS,) + kv_cache_shape
            try:
                kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
                    include_num_layers_dimension=self._cross_layers_blocks
                )
            except (AttributeError, NotImplementedError):
                assert self.tensor_shape is not None
                kv_cache_stride_order = tuple(range(len(self.tensor_shape)))

            # In case of cross layers permute kv_cache_shape according to
            # stride_order to retrieve physical position of block_size
            kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)

    @property
    def is_kv_layout_blocks_first(self) -> bool:
        return self._is_kv_layout_blocks_first

    @property
    def split_k_and_v(self) -> bool:
        # Whether to register regions for K and V separately (when present).
        return not (
            self._cross_layers_blocks or self.is_mla or self.is_kv_layout_blocks_first
        )

    @property
    def tp_size(self) -> int:
        return self.remote_tp_size[self.engine_id]

    @property
    def block_size(self) -> int:
        return self.remote_block_size[self.engine_id]

    @property
    def cross_layers_blocks(self) -> bool:
        return self._cross_layers_blocks

    def tp_ratio(
        self,
        remote_tp_size: int,
    ) -> int:
        """
        Calculate the tensor parallel ratio between local and remote TP.
        We can think of it as the number of local TP workers-per-remote TP
        workers. Local workers will read from the same remote TP worker in
        groups of size `tp_ratio`.If remote tp_size > local tp_size, the
        ratio is flipped (remote_size/local_size) and the returned value is
        negative.
        """
        if self.tp_size >= remote_tp_size:
            assert self.tp_size % remote_tp_size == 0, (
                f"Local tensor parallel size {self.tp_size} is not divisible "
                f"by remote tensor parallel size {remote_tp_size}."
            )
            return self.tp_size // remote_tp_size

        assert remote_tp_size % self.tp_size == 0, (
            f"Remote tensor parallel size {remote_tp_size} is not divisible "
            f"by local tensor parallel size {self.tp_size}."
        )
        # P TP > D TP case, return the ratio as negative
        return -remote_tp_size // self.tp_size

    def block_size_ratio(
        self,
        remote_block_size: int,
    ) -> int:
        """
        Calculate the block size ratio between local and remote TP.
        """
        assert self.block_size % remote_block_size == 0, (
            f"Local block size {self.block_size} is not divisible "
            f"by remote block size {remote_block_size} or vice versa."
        )
        return self.block_size // remote_block_size

    def tp_ratio_from_engine_id(
        self,
        remote_engine_id: EngineId,
    ) -> int:
        remote_tp_size = self.remote_tp_size[remote_engine_id]
        return self.tp_ratio(remote_tp_size)

    def block_size_ratio_from_engine_id(
        self,
        remote_engine_id: EngineId,
    ) -> int:
        remote_block_size = self.remote_block_size[remote_engine_id]
        return self.block_size_ratio(remote_block_size)

    def is_kv_replicated(self, engine_id: EngineId) -> bool:
        """
        Whether the KV cache is replicated across TP workers due to the
        number of TP workers being greater than the number of KV heads.
        When they are equal, each TP rank still owns one distinct KV head,
        so this is not considered replication.
        """
        tp_size = self.remote_tp_size[engine_id]
        return tp_size > self.total_num_kv_heads

    def replicates_kv_cache(self, remote_engine_id: EngineId) -> bool:
        # MLA is always replicated as the hidden dim can't be split.
        return self.is_mla or self.is_kv_replicated(remote_engine_id)

    def get_target_remote_ranks(
        self,
        remote_tp_size: int,
    ) -> list[int]:
        """
        Get the remote TP rank (on P) that the current local TP rank
        (on D) will read from. When remote tp_size > local tp_size, we
        read from multiple remote ranks.
        """
        tp_ratio = self.tp_ratio(remote_tp_size)
        if tp_ratio > 0:
            return [self.tp_rank // tp_ratio]

        # P TP > D TP case, D reads from |tp_ratio| remote workers.
        tp_ratio = -tp_ratio
        return [self.tp_rank * tp_ratio + i for i in range(tp_ratio)]

    def get_target_remote_ranks_from_engine_id(
        self,
        remote_engine_id: EngineId,
    ) -> list[int]:
        remote_tp_size = self.remote_tp_size[remote_engine_id]
        return self.get_target_remote_ranks(remote_tp_size)

    def get_transfer_cache_regions(
        self, cache: torch.Tensor, layer_spec: "KVCacheSpec"
    ) -> list[torch.Tensor] | torch.Tensor:
        """Return the cache tensor(s) to register as NIXL memory regions,
        also accounting for hybrid SSM models specificities.
        """
        if isinstance(layer_spec, MambaSpec):
            # Register the whole kv cache shared tensor, including SSM/Conv. This is
            # similar to FI with the difference that SSM/Conv have different sizes
            conv, ssm = cache
            return [conv]

        # Check may be hacky but it's matching `_update_hybrid_attention_mamba_layout`.
        if self.is_mamba and cache.shape[0] == 2:
            # When MAMBA is present, all backends are blocks first, so that blocks
            # can be shared between attention layers and mamba layers. Runner
            # `_update_hybrid_attention_mamba_layout` already adjusted strides
            # for FlashAttn-like backends so its num_blocks first.
            # Swap [2<>num_blocks] dims to get required layout for hybrid SSM.
            cache = cache.transpose(0, 1)

        # Regular case: backends like FA register K/V in separate regions
        return cache if self.split_k_and_v else [cache]


# ---- Mamba-HMA hetero-TP transfer config ----
#
# Key insight: with hetero-TP (P_TP > D_TP), FA KV cache may be
# replicated across P ranks (when P_TP > num_kv_heads), but Mamba
# conv/SSM state is almost always uniquely sharded per P rank.  So the
# number of P ranks D must read from can differ between FA and Mamba,
# and they must be handled separately.


def _physical_head_range(tp_size: int, num_heads: int, rank: int) -> range:
    """Physical KV head range stored in a rank's KV cache tensor.

    When ``tp_size <= num_heads``: sharded, K/TP contiguous heads per rank.
    When ``tp_size > num_heads``: 1 physical head per rank.  Heads are
    distributed **contiguously** (matching vLLM's GQA weight partitioning):
    consecutive ranks share a head before moving to the next one.
    """
    if tp_size <= num_heads:
        assert num_heads % tp_size == 0
        per_rank = num_heads // tp_size
        return range(rank * per_rank, (rank + 1) * per_rank)
    else:
        h = rank * num_heads // tp_size
        return range(h, h + 1)


def _range_overlap(a: range, b: range) -> range:
    start = max(a.start, b.start)
    stop = min(a.stop, b.stop)
    return range(start, max(start, stop))


@dataclass
class HeteroTPTransferConfig:
    """Precomputed transfer plan for one (D rank, P engine) pair.

    Currently only instantiated for Mamba-HMA (hybrid SSM+Attention) models
    where FA and mamba require different splitting factors.  Could be extended
    to other model types that need non-uniform hetero-TP transfer sizing.

    All descriptor sizes are computed here.  The guarantee is:
        local_entry_size == remote_entry_size   (for NIXL)

    Attributes that start with ``fa_`` concern FlashAttention KV cache.
    Attributes that start with ``mamba_`` concern Mamba conv/SSM state.
    """

    # ---- Input parameters (from handshake) ----
    tp_ratio: int
    K: int  # total_num_kv_heads (before TP sharding)
    d_tp: int  # D engine's tensor_parallel_size
    p_tp: int  # P engine's tensor_parallel_size
    d_rank: int  # this D worker's TP rank
    use_mla: bool

    # Per-layer block lengths (bytes, K+V combined for blocks_first).
    # Uniform across layers for current models.
    d_block_len: int  # D's block_len_per_layer (representative)
    p_block_len: int  # P's block_len_per_layer (from handshake)
    is_blocks_first: bool  # kv_topo.is_kv_layout_blocks_first

    # ---- Derived: computed in __post_init__ ----
    #
    # Physical heads per rank (what the KV tensor actually stores)
    d_physical_heads: int = field(init=False)
    p_physical_heads: int = field(init=False)

    # How many distinct P ranks D needs for FA data
    physical_fa_num_reads: int = field(init=False)

    # Which P ranks contribute unique FA heads (ordered by head index)
    fa_read_targets: list[int] = field(init=False)

    # All P ranks needed for mamba (always abs_tp for tp_ratio < 0)
    mamba_num_reads: int = field(init=False)

    # All P ranks this D rank communicates with (FA ∪ mamba)
    transfer_targets: list[int] = field(init=False)

    # FA descriptor entry size (K or V side, for blocks_first layout)
    # Guaranteed: fa_entry_size is the SAME for local handle AND remote desc.
    fa_entry_size: int = field(init=False)

    # Replication flags
    is_d_replicated: bool = field(init=False)
    is_p_replicated: bool = field(init=False)

    # Pre-built set for fast lookup
    _fa_target_set: frozenset[int] = field(init=False, repr=False)
    # Map: P rank → index in fa_read_targets (for head slot offset)
    _fa_target_index: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        K = self.K
        self.is_d_replicated = self.d_tp > K
        self.is_p_replicated = self.p_tp > K

        self.d_physical_heads = max(1, K // self.d_tp)
        self.p_physical_heads = max(1, K // self.p_tp)

        abs_tp = -self.tp_ratio if self.tp_ratio < 0 else 1

        # ---- Mamba range (computed first so FA can prefer ranks in it) ----
        mamba_range: range | None = None
        if self.tp_ratio < 0:
            mamba_range = range(self.d_rank * abs_tp, (self.d_rank + 1) * abs_tp)

        # ---- FA read targets ----
        if self.use_mla or self.tp_ratio >= 0:
            self.physical_fa_num_reads = 1
            self.fa_read_targets = (
                [0]
                if self.use_mla
                # Must match kv_topo.get_target_remote_ranks (d_rank // tp_ratio).
                else [
                    self.d_rank // self.tp_ratio if self.tp_ratio > 0 else self.d_rank
                ]
            )
        else:
            d_needs = _physical_head_range(self.d_tp, K, self.d_rank)
            # When mamba range exists, prefer P ranks within it so that
            # FA targets are a subset of mamba transfer_targets (avoids
            # orphaned FA targets outside the transfer loop).
            search_range = mamba_range if mamba_range is not None else range(self.p_tp)
            seen: set[tuple[int, int]] = set()
            targets: list[int] = []
            for p in search_range:
                p_has = _physical_head_range(self.p_tp, K, p)
                ov = _range_overlap(d_needs, p_has)
                if len(ov) > 0:
                    key = (ov.start, ov.stop)
                    if key not in seen:
                        seen.add(key)
                        targets.append(p)
            if not targets:
                # Fallback: search globally (should not happen in practice)
                for p in range(self.p_tp):
                    p_has = _physical_head_range(self.p_tp, K, p)
                    ov = _range_overlap(d_needs, p_has)
                    if len(ov) > 0:
                        key = (ov.start, ov.stop)
                        if key not in seen:
                            seen.add(key)
                            targets.append(p)
            self.fa_read_targets = targets
            self.physical_fa_num_reads = len(targets)

        self._fa_target_set = frozenset(self.fa_read_targets)
        self._fa_target_index = {r: i for i, r in enumerate(self.fa_read_targets)}

        # ---- Mamba targets ----
        if mamba_range is not None and abs_tp > self.physical_fa_num_reads:
            self.mamba_num_reads = abs_tp
            self.transfer_targets = list(mamba_range)
        else:
            self.mamba_num_reads = self.physical_fa_num_reads
            self.transfer_targets = list(self.fa_read_targets)

        # ---- FA entry size ----
        # For blocks_first: block_len_per_layer includes K+V; // 2 gives K (or V).
        # Use min(D, P) because D indexes into P when tp_ratio > 0,
        # and P is the natural unit when tp_ratio < 0.
        effective_block_len = min(self.d_block_len, self.p_block_len)
        if self.is_blocks_first:
            self.fa_entry_size = effective_block_len // 2
        else:
            self.fa_entry_size = effective_block_len

        self._validate()

    def _validate(self) -> None:
        """Cross-check internal consistency."""
        if self.is_d_replicated and self.is_p_replicated and self.tp_ratio > 0:
            logger.info(
                "Both-replicated hetero-TP: D_TP=%d > P_TP=%d > K=%d. "
                "Using d_rank // tp_ratio routing with relative head offset.",
                self.d_tp,
                self.p_tp,
                self.K,
            )

        # FA targets must be a subset of transfer_targets
        tt_set = set(self.transfer_targets)
        for t in self.fa_read_targets:
            if t not in tt_set:
                logger.error(
                    "FA target P rank %d is NOT in transfer_targets %s. "
                    "This will cause missed FA reads!",
                    t,
                    self.transfer_targets,
                )

        # For tp_ratio < 0 with blocks_first: D_K_half / reads should == P_K_half
        if (
            self.is_blocks_first
            and self.tp_ratio < 0
            and self.physical_fa_num_reads > 0
        ):
            d_k_half = self.d_block_len // 2
            p_k_half = self.p_block_len // 2
            expected_local = d_k_half // self.physical_fa_num_reads
            if expected_local != p_k_half:
                logger.warning(
                    "FA size mismatch: D_K_half=%d / reads=%d = %d, "
                    "but P_K_half=%d.  This may indicate a head count or "
                    "Mamba-HMA inflation inconsistency.",
                    d_k_half,
                    self.physical_fa_num_reads,
                    expected_local,
                    p_k_half,
                )

    # ---- Query methods ----

    def should_skip_fa(self, p_rank: int) -> bool:
        """Whether to skip FA groups for this P rank (mamba-only transfer)."""
        return p_rank not in self._fa_target_set

    def fa_head_slot(self, p_rank: int) -> int:
        """Index into D's FA block for this P rank's head data.

        For P ranks in fa_read_targets, returns 0, 1, ..., reads-1.
        For P ranks NOT in fa_read_targets (replicated duplicates),
        returns the slot of the matching FA target with the same head.
        """
        if p_rank in self._fa_target_index:
            return self._fa_target_index[p_rank]
        # Duplicate head: find which fa_target has the same physical head
        p_head = _physical_head_range(self.p_tp, self.K, p_rank)
        for target in self.fa_read_targets:
            t_head = _physical_head_range(self.p_tp, self.K, target)
            if _range_overlap(p_head, t_head):
                return self._fa_target_index[target]
        return 0  # fallback

    def fa_rank_offset(self, remote_kv_block_len: int) -> int:
        """Byte offset into P's FA block for this D rank.

        When D is replicated (D_TP > K), multiple D ranks share a head.
        Computes offset *relative to the target P rank's first head*
        so it works regardless of how many heads P has.
        When neither side replicates, falls back to tp_rank % tp_ratio.
        Returns 0 when D does not index into P's block.
        """
        if self.use_mla or self.tp_ratio <= 0:
            return 0
        if self.is_d_replicated:
            d_head = self.d_rank * self.K // self.d_tp
            p_rank = self.fa_read_targets[0]
            p_start = p_rank * self.K // self.p_tp
            return (d_head - p_start) * remote_kv_block_len
        return self.d_rank % self.tp_ratio * remote_kv_block_len

    @property
    def needs_split_handles(self) -> bool:
        """Whether per-P-rank split handles are needed.

        True when FA and mamba have different read counts, requiring
        different splitting factors in the local handle.
        """
        return self.tp_ratio < 0 and not self.use_mla and len(self.transfer_targets) > 1

    def compute_split_handle_data(
        self,
        src_blocks_data: list[tuple[int, int, int]],
        num_fa_descs: int,
        abs_tp: int,
    ) -> list[list[tuple[int, int, int]]]:
        """Compute per-P-rank (addr, len, tp) triples for Mamba-HMA split handles.

        FA descriptors (indices < num_fa_descs) are sliced by
        ``physical_fa_num_reads``; mamba descriptors are sliced uniformly
        by ``abs_tp``.

        Returns one list of triples per transfer target.
        """
        all_handle_data: list[list[tuple[int, int, int]]] = []
        for p_idx, p_rank in enumerate(self.transfer_targets):
            handle_data: list[tuple[int, int, int]] = []
            skip_fa = self.should_skip_fa(p_rank)
            fa_slot = self.fa_head_slot(p_rank) if not skip_fa else 0

            for j, (addr, local_len, tp) in enumerate(src_blocks_data):
                if j < num_fa_descs:
                    assert self.physical_fa_num_reads >= 1
                    fa_chunk = local_len // self.physical_fa_num_reads
                    handle_data.append((addr + fa_slot * fa_chunk, fa_chunk, tp))
                else:
                    mamba_chunk = local_len // abs_tp
                    handle_data.append((addr + p_idx * mamba_chunk, mamba_chunk, tp))
            all_handle_data.append(handle_data)
        return all_handle_data

    def filter_block_ids_for_rank(
        self,
        remote_rank: int,
        local_ids: BlockIds,
        remote_ids: BlockIds,
        is_mamba_group: list[bool],
    ) -> tuple[BlockIds, BlockIds]:
        """Zero out FA groups for P ranks outside fa_read_targets.

        Returns (filtered_local_ids, filtered_remote_ids).  When the
        remote rank carries FA data for this D rank, returns the inputs
        unchanged.
        """
        if not self.should_skip_fa(remote_rank):
            return local_ids, remote_ids
        num_groups = len(local_ids)
        filtered_local: list[list[int]] = [
            [] if not is_mamba_group[g] else local_ids[g] for g in range(num_groups)
        ]
        filtered_remote: list[list[int]] = [
            [] if not is_mamba_group[g] else remote_ids[g] for g in range(num_groups)
        ]
        return filtered_local, filtered_remote

    def describe(self) -> str:
        """One-line summary for logging."""
        return (
            f"HeteroTPTransferConfig("
            f"tp_ratio={self.tp_ratio}, K={self.K}, "
            f"d_tp={self.d_tp}, p_tp={self.p_tp}, d_rank={self.d_rank}, "
            f"physical_fa_reads={self.physical_fa_num_reads}, "
            f"mamba_reads={self.mamba_num_reads}, "
            f"fa_targets={self.fa_read_targets}, "
            f"transfer_targets={self.transfer_targets}, "
            f"fa_entry_size={self.fa_entry_size}, "
            f"d_block_len={self.d_block_len}, p_block_len={self.p_block_len})"
        )


def get_current_attn_backends(
    vllm_config: VllmConfig, layer_names: list[str] | None = None
) -> list[type[AttentionBackend]]:
    """Get all distinct attention backends for the given layers.

    Args:
        vllm_config: The current vLLM configuration.
        layer_names: Optional list of layer names to scope the lookup.
            When None, all attention layers are considered.

    Returns:
        Deduplicated list of attention backend classes.
    """
    layer_type = cast(type[Any], AttentionLayerBase)
    layers = get_layers_from_vllm_config(vllm_config, layer_type, layer_names)
    if layers:
        seen: dict[str, type[AttentionBackend]] = {}
        for layer in layers.values():
            backend = layer.get_attn_backend()
            seen[backend.full_cls_name()] = backend
        return list(seen.values())

    # Fallback for tests, when static_forward_context is empty.
    logger.debug(
        "No layers found in the vLLM config. Falling back to default attention backend."
    )
    from vllm.v1.attention.selector import get_attn_backend

    return [
        get_attn_backend(
            head_size=vllm_config.model_config.get_head_size(),
            dtype=vllm_config.model_config.dtype,
            kv_cache_dtype=vllm_config.cache_config.cache_dtype,
            use_mla=vllm_config.model_config.use_mla,
        )
    ]


def get_current_attn_backend(
    vllm_config: VllmConfig, layer_names: list[str] | None = None
) -> type[AttentionBackend]:
    """Get the first attention backend for the given layers."""
    return get_current_attn_backends(vllm_config, layer_names)[0]


# TODO (ZhanqiuHu): Consolidate TpKVTopology and HeteroTPTransferConfig
# into a single engine-agnostic TransferTopology class.
# 6 of 9 HeteroTPTransferConfig init fields duplicate TpKVTopology data.
#
# @dataclass
# class EngineTransferInfo:
#     """Per-remote-engine transfer state, computed at handshake."""
#     p_tp: int
#     tp_ratio: int
#     p_block_len: int
#     block_size: int
#     # Mamba-specific (None for non-mamba models)
#     fa_read_targets: list[int] | None = None
#     transfer_targets: list[int] | None = None
#     physical_fa_num_reads: int | None = None
#     mamba_num_reads: int | None = None
#     fa_entry_size: int | None = None
#
# class TransferTopology:
#     """Single source of truth for TP topology + transfer sizing."""
#     # Shared (set once at init, replaces duplicate fields)
#     tp_rank: int          # == TpKVTopology.tp_rank == HeteroTP.d_rank
#     tp_size: int          # == TpKVTopology.tp_size == HeteroTP.d_tp
#     total_num_kv_heads: int  # == HeteroTP.K
#     is_mla: bool          # == HeteroTP.use_mla
#     is_mamba: bool
#     is_blocks_first: bool # == HeteroTP.is_blocks_first
#     d_block_len: int
#
#     # Per-engine (populated via register_engine() at handshake)
#     _engines: dict[EngineId, EngineTransferInfo]
#
#     def register_engine(self, engine_id, p_tp, p_block_len, ...): ...
#
#     # General (from TpKVTopology)
#     def tp_ratio(self, engine_id) -> int: ...
#     def target_remote_ranks(self, engine_id) -> list[int]: ...
#     def is_kv_replicated(self, engine_id) -> bool: ...
#
#     # Mamba-specific (from HeteroTPTransferConfig, gated by is_mamba)
#     def fa_rank_offset(self, engine_id, block_len) -> int: ...
#     def physical_fa_num_reads(self, engine_id) -> int: ...
#     def transfer_targets(self, engine_id) -> list[int]: ...
#     def should_skip_fa(self, engine_id, p_rank) -> bool: ...
#     def filter_block_ids_for_rank(self, engine_id, ...) -> ...: ...
