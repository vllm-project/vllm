# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
KV cache helper for store.
"""

from collections.abc import Iterator
from dataclasses import dataclass
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
) -> Iterator[tuple[str, tuple[list[int], ...] | None, bool]]:
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


# ---- Per-engine transfer info ----


@dataclass(frozen=True)
class EngineTransferInfo:
    """Common per-remote-engine transfer state, computed at handshake.

    Stored per ``engine_id`` inside ``TransferTopology._engines``.
    """

    remote_tp_size: int

    remote_block_len: int
    """Block length (bytes)"""

    remote_block_size: int
    """Tokens per block."""

    remote_physical_blocks_per_logical: int
    """Physical blocks per logical block."""


@dataclass(frozen=True)
class MambaEngineTransferInfo(EngineTransferInfo):
    """Extends ``EngineTransferInfo`` with Mamba-hybrid transfer geometry.

    For hybrid SSM+Attention models, FA and Mamba layers may require
    different numbers of reads from different remote ranks.  This
    dataclass captures that per-engine transfer plan.
    """

    remote_fa_source_ranks: tuple[int, ...]
    """Remote ranks carrying unique FA heads for this local rank."""

    remote_all_source_ranks: tuple[int, ...]
    """All remote ranks this local rank reads from (FA + Mamba)."""

    remote_num_fa_reads: int
    """Number of distinct remote ranks needed for FA data."""

    remote_num_mamba_reads: int
    """Number of distinct remote ranks needed for Mamba data."""

    remote_fa_descriptor_bytes: int
    """Byte size of one FA K (or V) descriptor entry."""

    is_remote_replicated: bool
    """Whether the remote engine has replicated KV heads
    (remote_tp_size > total_num_kv_heads)."""

    remote_physical_heads: int
    """Physical KV heads stored per remote rank."""


# ---- Transfer topology ----


@dataclass
class TransferTopology:
    """Single source of truth for local TP identity and per-engine remote info."""

    tp_rank: int
    tp_size: int
    block_size: int
    engine_id: EngineId
    is_mla: bool
    is_mamba: bool
    total_num_kv_heads: int
    attn_backends: list[type[AttentionBackend]]
    tensor_shape: torch.Size | None = None

    def __post_init__(self):
        self.local_physical_heads = max(1, self.total_num_kv_heads // self.tp_size)

        self._engines: dict[EngineId, EngineTransferInfo] = {}
        self._fa_source_sets: dict[EngineId, frozenset[int]] = {}
        self._fa_source_indices: dict[EngineId, dict[int, int]] = {}

        # Figure out whether the first dimension of the cache is K/V
        # or num_blocks.
        attn_backend = self.attn_backends[0]
        if not self.is_mamba:
            _MOCK_BLOCK_SIZE = 16
            kv_cache_shape: tuple[int, ...] = attn_backend.get_kv_cache_shape(
                num_blocks=1,
                block_size=_MOCK_BLOCK_SIZE,
                num_kv_heads=1,
                head_size=1,
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

        if self._cross_layers_blocks:
            logger.debug("Using cross-layer KV cache")
            _MOCK_NUM_LAYERS = 80
            kv_cache_shape = (_MOCK_NUM_LAYERS,) + kv_cache_shape
            try:
                kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
                    include_num_layers_dimension=self._cross_layers_blocks
                )
            except (AttributeError, NotImplementedError):
                assert self.tensor_shape is not None
                kv_cache_stride_order = tuple(range(len(self.tensor_shape)))
            kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)

    # ============================================================
    # Engine registration
    # ============================================================

    def register_remote_engine(
        self,
        remote_engine_id: EngineId,
        remote_tp_size: int,
        remote_block_size: int,
        remote_block_len: int,
        remote_physical_blocks_per_logical: int,
        *,
        local_block_len: int = 0,
    ) -> EngineTransferInfo:
        """Register a remote engine, unifying worker dicts state.

        Only remote engines should be registered here — the local engine's
        identity (tp_size, block_size, etc.) is set via ``__init__`` params.

        For Mamba models, also computes the Mamba transfer plan and
        builds the FA source lookup caches.

        Args:
            local_block_len: Local representative block_len (bytes).
                Required for Mamba models to compute ``fa_descriptor_bytes``.
        """
        assert remote_engine_id != self.engine_id, (
            f"Cannot register local engine {self.engine_id} as remote. "
            f"Local identity is set via __init__ params."
        )
        if remote_engine_id in self._engines:
            return self._engines[remote_engine_id]
        info: EngineTransferInfo
        if self.is_mamba:
            info = self._build_mamba_info(
                remote_tp_size=remote_tp_size,
                remote_block_size=remote_block_size,
                remote_block_len=remote_block_len,
                remote_physical_blocks_per_logical=(remote_physical_blocks_per_logical),
                local_block_len=local_block_len,
            )
            assert isinstance(info, MambaEngineTransferInfo)
            self._fa_source_sets[remote_engine_id] = frozenset(
                info.remote_fa_source_ranks
            )
            self._fa_source_indices[remote_engine_id] = {
                r: i for i, r in enumerate(info.remote_fa_source_ranks)
            }
        else:
            info = EngineTransferInfo(
                remote_tp_size=remote_tp_size,
                remote_block_len=remote_block_len,
                remote_block_size=remote_block_size,
                remote_physical_blocks_per_logical=(remote_physical_blocks_per_logical),
            )
        self._engines[remote_engine_id] = info
        return info

    def get_engine_info(self, remote_engine_id: EngineId) -> EngineTransferInfo:
        return self._engines[remote_engine_id]

    # ============================================================
    # Layout properties
    # ============================================================

    @property
    def is_kv_layout_blocks_first(self) -> bool:
        return self._is_kv_layout_blocks_first

    @property
    def cross_layers_blocks(self) -> bool:
        return self._cross_layers_blocks

    @property
    def split_k_and_v(self) -> bool:
        # Whether to register regions for K and V separately (when present).
        return not (
            self._cross_layers_blocks or self.is_mla or self.is_kv_layout_blocks_first
        )

    # ============================================================
    # Common methods
    # ============================================================

    def tp_ratio(self, remote_tp_size: int) -> int:
        """Calculate the tensor parallel ratio between local and remote TP.

        Positive when local_tp >= remote_tp (local workers read from the
        same remote worker in groups of size ``tp_ratio``).  Negative when
        remote_tp > local_tp (ratio is flipped).
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
        return -(remote_tp_size // self.tp_size)

    def block_size_ratio(self, remote_block_size: int) -> int:
        """Calculate the block size ratio between local and remote."""
        assert self.block_size % remote_block_size == 0, (
            f"Local block size {self.block_size} is not divisible "
            f"by remote block size {remote_block_size} or vice versa."
        )
        return self.block_size // remote_block_size

    def is_kv_replicated(self, remote_engine_id: EngineId) -> bool:
        """Whether the KV cache is replicated across TP workers due to the
        number of TP workers being greater than the number of KV heads.
        """
        return self._engines[remote_engine_id].remote_tp_size > self.total_num_kv_heads

    def replicates_kv_cache(self, remote_engine_id: EngineId) -> bool:
        # MLA is always replicated as the hidden dim can't be split.
        return self.is_mla or self.is_kv_replicated(remote_engine_id)

    @property
    def local_replicates_kv_cache(self) -> bool:
        """Whether the local engine's KV cache is replicated."""
        return self.is_mla or self.tp_size > self.total_num_kv_heads

    def handshake_target_ranks(self, remote_tp_size: int) -> list[int]:
        """Pre-registration: compute which remote TP ranks to handshake with.

        Pure math based on local/remote TP sizes — does not require
        the remote engine to be registered yet.
        """
        tp_ratio = self.tp_ratio(remote_tp_size)
        if tp_ratio > 0:
            return [self.tp_rank // tp_ratio]
        abs_ratio = -tp_ratio
        return [self.tp_rank * abs_ratio + i for i in range(abs_ratio)]

    def target_remote_ranks(self, remote_engine_id: EngineId) -> list[int]:
        """Get the remote TP rank(s) that the current local TP rank will
        read from.  When remote tp_size > local tp_size, reads from
        multiple remote ranks.

        For Mamba models, returns the precomputed ``all_source_ranks``
        (FA + Mamba union).
        """
        info = self._engines[remote_engine_id]
        if isinstance(info, MambaEngineTransferInfo):
            return list(info.remote_all_source_ranks)

        tp_ratio = self.tp_ratio(info.remote_tp_size)
        if tp_ratio > 0:
            return [self.tp_rank // tp_ratio]
        # remote TP > local TP: read from |tp_ratio| remote workers
        abs_ratio = -tp_ratio
        return [self.tp_rank * abs_ratio + i for i in range(abs_ratio)]

    def get_transfer_cache_regions(
        self, cache: torch.Tensor, layer_spec: "KVCacheSpec"
    ) -> list[torch.Tensor] | torch.Tensor:
        """Return the cache tensor(s) to register as NIXL memory regions,
        also accounting for hybrid SSM models specificities.
        """
        if isinstance(layer_spec, MambaSpec):
            # Register the whole kv cache shared tensor, including
            # SSM/Conv.
            conv, ssm = cache
            return [conv]

        # Check may be hacky but it's matching
        # `_update_hybrid_attention_mamba_layout`.
        if self.is_mamba and cache.shape[0] == 2:
            # When MAMBA is present, all backends are blocks first, so
            # that blocks can be shared between attention layers and mamba
            # layers.  Runner already adjusted strides for FlashAttn-like
            # backends so its num_blocks first.
            # Swap [2<>num_blocks] dims for hybrid SSM layout.
            cache = cache.transpose(0, 1)

        # Regular case: backends like FA register K/V in separate regions
        return cache if self.split_k_and_v else [cache]

    # ============================================================
    # Mamba-specific methods
    # ============================================================

    def should_skip_fa(self, remote_engine_id: EngineId, remote_rank: int) -> bool:
        """Whether to skip FA groups for this remote rank (mamba-only)."""
        return remote_rank not in self._fa_source_sets[remote_engine_id]

    def fa_head_slot(self, remote_engine_id: EngineId, remote_rank: int) -> int:
        """Index into local FA block for this remote rank's head data.

        For remote ranks in ``fa_source_ranks``, returns 0, 1, …, reads-1.
        For ranks NOT in ``fa_source_ranks`` (replicated duplicates),
        returns the slot of the matching source rank with the same head.
        """
        fa_index = self._fa_source_indices[remote_engine_id]
        if remote_rank in fa_index:
            return fa_index[remote_rank]
        mamba_info = self._engines[remote_engine_id]
        assert isinstance(mamba_info, MambaEngineTransferInfo)
        K = self.total_num_kv_heads
        remote_tp = mamba_info.remote_tp_size
        r_head = self._physical_head_range(remote_tp, K, remote_rank)
        for target in mamba_info.remote_fa_source_ranks:
            t_head = self._physical_head_range(remote_tp, K, target)
            if self._range_overlap(r_head, t_head):
                return fa_index[target]
        return 0

    def fa_rank_offset(
        self, remote_engine_id: EngineId, remote_kv_block_len: int
    ) -> int:
        """Byte offset into remote FA block for this local rank.

        When local TP is replicated (local_tp > K), multiple local ranks
        share a head.  Computes offset *relative to the target remote
        rank's first head* so it works regardless of how many heads the
        remote has.  Returns 0 when local does not index into remote.
        """
        mamba_info = self._engines[remote_engine_id]
        assert isinstance(mamba_info, MambaEngineTransferInfo)
        tp_ratio = self.tp_ratio(mamba_info.remote_tp_size)
        if self.is_mla or tp_ratio <= 0:
            return 0
        K = self.total_num_kv_heads
        is_local_replicated = self.tp_size > K
        if is_local_replicated:
            local_head = self.tp_rank * K // self.tp_size
            p_rank = mamba_info.remote_fa_source_ranks[0]
            p_start = p_rank * K // mamba_info.remote_tp_size
            return (local_head - p_start) * remote_kv_block_len
        return self.tp_rank % tp_ratio * remote_kv_block_len

    def needs_split_handles(self, remote_engine_id: EngineId) -> bool:
        """Whether per-remote-rank split handles are needed.

        True when FA and mamba have different read counts, requiring
        different splitting factors in the local handle.
        """
        mamba_info = self._engines[remote_engine_id]
        assert isinstance(mamba_info, MambaEngineTransferInfo)
        tp_ratio = self.tp_ratio(mamba_info.remote_tp_size)
        return (
            tp_ratio < 0
            and not self.is_mla
            and len(mamba_info.remote_all_source_ranks) > 1
        )

    def compute_split_handle_data(
        self,
        remote_engine_id: EngineId,
        src_blocks_data: list[tuple[int, int, int]],
        num_fa_descs: int,
        abs_tp: int,
    ) -> list[list[tuple[int, int, int]]]:
        """Per-remote-rank (addr, len, dev) triples for Mamba-HMA split
        handles.

        FA descriptors (indices < num_fa_descs) are sliced by
        ``remote_num_fa_reads``; mamba descriptors are sliced uniformly
        by ``abs_tp``.
        """
        mamba_info = self._engines[remote_engine_id]
        assert isinstance(mamba_info, MambaEngineTransferInfo)
        all_handle_data: list[list[tuple[int, int, int]]] = []
        for p_idx, p_rank in enumerate(mamba_info.remote_all_source_ranks):
            handle_data: list[tuple[int, int, int]] = []
            skip_fa = self.should_skip_fa(remote_engine_id, p_rank)
            fa_slot = self.fa_head_slot(remote_engine_id, p_rank) if not skip_fa else 0
            for j, (addr, local_len, dev) in enumerate(src_blocks_data):
                if j < num_fa_descs:
                    assert mamba_info.remote_num_fa_reads >= 1
                    fa_chunk = local_len // mamba_info.remote_num_fa_reads
                    handle_data.append((addr + fa_slot * fa_chunk, fa_chunk, dev))
                else:
                    mamba_chunk = local_len // abs_tp
                    handle_data.append((addr + p_idx * mamba_chunk, mamba_chunk, dev))
            all_handle_data.append(handle_data)
        return all_handle_data

    def filter_block_ids_for_rank(
        self,
        remote_engine_id: EngineId,
        remote_rank: int,
        local_ids: BlockIds,
        remote_ids: BlockIds,
        is_mamba_group: list[bool],
    ) -> tuple[BlockIds, BlockIds]:
        """Zero out FA groups for remote ranks outside ``fa_source_ranks``.

        Returns (filtered_local_ids, filtered_remote_ids).  When the
        remote rank carries FA data for this local rank, returns the
        inputs unchanged.
        """
        if not self.should_skip_fa(remote_engine_id, remote_rank):
            return local_ids, remote_ids
        num_groups = len(local_ids)
        filtered_local: list[list[int]] = [
            [] if not is_mamba_group[g] else local_ids[g] for g in range(num_groups)
        ]
        filtered_remote: list[list[int]] = [
            [] if not is_mamba_group[g] else remote_ids[g] for g in range(num_groups)
        ]
        return filtered_local, filtered_remote

    def describe(self, remote_engine_id: EngineId) -> str:
        """One-line summary of transfer config for logging."""
        info = self._engines[remote_engine_id]
        base = (
            f"tp_ratio={self.tp_ratio(info.remote_tp_size)}, "
            f"K={self.total_num_kv_heads}, "
            f"local_tp={self.tp_size}, "
            f"remote_tp={info.remote_tp_size}, "
            f"local_rank={self.tp_rank}, "
            f"remote_block_len={info.remote_block_len}"
        )
        if isinstance(info, MambaEngineTransferInfo):
            return (
                f"TransferTopology.mamba({base}, "
                f"fa_reads={info.remote_num_fa_reads}, "
                f"mamba_reads={info.remote_num_mamba_reads}, "
                f"fa_sources={list(info.remote_fa_source_ranks)}, "
                f"all_sources={list(info.remote_all_source_ranks)}, "
                f"fa_desc_bytes={info.remote_fa_descriptor_bytes})"
            )
        return f"TransferTopology({base})"

    # ============================================================
    # Private helpers
    # ============================================================
    # Mamba-HMA hetero-TP transfer config:
    # With hetero-TP (P_TP > D_TP), FA KV cache may be replicated across
    # P ranks (when P_TP > num_kv_heads), but Mamba conv/SSM state is
    # almost always uniquely sharded per P rank.  So the number of P
    # ranks D must read from can differ between FA and Mamba, and they
    # must be handled separately.

    @staticmethod
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

    @staticmethod
    def _range_overlap(a: range, b: range) -> range:
        start = max(a.start, b.start)
        stop = min(a.stop, b.stop)
        return range(start, max(start, stop))

    # ============================================================
    # Private: build Mamba transfer info
    # ============================================================

    def _build_mamba_info(
        self,
        remote_tp_size: int,
        remote_block_size: int,
        remote_block_len: int,
        remote_physical_blocks_per_logical: int,
        local_block_len: int,
    ) -> MambaEngineTransferInfo:
        """Compute Mamba transfer plan."""
        K = self.total_num_kv_heads
        local_tp = self.tp_size
        local_rank = self.tp_rank

        is_remote_replicated = remote_tp_size > K
        remote_physical_heads = max(1, K // remote_tp_size)

        if local_tp >= remote_tp_size:
            assert local_tp % remote_tp_size == 0
            tp_ratio = local_tp // remote_tp_size
        else:
            assert remote_tp_size % local_tp == 0
            tp_ratio = -(remote_tp_size // local_tp)

        abs_tp = -tp_ratio if tp_ratio < 0 else 1

        mamba_range: range | None = None
        if tp_ratio < 0:
            mamba_range = range(local_rank * abs_tp, (local_rank + 1) * abs_tp)

        # ---- FA read targets ----
        if self.is_mla or tp_ratio >= 0:
            num_fa_reads = 1
            fa_source_ranks: list[int] = (
                [0]
                if self.is_mla
                else [local_rank // tp_ratio if tp_ratio > 0 else local_rank]
            )
        else:
            local_needs = self._physical_head_range(local_tp, K, local_rank)
            search_range = (
                mamba_range if mamba_range is not None else range(remote_tp_size)
            )
            seen: set[tuple[int, int]] = set()
            fa_source_ranks = []
            for p in search_range:
                p_has = self._physical_head_range(remote_tp_size, K, p)
                ov = self._range_overlap(local_needs, p_has)
                if len(ov) > 0:
                    key = (ov.start, ov.stop)
                    if key not in seen:
                        seen.add(key)
                        fa_source_ranks.append(p)
            if not fa_source_ranks:
                for p in range(remote_tp_size):
                    p_has = self._physical_head_range(remote_tp_size, K, p)
                    ov = self._range_overlap(local_needs, p_has)
                    if len(ov) > 0:
                        key = (ov.start, ov.stop)
                        if key not in seen:
                            seen.add(key)
                            fa_source_ranks.append(p)
            num_fa_reads = len(fa_source_ranks)

        # ---- All source ranks (mamba + FA) ----
        if mamba_range is not None and abs_tp > num_fa_reads:
            num_mamba_reads = abs_tp
            all_source_ranks = list(mamba_range)
        else:
            num_mamba_reads = num_fa_reads
            all_source_ranks = list(fa_source_ranks)

        # ---- FA descriptor bytes ----
        effective_block_len = min(local_block_len, remote_block_len)
        if self.is_kv_layout_blocks_first:
            fa_descriptor_bytes = effective_block_len // 2
        else:
            fa_descriptor_bytes = effective_block_len

        # ---- Validation ----
        is_local_replicated = local_tp > K
        if is_local_replicated and is_remote_replicated and tp_ratio > 0:
            logger.info(
                "Both-replicated hetero-TP: local_tp=%d > remote_tp=%d > K=%d.",
                local_tp,
                remote_tp_size,
                K,
            )
        tt_set = set(all_source_ranks)
        for t in fa_source_ranks:
            if t not in tt_set:
                logger.error(
                    "FA source rank %d NOT in all_source_ranks %s.",
                    t,
                    all_source_ranks,
                )
        if self.is_kv_layout_blocks_first and tp_ratio < 0 and num_fa_reads > 0:
            local_k_half = local_block_len // 2
            remote_k_half = remote_block_len // 2
            expected = local_k_half // num_fa_reads
            if expected != remote_k_half:
                logger.warning(
                    "FA size mismatch: local_k_half=%d / reads=%d = %d, "
                    "but remote_k_half=%d.",
                    local_k_half,
                    num_fa_reads,
                    expected,
                    remote_k_half,
                )

        return MambaEngineTransferInfo(
            remote_tp_size=remote_tp_size,
            remote_block_len=remote_block_len,
            remote_block_size=remote_block_size,
            remote_physical_blocks_per_logical=(remote_physical_blocks_per_logical),
            remote_fa_source_ranks=tuple(fa_source_ranks),
            remote_all_source_ranks=tuple(all_source_ranks),
            remote_num_fa_reads=num_fa_reads,
            remote_num_mamba_reads=num_mamba_reads,
            remote_fa_descriptor_bytes=fa_descriptor_bytes,
            is_remote_replicated=is_remote_replicated,
            remote_physical_heads=remote_physical_heads,
        )
