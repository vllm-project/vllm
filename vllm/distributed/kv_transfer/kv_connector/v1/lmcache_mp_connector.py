# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Builtin LMCache multiprocess connector.

Synced from `lmcache/integration/vllm/lmcache_mp_connector.py` at LMCache
v0.5.7.post1, plus the LMCache fixes that release needs to run on this vLLM:

- LMCache/LMCache#6457: `CachedRequestData.resumed_req_ids` is gone.
- LMCache/LMCache#6487: declare `kv_cache_access`.
- LMCache/LMCache#6608: read `KVCacheGroup` instead of `KVCacheGroupSpec`
  (see `lmcache_mp_kv_cache_groups` and `lmcache_mp_kv_cache_group_edits`).

`LMCacheMPConnector` at the bottom of this module prefers the connector the
installed `lmcache` package ships and falls back to this one.
"""

import math
import os
import sys
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
import zmq
from lmcache.banner import print_banner_once
from lmcache.integration.vllm.experimental import dispatch
from lmcache.integration.vllm.lazy_offload_manager import LazyOffloadManager
from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPConnectorMetadata,
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
    LMCacheMPWorkerMetadata,
)
from lmcache.integration.vllm.lmcache_mp_metrics import (
    LMCacheMPConnectorStats,
    LMCacheMPPromMetrics,
)
from lmcache.integration.vllm.mp_server_launcher import (
    is_mp_server_autostart_enabled,
)
from lmcache.integration.vllm.utils import (
    mla_only,
    vllm_layout_hints,
)
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPSchedulerAdapter,
    LMCacheMPWorkerAdapter,
    ParallelStrategy,
)
from lmcache.utils import CacheRemoveEvent
from lmcache.utils import init_logger as lmcache_init_logger
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord as RequestAllocationRecord,
)

from vllm.config import KVTransferConfig, VllmConfig
from vllm.config.kv_transfer import KVCacheAccess
from vllm.distributed.kv_events import (
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
    KVConnectorKVEvents,
    KVEventAggregator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorTransferResults,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_kv_cache_group_edits import (  # noqa: E501
    apply_kv_cache_group_edits,
    validate_kv_cache_groups,
)
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_kv_cache_groups import (
    create_engine_group_infos_from_vllm,
    group_prefix_replay_tokens,
    group_specs,
    is_scratch_group,
    tokens_per_group_block,
)
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.kv_cache_utils import BlockHash, maybe_convert_block_hash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
        KVConnectorPromMetrics,
        KVConnectorStats,
        PromMetric,
        PromMetricT,
    )
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = lmcache_init_logger(__name__)

_DCP_LAYOUT_NAMESPACE = "##lmcache-dcp-layout-v1-"
_MAX_LCM_EXPANSION_FACTOR = 4


def _convert_kv_event_hash(block_hash: bytes | int | None) -> bytes | int | None:
    """Convert LMCache event hashes to vLLM's configured external format."""
    if isinstance(block_hash, bytes):
        return maybe_convert_block_hash(BlockHash(block_hash))
    return block_hash


class LMCacheMPKVEvents(KVConnectorKVEvents):
    """KV event container used by LMCache multiprocess workers."""

    def __init__(self, num_workers: int) -> None:
        self._aggregator = KVEventAggregator(num_workers)
        self._batches: list[list[KVCacheEvent]] = []

    def add_events(self, events: list[KVCacheEvent]) -> None:
        """Add events from one worker."""
        self._aggregator.add_events(events)
        self._batches.append(list(events))

    def aggregate(self) -> "LMCacheMPKVEvents":
        """Keep all pollers' events, preserving transitions within each batch."""
        remaining = set(self._aggregator.get_all_events())
        merged: list[KVCacheEvent] = []
        for events in self._batches:
            merged.extend(event for event in events if event in remaining)
            remaining.difference_update(events)
        self._batches = [merged] if merged else []
        self._aggregator.clear_events()
        self._aggregator.add_events(merged)
        self._aggregator.reset_workers()
        return self

    def increment_workers(self, count: int = 1) -> None:
        """Track an additional worker contribution."""
        self._aggregator.increment_workers(count)

    def get_all_events(self) -> list[KVCacheEvent]:
        """Return all buffered events in worker order."""
        return [event for batch in self._batches for event in batch]

    def get_number_of_workers(self) -> int:
        """Return the number of contributing workers."""
        return self._aggregator.get_number_of_workers()

    def clear_events(self) -> None:
        """Clear buffered events and reset the worker count."""
        self._aggregator.clear_events()
        self._aggregator.reset_workers()
        self._batches.clear()


# Helper functions
def _has_preemption_reqs(scheduler_output: SchedulerOutput) -> bool:
    """Return whether the scheduler output contains preemption-related requests.

    Checks for the presence of resumed or preempted requests in the
    scheduler output.

    A preemption is detected if:
    - `scheduled_cached_reqs.resumed_req_ids`: Requests resumed from
      preemption this step.
    - `scheduler_output.preempted_req_ids`: Requests preempted this step.

    Args:
        scheduler_output: The vLLM scheduler output for this step.

    Returns:
        True if preemption-related requests exist, False otherwise.

    """
    cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)

    # Primary signal: requests resumed from preemption this step.
    resumed_ids = getattr(cached_reqs, "resumed_req_ids", None)
    if resumed_ids:
        logger.warning("<preempted> by resumed requests: %s", resumed_ids)
        return True

    # Primary signal: requests preempted this step.
    preempted_ids = getattr(scheduler_output, "preempted_req_ids", None)
    if preempted_ids:
        logger.warning("<preempted> by preempted requests: %s", preempted_ids)
        return True

    return False


def _iter_kv_cache_specs(kv_cache_config: "KVCacheConfig | None") -> Iterable[Any]:
    """Yield leaf KV cache specs without depending on a vLLM helper API."""
    if kv_cache_config is None:
        return
    for group in kv_cache_config.kv_cache_groups:
        yield from group_specs(group)


def get_group_tokens_per_block(
    vllm_config: VllmConfig,
    kv_cache_config: "KVCacheConfig | None",
) -> list[int]:
    """Return each KV cache group's effective block span in tokens.

    Attention pages are local DCP shards and therefore cover
    `spec.block_size * dcp_size` global tokens. Recurrent-state pages are
    replicated and retain their physical `spec.block_size` span. Scratch
    groups cover no tokens and report `0`. When vLLM does not provide
    group metadata, preserve the legacy single-group rule.

    Args:
        vllm_config: The active vLLM configuration.
        kv_cache_config: vLLM's resolved KV cache group configuration.

    Returns:
        The effective token span of each KV cache group, `0` for scratch
        groups that never store or retrieve.

    """
    dcp_size = getattr(vllm_config.parallel_config, "decode_context_parallel_size", 1)
    groups = (
        getattr(kv_cache_config, "kv_cache_groups", ()) or ()
        if kv_cache_config is not None
        else ()
    )
    return [tokens_per_group_block(group, dcp_size) for group in groups] or [
        vllm_config.cache_config.block_size * dcp_size
    ]


def get_prefix_replay_tokens(kv_cache_config: "KVCacheConfig | None") -> int:
    """Return vLLM's prefix-replay window, or 0 when no group declares one.

    Sliding-window bounded replay (DeepSeek V4.1) makes the scheduler floor a
    connector's external hit to the scheduler block boundary before it
    allocates, so the connector must hand back block-aligned counts.
    """
    groups = (
        getattr(kv_cache_config, "kv_cache_groups", ()) or ()
        if kv_cache_config is not None
        else ()
    )
    return max(
        (group_prefix_replay_tokens(group) for group in groups),
        default=0,
    )


def get_vllm_scheduler_block_size(
    vllm_config: VllmConfig,
    kv_cache_config: "KVCacheConfig | None",
) -> int:
    """Return vLLM's scheduler block size for the resolved cache groups.

    The scheduler boundary must align with every cache group, so vLLM uses the
    least common multiple of their effective block spans. Scratch groups
    (span `0`) do not take part.

    Args:
        vllm_config: The active vLLM configuration.
        kv_cache_config: vLLM's resolved KV cache group configuration.

    Returns:
        The scheduler block size in tokens.

    """
    group_spans = [
        span
        for span in get_group_tokens_per_block(vllm_config, kv_cache_config)
        if span > 0
    ]
    scheduler_block_size = math.lcm(*group_spans)
    largest_group_span = max(group_spans)
    if scheduler_block_size > largest_group_span * _MAX_LCM_EXPANSION_FACTOR:
        logger.warning(
            "LMCache resolved a scheduler block of %d tokens from group spans "
            "%s (%0.1fx the largest span). Near-coprime group geometry can "
            "make cache chunks too coarse to be useful.",
            scheduler_block_size,
            group_spans,
            scheduler_block_size / largest_group_span,
        )
    return scheduler_block_size


def get_dcp_decorated_model_name(
    vllm_config: VllmConfig,
    kv_cache_config: "KVCacheConfig | None" = None,
) -> str:
    """Decorate the model name with its non-trivial DCP interleave layout.

    Embedding the DCP interleave size in the LMCache model name prevents a
    changed interleave from hitting objects with an incompatible byte layout.

    Args:
        vllm_config: The active vLLM configuration.
        kv_cache_config: vLLM's resolved KV cache group configuration.

    Returns:
        The decorated cache model name, or the original model name when DCP or
        interleaving is inactive.

    """
    model_name = vllm_config.model_config.model
    parallel_config = vllm_config.parallel_config
    dcp_size = getattr(parallel_config, "decode_context_parallel_size", 1)
    interleave = original = getattr(parallel_config, "cp_kv_cache_interleave_size", 1)
    # adjust_dcp_kv_cache_interleave_size runs per worker, so the scheduler's
    # config keeps the pre-adjustment value. Delegate to vLLM (idempotent, and
    # gated differently across versions), then restore what it owns.
    adjust = getattr(vllm_config, "adjust_dcp_kv_cache_interleave_size", None)
    if adjust is not None and kv_cache_config is not None:
        try:
            adjust(kv_cache_config)
            interleave = getattr(
                parallel_config, "cp_kv_cache_interleave_size", original
            )
        finally:
            parallel_config.cp_kv_cache_interleave_size = original
    if dcp_size <= 1 or interleave == 1:
        return model_name
    return f"{model_name}{_DCP_LAYOUT_NAMESPACE}d{dcp_size}-interleave{interleave}"


def get_resolved_attention_block_sizes(
    vllm_config: VllmConfig,
    kv_cache_config: "KVCacheConfig | None",
) -> set[int]:
    """Return physical block sizes for resolved attention cache groups.

    Scratch groups are attention specs by class but cover no tokens, so they
    are left out.

    Args:
        vllm_config: The active vLLM configuration.
        kv_cache_config: vLLM's resolved KV cache group configuration.

    Returns:
        The resolved attention block sizes. Falls back to vLLM's cache block
        size when group metadata is unavailable.

    """
    groups = kv_cache_config.kv_cache_groups if kv_cache_config is not None else ()
    block_sizes = {
        spec.block_size
        for group in groups
        if not is_scratch_group(group)
        for spec in group_specs(group)
        if any(cls.__name__ == "AttentionSpec" for cls in type(spec).__mro__)
    }
    if block_sizes:
        return block_sizes
    return {vllm_config.cache_config.block_size}


def validate_mamba_step_alignment(
    vllm_config: VllmConfig,
    kv_cache_config: "KVCacheConfig | None" = None,
) -> None:
    """Reject scheduler configs whose steps cannot advance a whole Mamba block.

    In `mamba_cache_mode="align"` vLLM snapshots the recurrent state only at
    the end of each scheduler step, on the last block the step advanced. A step
    advancing more than one block fills the skipped block-table positions with
    the null block (`MambaManager.allocate_new_blocks`); LMCache handles those
    safely -- the request tracker nulls the slot of a relocated speculative
    block, `store` never commits an all-null-block chunk and `retrieve`
    loads only each object group's sliding-window suffix -- so
    `max_num_batched_tokens` may exceed `2 * block_size` (with
    `--separate-object-groups`). Only the lower bound remains: a step must
    advance at least one full block, or vLLM's block-aligned splitting
    (`Scheduler._mamba_block_aligned_split`) yields empty chunks and prefill
    cannot progress.

    Args:
        vllm_config: The vLLM config; only Mamba-hybrid models in `align`
            cache mode are constrained, others pass.
        kv_cache_config: vLLM's resolved KV cache group configuration.

    Returns:
        None.

    Raises:
        ValueError: If Mamba groups have inconsistent block sizes or
            `max_num_batched_tokens < block_size`.

    """
    if getattr(vllm_config.cache_config, "mamba_cache_mode", "none") != "align":
        return
    mamba_block_sizes = {
        spec.block_size
        for spec in _iter_kv_cache_specs(kv_cache_config)
        if any(cls.__name__ == "MambaSpec" for cls in type(spec).__mro__)
    }
    if len(mamba_block_sizes) > 1:
        raise ValueError(
            "All Mamba KV cache groups must use one physical block size; got "
            f"{sorted(mamba_block_sizes)}."
        )
    block_size = (
        next(iter(mamba_block_sizes))
        if mamba_block_sizes
        else vllm_config.cache_config.block_size
    )
    max_batched = vllm_config.scheduler_config.max_num_batched_tokens
    if max_batched < block_size:
        raise ValueError(
            f"Mamba-hybrid models with LMCache require "
            f"max_num_batched_tokens >= block_size so every prefill step "
            f"advances at least one full block; got "
            f"max_num_batched_tokens={max_batched}, block_size={block_size}. "
            f"Set --max-num-batched-tokens to at least {block_size}."
        )


def validate_dcp_support(
    vllm_config: VllmConfig,
    n_servers: int,
    kv_cache_config: "KVCacheConfig | None" = None,
) -> None:
    """Reject decode-context-parallel topologies this connector cannot serve.

    These would silently store wrong or incomplete KV, so fail at startup
    instead. `dcp > tp` is not re-checked; vLLM's `ParallelConfig`
    already rejects it.

    Args:
        vllm_config: The active vLLM configuration.
        n_servers: The number of LMCache servers backing the deployment.
        kv_cache_config: vLLM's resolved KV cache group configuration.

    Returns:
        None.

    Raises:
        ValueError: On an unsupported DCP topology (`dcp_size == 1`
            always passes).

    """
    pc = vllm_config.parallel_config
    dcp_size = getattr(pc, "decode_context_parallel_size", 1)
    if dcp_size <= 1:
        return

    pcp_size = getattr(pc, "prefill_context_parallel_size", 1)
    if pcp_size > 1:
        raise ValueError(
            "LMCacheMPConnector does not support prefill-context parallelism "
            f"together with DCP (got pcp={pcp_size}, dcp={dcp_size})."
        )

    interleave = getattr(pc, "cp_kv_cache_interleave_size", 1)
    attention_block_sizes = get_resolved_attention_block_sizes(
        vllm_config, kv_cache_config
    )
    incompatible_block_sizes = sorted(
        block_size
        for block_size in attention_block_sizes
        if interleave > block_size or block_size % interleave != 0
    )
    if incompatible_block_sizes:
        raise ValueError(
            f"cp_kv_cache_interleave_size ({interleave}) must be no greater "
            "than and evenly divide every resolved attention block size; "
            f"incompatible sizes: {incompatible_block_sizes}."
        )

    ranks_per_server = pc.world_size // n_servers
    if ranks_per_server < dcp_size:
        raise ValueError(
            f"Each LMCache server needs at least decode_context_parallel_size "
            f"({dcp_size}) ranks to hold a complete set of shards, but "
            f"{n_servers} server(s) leave only {ranks_per_server} rank(s) "
            "each. Use fewer servers or a smaller DCP size."
        )


def build_parallel_strategy_from_vllm_config(
    vllm_config: "VllmConfig",
    n_servers: int,
) -> ParallelStrategy:
    """Build a ParallelStrategy from a vLLM config.

    Centralises the (vllm_config -> KV parallel geometry) mapping.

    Args:
        vllm_config: The vLLM configuration object.
        n_servers: Number of LMCache servers backing this deployment.

    Returns:
        The constructed ParallelStrategy.

    """
    pc = vllm_config.parallel_config
    return ParallelStrategy(
        mla_only=mla_only(vllm_config.model_config),
        vllm_world_size=pc.world_size,
        vllm_worker_id=pc.rank,
        tp_size=pc.tensor_parallel_size,
        pp_size=pc.pipeline_parallel_size,
        n_servers=n_servers,
        dcp_size=getattr(pc, "decode_context_parallel_size", 1),
    )


def _ensure_transport_scheme(server_url: str) -> str:
    """Ensure a multiprocess server URL carries a transport scheme.

    Preserve explicit schemes so the request client factory can select the
    transport. Bare endpoints retain the legacy ZMQ TCP behavior.

    Args:
        server_url: A server URL, with or without a `<scheme>://` prefix.

    Returns:
        The URL with a transport scheme, defaulting to `tcp://`.

    """
    if "://" in server_url:
        return server_url
    return f"tcp://{server_url}"


class LMCacheMPConnectorUpstream(KVConnectorBase_V1, SupportsHMA):
    """The connector for LMCache multi-process mode.

    Extra configs (kv_transfer_config.extra_config):

    Multi-server deployment:
    - lmcache.mp.server_urls: server URL list or comma-separated string,
      e.g. "tcp://host1:6667,tcp://host2:6667".

    Single-server deployment:
    - lmcache.mp.host: the host of the LMCache server.
    - lmcache.mp.port: the port of the LMCache server.

    - lmcache.mp.mq_timeout: timeout (seconds) for message queue requests.
    - lmcache.mp.heartbeat_interval: interval (seconds) between server
      heartbeat pings.
    - lmcache.mp.eager_prefetch: submit the LMCache lookup when a request
      enters vLLM's waiting queue. Disabled by default.
    - lmcache.mp.save_decode_cache: also store chunks containing generated
      tokens. Defaults to False; only complete prompt chunks are stored.
    """

    # Tail block slots vLLM may relocate for one request; 0 means vLLM only
    # appends. The scheduler role sets it from the vLLM config.
    _mamba_relocation_window: int = 0

    @classmethod
    def kv_cache_access(cls, kv_transfer_config: KVTransferConfig) -> KVCacheAccess:
        # Shares the KV cache with the LMCache server via cudaIpcGetMemHandle.
        return KVCacheAccess.LEGACY_IPC

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ) -> None:
        """Initialize a worker or scheduler connector from vLLM configuration.

        Args:
            vllm_config: Engine configuration, including connector extra config.
            role: Scheduler or worker role.
            kv_cache_config: Resolved cache groups, if supplied by vLLM.

        Raises:
            ValueError: If cache geometry or auto-start configuration is invalid,
                including auto-start with multiple server endpoints.
            ConnectionError: If the configured MP server cannot be reached.

        """
        # Older supported vLLM releases allow connectors to omit this value,
        # while current vLLM's type declaration requires it.
        super().__init__(vllm_config, role, kv_cache_config)  # type: ignore[arg-type]
        self._connector_stats = LMCacheMPConnectorStats()

        # Fail fast, before the server handshake below.
        kv_cache_config = getattr(self, "_kv_cache_config", None)
        validate_mamba_step_alignment(vllm_config, kv_cache_config)
        validate_kv_cache_groups(kv_cache_config)
        # Flat block errors cannot describe multi-group failures on new vLLM.
        # Keep single-group reporting so valid partial prefixes remain reusable.
        self._report_request_errors = (
            kv_cache_config is not None and len(kv_cache_config.kv_cache_groups) > 1
        )

        group_tokens_per_block = get_group_tokens_per_block(
            vllm_config, kv_cache_config
        )
        self._prefix_replay_tokens = get_prefix_replay_tokens(kv_cache_config)
        mamba_cache_mode = getattr(vllm_config.cache_config, "mamba_cache_mode", "none")
        self._reserve_last_token_for_lookup = mamba_cache_mode in ("align", "all")
        scheduler_block_size = get_vllm_scheduler_block_size(
            vllm_config, kv_cache_config
        )
        cache_model_name = get_dcp_decorated_model_name(vllm_config, kv_cache_config)

        assert vllm_config.kv_transfer_config is not None
        self._can_store = vllm_config.kv_transfer_config.is_kv_producer
        self._save_decode_cache = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache.mp.save_decode_cache", False
        )
        if not isinstance(self._save_decode_cache, bool):
            raise ValueError("lmcache.mp.save_decode_cache must be a boolean")

        kv_events_config = vllm_config.kv_events_config
        self._enable_kv_events = bool(
            kv_events_config is not None and kv_events_config.enable_kv_cache_events
        )

        self._eager_prefetch: bool = bool(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "lmcache.mp.eager_prefetch", False
            )
        )

        # Multi-server: prefer lmcache.mp.server_urls (list or comma-separated
        # string) over the single-server lmcache.mp.host / lmcache.mp.port.
        server_urls_cfg = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache.mp.server_urls", None
        )
        if server_urls_cfg:
            if isinstance(server_urls_cfg, list):
                server_urls = [u.strip() for u in server_urls_cfg if u.strip()]
            else:
                server_urls = [
                    u.strip() for u in server_urls_cfg.split(",") if u.strip()
                ]
        else:
            # Legacy single-server fallback.
            server_host = vllm_config.kv_transfer_config.get_from_extra_config(
                "lmcache.mp.host", "tcp://localhost"
            )
            server_port = vllm_config.kv_transfer_config.get_from_extra_config(
                "lmcache.mp.port", 5555
            )
            server_urls = [f"{server_host}:{server_port}"]

        # Preserve explicit transport schemes and default bare host:port
        # endpoints to the legacy ZMQ TCP transport.
        server_urls = [_ensure_transport_scheme(u) for u in server_urls]

        # The server count is derived from lmcache.mp.server_urls.
        n_servers = len(server_urls)
        if (
            is_mp_server_autostart_enabled(
                vllm_config.kv_transfer_config.kv_connector_extra_config
            )
            and n_servers > 1
        ):
            raise ValueError(
                "LMCache MP auto-start only supports a single server; "
                "start multiple servers separately and disable lmcache.mp.autostart."
            )

        validate_dcp_support(vllm_config, n_servers, kv_cache_config)

        logger.info(
            "Resolved LMCache MP geometry: group_tokens_per_block=%s, "
            "scheduler_block_size=%d, cache_model_name=%s",
            group_tokens_per_block,
            scheduler_block_size,
            cache_model_name,
        )

        assert vllm_config.parallel_config.world_size % n_servers == 0, (
            f"world_size ({vllm_config.parallel_config.world_size}) must be "
            f"divisible by n_servers ({n_servers})"
        )

        # Multi-server + DP is not supported yet.
        dp_size = getattr(vllm_config.parallel_config, "data_parallel_size", 1)
        if n_servers > 1 and dp_size > 1:
            raise ValueError(
                "LMCacheMPConnector multi-server mode (n_servers > 1) does not "
                f"support data parallelism yet; got dp_size={dp_size}. "
                "DP across multiple LMCache servers will be "
                "supported in a follow-up PR."
            )

        # Multi-server + MLA: only TP is supported (no PP).
        # PP splits layers across nodes, which would cause per-piece
        # reader counts to vary per (server, pp_stage) pair and break
        # the single-``tp_size`` LOOKUP / FREE_LOOKUP_LOCKS protocol.
        # Non-MLA mode is not affected by this restriction.
        if n_servers > 1:
            pp_size = vllm_config.parallel_config.pipeline_parallel_size
            if pp_size > 1:
                raise ValueError(
                    "LMCacheMPConnector multi-server mode only supports "
                    "tensor parallelism (TP), not pipeline parallelism (PP). "
                    f"Got pp_size={pp_size}."
                )

        zmq_context = zmq.Context.instance()
        parallel_strategy = build_parallel_strategy_from_vllm_config(
            vllm_config, n_servers
        )

        self.dispatcher = None

        dcp_size = parallel_strategy.dcp_size
        self._dcp_size = dcp_size

        # Lazy offload configuration: when enabled, store operations are
        # buffered and drained by the configured policy.
        self.lazy_offload = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache.mp.lazy_offload", False
        )
        if self.lazy_offload and not vllm_config.cache_config.enable_prefix_caching:
            # Eviction detection relies on block hashes, which only exist
            # when vLLM's prefix caching maintains them.
            raise ValueError(
                "lmcache.mp.lazy_offload requires vLLM prefix caching "
                "(enable_prefix_caching=True)"
            )

        if self.role == KVConnectorRole.SCHEDULER:
            # Banner from the scheduler role only, so tensor-parallel
            # deployments print it once rather than once per worker.
            print_banner_once(sys.stderr)
            self.scheduler_adapter = LMCacheMPSchedulerAdapter(
                server_urls=server_urls,
                context=zmq_context,
                model_name=cache_model_name,
                vllm_block_size=scheduler_block_size,
                parallel_strategy=parallel_strategy,
                extra_config=vllm_config.kv_transfer_config.kv_connector_extra_config,
                required_chunk_alignment=scheduler_block_size,
            )
            self.request_trackers: dict[str, LMCacheMPRequestTracker] = {}
            # Worker events aggregated per step, awaiting take_events().
            self._kv_cache_events: list[KVCacheEvent] = []

            # Align-mode Mamba keeps one speculative block per draft token at
            # the tail of a request's block list.
            spec_config = getattr(vllm_config, "speculative_config", None)
            self._mamba_relocation_window = (
                spec_config.num_speculative_tokens or 0
                if mamba_cache_mode == "align" and spec_config is not None
                else 0
            )

            # GPU block pool reference
            self._gpu_block_pool: BlockPool | None = None
        elif self.role == KVConnectorRole.WORKER:
            # Node routing: a worker connects only to its local LMCache server.
            # Global ranks are assigned to nodes in contiguous blocks:
            #   node 0 → ranks [0, ranks_per_node),
            #   node 1 → [ranks_per_node, 2 * ranks_per_node), ...
            ranks_per_node = parallel_strategy.vllm_world_size // n_servers
            local_server_url = server_urls[
                parallel_strategy.vllm_worker_id // ranks_per_node
            ]
            self.worker_adapter = self.create_lmcache_mp_worker_adapter(
                server_url=local_server_url,
                context=zmq_context,
                model_name=cache_model_name,
                vllm_block_size=scheduler_block_size,
                parallel_strategy=parallel_strategy,
                extra_config=vllm_config.kv_transfer_config.kv_connector_extra_config,
                enable_kv_events=self._enable_kv_events,
                required_chunk_alignment=scheduler_block_size,
            )
            # Whether vLLM called get_kv_connector_kv_cache_events this step; it
            # never does for a MultiConnector child (see build_connector_worker_meta).
            self._kv_events_hook_called = False
            if self.transfer_intermediate_tensors:
                from lmcache.integration.vllm.experimental import (
                    FeatureContext,
                    init_dispatcher,
                )
                from lmcache.v1.multiprocess.custom_types import TRANSFER_QUERY

                ctx = FeatureContext(
                    worker_adapter=self.worker_adapter,
                )
                requested = (
                    {TRANSFER_QUERY} if self.transfer_intermediate_tensors else set()
                )
                self.dispatcher = init_dispatcher(ctx, requested)
                self.worker_adapter.dispatcher = self.dispatcher
        else:
            raise ValueError(f"Unknown KVConnectorRole: {self.role}")

        # Tokens covered by one paged chunk (one block ID) of each engine
        # group, from the group's KV cache spec. Hybrid models can mix
        # different values (e.g. gemma-4: sliding-window groups 32,
        # full-attention groups 16; DeepSeek V4: 256/64/8/4); scratch
        # groups report 0 and take no part in caching. Falls back to
        # the engine's base block size when no group metadata is available
        # (single non-hybrid group).
        self._group_tokens_per_block = group_tokens_per_block
        cached_spans = [span for span in group_tokens_per_block if span > 0]
        if not cached_spans or min(group_tokens_per_block) < 0:
            raise ValueError(
                f"group tokens_per_block {group_tokens_per_block} must be "
                "non-negative with at least one cacheable group"
            )
        # Smallest token count aligned to every group's paged-chunk
        # boundary; used to round down vLLM APC hit counts.
        self._hit_alignment_tokens = math.lcm(*cached_spans)
        if self.role == KVConnectorRole.SCHEDULER:
            # Chunk boundaries must land on every group's paged-chunk
            # boundary so per-group block-id slicing stays aligned.
            lmcache_tokens_per_chunk = self.scheduler_adapter.lmcache_tokens_per_chunk
            for engine_group_idx, tokens_per_block in enumerate(
                self._group_tokens_per_block
            ):
                if tokens_per_block and lmcache_tokens_per_chunk % tokens_per_block:
                    raise ValueError(
                        f"LMCache chunk size {lmcache_tokens_per_chunk} must be "
                        f"a multiple of group {engine_group_idx} "
                        f"tokens_per_block {tokens_per_block}"
                    )
            if self.lazy_offload:
                self._lazy_offload_manager = LazyOffloadManager(
                    vllm_config.kv_transfer_config.kv_connector_extra_config,
                    self._group_tokens_per_block,
                    self.scheduler_adapter,
                )

    @property
    def role(self) -> KVConnectorRole:
        return self._role

    @property
    def transfer_intermediate_tensors(self) -> bool:
        cfg = self._vllm_config.kv_transfer_config
        if cfg is None:
            return False
        vllm_transfer = cfg.get_from_extra_config(
            "lmcache.mp.transfer_intermediate_tensors", False
        )
        return vllm_transfer

    # ==============================
    # Worker-side methods
    # ==============================

    def create_lmcache_mp_worker_adapter(
        self, **adapter_kwargs: Any
    ) -> LMCacheMPWorkerAdapter:
        """Build the worker adapter.

        Subclasses override this to use their own adapter. The keyword
        arguments are those of `LMCacheMPWorkerAdapter`.
        """
        return LMCacheMPWorkerAdapter(**adapter_kwargs)

    def _get_connector_metadata(self) -> KVConnectorMetadata:
        """Get the connector metadata.

        This function should only be called inside the connector.

        Returns:
            ConnectorMetadata: the connector metadata.

        """
        # Should only be called while set to valid metadata.
        assert self._connector_metadata is not None
        return self._connector_metadata

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args:
            kv_caches: dictionary of layer names, kv cache

        """
        logger.info("Registering kv caches!")
        kv_cache_config = getattr(self, "_kv_cache_config", None)
        # Must precede both group-info creation and transfer registration so
        # they see the same edited views.
        layout_hints = vllm_layout_hints(self._vllm_config)
        kv_caches = apply_kv_cache_group_edits(
            kv_cache_config, kv_caches, layout_hints=layout_hints
        )
        engine_group_infos = create_engine_group_infos_from_vllm(
            kv_cache_config,
            kv_caches,
            layout_hints=layout_hints,
            dcp_size=self._dcp_size,
        )
        self.worker_adapter.register_kv_caches(
            kv_caches,
            engine_group_infos=engine_group_infos,
            layout_hints=layout_hints,
        )
        if self.dispatcher is not None:
            dispatch(
                self.dispatcher,
                "register",
                kv_caches=kv_caches,
                kv_cache_config=kv_cache_config,
                vllm_config=self._vllm_config,
            )
        return

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, LMCacheMPConnectorMetadata)

        request_ids = []
        ops = []
        cache_salts = []
        request_configs_list = []

        for meta in metadata.requests:
            if meta.direction != "RETRIEVE":
                continue
            request_ids.append(meta.request_id)
            ops.append(meta.op)
            cache_salts.append(meta.cache_salt)
            request_configs_list.append(meta.request_configs)

        if len(request_ids) == 0:
            return

        event = self.worker_adapter.create_recorded_event()

        self.worker_adapter.batched_submit_retrieve_requests(
            request_ids,
            ops,
            event,
            cache_salts=cache_salts,
            request_configs_list=request_configs_list,
        )

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer

        """
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        """Start saving a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.

        """
        if self.dispatcher is not None:
            dispatch(
                self.dispatcher,
                "save_kv_layer",
                layer_name=layer_name,
                metadata=self._get_connector_metadata(),
                attn_metadata=attn_metadata,
                **kwargs,
            )
        return

    def wait_for_save(self):
        """Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, LMCacheMPConnectorMetadata)

        request_ids = []
        ops = []
        cache_salts = []
        request_configs_list = []
        for meta in metadata.requests:
            if meta.direction != "STORE":
                continue
            request_ids.append(meta.request_id)
            ops.append(meta.op)
            cache_salts.append(meta.cache_salt)
            request_configs_list.append(meta.request_configs)

        if len(request_ids) == 0:
            if self.dispatcher is not None:
                dispatch(self.dispatcher, "wait_for_save", event=None)
            return

        event = self.worker_adapter.create_recorded_event()

        self.worker_adapter.batched_submit_store_requests(
            request_ids,
            ops,
            event,
            cache_salts=cache_salts,
            request_configs_list=request_configs_list,
        )
        if self.dispatcher is not None:
            dispatch(self.dispatcher, "wait_for_save", event=event)

    # TODO: How does lmcache driven path handle preemption?
    # NOTE1: handle_preemptions is called by vllm each step regardless
    #        preemption really happens or not.
    # NOTE2: preemption hint is managed by KVConnectorRole.SCHEDULER,
    #        that's why here we have to judge preemption by
    #        need_flush_before_forward flag which is set by SCHEDULER.
    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        """Flush async engine-driven stores only when scheduler metadata requests it.

        Args:
            kv_connector_metadata: Connector metadata produced by the scheduler;
                only acts when it is a `LMCacheMPConnectorMetadata` with
                `need_flush_before_forward=True`.

        """
        worker_adapter = getattr(self, "worker_adapter", None)
        if self.role != KVConnectorRole.WORKER or worker_adapter is None:
            return
        need_flush_before_forward = (
            isinstance(kv_connector_metadata, LMCacheMPConnectorMetadata)
            and kv_connector_metadata.need_flush_before_forward
        )
        worker_adapter.handle_preemptions(need_flush_before_forward)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Notifies worker-side connector ids of requests that have
        finished generating tokens on the worker.
        The scheduler process (via the Executors) will use this output
        to track which workers are done.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).

        """
        if self.lazy_offload:
            val = self.worker_adapter.get_finished_with_lazy_offload()
        else:
            # The adapter reports engine-finished IDs even without a STORE.
            # Consumers never delay frees for saves, but must still poll retrieves.
            val = self.worker_adapter.get_finished(
                finished_req_ids if self._can_store else set()
            )
        # logger.error("Finished req ids: %s, %s", val[0], val[1])
        return val

    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> KVConnectorTransferResults:
        """Poll completed transfers and report multi-group failures by request.

        Args:
            finished_req_ids: Requests whose generation has finished in vLLM.

        Returns:
            Completed sends and receives, with failed multi-group receives also
            in `failed_recving`. vLLM aggregates all workers' completions
            before recovering the request. Single-group errors remain available
            through `get_block_ids_with_load_errors`.

        """
        sending, receiving = self.get_finished(finished_req_ids)
        return KVConnectorTransferResults(
            finished_sending=set(sending or ()),
            finished_recving=set(receiving or ()),
            failed_recving=(
                self.worker_adapter.get_failed_request_ids()
                if self._report_request_errors
                else set()
            ),
        )

    def build_connector_worker_meta(self) -> LMCacheMPWorkerMetadata | None:
        """Build this step's worker-to-scheduler metadata.

        Carries lazy-offload store results, and the step's KV events when vLLM
        did not call `get_kv_connector_kv_cache_events` first (e.g. as a
        `MultiConnector` child, which never forwards that hook).

        Returns:
            The worker metadata, or None when there is nothing to report.

        """
        # TODO: Remove this KV event fallback (and LMCacheMPWorkerMetadata.kv_events)
        # once upstream vLLM forwards get_kv_connector_kv_cache_events through
        # MultiConnector and our minimum supported vLLM includes it.
        kv_events = (
            None
            if self._kv_events_hook_called
            else self.get_kv_connector_kv_cache_events()
        )
        self._kv_events_hook_called = False
        if not self.lazy_offload:
            if kv_events is None:
                return None
            return LMCacheMPWorkerMetadata(
                completed_store_requests={}, kv_events=kv_events
            )
        completed_store_requests = self.worker_adapter.get_completed_store_requests()
        failed_store_requests = self.worker_adapter.get_failed_store_requests()
        if completed_store_requests or failed_store_requests or kv_events:
            return LMCacheMPWorkerMetadata(
                completed_store_requests=completed_store_requests or {},
                failed_store_requests=failed_store_requests or set(),
                kv_events=kv_events,
            )
        else:
            return None

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Get failed block IDs for single-group or legacy vLLM recovery.

        Returns:
            Set of block IDs that encountered load errors.
            Empty if no load errors occurred or failures are reported by request.

        Notes:
            - Applies to both sync- and async-loading requests.
            - Async loading: failed blocks may be reported in any forward pass
              up to and including the pass where the request ID is returned by
              `get_finished()`. Even if failures occur, the request must still
              be reported via `get_finished()`, and the failed block IDs must
              appear here no later than that same pass.
            - Sync loading: failed blocks should be reported in the forward
              pass in which they are detected.

        """
        if self._report_request_errors:
            return set()
        return self.worker_adapter.get_block_ids_with_load_errors()

    def get_kv_connector_kv_cache_events(self) -> LMCacheMPKVEvents | None:
        """Return worker-side KV cache events collected since the last step.

        Returns:
            A vLLM KV event container with `BlockStored` events for LMCache
            stores (own completed stores and stores learned from the MP
            server) and `BlockRemoved` events for CPU evictions, or None
            when disabled or when nothing happened.

        """
        self._kv_events_hook_called = True
        if not self._enable_kv_events:
            return None
        events = self.worker_adapter.get_kv_events()
        if not events:
            return None

        blocks: list[KVCacheEvent] = []
        for event in events:
            block_hashes = [
                _convert_kv_event_hash(block_hash) for block_hash in event.block_hashes
            ]
            if isinstance(event, CacheRemoveEvent):
                blocks.append(
                    BlockRemoved(block_hashes=block_hashes, medium=event.medium)
                )
                continue
            blocks.append(
                BlockStored(
                    block_hashes=block_hashes,
                    parent_block_hash=_convert_kv_event_hash(event.parent_block_hash),
                    token_ids=event.token_ids,
                    lora_id=event.lora_id,
                    block_size=event.block_size,
                    medium=event.medium,
                    lora_name=event.lora_name,
                )
            )
        kv_events = LMCacheMPKVEvents(num_workers=1)
        kv_events.add_events(blocks)
        return kv_events

    def shutdown(self) -> None:
        """Shutdown the connector. This is called when the owning process
        (worker or scheduler) is shutting down to ensure that all the async
        operations are completed and the connector is cleaned up properly.
        On the scheduler side it also logs the lazy-offload counter ledger
        so the drop rate is auditable from the log.
        """
        if hasattr(self, "worker_adapter"):
            self.worker_adapter.shutdown()
        if hasattr(self, "scheduler_adapter"):
            if self.lazy_offload:
                self._lazy_offload_manager.log_final_stats()
            self.scheduler_adapter.shutdown()
        return None

    def get_kv_connector_stats(self) -> "KVConnectorStats | None":
        """Get the KV connector stats collected during the last interval."""
        stats = self._connector_stats.clone_and_reset()
        return None if stats.is_empty() else stats

    # ==============================
    # Scheduler-side methods
    # ==============================

    def reset_cache(self) -> bool | None:
        """Request a best-effort LMCache MP cache clear from the scheduler.

        Active request trackers are preserved. Backing servers retain objects
        protected by in-flight read or write locks.

        Returns:
            True when every MP server answers the clear, False on timeout or
            RPC failure, and None for worker-role connectors.

        """
        if self.role != KVConnectorRole.SCHEDULER:
            return None

        return self.scheduler_adapter.reset_cache()

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        """Bind GPU block pool so that we can touch blocks during stores.
        Called by Scheduler after kv_cache_manager is ready."""
        if self.role == KVConnectorRole.SCHEDULER:
            logger.info("Bind GPU block pool in LMCacheMPConnector scheduler")
            self._gpu_block_pool = gpu_block_pool
            if self.lazy_offload:
                self._lazy_offload_manager.bind_block_pool(gpu_block_pool)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - An optional number of tokens that can be loaded from the
                  external KV cache beyond what is already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if external KV cache tokens will be loaded
                  asynchronously (between scheduler steps). Must be
                  'False' if the first element is 0.

        Notes:
            The connector should only consider the largest prefix of prompt-
            tokens for which KV cache is actually available at the time of the
            call. If the cache cannot be loaded for some tokens (e.g., due to
            connectivity issues or eviction), those tokens must not be taken
            into account.

        """
        tracker = self._get_or_create_request_tracker(request)

        # A failed asynchronous load is bypassed until vLLM admits the request
        # for local computation via update_state_after_alloc().  The scheduler
        # may poll this method repeatedly before that admission; do not submit
        # another lookup or re-enter WAITING_FOR_REMOTE_KVS in the meantime.
        if tracker.state == LMCacheMPRequestState.BYPASS_LMCACHE:
            return 0, False

        # A completed async load normally leaves num_computed_tokens > 0, so
        # the scheduler does not call this method again.  If vLLM reset the
        # request to zero after the worker reported invalid blocks, however,
        # the existing tracker is still READY and describes the failed load.
        # Reusing it would report another external hit without transitioning
        # back through WAITING_FOR_LOAD, leaving the request stuck forever in
        # WAITING_FOR_REMOTE_KVS.  Fail closed for this request instead: drop
        # the stale local lookup/tracker state and let vLLM recompute locally.
        # The server has already released each failed worker's reader share.
        if (
            tracker.state == LMCacheMPRequestState.READY
            and request.num_computed_tokens == 0
            and tracker.num_lmcache_hit_tokens > 0
        ):
            logger.warning(
                "Bypassing LMCache for request %s after a failed async KV load; "
                "the prompt will be recomputed locally.",
                request.request_id,
            )
            self.scheduler_adapter.cleanup_lookup_result(request.request_id)
            tracker.allocated_block_ids.clear()
            tracker.num_stored_tokens = 0
            tracker.num_vllm_hit_tokens = 0
            tracker.num_lmcache_hit_tokens = 0
            tracker.apc_covered_tokens = 0
            tracker.state = LMCacheMPRequestState.BYPASS_LMCACHE
            return 0, False

        # APC-covered lookup start (0 = none; stays 0 after a shrink):
        # the vLLM hit floored to every group's paged-chunk boundary and the
        # LMCache chunk size. It never exceeds the aligned lookup end since
        # vLLM caps the APC hit at num_prompt_tokens - 1; an all-covered
        # lookup is served by the server's empty prefetch job.
        lookup_start = 0
        if not tracker.need_full_lookup:
            tokens_per_chunk = self.scheduler_adapter.lmcache_tokens_per_chunk
            lookup_start = (
                num_computed_tokens
                // self._hit_alignment_tokens
                * self._hit_alignment_tokens
                // tokens_per_chunk
                * tokens_per_chunk
            )

        if tracker.lookup_started_at is None:
            tracker.lookup_started_at = time.monotonic()
        submitted = self.scheduler_adapter.maybe_submit_lookup_request(
            request.request_id,
            packed_token_ids=tracker.packed_token_ids(),
            cache_salt=tracker.cache_salt,
            # vLLM's prefix-cache hit, for the server's KV load policy.
            request_configs={
                **(tracker.request_configs or {}),
                "lmcache.hint.engine_computed_tokens": num_computed_tokens,
            },
            reserve_last_token=self._reserve_last_token_for_lookup,
            start=lookup_start,
        )
        # Freeze only on a real submit: a no-op poll would re-freeze a
        # boundary the in-flight lookup never used.
        if submitted:
            tracker.apc_covered_tokens = lookup_start

        ret = self.scheduler_adapter.check_lookup_result(request.request_id)
        if ret is None:
            return None, True
        assert tracker.lookup_started_at is not None
        self._connector_stats.record_lookup(
            time.monotonic() - tracker.lookup_started_at
        )
        tracker.lookup_started_at = None

        # Save the vLLM hit count even when LMCache misses. It is rounded
        # down to a boundary aligned for every engine group (a full-prompt
        # APC hit reports num_prompt_tokens - 1), so the retrieve-skip
        # range stays paged-chunk-aligned in all groups.
        tracker.num_vllm_hit_tokens = (
            num_computed_tokens
            // self._hit_alignment_tokens
            * self._hit_alignment_tokens
        )

        # APC shrank below the frozen boundary: stop skipping (sticky),
        # remember the abandoned lookup's locked range for the next FREE,
        # and re-look-up the full prefix in the same scheduler step.
        if (
            tracker.apc_covered_tokens > 0
            and tracker.num_vllm_hit_tokens < tracker.apc_covered_tokens
        ):
            logger.info(
                "APC hit for request %s shrank below the covered boundary "
                "(%d < %d); re-looking-up the full prefix from token 0.",
                request.request_id,
                tracker.num_vllm_hit_tokens,
                tracker.apc_covered_tokens,
            )
            if ret > tracker.apc_covered_tokens:
                tracker.pending_free_ranges.append((tracker.apc_covered_tokens, ret))
            self.scheduler_adapter.cleanup_lookup_result(request.request_id)
            tracker.need_full_lookup = True
            tracker.lookup_started_at = None
            tracker.apc_covered_tokens = 0
            tracker.num_stored_tokens = 0
            tracker.num_lmcache_hit_tokens = 0
            if self.scheduler_adapter.maybe_submit_lookup_request(
                request.request_id,
                packed_token_ids=tracker.packed_token_ids(),
                cache_salt=tracker.cache_salt,
                request_configs={
                    **(tracker.request_configs or {}),
                    "lmcache.hint.engine_computed_tokens": num_computed_tokens,
                },
                reserve_last_token=self._reserve_last_token_for_lookup,
            ):
                tracker.lookup_started_at = time.monotonic()
            return None, True

        if ret == 0:
            return 0, False

        assert ret % self.scheduler_adapter.lmcache_tokens_per_chunk == 0

        tracker.num_stored_tokens = ret
        tracker.num_lmcache_hit_tokens = ret

        need_to_load = max(0, ret - num_computed_tokens)

        if request.status == RequestStatus.PREEMPTED:
            logger.info(
                "<resume-load> req=%s apc=%d lmcache=%d load=%d",
                request.request_id,
                num_computed_tokens,
                ret,
                need_to_load,
            )

        # In full-prompt-hit case, we need to recompute the last token.
        # Without this, num_computed_tokens would equal request.num_tokens,
        # causing num_new_tokens to be 0 and triggering the
        # `assert num_new_tokens > 0` in the scheduler.
        if ret == len(request.all_token_ids):
            if self._prefix_replay_tokens > 0:
                # SWA bounded replay floors the hit to a block boundary before
                # allocating, so give back a whole chunk instead of one token.
                chunk = self.scheduler_adapter.lmcache_tokens_per_chunk
                aligned = max(0, (ret - 1) // chunk * chunk)
                if aligned < ret:
                    tracker.pending_free_ranges.append((aligned, ret))
                    tracker.num_lmcache_hit_tokens = aligned
                need_to_load = max(0, aligned - num_computed_tokens)
            else:
                need_to_load = max(0, need_to_load - 1)

        logger.debug(
            "vLLM hit is: %d, Need to load is %d", num_computed_tokens, need_to_load
        )
        return need_to_load, need_to_load > 0

    def on_new_request(self, request: "Request") -> None:
        """Submit an LMCache lookup when a request enters the waiting queue.

        Args:
            request (Request): The request object.

        """
        if self.role != KVConnectorRole.SCHEDULER:
            return
        if not self._eager_prefetch or request.resumable:
            return

        tracker = self._get_or_create_request_tracker(request)
        if tracker.lookup_started_at is None:
            tracker.lookup_started_at = time.monotonic()
        self.scheduler_adapter.maybe_submit_lookup_request(
            request.request_id,
            packed_token_ids=tracker.packed_token_ids(),
            cache_salt=tracker.cache_salt,
            request_configs=tracker.request_configs,
            reserve_last_token=self._reserve_last_token_for_lookup,
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """Update KVConnector state after block allocation.

        If get_num_new_matched_tokens previously returned True for a
        request, this function may be called twice for that same request -
        first when blocks are allocated for the connector tokens to be
        asynchronously loaded into, and second when any additional blocks
        are allocated, after the load/transfer is complete.

        Args:
            request (Request): the request object.
            blocks (KVCacheBlocks): the blocks allocated for the request.
            num_external_tokens (int): the number of tokens that will be
                loaded from the external KV cache.

        """
        # NOTE: `blocks` comes from kv_cache_manager.get_blocks(request_id),
        # which returns ALL blocks for the request (not just newly allocated).
        # This function may be called twice for async-load requests:
        #   1st call: blocks = initial allocation (APC + fresh)
        #   2nd call: blocks = all blocks
        #  (initial + newly allocated for remaining tokens)
        # We must only append the NEW blocks beyond what's already tracked
        # to avoid duplication, which would corrupt the store path's block indexing.
        tracker = self._get_request_tracker(request.request_id)
        block_ids = blocks.get_block_ids() or ()

        # Only append blocks beyond what's already tracked, per engine group.
        existing_counts = tracker.num_allocated_blocks()
        new_block_ids: list[list[int]] = []
        for engine_group_idx, group_blocks in enumerate(block_ids):
            existing = existing_counts.get(engine_group_idx, 0)
            new_block_ids.append(list(group_blocks[existing:]))
        if any(new_block_ids):
            tracker.append_block_ids(
                tuple(new_block_ids), self._mamba_relocation_window
            )

        # Update the state of the tracker
        if tracker.state == LMCacheMPRequestState.BYPASS_LMCACHE:
            # Returning zero external tokens admitted this request for local
            # computation.  Once vLLM publishes that allocation, normal READY
            # tracking (including later stores) can resume.
            tracker.state = LMCacheMPRequestState.READY
            return

        condition = num_external_tokens > 0 and tracker.needs_retrieve()
        if tracker.state == LMCacheMPRequestState.PREFETCHING:
            # If need to retrieve, change to WAITING_FOR_LOAD
            # Otherwise, change to READY
            tracker.state = (
                LMCacheMPRequestState.WAITING_FOR_LOAD
                if condition
                else LMCacheMPRequestState.READY
            )
            # Clean up lookup future in scheduler adapter
            self.scheduler_adapter.cleanup_lookup_result(request.request_id)

            # Free locks on chunks that vLLM already computed and won't
            # retrieve from LMCache, plus any ranges left locked by this
            # request's abandoned lookups (APC shrink re-lookups).
            free_end = 0
            if tracker.num_lmcache_hit_tokens > 0:
                if not condition:
                    # No retrieve needed — free ALL locked chunks
                    free_end = tracker.num_lmcache_hit_tokens
                else:
                    # Note(Roy): Boundary misalignment between vLLM blocks and LMCache
                    # blocks is handled in free_lookup_locks. It makes sure that if
                    # the last vLLM computed block ends in the middle of a LMCache
                    # block, the end LMCache block is not freed (i.e., floor division)
                    # since it will still be needed by vLLM and such block's lock will
                    # be freed by vLLM's retrieve.
                    free_end = tracker.num_vllm_hit_tokens

            if free_end > 0 or tracker.pending_free_ranges:
                self.scheduler_adapter.free_lookup_locks(
                    packed_token_ids=tracker.packed_token_ids(),
                    start=0,
                    end=free_end,
                    request_id=request.request_id,
                    cache_salt=tracker.cache_salt,
                    request_configs=tracker.request_configs,
                    extra_ranges=tuple(tracker.pending_free_ranges),
                )
                tracker.pending_free_ranges = []
                logger.debug(
                    "Free locks of tokens %d-%d since it is cached by vLLM.",
                    0,
                    free_end,
                )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.

        """
        metadata = LMCacheMPConnectorMetadata()
        metadata.need_flush_before_forward = _has_preemption_reqs(scheduler_output)

        self._process_retrieve_requests(metadata)
        self._process_new_requests(scheduler_output, metadata)
        self._process_cached_requests(scheduler_output, metadata)

        if self.lazy_offload:
            actions = self._lazy_offload_manager.on_scheduler_step(scheduler_output)
            for store_metadata in actions.stores_to_submit:
                metadata.add_request_metadata(store_metadata)
            for request_id in actions.sessions_to_end:
                self.scheduler_adapter.end_session(request_id)

        if len(metadata) > 0:
            logger.debug("Final connector metadata: %s", metadata)

        # Report block allocation deltas to LMCache for observability
        self._report_block_allocation_deltas(scheduler_output)

        return metadata

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.

        """
        # Worker metadata carries the events only when the hook is not called.
        kv_cache_events = connector_output.kv_cache_events or getattr(
            connector_output.kv_connector_worker_meta, "kv_events", None
        )
        if kv_cache_events and isinstance(kv_cache_events, LMCacheMPKVEvents):
            # One aggregated batch per step keeps the workers' event order.
            self._kv_cache_events.extend(kv_cache_events.aggregate().get_all_events())

        if not self.lazy_offload:
            return
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, LMCacheMPWorkerMetadata):
            return
        actions = self._lazy_offload_manager.on_store_results(
            meta.failed_store_requests,
            meta.completed_store_requests,
        )
        for request_id in actions.sessions_to_end:
            self.scheduler_adapter.end_session(request_id)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called exactly once when a request has finished, before its blocks are
        freed.

        The connector may assumes responsibility for freeing the blocks
        asynchronously by returning True.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.

        """
        params: dict[str, Any] | None = getattr(request, "kv_transfer_params", None)
        return_params: dict[str, Any] | None = {} if params is not None else None

        if (
            params is not None
            and return_params is not None
            and "cached_token_stats" in params
        ):
            request_tracker = self._get_request_tracker(request.request_id)
            num_vllm = request_tracker.num_vllm_hit_tokens
            num_lmcache = request_tracker.num_lmcache_hit_tokens
            return_params["cached_token_stats"] = {
                "num_vllm_cached_tokens": num_vllm,
                "num_lmcache_cached_tokens": num_lmcache,
                "num_lmcache_extra_cached_tokens": max(0, num_lmcache - num_vllm),
            }

        # Clean up request tracker to prevent memory leak
        self._cleanup_request_tracker(request.request_id)

        # Drop lookup state for a request aborted before its lookup was
        # consumed (update_state_after_alloc never ran for it). Both the eager
        # and the lazy path need this, so it runs before they diverge.
        self.scheduler_adapter.cleanup_lookup_result(request.request_id)

        if self.lazy_offload:
            # Blocks return to the free queue (False) and remain observable;
            # the manager ends the LMCache session after all deferred stores
            # have either completed or been dropped.
            actions = self._lazy_offload_manager.on_request_finished(request.request_id)
            for request_id in actions.sessions_to_end:
                self.scheduler_adapter.end_session(request_id)
            return False, (return_params or None)

        # Notify LMCache to end the session for this request
        self.scheduler_adapter.end_session(request.request_id)
        return self._can_store, (return_params or None)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """HMA request-finished entry point; cleanup is request-id based."""
        return self.request_finished(request, block_ids[0] if block_ids else [])

    def take_events(self) -> Iterable["KVCacheEvent"]:
        """Take the KV cache events from the connector.

        Returns:
            The KV cache events collected since the last call, in the order
            the workers reported them. The buffer is cleared.

        """
        events = self._kv_cache_events
        self._kv_cache_events = []
        return events

    def has_pending_push_work(self) -> bool:
        """Return whether vLLM should keep stepping for pending push work.

        Returns:
            True when scheduler-side lazy offload has submitted stores waiting
            for worker completion. Queued stores are intentionally excluded:
            they require a model-token step for submission and cannot progress
            during a connector-only step. Non-lazy mode uses vLLM's normal
            delayed-free path and does not need this keepalive.

        """
        if self.role != KVConnectorRole.SCHEDULER or not self.lazy_offload:
            return False

        return self._lazy_offload_manager.has_inflight_store_work()

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        """Defer to vLLM; a connector preference is unsafe for now.

        Connector preferences outrank the default in vLLM's layout
        resolution, but backends that assume a layout without declaring it
        via `supported_kv_cache_layouts` (e.g. the DeepSeek V4 sparse
        attention, which hardcodes NHD) silently read a pool laid out
        differently. LMCache handles every resolved layout, so express no
        preference until such backends declare theirs.
        """
        return None

    def get_finished_count(self) -> int | None:
        """Get the count of requests expected to complete send/receive operations
        via this connector. This method is used to initialize the
        KVOutputAggregator, overwriting the default world_size.

        Returns:
            int: expected sending or receiving completion count.

        """
        return None

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> "KVConnectorStats | None":
        """KVConnectorStats resolution method. This method allows dynamically
        registered connectors to return their own KVConnectorStats object,
        which can implement custom aggregation logic on the data dict.
        """
        return LMCacheMPConnectorStats(data=data or {})

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: "VllmConfig",
        metric_types: dict[type["PromMetric"], type["PromMetricT"]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> "KVConnectorPromMetrics | None":
        """Create a KVConnectorPromMetrics subclass which should register
        per-connector Prometheus metrics and implement observe() to
        expose connector transfer stats via Prometheus.
        """
        return LMCacheMPPromMetrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )

    ##############################
    # Helper functions
    ##############################
    def _process_retrieve_requests(
        self,
        metadata: LMCacheMPConnectorMetadata,
    ) -> None:
        lmcache_tokens_per_chunk = self.scheduler_adapter.lmcache_tokens_per_chunk

        for request_tracker in self.request_trackers.values():
            if request_tracker.state != LMCacheMPRequestState.WAITING_FOR_LOAD:
                continue
            r_metadata = LMCacheMPRequestMetadata.GetRetrieveMetadata(
                request_tracker,
                lmcache_tokens_per_chunk,
                group_tokens_per_block=self._group_tokens_per_block,
            )
            if r_metadata is not None:
                metadata.add_request_metadata(r_metadata)
            request_tracker.state = LMCacheMPRequestState.READY

    def _process_new_requests(
        self,
        scheduler_output: SchedulerOutput,
        metadata: LMCacheMPConnectorMetadata,
    ) -> None:
        lmcache_tokens_per_chunk = self.scheduler_adapter.lmcache_tokens_per_chunk

        for new_request in scheduler_output.scheduled_new_reqs:
            request_tracker = self._get_request_tracker(new_request.req_id)

            num_new_tokens = scheduler_output.num_scheduled_tokens[new_request.req_id]
            request_tracker.increase_num_scheduled_tokens(num_new_tokens)
            request_tracker.update_num_engine_computed_tokens(
                new_request.num_computed_tokens, num_new_tokens
            )

            if not self._can_store:
                continue

            r_meta = LMCacheMPRequestMetadata.GetStoreMetadata(
                request_tracker,
                lmcache_tokens_per_chunk,
                self._group_tokens_per_block,
                save_decode_cache=self._save_decode_cache,
            )
            if r_meta is not None:
                # In lazy_offload mode, add to pending queue instead of immediate store
                if self.lazy_offload:
                    self._lazy_offload_manager.add_store_candidate(r_meta)
                else:
                    metadata.add_request_metadata(r_meta)

    def _process_cached_requests(
        self,
        scheduler_output: SchedulerOutput,
        metadata: LMCacheMPConnectorMetadata,
    ) -> None:
        lmcache_tokens_per_chunk = self.scheduler_adapter.lmcache_tokens_per_chunk

        cached_reqs = scheduler_output.scheduled_cached_reqs
        # vLLM MRV2 readmits resumed requests as full new requests.
        resumed_ids = getattr(cached_reqs, "resumed_req_ids", ())
        for idx, request_id in enumerate(cached_reqs.req_ids):
            request_tracker = self._get_request_tracker(request_id)

            # Update block ids
            new_block_ids = cached_reqs.new_block_ids[idx] or ()
            if request_id not in resumed_ids:
                request_tracker.append_block_ids(
                    new_block_ids, self._mamba_relocation_window
                )

            # Use the incremental num_scheduled_tokens to
            # stay consistent with _process_new_requests.
            num_new_tokens = scheduler_output.num_scheduled_tokens[request_id]
            request_tracker.increase_num_scheduled_tokens(num_new_tokens)
            request_tracker.update_num_engine_computed_tokens(
                cached_reqs.num_computed_tokens[idx], num_new_tokens
            )

            if not self._can_store:
                continue

            r_meta = LMCacheMPRequestMetadata.GetStoreMetadata(
                request_tracker,
                lmcache_tokens_per_chunk,
                self._group_tokens_per_block,
                save_decode_cache=self._save_decode_cache,
            )

            if r_meta is not None:
                # In lazy_offload mode, add to pending queue instead of immediate store
                if self.lazy_offload:
                    self._lazy_offload_manager.add_store_candidate(r_meta)
                else:
                    metadata.add_request_metadata(r_meta)

    def _report_block_allocation_deltas(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Gather per-request block allocation deltas and report to LMCache.

        For new requests: all allocated_block_ids and token_ids are new.
        For cached requests: only newly appended block_ids and token_ids.
        The L0 allocation telemetry is flat today, so HMA reports engine group 0.
        """
        records: list[RequestAllocationRecord] = []

        # New requests: send all tokens covering all allocated blocks so
        # the L0 metrics subscriber can correctly map each block to its
        # actual token content (not just the newly-scheduled slice).
        for new_request in scheduler_output.scheduled_new_reqs:
            tracker = self.request_trackers.get(new_request.req_id)
            if tracker is None:
                continue
            primary_block_ids = tracker.allocated_block_ids.get(0, [])
            num_blocks = len(primary_block_ids)
            total_tokens = num_blocks * self._group_tokens_per_block[0]
            records.append(
                RequestAllocationRecord(
                    req_id=new_request.req_id,
                    new_block_ids=list(primary_block_ids),
                    new_token_ids=tracker.get_token_ids()[:total_tokens],
                )
            )

        # Cached requests: only the newly added blocks and their full
        # token content.  We send all tokens covered by the new blocks
        # (not just the tokens scheduled this step) so the L0 subscriber
        # can correctly identify block content.
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for idx, request_id in enumerate(cached_reqs.req_ids):
            # The L0 subscriber works on the primary (group 0) block-id list.
            new_group_block_ids = cached_reqs.new_block_ids[idx]
            new_block_ids = new_group_block_ids[0] if new_group_block_ids else []
            if not new_block_ids:
                continue
            tracker = self.request_trackers.get(request_id)
            if tracker is None:
                continue
            # The new blocks sit at the end of the request's block list.
            # Compute the token range they cover.
            total_blocks = len(tracker.allocated_block_ids.get(0, []))
            num_new_blocks = len(new_block_ids)
            tokens_per_block = self._group_tokens_per_block[0]
            start_token = (total_blocks - num_new_blocks) * tokens_per_block
            end_token = total_blocks * tokens_per_block
            new_token_ids = tracker.get_token_ids()[start_token:end_token]
            records.append(
                RequestAllocationRecord(
                    req_id=request_id,
                    new_block_ids=new_block_ids,
                    new_token_ids=new_token_ids,
                )
            )

        if records:
            self.scheduler_adapter.report_block_allocations(records)

    def _get_request_tracker(self, request_id: str) -> LMCacheMPRequestTracker:
        assert request_id in self.request_trackers, (
            f"Request tracker for request_id {request_id} not found. "
        )
        return self.request_trackers[request_id]

    def _get_or_create_request_tracker(
        self, request: "Request"
    ) -> LMCacheMPRequestTracker:
        request_id = request.request_id
        # Remove the old trackers that is created before the preemption
        if (
            request.status == RequestStatus.PREEMPTED
            and request_id in self.request_trackers
        ):
            tracker = self.request_trackers[request_id]

            # NOTE: since this function may be called multiple times
            # for a single request (because get_num_new_matched_tokens
            # may be called multiple times) for the same request, we
            # will only do the remove if the tracker is not in the "fresh"
            # state, i.e., PREFETCHING
            if tracker.state != LMCacheMPRequestState.PREFETCHING:
                self.request_trackers.pop(request_id)
                if self.lazy_offload:
                    # The recreated tracker restarts at token zero, so its
                    # manager discards overlapping buffered metadata.
                    self._lazy_offload_manager.on_request_reset(request_id)

        if request_id not in self.request_trackers:
            if self.lazy_offload:
                actions = self._lazy_offload_manager.on_request_arrived(request_id)
                for session_id in actions.sessions_to_end:
                    self.scheduler_adapter.end_session(session_id)
            new_tracker = LMCacheMPRequestTracker(request)
            self.request_trackers[request_id] = new_tracker
        return self.request_trackers[request_id]

    def _cleanup_request_tracker(self, request_id: str) -> None:
        """Clean up request tracker and associated lookup future for a request.
        This should be called when a request is finished to prevent memory leak.
        """
        # Clean up request tracker
        if self.request_trackers.pop(request_id, None):
            logger.debug(
                "[KVConnector] Cleaned up request_tracker for request %s",
                request_id,
            )


# At module load time, prefer the external LMCacheMPConnector shipped with the
# ``lmcache`` package. This avoids forcing users to set
# ``kv_connector_module_path`` when they only configure ``kv_connector``. If
# the external module is unavailable (e.g. older lmcache version that does
# not ship this submodule, or any import error), fall back to the builtin
# implementation defined above.
def _resolve_lmcache_mp_connector() -> type[KVConnectorBase_V1]:
    if os.environ.get("LMCACHE_USE_UPSTREAM_MP"):
        logger.info(
            "Force use builtin LMCacheMPConnectorUpstream in vLLM.",
        )
        return LMCacheMPConnectorUpstream

    try:
        from lmcache.integration.vllm.lmcache_mp_connector import (
            LMCacheMPConnector as _ExternalLMCacheMPConnector,
        )

        logger.info(
            "Using external LMCacheMPConnector from "
            "lmcache.integration.vllm.lmcache_mp_connector"
        )
        return _ExternalLMCacheMPConnector
    except ImportError as e:
        logger.info(
            "External LMCacheMPConnector is not available (%s), "
            "falling back to builtin implementation in vLLM.",
            e,
        )
        return LMCacheMPConnectorUpstream


LMCacheMPConnector = _resolve_lmcache_mp_connector()
