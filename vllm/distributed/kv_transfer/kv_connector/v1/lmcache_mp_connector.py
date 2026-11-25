# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Builtin LMCache multiprocess connector.

Synced from `lmcache/integration/vllm/lmcache_mp_connector.py` at LMCache
v0.5.8. Apart from changes for vLLM's linters, it differs only in passing the
KV cache groups, not the `KVCacheConfig`, to
`VllmConfig.adjust_dcp_kv_cache_interleave_size`, and in not reading
`CachedRequestData.resumed_req_ids`, which this vLLM no longer has.

`LMCacheMPConnector` at the bottom of this module prefers the connector the
installed `lmcache` package ships and falls back to this one.
"""

import math
import os
import sys
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_events import (
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorTransferResults,
    )
except ImportError:
    # Older engines consume get_finished() and block-level errors instead.
    KVConnectorTransferResults = None  # type: ignore[assignment,misc]

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA
except ImportError:
    # Older vLLM builds do not expose HMA. They cannot route per-group
    # request-finished calls, but keeping the class importable preserves
    # legacy single-group behavior.
    class SupportsHMA:  # type: ignore[no-redef]
        pass


try:
    from vllm.config.kv_transfer import KVCacheAccess
except ImportError:
    # vLLM without the kv_cache_access hook (vllm-project/vllm#66605).
    KVCacheAccess = None  # type: ignore[assignment,misc]


import torch
import zmq
from lmcache.banner import print_banner_once
from lmcache.integration.vllm.experimental import dispatch
from lmcache.integration.vllm.kv_cache_group_edits import (
    apply_kv_cache_group_edits,
    validate_kv_cache_groups,
)
from lmcache.integration.vllm.kv_cache_groups import (
    create_engine_group_infos_from_vllm,
    group_prefix_replay_tokens,
    group_specs,
    is_align_mode_mamba_group,
    is_per_request_state_group,
    is_scratch_group,
    tokens_per_group_block,
)
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
    vllm_layout_hints,
)
from lmcache.utils import CacheRemoveEvent
from lmcache.utils import init_logger as lmcache_init_logger
from lmcache.v1.client.worker_transfer import LMCacheDrivenTransferContext

from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.kv_cache_utils import BlockHash, maybe_convert_block_hash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

try:
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        ExtraConfigDefault,
        LMCacheMPSchedulerAdapter,
        LMCacheMPWorkerAdapter,
        ParallelStrategy,
        resolve_extra_config_value,
    )

    try:
        from lmcache.v1.common.custom_types import (  # type: ignore[attr-defined]
            RequestAllocationRecord,
        )
    except ImportError:
        from lmcache.v1.common.custom_types import (
            BlockAllocationRecord as RequestAllocationRecord,
        )
except ImportError:
    from lmcache.v1.common.custom_types import (
        BlockAllocationRecord as RequestAllocationRecord,
    )

    from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration import (  # type: ignore[assignment,no-redef]
        LMCacheMPSchedulerAdapter,
        LMCacheMPWorkerAdapter,
        ParallelStrategy,
    )

if TYPE_CHECKING:
    from vllm.config import KVTransferConfig
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
# Request-config key whose value becomes ``ObjectKeyIdentity.tags["revision"]``.
# See ``get_request_tags`` in ``lmcache.v1.distributed.api``.
_REVISION_TAG_KEY = "lmcache.tag.revision"
_MAX_LCM_EXPANSION_FACTOR = 4
_USE_VMM_API_KEY = "lmcache.mp.use_vmm_api"


def _worker_extra_config(vllm_config: "VllmConfig") -> dict[str, Any]:
    """Resolve the worker's VMM IPC mode from the active vLLM config.

    An explicit `lmcache.mp.use_vmm_api` value remains authoritative. When
    it is absent, the connector follows vLLM's resolved allocation mode:
    extensible KV cache and the cuMem allocator both produce VMM-backed KV
    memory.

    Args:
        vllm_config: The active, resolved vLLM engine configuration.

    Returns:
        Connector extra config with `lmcache.mp.use_vmm_api` resolved for
        the worker adapter. The user-owned config mapping is not modified.

    """
    assert vllm_config.kv_transfer_config is not None
    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
    if _USE_VMM_API_KEY in extra_config:
        return extra_config

    model_config = getattr(vllm_config, "model_config", None)
    use_vmm_api = bool(
        getattr(vllm_config.cache_config, "enable_extensible_kv_cache", False)
        or (
            model_config is not None
            and getattr(model_config, "enable_cumem_allocator", False)
        )
    )
    return {**extra_config, _USE_VMM_API_KEY: use_vmm_api}


def _convert_kv_event_hash(block_hash: bytes | int | None) -> bytes | int | None:
    """Convert LMCache event hashes to vLLM's configured external format."""
    if isinstance(block_hash, bytes):
        return maybe_convert_block_hash(BlockHash(block_hash))
    return block_hash


# Helper functions
def _has_preemption_reqs(scheduler_output: SchedulerOutput) -> bool:
    """Return whether any request was preempted this step.

    Args:
        scheduler_output: The vLLM scheduler output for this step.

    Returns:
        True if `scheduler_output.preempted_req_ids` is non-empty.

    """
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
            adjust(kv_cache_config.kv_cache_groups)
            interleave = getattr(
                parallel_config, "cp_kv_cache_interleave_size", original
            )
        finally:
            parallel_config.cp_kv_cache_interleave_size = original
    if dcp_size <= 1 or interleave == 1:
        return model_name
    return f"{model_name}{_DCP_LAYOUT_NAMESPACE}d{dcp_size}-interleave{interleave}"


def _revision_cache_token(vllm_config: VllmConfig) -> str | None:
    """Return the model revision that must be part of the cache identity.

    vLLM resolves `--revision` to a commit hash for Hub models and exposes
    the hash as `ResolvedRevision.resolved`. Two engines that pin different
    commits of one model ID produce different KV bytes for the same tokens, so
    the hash is what the identity has to track: the requested ref alone would
    not change when the branch moves. A revision vLLM could not resolve (local
    path, ModelScope, offline lookup) is used verbatim.

    An engine that passes no `--revision` still gets the default branch
    resolved, to `ResolvedRevision(initial=None, ...)`. Tracking that hash
    would rename every existing cache namespace on upgrade without any
    guarantee about the served weights, so only an explicitly pinned revision
    takes part in the identity.

    Args:
        vllm_config: The active vLLM configuration.

    Returns:
        The commit hash, the requested revision, or `None` when the engine
        does not pin a revision.

    """
    model_config = getattr(vllm_config, "model_config", None)
    revision = (
        getattr(model_config, "revision", None) if model_config is not None else None
    )
    if not revision:
        return None
    requested = getattr(revision, "initial", revision)
    if requested is None:
        return None
    return str(getattr(revision, "resolved", None) or requested)


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
    `max_num_batched_tokens` may exceed `2 * block_size` (with object-group
    separation). Only the lower bound remains: a step must
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

    # A rank's token shard is its position among its server's TP ranks, so
    # each server's TP ranks must form whole shard sets.
    kv_tp_size = pc.tensor_parallel_size // n_servers
    if kv_tp_size % dcp_size != 0:
        raise ValueError(
            "Each LMCache server's tensor-parallel ranks must hold a complete "
            f"set of shards: {n_servers} server(s) leave {kv_tp_size} rank(s) "
            f"each for decode_context_parallel_size={dcp_size}. Use fewer "
            "servers or a smaller DCP size."
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
    - lmcache.mp.host: the host of the LMCache server, e.g. "127.0.0.1" or
      "tcp://127.0.0.1" (default "tcp://localhost"). The final URL is
      "{host}:{port}". If no transport scheme is provided, "tcp://" is
      automatically prepended.
    - lmcache.mp.port: the port of the LMCache server.

    - lmcache.mp.mq_timeout: timeout (seconds) for message queue requests.
    - lmcache.mp.heartbeat_interval: interval (seconds) between server
      heartbeat pings.
    - lmcache.mp.eager_prefetch: submit the LMCache lookup when a request
      enters vLLM's waiting queue. Disabled by default.
    - lmcache.mp.save_decode_cache: also store chunks containing generated
      tokens. Defaults to False; only complete prompt chunks are stored.
    - lmcache.mp.save_unfull_chunk: also store the prompt's unfull chunk, with
      its boundary (token count) on vLLM's prefix-cache grid below the
      prompt's last token, and let a lookup hit stop mid-chunk. Defaults to
      False. A request whose `kv_transfer_params` set
      `lmcache.temporary_unfull_chunk` stores it read-once. Not yet
      supported for models with per-request ring state
      (`CircularBufferSpec`). See
      `docs/design/v1/multiprocess/unfull_chunk.md`.
    """

    # Tail block slots vLLM may relocate for one request; 0 means vLLM only
    # appends. The scheduler role sets it from the vLLM config.
    _mamba_relocation_window: int = 0

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
            KVConnectorTransferResults is not None
            and kv_cache_config is not None
            and len(kv_cache_config.kv_cache_groups) > 1
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
        # Identity tag sent with every request; empty when the engine pinned no
        # revision, so unpinned engines keep their existing cache namespace.
        self._revision_tag = _revision_cache_token(vllm_config) or ""

        assert vllm_config.kv_transfer_config is not None
        self._can_store = vllm_config.kv_transfer_config.is_kv_producer
        self._save_decode_cache = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache.mp.save_decode_cache", False
        )
        if not isinstance(self._save_decode_cache, bool):
            raise ValueError("lmcache.mp.save_decode_cache must be a boolean")
        self._save_unfull_chunk = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache.mp.save_unfull_chunk", False
        )
        if not isinstance(self._save_unfull_chunk, bool):
            raise ValueError("lmcache.mp.save_unfull_chunk must be a boolean")
        spec_config = getattr(vllm_config, "speculative_config", None)
        if (
            self._save_unfull_chunk
            and spec_config is not None
            and (spec_config.num_speculative_tokens or 0) > 0
        ):
            # Draft rows sit past the prompt end in relocated Mamba blocks,
            # so the state at the prompt end is not isolated.
            raise ValueError(
                "lmcache.mp.save_unfull_chunk does not support speculative decoding"
            )
        if (
            self._save_unfull_chunk
            and kv_cache_config is not None
            and any(
                is_per_request_state_group(group)
                for group in kv_cache_config.kv_cache_groups
            )
        ):
            # Resuming mid-chunk needs the ring's open compression group,
            # which LMCache does not store yet: the hit would load garbage.
            raise ValueError(
                "lmcache.mp.save_unfull_chunk does not support models with "
                "per-request ring state (CircularBufferSpec) yet"
            )

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
            "scheduler_block_size=%d, cache_model_name=%s, revision=%s",
            group_tokens_per_block,
            scheduler_block_size,
            cache_model_name,
            self._revision_tag or "-",
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

        self._dcp_size = parallel_strategy.dcp_size
        self._kv_tp_size = parallel_strategy.kv_tp_size

        # Lazy offload configuration: when enabled, store operations are
        # buffered and drained by the configured policy.
        self.lazy_offload = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache.mp.lazy_offload", False
        )
        if self.lazy_offload and self._save_unfull_chunk:
            raise ValueError(
                "lmcache.mp.save_unfull_chunk does not support lmcache.mp.lazy_offload"
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
                enable_kv_events=self._enable_kv_events,
            )
            self.request_trackers: dict[str, LMCacheMPRequestTracker] = {}
            # Requests whose request_finished returned True, sent to the
            # workers with the next step's metadata.
            self._delay_free_req_ids: set[str] = set()

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
            # Boundary-state blocks pinned for each request's unfull-chunk
            # stores, released once the request's stores finish.
            self._pinned_boundary_state_blocks: dict[str, list[int]] = {}
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
                extra_config=_worker_extra_config(vllm_config),
                required_chunk_alignment=scheduler_block_size,
            )
            # Requests whose blocks vLLM keeps until they are in finished_sending.
            self._delay_free_req_ids = set()
            if self.transfer_intermediate_tensors:
                from lmcache.integration.vllm.experimental import (
                    FeatureContext,
                    init_dispatcher,
                )
                from lmcache.v1.common.custom_types import TRANSFER_QUERY

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
        # Align-mode Mamba groups: their state at an unfull chunk's boundary
        # comes from vLLM's boundary-state offloads, not the block table.
        self._boundary_state_groups = (
            [
                engine_group_idx
                for engine_group_idx, group in enumerate(
                    kv_cache_config.kv_cache_groups
                )
                if is_align_mode_mamba_group(group)
            ]
            if kv_cache_config is not None
            else []
        )
        # Grid of unfull-chunk boundaries: boundary-state offloads lie on vLLM's
        # prefix-cache hash grid; otherwise prefix-cache hits land on the
        # scheduler block.
        self._unfull_chunk_grid = self._hit_alignment_tokens
        if (
            self._save_unfull_chunk
            and self._boundary_state_groups
            and kv_cache_config is not None
        ):
            if "kv_connector_block_state" not in SchedulerOutput.__dataclass_fields__:
                raise ValueError(
                    "lmcache.mp.save_unfull_chunk with align-mode Mamba needs a "
                    "vLLM that hands boundary states to connectors"
                )
            from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

            _, self._unfull_chunk_grid = resolve_kv_cache_block_sizes(
                kv_cache_config, vllm_config
            )
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
            kv_tp_size=self._kv_tp_size,
        )
        self.worker_adapter.register_kv_caches(
            kv_caches,
            engine_group_infos=engine_group_infos,
            layout_hints=layout_hints,
        )
        if self._save_unfull_chunk and not isinstance(
            self.worker_adapter.transfer_ctx, LMCacheDrivenTransferContext
        ):
            # The engine-driven server paths copy whole chunks only.
            raise ValueError(
                "lmcache.mp.save_unfull_chunk requires the lmcache_driven transfer mode"
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

    def unregister_kv_caches(self) -> None:
        """Drop the LMCache server's registration of the KV caches before
        vLLM releases their memory (sleep, KV cache release, weight update).
        Idempotent.
        """
        self.worker_adapter.unregister_kv_caches()

    def restore_kv_caches(self) -> None:
        """Register the KV caches with the LMCache server again after vLLM
        re-backs their memory. Idempotent.

        Raises:
            ConnectionError: if the server does not respond in time. Calling
                this again retries.

        """
        self.worker_adapter.restore_kv_caches()

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
        self._delay_free_req_ids.update(metadata.delay_free_req_ids)

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
            finished_sending, finished_receiving = self.worker_adapter.get_finished(
                finished_req_ids if self._can_store else set()
            )
            # Report only requests whose request_finished returned True: vLLM
            # asserts on a finished_sending report for a request it freed.
            finished_sending = set(finished_sending or ()) & self._delay_free_req_ids
            self._delay_free_req_ids -= finished_sending
            val = (finished_sending, finished_receiving)
        # logger.error("Finished req ids: %s, %s", val[0], val[1])
        return val

    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> "KVConnectorTransferResults":
        """Poll completed transfers and report multi-group failures by request.

        Args:
            finished_req_ids: Requests whose generation has finished in vLLM.

        Returns:
            Completed sends and receives, with failed multi-group receives also
            in `failed_recving`. vLLM aggregates all workers' completions
            before recovering the request. Single-group errors remain available
            through `get_block_ids_with_load_errors`.

        """
        assert KVConnectorTransferResults is not None
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
        """Build this step's lazy-offload store results.

        Returns:
            Completion counts and failed request IDs, or None when lazy
            offload is disabled or there are no new results.

        """
        if not self.lazy_offload:
            return None
        completed_store_requests = self.worker_adapter.get_completed_store_requests()
        failed_store_requests = self.worker_adapter.get_failed_store_requests()
        if completed_store_requests or failed_store_requests:
            return LMCacheMPWorkerMetadata(
                completed_store_requests=completed_store_requests or {},
                failed_store_requests=failed_store_requests or set(),
            )
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
            request_configs=self._lookup_request_configs(tracker, num_computed_tokens),
            reserve_last_token=self._reserve_last_token_for_lookup,
            include_unfull_chunk=self._save_unfull_chunk,
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
                request_configs=self._lookup_request_configs(
                    tracker, num_computed_tokens
                ),
                reserve_last_token=self._reserve_last_token_for_lookup,
                include_unfull_chunk=self._save_unfull_chunk,
            ):
                tracker.lookup_started_at = time.monotonic()
            return None, True

        if ret == 0:
            return 0, False

        # A hit ending with an unfull chunk loads it even when vLLM's prefix
        # cache already holds part of it: the retrieve skips those tokens
        # (``skip_first_n_tokens``), as it does for a full chunk.
        full_chunk_tokens = ret - ret % self.scheduler_adapter.lmcache_tokens_per_chunk
        tracker.num_stored_tokens = full_chunk_tokens
        tracker.num_lmcache_hit_tokens = ret
        if ret != full_chunk_tokens:
            tracker.unfull_chunk_boundaries.add(ret)

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
            include_unfull_chunk=self._save_unfull_chunk,
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
        metadata.delay_free_req_ids = self._delay_free_req_ids
        self._delay_free_req_ids = set()

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
        # A request's stores are done once its workers report it sent.
        for request_id in connector_output.finished_sending or ():
            self._release_boundary_state_blocks(request_id)

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

        # Keep the blocks only while a store may still read them. A preempted
        # request starts over with a new tracker on its next lookup.
        tracker = self.request_trackers.get(request.request_id)
        delay_free = tracker is not None and tracker.store_scheduled

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

        # A request preempted after an unfull-chunk store may store nothing
        # after it resumes. Its boundary-state blocks stay pinned until it is
        # reported in finished_sending, so it still delays.
        delay_free = (
            delay_free or request.request_id in self._pinned_boundary_state_blocks
        )

        # Notify LMCache to end the session for this request
        self.scheduler_adapter.end_session(request.request_id)
        if delay_free:
            self._delay_free_req_ids.add(request.request_id)
        return delay_free, (return_params or None)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """HMA request-finished entry point; cleanup is request-id based."""
        return self.request_finished(request, block_ids[0] if block_ids else [])

    def take_events(self) -> Iterable["KVCacheEvent"]:
        """Drain scheduler CPU events in order, independently of worker output.

        Returns:
            vLLM events received since the previous drain, or an empty list
            when events are disabled or this connector belongs to a worker.

        """
        if self.role != KVConnectorRole.SCHEDULER or not self._enable_kv_events:
            return []
        events = self.scheduler_adapter.get_kv_events()
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
        return blocks

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

    @classmethod
    def kv_cache_access(  # type: ignore[override]
        cls, kv_transfer_config: "KVTransferConfig"
    ) -> "KVCacheAccess | None":
        """Report how the LMCache server reaches the device KV cache.

        Args:
            kv_transfer_config: This connector's transfer config. An explicit
                `lmcache.mp.use_vmm_api` value overrides automatic VMM
                detection; `lmcache.mp.isolated_ipc` selects the export
                mechanism used by VMM allocations.

        Returns:
            - `LEGACY_IPC` when `use_vmm_api` is explicitly false.
            - `EXPORTS_FABRIC` in automatic or explicit VMM mode with `isolated_ipc`.
            - `EXPORTS_FD` in automatic or explicit VMM mode otherwise.
            - None when vLLM has no `KVCacheAccess`.

        """
        if KVCacheAccess is None:
            return None
        extra_config = kv_transfer_config.kv_connector_extra_config
        if _USE_VMM_API_KEY in extra_config and not resolve_extra_config_value(
            extra_config, ExtraConfigDefault.use_vmm_api
        ):
            # Legacy CUDA IPC handles cannot express driver-mapped memory, so
            # the KV cache must stay a plain allocation.
            return KVCacheAccess.LEGACY_IPC
        if resolve_extra_config_value(extra_config, ExtraConfigDefault.isolated_ipc):
            # Isolated IPC refuses POSIX fds, and an allocation that can
            # export one is always shared through it, so ask for FABRIC alone.
            return KVCacheAccess.EXPORTS_FABRIC
        return KVCacheAccess.EXPORTS_FD

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
                    request_tracker.store_scheduled = True
            if self._save_unfull_chunk:
                self._add_unfull_chunk_stores(
                    scheduler_output, request_tracker, metadata
                )

    def _process_cached_requests(
        self,
        scheduler_output: SchedulerOutput,
        metadata: LMCacheMPConnectorMetadata,
    ) -> None:
        lmcache_tokens_per_chunk = self.scheduler_adapter.lmcache_tokens_per_chunk

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for idx, request_id in enumerate(cached_reqs.req_ids):
            request_tracker = self._get_request_tracker(request_id)

            # Update block ids
            new_block_ids = cached_reqs.new_block_ids[idx] or ()
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
                    request_tracker.store_scheduled = True
            if self._save_unfull_chunk:
                self._add_unfull_chunk_stores(
                    scheduler_output, request_tracker, metadata
                )

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
                    new_token_ids=tracker.get_token_ids(0, total_tokens),
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
            new_token_ids = tracker.get_token_ids(start_token, end_token)
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
            self._apply_revision_tag(new_tracker)
            self.request_trackers[request_id] = new_tracker
        return self.request_trackers[request_id]

    def _apply_revision_tag(self, tracker: LMCacheMPRequestTracker) -> None:
        """Carry the engine revision as an `lmcache.tag.revision` tag.

        The tag travels in the request configuration, so the server derives it
        with `get_request_tags` next to every other identity tag and no extra
        wire field is needed. Engines without a pinned revision add nothing.

        A request that already sets the key to another value would silently
        write objects under a revision that is not the one serving it, so the
        engine revision wins and the conflict is logged.
        """
        revision_tag = getattr(self, "_revision_tag", "")
        if not revision_tag:
            return
        configs = dict(getattr(tracker, "request_configs", None) or {})
        configured = configs.get(_REVISION_TAG_KEY)
        if configured is not None and configured != revision_tag:
            logger.warning(
                "Request %s sets %s=%r, overriding it with the engine revision %r",
                tracker.request_id,
                _REVISION_TAG_KEY,
                configured,
                revision_tag,
            )
        configs[_REVISION_TAG_KEY] = revision_tag
        tracker.request_configs = configs

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

    def _lookup_request_configs(
        self, tracker: LMCacheMPRequestTracker, num_computed_tokens: int
    ) -> dict[str, Any]:
        """Return a lookup's request configs: the request's, plus hints.

        Args:
            tracker: The request's tracker.
            num_computed_tokens: vLLM's prefix-cache hit, for the server's KV
                load policy.

        Returns:
            The request configs to send with the lookup.

        """
        request_configs: dict[str, Any] = {
            **(tracker.request_configs or {}),
            "lmcache.hint.engine_computed_tokens": num_computed_tokens,
        }
        if self._save_unfull_chunk:
            # Unfull-chunk boundaries lie on this grid, so the server probes
            # only it.
            request_configs["lmcache.hint.unfull_chunk_grid"] = self._unfull_chunk_grid
        return request_configs

    def _add_unfull_chunk_stores(
        self,
        scheduler_output: SchedulerOutput,
        tracker: LMCacheMPRequestTracker,
        metadata: LMCacheMPConnectorMetadata,
    ) -> None:
        """Queue the stores of the request's newly storable unfull chunks.

        For align-mode Mamba groups, a grid unfull chunk's state block is a
        boundary-state block vLLM offered this step
        (`boundary_state_offloads`). It is not part of the request's block
        table, so it is pinned until the request's stores finish.

        Args:
            scheduler_output: This step's scheduler output.
            tracker: The request's tracker.
            metadata: The step's connector metadata to add the stores to.

        """
        boundary_state_blocks: dict[int, dict[int, int]] = {}
        pool = self._gpu_block_pool
        if self._boundary_state_groups:
            block_state = scheduler_output.kv_connector_block_state
            # Without the pool a boundary-state block cannot be pinned.
            if block_state is None or pool is None:
                return
            for group, block_id, boundary in block_state.boundary_state_offloads.get(
                tracker.request_id, ()
            ):
                boundary_state_blocks.setdefault(boundary, {})[group] = block_id
        for (
            r_meta,
            boundary_state_block_ids,
        ) in LMCacheMPRequestMetadata.GetUnfullChunkStoreMetadata(
            tracker,
            self.scheduler_adapter.lmcache_tokens_per_chunk,
            self._group_tokens_per_block,
            self._unfull_chunk_grid,
            boundary_state_blocks,
            self._boundary_state_groups,
        ):
            if pool is not None and boundary_state_block_ids:
                pool.touch(
                    [pool.blocks[block_id] for block_id in boundary_state_block_ids]
                )
                self._pinned_boundary_state_blocks.setdefault(
                    tracker.request_id, []
                ).extend(boundary_state_block_ids)
            metadata.add_request_metadata(r_meta)
            tracker.store_scheduled = True

    def _release_boundary_state_blocks(self, request_id: str) -> None:
        """Unpin the boundary-state blocks a request's unfull-chunk stores read.

        Args:
            request_id: The request whose stores finished.

        """
        block_ids = self._pinned_boundary_state_blocks.pop(request_id, [])
        if block_ids and self._gpu_block_pool is not None:
            pool = self._gpu_block_pool
            pool.free_blocks([pool.blocks[block_id] for block_id in block_ids])


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
