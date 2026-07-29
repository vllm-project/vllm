# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Patch Author: Anton Alexander
"""Metadata dataclasses and helpers for the NIXL connector."""

from dataclasses import dataclass, field
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds, EngineId
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

TransferHandle = int
ReqId = str

GET_META_MSG = b"get_meta_msg"
#
# NIXL Connector Version
#
# Increment this version whenever there is an incompatible change to:
#   - NixlAgentMetadata schema
#   - kv_transfer_params schema or semantics
#   - NIXL transfer protocol or wire format
#   - KV cache memory layout or block organization
#   - Any other change that breaks P/D interoperability
#
# Version History:
#   1: Initial version with compatibility checking
#   2: Add remote_request_id to kv_transfer_params
#   3: Add physical_blocks_per_logical_kv_block to NixlAgentMetadata
#   4: Add KV block lease renewal through heartbeats
#   5: Add pipeline-parallel producer metadata (pp_rank, pp_size,
#      start_layer, end_layer, registered_layer_indices,
#      registered_layer_names) and per-request pp_size. NIXL regions are
#      advertised per layer-name so HMA pool composition may differ across PP
#      producer shards and the decode consumer.
#   6: Add region_members so each advertised NIXL region declares ALL the
#      (global_layer_index, kv_group_index) members sharing it — including HMA
#      cross-group pooled members (e.g. an swa_cache pooled onto a later layer's
#      main-attn region) that are dedup'd out of registered_layer_names. Without
#      this, a pooled member belonging to a different kv group than the region's
#      representative is never transferred (its blocks are dropped), corrupting
#      KV under PP+HMA disaggregation.
#   7: Add fa_members_registered (P6' FIX-2, #46407). Under PP>1 + HMA the
#      producer dedups the FullAttention layer aliased onto a Mamba-represented
#      pooled tensor out of registered_layer_names; with this fix the producer
#      RE-ADVERTISES it as its own FA-typed region (same base addr, FA page
#      stride) and sets this flag True so the consumer knows real FA regions
#      now exist. False ⇒ legacy behavior (no extra FA regions). The flag is
#      read defensively via getattr so a v6 consumer ignores it; a v7 consumer
#      paired with a v6 producer sees False (both safe). NOTE: the producer
#      change ALSO grows registered_layer_names/region_group_ids (the FA regions
#      become their own representatives), so the legacy FA loop — not the v2
#      region_members recovery — emits the FA descriptors; the flag mainly
#      documents capability + keeps the recovery path correctly gated.
#   8: Add ssm_members_registered (P6' FIX-9, #46407 generalization). The HMA
#      general-case allocator pools ONE layer from EVERY kv_cache_group into a
#      shared tensor; for a multi-Mamba-group hybrid (e.g. Nemotron-3-Ultra: 4
#      Mamba groups + 1 FullAttention) each tensor holds N>1 Mamba layers + 1 FA.
#      FIX-2 (v7) recovered only the dedup'd FA member; the (N-1) dedup'd Mamba
#      members stayed dropped, so mamba_region_group_ids collapsed to [0]*N and
#      only the group-0 Mamba layers transferred -> the rest ran on stale SSM
#      state -> coherent-but-wrong decode. v8 producer ALSO re-advertises every
#      dedup'd Mamba member as its own region (same base, Mamba page stride) and
#      sets this flag. Consumer applies the SSM-only filter when
#      (fa_members_registered or ssm_members_registered). Read via getattr so a
#      v7 consumer ignores it; a v8 consumer paired with a v7 producer sees False
#      (both safe). Live-verified on Nemotron-3-Ultra 550B TP8/PP2: PP2 decode
#      byte-identical to the PP1 aggregated control (2026-06-26).
#
NIXL_CONNECTOR_VERSION: int = 8


@dataclass
class NixlAgentMetadata:
    engine_id: str
    agent_metadata: bytes
    kv_caches_base_addr: list[int]
    device_id: int
    num_blocks: int
    block_lens: list[int]
    kv_cache_layout: str
    block_size: int
    ssm_sizes: tuple[int, int]
    attn_backend_name: str
    physical_blocks_per_logical_kv_block: int
    pp_rank: int
    pp_size: int
    registered_layer_names: list[str]
    # Parallel to the advertised regions (registered_layer_names order): for
    # each NIXL region, the full list of layer names whose transfer caches
    # physically share that region. Captures HMA cross-group pooled members
    # that registered_layer_names (representatives only) omits, so the transfer
    # can cover every member's blocks in a shared region. Keyed by layer name
    # (not (layer_index, kv_group_index)) because distinct caches can merge into
    # one kv group via UniformTypeKVCacheSpecs (e.g. an MLA layer's main latent
    # and its indexer k_cache both land in the full-attention group): a
    # (layer_index, group) pair is then non-unique across regions, whereas the
    # layer name uniquely identifies the region and is stable across the PP
    # producer shard and the full-model consumer. Defaults to empty for backward
    # construction; populated in register_kv_caches.
    region_members: list[list[str]] = field(default_factory=list)
    # P6' FIX-2 (#46407): True iff the producer re-advertised at least one
    # dedup'd FullAttention member (aliased onto a Mamba-represented pooled
    # tensor under PP>1 + HMA) as its own FA-typed NIXL region. The consumer's
    # gated FA-recovery (worker._region_fa_recovery_for_members) reads this via
    # getattr(meta, "fa_members_registered", False) and only emits recovery
    # descriptors when True. Defaults False: PP1 (FA already a representative),
    # all-attention, and MLA producers never set it; a v6 producer omits the
    # field entirely and the getattr default keeps the consumer safe.
    fa_members_registered: bool = False
    # P6' FIX-9 (#46407 generalization): True iff the producer re-advertised at
    # least one dedup'd SSM/Mamba member (a non-representative Mamba layer from
    # group 1..N pooled behind the group-0 Mamba representative at the same base
    # under PP>1 + HMA) as its own region. Without this, only the group-0 Mamba
    # rep transfers and (num_mamba_groups-1)/num_mamba_groups of the Mamba layers
    # run on stale SSM state -> coherent-but-wrong decode. The consumer uses
    # (fa_members_registered or ssm_members_registered) to decide whether to apply
    # the SSM-only filter before the mamba descriptor builder. Defaults False:
    # PP1, single-Mamba-group, all-attention, and MLA producers never set it; a
    # v6/v7 producer omits the field and the getattr default keeps consumers safe.
    ssm_members_registered: bool = False


@dataclass
class NixlHandshakePayload(KVConnectorHandshakeMetadata):
    """
    Wrapper for NIXL handshake sent over the wire.

    Enables two-phase decoding for graceful compatibility checking:
    1. Decode NixlHandshakePayload to get compatibility_hash
    2. Compute local hash and compare
    3. Only if hashes match, decode agent_metadata_bytes

    This prevents decoder errors when NixlAgentMetadata schema is
    incompatible, allowing graceful failure with clear error message.
    """

    compatibility_hash: str
    agent_metadata_bytes: bytes  # NixlAgentMetadata encoded


def compute_nixl_compatibility_hash(
    vllm_config: VllmConfig, attn_backend_name: str, cross_layers_blocks: bool
) -> str:
    """
    Compute compatibility hash for NIXL KV transfer.

    Hash only the factors that affect whether two NIXL instances can
    successfully transfer KV cache data.

    Factors included:
    - vLLM version and NIXL connector version
    - Model architecture (name, dtype, KV heads, layers)
    - KV cache format (dtype, sliding window)
    - Attention backend

    Note: Factors like tensor_parallel_size, block_size, and kv_cache_layout
    are validated at runtime in _validate_remote_agent_handshake and are not
    included in this hash to support heterogeneous deployments.

    Note - the set of factors are likely to evolve significantly over
    time to be more or less permissive.

    Returns:
        SHA-256 hex digest
    """
    from vllm import __version__ as vllm_version
    from vllm.config.utils import hash_factors

    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config
    is_hma_enabled = not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager

    factors = {
        # Version compatibility
        "vllm_version": vllm_version,
        "nixl_connector_version": NIXL_CONNECTOR_VERSION,
        # Model architecture - affects KV cache shape
        "model": model_config.model,
        "dtype": str(model_config.dtype),
        "num_kv_heads": model_config.get_total_num_kv_heads(),
        "head_size": model_config.get_head_size(),
        "num_hidden_layers": model_config.get_total_num_hidden_layers(),
        # Attention backend and KV cache dtype affect memory layout
        "attn_backend_name": attn_backend_name,
        "cache_dtype": str(cache_config.cache_dtype),
        "cross_layers_blocks": cross_layers_blocks,
        "is_hma_enabled": is_hma_enabled,
    }

    compat_hash = hash_factors(factors)
    logger.debug(
        "NIXL compatibility hash: %s (model=%s, dtype=%s, num_kv_heads=%d, "
        "cache_dtype=%s, attn_backend=%s)",
        compat_hash,
        factors["model"],
        factors["dtype"],
        factors["num_kv_heads"],
        factors["cache_dtype"],
        attn_backend_name,
    )
    return compat_hash


@dataclass
class HeartbeatInfo:
    """Heartbeat data for a single remote engine, sent from D worker to P."""

    req_ids: set[ReqId]
    host: str
    port: int
    tp_size: int
    pp_size: int


@dataclass
class RemoteMeta:
    block_ids: BlockIds
    host: str
    port: int
    engine_id: str
    request_id: str


@dataclass
class ReqMeta:
    local_block_ids: BlockIds
    # To be used when logical block size does not match the kernel block size
    local_physical_block_ids: BlockIds
    tp_size: int
    pp_size: int = 1
    remote: RemoteMeta | None = None


class NixlConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.reqs_to_recv: dict[ReqId, ReqMeta] = {}
        self.reqs_to_save: dict[ReqId, ReqMeta] = {}
        self.reqs_to_send: dict[ReqId, float] = {}
        self.reqs_in_batch: set[ReqId] = set()
        self.reqs_not_processed: set[ReqId] = set()
        # Heartbeat data grouped by remote engine, sent by D worker to P.
        self.heartbeat_by_engine: dict[EngineId, HeartbeatInfo] = {}

    def _add_new_req(
        self,
        local_block_ids: BlockIds,
        kv_transfer_params: dict[str, Any],
    ) -> ReqMeta:
        return ReqMeta(
            local_block_ids=local_block_ids,
            local_physical_block_ids=local_block_ids,
            # P workers don't need to receive tp_size from proxy here.
            tp_size=kv_transfer_params.get("tp_size", 1),
            pp_size=kv_transfer_params.get("pp_size", 1),
        )

    def add_new_req_to_save(
        self,
        request_id: ReqId,
        local_block_ids: BlockIds,
        kv_transfer_params: dict[str, Any],
    ):
        self.reqs_to_save[request_id] = self._add_new_req(
            local_block_ids, kv_transfer_params
        )

    def add_new_req_to_recv(
        self,
        request_id: ReqId,
        local_block_ids: BlockIds,
        kv_transfer_params: dict[str, Any],
    ):
        req = self._add_new_req(local_block_ids, kv_transfer_params)
        req.remote = RemoteMeta(
            block_ids=kv_transfer_params["remote_block_ids"],
            engine_id=kv_transfer_params["remote_engine_id"],
            request_id=kv_transfer_params["remote_request_id"],
            host=kv_transfer_params["remote_host"],
            port=kv_transfer_params["remote_port"],
        )
        self.reqs_to_recv[request_id] = req

