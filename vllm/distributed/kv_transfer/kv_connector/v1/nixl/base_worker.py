# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base worker-side logic for the NIXL connector."""

import contextlib
import itertools
import logging
import math
import os
import queue
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import msgspec
import numpy as np
import regex as re
import torch
import zmq

from vllm.config import get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    EngineId,
    EngineTransferInfo,
    TransferTopology,
    byte_view,
    check_compressed_kernel_split,
    coalesce_byte_ranges,
    contiguous_byte_ranges,
    deserialize_layer_meta_tensors,
    expand_block_ids,
    get_current_attn_backends,
    get_current_attn_backends_and_specs,
    kv_postprocess_blksize_and_layout_on_receive,
    kv_postprocess_blksize_on_receive,
    pair_shards,
    serialize_layer_meta_tensors,
    transfer_block_positions,
    view_block_size_ratio,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    CopyBlocksOp,
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    HB_NOTIF_PREFIX,
    NixlAgentMetadata,
    NixlConnectorMetadata,
    NixlHandshakePayload,
    ReqId,
    ReqMeta,
    TransferHandle,
    compute_nixl_compatibility_hash,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import (
    NixlKVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    _NIXL_SUPPORTED_DEVICE,
    _is_ssm_spec,
    get_dcp_replicated_group_ids,
    get_representative_spec_type,
    kv_lease_extension,
    zmq_ctx,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.validation import (
    check_remote_pp_supported,
    validate_csa_linear_tp_layout,
    validate_push_block_sizes,
    validate_remote_agent_handshake,
    validate_remote_parallel_config,
)
from vllm.distributed.nixl_utils import (
    NixlWrapper,
    check_nixl_version,
    nixl_agent_config,
    nixl_thread_sync_t,
    nixlRemoteDisconnectError,
)
from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.platforms import current_platform
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.network_utils import make_zmq_path
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.core.single_type_kv_cache_manager import (
    CircularBufferManager,
    FullAttentionManager,
)
from vllm.v1.hisparse.runtime import get_hisparse_host_leader_rank
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheLayout,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SparseCacheRole,
    compute_layer_kv_cache_shape_bytes,
)
from vllm.v1.worker.utils import select_common_block_size

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


# Re-handshake delay bounds for a pull peer whose local metadata keeps getting
# invalidated, eg due to a persistent RDMA fault.
_INVALID_PEER_MIN_BACKOFF_S = 1.0
_INVALID_PEER_MAX_BACKOFF_S = 60.0
_MAX_HYBRID_BLOCK_SPLIT = 128


def _region_sort_key(layer_name: str) -> tuple[tuple[int, int | str], ...]:
    """Sort transfer regions in model-layer order, then by cache name."""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", layer_name)
    )


# (remote region, local region, local cache group) of a pair of regions holding
# the same layers. A region shared by several cache groups pairs once per group.
RegionPair = tuple[int, int, int]


# Max slack (s) kept before a remote lease deadline when heartbeating or reading:
# absorbs clock-offset estimation error and notification/read latency.
_KV_BLOCKS_EXPIRY_SAFETY_MARGIN = 5.0


@dataclass(slots=True)
class _LocalDescs:
    """A prepared local desc list and the first desc id of each byte range."""

    handle: int
    starts: dict[tuple[int, int, int, int], int]
    # Mixed device/host regions: the host part, and per desc whether it is
    # host memory and its position in its own list.
    dram_handle: int | None = None
    desc_is_dram: np.ndarray | None = None
    desc_pos: np.ndarray | None = None


@dataclass(slots=True)
class _RemoteStage:
    """The KV regions of one PP stage of a remote engine, paired with this
    worker's regions, and what moves between them. An engine that runs PP=1
    is one stage."""

    pp_rank: int
    # The regions of both sides holding the same layers.
    pairs: list[RegionPair]
    # The local layers that transfers with the stage cover.
    layers: list[str]
    # Number of blocks of each of the stage's regions.
    region_num_blocks: list[int]
    # Remote TP rank -> prepped descriptor list of that rank's paired regions.
    xfer_side_handles: dict[int, int] = field(default_factory=dict)
    # What each pair of regions moves with each remote TP rank:
    #   byte_ranges[k][remote_rank] = [(local_offset, remote_offset, num_bytes)]
    # for the k-th pair of `pairs`, in bytes from the start of a transfer
    # block on each side. A rank that moves none of the pair is left out.
    byte_ranges: list[dict[int, list[tuple[int, int, int]]]] = field(
        default_factory=list
    )
    # Per remote TP rank and cache group, the local and remote first desc id of each
    # range moved at the group's blocks: e.g. FA group 0 over 2 layers of 1000 blocks
    # + SSM group 1 over 1 layer's conv and ssm views of 100 state blocks:
    #   descs: | L0 0..999 | L1 0..999 | conv 0..99 | ssm 0..99 |
    #   desc_starts[rank] = {0: [0, 1000], 1: [2000, 2100]}
    # so a request's block_ids [[7, 8], [4]] read descs
    #   group 0: [0, 1000][:, None] + [7, 8] -> [7, 8, 1007, 1008]
    #   group 1: [2000, 2100][:, None] + [4] -> [2004, 2104]
    # in one add per group, with nothing computed per request.
    desc_starts: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = field(
        default_factory=dict
    )
    # Local desc list covering this stage's byte ranges (shared via
    # `_local_desc_lists` when another stage moves the same ranges).
    local_descs: _LocalDescs | None = None


class NixlBaseConnectorWorker:
    """Base implementation of Worker side methods shared by pull and push."""

    # Transfer mode included in the NIXL compatibility hash so that a push
    # (WRITE) connector and a pull (READ) connector never handshake together.
    # Overridden by NixlPushConnectorWorker.
    _TRANSFER_MODE: str = "pull"

    # Default UCX `num_threads` (kv_connector_extra_config can override it).
    _DEFAULT_NUM_THREADS: int = 4

    # Push additionally supports MLA HMA and packed PP; attention-only HMA
    # pull uses the shared layer pairing without these paths.
    _supports_pp_hma = False

    # Declared for methods defined before __init__ (mypy has-type).
    transfer_tp_size: int
    transfer_tp_rank: int
    _group_spec_types: tuple[type[KVCacheSpec], ...]
    block_len_per_layer: list[int]
    _physical_blocks_per_group: list[int]
    model_config: Any
    attn_backends: list[Any]
    block_size: int
    num_blocks: int
    pp_size: int
    _has_mamba: bool
    vllm_config: "VllmConfig"
    kv_cache_config: "KVCacheConfig"
    region_members: list[list[str]]
    _layer_specs: dict[str, KVCacheSpec]
    _remote_block_splits: dict[EngineId, int]
    _physical_blocks_per_logical_kv_block: int
    _logical_num_blocks: int

    @property
    def transfer_kv_cache_layout(self) -> str:
        """The layout of the blocks that transfer: the host buffer's, if any."""
        if self.use_host_buffer:
            return self.host_buffer_kv_cache_layout
        return self.kv_cache_layout

    def _transfer_block(
        self,
        spec: KVCacheSpec,
        group: int,
        block_len: int,
        layout: Sequence[int],
        block_size_ratio: int = 1,
    ) -> torch.Tensor:
        """One `[1, H, N, C]` block of bytes of a layer on the meta device, or
        `block_len` opaque bytes. A block in the layout of a remote with
        blocks `block_size_ratio` times smaller is one of its blocks."""
        if not spec.has_layer_views:
            shape: tuple[int, ...] = (1, 1, 1, block_len)
        else:
            tokens = spec.block_size // (
                self._physical_blocks_per_group[group] * block_size_ratio
            )
            shape = (1, *compute_layer_kv_cache_shape_bytes(spec, 1, tokens)[1:])
        block = torch.empty_permuted(shape, layout, dtype=torch.int8, device="meta")
        return block.as_strided(shape, (block_len, *block.stride()[1:]))

    def _tp_byte_ranges(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        regions: list[tuple[int, int]],
        remote_tp_size: int,
        remote_dcp_size: int,
        receiving: bool,
    ) -> list[dict[int, list[tuple[int, int, int]]]]:
        """What each `(local region, cache group)` moves with each remote
        rank, from both sides' `slice_for_tp_transfer` views:

            ranges[k][remote_rank] = [(local_offset, remote_offset, num_bytes)]

        in bytes from the start of a transfer block on each side. A rank that
        moves none of a region is left out.
        """
        assert self.transfer_topo is not None
        topo = self.transfer_topo
        tp_size, tp_rank = self.transfer_tp_size, self.transfer_tp_rank
        remote_views_by_layer = {
            name: deserialize_layer_meta_tensors(views)
            for name, views in nixl_agent_meta.layer_meta_tensors.items()
        }
        all_ranges = []
        for local, group in regions:
            # A region's layers view the same blocks: any one stands for all.
            name = self.region_members[local][0]
            spec = self._layer_specs[name]
            if name not in remote_views_by_layer:
                raise ValueError(f"The remote advertised no views of {name!r}.")
            views, contiguous = self._layer_views[name]
            remote_views = remote_views_by_layer[name]

            # Block size and layout: the transfer block. The receiver decides
            # where bytes land; a sender pairs the views of its cache with the
            # receiver's. Blocks of our layout and block size land where they
            # belong. Others land as the remote stores them, `ratio` to a
            # contiguous block of ours, and are reordered after the transfer.
            ratio = view_block_size_ratio(views, remote_views)
            block_len = self.block_len_per_layer[local] // ratio
            layout = nixl_agent_meta.kv_cache_layout
            if ratio != 1 or layout != self.transfer_kv_cache_layout:
                if not receiving:
                    raise NotImplementedError(
                        f"Sending {name!r} to a remote of another KV cache "
                        "layout or block size is not supported."
                    )
                if not contiguous:
                    raise NotImplementedError(
                        f"The blocks of {name!r} are not contiguous, so the "
                        "remote's cannot land in them: use the same KV cache "
                        "layout and block size on both sides."
                    )
                order = KVCacheLayout[layout].layer_view_order
                views = spec.slice_for_tp_transfer(
                    self._transfer_block(spec, group, block_len, order, ratio)
                )
            # Both sides lay out every block alike: pair the first of each.
            views = {component: view[:1] for component, view in views.items()}
            remote_views = {
                component: view[:1] for component, view in remote_views.items()
            }

            # TP: pair our shards with those of the remote. DCP shares out a
            # request's blocks among ranks that hold the same shards, so a
            # sender's ranks count by DCP group. It leaves recurrent state
            # alone.
            dcp_size, remote_dcp = (
                (topo.dcp_size, remote_dcp_size)
                if isinstance(spec, AttentionSpec)
                else (1, 1)
            )
            if not receiving and dcp_size != remote_dcp:
                # A push pairs the two ranks' blocks of a request in order.
                raise NotImplementedError(
                    f"Sending {name!r} to a remote of another DCP size is not "
                    "supported."
                )
            group_size, remote_group_size = (
                (1, remote_dcp) if receiving else (dcp_size, 1)
            )
            pairs_by_group = pair_shards(
                views,
                tp_size // group_size,
                tp_rank // group_size,
                remote_views,
                remote_tp_size // remote_group_size,
                receiving=receiving,
            )
            # DCP: of a paired group, the ranks that hold blocks of a request
            # in common with this rank.
            pairs_by_rank = {
                rank: pairs_by_group[rank // remote_group_size]
                for rank in range(remote_tp_size)
                if rank // remote_group_size in pairs_by_group
                and topo.shares_blocks(rank, remote_dcp)
            }

            # Bytes: the runs of each pair that are contiguous on both sides,
            # in the order of ours. A block the two sides hold with the same
            # views needs no pairing and moves whole, with the padding that
            # its views do not cover; as on main, recurrent state sharded
            # across TP leaves its padding behind.
            sharded_state = isinstance(spec, MambaSpec) and not spec.tp_replicated
            ours, theirs = map(serialize_layer_meta_tensors, (views, remote_views))
            whole = contiguous and not sharded_state and ours == theirs
            all_ranges.append(
                {
                    rank: [(0, 0, block_len)]
                    if whole
                    else sorted(contiguous_byte_ranges(pairs))
                    for rank, pairs in pairs_by_rank.items()
                }
            )
        return all_ranges

    @classmethod
    def _byte_range_descs(
        cls,
        byte_ranges: list[tuple[int, int, int, int]],
        regions: Sequence[tuple[int, int, int, int]],
        device_id: int,
    ) -> tuple[np.ndarray, dict[tuple[int, int, int, int], int], list[int]]:
        """Nx5 strided descs of each distinct `(region, offset, num_bytes,
        block_size_ratio)` byte range of a region's blocks, the first desc id
        of each, and the region of each desc row.

        `regions[i]` is `(base_addr, block_len, block_stride, num_blocks)`; a
        range moves `num_bytes` from `offset` of each of the
        `block_size_ratio` units of every block.
        """
        starts: dict[tuple[int, int, int, int], int] = {}
        parts = [np.empty((0, 5), dtype=np.uint64)]
        row_regions: list[int] = []
        num_descs = 0
        for byte_range in byte_ranges:
            if byte_range in starts:
                continue
            region, offset, num_bytes, ratio = byte_range
            base_addr, block_len, stride, num_blocks = regions[region]
            descs = cls._block_strided_descs(
                base_addr + offset,
                num_bytes,
                block_len // ratio,
                stride,
                num_blocks,
                ratio,
                device_id,
            )
            starts[byte_range] = num_descs
            num_descs += int(descs[:, 4].sum())
            parts.append(descs)
            row_regions += [region] * len(descs)
        return np.concatenate(parts), starts, row_regions

    @staticmethod
    def _compute_desc_ids(
        stage: _RemoteStage,
        remote_rank: int,
        local_block_ids: BlockIds,
        remote_block_ids: BlockIds,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Local and remote desc ids moving the block ids, both listed per
        local cache group.

        Each byte range moved with the remote rank moves its cache group's
        blocks, one desc per block, from the start fixed at handshake: one
        broadcast add per group.
        """
        starts = stage.desc_starts[remote_rank]
        local_ids: list[np.ndarray] = []
        remote_ids: list[np.ndarray] = []
        for group, local_blocks in enumerate(local_block_ids):
            if not len(local_blocks) or group not in starts:
                continue
            local_starts, remote_starts = starts[group]
            # NIXL takes int32 ids without a conversion copy.
            local_blocks = np.asarray(local_blocks, dtype=np.int32)
            remote_blocks = np.asarray(remote_block_ids[group], dtype=np.int32)
            local_ids.append((local_starts[:, None] + local_blocks).ravel())
            remote_ids.append((remote_starts[:, None] + remote_blocks).ravel())
        if len(local_ids) == 1:
            return local_ids[0], remote_ids[0]
        if not local_ids:
            empty = np.array([], dtype=np.int32)
            return empty, empty
        return np.concatenate(local_ids), np.concatenate(remote_ids)

    def _compat_backend_name(self) -> str:
        """Attention backend name folded into the compatibility hash.

        Backends of indexer caches do not count: a model may build indexers on
        some layers only, so a PP stage may start on, or hold only, layers
        without one.
        """
        indexer_layers = [
            name
            for group in self.kv_cache_config.kv_cache_groups
            for name, spec in group.kv_cache_specs.items()
            if isinstance(spec, AttentionSpec)
            and spec.cache_role is SparseCacheRole.INDEXER
        ]
        layer_type = cast(type[Any], AttentionLayerBase)
        indexer_backends = {
            layer.get_attn_backend()
            for layer in get_layers_from_vllm_config(
                self.vllm_config, layer_type, indexer_layers
            ).values()
        }
        backends = [b for b in self.attn_backends if b not in indexer_backends]
        if self._supports_pp_hma and self._has_packed_cache:
            # PP can reorder the same backends across stages of a packed model.
            return ",".join(sorted({backend.get_name() for backend in backends}))
        return backends[0].get_name()

    def _set_region_layers(self, region_layers: list[list[str]]) -> None:
        self.region_members: list[list[str]] = region_layers
        self._region_by_layer: dict[str, int] = {
            layer_name: region
            for region, layer_names in enumerate(region_layers)
            for layer_name in layer_names
        }

    def _pair_regions(
        self, nixl_agent_meta: NixlAgentMetadata, remote_pp_stage: bool = False
    ) -> tuple[list[RegionPair], dict[int, int]]:
        """Pair a peer's regions with the local ones through the layers both
        hold: the `(remote, local, group)` region pairs, and the peer's cache
        group each local group pairs with.

        The peer must hold every local layer; layers only the peer holds are
        skipped on a PP stage, which holds a slice of them.
        """
        engine_id = nixl_agent_meta.engine_id
        num_regions = len(nixl_agent_meta.kv_caches_base_addr)
        if any(
            values is not None and len(values) != num_regions
            for values in (
                nixl_agent_meta.block_lens,
                nixl_agent_meta.block_strides,
                nixl_agent_meta.region_num_blocks,
                nixl_agent_meta.region_group_ids,
                nixl_agent_meta.region_mem_types,
            )
        ):
            raise RuntimeError(
                f"Engine {engine_id} advertised KV region metadata of different "
                "lengths."
            )
        if len(nixl_agent_meta.region_members) != num_regions:
            raise RuntimeError(
                f"Engine {engine_id} did not advertise the layers of its KV regions."
            )
        remote_groups = nixl_agent_meta.region_group_ids
        remote_layer_groups = nixl_agent_meta.layer_group_ids
        group_by_layer = self.kv_cache_config.transfer_group_index_by_layer

        pairs: dict[RegionPair, None] = {}
        remote_group_by_group: dict[int, int] = {}
        layers: set[str] = set()
        for remote, layer_names in enumerate(nixl_agent_meta.region_members):
            for layer_name in layer_names:
                local = self._region_by_layer.get(layer_name)
                if local is None:
                    if self.pp_size > 1:
                        # This PP stage holds a slice of the peer's layers.
                        continue
                    raise RuntimeError(
                        f"Engine {engine_id} holds KV layer {layer_name!r}, which "
                        "has no counterpart here."
                    )
                group = group_by_layer[layer_name]
                layers.add(layer_name)
                # Block ids are listed per group: the peer's group of the layer
                # says which of its lists the layer's blocks are in.
                remote_group = remote_layer_groups.get(layer_name)
                if remote_group is None:
                    remote_group = remote_groups[remote] if remote_groups else group
                # A cache group's blocks pair with those of a single peer group.
                paired_group = remote_group_by_group.setdefault(group, remote_group)
                if paired_group != remote_group:
                    raise NotImplementedError(
                        f"Engine {engine_id} splits the layers of a local KV "
                        "cache group across several of its cache groups."
                    )
                pairs[(remote, local, group)] = None
        missing = self._region_by_layer.keys() - layers
        if missing and not remote_pp_stage:
            raise RuntimeError(
                f"Engine {engine_id} does not hold the KV layers "
                f"{sorted(missing, key=_region_sort_key)}."
            )
        if self._has_mamba and any(
            group != remote_group
            for group, remote_group in remote_group_by_group.items()
        ):
            raise NotImplementedError(
                f"Engine {engine_id} puts Mamba/SSM hybrid KV layers in other "
                "cache groups than this worker; use the same KV cache plan on P "
                "and D."
            )
        return list(pairs), remote_group_by_group

    def _groups_match(self, engine_id: EngineId) -> bool:
        """Whether every paired layer is in the same cache group on both
        sides."""
        return all(
            group == remote_group
            for group, remote_group in enumerate(self.remote_group_ids[engine_id])
        )

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        nixl_wrapper_cls = NixlWrapper
        if nixl_wrapper_cls is None:
            logger.error("NIXL is not available")
            raise RuntimeError("NIXL is not available")
        # Descriptor lists are prepared in NIXL's strided (Nx5) form.
        check_nixl_version("1.5.0")
        logger.info("Initializing NIXL wrapper")
        logger.info("Initializing NIXL worker %s", engine_id)

        # Config.
        self.vllm_config = vllm_config

        if vllm_config.kv_transfer_config is None:
            raise ValueError("kv_transfer_config must be set for NixlConnector")
        self.kv_transfer_config = vllm_config.kv_transfer_config

        self.nixl_backends = vllm_config.kv_transfer_config.get_from_extra_config(
            "backends", ["UCX"]
        )
        kv_lease_duration: int = vllm_config.kv_transfer_config.get_from_extra_config(
            "kv_lease_duration", 30
        )
        # NOTE (NickLucche): For now we use a hardcoded value for a simpler interface.
        self._lease_extension = kv_lease_extension(kv_lease_duration)
        # At most the heartbeat interval so heartbeats can keep a short lease
        # alive (assumes both sides share kv_lease_duration, like the interval).
        self._lease_safety_margin = min(
            _KV_BLOCKS_EXPIRY_SAFETY_MARGIN, kv_lease_duration / 6
        )

        self._bidirectional_kv_xfer_enabled: bool = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "bidirectional_kv_xfer", False
            )
        )

        self.kv_cache_config = kv_cache_config
        transfer_block_sizes = [
            group.block_size
            for group in kv_cache_config.transfer_groups
            if not _is_ssm_spec(get_representative_spec_type(group))
            and not issubclass(group.manager_cls, CircularBufferManager)
        ]
        self.block_size = (
            math.lcm(*transfer_block_sizes)
            if transfer_block_sizes
            else cast(int, vllm_config.cache_config.block_size)
        )
        self._layer_specs: dict[str, KVCacheSpec] = {
            name: spec
            for group in kv_cache_config.transfer_groups
            for name, spec in group.kv_cache_specs.items()
        }
        self._is_hma_required = (
            not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            and any(
                not issubclass(g.manager_cls, FullAttentionManager)
                for g in kv_cache_config.transfer_groups
            )
        )
        # A failed load also reports the request's blocks only when the
        # scheduler can map them back to requests: one KV cache group, counting
        # groups outside the transfer (e.g. HiSparse). Otherwise the scheduler
        # raises on invalid_block_ids, so failures go through failed_recving.
        # TODO: Drop this once _handle_invalid_blocks supports several groups.
        self._report_invalid_block_ids = (
            not self._is_hma_required and len(kv_cache_config.kv_cache_groups) == 1
        )

        # ---- Model state (derived from model config) ----
        self._has_mamba = any(
            isinstance(spec, MambaSpec) for spec in self._layer_specs.values()
        )
        assert not self._has_mamba or self._is_hma_required
        self._is_csa_linear = any(
            issubclass(group.manager_cls, CircularBufferManager)
            for group in kv_cache_config.kv_cache_groups
        )

        # Agent.
        non_ucx_backends = [b for b in self.nixl_backends if b != "UCX"]
        # Configure NIXL num_threads to avoid UAR exhaustion on Mellanox NICs.
        # Each UCX thread allocates UARs (doorbell pages) via DevX, and
        # excessive NIXL UAR usage can exhaust NIC UAR space. This can cause
        # components like NVSHMEM (used by DeepEP kernels) to fail during RDMA
        # initialization with "mlx5dv_devx_alloc_uar" errors.
        # Ref: https://network.nvidia.com/files/doc-2020/ethernet-adapters-programming-manual.pdf#page=63
        num_threads = vllm_config.kv_transfer_config.get_from_extra_config(
            "num_threads", self._DEFAULT_NUM_THREADS
        )
        if nixl_agent_config is None:
            config = None
        else:
            # Enable telemetry by default for NIXL 0.7.1 and above.
            config = (
                nixl_agent_config(backends=self.nixl_backends, capture_telemetry=True)
                if len(non_ucx_backends) > 0
                else nixl_agent_config(num_threads=num_threads, capture_telemetry=True)
            )

        # Extra UCX backend init params, e.g. {"ucx_error_handling_mode": "none"}.
        params = self.kv_transfer_config.get_from_extra_config("ucx_backend_params", {})
        ucx_backend_params = {k: str(v) for k, v in params.items()}
        sleep_mode = vllm_config.model_config.enable_sleep_mode and config is not None
        # With UCX params, create every backend here, in the configured order.
        explicit_backends = (config is not None and not sleep_mode) and (
            bool(ucx_backend_params) and "UCX" in self.nixl_backends
        )
        if sleep_mode or explicit_backends:
            config = nixl_agent_config(backends=[], capture_telemetry=True)
        if config is not None and nixl_thread_sync_t is not None:
            # Handshake threads load remote agents while others post, and NIXL
            # only locks the agent given a sync mode. RW keeps posting concurrent.
            config.sync_mode = nixl_thread_sync_t.NIXL_THREAD_SYNC_RW
        self.nixl_wrapper = nixl_wrapper_cls(str(uuid.uuid4()), config)
        if sleep_mode:
            # Keep slept pages unpinned, for this agent only; UCX_* env vars win.
            self.nixl_wrapper.create_backend(
                "UCX",
                {
                    "num_threads": str(num_threads),
                    "engine_config": "RCACHE_ENABLE=n,TLS=^cuda_ipc",
                    **ucx_backend_params,
                },
            )
        elif explicit_backends:
            for backend in self.nixl_backends:
                nixl_params: dict[str, str] = {}
                if backend == "UCX":
                    if not non_ucx_backends:
                        nixl_params["num_threads"] = str(num_threads)
                    nixl_params.update(ucx_backend_params)
                self.nixl_wrapper.create_backend(backend, nixl_params)
        # Map of engine_id -> {(pp_rank, tp_rank): agent_name, ...}.
        # non-PP remote uses pp_rank 0, i.e. (0, tp_rank).
        self._remote_agents: dict[EngineId, dict[tuple[int, int], str]] = defaultdict(
            dict
        )
        # Failed reads need a local metadata probe; invalid peers wait for drain.
        self._failed_remote_engines: set[EngineId] = set()
        # Failed reads that NIXL reported as a lost connection. Their metadata
        # can still be loaded, so they are invalidated without the probe.
        self._disconnected_remote_engines: set[EngineId] = set()
        self._invalid_remote_engines: set[EngineId] = set()
        # Invalid engine -> (backoff, earliest release time). Kept across release
        # so that repeat invalidations back off.
        self._invalid_engine_backoff: dict[EngineId, tuple[float, float]] = {}
        # Map of engine_id -> clock offset.
        self._engine_clock_offset: dict[EngineId, float] = {}
        # (remote engine, remote req id) -> lease deadline renewed by our heartbeats.
        self._remote_leases: dict[tuple[EngineId, str], float] = {}

        # Metadata.
        self.engine_id: EngineId = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.world_size = get_tensor_model_parallel_world_size()
        self.pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0

        # TP1 PCP+DCP owns distinct KV shards; replicated PCP uses rank zero.
        self.pcp_dcp_sharded = self.pcp_size > 1 and self.dcp_size > 1
        self.transfer_tp_size = (
            self.pcp_size if self.pcp_dcp_sharded else self.world_size
        )
        self.transfer_tp_rank = self.pcp_rank if self.pcp_dcp_sharded else self.tp_rank

        # MLA is either fully replicated or fully sharded across transfer ranks.
        self.dcp_rank = self.transfer_tp_rank % self.dcp_size

        self.num_blocks = kv_cache_config.num_blocks
        self.enable_permute_local_kv = False
        self.enable_heterogeneous_attn_post_process = False

        # KV Caches and nixl tracking data.
        self.device_type = current_platform.device_type
        self.kv_buffer_device: str = vllm_config.kv_transfer_config.kv_buffer_device
        if self.device_type not in _NIXL_SUPPORTED_DEVICE:
            raise RuntimeError(f"{self.device_type} is not supported.")
        elif self.kv_buffer_device not in _NIXL_SUPPORTED_DEVICE[self.device_type]:
            raise RuntimeError(
                f"{self.device_type} with {self.kv_buffer_device} kv_buffer "
                "is not supported."
            )
        self.device_kv_caches: dict[str, torch.Tensor] = {}

        # cpu kv buffer for xfer
        # used when device memory can not be registered under nixl
        self.host_xfer_buffers: dict[str, torch.Tensor] = {}
        if self.device_type == "cpu":
            self.use_host_buffer = False
        else:
            self.use_host_buffer = self.kv_buffer_device == "cpu"

        # reserve different cores for start_load_kv() from model_forward()
        if self.device_type == "cpu":
            numa_core_list = current_platform.discover_numa_topology()
            # setup one last core in each numa for kv transfer.
            rsv_cores_for_kv = [
                max(each_numa_core_list) for each_numa_core_list in numa_core_list
            ]

            if rsv_cores_for_kv:
                if not hasattr(os, "sched_setaffinity"):
                    raise NotImplementedError(
                        "os.sched_setaffinity is not available on this platform"
                    )
                os.sched_setaffinity(0, rsv_cores_for_kv)

        # support for oot platform which can't register nixl memory
        # type based on kv_buffer_device
        nixl_memory_type = current_platform.get_nixl_memory_type()
        if nixl_memory_type is None:
            if self.kv_buffer_device in ["cuda", "xpu"]:
                nixl_memory_type = "VRAM"
            elif self.kv_buffer_device == "cpu":
                nixl_memory_type = "DRAM"
        if nixl_memory_type is None:
            raise RuntimeError(
                f"{self.device_type} with {self.kv_buffer_device} kv_buffer "
                "is not supported."
            )
        self.nixl_memory_type = nixl_memory_type

        # Note: host xfer buffer ops when use_host_buffer is True
        self.copy_blocks: CopyBlocksOp | None = None

        # Map of engine_id -> kv_caches_base_addr. For TP case, each local
        self.device_id: int = 0
        # Current rank may pull from multiple remote TP workers.
        # EngineId, dict[int, list[int]] -> engine_id, tp_rank, base_addr_for_layer
        self.kv_caches_base_addr = defaultdict[EngineId, dict[int, list[int]]](dict)

        # Number of NIXL regions: one per layer, or per set of layers viewing
        # the same blocks.
        self.num_regions = 0
        self.region_mem_types: list[str] = []
        self.region_group_ids: list[int] = []
        self.region_num_blocks: list[int] = []
        self._mixed_mem_types = False
        # True on TP ranks that share a HiSparse host pool with their node's
        # host leader and leave its NIXL registration and DRAM reads to it.
        self._skip_dram_xfer = False

        # PP>1: this worker holds a contiguous layer slice. Push transfers it
        # into the PP=1 remote's regions that hold its layers; a PP=1 puller
        # reads each stage's regions into the local ones holding its layers.
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        if self.pp_size > 1:
            if (
                self._is_hma_required
                and not self._supports_pp_hma
                and (vllm_config.model_config.use_mla or self._is_csa_linear)
            ):
                raise NotImplementedError(
                    "NixlConnector (pull) supports pipeline_parallel_size > 1 with "
                    "hybrid KV cache layouts (HMA) only for attention KV without "
                    "MLA or CSA-linear caches; use NixlPushConnector for those."
                )
            # Mamba/SSM hybrid PP stages are not validated yet.
            if self._has_mamba:
                raise NotImplementedError(
                    "NIXL connectors do not support pipeline_parallel_size > 1 "
                    "with Mamba/SSM hybrid KV cache layouts yet."
                )
            # Decode-side PP is unsupported for push (completions counted per
            # consumer rank). Pull producers may be PP-sharded.
            if (
                self._supports_pp_hma
                and vllm_config.kv_transfer_config.kv_role == "kv_consumer"
            ):
                raise NotImplementedError(
                    "NixlPushConnector consumer (decode) does not support "
                    "pipeline_parallel_size > 1."
                )
        # Keep heartbeat handshakes to a PP-sharded producer notif-only.
        self._hb_handshake_notif_only = False

        # Local desc lists, keyed by the byte ranges they hold so stages
        # moving the same ranges share one.
        self._local_desc_lists: dict[tuple, _LocalDescs] = {}
        # Push decode side: per engine and PP stage, the ranks pushing here.
        self._push_senders = defaultdict[EngineId, dict[int, int]](dict)

        # Map of engine_id -> num_blocks. All ranks in the same deployment will
        # have the same number of blocks.
        self.dst_num_blocks: dict[EngineId, int] = {}
        # engine_id -> pp_rank -> that stage of the engine. A PP=1 remote is
        # a single stage 0.
        self._remote_stages: dict[EngineId, dict[int, _RemoteStage]] = {}
        # Engine -> physical blocks per logical block of each of its cache
        # groups, and how many smaller blocks each of its token blocks moves as.
        self._remote_physical_blocks_per_group: dict[EngineId, list[int]] = {}
        self._remote_block_splits: dict[EngineId, int] = {}
        self._registered_descs: list[Any] = []
        self._kv_unregistered = False

        # In progress transfers.
        # [req_id -> list[handle]]
        self._recving_metadata: dict[ReqId, ReqMeta] = {}
        self._recving_transfers = defaultdict[ReqId, list[TransferHandle]](list)
        # Track the expiration time of requests that are waiting to be sent.
        self._reqs_to_send: dict[ReqId, float] = {}
        # Replicated-KV PCP ranks > 0 never transfer; requests they are asked
        # to send are reported finished immediately (see get_finished).
        self._replicated_pcp_done_sending: set[ReqId] = set()
        # Set of requests that have been part of a batch, regardless of status.
        self._reqs_to_process: set[ReqId] = set()

        # Invalid blocks from failed NIXL operations (thread-safe queue of block ids)
        self._invalid_block_ids: queue.Queue[set[int]] = queue.Queue()
        # requests that skipped transfer (handshake or transfer failures)
        # Uses Queue for thread-safe cross-thread coordination with the
        # background handshake thread, matching the _ready_requests pattern.
        self._failed_recv_reqs: queue.Queue[ReqId] = queue.Queue()
        self._recv_failures: set[ReqId] = set()
        self._pending_recv_notifs: dict[ReqId, list[tuple[str, bytes]]] = {}

        # Handshake metadata of this worker for NIXL transfers.
        self.xfer_handshake_metadata: NixlHandshakePayload | None = None
        # Background thread for initializing new NIXL handshakes.
        self._handshake_initiation_executor = ThreadPoolExecutor(
            # NIXL is not guaranteed to be thread-safe, limit 1 worker.
            max_workers=1,
            thread_name_prefix="vllm-nixl-handshake-initiator",
        )
        self._ready_requests = queue.Queue[tuple[ReqId, ReqMeta]]()
        self._handshake_futures: dict[
            EngineId, Future[tuple[dict[tuple[int, int], str], float]]
        ] = {}
        # Protects _handshake_futures and _remote_agents.
        self._handshake_lock = threading.RLock()

        # TTL-based eviction of stale remote engine state.
        self._engine_last_active: dict[EngineId, float] = {}
        self._engine_ttl: float = vllm_config.kv_transfer_config.get_from_extra_config(
            "engine_ttl", 3600.0
        )
        # Map of pull peer (host, port) -> engine_id currently serving it.
        self._engine_by_address: dict[tuple[str, int], EngineId] = {}

        self.model_config = vllm_config.model_config

        self.use_mla = self.model_config.use_mla

        # Get the attention backend from the first layer
        # NOTE (NickLucche) models with multiple backends are not supported yet
        self.attn_backends = get_current_attn_backends(vllm_config)
        self.backend_name = self.attn_backends[0].get_name()

        self.kv_cache_layout = (
            vllm_config.cache_config.get_resolved_kv_cache_layout().name
        )
        self.host_buffer_kv_cache_layout = self.kv_cache_layout
        logger.info(
            "Detected attention backend(s) %s",
            [backend.get_name() for backend in self.attn_backends],
        )
        logger.info("Detected kv cache layout %s", self.kv_cache_layout)

        # lazy initialized in register_kv_caches
        self.compat_hash: str | None = None
        self.transfer_topo: TransferTopology | None = None

        # With heterogeneous TP (or DCP), P must wait for all assigned D
        # workers to finish reading before safely freeing the blocks.
        self.consumer_notification_counts_by_req = defaultdict[ReqId, int](int)
        self.expected_consumer_notifications_by_req: dict[ReqId, int] = {}
        self.xfer_stats = NixlKVConnectorStats()

        self._physical_blocks_per_logical_kv_block = 1
        self._sync_block_size_with_kernel()
        self._physical_blocks_per_group = [1] * len(kv_cache_config.transfer_groups)
        self._logical_num_blocks = getattr(self, "_logical_num_blocks", self.num_blocks)

        self._group_spec_types = tuple(
            get_representative_spec_type(g)
            for g in self.kv_cache_config.transfer_groups
        )
        # MLA drafts running at DCP=1 under a DCP target: a DCP peer holds
        # them in full on every rank.
        self._dcp_replicated_groups = get_dcp_replicated_group_ids(
            self.kv_cache_config.transfer_groups
        )
        # Per region, the length of its blocks.
        self.block_len_per_layer = list[int]()

        # Per-region block stride in bytes. Taken from the registered tensor's
        # stride(0) so it stays correct under layouts that interleave layers
        # within a block (BLHNC/BHLNC), where stride > block_len.
        self.block_stride_per_layer = list[int]()

        # Block-outermost layouts (BLHNC/BHLNC) interleave every layer's page of
        # a block in one allocation.
        self._has_packed_cache = KVCacheLayout[self.kv_cache_layout].is_block_outermost
        if (
            self._TRANSFER_MODE == "pull"
            and self.pp_size > 1
            and self._has_packed_cache
        ):
            raise NotImplementedError(
                "NixlConnector (pull) does not support PP with packed KV cache layout "
                f"{self.kv_cache_layout}. Use a supported non-packed layout; "
                "NixlPushConnector has separate packed PP restrictions."
            )
        # The layers of each region, which pair regions with a peer's.
        self.region_members = []
        self._region_by_layer = {}
        # Layer -> its `slice_for_tp_transfer` views, in bytes from its first
        # block, and whether each of its blocks is contiguous.
        self._layer_views: dict[str, tuple[dict[str, torch.Tensor], bool]] = {}
        # Per engine and cache group, the remote ranks its blocks move with.
        self.source_ranks: dict[EngineId, list[tuple[int, ...]]] = {}
        # Per engine and cache group, the remote cache group its blocks pair
        # with (`_pair_regions`).
        self.remote_group_ids: dict[EngineId, list[int]] = {}

        self.enforce_compat_hash = self.kv_transfer_config.get_from_extra_config(
            "enforce_handshake_compat", True
        )

    def _sync_block_size_with_kernel(self) -> None:
        backends, specs, layer_names = get_current_attn_backends_and_specs(
            self.vllm_config, self.kv_cache_config, self.attn_backends
        )
        kernel_block_size = select_common_block_size(
            self.block_size, backends, specs, layer_names
        )
        check_compressed_kernel_split(
            self.vllm_config, self.kv_cache_config, kernel_block_size
        )
        self._logical_num_blocks = self.num_blocks
        if self.block_size != kernel_block_size:
            logger.info_once(
                "User-specified logical block size (%s) does not match"
                " physical kernel block size (%s). Using the latter.",
                self.block_size,
                kernel_block_size,
            )
            assert self.block_size > kernel_block_size
            self._physical_blocks_per_logical_kv_block = (
                self.block_size // kernel_block_size
            )
            self.block_size = kernel_block_size
            self.num_blocks *= self._physical_blocks_per_logical_kv_block

    def _nixl_handshake(
        self,
        host: str,
        port: int,
        remote_tp_size: int,
        expected_engine_id: str,
        remote_dcp_size: int = 1,
        remote_pp_size: int = 1,
        notif_agents_only: bool = False,
    ) -> tuple[dict[tuple[int, int], str], float]:
        """Do a NIXL handshake with a remote instance."""
        if self._is_csa_linear:
            assert self.transfer_topo is not None
            validate_csa_linear_tp_layout(
                self.world_size, remote_tp_size, self.transfer_topo.total_num_kv_heads
            )

        # the first time we connect to a remote agent.
        # be careful, the handshake happens in a background thread.
        # it does not have an active cuda context until any cuda runtime
        # call is made. when UCX fails to find a valid cuda context, it will
        # disable any cuda ipc communication, essentially disabling any NVLink
        # communication.
        # when we are using device buffers, we need to set the device
        # explicitly to make sure the handshake background thread has a valid
        # cuda context.
        if not self.use_host_buffer:
            current_platform.set_device(self.device_id)

        # When target instance TP > local TP, we need to perform multiple
        # handshakes. Do it in a single background job for simplicity.
        # Regardless, only handshake with the remote TP rank(s) that current
        # local rank will read from. Note that With homogeneous TP,
        # this happens to be the same single rank_i.
        assert self.transfer_topo is not None
        p_remote_ranks = self.transfer_topo.handshake_target_ranks(
            remote_tp_size, remote_dcp_size
        )
        remote_rank_to_agent_name: dict[tuple[int, int], str] = {}
        path = make_zmq_path("tcp", host, port)
        # Clock offset to the peer, estimated from the handshake round-trip.
        # Keep the lowest-RTT sample: hop cost is ~uniform across ranks, so a
        # higher RTT is just noise that skews the midpoint estimate.
        best_rtt = float("inf")
        best_offset: float | None = None

        with zmq_ctx(zmq.REQ, path) as sock:
            for remote_pp_rank, remote_rank in itertools.product(
                range(remote_pp_size), p_remote_ranks
            ):
                logger.debug(
                    "Querying metadata on path: %s at remote pp rank %s, tp rank %s",
                    path,
                    remote_pp_rank,
                    remote_rank,
                )

                # Send query for the request.
                msg = msgspec.msgpack.encode(
                    (GET_META_MSG, remote_pp_rank, remote_rank)
                )
                # Set receive timeout to 5 seconds to avoid hanging on dead server
                sock.setsockopt(zmq.RCVTIMEO, 5000)  # milliseconds
                start_time = time.perf_counter()
                sock.send(msg)
                reply_parts = sock.recv_multipart()
                recv_time = time.perf_counter()
                assert len(reply_parts) == 2
                handshake_bytes = reply_parts[0]

                remote_perf = msgspec.msgpack.decode(reply_parts[1])
                rtt = recv_time - start_time
                if rtt < best_rtt:
                    best_rtt = rtt
                    best_offset = remote_perf - (start_time + recv_time) / 2

                # Decode handshake payload to get compatibility hash
                handshake_decoder = msgspec.msgpack.Decoder(NixlHandshakePayload)
                try:
                    handshake_payload = handshake_decoder.decode(handshake_bytes)
                except (msgspec.DecodeError, msgspec.ValidationError) as e:
                    raise RuntimeError(
                        f"Failed to decode NixlHandshakePayload. This likely indicates "
                        f"an incompatibility between connector version. Error: {e}"
                    ) from e

                got_metadata_time = time.perf_counter()
                logger.debug(
                    "NIXL handshake: get metadata took: %s",
                    got_metadata_time - start_time,
                )

                # Check compatibility hash BEFORE decoding agent metadata
                assert self.compat_hash is not None
                if (
                    self.enforce_compat_hash
                    and handshake_payload.compatibility_hash != self.compat_hash
                ):
                    raise RuntimeError(
                        f"NIXL compatibility hash mismatch. "
                        f"Local: {self.compat_hash}, "
                        f"Remote: {handshake_payload.compatibility_hash}. "
                        f"Prefill and decode instances have incompatible "
                        f"configurations. This may be due to: different vLLM versions,"
                        f" models, dtypes, KV cache layouts, attention backends, etc. "
                        f"Both instances must use identical configurations."
                        f"Disable this check using "
                        f'--kv-transfer-config \'{{"kv_connector_extra_config": '
                        f'{{"enforce_handshake_compat": false}}}}\''
                    )

                logger.info(
                    "NIXL compatibility check passed (hash: %s)",
                    handshake_payload.compatibility_hash,
                )

                # Decode agent metadata
                metadata_decoder = msgspec.msgpack.Decoder(NixlAgentMetadata)
                try:
                    metadata = metadata_decoder.decode(
                        handshake_payload.agent_metadata_bytes
                    )
                except (msgspec.DecodeError, msgspec.ValidationError) as e:
                    # This should not happen if hash matched
                    raise RuntimeError(
                        f"Failed to decode NixlAgentMetadata. Error: {e}"
                    ) from e

                validate_remote_parallel_config(self.pcp_size, self.dcp_size, metadata)
                if self._TRANSFER_MODE == "push":
                    validate_push_block_sizes(self, metadata)

                # Ensure engine id matches.
                if metadata.engine_id != expected_engine_id:
                    raise RuntimeError(
                        f"Remote NIXL agent engine ID mismatch. "
                        f"Expected {expected_engine_id},"
                        f"received {metadata.engine_id}."
                    )

                # Register Remote agent.
                if notif_agents_only:
                    remote_agent_name = self._add_notif_only_remote_agent(
                        metadata, remote_pp_rank, remote_tp_size, metadata.dcp_size
                    )
                else:
                    remote_agent_name = self.add_remote_agent(
                        metadata,
                        remote_rank,
                        remote_tp_size,
                        metadata.dcp_size,
                        remote_pp_rank=remote_pp_rank,
                        remote_pp_size=remote_pp_size,
                    )
                setup_agent_time = time.perf_counter()
                logger.debug(
                    "NIXL handshake: add agent took: %s (notif_agents_only=%s)",
                    setup_agent_time - got_metadata_time,
                    notif_agents_only,
                )
                remote_ranks = (remote_pp_rank, remote_rank)
                remote_rank_to_agent_name[remote_ranks] = remote_agent_name

        assert best_offset is not None
        return remote_rank_to_agent_name, best_offset

    def _add_notif_only_remote_agent(
        self,
        metadata: NixlAgentMetadata,
        remote_pp_rank: int,
        remote_tp_size: int,
        remote_dcp_size: int = 1,
    ) -> str:
        """Load a remote agent for notifs only on the push-mode decode side.

        Skips descriptor setup but records engine info for block accounting,
        and how many of the stage's ranks push to this one.
        """
        assert self.transfer_topo is not None
        group_by_layer = self.kv_cache_config.transfer_group_index_by_layer
        regions = {
            (self._region_by_layer[name], group_by_layer[name])
            for layer_names in metadata.region_members
            for name in layer_names
            if name in self._region_by_layer
        }
        senders = set[int]().union(
            *self._tp_byte_ranges(
                metadata, list(regions), remote_tp_size, 1, receiving=True
            )
        )
        self._push_senders[metadata.engine_id][remote_pp_rank] = len(senders)
        self.transfer_topo.register_remote_engine(
            metadata.engine_id,
            EngineTransferInfo(
                remote_tp_size=remote_tp_size,
                remote_block_size=metadata.block_size,
                remote_block_len=metadata.block_lens[0],
                remote_physical_blocks_per_logical=max(
                    metadata.physical_blocks_per_group, default=1
                ),
                remote_dcp_size=remote_dcp_size,
            ),
        )
        return self.nixl_wrapper.add_remote_agent(metadata.agent_metadata)

    def initialize_host_xfer_buffer(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Initialize transfer buffer in CPU mem for accelerators
        NOT directly supported by NIXL (e.g., tpu)
        """
        xfer_buffers: dict[str, torch.Tensor] = {}
        try:
            for layer_name, kv_cache in kv_caches.items():
                kv_shape = kv_cache.shape
                kv_dtype = kv_cache.dtype
                permute_shape = False
                inv_order = (0, 2, 1, 3)
                if not self.use_mla:
                    assert kv_cache.ndim == 4

                    if self.kv_cache_layout == "LBNHC":
                        if self.kv_transfer_config.enable_permute_local_kv:
                            logger.info_once(
                                "'enable_permute_local_kv' flag is enabled while "
                                "device KV Layout is LBNHC. Init host buffer with"
                                " LBHNC to better support Decode/Prefill TP_ratio > 1."
                            )
                            # Since LBNHC will not support Decode/Prefill TP_ratio > 1,
                            # we can leverage host_buffer for permute.
                            self.host_buffer_kv_cache_layout = "LBHNC"
                        else:
                            # Packed KV layout is logical (B, H, N, 2*D). Allocate
                            # (B, N, H, 2*D) and view it as logical (B, H, N, 2*D)
                            # so raw NIXL transfers see NHD physical strides.
                            kv_shape = tuple(kv_shape[i] for i in inv_order)
                            permute_shape = True

                xfer_buffers[layer_name] = torch.empty(
                    kv_shape, dtype=kv_dtype, device="cpu"
                )
                if permute_shape:
                    xfer_buffers[layer_name] = xfer_buffers[layer_name].permute(
                        inv_order
                    )
        except MemoryError as e:
            logger.error("NIXLConnectorWorker gets %s.", e)
            raise

        self.host_xfer_buffers = xfer_buffers

    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp):
        """Assign copy (d2h, h2d) operations when host buffer is used."""
        # Set a no-op if the host buffer is not cpu.
        if self.kv_buffer_device != "cpu":
            return
        # Set a no-op if self.device_type is 'cpu'.
        if self.device_type == "cpu":
            return
        assert self.use_host_buffer
        self.copy_blocks = copy_operation

    def _log_failure(
        self,
        failure_type: str,
        req_id: str | None,
        msg: str = "",
        error: BaseException | None = None,
        meta: ReqMeta | None = None,
        **extra_context,
    ):
        """Log transfer failure with structured context for easier debugging."""
        context: dict[str, Any] = {
            "failure_type": failure_type,
            "request_id": req_id,
            "engine_id": self.engine_id,
        }
        if meta is None and req_id is not None:
            # Try to get metadata from in progress transfers when not provided
            meta = self._recving_metadata.get(req_id)

        if meta and meta.remote:
            context.update(
                {
                    "remote_engine_id": meta.remote.engine_id,
                    "remote_request_id": meta.remote.request_id,
                    "remote_host": meta.remote.host,
                    "remote_port": meta.remote.port,
                    "num_local_blocks": sum(
                        len(group) for group in meta.local_block_ids
                    ),
                    "num_remote_blocks": sum(
                        len(group) for group in meta.remote.block_ids
                    ),
                    "local_block_ids_sample": meta.local_block_ids[0][:10]
                    if meta.local_block_ids
                    else [],
                }
            )

        context.update(extra_context)
        if msg:
            failure_type = f"{failure_type}. {msg}"

        logger.error(
            "NIXL transfer failure: %s | Context: %s",
            failure_type,
            context,
            exc_info=error is not None,
            stacklevel=2,
        )

    def _ensure_handshake(
        self,
        engine_id: EngineId,
        host: str,
        port: int,
        tp_size: int,
        dcp_size: int = 1,
        pp_size: int = 1,
        notif_agents_only: bool = False,
    ) -> Future[tuple[dict[tuple[int, int], str], float]] | None:
        """Ensure a handshake is in-flight (or already done) for *engine_id*.

        Returns the `Future` if a handshake is pending (or was just
        started), or `None` if the handshake already completed
        successfully.  Callers can attach per-request callbacks to the
        returned future.
        Failures to handshake are logged and the request is marked as failed.
        """
        self._evict_stale_engines()
        with self._handshake_lock:
            if engine_id in self._remote_agents:
                return None
            fut = self._handshake_futures.get(engine_id)
            if fut is not None:
                return fut
            fut = self._handshake_initiation_executor.submit(
                self._nixl_handshake,
                host,
                port,
                tp_size,
                engine_id,
                dcp_size,
                pp_size,
                notif_agents_only,
            )
            self._handshake_futures[engine_id] = fut

            def done_callback(
                f: Future[tuple[dict[tuple[int, int], str], float]],
                eid=engine_id,
            ):
                with self._handshake_lock:
                    del self._handshake_futures[eid]
                    try:
                        remote_agents, clock_offset = f.result()
                        self._remote_agents[eid] = remote_agents
                        self._engine_clock_offset[eid] = clock_offset
                        # Exported lease deadlines are rebased with this offset.
                        logger.info(
                            "NIXL clock offset to engine %s: %.3fs", eid, clock_offset
                        )
                        self._engine_last_active[eid] = time.perf_counter()
                        self._track_remote_engine_replacement(eid, host, port)
                    except Exception as e:
                        self._log_failure(
                            failure_type="handshake_setup_failed",
                            req_id=None,
                            error=e,
                            remote_engine_id=eid,
                        )
                        # Count once per handshake, regardless of waiting requests.
                        self.xfer_stats.record_failed_handshake()

            fut.add_done_callback(done_callback)
            return fut

    def _background_nixl_handshake(
        self, req_id: str, remote_engine_id: EngineId, meta: ReqMeta
    ):
        # Do NIXL handshake in background and add to _ready_requests when done.
        assert meta.remote is not None
        fut = self._ensure_handshake(
            remote_engine_id,
            meta.remote.host,
            meta.remote.port,
            meta.tp_size,
            meta.dcp_size,
            meta.pp_size,
        )
        if fut is None:
            # Already handshaked — only happens if caller does not pre-check.
            self._ready_requests.put((req_id, meta))
            return

        # Check handshake success before proceeding with request.
        def request_ready(f: Future[Any], entry=(req_id, meta)):
            try:
                f.result()
                self._ready_requests.put(entry)
            except Exception as e:
                self._log_failure(
                    failure_type="handshake_failed",
                    req_id=req_id,
                    error=e,
                    meta=meta,
                )
                self._failed_recv_reqs.put(req_id)

        fut.add_done_callback(request_ready)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in nixl."""
        # The allocator views each layer's cache in the blocks its kernels
        # use, which may split a logical block; read how many per cache group.
        group_by_layer = self.kv_cache_config.transfer_group_index_by_layer
        config = self.kv_cache_config
        group_num_blocks = [
            (config.hisparse_host_num_blocks if group.host_resident else 0)
            or config.num_blocks
            for group in config.transfer_groups
        ]
        physical_blocks_per_group = [1] * len(group_num_blocks)
        for name, cache in kv_caches.items():
            group_id = group_by_layer.get(name)
            if group_id is not None and name in self._layer_specs and cache.ndim > 1:
                physical_blocks_per_group[group_id] = max(
                    1, cache.shape[0] // group_num_blocks[group_id]
                )
        # Per-group ratios come from #65232 (needed for hybrid MLA+SSM where
        # attention is kernel-split and SSM is not). Scalar block_size /
        # num_blocks / _physical_blocks_per_logical_kv_block are already set
        # by #61452's _sync_block_size_with_kernel in __init__ — do not
        # re-divide here (that double-applies and collapses block_size).
        self._physical_blocks_per_group = physical_blocks_per_group
        self.transfer_topo = TransferTopology(
            tp_rank=self.transfer_tp_rank,
            tp_size=self.transfer_tp_size,
            block_size=self.block_size,
            engine_id=self.engine_id,
            is_mla=self.use_mla,
            total_num_kv_heads=1
            if self.use_mla
            else self.model_config.get_total_num_kv_heads(),
            attn_backends=self.attn_backends,
            dcp_size=self.dcp_size,
            # SSM States come in tuples (ssm, conv)
            tensor_shape=next(iter(kv_caches.values())).shape
            if not self._has_mamba
            else None,
            is_mamba=self._has_mamba,
        )
        self.compat_hash = compute_nixl_compatibility_hash(
            self.vllm_config,
            self._compat_backend_name(),
            transfer_mode=self._TRANSFER_MODE,
        )

        if self._is_csa_linear and self.use_host_buffer:
            raise NotImplementedError(
                "NIXL host staging does not preserve CSA-linear shared tensors."
            )

        if self.use_host_buffer:
            self.initialize_host_xfer_buffer(kv_caches=kv_caches)
            assert len(self.host_xfer_buffers) == len(kv_caches), (
                f"host_buffer: {len(self.host_xfer_buffers)}, "
                f"kv_caches: {len(kv_caches)}"
            )
            xfer_buffers = self.host_xfer_buffers
        else:
            xfer_buffers = kv_caches
            assert not self.host_xfer_buffers, (
                "host_xfer_buffer should not be initialized when "
                f"kv_buffer_device is {self.kv_buffer_device}"
            )

        logger.info(
            "Registering KV_Caches. use_mla: %s, kv_buffer_device: %s, "
            "use_host_buffer: %s",
            self.use_mla,
            self.kv_buffer_device,
            self.use_host_buffer,
        )

        registration_ranges: dict[tuple[int, str], tuple[int, int, int]] = {}
        layout = KVCacheLayout[self.kv_cache_layout].layer_view_order
        region_mem_types: list[str] = []
        base_addrs: list[int] = []
        region_layers: list[list[str]] = []
        regions: list[tuple[int, int, int, int, str]] = []
        if (
            self._has_packed_cache
            and self._supports_pp_hma
            and self.pp_size > 1
            and not self._has_mamba
            and any(
                not isinstance(spec, MLAAttentionSpec)
                for spec in self._layer_specs.values()
            )
        ):
            raise NotImplementedError("PP push with packed KV caches requires MLA")

        # Each layer is a region: an array of kernel blocks. P and D may
        # allocate equivalent layers in different cache-group orders; model-layer
        # order keeps their region lists aligned.
        layer_names = (
            xfer_buffers
            if self._is_csa_linear
            else sorted(xfer_buffers, key=_region_sort_key)
        )
        for layer_name in layer_names:
            cache = xfer_buffers[layer_name]
            # NOTE (NickLucche) Hybrid SSM mamba/FA physical page_size may differ when
            # kernel requires a specific block size. This leads to SSM and FA layers
            # having different num_blocks.
            # `_physical_blocks_per_group` ratio is used to adjust for this.
            layer_spec = self._layer_specs.get(layer_name)
            if layer_spec is None:
                logger.debug(
                    "Skipping layer %s as no KVCache spec is present. "
                    "This is likely because the layer is sharing its KV cache",
                    layer_name,
                )
                continue
            group_id = self.kv_cache_config.transfer_group_index_by_layer[layer_name]
            group = self.kv_cache_config.transfer_groups[group_id]
            physical_blocks = self._physical_blocks_per_group[group_id]
            block_len = layer_spec.page_size_bytes // physical_blocks
            num_blocks = group_num_blocks[group_id] * physical_blocks
            logger.debug(
                "Registering layer %s with cache shape: %s", layer_name, cache.shape
            )
            storage = cache.untyped_storage()
            if cache.device.type == "cpu":
                mem_type = "DRAM"
                region_device_id = 0
            else:
                mem_type = self.nixl_memory_type
                region_device_id = max(cache.get_device(), 0)
                self.device_id = region_device_id
            if group.host_resident:
                registration_base = cache.data_ptr()
                registration_len = cache.nbytes
            else:
                registration_base = storage.data_ptr()
                registration_len = storage.nbytes()
            registration_ranges.setdefault(
                (registration_base, mem_type),
                (
                    registration_base,
                    registration_base + registration_len,
                    region_device_id,
                ),
            )

            # When there's a mismatch between kbs<>bs, we rely on HMA to ensure
            # caches are either [NB, PS] or [NB*r, PS/r] where r is bs/kbs.
            if physical_blocks == 1 and cache.shape[0] < num_blocks:
                raise AssertionError(
                    "KV cache tensor has too few "
                    f"blocks; layer={layer_name}, "
                    f"expected_num_blocks={num_blocks}, "
                    f"cache_shape={tuple(cache.shape)}, "
                    f"cache_stride={tuple(cache.stride())}, "
                    f"layer_spec={type(layer_spec).__name__}, "
                    f"backend={self.backend_name}, "
                    "all_backends="
                    f"{[backend.get_name() for backend in self.attn_backends]}, "
                    f"kv_cache_layout={self.kv_cache_layout}"
                )

            if cache.ndim == 1:
                # Flat byte view: HMA tensors shared between layer types carry
                # no block dimension, so stride(0) is 1 byte. Blocks abut.
                block_stride = block_len
            else:
                block_stride = cache.stride(0) * cache.element_size()
            if cache.ndim == 4 and layer_spec.has_layer_views:
                block = byte_view(cache)
                # A block is one run of bytes unless its heads are apart in
                # memory (LHBNC or BHLNC with several heads): its last byte
                # is then past its `numel`-th.
                last_byte = block[0, -1, -1, -1].storage_offset()
                contiguous = last_byte + 1 == block[0].numel()
            else:
                block = self._transfer_block(layer_spec, group_id, block_len, layout)
                contiguous = True
            self._layer_views[layer_name] = (
                layer_spec.slice_for_tp_transfer(block),
                contiguous,
            )
            # Layers that view the same blocks are one region.
            region = (cache.data_ptr(), block_len, block_stride, group_id, mem_type)
            if region in regions:
                region_layers[regions.index(region)].append(layer_name)
                continue
            regions.append(region)
            region_layers.append([layer_name])
            base_addrs.append(cache.data_ptr())
            self.block_len_per_layer.append(block_len)
            self.block_stride_per_layer.append(block_stride)
            self.region_group_ids.append(group_id)
            self.region_num_blocks.append(num_blocks)
            region_mem_types.append(mem_type)

        logger.debug(
            "Different block lengths collected: %s", set(self.block_len_per_layer)
        )
        self.kv_caches_base_addr[self.engine_id][self.tp_rank] = base_addrs
        self.num_regions = len(base_addrs)
        self.region_mem_types = region_mem_types
        self._set_region_layers(region_layers)
        self._mixed_mem_types = len(set(region_mem_types)) > 1
        self._skip_dram_xfer = self._leaves_dram_xfer_to_host_leader()
        if self._mixed_mem_types:
            assert self.use_mla and not self._has_mamba, (
                "Mixed-device KV registration is only supported for MLA "
                "models without Mamba layers."
            )
        for mem_type in sorted(set(region_mem_types)):
            if mem_type == "DRAM" and self._skip_dram_xfer:
                continue
            ranges_for_mem_type = [
                (start, end - start, device_id, "")
                for (_, cache_mem_type), (
                    start,
                    end,
                    device_id,
                ) in registration_ranges.items()
                if cache_mem_type == mem_type
            ]
            descs = self.nixl_wrapper.get_reg_descs(ranges_for_mem_type, mem_type)
            self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
            self._registered_descs.append(descs)

        self.device_kv_caches = kv_caches
        self.dst_num_blocks[self.engine_id] = self.num_blocks

        # After KV Caches registered, listen for new connections.
        agent_metadata = NixlAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.nixl_wrapper.get_agent_metadata(),
            device_id=self.device_id,
            kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id][self.tp_rank],
            num_blocks=self.num_blocks,
            block_lens=self.block_len_per_layer,
            block_strides=self.block_stride_per_layer,
            kv_cache_layout=self.transfer_kv_cache_layout,
            block_size=self.block_size,
            attn_backend_name=self.backend_name,
            physical_blocks_per_group=self._physical_blocks_per_group,
            region_num_blocks=self.region_num_blocks,
            region_group_ids=self.region_group_ids,
            region_mem_types=self.region_mem_types,
            dcp_size=self.dcp_size,
            pcp_size=self.pcp_size,
            region_members=self.region_members,
            layer_group_ids={
                name: self.kv_cache_config.transfer_group_index_by_layer[name]
                for names in self.region_members
                for name in names
            },
            layer_meta_tensors={
                name: serialize_layer_meta_tensors(views)
                for name, (views, _) in self._layer_views.items()
            },
        )
        # Wrap metadata in payload with hash for defensive decoding
        assert self.compat_hash is not None
        encoder = msgspec.msgpack.Encoder()
        self.xfer_handshake_metadata = NixlHandshakePayload(
            compatibility_hash=self.compat_hash,
            agent_metadata_bytes=encoder.encode(agent_metadata),
        )

    def unregister_kv_caches(self) -> None:
        if self._kv_unregistered or not self._registered_descs:
            return
        self._release_local_xfer_handles()
        # Remote engines hold local handles too; they handshake again on demand.
        for engine_id in list(self._remote_agents):
            self._cleanup_remote_engine(engine_id, log_eviction=False)
        for descs in self._registered_descs:
            self.nixl_wrapper.deregister_memory(descs)
        self._kv_unregistered = True

    def restore_kv_caches(self) -> None:
        if not self._kv_unregistered:
            return
        for descs in self._registered_descs:
            self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
        # Peers load the new registration from the agent metadata.
        payload = self.xfer_handshake_metadata
        assert payload is not None
        agent_metadata = msgspec.msgpack.decode(
            payload.agent_metadata_bytes, type=NixlAgentMetadata
        )
        agent_metadata.agent_metadata = self.nixl_wrapper.get_agent_metadata()
        payload.agent_metadata_bytes = msgspec.msgpack.encode(agent_metadata)
        self._kv_unregistered = False

    @staticmethod
    def _block_strided_descs(
        base_addr: int,
        num_bytes: int,
        unit_len: int,
        block_stride: int,
        num_blocks: int,
        block_size_ratio: int,
        device_id: int,
    ) -> np.ndarray:
        """Nx5 strided descs of `num_bytes` in each of the `block_size_ratio`
        units of `unit_len` bytes that `num_blocks` blocks split into.

        Evenly spaced units collapse into one strided desc. In a padded
        block the units abut only within the block, so each block gets its
        own.
        """
        if block_size_ratio == 1 or block_stride == block_size_ratio * unit_len:
            stride = block_stride // block_size_ratio
            count = num_blocks * block_size_ratio
            return np.array(
                [[base_addr, num_bytes, device_id, stride, count]], dtype=np.uint64
            )
        strided_descs = np.empty((num_blocks, 5), dtype=np.uint64)
        strided_descs[:, 0] = (
            base_addr + np.arange(num_blocks, dtype=np.uint64) * block_stride
        )
        strided_descs[:, 1] = num_bytes
        strided_descs[:, 2] = device_id
        strided_descs[:, 3] = unit_len
        strided_descs[:, 4] = block_size_ratio
        return strided_descs

    @staticmethod
    def _strided_desc_starts(strided_descs: np.ndarray) -> np.ndarray:
        """Flat block index of each strided desc's first block."""
        counts = strided_descs[:, 4]
        return np.cumsum(counts) - counts

    def _prep_descs(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        remote_tp_rank: int,
        remote_agent_name: str,
        block_size_ratio: int,
        stage: _RemoteStage,
    ) -> None:
        """Prepare the local and remote desc lists moving each region pair's
        byte ranges, one strided desc per range and block. Byte ranges of a
        cache group that are contiguous on both sides are merged into a single
        desc (e.g. adjacent TP shards, or layers laid out back to back in a
        block)."""
        pairs = stage.pairs
        remote_split = self._remote_block_splits.get(nixl_agent_meta.engine_id, 1)
        local_regions = list(
            zip(
                self.kv_caches_base_addr[self.engine_id][self.tp_rank],
                self.block_len_per_layer,
                self.block_stride_per_layer,
                self.region_num_blocks,
            )
        )
        num_remote = len(nixl_agent_meta.kv_caches_base_addr)
        remote_regions = list(
            zip(
                nixl_agent_meta.kv_caches_base_addr,
                nixl_agent_meta.block_lens,
                nixl_agent_meta.block_strides,
                stage.region_num_blocks,
            )
        )
        remote_mem_types = (
            nixl_agent_meta.region_mem_types or [self.nixl_memory_type] * num_remote
        )
        # Every range moved, as its address in the first block of each side and
        # then the region and offset it is at there. Ranges are listed with
        # those they can coalesce with: of a remote rank and cache group, in
        # blocks that repeat alike.
        moved: dict[tuple, list[tuple]] = defaultdict(list)
        for (remote, local, group), ranges_by_rank in zip(pairs, stage.byte_ranges):
            ssm = _is_ssm_spec(self._group_spec_types[group])
            units = (1, 1) if ssm else (block_size_ratio, remote_split)
            addr, _, *blocks = local_regions[local]
            remote_addr, _, *remote_blocks = remote_regions[remote]
            mem_types = self.region_mem_types[local], remote_mem_types[remote]
            # A block of several units repeats a range in each: those stay
            # apart.
            alike: tuple = (local, remote)
            if units == (1, 1):
                alike = (*blocks, *remote_blocks, *mem_types)
            for rank, ranges in ranges_by_rank.items():
                for offset, remote_offset, num_bytes in ranges:
                    start = addr + offset, remote_addr + remote_offset
                    at = (local, offset), (remote, remote_offset)
                    moved[rank, group, units, alike].append((*start, num_bytes, *at))
        # Ranges that follow each other on both sides move as one: the shards
        # a rank holds next to each other, and the pages that the layers of a
        # cache group hold back to back in a block-outermost block, which
        # then moves as one desc, not one per layer.
        # (remote rank, cache group, local and remote byte range) of every
        # range moved; the local list holds every rank's.
        xfers = [
            (rank, group, (*ours, n, ratio), (*theirs, n, split))
            for (rank, group, (ratio, split), _), ranges in moved.items()
            for _, _, n, ours, theirs in coalesce_byte_ranges(ranges)
        ]

        local_ranges = tuple(local_range for _, _, local_range, _ in xfers)
        if local_ranges not in self._local_desc_lists:
            descs, starts, row_regions = self._byte_range_descs(
                list(local_ranges), local_regions, self.device_id
            )
            if not self._mixed_mem_types:
                handle = self.nixl_wrapper.prep_xfer_dlist(
                    "NIXL_INIT_AGENT", descs, mem_type=self.region_mem_types[0]
                )
                self._local_desc_lists[local_ranges] = _LocalDescs(handle, starts)
            else:
                # Split by memory type; DRAM descriptors are registered under
                # CPU device 0.
                row_is_dram = np.array(
                    [self.region_mem_types[r] == "DRAM" for r in row_regions],
                    dtype=bool,
                )
                descs[row_is_dram, 2] = 0
                desc_is_dram = np.repeat(row_is_dram, descs[:, 4].astype(np.int64))
                desc_pos = np.empty(len(desc_is_dram), dtype=np.int64)
                for mask in (desc_is_dram, ~desc_is_dram):
                    desc_pos[mask] = np.arange(int(mask.sum()), dtype=np.int64)
                dram_handle = None
                if not self._skip_dram_xfer:
                    dram_handle = self.nixl_wrapper.prep_xfer_dlist(
                        "NIXL_INIT_AGENT", descs[row_is_dram], mem_type="DRAM"
                    )
                handle = self.nixl_wrapper.prep_xfer_dlist(
                    "NIXL_INIT_AGENT",
                    descs[~row_is_dram],
                    mem_type=self.nixl_memory_type,
                )
                self._local_desc_lists[local_ranges] = _LocalDescs(
                    handle, starts, dram_handle, desc_is_dram, desc_pos
                )
        stage.local_descs = local_descs = self._local_desc_lists[local_ranges]

        rank_xfers = [
            (group, local_range, remote_range)
            for rank, group, local_range, remote_range in xfers
            if rank == remote_tp_rank
        ]
        descs, remote_starts, _ = self._byte_range_descs(
            [remote_range for _, _, remote_range in rank_xfers],
            remote_regions,
            nixl_agent_meta.device_id,
        )
        # Per cache group, the first desc id of each of its ranges on each
        # side, so a request's desc ids are one add away from its block ids.
        starts_by_group: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for group, local_range, remote_range in rank_xfers:
            starts_by_group[group].append(
                (local_descs.starts[local_range], remote_starts[remote_range])
            )
        rank_starts = stage.desc_starts[remote_tp_rank] = {}
        for group, group_starts in starts_by_group.items():
            local_ids, remote_ids = np.array(group_starts, dtype=np.int32).T.copy()
            rank_starts[group] = (local_ids, remote_ids)
        paired_mem_types = {remote_mem_types[remote] for remote, _, _ in pairs}
        if len(paired_mem_types) != 1:
            raise NotImplementedError(
                "NIXL pull does not yet support a mixed-memory producer"
            )
        if len(descs):
            handle = self.nixl_wrapper.prep_xfer_dlist(
                remote_agent_name, descs, mem_type=paired_mem_types.pop()
            )
            stage.xfer_side_handles[remote_tp_rank] = handle

    def _split_remote_blocks(self, meta: NixlAgentMetadata) -> None:
        """View remote blocks as gcd-sized kernel blocks when the local block
        size is not a multiple of the remote one.

        Hybrid blocks are sized to cover the TP-sharded SSM page, so P and D
        differ with TP. Backends that accept any kernel block size keep them
        whole, and non-multiples cannot otherwise be paired.
        """
        if self.block_size % meta.block_size == 0:
            return
        unit = math.gcd(self.block_size, meta.block_size)
        split = meta.block_size // unit
        if max(split, self.block_size // unit) > _MAX_HYBRID_BLOCK_SPLIT:
            logger.warning(
                "Not pairing remote %d-token blocks with local %d-token blocks: "
                "their %d-token common unit would split each block into more "
                "than %d transfers.",
                meta.block_size,
                self.block_size,
                unit,
                _MAX_HYBRID_BLOCK_SPLIT,
            )
            return
        # Only token blocks split; recurrent state blocks move whole.
        group_by_layer = self.kv_cache_config.transfer_group_index_by_layer
        if any(
            block_len % split
            for block_len, names in zip(meta.block_lens, meta.region_members)
            if not all(
                _is_ssm_spec(self._group_spec_types[group_by_layer[n]]) for n in names
            )
        ):
            logger.warning(
                "Not pairing remote %d-token blocks with local %d-token blocks: "
                "the remote KV layout cannot be split into %d-token units.",
                meta.block_size,
                self.block_size,
                unit,
            )
            return
        logger.info_once(
            "Transferring remote %d-token blocks as %d-token kernel blocks to "
            "pair them with local %d-token blocks.",
            meta.block_size,
            unit,
            self.block_size,
        )
        meta.block_size //= split
        meta.num_blocks *= split
        meta.physical_blocks_per_group = [
            num * (1 if _is_ssm_spec(self._group_spec_types[group]) else split)
            for group, num in enumerate(meta.physical_blocks_per_group)
        ]
        # An MLA block has one head, so a cut in its tokens is contiguous.
        for name, views in meta.layer_meta_tensors.items():
            if not isinstance(self._layer_specs.get(name), MambaSpec):
                meta.layer_meta_tensors[name] = {
                    component: (
                        offset // split,
                        (blocks, heads, tokens // split, channels),
                        (stride // split, *strides),
                    )
                    for component, (
                        offset,
                        (blocks, heads, tokens, channels),
                        (stride, *strides),
                    ) in views.items()
                }
        self._remote_block_splits[meta.engine_id] = split

    def add_remote_agent(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        remote_tp_rank: int = 0,
        remote_tp_size: int = 1,
        remote_dcp_size: int = 1,
        remote_pp_rank: int = 0,
        remote_pp_size: int = 1,
    ) -> str:
        """Add the remote NIXL agent and prepare the descriptors for reading cache
        blocks from remote.

        A PP-sharded remote registers each stage separately: every stage holds a
        slice of the layers and is read into the local regions holding them.

        In particular, handle both homogeneous and heterogeneous TP. The former
        requires local rank_i to read from remote rank_i.
        The latter, in the case of D.world_size < P.world_size, requires that a
        local (D) TP worker reads from multiple remote (P) TP workers.
        Conversely, assuming D.world_size > P.world_size, two or more local TP
        workers will read from a single remote TP worker.

        Here's an example for the last case described above (non-MLA):

        rank_offset     p_remote_tp_rank
        (kv split no)
        --------------------------------
            0                 0      Worker0  ---- 1st half of KV ----> Worker0  [ KV Cache ]
                                                                        /
            1                 0      Worker1  ---- 2nd half of KV -----/

            0                 1      Worker2  ---- 1st half of KV ----> Worker1  [ KV Cache ]
                                                                        /
            1                 1      Worker3  ---- 2nd half of KV -----/


                                Decoder TP workers                     Prefix TP workers
                                  (world_size=4)                         (world_size=2)
                                                 tp_ratio = 4 // 2 = 2

        Considering the KV Caches, if P-Worker_i has cache size [2, num_blocksP, kv_heads, block_size, head_dim]
        then D-Worker_j has [2, num_blocksD, kv_heads//tp_ratio, block_size, head_dim]. Mind the "LBHNC" layout format.
        Assuming num_blocksD >= num_blocksP, D-Worker0 reads from P-Worker0 by preparing the kv_heads//tp_ratio
        first heads from all the slots of all the blocks. D-Worker1 will do the same, but reading the second split
        along the kv_heads dimension, and so forth until "tp_ratio" D TP workers have pulled from P-Worker0.

        Note that the above will also hold true for the homogeneous TP case, where tp_ratio evaluates to 1.

        Regarding MLA case, the cache is replicated across TP workers so the rank_offset will just always be 0
        so that the whole cache is shared by "tp_ratio" D TP workers.

        For Mamba hetero-TP, both tp_ratio > 0 (D_TP > P_TP) and
        tp_ratio < 0 (P_TP > D_TP) are supported by the 3-read transfer.
        """  # noqa: E501
        engine_id = nixl_agent_meta.engine_id
        # TODO re-evaluate refreshing for scaling/recovery
        remote_ranks = (remote_pp_rank, remote_tp_rank)
        if remote_ranks in self._remote_agents.get(engine_id, {}):
            logger.debug(
                "Remote agent with engine_id %s and rank"
                "%s already exchanged metadata, skip handshake.",
                engine_id,
                remote_ranks,
            )
            return self._remote_agents[engine_id][remote_ranks]

        assert self.transfer_topo is not None
        transfer_topo = self.transfer_topo
        if (
            self._supports_pp_hma
            and self.pp_size > 1
            and (self._is_hma_required or self._has_packed_cache)
        ):
            if self.block_size != nixl_agent_meta.block_size:
                raise NotImplementedError(
                    "Attention-HMA push requires identical P/D block sizes."
                )
            if remote_tp_size > transfer_topo.tp_size and not self.use_mla:
                raise NotImplementedError(
                    "Attention-HMA push does not support decode TP greater than "
                    "prefill TP yet"
                )
        if remote_pp_size > 1:
            check_remote_pp_supported(self, nixl_agent_meta)
        pairs, remote_groups = self._pair_regions(
            nixl_agent_meta, remote_pp_stage=remote_pp_size > 1
        )
        # Every stage adds the remote cache group its groups pair with.
        remote_group_ids = self.remote_group_ids.setdefault(
            engine_id, list(range(len(self._group_spec_types)))
        )
        for group, remote_group in remote_groups.items():
            remote_group_ids[group] = remote_group
        if remote_pp_size > 1 and not self._groups_match(engine_id):
            raise NotImplementedError(
                "Pulling KV from a pipeline-parallel producer requires the same "
                "KV cache groups on both sides."
            )
        # The push WRITE path is not validated with split remote blocks.
        if self._has_mamba and self.use_mla and self._TRANSFER_MODE == "pull":
            self._split_remote_blocks(nixl_agent_meta)
        stage = self._add_remote_stage(
            nixl_agent_meta, pairs, remote_pp_rank, remote_pp_size
        )

        ### Register remote engine in TransferTopology (idempotent).
        transfer_info = EngineTransferInfo(
            remote_tp_size=remote_tp_size,
            remote_block_size=nixl_agent_meta.block_size,
            remote_block_len=nixl_agent_meta.block_lens[0],
            remote_physical_blocks_per_logical=max(
                nixl_agent_meta.physical_blocks_per_group, default=1
            ),
            remote_dcp_size=remote_dcp_size,
        )
        self._remote_physical_blocks_per_group[engine_id] = (
            nixl_agent_meta.physical_blocks_per_group
        )
        transfer_topo.register_remote_engine(engine_id, transfer_info)
        logger.info("Transfer plan: %s", transfer_topo.describe(engine_id))

        remote_agent_name = self.nixl_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata
        )

        self.dst_num_blocks.setdefault(engine_id, nixl_agent_meta.num_blocks)

        # Keep track of remote agent kv caches base addresses.
        self.kv_caches_base_addr[engine_id][remote_tp_rank] = (
            nixl_agent_meta.kv_caches_base_addr
        )
        validate_remote_agent_handshake(
            self, nixl_agent_meta, remote_tp_size, remote_dcp_size
        )

        # The byte ranges of each block moved with each remote rank, per the
        # specs. block_size_ratio > 1 moves each local token block as that
        # many remote ones; recurrent state blocks move whole:
        # remote:               | 0| 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|
        # local origin:|          0|          1|          8|         12|
        # local mapped:| 0| 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|
        block_size_ratio = transfer_topo.block_size_ratio(nixl_agent_meta.block_size)
        if not stage.byte_ranges:
            stage.byte_ranges = self._tp_byte_ranges(
                nixl_agent_meta,
                [(local, group) for _, local, group in pairs],
                remote_tp_size,
                remote_dcp_size,
                receiving=self._TRANSFER_MODE == "pull",
            )
            # Every stage adds the ranks its cache groups move with.
            source_ranks = [set(r) for r in self.source_ranks.get(engine_id, [])]
            source_ranks += [set() for _ in self._group_spec_types[len(source_ranks) :]]
            for (_, _, group), ranges_by_rank in zip(pairs, stage.byte_ranges):
                source_ranks[group].update(ranges_by_rank)
            self.source_ranks[engine_id] = [tuple(sorted(r)) for r in source_ranks]
        self._prep_descs(
            nixl_agent_meta,
            remote_tp_rank,
            remote_agent_name,
            block_size_ratio,
            stage,
        )
        return remote_agent_name

    def _add_remote_stage(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        pairs: list[RegionPair],
        pp_rank: int,
        pp_size: int,
    ) -> _RemoteStage:
        """Register one PP stage of a remote engine (the whole engine when
        `pp_size` is 1) once for all its TP ranks."""
        engine_id = nixl_agent_meta.engine_id
        num_regions = len(nixl_agent_meta.kv_caches_base_addr)
        stage = _RemoteStage(
            pp_rank=pp_rank,
            pairs=pairs,
            layers=sorted(
                {
                    name
                    for members in nixl_agent_meta.region_members
                    for name in members
                    if name in self._region_by_layer
                }
            ),
            region_num_blocks=nixl_agent_meta.region_num_blocks
            or [nixl_agent_meta.num_blocks] * num_regions,
        )
        stages = self._remote_stages.setdefault(engine_id, {})
        # Another TP rank of the stage may have registered it already.
        if stages.setdefault(pp_rank, stage).pairs != pairs:
            raise RuntimeError(
                f"The TP ranks of PP stage {pp_rank} of engine {engine_id} lay "
                "out their KV regions differently."
            )
        # Checked whenever every stage is known, so a retried handshake
        # checks again.
        if len(stages) == pp_size and sorted(
            name for s in stages.values() for name in s.layers
        ) != sorted(self._region_by_layer):
            raise RuntimeError(
                f"The {pp_size} PP stages of engine {engine_id} do not hold "
                "every local KV layer exactly once."
            )
        return stages[pp_rank]

    def _single_remote_stage(self, engine_id: EngineId) -> _RemoteStage:
        """The only stage of a remote engine that runs PP=1."""
        (stage,) = self._remote_stages[engine_id].values()
        return stage

    def sync_recved_kv_to_device(self, req_id: str, meta: ReqMeta):
        """Copy recved kv from host buffer to device."""
        assert self.use_host_buffer
        assert self.copy_blocks is not None

        local_block_ids = meta.local_physical_block_ids
        # Every KV cache group allocates from the same BlockPool, so block ids
        # are unique across groups and the per-group copies can be issued as one.
        block_ids = [block_id for group in local_block_ids for block_id in group]
        # The h2d block copy below is intentionally synchronous.
        with gpu_sync_allowed():
            self.copy_blocks(
                self.host_xfer_buffers,
                self.device_kv_caches,
                block_ids,
                block_ids,
                "h2d",
            )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "synced recved kv of request[%s] to device kv buffer,"
                "local_block_ids: %s. ",
                req_id,
                ",".join(map(str, local_block_ids)),
            )

    def save_kv_to_host(self, metadata: NixlConnectorMetadata):
        """Copy kv from device to host buffer."""
        assert self.use_host_buffer
        assert self.copy_blocks is not None

        # The d2h block copies below are intentionally synchronous.
        with gpu_sync_allowed():
            for req_id, meta in metadata.reqs_to_save.items():
                meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                    meta.local_block_ids, self._physical_blocks_per_group
                )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "save_load_kv for request[%s] to host xfer buffer."
                        "local_block_ids: %s. ",
                        req_id,
                        ",".join(map(str, meta.local_physical_block_ids)),
                    )
                # blocking
                block_ids = [
                    block_id
                    for group in meta.local_physical_block_ids
                    for block_id in group
                ]
                self.copy_blocks(
                    self.device_kv_caches,
                    self.host_xfer_buffers,
                    block_ids,
                    block_ids,
                    "d2h",
                )

    def post_process_device_kv_on_receive(
        self,
        block_size_ratio: int,
        block_ids_list: list[tuple[int, list[int], range]],
        convert: bool = True,
    ):
        """Post process device kv cache after receiving from remote.

        3 types of conversion supported (`convert`):
            * kv_cache_postprocess_layout => convert from HND to NHD
            * kv_cache_postprocess_blksize => convert from small block size
              to large block size
            * kv_cache_postprocess_blksize_and_layout => convert from small
              block size to large block size and convert from HND to NHD

        The transfer only covers `covered_sub_blocks` remote-sized
        sub-blocks of each request's local attention blocks; the rest was
        clipped when pairing local and remote blocks from the last token
        (`_pair_blocks`). Those blocks were
        excluded from the scheduler's alloc-time KV zeroing (which would race
        the RDMA write), so everything outside the covered range is zeroed here.
        Stale bytes would otherwise surface as garbage or NaNs once decode
        grows into the untransferred tail.
        """
        if len(self.device_kv_caches) == 0:
            return
        assert block_size_ratio >= 1, "Only nP < nD supported currently."
        assert self.transfer_topo is not None
        if not convert:
            logger.debug(
                "Post-processing device kv cache on receive by zeroing "
                "untransferred blocks."
            )
        elif self.enable_permute_local_kv and block_size_ratio > 1:
            logger.debug(
                "Post-processing device kv cache on receive by converting "
                "block_size with %sx bigger and permuting layout from HND"
                " to NHD.",
                block_size_ratio,
            )
        elif self.enable_permute_local_kv:
            logger.debug(
                "Post-processing device kv cache on receive by permuting layout"
                "from HND to NHD."
            )
        else:
            logger.debug(
                "Post-processing device kv cache on receive by converting "
                "block_size with %sx bigger.",
                block_size_ratio,
            )

        for group, block_ids, covered_sub_blocks in block_ids_list:
            attn_caches = [
                self.device_kv_caches[name]
                for name in self.kv_cache_config.transfer_groups[group].layer_names
            ]
            # Older call sites pass a covered count; `_pair_blocks`
            # records a range of remote-sized sub-blocks.
            if isinstance(covered_sub_blocks, int):
                covered_sub_blocks = range(covered_sub_blocks)
            # Blocks the transfer didn't write: the token tail of the last
            # partially covered block, then everything beyond it, and the token
            # head of the first one, then everything before it.
            covered_blocks, sub_blocks_in_last = divmod(
                covered_sub_blocks.stop, block_size_ratio
            )
            first_stale = covered_blocks + (1 if sub_blocks_in_last else 0)
            head_blocks, sub_blocks_in_head = divmod(
                covered_sub_blocks.start, block_size_ratio
            )
            has_stale = head_blocks > 0 or first_stale < len(block_ids)
            indices = None
            if convert or has_stale:
                indices = async_tensor_h2d(block_ids, attn_caches[0].device, torch.long)

            if convert:
                for cache in attn_caches:
                    if self.enable_permute_local_kv:
                        kv_postprocess_blksize_and_layout_on_receive(
                            cache, indices, block_size_ratio
                        )
                    elif KVCacheLayout[self.kv_cache_layout].is_block_contiguous:
                        # Token-major blocks get the sub-blocks in token order.
                        kv_postprocess_blksize_on_receive(
                            cache, indices, block_size_ratio
                        )

            if sub_blocks_in_last:
                last_block_id = block_ids[covered_blocks]
                for cache in attn_caches:
                    # Attention caches are [B, H, N, C]: tokens on dim 2.
                    sub_block_tokens = cache.shape[2] // block_size_ratio
                    zero_from = sub_blocks_in_last * sub_block_tokens
                    cache[last_block_id, :, zero_from:].zero_()
            if sub_blocks_in_head:
                for cache in attn_caches:
                    sub_block_tokens = cache.shape[2] // block_size_ratio
                    zero_to = sub_blocks_in_head * sub_block_tokens
                    cache[block_ids[head_blocks], :, :zero_to].zero_()
            if has_stale:
                assert indices is not None
                stale_ids = torch.cat((indices[:head_blocks], indices[first_stale:]))
                for cache in attn_caches:
                    cache.index_fill_(0, stale_ids, 0)

    def post_process_device_kv_on_receive_heterogeneous_attn(
        self, block_ids: list[int]
    ):
        """Post process device kv cache after receiving from remote
        for heterogeneous attention.
        """
        assert self.enable_heterogeneous_attn_post_process

        indices = torch.tensor(block_ids, device=self.device_type, dtype=torch.long)

        for _, cache_or_caches in self.device_kv_caches.items():
            current_platform.pack_kv_cache(
                kv_cache=cache_or_caches,
                indices=indices,
            )

    def _leaves_dram_xfer_to_host_leader(self) -> bool:
        """Whether this rank leaves the DRAM regions' NIXL work to its node's
        HiSparse host leader.

        A HiSparse host pool shared by a node's TP ranks (one mmap per node)
        is registered with NIXL and filled by the node's host leader alone:
        every rank would otherwise register the same pages on every NIC and
        READ the same host KV into them. Private pools are unaffected.
        """
        if not (
            self._mixed_mem_types and self.kv_cache_config.hisparse_shared_host_pool
        ):
            return False
        leader = get_hisparse_host_leader_rank(self.vllm_config, self.tp_rank)
        skip = self.tp_rank != leader
        logger.info(
            "HiSparse host pool: NIXL DRAM registration and reads on host "
            "leader TP rank %d only (tp_rank=%d, skip=%s)",
            leader,
            self.tp_rank,
            skip,
        )
        return skip

    def get_transfer_results(self) -> KVConnectorTransferResults:
        """Get transfers that completed on this specific worker.

        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        assert self.transfer_topo is not None
        done_sending = self._get_new_notifs()
        done_recving, newly_failed = self._pop_done_transfers(self._recving_transfers)
        if newly_failed:
            self._recv_failures.update(newly_failed)

        done_sending.update(self._replicated_pcp_done_sending)
        self._replicated_pcp_done_sending.clear()

        # Process receive failures reported by background threads. Each
        # producer already recorded a specific metric (handshake or
        # notification), so only the failure bookkeeping runs here.
        while not self._failed_recv_reqs.empty():
            try:
                req_id = self._failed_recv_reqs.get_nowait()
            except queue.Empty:
                break
            self._handle_failed_transfer(
                req_id, None, self._recv_failures, record_failed_transfer=False
            )

        failed_recv_reqs: set[ReqId] = set()
        if self._recv_failures:
            # Keep failure bookkeeping independent of healthy request count.
            for req_id in tuple(self._recv_failures):
                if req_id not in self._recving_metadata:
                    self._recv_failures.remove(req_id)
                elif req_id not in self._recving_transfers:
                    self._recv_failures.remove(req_id)
                    failed_recv_reqs.add(req_id)
            done_recving.update(failed_recv_reqs)
        # KV read while the remote may have released the blocks must not be used.
        failed_recv_reqs |= self._get_lease_lapsed_recvs(
            done_recving - failed_recv_reqs
        )

        if len(done_sending) > 0 or len(done_recving) > 0:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving (%s failed)",
                self.tp_rank,
                len(done_sending),
                len(done_recving),
                len(failed_recv_reqs),
            )

        block_ids_for_blocksize_post_process = defaultdict(list)
        block_ids_for_heterogeneous_attn_post_process = list[list[int]]()
        direct_device_recving = set[str]()
        for req_id in done_recving:
            # clean up metadata for completed requests
            meta = self._recving_metadata.pop(req_id, None)
            assert meta is not None, f"{req_id} not found in recving_metadata list"

            # Skip KV sync and post-processing for failed requests
            if req_id in failed_recv_reqs:
                self._pending_recv_notifs.pop(req_id, None)
                if self._report_invalid_block_ids:
                    self._invalid_block_ids.put(set(meta.local_block_ids[0]))
                logger.warning(
                    "Skipping KV post-processing for failed request %s",
                    req_id,
                )
                continue

            self._send_pending_recv_notifs(req_id)
            assert meta.remote is not None
            if self.use_host_buffer:
                self.sync_recved_kv_to_device(req_id, meta)

            direct_device_recving.add(req_id)

            # Post processing for heteroblocksize/layout, and for blocks the
            # transfer clipped. _pair_blocks records the covered
            # sub-block range per request in meta.covered_sub_blocks.
            remote_info = self.transfer_topo.get_engine_info(meta.remote.engine_id)
            block_size_ratio = self.transfer_topo.block_size_ratio(
                remote_info.remote_block_size
            )
            hetero_ppl = (
                remote_info.remote_physical_blocks_per_logical
                != self._physical_blocks_per_logical_kv_block
            )
            if block_size_ratio > 1 or self.enable_permute_local_kv or hetero_ppl:
                for g, local_group in enumerate(meta.local_physical_block_ids):
                    if not local_group or _is_ssm_spec(self._group_spec_types[g]):
                        continue
                    # Range of remote-sized sub-blocks the transfer covered;
                    # everything outside it was clipped and must be zeroed.
                    covered_sub_blocks = (
                        meta.covered_sub_blocks[g]
                        if meta.covered_sub_blocks is not None
                        else range(
                            min(
                                len(local_group) * block_size_ratio,
                                len(meta.remote.block_ids[g]),
                            )
                        )
                    )
                    block_ids_for_blocksize_post_process[block_size_ratio].append(
                        (g, local_group, covered_sub_blocks)
                    )
            # post processing for heterogeneous attention
            if self.enable_heterogeneous_attn_post_process:
                block_ids_for_heterogeneous_attn_post_process.append(
                    meta.local_physical_block_ids[0]
                )
        for (
            block_size_ratio,
            block_ids_list,
        ) in block_ids_for_blocksize_post_process.items():
            # MLA never needs the block-size/layout conversion, but its
            # clipped blocks still need zeroing.
            convert = not self.use_mla and (
                block_size_ratio > 1 or self.enable_permute_local_kv
            )
            self.post_process_device_kv_on_receive(
                block_size_ratio, block_ids_list, convert
            )

        for block_ids in block_ids_for_heterogeneous_attn_post_process:
            self.post_process_device_kv_on_receive_heterogeneous_attn(block_ids)

        self._sync_device_after_direct_recv(direct_device_recving)

        # Handle timeout to avoid stranding blocks on remote.
        self._reap_expired_send_leases(done_sending)

        self._cleanup_remote_engines()
        return KVConnectorTransferResults(
            finished_sending=done_sending,
            finished_recving=done_recving,
            failed_recving=failed_recv_reqs,
        )

    def get_finished(self) -> tuple[set[str], set[str]]:
        """Compatibility wrapper for the legacy completion API."""
        results = self.get_transfer_results()
        return results.finished_sending, results.finished_recving

    def _sync_device_after_direct_recv(self, done_recving: set[str]) -> None:
        """Make direct NIXL writes visible before model execution."""
        requires_sync = current_platform.is_rocm() and self._has_mamba
        if self.use_host_buffer or not done_recving or not requires_sync:
            return

        torch.accelerator.synchronize()

    def _get_new_notifs(self) -> set[str]:
        """Get req_ids which got a remote xfer notification.

        Subclasses must implement this to handle mode-specific notifications.
        """
        raise NotImplementedError

    def _get_lease_lapsed_recvs(self, done_recving: set[ReqId]) -> set[ReqId]:
        """Completed receives whose remote lease may have lapsed mid-read."""
        return set()

    def _reap_expired_send_leases(self, done_sending: set[str]) -> None:
        """Reclaim expired send-side KV leases into `done_sending`.

        `_reqs_to_send` is not ordered by expiry: heartbeats update the
        deadline in place, and mixed TTLs share the map, so a live head
        entry can sit in front of already-expired ones. Scan every entry
        rather than stopping at the first still-live request.
        """
        if not self._reqs_to_send:
            return
        now = time.perf_counter()
        # Blocks still being sent stay leased until their transfer ends.
        expired = [
            req_id
            for req_id, expires in self._reqs_to_send.items()
            if now >= expires and not self._has_inflight_send(req_id)
        ]
        for req_id in expired:
            count = self.consumer_notification_counts_by_req.pop(req_id, 0)
            self.expected_consumer_notifications_by_req.pop(req_id, None)
            self.xfer_stats.record_kv_expired_req()
            logger.warning(
                "Releasing expired KV blocks for request %s which were "
                "retrieved by %d remote worker(s) before lease expired.",
                req_id,
                count,
            )
            self._reqs_to_process.remove(req_id)
            del self._reqs_to_send[req_id]
            done_sending.add(req_id)

    def _has_inflight_send(self, req_id: str) -> bool:
        """Whether this worker is still sending the request's KV blocks."""
        return False

    def _handle_heartbeat(self, payload: str) -> None:
        """Extend leases for requests referenced in a heartbeat.

        Args:
            payload: comma-separated P-side request IDs, e.g.
                     "req_abc,req_def".

        """
        new_expiry = time.perf_counter() + self._lease_extension
        for req_id in payload.split(","):
            if req_id in self._reqs_to_send:
                old = self._reqs_to_send[req_id]
                self._reqs_to_send[req_id] = max(old, new_expiry)
                logger.debug(
                    "Heartbeat extended lease for request %s "
                    "by %ds (old_expiry=%.1f, new_expiry=%.1f)",
                    req_id,
                    self._lease_extension,
                    old,
                    new_expiry,
                )

    def _pop_done_transfers(
        self, transfers: dict[str, list[int]]
    ) -> tuple[set[str], set[str]]:
        """Poll transfers, retaining handles until they can be released.

        Args:
            transfers: Outstanding handles by request, updated in place.

        Returns:
            Requests with no outstanding handles and requests with observed
            failures. Failed requests may still have outstanding handles.

        """
        done_req_ids: set[str] = set()
        failed_req_ids: set[str] = set()
        for req_id, handles in list(transfers.items()):
            in_progress = []
            for handle in handles:
                try:
                    xfer_state = self.nixl_wrapper.check_xfer_state(handle)
                    if xfer_state == "DONE":
                        res = self.nixl_wrapper.get_xfer_telemetry(handle)
                        self.xfer_stats.record_transfer(res)
                        self.nixl_wrapper.release_xfer_handle(handle)
                    elif xfer_state == "PROC":
                        in_progress.append(handle)
                    else:
                        self._log_failure(
                            failure_type="transfer_failed",
                            req_id=req_id,
                            xfer_state=xfer_state,
                        )
                        if not self._handle_failed_transfer(
                            req_id, handle, failed_req_ids
                        ):
                            in_progress.append(handle)
                except Exception as e:
                    self._log_failure(
                        failure_type="transfer_exception",
                        req_id=req_id,
                        error=e,
                    )
                    if not self._handle_failed_transfer(
                        req_id, handle, failed_req_ids, error=e
                    ):
                        in_progress.append(handle)

            if not in_progress:
                done_req_ids.add(req_id)
                del transfers[req_id]
            else:
                transfers[req_id] = in_progress
        return done_req_ids, failed_req_ids

    def _try_release_xfer_handle(self, req_id: str, handle: int) -> bool:
        """Release a handle, returning False if the caller must retain it."""
        try:
            self.nixl_wrapper.release_xfer_handle(handle)
        except Exception as e:
            # A status error does not guarantee that the backend stopped DMA.
            self._log_failure(
                failure_type="transfer_release_failed",
                msg="Retaining handle and blocks until release succeeds",
                req_id=req_id,
                error=e,
            )
            return False
        return True

    def _handle_failed_transfer(
        self,
        req_id: str,
        handle: int | None,
        failed_req_ids: set[str] | None = None,
        record_failed_transfer: bool = True,
        error: BaseException | None = None,
    ) -> bool:
        """Record a failure and release its handle, returning False to retain it.

        Args:
            req_id: The request ID.
            handle: The transfer handle.
            failed_req_ids: Requests observed as failed; each stays listed
                until it has no outstanding handles.
            record_failed_transfer: Whether to count the failure toward the
                transport-failure metric. Callers that already recorded a more
                specific metric (eg KV expiry, reported separately from
                transport failures) pass False.
            error: The exception that failed the transfer, if any. A lost
                connection invalidates the remote engine.

        """
        if record_failed_transfer:
            self.xfer_stats.record_failed_transfer()
            if (
                self._TRANSFER_MODE == "pull"
                and (meta := self._recving_metadata.get(req_id)) is not None
                and meta.remote is not None
            ):
                self._failed_remote_engines.add(meta.remote.engine_id)
                if nixlRemoteDisconnectError is not None and isinstance(
                    error, nixlRemoteDisconnectError
                ):
                    self._disconnected_remote_engines.add(meta.remote.engine_id)
        if failed_req_ids is not None:
            failed_req_ids.add(req_id)
        return handle is None or self._try_release_xfer_handle(req_id, handle)

    def _send_pending_recv_notifs(self, req_id: str) -> None:
        """Send notifications deferred by split DRAM/VRAM reads."""
        for agent_name, notif_id in self._pending_recv_notifs.pop(req_id, []):
            try:
                self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
            except Exception as e:
                self._log_failure(
                    failure_type="notification_failed",
                    msg="P worker blocks will be freed after timeout. "
                    "This may indicate network issues.",
                    req_id=req_id,
                    error=e,
                    remote_agent_name=agent_name,
                )
                self.xfer_stats.record_failed_notification()

    def _send_heartbeats(self, metadata: NixlConnectorMetadata) -> None:
        """Send heartbeats to remote engines, extending the lease on KV blocks."""
        now = time.perf_counter()
        for engine_id, hb_info in metadata.heartbeat_by_engine.items():
            # Proactive handshake (this request may still be in waiting queue) so
            # the **next** heartbeat for this remote can go through.
            if (
                self._ensure_handshake(
                    engine_id,
                    hb_info.host,
                    hb_info.port,
                    hb_info.tp_size,
                    hb_info.dcp_size,
                    hb_info.pp_size,
                    self._hb_handshake_notif_only and hb_info.pp_size > 1,
                )
                is not None
            ):
                continue  # handshake is still pending

            # Refresh the TTL: a heartbeat means the router still paired this P-D,
            # so requests are still waiting on this engine
            self._engine_last_active[engine_id] = time.perf_counter()
            hb_msg = (HB_NOTIF_PREFIX + ",".join(hb_info.req_ids)).encode()
            all_sent = True
            for agent_name in self._remote_agents[engine_id].values():
                try:
                    self.nixl_wrapper.send_notif(agent_name, notif_msg=hb_msg)
                except Exception:
                    all_sent = False
                    logger.debug(
                        "Failed to send heartbeat to engine %s",
                        engine_id,
                        exc_info=True,
                    )
            clock_offset = self._engine_clock_offset.get(engine_id)
            if all_sent and clock_offset is not None:
                # Only count rounds that reached every remote rank.
                for remote_req_id, (expiry, extension) in hb_info.leases.items():
                    if "," in remote_req_id:
                        # Heartbeats are comma-separated: P can't match this id.
                        continue
                    self._record_heartbeat_sent(
                        (engine_id, remote_req_id),
                        now,
                        expiry - clock_offset,
                        extension,
                    )
        if metadata.heartbeat_by_engine:
            # Expired leases no longer matter: the read is declined either way.
            self._remote_leases = {
                k: d for k, d in self._remote_leases.items() if d > now
            }

    def _record_heartbeat_sent(
        self, key: tuple[EngineId, str], now: float, deadline: float, extension: int
    ) -> None:
        """Mirror the remote's `_handle_heartbeat` for a heartbeat sent at `now`:
        a live lease (initial `deadline`, our clock) renews to `now + extension`;
        a lapsed one stays lapsed, as its deadline no longer moves."""
        lease = self._remote_leases.setdefault(key, deadline)
        if now + self._lease_safety_margin <= lease:
            self._remote_leases[key] = max(lease, now + extension)

    def _pair_blocks(
        self,
        engine_id: EngineId,
        remote_rank: int,
        group: int,
        local_block_ids: Sequence[int],
        remote_group: int,
        remote_block_ids: Sequence[int],
        *,
        num_computed_blocks: int | None = None,
        num_tokens: int | None = None,
    ) -> tuple[list[int], list[int], range]:
        """Pair the transfer blocks a cache group holds of a request with
        those a remote rank holds of it: those at the same position in the
        request. A transfer block is a remote kernel block, and a kernel
        block of ours is `block_size_ratio` of them.

        `local_block_ids` start `num_computed_blocks` into this rank's
        blocks of the request. Without that count the lists are not shared
        out by DCP and positions count back from the last token, whose block
        ends each list: a prefix hit here or sliding-window clipping there
        can start the two at different tokens.

            ours             | a | b | c |      after a prefix hit on two
            theirs   | p | q | r | s | t |      pairs (a, r) (b, s) (c, t)

        Given `num_tokens`, the transfer blocks allocated past it stay
        behind.

        Returns the paired lists and the range they cover of our transfer
        blocks that the remote rank should hold, which starts past 0 when it
        lacks some.
        """
        assert self.transfer_topo is not None
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        ratio = self.transfer_topo.block_size_ratio(remote_info.remote_block_size)
        per_block = self._physical_blocks_per_group[group] * ratio
        remote_per_block = self._remote_physical_blocks_per_group[engine_id][
            remote_group
        ]
        local = expand_block_ids(local_block_ids, ratio)
        remote = np.asarray(remote_block_ids, dtype=np.int64)
        end = None
        if num_tokens is not None:
            end = cdiv(num_tokens, remote_info.remote_block_size)
        if num_computed_blocks is None:
            remote_dcp_size, remote_dcp_rank = 1, 0
            positions = np.arange(-len(local), 0) + round_up(end or 0, per_block)
            remote_positions = np.arange(-len(remote), 0) + round_up(
                end or 0, remote_per_block
            )
        else:
            remote_dcp_size = remote_info.remote_dcp_size
            remote_dcp_rank = remote_rank % remote_dcp_size
            positions = transfer_block_positions(
                len(local),
                per_block,
                dcp_size=self.dcp_size,
                dcp_rank=self.dcp_rank,
                num_computed_blocks=num_computed_blocks,
            )
            remote_positions = transfer_block_positions(
                len(remote),
                remote_per_block,
                dcp_size=remote_dcp_size,
                dcp_rank=remote_dcp_rank,
            )
        # Ours that the remote rank holds too, and its own, within the request.
        wanted = positions // per_block % remote_dcp_size == remote_dcp_rank
        available = np.ones(len(remote), dtype=bool)
        if end is not None:
            wanted &= positions < end
            available &= remote_positions < end
        # The same position in the request: the same transfer block.
        _, ours, theirs = np.intersect1d(
            positions[wanted], remote_positions[available], return_indices=True
        )
        num_wanted = int(wanted.sum())
        return (
            local[wanted][ours].tolist(),
            remote[available][theirs].tolist(),
            range(num_wanted - len(ours), num_wanted),
        )

    @staticmethod
    def _logical_to_kernel_block_ids(
        block_ids: BlockIds, physical_blocks_per_group: Sequence[int]
    ) -> BlockIds:
        """Expand each group's block ids to the kernel blocks they span, e.g.
        `[1, 3]` at 2 kernel blocks per block to `[2, 3, 6, 7]`."""
        if all(num == 1 for num in physical_blocks_per_group):
            return block_ids
        return [
            expand_block_ids(group, num).tolist()
            for group, num in zip(block_ids, physical_blocks_per_group)
        ]

    def _apply_prefix_caching(
        self,
        decode_block_ids: BlockIds,
        prefill_block_ids: BlockIds,
        decode_physical_per_logical: int,
        prefill_physical_per_logical: int,
        prefill_group_ids: Sequence[int] | None = None,
    ) -> tuple[BlockIds, BlockIds]:
        """Trim block ID lists so only the uncomputed suffix is transferred.

        Inputs are *kernel* (physical) block IDs, already expanded from logical IDs
        with each side's physical-per-logical ratio. Both pull and push call this after
        that expansion so the trim happens at kernel granularity.

        The prefix-cache hit is always on the decode (D) side, so `decode`
        holds only its uncomputed blocks while prefill (P) holds the full
        sequence. This is mode-independent: pull passes its own (D) blocks as
        `decode`, push passes the remote D registration as `decode`.

        For non-Mamba models: end-trim `prefill` to match `decode` count, so
        already-cached prefix blocks are skipped in the transfer.

        For Mamba hybrid: SSM groups pair state slots by position and FA
        groups end-trim to the uncomputed suffix when physical-per-logical
        matches.

        Each decode group `i` pairs with prefill group `prefill_group_ids[i]`,
        the one holding the same layers on that side (`_pair_regions`); the
        same group by default. The returned lists are indexed by decode group.
        """
        if prefill_group_ids is None:
            prefill_block_ids = list(prefill_block_ids)
        else:
            prefill_block_ids = [prefill_block_ids[g] for g in prefill_group_ids]
        # Partial prefix cache hit: just transfer uncomputed blocks.
        # Skip mamba groups — their blocks represent full state (conv+ssm),
        # not per-token data, so trimming would corrupt the transfer.
        if not self._has_mamba:
            for i, prefill_group in enumerate(prefill_block_ids):
                num_decode_blocks = len(decode_block_ids[i])
                assert num_decode_blocks <= len(prefill_group)
                if num_decode_blocks < len(prefill_group):
                    prefill_block_ids[i] = prefill_group[-num_decode_blocks:]
        else:
            # (NOTE: ZhanqiuHu) HeteroTP can cause different kernel block counts
            # due to logical block rounding.
            # Example: 640 prompt tokens, kernel_block_size=64
            #   prefill physical_per_logical=10, decode physical_per_logical=6
            #   prefill logical ids from kv_transfer_params = [0]
            #   decode logical ids allocated = [0, 1]
            #   prefill kernel blocks: [0..9]  (1*10=10)
            #   decode kernel blocks:  [0..11] (2*6=12)
            #   actual data blocks = ceil(640/64) = 10, trim both to 10
            # Vice versa (prefill physical_per_logical=6, decode=10):
            #   prefill logical ids = [0, 1], decode logical ids = [0]
            #   prefill kernel blocks: [0..11] (2*6=12)
            #   decode kernel blocks:  [0..9]  (1*10=10)
            #   actual data blocks = ceil(640/64) = 10, trim both to 10
            decode_block_ids = list(decode_block_ids)
            for i, prefill_group in enumerate(prefill_block_ids):
                num_decode_blocks = len(decode_block_ids[i])
                num_prefill_blocks = len(prefill_group)
                if _is_ssm_spec(self._group_spec_types[i]):
                    if num_decode_blocks == num_prefill_blocks:
                        continue
                    # Only state-bearing slots reach here, normally just one
                    # (see get_exchange_clipped_blocks). A longer prefill list
                    # carries earlier positions the decode side already has
                    # (prefix hit) -> take its tail; a longer decode list holds
                    # the position D recomputes itself, which gets no prefill
                    # state.
                    assert num_decode_blocks - num_prefill_blocks <= 1, (
                        f"Group {i}: unpairable SSM state slots, "
                        f"decode={num_decode_blocks} prefill={num_prefill_blocks}"
                    )
                    num_blocks = min(num_decode_blocks, num_prefill_blocks)
                    if num_decode_blocks < num_prefill_blocks:
                        prefill_block_ids[i] = prefill_group[-num_blocks:]
                    else:
                        decode_block_ids[i] = decode_block_ids[i][:num_blocks]
                elif (
                    decode_physical_per_logical == prefill_physical_per_logical
                    and num_decode_blocks < num_prefill_blocks
                ):
                    # Partial prefix cache hit for FA group.
                    prefill_block_ids[i] = prefill_group[-num_decode_blocks:]
                else:
                    # TODO Handle prefix caching with different block_sizes
                    # Allocation rounding legitimately leaves up to
                    # ppl - 1 trailing dead kernel blocks per side (plus one
                    # extra decode block for the recomputed final token), so
                    # the counts may differ by up to the sum of the two
                    # ratios; anything larger indicates mismatched lists.
                    max_padding = (
                        decode_physical_per_logical + prefill_physical_per_logical
                    )
                    assert abs(num_decode_blocks - num_prefill_blocks) <= max_padding, (
                        f"Group {i}: |{num_decode_blocks} - "
                        f"{num_prefill_blocks}| > {max_padding}"
                    )
                    num_blocks = min(num_decode_blocks, num_prefill_blocks)
                    decode_block_ids[i] = decode_block_ids[i][:num_blocks]
                    prefill_block_ids[i] = prefill_group[:num_blocks]
        return decode_block_ids, prefill_block_ids

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """Get the KV transfer stats for the connector."""
        # Clear stats for next iteration
        if not self.xfer_stats.is_empty():
            return self.xfer_stats.clone_and_reset()
        return None

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return and clear the set of block IDs that failed to load.

        This is called by the scheduler to identify blocks that need
        to be retried after a NIXL transfer failure.
        """
        # Drain the queue (thread-safe, no lock needed).
        result: set[int] = set()
        while not self._invalid_block_ids.empty():
            try:
                result.update(self._invalid_block_ids.get_nowait())
            except queue.Empty:
                break
        return result

    def _evict_stale_engines(self) -> None:
        """Scan for and evict remote engines that have exceeded their TTL.

        Called from the main thread in when a new remote engine appears.
        We can only go OOM as we discover and register a new remote, therefore we make
        sure we clean up stale engine data structures before then. This invariant
        prevents us from using background threads, though memory usage is not guaranteed
        to be "optimal" until a new handshake is performed.

        Engines with active transfers, heartbeats or pending handshakes cannot
        be stale:
        - Reads stamp _engine_last_active when they are issued, and engines a
          transfer is still reading from are held back explicitly, since the
          stamp is not refreshed while the read runs.
        - Heartbeats stamp it too, so an engine stays registered while this
          instance holds requests waiting on it.
        - Pending handshakes don't have an _engine_last_active entry yet
        """
        # NOTE (NickLucche): This does NOT currently prevent OOMing if a huge number
        # of remote engines is registered all at once (adding a background cleanup
        # thread wouldnt help either).
        # If that scenario is plausible, we can follow up with an LRU eviction policy.
        if self._engine_ttl <= 0:
            return

        now = time.perf_counter()
        stale = [
            eid
            for eid, last_active in list(self._engine_last_active.items())
            if now - last_active > self._engine_ttl
        ]
        if not stale:
            return
        busy = self._engines_with_inflight_transfers()
        for eid in stale:
            if eid not in busy:
                self._cleanup_remote_engine(eid)

    def _engines_with_inflight_transfers(self) -> set[EngineId]:
        """Remote engines a transfer is still reading from.

        The timestamp is stamped when a read is issued and not refreshed while
        it runs, so a transfer that outlives the TTL leaves its engine looking
        idle. A peer that has lost its NIC holds one indefinitely.
        """
        return {
            meta.remote.engine_id
            # Snapshot: push evicts from its writer thread too.
            for req_id in list(self._recving_transfers)
            if (meta := self._recving_metadata.get(req_id)) is not None
            and meta.remote is not None
        }

    def _track_remote_engine_replacement(
        self, engine_id: EngineId, host: str, port: int
    ) -> None:
        """Record the engine currently serving a pull peer address."""
        # TODO: Also handle push mode, which handshakes in both directions.
        if self._TRANSFER_MODE != "pull":
            return
        previous = self._engine_by_address.get((host, port))
        self._engine_by_address[(host, port)] = engine_id
        if previous is None or previous == engine_id:
            return
        logger.info(
            "Remote engine at %s:%d changed from %s to %s.",
            host,
            port,
            previous,
            engine_id,
        )

    def _cleanup_remote_engines(self) -> None:
        """Release replaced or locally invalid pull peers after their reads drain."""
        # TODO: Also handle push mode, which handshakes in both directions.
        if self._TRANSFER_MODE != "pull":
            return
        # A background handshake may be loading native metadata with the GIL
        # released. Defer both probing and cleanup until it has finished.
        with self._handshake_lock:
            if self._handshake_futures:
                return
            replaced = self._remote_agents.keys() - self._engine_by_address.values()
        now = time.perf_counter()
        for engine_id in self._failed_remote_engines - self._invalid_remote_engines:
            if engine_id in self._disconnected_remote_engines:
                # Skip an engine whose agents were already removed (no NIXL state left).
                if engine_id in self._remote_agents:
                    # The metadata may still be loaded; the connection is not.
                    logger.warning(
                        "NIXL lost the connection to remote engine %s.", engine_id
                    )
                    self._mark_remote_engine_invalid(engine_id, now)
                continue
            agents = self._remote_agents.get(engine_id, {}).values()
            try:
                # No descriptors: query local existence, not peer health.
                if not all(map(self.nixl_wrapper.check_remote_metadata, agents)):
                    self._mark_remote_engine_invalid(engine_id, now)
            except Exception:
                logger.warning(
                    "Could not check local NIXL metadata for engine %s.",
                    engine_id,
                    exc_info=True,
                )
        self._failed_remote_engines.clear()
        self._disconnected_remote_engines.clear()
        # Invalid peers keep rejecting new reads until their backoff has elapsed.
        invalid = {
            engine_id
            for engine_id in self._invalid_remote_engines
            if self._invalid_engine_backoff.get(engine_id, (0.0, 0.0))[1] <= now
        }
        if not replaced and not invalid:
            return
        # _recving_metadata is only accessed from this (main) thread.
        busy = {
            meta.remote.engine_id
            for meta in self._recving_metadata.values()
            if meta.remote is not None
        }
        # Replaced peers can still serve queued requests. Invalid peers reject
        # new reads, but must retain descriptors until outstanding handles end.
        invalid -= self._engines_with_inflight_transfers()
        for engine_id in (replaced - busy) | invalid:
            self._cleanup_remote_engine(engine_id, log_eviction=False)
            logger.info(
                "Released NIXL state for replaced or locally invalid remote engine %s.",
                engine_id,
            )

    def _mark_remote_engine_invalid(self, engine_id: EngineId, now: float) -> None:
        """Schedule release of an engine whose local NIXL metadata is gone.

        The first invalidation is released as soon as reads drain. Repeats
        before the peer has stayed healthy for the max backoff double the
        delay, during which reads fail fast instead of re-handshaking.
        """
        # Forget engines that have stayed healthy long enough.
        for eid, (_, release_at) in list(self._invalid_engine_backoff.items()):
            if now - release_at > _INVALID_PEER_MAX_BACKOFF_S:
                del self._invalid_engine_backoff[eid]
        prev = self._invalid_engine_backoff.get(engine_id)
        backoff = 0.0
        if prev is not None:
            backoff = min(
                max(2 * prev[0], _INVALID_PEER_MIN_BACKOFF_S),
                _INVALID_PEER_MAX_BACKOFF_S,
            )
        self._invalid_engine_backoff[engine_id] = (backoff, now + backoff)
        self._invalid_remote_engines.add(engine_id)
        logger.warning(
            "Local NIXL metadata for remote engine %s was invalidated; "
            "re-handshaking after reads drain%s.",
            engine_id,
            f" and a {backoff:.0f}s backoff" if backoff else "",
        )

    def _cleanup_remote_engine(
        self, engine_id: EngineId, *, log_eviction: bool = True
    ) -> None:
        """Remove all state for a single remote engine.

        Releases NIXL resources (dlist handles, remote agents) and clears
        all per-engine data structures. Used by TTL eviction, replaced/invalid
        peer cleanup and shutdown.
        """
        assert engine_id in self._remote_agents

        # Notif-only engines (push-mode D side) have no stages.
        for stage in self._remote_stages.pop(engine_id, {}).values():
            for handle in stage.xfer_side_handles.values():
                self.nixl_wrapper.release_dlist_handle(handle)
        # Pop under the handshake lock; NIXL teardown stays outside it.
        with self._handshake_lock:
            agents = self._remote_agents.pop(engine_id)
            for address, eid in list(self._engine_by_address.items()):
                if eid == engine_id:
                    del self._engine_by_address[address]
        for agent_name in agents.values():
            self.nixl_wrapper.remove_remote_agent(agent_name)

        self._failed_remote_engines.discard(engine_id)
        self._disconnected_remote_engines.discard(engine_id)
        self._invalid_remote_engines.discard(engine_id)

        self.kv_caches_base_addr.pop(engine_id, None)
        self.dst_num_blocks.pop(engine_id, None)
        self.source_ranks.pop(engine_id, None)
        self.remote_group_ids.pop(engine_id, None)
        self._remote_physical_blocks_per_group.pop(engine_id, None)
        self._remote_block_splits.pop(engine_id, None)
        self._push_senders.pop(engine_id, None)
        if self.transfer_topo is not None:
            self.transfer_topo.unregister_remote_engine(engine_id)

        # Drop the cached clock offset; it is re-measured on the next handshake.
        self._engine_clock_offset.pop(engine_id, None)
        # A just-completed handshake may not have recorded activity yet, so
        # tolerate a missing entry.
        last_active = self._engine_last_active.pop(engine_id, None)
        if log_eviction and last_active is not None:
            logger.info(
                "Evicted stale remote engine %s (inactive for %.1fs).",
                engine_id,
                time.perf_counter() - last_active,
            )

    def __del__(self):
        with contextlib.suppress(Exception):
            self.shutdown()

    def _release_local_xfer_handles(self) -> None:
        try:
            for descs in self._local_desc_lists.values():
                for handle in (descs.handle, descs.dram_handle):
                    if handle is not None:
                        self.nixl_wrapper.release_dlist_handle(handle)
        finally:
            self._local_desc_lists.clear()

    def _finish_shutdown(self) -> None:
        self._recving_transfers.clear()
        try:
            self._release_local_xfer_handles()
        except Exception:
            logger.exception("NIXL dlist-handle release failed at shutdown.")
        try:
            for engine_id in list(self._remote_agents):
                self._cleanup_remote_engine(engine_id, log_eviction=False)
        except Exception:
            logger.exception("NIXL remote-engine cleanup failed at shutdown.")
        try:
            if not self._kv_unregistered:
                for desc in self._registered_descs:
                    self.nixl_wrapper.deregister_memory(desc)
        finally:
            self._registered_descs.clear()
            # Drop cache references before their owners release registered
            # host memory; handshake futures may outlive model-runner shutdown.
            self.device_kv_caches = {}
            self.host_xfer_buffers = {}

    def shutdown(self) -> None:
        """Shutdown the connector worker."""
        if not hasattr(self, "_handshake_initiation_executor"):
            # error happens during init, no need to shutdown
            return
        self._handshake_initiation_executor.shutdown(wait=False)
        for handles in self._recving_transfers.values():
            for handle in handles:
                self.nixl_wrapper.release_xfer_handle(handle)
        self._recving_transfers.clear()
        self._finish_shutdown()
