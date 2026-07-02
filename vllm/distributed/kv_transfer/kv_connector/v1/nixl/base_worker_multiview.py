# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-view descriptor layer for the NIXL connector.

Replaces the per-spec builder methods (_build_fa_local/remote,
_build_mamba_local/remote) with a unified DescriptorView abstraction
that delegates layout decisions to KVCacheSpec.transfer_shapes() and
KVCacheSpec.slice_for_tp_transfer().

Wire-in: change the import in pull_worker.py / push_worker.py to use
NixlBaseConnectorWorkerMultiview as the base class.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    EngineTransferInfo,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
    NixlHandshakePayload,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    TPMapping,
    compute_tp_mapping,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

# 4D dim index for the batch/block dimension.
_DIM4_B = 0


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------


def build_region_meta(
    spec: KVCacheSpec,
    num_blocks: int,
    block_size: int,
    block_stride_bytes: int,
    region_content_bytes: int,
    virtually_split: bool = False,
) -> torch.Tensor:
    """Build a ``(B, H, N, C)`` meta tensor for one KV cache region.

    Shape is derived from ``spec.compute_transfer_shape()`` which knows the
    logical layout (heads, tokens, head_dim) for each spec type.

    ``slice_for_tp_transfer`` later narrows/splits this into descriptor streams.
    ``_view_to_descriptors`` consumes only ``stride(0)``, ``shape``, and
    ``storage_offset``.
    """
    dtype = getattr(spec, "dtype", torch.int8)
    elem = get_dtype_size(dtype)
    H, N, C = spec.compute_transfer_shape(
        region_content_bytes, block_size, virtually_split
    )

    meta = torch.as_strided(
        torch.empty(1, dtype=dtype, device="meta"),
        size=(num_blocks, H, N, C),
        stride=(block_stride_bytes // elem, N * C, C, 1),
        storage_offset=0,
    )
    return meta


@dataclass
class DescriptorView:
    """One descriptor view in the flat descriptor list.

    Each view owns a contiguous slice of the descriptor space.
    Layout: [view0 descs | view1 descs | ...].

    Per-region strides and contents are pre-computed at construction
    so that builders can consume them without branching on spec type.

    The number of descriptor streams per region is determined dynamically
    by len(spec.slice_for_tp_transfer(...)) — not pre-computed.
    """

    spec: KVCacheSpec
    region_indices: list[int]
    strides: list[int]
    contents: list[int]
    num_blocks: int
    descs_per_region: int
    virtually_split: bool

    @property
    def num_view_regions(self) -> int:
        return len(self.region_indices) * self.descs_per_region

    def num_descs(self) -> int:
        return self.num_view_regions * self.num_blocks

    def remote_num_blocks(
        self,
        remote_kernel_blocks: int,
        remote_physical_per_logical: int,
    ) -> int:
        if isinstance(self.spec, MambaSpec):
            return remote_kernel_blocks // remote_physical_per_logical
        return remote_kernel_blocks

    def remote_stride(
        self, remote_block_len: int, remote_physical_per_logical: int
    ) -> int:
        if isinstance(self.spec, MambaSpec):
            return remote_block_len * remote_physical_per_logical
        return remote_block_len

    def remote_content(self, remote_block_len: int, remote_ssm_content: int) -> int:
        if isinstance(self.spec, MambaSpec):
            return remote_ssm_content
        return remote_block_len


# ---------------------------------------------------------------------------
# Multi-view subclass
# ---------------------------------------------------------------------------


class NixlBaseConnectorWorkerMultiview(NixlBaseConnectorWorker):
    """Extends NixlBaseConnectorWorker with unified DescriptorView logic.

    Overrides:
      - register_kv_caches  (adds view construction + block_stride tracking)
      - _compute_desc_ids   (view-based offset computation)
      - register_local_xfer_handler  (delegates to _build_view_local)
      - add_remote_agent    (delegates to _build_view_remote)
      - _build_local_splits_from_plan (view-aware splitting)
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        self._views: list[DescriptorView] = []
        self._group_to_view_idx: dict[int, int] = {}
        # Per-region physical stride (bytes between consecutive blocks).
        # Populated in register_kv_caches.
        self.block_stride_per_layer: list[int] = []
        # Per-region spec, for slice_for_tp_transfer dispatch.
        self.spec_per_region: list[KVCacheSpec] = []

    # ------------------------------------------------------------------
    # Descriptor ID computation
    # ------------------------------------------------------------------

    def _compute_desc_ids(
        self,
        block_ids: BlockIds,
        dst_num_blocks: int,
        block_size_ratio: float | None,
        physical_blocks_per_logical: int,
    ) -> np.ndarray:
        """Compute NIXL descriptor IDs using per-view offsets."""
        kernel_blocks = dst_num_blocks
        if block_size_ratio is not None:
            kernel_blocks = int(kernel_blocks * block_size_ratio)

        view_offsets: list[int] = []
        view_block_counts: list[int] = []
        cumulative_offset = 0
        for view in self._views:
            view_offsets.append(cumulative_offset)
            view_block_counts.append(
                view.remote_num_blocks(kernel_blocks, physical_blocks_per_logical)
            )
            cumulative_offset += view.num_view_regions * view_block_counts[-1]

        all_descs: list[np.ndarray] = []
        for group_idx, group_block_ids in enumerate(block_ids):
            if group_idx not in self._group_to_view_idx:
                raise ValueError(
                    f"Unknown spec type {self._group_spec_types[group_idx]} "
                    f"at group index {group_idx}"
                )
            view_idx = self._group_to_view_idx[group_idx]
            all_descs.append(
                (
                    view_offsets[view_idx]
                    + np.arange(self._views[view_idx].num_view_regions)[:, None]
                    * view_block_counts[view_idx]
                    + np.asarray(group_block_ids)[None, :]
                ).flatten()
            )

        return np.concatenate(all_descs)

    # ------------------------------------------------------------------
    # View construction
    # ------------------------------------------------------------------

    def _build_descriptor_views(
        self,
    ) -> tuple[list[DescriptorView], dict[int, int]]:
        """Create one DescriptorView per KV cache group.

        Returns (views, group_to_view_idx).
        Per-region strides and contents are pre-computed here so that
        builders can consume them without branching on spec type.

        descs_per_region is determined by calling slice_for_tp_transfer
        with identical local TP params (homogeneous case) to count the
        number of descriptor streams.
        """
        assert self.transfer_topo is not None
        all_indices = list(range(len(self.spec_per_region)))

        views: list[DescriptorView] = []
        group_to_view: dict[int, int] = {}

        for group_idx, group in enumerate(self.kv_cache_config.kv_cache_groups):
            group_spec = group.kv_cache_spec
            if isinstance(group_spec, UniformTypeKVCacheSpecs):
                representative_spec = next(iter(group_spec.kv_cache_specs.values()))
            else:
                representative_spec = group_spec

            if isinstance(representative_spec, MambaSpec):
                strides = [
                    self.block_len_per_layer[i]
                    * self._physical_blocks_per_logical_kv_block
                    for i in all_indices
                ]
                contents = [sum(self._mamba_ssm_size) for _ in all_indices]
                num_blocks = self._logical_num_blocks
                view_virtually_split = False
            elif isinstance(representative_spec, AttentionSpec):
                strides = [self.block_stride_per_layer[i] for i in all_indices]
                contents = [self.block_len_per_layer[i] for i in all_indices]
                num_blocks = self.num_blocks
                view_virtually_split = self.transfer_topo.virtually_split_kv_in_blocks
            else:
                raise ValueError(
                    f"Unsupported spec type for DescriptorView: "
                    f"{type(representative_spec).__name__}"
                )

            # Determine descs_per_region by doing a probe slice with local TP.
            probe_meta = build_region_meta(
                spec=representative_spec,
                num_blocks=1,
                block_size=self.block_size,
                block_stride_bytes=strides[0],
                region_content_bytes=contents[0],
                virtually_split=view_virtually_split,
            )
            probe_slices = representative_spec.slice_for_tp_transfer(
                probe_meta,
                self.world_size,
                self.tp_rank,
                self.world_size,
                self.tp_rank,
                self.model_config,
                virtually_split=view_virtually_split,
            )
            descs_per_region = len(probe_slices)

            views.append(
                DescriptorView(
                    spec=representative_spec,
                    region_indices=all_indices,
                    strides=strides,
                    contents=contents,
                    num_blocks=num_blocks,
                    descs_per_region=descs_per_region,
                    virtually_split=view_virtually_split,
                )
            )
            group_to_view[group_idx] = len(views) - 1

        return views, group_to_view

    # ------------------------------------------------------------------
    # register_kv_caches override
    # ------------------------------------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register KV caches and build multi-view descriptors."""
        import msgspec

        from vllm.distributed.kv_transfer.kv_connector.utils import TransferTopology
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
            compute_nixl_compatibility_hash,
        )

        # Detect packed allocation (DSv4-style contiguous per-block packing).
        # All tensors are strided views into the same backing storage.
        # The packed path uses 1 region / 1 desc-per-block and doesn't need
        # multi-view machinery, so delegate to the base class.
        if len(kv_caches) > 1 and not self._has_mamba:
            storage = next(iter(kv_caches.values())).untyped_storage()
            storage_ptrs = {
                cache.untyped_storage().data_ptr() for cache in kv_caches.values()
            }
            data_ptrs = {cache.data_ptr() for cache in kv_caches.values()}
            if len(storage_ptrs) == 1 and len(data_ptrs) > 1:
                self._register_packed_kv_cache(storage)
                self.device_kv_caches = kv_caches
                return

        self.transfer_topo = TransferTopology(
            tp_rank=self.tp_rank,
            tp_size=self.world_size,
            block_size=self.block_size,
            engine_id=self.engine_id,
            is_mla=self.use_mla,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backends=self.attn_backends,
            tensor_shape=next(iter(kv_caches.values())).shape
            if not self._has_mamba
            else None,
            is_mamba=self._has_mamba,
        )
        self.compat_hash = compute_nixl_compatibility_hash(
            self.vllm_config, self.backend_name, self.transfer_topo.cross_layers_blocks
        )

        if self.use_host_buffer:
            self.initialize_host_xfer_buffer(kv_caches=kv_caches)
            assert len(self.host_xfer_buffers) == len(kv_caches)
            xfer_buffers = self.host_xfer_buffers
        else:
            xfer_buffers = kv_caches
            assert not self.host_xfer_buffers

        logger.info(
            "Registering KV_Caches (multiview). use_mla: %s, "
            "kv_buffer_device: %s, use_host_buffer: %s",
            self.use_mla,
            self.kv_buffer_device,
            self.use_host_buffer,
        )

        caches_data = []
        seen_base_addresses: list[int] = []
        self.block_len_per_layer = []
        self.block_stride_per_layer = []
        self.spec_per_region = []
        tensor_size_bytes = None

        for layer_name, cache_or_caches in xfer_buffers.items():
            layer_spec = self._layer_specs.get(layer_name)
            if layer_spec is None:
                logger.debug(
                    "Skipping layer %s (no KVCache spec, likely sharing)", layer_name
                )
                continue
            if isinstance(layer_spec, UniformTypeKVCacheSpecs):
                layer_spec = layer_spec.kv_cache_specs[layer_name]

            cache_list = self.transfer_topo.get_transfer_cache_regions(
                cache_or_caches, layer_spec
            )
            physical_page_size = (
                layer_spec.page_size_bytes
                if isinstance(layer_spec, MambaSpec)
                else layer_spec.page_size_bytes
                // self._physical_blocks_per_logical_kv_block
            )
            physical_page_size = physical_page_size // len(cache_list)
            if self.transfer_topo._cross_layers_blocks:
                physical_page_size = physical_page_size * len(
                    self.kv_cache_config.kv_cache_tensors
                )
            num_blocks = (
                self._logical_num_blocks
                if isinstance(layer_spec, MambaSpec)
                else self.num_blocks
            )
            curr_tensor_size_bytes = num_blocks * physical_page_size
            if tensor_size_bytes is None:
                tensor_size_bytes = curr_tensor_size_bytes

            for cache in cache_list:
                base_addr = cache.data_ptr()
                if base_addr in seen_base_addresses:
                    logger.debug("Skipping %s (already seen)", layer_name)
                    continue
                logger.debug(
                    "Registering layer %s with cache shape: %s", layer_name, cache.shape
                )
                seen_base_addresses.append(base_addr)

                if isinstance(layer_spec, MambaSpec):
                    self.block_len_per_layer.append(
                        physical_page_size // self._physical_blocks_per_logical_kv_block
                    )
                    self.block_stride_per_layer.append(self.block_len_per_layer[-1])
                else:
                    self.block_len_per_layer.append(physical_page_size)
                    self.block_stride_per_layer.append(
                        cache.stride(0) * cache.element_size()
                    )
                self.spec_per_region.append(layer_spec)

                is_mla_region = isinstance(
                    layer_spec, (MLAAttentionSpec, SlidingWindowMLASpec)
                )
                self._region_is_mla.append(is_mla_region)

                if not is_mla_region:
                    assert tensor_size_bytes == curr_tensor_size_bytes, (
                        "All non-MLA kv cache tensors must have the same size"
                    )

                if cache.shape[0] != num_blocks:
                    raise AssertionError(
                        f"Block count mismatch; layer={layer_name}, "
                        f"expected={num_blocks}, got={cache.shape[0]}"
                    )

                self.device_id = max(cache.get_device(), 0)
                caches_data.append(
                    (base_addr, curr_tensor_size_bytes, self.device_id, "")
                )

        assert len(self.block_len_per_layer) == len(seen_base_addresses)
        assert len(self.block_stride_per_layer) == len(seen_base_addresses)
        assert len(self.spec_per_region) == len(seen_base_addresses)

        self.kv_caches_base_addr[self.engine_id][self.tp_rank] = seen_base_addresses
        self.num_regions = len(caches_data)

        if self.transfer_topo.virtually_split_kv_in_blocks:
            self.num_regions = sum(
                1 if self._is_region_replicated(i) else 2
                for i in range(len(self._region_is_mla))
            )

        self.num_descs = self.num_regions * self.num_blocks

        if not self.use_host_buffer:
            current_platform.set_device(self.device_id)

        descs = self.nixl_wrapper.get_reg_descs(caches_data, self.nixl_memory_type)
        self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
        self._registered_descs.append(descs)

        self.device_kv_caches = kv_caches
        self.dst_num_blocks[self.engine_id] = self.num_blocks

        # Build multi-view descriptors.
        self._views, self._group_to_view_idx = self._build_descriptor_views()

        from collections import Counter

        spec_counts = Counter(type(s).__name__ for s in self.spec_per_region)
        spec_summary = ", ".join(
            f"{name}={cnt}" for name, cnt in spec_counts.most_common()
        )

        if self._has_mamba:
            logger.info(
                "Hybrid SSM registration (multiview): num_blocks=%s, "
                "logical_num_blocks=%s, ratio=%s, num_regions=%s, "
                "num_descs=%s, mamba_ssm_size=%s, views=%s, "
                "region_specs={%s}",
                self.num_blocks,
                self._logical_num_blocks,
                self._physical_blocks_per_logical_kv_block,
                self.num_regions,
                self.num_descs,
                self._mamba_ssm_size,
                len(self._views),
                spec_summary,
            )
        else:
            logger.info(
                "KV cache registration (multiview): num_blocks=%s, "
                "logical_num_blocks=%s, ratio=%s, num_regions=%s, "
                "num_descs=%s, views=%s, region_specs={%s}",
                self.num_blocks,
                self._logical_num_blocks,
                self._physical_blocks_per_logical_kv_block,
                self.num_regions,
                self.num_descs,
                len(self._views),
                spec_summary,
            )

        # Register local xfer handler.
        self.src_xfer_handles_by_block_size[self.block_size], self.src_blocks_data = (
            self.register_local_xfer_handler(self.block_size)
        )

        # Publish handshake metadata.
        agent_metadata = NixlAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.nixl_wrapper.get_agent_metadata(),
            device_id=self.device_id,
            kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id][self.tp_rank],
            num_blocks=self.num_blocks,
            block_lens=self.block_len_per_layer,
            block_strides=self.block_stride_per_layer,
            kv_cache_layout=self.kv_cache_layout
            if not self.use_host_buffer
            else self.host_buffer_kv_cache_layout,
            block_size=self.block_size,
            ssm_sizes=self._mamba_ssm_size,
            attn_backend_name=self.backend_name,
            physical_blocks_per_logical_kv_block=(
                self._physical_blocks_per_logical_kv_block
            ),
        )
        assert self.compat_hash is not None
        encoder = msgspec.msgpack.Encoder()
        self.xfer_handshake_metadata = NixlHandshakePayload(
            compatibility_hash=self.compat_hash,
            agent_metadata_bytes=encoder.encode(agent_metadata),
        )

    # ------------------------------------------------------------------
    # View-based local descriptor building
    # ------------------------------------------------------------------

    @staticmethod
    def _view_to_descriptors(
        view: torch.Tensor,
        base_addr: int,
        device_id: int,
    ) -> list[tuple[int, int, int]]:
        """Extract NIXL (addr, len, device_id) descriptors from a 4D meta."""
        elem = view.element_size()
        block_stride = view.stride(_DIM4_B) * elem
        payload = prod(view.shape[1:]) * elem
        offset = view.storage_offset() * elem
        return [
            (base_addr + offset + block_idx * block_stride, payload, device_id)
            for block_idx in range(view.shape[_DIM4_B])
        ]

    def _build_view_local(
        self,
        view: DescriptorView,
        base_addresses: list[int],
        block_size_ratio: int,
    ) -> list[tuple[int, int, int]]:
        """Build local descriptors for one view."""
        if isinstance(view.spec, MambaSpec):
            assert block_size_ratio == 1, (
                "Mamba 3-read transfer with block_size_ratio != 1 "
                f"is not tested. Got {block_size_ratio}."
            )

        num_blocks = view.num_blocks * block_size_ratio
        result: list[tuple[int, int, int]] = []

        for region_pos, region_idx in enumerate(view.region_indices):
            meta = build_region_meta(
                spec=view.spec,
                num_blocks=num_blocks,
                block_size=self.block_size,
                block_stride_bytes=view.strides[region_pos] // block_size_ratio,
                region_content_bytes=view.contents[region_pos] // block_size_ratio,
                virtually_split=view.virtually_split,
            )
            slices = view.spec.slice_for_tp_transfer(
                meta,
                self.world_size,
                self.tp_rank,
                self.world_size,
                self.tp_rank,
                self.model_config,
                virtually_split=view.virtually_split,
            )
            for slice_view in slices:
                descs = self._view_to_descriptors(
                    slice_view, base_addresses[region_idx], self.device_id
                )
                if region_pos == 0 and len(result) == 0:
                    logger.debug(
                        "_build_view_local: spec=%s region=%d "
                        "num_slices=%d slice_shape=%s "
                        "payload=%d stride=%d",
                        type(view.spec).__name__,
                        region_idx,
                        len(slices),
                        list(slice_view.shape),
                        descs[0][1] if descs else 0,
                        view.strides[region_pos] // block_size_ratio,
                    )
                result.extend(descs)

        return result

    def register_local_xfer_handler(
        self,
        block_size: int,
    ) -> tuple[int, list[tuple[int, int, int]]]:
        """Register local xfer handler using multi-view descriptors."""
        assert self.transfer_topo is not None
        block_size_ratio = self.block_size // block_size
        local_base_addresses = self.kv_caches_base_addr[self.engine_id][self.tp_rank]

        blocks_data: list[tuple[int, int, int]] = []
        for view in self._views:
            blocks_data.extend(
                self._build_view_local(view, local_base_addresses, block_size_ratio)
            )

        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        return self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs), blocks_data

    # ------------------------------------------------------------------
    # View-based remote descriptor building
    # ------------------------------------------------------------------

    def _build_view_remote(
        self,
        view: DescriptorView,
        nixl_agent_meta: NixlAgentMetadata,
        remote_physical_per_logical: int,
        remote_tp_rank: int,
        remote_tp_size: int,
    ) -> list[tuple[int, int, int]]:
        """Build remote descriptors for one view."""
        assert self.transfer_topo is not None
        num_blocks = view.remote_num_blocks(
            nixl_agent_meta.num_blocks, remote_physical_per_logical
        )
        result: list[tuple[int, int, int]] = []

        for region_pos, region_idx in enumerate(view.region_indices):
            meta = build_region_meta(
                spec=view.spec,
                num_blocks=num_blocks,
                block_size=nixl_agent_meta.block_size,
                block_stride_bytes=view.remote_stride(
                    nixl_agent_meta.block_lens[region_idx],
                    remote_physical_per_logical,
                ),
                region_content_bytes=view.remote_content(
                    nixl_agent_meta.block_lens[region_idx],
                    sum(nixl_agent_meta.ssm_sizes),
                ),
                virtually_split=view.virtually_split,
            )
            slices = view.spec.slice_for_tp_transfer(
                meta,
                self.transfer_topo.tp_size,
                self.transfer_topo.tp_rank,
                remote_tp_size,
                remote_tp_rank,
                self.model_config,
                virtually_split=view.virtually_split,
            )
            for slice_view in slices:
                descs = self._view_to_descriptors(
                    slice_view,
                    nixl_agent_meta.kv_caches_base_addr[region_idx],
                    nixl_agent_meta.device_id,
                )
                if region_pos == 0 and len(result) == 0:
                    logger.debug(
                        "_build_view_remote: spec=%s region=%d "
                        "num_slices=%d slice_shape=%s "
                        "payload=%d stride=%d "
                        "my_tp=%d my_rank=%d other_tp=%d other_rank=%d",
                        type(view.spec).__name__,
                        region_idx,
                        len(slices),
                        list(slice_view.shape),
                        descs[0][1] if descs else 0,
                        view.strides[region_pos],
                        self.transfer_topo.tp_size,
                        self.transfer_topo.tp_rank,
                        remote_tp_size,
                        remote_tp_rank,
                    )
                result.extend(descs)

        return result

    # ------------------------------------------------------------------
    # add_remote_agent override
    # ------------------------------------------------------------------

    def add_remote_agent(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        remote_tp_rank: int = 0,
        remote_tp_size: int = 1,
    ) -> str:
        """Add remote NIXL agent using multi-view descriptors."""

        engine_id = nixl_agent_meta.engine_id
        if remote_tp_rank in self._remote_agents.get(engine_id, {}):
            logger.debug(
                "Remote agent (%s, rank %s) already registered, skipping.",
                engine_id,
                remote_tp_rank,
            )
            return self._remote_agents[engine_id][remote_tp_rank]

        assert self.transfer_topo is not None
        transfer_topo = self.transfer_topo
        physical_blocks_per_logical = (
            nixl_agent_meta.physical_blocks_per_logical_kv_block
        )
        transfer_info = EngineTransferInfo(
            remote_tp_size=remote_tp_size,
            remote_block_size=nixl_agent_meta.block_size,
            remote_block_len=nixl_agent_meta.block_lens[0],
            remote_physical_blocks_per_logical=physical_blocks_per_logical,
        )
        transfer_topo.register_remote_engine(engine_id, transfer_info)
        logger.info("Transfer plan: %s", transfer_topo.describe(engine_id))

        self.tp_mappings[engine_id] = compute_tp_mapping(
            transfer_topology=transfer_topo,
            remote_tp_size=remote_tp_size,
            group_spec_types=self._group_spec_types,
        )

        remote_agent_name = self.nixl_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata
        )

        block_size_ratio = transfer_topo.block_size_ratio(nixl_agent_meta.block_size)

        if engine_id not in self.dst_num_blocks:
            self.dst_num_blocks[engine_id] = nixl_agent_meta.num_blocks

        self.kv_caches_base_addr[engine_id][remote_tp_rank] = (
            nixl_agent_meta.kv_caches_base_addr
        )
        self._validate_remote_agent_handshake(nixl_agent_meta, remote_tp_size)

        tp_ratio = transfer_topo.tp_ratio(remote_tp_size)

        logger.debug(
            "Registering remote agent (%s, rank %s) with tp_ratio %s",
            engine_id,
            remote_tp_rank,
            tp_ratio,
        )

        plan = self.tp_mappings[engine_id]

        # (Optional) Register local splits for P_TP > D_TP.
        if (
            tp_ratio < 0
            and not self.use_mla
            and tp_ratio not in self.src_xfer_handles_by_tp_ratio
        ):
            self.src_xfer_handles_by_tp_ratio[tp_ratio] = []
            for handle_data in self._build_local_splits_from_plan(
                plan,
                self.src_blocks_data,
            ):
                descs = self.nixl_wrapper.get_xfer_descs(
                    handle_data, self.nixl_memory_type
                )
                handle = self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs)
                self.src_xfer_handles_by_tp_ratio[tp_ratio].append(handle)

        # Register remote agent memory regions using views.
        blocks_data: list[tuple[int, int, int]] = []
        for view in self._views:
            blocks_data.extend(
                self._build_view_remote(
                    view,
                    nixl_agent_meta,
                    physical_blocks_per_logical,
                    remote_tp_rank=remote_tp_rank,
                    remote_tp_size=remote_tp_size,
                )
            )

        logger.info(
            "Remote descs: engine=%s rank=%d total_descs=%d "
            "local_src_descs=%d tp_ratio=%d",
            engine_id[:8],
            remote_tp_rank,
            len(blocks_data),
            len(self.src_blocks_data),
            tp_ratio,
        )
        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        self.dst_xfer_side_handles[engine_id][remote_tp_rank] = (
            self.nixl_wrapper.prep_xfer_dlist(remote_agent_name, descs)
        )

        if block_size_ratio > 1:
            self.src_xfer_handles_by_block_size[nixl_agent_meta.block_size] = (
                self.register_local_xfer_handler(nixl_agent_meta.block_size)[0]
            )

        return remote_agent_name

    # ------------------------------------------------------------------
    # _build_local_splits_from_plan override
    # ------------------------------------------------------------------

    def _build_local_splits_from_plan(
        self,
        plan: TPMapping,
        src_blocks_data: list[tuple[int, int, int]],
        num_fa_descs: int | None = None,
    ) -> Iterator[list[tuple[int, int, int]]]:
        """Build split handle data for P_TP > D_TP using view layout.

        Unlike the base class which splits FA and SSM descriptors separately,
        this uses the view structure to determine split boundaries.
        """
        # For now, delegate to parent's implementation which works at the
        # descriptor level. The view-based _compute_desc_ids already handles
        # the mapping correctly. If num_fa_descs is not provided, compute it
        # from views.
        if num_fa_descs is None:
            # FA descs = sum of all attention view descriptors
            num_fa_descs = sum(
                view.num_descs()
                for view in self._views
                if isinstance(view.spec, AttentionSpec)
            )
        yield from super()._build_local_splits_from_plan(
            plan, src_blocks_data, num_fa_descs
        )
