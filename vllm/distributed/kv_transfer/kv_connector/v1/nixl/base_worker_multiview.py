# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-view descriptor layer for the NIXL connector.

Replaces the per-spec builder methods (_build_fa_local/remote,
_build_mamba_local/remote) with a unified DescriptorView abstraction
that delegates layout decisions to KVCacheSpec.compute_transfer_shape()
and KVCacheSpec.slice_for_tp_transfer().

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
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    TPMapping,
)
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

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
    H, N, C = spec.compute_transfer_shape(region_content_bytes, block_size)

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
      - _on_kv_caches_registered  (builds DescriptorViews after region setup)
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
        if self._is_packed_kv:
            return super()._compute_desc_ids(
                block_ids,
                dst_num_blocks,
                block_size_ratio,
                physical_blocks_per_logical,
            )
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
            elif isinstance(representative_spec, AttentionSpec):
                strides = [self.block_stride_per_layer[i] for i in all_indices]
                contents = [self.block_len_per_layer[i] for i in all_indices]
                num_blocks = self.num_blocks
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
            )
            probe_slices = representative_spec.slice_for_tp_transfer(
                probe_meta,
                self.world_size,
                self.tp_rank,
                self.world_size,
                self.tp_rank,
                self.model_config,
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
                )
            )
            group_to_view[group_idx] = len(views) - 1

        return views, group_to_view

    # ------------------------------------------------------------------
    # Hook: build views after base class registers KV regions
    # ------------------------------------------------------------------

    def _on_kv_caches_registered(self) -> None:
        """Build DescriptorViews after NIXL memory regions are registered."""
        self._views, self._group_to_view_idx = self._build_descriptor_views()

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
            )
            slices = view.spec.slice_for_tp_transfer(
                meta,
                self.world_size,
                self.tp_rank,
                self.world_size,
                self.tp_rank,
                self.model_config,
            )
            for slice_view in slices:
                descs = self._view_to_descriptors(
                    slice_view, base_addresses[region_idx], self.device_id
                )
                result.extend(descs)

        return result

    def register_local_xfer_handler(
        self,
        block_size: int,
    ) -> tuple[int, list[tuple[int, int, int]]]:
        """Register local xfer handler using multi-view descriptors."""
        if self._is_packed_kv:
            return super().register_local_xfer_handler(block_size)
        assert self.transfer_topo is not None
        block_size_ratio = self.block_size // block_size
        local_base_addresses = self.kv_caches_base_addr[self.engine_id][self.tp_rank]

        blocks_data: list[tuple[int, int, int]] = []
        for view in self._views:
            view_descs = self._build_view_local(
                view, local_base_addresses, block_size_ratio
            )
            blocks_data.extend(view_descs)

        expected_descs = sum(v.num_descs() for v in self._views) * block_size_ratio
        assert len(blocks_data) == expected_descs, (
            f"View descriptor count mismatch: built {len(blocks_data)} descs, "
            f"expected {expected_descs} (views={len(self._views)}, "
            f"block_size_ratio={block_size_ratio})"
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
                    nixl_agent_meta.block_strides[region_idx],
                    remote_physical_per_logical,
                ),
                region_content_bytes=view.remote_content(
                    nixl_agent_meta.block_lens[region_idx],
                    sum(nixl_agent_meta.ssm_sizes),
                ),
            )
            slices = view.spec.slice_for_tp_transfer(
                meta,
                self.transfer_topo.tp_size,
                self.transfer_topo.tp_rank,
                remote_tp_size,
                remote_tp_rank,
                self.model_config,
            )
            for slice_view in slices:
                descs = self._view_to_descriptors(
                    slice_view,
                    nixl_agent_meta.kv_caches_base_addr[region_idx],
                    nixl_agent_meta.device_id,
                )
                result.extend(descs)

        return result

    # ------------------------------------------------------------------
    # Remote descriptor building override
    # ------------------------------------------------------------------

    def _build_all_remote_descs(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        plan: TPMapping,
        block_size_ratio: int,
        tp_ratio: int,
        transfer_info: EngineTransferInfo,
        physical_blocks_per_logical: int,
        remote_tp_rank: int,
        remote_tp_size: int,
    ) -> list[tuple[int, int, int]]:
        """Build remote descriptors using multi-view layout."""
        if self._is_packed_kv:
            return super()._build_all_remote_descs(
                nixl_agent_meta,
                plan,
                block_size_ratio,
                tp_ratio,
                transfer_info,
                physical_blocks_per_logical,
                remote_tp_rank,
                remote_tp_size,
            )

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
        return blocks_data

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

        Unlike the base class which uses a flat num_fa_descs boundary,
        this iterates per-view to determine split counts and slot
        mappings, making it independent of view ordering.
        """
        if self._is_packed_kv:
            yield from super()._build_local_splits_from_plan(
                plan,
                src_blocks_data,
                num_fa_descs if num_fa_descs is not None else 0,
            )
            return

        # Per-view: (start, end, num_splits, rank_to_slot).
        view_info: list[tuple[int, int, int, dict[int, int]]] = []
        position = 0
        for view_idx, view in enumerate(self._views):
            representative_group = next(
                group_idx
                for group_idx, mapped in self._group_to_view_idx.items()
                if mapped == view_idx
            )
            if isinstance(view.spec, MambaSpec):
                rank_to_slot = {r: idx for idx, r in enumerate(plan.all_source_ranks)}
                num_splits = len(plan.source_ranks_per_group[representative_group])
            else:
                rank_to_slot = {
                    r: plan.rank_to_attention_slot.get(r, 0)
                    for r in plan.all_source_ranks
                }
                num_splits = len(
                    set(
                        plan.rank_to_attention_slot[r]
                        for r in plan.source_ranks_per_group[representative_group]
                    )
                )
            view_info.append(
                (
                    position,
                    position + view.num_descs(),
                    num_splits,
                    rank_to_slot,
                )
            )
            position += view.num_descs()

        for _source_idx, source_rank in enumerate(plan.all_source_ranks):
            handle: list[tuple[int, int, int]] = []
            for start, end, num_splits, rank_to_slot in view_info:
                slot = rank_to_slot[source_rank]
                for j in range(start, end):
                    addr, local_len, dev = src_blocks_data[j]
                    chunk = local_len // num_splits
                    handle.append((addr + slot * chunk, chunk, dev))
            yield handle
