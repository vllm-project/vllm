# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiSparse destination policy for NIXL prefix imports."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.v1.core.kv_cache_utils import HISPARSE_RESIDENT_SUFFIX
from vllm.v1.kv_cache_interface import (
    HiSparseResidentSpec,
    KVCacheGroupRole,
    KVCacheGroupSpec,
    KVCacheSpec,
)

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
        NixlBaseConnectorWorker,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import ReqMeta
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import TPMapping
    from vllm.v1.kv_cache_interface import KVCacheConfig


class HiSparseNixlAdapter:
    """Map NIXL regions and reads onto HiSparse's stable host pages."""

    def __init__(self, kv_cache_config: KVCacheConfig) -> None:
        host_num_blocks = kv_cache_config.hisparse_host_num_blocks
        assert host_num_blocks is not None
        self.host_num_blocks = host_num_blocks
        self.host_regions: list[tuple[int, int, int] | None] = []
        self._host_registration_ranges: dict[int, int] = {}
        self._host_desc_offsets: list[int | None] = []
        self._host_xfer_handle: int | None = None

    def reset_regions(self) -> None:
        self.host_regions.clear()
        self._host_registration_ranges.clear()
        self._host_desc_offsets.clear()
        self._host_xfer_handle = None

    @staticmethod
    def transfer_layer_name(layer_name: str) -> str:
        if layer_name.endswith(HISPARSE_RESIDENT_SUFFIX):
            return layer_name[: -len(HISPARSE_RESIDENT_SUFFIX)]
        return layer_name

    def logical_num_blocks(
        self, group: KVCacheGroupSpec, default_num_blocks: int
    ) -> int:
        if group.role is KVCacheGroupRole.HISPARSE_SOURCE:
            return self.host_num_blocks
        if group.block_pool_id is None:
            raise ValueError("NIXL transfer group has no block pool")
        return default_num_blocks

    @staticmethod
    def is_mla_region(layer_spec: KVCacheSpec) -> bool:
        return isinstance(layer_spec, HiSparseResidentSpec)

    def register_region(
        self,
        layer_name: str,
        group: KVCacheGroupSpec,
        region_block_len: int,
        kv_caches: dict[str, torch.Tensor],
    ) -> None:
        host_layer_name = None
        if layer_name.endswith(HISPARSE_RESIDENT_SUFFIX):
            host_layer_name = self.transfer_layer_name(layer_name)
        if host_layer_name is None:
            self.host_regions.append(None)
            return

        host_cache = kv_caches.get(host_layer_name)
        if host_cache is None:
            raise ValueError(
                "HiSparse host transfer cache is missing: "
                f"layer={layer_name}, source={host_layer_name}"
            )
        assert host_cache.device.type == "cpu"
        host_stride = host_cache.stride(0) * host_cache.element_size()
        if host_stride < region_block_len:
            raise ValueError(
                "HiSparse host page is smaller than its transfer region: "
                f"layer={layer_name}, host_stride={host_stride}, "
                f"region_block_len={region_block_len}"
            )
        self.host_regions.append(
            (host_cache.data_ptr(), host_stride, host_cache.shape[0])
        )
        storage = host_cache.untyped_storage()
        self._host_registration_ranges[storage.data_ptr()] = storage.nbytes()

    def register_host_memory(self, worker: NixlBaseConnectorWorker) -> None:
        """Register stable host pages and prepare their local descriptor list."""
        registration_data = [
            (address, length, 0, "")
            for address, length in self._host_registration_ranges.items()
        ]
        registration_descs = worker.nixl_wrapper.get_reg_descs(
            registration_data, "DRAM"
        )
        worker.nixl_wrapper.register_memory(
            registration_descs, backends=worker.nixl_backends
        )
        worker._registered_descs.append(registration_descs)

        blocks: list[tuple[int, int, int]] = []
        for region, block_len in zip(
            self.host_regions, worker.block_len_per_layer, strict=True
        ):
            if region is None:
                self._host_desc_offsets.append(None)
                continue
            base, stride, capacity = region
            self._host_desc_offsets.append(len(blocks))
            blocks.extend(
                (base + page_id * stride, block_len, 0) for page_id in range(capacity)
            )
        descs = worker.nixl_wrapper.get_xfer_descs(blocks, "DRAM")
        self._host_xfer_handle = worker.nixl_wrapper.prep_xfer_dlist(
            "NIXL_INIT_AGENT", descs
        )

    def release(self, worker: NixlBaseConnectorWorker) -> None:
        if self._host_xfer_handle is not None:
            worker.nixl_wrapper.release_dlist_handle(self._host_xfer_handle)
            self._host_xfer_handle = None

    def read_host_blocks(
        self,
        worker: NixlBaseConnectorWorker,
        req_id: str,
        meta: ReqMeta,
        plan: TPMapping,
        remote_region_groups: list[int],
    ) -> None:
        """Land an oversized external prefix in stable HiSparse host pages."""
        assert meta.remote is not None and worker.transfer_topo is not None
        if worker._has_mamba or not worker.use_mla:
            raise NotImplementedError(
                "Host-backed HiSparse imports currently require a pure MLA model"
            )
        if len(plan.all_source_ranks) != 1:
            raise NotImplementedError(
                "Host-backed HiSparse imports require replicated NIXL regions"
            )

        engine_id = meta.remote.engine_id
        remote_rank = plan.all_source_ranks[0]
        remote_by_region = worker._block_ids_by_region(
            meta.remote.block_ids, remote_region_groups
        )
        remote_by_region = worker._split_block_ids_by_region(
            remote_by_region, worker.dst_region_split_ratios[engine_id]
        )
        local_by_region = worker._block_ids_by_region(
            meta.local_physical_block_ids, worker.region_group_ids
        )

        host_blocks = meta.hisparse_host_block_ids
        assert host_blocks is not None
        destination_pages: list[list[int]] = []
        for region, local_blocks in zip(
            self.host_regions, local_by_region, strict=True
        ):
            if region is None:
                destination_pages.append(list(local_blocks))
                continue
            _, _, host_page_capacity = region
            if host_page_capacity % self.host_num_blocks:
                raise ValueError("HiSparse host region capacity is not block aligned")
            pages_per_block = host_page_capacity // self.host_num_blocks
            destination_pages.append(
                [
                    block_id * pages_per_block + page_offset
                    for block_id in host_blocks
                    for page_offset in range(pages_per_block)
                ]
            )

        trimmed_destination_pages, trimmed_remote_by_region = (
            worker._apply_prefix_caching_by_region(destination_pages, remote_by_region)
        )
        destination_pages = [list(ids) for ids in trimmed_destination_pages]
        remote_by_region = [list(ids) for ids in trimmed_remote_by_region]
        remote_desc_ids = worker._compute_desc_ids(
            block_ids=remote_by_region,
            dst_num_blocks=worker.dst_num_blocks[engine_id],
            block_size_ratio=None,
            physical_blocks_per_logical=(
                worker.transfer_topo.get_engine_info(
                    engine_id
                ).remote_physical_blocks_per_logical
            ),
            region_num_blocks=worker.dst_region_num_blocks[engine_id],
            region_group_ids=list(range(worker.num_regions)),
        )

        remote_ids_by_region: list[np.ndarray] = []
        cursor = 0
        for blocks in remote_by_region:
            next_cursor = cursor + len(blocks)
            remote_ids_by_region.append(remote_desc_ids[cursor:next_cursor])
            cursor = next_cursor
        assert cursor == len(remote_desc_ids)

        host_local_ids: list[int] = []
        host_remote_ids: list[int] = []
        device_pages: list[list[int]] = []
        device_remote_ids: list[int] = []
        for region_idx, (pages, mirrors, region_remote_ids) in enumerate(
            zip(
                destination_pages,
                local_by_region,
                remote_ids_by_region,
                strict=True,
            )
        ):
            host_region = self.host_regions[region_idx]
            if host_region is None:
                device_pages.append(pages)
                device_remote_ids.extend(region_remote_ids)
                continue

            host_offset = self._host_desc_offsets[region_idx]
            assert host_offset is not None
            host_local_ids.extend(host_offset + page_id for page_id in pages)
            host_remote_ids.extend(region_remote_ids)

            mirror_offset = len(pages) - len(mirrors)
            if mirror_offset < 0:
                raise ValueError(
                    "HiSparse resident mirror exceeds imported host history"
                )
            device_pages.append(list(mirrors))
            device_remote_ids.extend(region_remote_ids[mirror_offset:])

        remote_info = worker.transfer_topo.get_engine_info(engine_id)
        local_device_ids = worker._compute_desc_ids(
            block_ids=device_pages,
            dst_num_blocks=worker.dst_num_blocks[worker.engine_id],
            block_size_ratio=worker.transfer_topo.block_size_ratio(
                remote_info.remote_block_size
            ),
            physical_blocks_per_logical=(worker._physical_blocks_per_logical_kv_block),
            region_num_blocks=worker.region_num_blocks,
            region_group_ids=list(range(worker.num_regions)),
        )
        if len(local_device_ids) != len(device_remote_ids):
            raise ValueError(
                "HiSparse device mirror descriptor count mismatch: "
                f"remote={len(device_remote_ids)}, local={len(local_device_ids)}"
            )

        assert self._host_xfer_handle is not None
        transfer_specs = [
            (
                self._host_xfer_handle,
                np.asarray(host_local_ids, dtype=np.int64),
                np.asarray(host_remote_ids, dtype=np.int64),
            ),
            (
                worker.src_xfer_handles_by_block_size[remote_info.remote_block_size],
                local_device_ids,
                np.asarray(device_remote_ids, dtype=np.int64),
            ),
        ]
        transfer_specs = [spec for spec in transfer_specs if len(spec[1])]

        notif_id = f"{meta.remote.request_id}:{worker.world_size}".encode()
        agents = worker._remote_agents[engine_id]
        if worker.transfer_topo.tp_ratio(meta.tp_size) < 0:
            pending_notifs = [(agent, notif_id) for agent in agents.values()]
        else:
            pending_notifs = [(agents[(0, remote_rank)], notif_id)]
        worker._pending_recv_notifs.setdefault(req_id, []).extend(pending_notifs)

        handles: list[int] = []
        try:
            remote_handle = worker.dst_xfer_side_handles[engine_id][remote_rank]
            for local_handle, local_ids, selected_remote_ids in transfer_specs:
                handles.append(
                    worker.nixl_wrapper.make_prepped_xfer(
                        "READ",
                        local_handle,
                        local_ids,
                        remote_handle,
                        selected_remote_ids,
                    )
                )
            worker._recving_transfers.setdefault(req_id, [])
            for handle in handles:
                worker.nixl_wrapper.transfer(handle)
                worker._recving_transfers[req_id].append(handle)
        except Exception as error:
            started = set(worker._recving_transfers.get(req_id, ()))
            for handle in handles:
                if handle not in started:
                    worker.nixl_wrapper.release_xfer_handle(handle)
            worker._log_failure(
                failure_type="transfer_setup_failed",
                req_id=req_id,
                msg="Host-backed HiSparse import setup failed",
                error=error,
                dst_engine_id=engine_id,
                remote_rank=remote_rank,
            )
            worker._handle_failed_transfer(req_id, None)


def make_hisparse_nixl_adapter(
    kv_cache_config: KVCacheConfig,
) -> HiSparseNixlAdapter | None:
    if kv_cache_config.hisparse_host_num_blocks is None:
        return None
    return HiSparseNixlAdapter(kv_cache_config)
