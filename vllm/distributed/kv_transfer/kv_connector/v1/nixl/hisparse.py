# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiSparse destination policy for NIXL prefix imports."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm import envs
from vllm.v1.core.kv_cache_utils import (
    HISPARSE_INDEXER_SOURCE_SUFFIX,
    HISPARSE_RESIDENT_SUFFIX,
)
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
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.host_staging import (
        HostReadStager,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import ReqMeta
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import TPMapping
    from vllm.v1.kv_cache_interface import KVCacheConfig

_CUDA_MEMCPY_DEVICE_TO_HOST = 2
_CUDA_MEMCPY_DEVICE_TO_DEVICE = 3


class HiSparseNixlAdapter:
    """Map NIXL regions and reads onto HiSparse's stable host pages."""

    def __init__(self, kv_cache_config: KVCacheConfig) -> None:
        host_num_blocks = kv_cache_config.hisparse_host_num_blocks
        assert host_num_blocks is not None
        self.host_num_blocks = host_num_blocks
        self.host_regions: list[tuple[int, int, int] | None] = []
        self.host_stager: HostReadStager | None = None

    def reset_regions(self) -> None:
        self.host_regions.clear()

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
        elif group.role is KVCacheGroupRole.HISPARSE_INDEXER:
            host_layer_name = f"{layer_name}{HISPARSE_INDEXER_SOURCE_SUFFIX}"
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

    def get_host_stager(self, worker: NixlBaseConnectorWorker) -> HostReadStager:
        if self.host_stager is not None:
            return self.host_stager
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.host_staging import (
            HostReadStager,
        )

        stage_bytes = envs.VLLM_NIXL_HOST_STAGE_BYTES
        if stage_bytes <= 0:
            raise RuntimeError(
                "Host-backed HiSparse P/D imports require "
                "VLLM_NIXL_HOST_STAGE_BYTES > 0"
            )
        lengths = np.asarray(worker.block_len_per_layer, dtype=np.int64)
        self.host_stager = HostReadStager(
            desc_lens=lengths,
            host_addrs=np.zeros(len(lengths), dtype=np.uint64),
            device=torch.device(f"cuda:{worker.device_id}"),
            nixl_wrapper=worker.nixl_wrapper,
            memory_type=worker.nixl_memory_type,
            backends=worker.nixl_backends,
            stage_bytes=stage_bytes,
            num_slots=max(envs.VLLM_NIXL_HOST_STAGE_SLOTS, 1),
        )
        return self.host_stager

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

        local_bases = worker.kv_caches_base_addr[worker.engine_id][worker.tp_rank]
        dst_addrs: list[int] = []
        desc_lens: list[int] = []
        copy_kinds: list[int] = []
        mirror_addrs: list[int] = []
        for region_idx, (pages, mirrors) in enumerate(
            zip(destination_pages, local_by_region, strict=True)
        ):
            host_region = self.host_regions[region_idx]
            block_len = worker.block_len_per_layer[region_idx]
            mirror_offset = len(pages) - len(mirrors)
            if mirror_offset < 0:
                raise ValueError(
                    "HiSparse resident mirror exceeds imported host history"
                )
            mirror_by_page = {
                mirror_offset + idx: block_id for idx, block_id in enumerate(mirrors)
            }
            for page_idx, page_id in enumerate(pages):
                if host_region is None:
                    dst_addrs.append(
                        local_bases[region_idx]
                        + page_id * worker.region_strides[region_idx]
                    )
                    copy_kinds.append(_CUDA_MEMCPY_DEVICE_TO_DEVICE)
                    mirror_addrs.append(0)
                else:
                    host_base, host_stride, _ = host_region
                    dst_addrs.append(host_base + page_id * host_stride)
                    copy_kinds.append(_CUDA_MEMCPY_DEVICE_TO_HOST)
                    mirror_block = mirror_by_page.get(page_idx)
                    mirror_addrs.append(
                        0
                        if mirror_block is None
                        else local_bases[region_idx]
                        + mirror_block * worker.region_strides[region_idx]
                    )
                desc_lens.append(block_len)

        if len(remote_desc_ids) != len(dst_addrs):
            raise ValueError(
                "HiSparse host import descriptor count mismatch: "
                f"remote={len(remote_desc_ids)}, local={len(dst_addrs)}"
            )
        notif_id = f"{meta.remote.request_id}:{worker.world_size}".encode()
        agents = worker._remote_agents[engine_id]
        if worker.transfer_topo.tp_ratio(meta.tp_size) < 0:
            pending_notifs = [(agent, notif_id) for agent in agents.values()]
        else:
            pending_notifs = [(agents[(0, remote_rank)], notif_id)]
        worker._pending_recv_notifs.setdefault(req_id, []).extend(pending_notifs)
        try:
            self.get_host_stager(worker).submit(
                req_id,
                remote_desc_ids,
                np.arange(len(remote_desc_ids), dtype=np.int64),
                worker.dst_xfer_side_handles[engine_id][remote_rank],
                dst_addrs=np.asarray(dst_addrs, dtype=np.uint64),
                desc_lens=np.asarray(desc_lens, dtype=np.int64),
                copy_kinds=np.asarray(copy_kinds, dtype=np.int32),
                mirror_addrs=np.asarray(mirror_addrs, dtype=np.uint64),
            )
        except Exception as error:
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
