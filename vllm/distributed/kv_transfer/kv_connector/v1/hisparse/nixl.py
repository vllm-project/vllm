# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL destination selection for HiSparse prefix imports."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.v1.core.kv_cache_utils import HISPARSE_RESIDENT_SUFFIX
from vllm.v1.kv_cache_interface import HiSparseResidentSpec

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
        NixlBaseConnectorWorker,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import ReqMeta
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import TPMapping
    from vllm.v1.hisparse.runtime import HiSparseRuntime
    from vllm.v1.kv_cache_interface import KVCacheConfig


class HiSparseNixlDestination:
    """Provide a host destination list parallel to HiSparse GPU regions."""

    def __init__(self, kv_cache_config: KVCacheConfig, vllm_config: VllmConfig) -> None:
        host_num_blocks = kv_cache_config.hisparse_host_num_blocks
        assert host_num_blocks is not None
        self.host_num_blocks = host_num_blocks
        self._forward_context = vllm_config.compilation_config.static_forward_context
        self.host_regions: list[tuple[int, int] | None] = []
        self._host_pools: dict[int, tuple[torch.Tensor, list[HiSparseRuntime]]] = {}
        self._host_desc_lens = np.empty(0, dtype=np.int64)
        self._host_addrs = np.empty(0, dtype=np.uint64)
        self._descriptor_offsets: list[int | None] = []
        self._xfer_handle: int | None = None

    def reset_regions(self) -> None:
        self.host_regions.clear()
        self._host_pools.clear()
        self._host_desc_lens = np.empty(0, dtype=np.int64)
        self._host_addrs = np.empty(0, dtype=np.uint64)
        self._descriptor_offsets.clear()
        self._xfer_handle = None

    @staticmethod
    def transfer_layer_name(layer_name: str) -> str:
        if layer_name.endswith(HISPARSE_RESIDENT_SUFFIX):
            return layer_name[: -len(HISPARSE_RESIDENT_SUFFIX)]
        return layer_name

    @staticmethod
    def is_mla_region(layer_spec: object) -> bool:
        return isinstance(layer_spec, HiSparseResidentSpec)

    def register_region(
        self,
        layer_name: str,
        region_block_len: int,
        kv_caches: dict[str, torch.Tensor],
    ) -> None:
        if not layer_name.endswith(HISPARSE_RESIDENT_SUFFIX):
            self.host_regions.append(None)
            return

        source_name = self.transfer_layer_name(layer_name)
        host_cache = kv_caches.get(source_name)
        if host_cache is None or host_cache.device.type != "cpu":
            raise ValueError(f"Missing HiSparse host cache for {source_name}")
        if host_cache.shape[0] != self.host_num_blocks:
            raise ValueError(
                f"HiSparse host cache {source_name} has {host_cache.shape[0]} "
                f"blocks, expected {self.host_num_blocks}"
            )
        stride = host_cache.stride(0) * host_cache.element_size()
        if stride != region_block_len:
            raise ValueError(
                f"HiSparse host and GPU pages differ for {source_name}: "
                f"host={stride}, gpu={region_block_len}"
            )
        attention_layer = self._forward_context[source_name]
        cache_handle = attention_layer.hisparse_cache
        assert cache_handle is not None
        runtime = cache_handle.runtime
        registered_pool = runtime.registered_host_pool
        pool_start = registered_pool.data_ptr()
        pool_end = pool_start + registered_pool.nbytes
        host_end = host_cache.data_ptr() + host_cache.nbytes
        if not pool_start <= host_cache.data_ptr() < host_end <= pool_end:
            raise ValueError(
                f"HiSparse host cache {source_name} is outside its registered pool"
            )
        pool, runtimes = self._host_pools.setdefault(pool_start, (registered_pool, []))
        if pool.nbytes != registered_pool.nbytes:
            raise ValueError("HiSparse host pool aliases have inconsistent sizes")
        if runtime not in runtimes:
            runtimes.append(runtime)
        self.host_regions.append((host_cache.data_ptr(), stride))

    def prepare_host_descriptors(
        self,
        worker: NixlBaseConnectorWorker,
    ) -> None:
        torch.accelerator.synchronize()
        cudart = torch.cuda.cudart()
        for pool, runtimes in self._host_pools.values():
            if not all(
                runtime.host_pool_registration_owned_by_hisparse for runtime in runtimes
            ):
                raise RuntimeError("HiSparse host pool registration has no owner")
            error = cudart.cudaHostUnregister(pool.data_ptr())
            if error.value != 0:
                raise RuntimeError(
                    "HiSparse cudaHostUnregister failed before NIXL registration: "
                    f"{error}"
                )
            for runtime in runtimes:
                runtime.host_pool_registration_owned_by_hisparse = False

        registration_descs = worker.nixl_wrapper.get_reg_descs(
            [
                (pool.data_ptr(), pool.nbytes, 0, "")
                for pool, _ in self._host_pools.values()
            ],
            "DRAM",
        )
        worker.nixl_wrapper.register_memory(
            registration_descs, backends=worker.nixl_backends
        )
        worker._registered_descs.append(registration_descs)

        blocks: list[tuple[int, int, int]] = []
        for region in self.host_regions:
            if region is None:
                self._descriptor_offsets.append(None)
                continue
            base, stride = region
            self._descriptor_offsets.append(len(blocks))
            blocks.extend(
                (base + block_id * stride, stride, 0)
                for block_id in range(self.host_num_blocks)
            )
        descs = worker.nixl_wrapper.get_xfer_descs(blocks, "DRAM")
        self._host_addrs = np.asarray([block[0] for block in blocks], dtype=np.uint64)
        self._host_desc_lens = np.asarray(
            [block[1] for block in blocks], dtype=np.int64
        )
        self._xfer_handle = worker.nixl_wrapper.prep_xfer_dlist(
            "NIXL_INIT_AGENT", descs, backends=worker.nixl_backends
        )

    def release(self, worker: NixlBaseConnectorWorker) -> None:
        if self._xfer_handle is not None:
            worker.nixl_wrapper.release_dlist_handle(self._xfer_handle)
            self._xfer_handle = None

    def read_host_blocks(
        self,
        worker: NixlBaseConnectorWorker,
        request_id: str,
        meta: ReqMeta,
        plan: TPMapping,
        remote_region_groups: list[int],
    ) -> None:
        """Read sparse regions to host and device-only regions to GPU."""
        assert meta.remote is not None and worker.transfer_topo is not None
        if worker._has_mamba or not worker.use_mla:
            raise NotImplementedError("HiSparse host imports require a pure MLA model")
        if len(plan.all_source_ranks) != 1:
            raise NotImplementedError(
                "HiSparse host imports require replicated NIXL regions"
            )

        engine_id = meta.remote.engine_id
        remote_rank = plan.all_source_ranks[0]
        remote_by_region = worker._block_ids_by_region(
            meta.remote.block_ids, remote_region_groups
        )
        device_by_region = worker._block_ids_by_region(
            meta.local_physical_block_ids, worker.region_group_ids
        )
        host_blocks = meta.hisparse_host_block_ids
        assert host_blocks is not None
        local_by_region = [
            list(host_blocks) if host_region is not None else list(device_blocks)
            for host_region, device_blocks in zip(
                self.host_regions, device_by_region, strict=True
            )
        ]
        trimmed_local, trimmed_remote = worker._apply_prefix_caching_by_region(
            local_by_region, remote_by_region
        )
        local_by_region = [list(blocks) for blocks in trimmed_local]
        remote_by_region = [list(blocks) for blocks in trimmed_remote]

        remote_info = worker.transfer_topo.get_engine_info(engine_id)
        remote_ids = worker._compute_desc_ids(
            block_ids=remote_by_region,
            dst_num_blocks=worker.dst_num_blocks[engine_id],
            block_size_ratio=None,
            physical_blocks_per_logical=(
                remote_info.remote_physical_blocks_per_logical
            ),
            region_num_blocks=worker.dst_region_num_blocks[engine_id],
            region_group_ids=list(range(worker.num_regions)),
        )
        remote_ids_by_region: list[np.ndarray] = []
        cursor = 0
        for region_blocks in remote_by_region:
            end = cursor + len(region_blocks)
            remote_ids_by_region.append(remote_ids[cursor:end])
            cursor = end
        assert cursor == len(remote_ids)

        host_local_ids: list[int] = []
        host_remote_ids: list[int] = []
        device_regions: list[list[int]] = []
        device_remote_ids: list[int] = []
        for region_index, (local_blocks, region_remote_ids) in enumerate(
            zip(local_by_region, remote_ids_by_region, strict=True)
        ):
            host_offset = self._descriptor_offsets[region_index]
            if host_offset is None:
                device_regions.append(local_blocks)
                device_remote_ids.extend(region_remote_ids)
            else:
                device_regions.append([])
                host_local_ids.extend(
                    host_offset + block_id for block_id in local_blocks
                )
                host_remote_ids.extend(region_remote_ids)

        device_local_ids = worker._compute_desc_ids(
            block_ids=device_regions,
            dst_num_blocks=worker.dst_num_blocks[worker.engine_id],
            block_size_ratio=worker.transfer_topo.block_size_ratio(
                remote_info.remote_block_size
            ),
            physical_blocks_per_logical=(worker._physical_blocks_per_logical_kv_block),
            region_num_blocks=worker.region_num_blocks,
            region_group_ids=list(range(worker.num_regions)),
        )
        if len(device_local_ids) != len(device_remote_ids):
            raise ValueError("HiSparse device descriptor count mismatch")

        assert self._xfer_handle is not None
        local_device_handle = worker.src_xfer_handles_by_block_size[
            remote_info.remote_block_size
        ]
        host_local_ids_array = np.asarray(host_local_ids, dtype=np.int64)
        host_remote_ids_array = np.asarray(host_remote_ids, dtype=np.int64)
        host_stager = worker._maybe_init_host_stager_for_buffers(
            meta.remote.host,
            self._host_desc_lens,
            self._host_addrs,
            [pool for pool, _ in self._host_pools.values()],
        )
        transfer_specs = [
            (
                local_device_handle,
                device_local_ids,
                np.asarray(device_remote_ids, dtype=np.int64),
            )
        ]
        if host_stager is None:
            transfer_specs.insert(
                0, (self._xfer_handle, host_local_ids_array, host_remote_ids_array)
            )

        notif_id = f"{meta.remote.request_id}:{plan.local_consumers}".encode()
        agents = worker._remote_agents[engine_id]
        if worker.transfer_topo.tp_ratio(meta.tp_size) < 0:
            pending_notifs = [(agent, notif_id) for agent in agents.values()]
        else:
            pending_notifs = [(agents[(0, remote_rank)], notif_id)]
        worker._pending_recv_notifs.setdefault(request_id, []).extend(pending_notifs)

        handles: list[int] = []
        try:
            remote_handle = worker.dst_xfer_side_handles[engine_id][remote_rank]
            for local_handle, local_ids, selected_remote_ids in transfer_specs:
                if not len(local_ids):
                    continue
                handles.append(
                    worker.nixl_wrapper.make_prepped_xfer(
                        "READ",
                        local_handle,
                        local_ids,
                        remote_handle,
                        selected_remote_ids,
                    )
                )
            if host_stager is not None and len(host_local_ids_array):
                host_stager.submit(
                    request_id,
                    host_remote_ids_array,
                    host_local_ids_array,
                    remote_handle,
                )
            worker._recving_transfers.setdefault(request_id, [])
            for handle in handles:
                worker.nixl_wrapper.transfer(handle)
                worker._recving_transfers[request_id].append(handle)
        except Exception as error:
            started = set(worker._recving_transfers.get(request_id, ()))
            for handle in handles:
                if handle not in started:
                    worker.nixl_wrapper.release_xfer_handle(handle)
            worker._log_failure(
                failure_type="transfer_setup_failed",
                req_id=request_id,
                msg="HiSparse host import setup failed",
                error=error,
                dst_engine_id=engine_id,
                remote_rank=remote_rank,
            )
            worker._handle_failed_transfer(request_id, None)


def make_hisparse_nixl_destination(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> HiSparseNixlDestination | None:
    if kv_cache_config.hisparse_host_num_blocks is None or not any(
        isinstance(group.kv_cache_spec, HiSparseResidentSpec)
        for group in kv_cache_config.transfer_groups
    ):
        return None
    return HiSparseNixlDestination(kv_cache_config, vllm_config)
