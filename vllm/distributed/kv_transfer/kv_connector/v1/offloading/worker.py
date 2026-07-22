# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict
from dataclasses import replace

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    ReqId,
)
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingSpec,
    OffloadingWorker,
)

logger = init_logger(__name__)


class OffloadingConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(
        self,
        spec: OffloadingSpec,
        kv_cache_config: KVCacheConfig,
    ):
        self.spec = spec
        self.kv_cache_config = kv_cache_config
        self.worker: OffloadingWorker | None = None

        # job_id -> req_id for in-flight loads.
        self._load_jobs: dict[int, ReqId] = {}
        self._unsubmitted_store_jobs: list[
            tuple[int, GPULoadStoreSpec, LoadStoreSpec]
        ] = []
        self._connector_worker_meta = OffloadingWorkerMetadata()

    def _init_worker(self, kv_caches: CanonicalKVCaches) -> None:
        self.worker = self.spec.get_worker(kv_caches)

    def register_kv_caches(
        self, kv_caches: dict[str, torch.Tensor | list[torch.Tensor]]
    ):
        kv_cache_config = self.kv_cache_config
        num_blocks = kv_cache_config.num_blocks

        # layer_name -> (num_blocks, page_size_bytes) int8 view.
        # Standardized layouts always have num_blocks as the leading dim.
        tensors_per_block: dict[str, tuple[torch.Tensor, ...]] = {}
        # layer_name -> size of (un-padded) page in bytes
        unpadded_page_size_bytes: dict[str, int] = {}
        # layer_name -> size of page in bytes
        page_size_bytes: dict[str, int] = {}
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            group_layer_names = kv_cache_group.layer_names
            group_kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(group_kv_cache_spec, UniformTypeKVCacheSpecs):
                per_layer_specs = group_kv_cache_spec.kv_cache_specs
            else:
                per_layer_specs = {}
            for layer_name in group_layer_names:
                layer_kv_cache_spec = per_layer_specs.get(
                    layer_name, group_kv_cache_spec
                )
                layer_kv_cache = kv_caches[layer_name]
                # AttentionSpec yields a single tensor; MambaSpec yields a
                # list of typed state tensors that share one underlying
                # buffer. Either way, the first tensor's storage_offset
                # marks the start of this layer's region.
                ref = (
                    layer_kv_cache[0]
                    if isinstance(layer_kv_cache, list)
                    else layer_kv_cache
                )
                page = layer_kv_cache_spec.page_size_bytes
                elem_size = ref.element_size()
                byte_offset = ref.storage_offset() * elem_size
                # Packed layouts (e.g. DSv4) interleave layers per block, so
                # the attention tensor's stride(0) (the manager-block stride)
                # exceeds page_size_bytes. Non-packed layouts have
                # stride(0) == page_size_bytes.
                block_stride_bytes = (
                    ref.stride(0) * elem_size
                    if isinstance(layer_kv_cache_spec, AttentionSpec)
                    else page
                )
                tensors_per_block[layer_name] = (
                    torch.tensor([], dtype=torch.int8, device=ref.device).set_(
                        ref.untyped_storage(),
                        byte_offset,
                        (num_blocks, page),
                        (block_stride_bytes, 1),
                    ),
                )
                page_size_bytes[layer_name] = page

                if isinstance(layer_kv_cache_spec, AttentionSpec):
                    unpadded_page_size_bytes[layer_name] = (
                        layer_kv_cache_spec.unpadded_page_size_bytes
                    )
                elif isinstance(layer_kv_cache_spec, MambaSpec):
                    unpadded_page_size_bytes[layer_name] = replace(
                        layer_kv_cache_spec, page_size_padded=None
                    ).page_size_bytes
                else:
                    raise NotImplementedError

        # Packed layouts (e.g. DSv4) interleave all layers within each
        # manager block: a layer view's block stride exceeds its page size.
        # Offload the whole packed block as a single transfer region.
        packed_layer_name = next(
            (
                layer_name
                for layer_name, (tensor,) in tensors_per_block.items()
                if tensor.stride(0) != tensor.shape[1]
            ),
            None,
        )
        if packed_layer_name is not None:
            (tensor,) = tensors_per_block[packed_layer_name]
            block_stride = tensor.stride(0)
            packed_tensor = tensor.as_strided(
                (num_blocks, block_stride),
                (block_stride, 1),
                storage_offset=0,
            )
            self._init_worker(
                CanonicalKVCaches(
                    [CanonicalKVCacheTensor(packed_tensor, block_stride)],
                    [
                        [CanonicalKVCacheRef(0, block_stride)]
                        for _ in kv_cache_config.kv_cache_groups
                    ],
                )
            )
            return

        block_tensors: list[CanonicalKVCacheTensor] = []
        block_data_refs: dict[str, list[CanonicalKVCacheRef]] = defaultdict(list)
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            for slot_layers in kv_cache_tensor.shared_by:
                # Filter to layers that were actually processed above.
                # Some slots may have no corresponding model layer (reserved
                # memory with no group layer at that index).
                tensor_layer_names = [n for n in slot_layers if n in tensors_per_block]
                if not tensor_layer_names:
                    continue

                # Verify all layers in the slot reference the same tensors.
                assert len({len(tensors_per_block[n]) for n in tensor_layer_names}) == 1
                data_ptrs = {
                    n: tensors_per_block[n][0].data_ptr() for n in tensor_layer_names
                }
                assert len(set(data_ptrs.values())) == 1, data_ptrs
                assert (
                    len({tensors_per_block[n][0].stride() for n in tensor_layer_names})
                    == 1
                )

                first_layer_name = tensor_layer_names[0]
                for tensor in tensors_per_block[first_layer_name]:
                    block_tensors.append(
                        CanonicalKVCacheTensor(
                            tensor=tensor,
                            page_size_bytes=page_size_bytes[first_layer_name],
                        )
                    )

                    curr_tensor_idx = len(block_tensors) - 1
                    for layer_name in tensor_layer_names:
                        block_data_refs[layer_name].append(
                            CanonicalKVCacheRef(
                                tensor_idx=curr_tensor_idx,
                                page_size_bytes=(unpadded_page_size_bytes[layer_name]),
                            )
                        )

        group_data_refs: list[list[CanonicalKVCacheRef]] = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            group_refs: list[CanonicalKVCacheRef] = []
            for layer_name in kv_cache_group.layer_names:
                group_refs += block_data_refs[layer_name]
            group_data_refs.append(group_refs)

        canonical_kv_caches = CanonicalKVCaches(
            tensors=block_tensors,
            group_data_refs=group_data_refs,
        )

        self._init_worker(canonical_kv_caches)

    def handle_preemptions(self, kv_connector_metadata: OffloadingConnectorMetadata):
        assert self.worker is not None

        # Pop jobs_to_flush from store_jobs into _unsubmitted_store_jobs
        # so the existing submission loop below submits them before wait().
        if kv_connector_metadata.jobs_to_flush:
            for job_id in kv_connector_metadata.jobs_to_flush:
                entry = kv_connector_metadata.store_jobs.pop(job_id, None)
                if entry is not None:
                    assert isinstance(entry.src_spec, GPULoadStoreSpec)
                    self._unsubmitted_store_jobs.append(
                        (job_id, entry.src_spec, entry.dst_spec)
                    )

        # Submit deferred stores from previous step (and jobs_to_flush above).
        for job_id, src_spec, dst_spec in self._unsubmitted_store_jobs:
            assert isinstance(src_spec, GPULoadStoreSpec)
            success = self.worker.submit_store(job_id, src_spec, dst_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        if kv_connector_metadata.jobs_to_flush:
            self.worker.wait(kv_connector_metadata.jobs_to_flush)

    def start_kv_transfers(self, metadata: OffloadingConnectorMetadata):
        assert self.worker is not None
        for job_id, src_spec, dst_spec in self._unsubmitted_store_jobs:
            success = self.worker.submit_store(job_id, src_spec, dst_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        for job_id, entry in metadata.load_jobs.items():
            self._load_jobs[job_id] = entry.req_id
            assert isinstance(entry.dst_spec, GPULoadStoreSpec)
            success = self.worker.submit_load(job_id, entry.src_spec, entry.dst_spec)
            assert success

    def prepare_store_kv(self, metadata: OffloadingConnectorMetadata):
        for job_id, entry in metadata.store_jobs.items():
            # NOTE(orozery): defer the store to the beginning of the next
            # engine step, so that offloading starts AFTER transfers related
            # to token sampling, thereby avoiding delays to token generation.
            assert isinstance(entry.src_spec, GPULoadStoreSpec)
            self._unsubmitted_store_jobs.append(
                (job_id, entry.src_spec, entry.dst_spec)
            )

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """
        Returns:
            tuple of (finished_sending, finished_recving). Stores never
            emit finished_sending — the scheduler tracks store completion
            via kv_connector_worker_meta.completed_jobs and fences any
            block reuse via jobs_to_flush. Loads still emit
            finished_recving so the base scheduler can resume requests
            blocked on remote KV (and free aborted-during-load reqs).
        """
        assert self.worker is not None
        finished_recving: set[str] = set()
        for transfer_result in self.worker.get_finished():
            # we currently do not support job failures
            job_id = transfer_result.job_id
            assert transfer_result.success
            is_load = job_id in self._load_jobs
            if (
                transfer_result.transfer_time is not None
                and transfer_result.transfer_size is not None
            ):
                if is_load:
                    stats = self._connector_worker_meta.transfer_stats.load
                else:
                    stats = self._connector_worker_meta.transfer_stats.store
                stats.record(
                    transfer_result.transfer_size,
                    transfer_result.transfer_time,
                )

            self._connector_worker_meta.mark_completed(job_id)
            req_id = self._load_jobs.pop(job_id, None)
            if req_id is not None:
                finished_recving.add(req_id)

        return set(), finished_recving

    def build_connector_worker_meta(self) -> OffloadingWorkerMetadata | None:
        """Return completed transfer job IDs since the last call."""
        if not self._connector_worker_meta.completed_jobs:
            return None
        meta = self._connector_worker_meta
        self._connector_worker_meta = OffloadingWorkerMetadata()
        return meta

    def shutdown(self) -> None:
        self._unsubmitted_store_jobs.clear()
        self._load_jobs.clear()
        self._connector_worker_meta = OffloadingWorkerMetadata()
        if self.worker is not None:
            self.worker.shutdown()
