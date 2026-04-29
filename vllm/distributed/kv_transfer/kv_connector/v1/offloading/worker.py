# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict
from dataclasses import replace

import torch

from vllm.config import get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    ReqId,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.spec import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    OffloadingSpec,
)
from vllm.v1.kv_offload.worker.worker import (
    OffloadingWorker,
    TransferSpec,
)

logger = init_logger(__name__)


class OffloadingConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, spec: OffloadingSpec):
        self.spec = spec
        self.worker = OffloadingWorker()

        self.kv_connector_stats = OffloadingConnectorStats()
        # job_id -> req_id for in-flight loads.
        self._load_jobs: dict[int, ReqId] = {}
        self._unsubmitted_store_jobs: list[tuple[int, TransferSpec]] = []
        self._connector_worker_meta = OffloadingWorkerMetadata()

    def _register_handlers(self, kv_caches: CanonicalKVCaches):
        for src_cls, dst_cls, handler in self.spec.get_handlers(kv_caches):
            self.worker.register_handler(src_cls, dst_cls, handler)

    def register_kv_caches(
        self, kv_caches: dict[str, torch.Tensor | list[torch.Tensor]]
    ):
        layer_names = list(kv_caches.keys())
        layers = get_layers_from_vllm_config(
            self.spec.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
            layer_names,
        )
        attn_backends = {
            layer_name: layers[layer_name].get_attn_backend()
            for layer_name in layer_names
            if layer_name in layers
        }

        num_blocks = self.spec.kv_cache_config.num_blocks

        # layer_name -> list of matching KV cache tensors
        # such that each tensor starts with the num_blocks dimension.
        # FlashAttention layers which use the (2, num_blocks, ...) layout
        # will possibly map to 2 tensors, one per K and one per V.
        # All other layers will probably map to a single tensor.
        tensors_per_block: dict[str, tuple[torch.Tensor, ...]] = {}
        # layer_name -> size of (un-padded) page in bytes
        unpadded_page_size_bytes: dict[str, int] = {}
        # layer_name -> size of page in bytes
        page_size_bytes: dict[str, int] = {}
        for kv_cache_group in self.spec.kv_cache_config.kv_cache_groups:
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
                if isinstance(layer_kv_cache_spec, AttentionSpec):
                    layer_kv_cache = kv_caches[layer_name]
                    assert isinstance(layer_kv_cache, torch.Tensor)
                    assert layer_kv_cache.storage_offset() == 0

                    # get the logical dimension for num_blocks
                    test_shape = attn_backends[layer_name].get_kv_cache_shape(
                        num_blocks=1234,
                        block_size=16,
                        num_kv_heads=1,
                        head_size=256,
                    )
                    num_blocks_logical_dim = test_shape.index(1234)

                    # sort the logical dimensions by stride (high to low)
                    # to get a physical-to-logical mapping:
                    # physical_to_logical[physical_pos] = logical_dim
                    logical_strides = layer_kv_cache.stride()
                    physical_to_logical = sorted(
                        range(len(logical_strides)),
                        key=lambda idx: logical_strides[idx],
                        reverse=True,
                    )

                    num_blocks_physical_dim = physical_to_logical.index(
                        num_blocks_logical_dim
                    )
                    if num_blocks_physical_dim == 0:
                        storage = layer_kv_cache.untyped_storage()
                        page = layer_kv_cache_spec.page_size_bytes
                        tensors_per_block[layer_name] = (
                            torch.tensor(
                                [],
                                dtype=torch.int8,
                                device=layer_kv_cache.device,
                            )
                            .set_(storage)
                            .view(num_blocks, page),
                        )
                        page_size_bytes[layer_name] = (
                            layer_kv_cache_spec.page_size_bytes
                        )
                    else:
                        # Flash Attention case: (2, num_blocks, ...)
                        assert test_shape[0] == 2
                        assert physical_to_logical[0] == 0
                        assert num_blocks_physical_dim == 1

                        # unbind the tensor to separate K and V tensors
                        half_page_size = layer_kv_cache_spec.page_size_bytes // 2
                        storage = layer_kv_cache.untyped_storage()
                        raw = (
                            torch.tensor(
                                [],
                                dtype=torch.int8,
                                device=layer_kv_cache.device,
                            )
                            .set_(storage)
                            .view(2, num_blocks, half_page_size)
                        )
                        tensors_per_block[layer_name] = tuple(raw.unbind(0))

                        page_size_bytes[layer_name] = half_page_size

                    unpadded_page_size_bytes[layer_name] = page_size_bytes[layer_name]

                elif isinstance(layer_kv_cache_spec, MambaSpec):
                    state_tensors = kv_caches[layer_name]
                    assert isinstance(state_tensors, list)

                    # re-construct the raw (num_blocks, page_size) tensor
                    # from the first state tensor
                    assert len(state_tensors) > 0
                    first_state_tensor = state_tensors[0]
                    assert first_state_tensor.storage_offset() == 0
                    tensor = (
                        torch.tensor(
                            [],
                            dtype=torch.int8,
                            device=first_state_tensor.device,
                        )
                        .set_(first_state_tensor.untyped_storage())
                        .view((num_blocks, layer_kv_cache_spec.page_size_bytes))
                    )
                    tensors_per_block[layer_name] = (tensor,)

                    page_size_bytes[layer_name] = layer_kv_cache_spec.page_size_bytes
                    unpadded_page_size_bytes[layer_name] = replace(
                        layer_kv_cache_spec, page_size_padded=None
                    ).page_size_bytes

                else:
                    raise NotImplementedError

        block_tensors: list[CanonicalKVCacheTensor] = []
        block_data_refs: dict[str, list[CanonicalKVCacheRef]] = defaultdict(list)
        for kv_cache_tensor in self.spec.kv_cache_config.kv_cache_tensors:
            tensor_layer_names = kv_cache_tensor.shared_by

            # verify all layers in the group reference the exact same tensors
            assert len({len(tensors_per_block[n]) for n in tensor_layer_names}) == 1
            assert (
                len({tensors_per_block[n][0].data_ptr() for n in tensor_layer_names})
                == 1
            )
            assert (
                len({tensors_per_block[n][0].stride() for n in tensor_layer_names}) == 1
            )

            # pick the first layer to represent the group
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
        for kv_cache_group in self.spec.kv_cache_config.kv_cache_groups:
            group_refs: list[CanonicalKVCacheRef] = []
            for layer_name in kv_cache_group.layer_names:
                group_refs += block_data_refs[layer_name]
            group_data_refs.append(group_refs)

        canonical_kv_caches = CanonicalKVCaches(
            tensors=block_tensors,
            group_data_refs=group_data_refs,
        )

        self._register_handlers(canonical_kv_caches)

    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ):
        # verify that num_blocks is at physical position 0 in the cross-layers
        # tensor layout.
        test_shape = attn_backend.get_kv_cache_shape(
            num_blocks=1234, block_size=16, num_kv_heads=1, head_size=256
        )
        num_blocks_logical_dim = test_shape.index(1234) + 1
        physical_to_logical = attn_backend.get_kv_cache_stride_order(
            include_num_layers_dimension=True
        )
        num_blocks_physical_dim = physical_to_logical.index(num_blocks_logical_dim)
        assert num_blocks_physical_dim == 0

        kv_cache_groups = self.spec.kv_cache_config.kv_cache_groups
        assert len(kv_cache_groups) == 1
        kv_cache_spec = kv_cache_groups[0].kv_cache_spec
        num_layers = len(kv_cache_groups[0].layer_names)
        page_size_bytes = kv_cache_spec.page_size_bytes * num_layers

        assert kv_cache.storage_offset() == 0
        storage = kv_cache.untyped_storage()
        assert len(storage) % page_size_bytes == 0
        num_blocks = len(storage) // page_size_bytes
        tensor = (
            torch.tensor(
                [],
                dtype=torch.int8,
                device=kv_cache.device,
            )
            .set_(storage)
            .view(num_blocks, page_size_bytes)
        )
        kv_cache_tensor = CanonicalKVCacheTensor(
            tensor=tensor, page_size_bytes=page_size_bytes
        )
        # in cross layers layout, there's currently only a single group
        kv_cache_data_ref = CanonicalKVCacheRef(
            tensor_idx=0, page_size_bytes=page_size_bytes
        )
        canonical_kv_caches = CanonicalKVCaches(
            tensors=[kv_cache_tensor], group_data_refs=[[kv_cache_data_ref]]
        )

        self._register_handlers(canonical_kv_caches)

    def handle_preemptions(self, kv_connector_metadata: OffloadingConnectorMetadata):
        for job_id, transfer_spec in self._unsubmitted_store_jobs:
            success = self.worker.transfer_async(job_id, transfer_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        if kv_connector_metadata.jobs_to_flush:
            self.worker.wait(kv_connector_metadata.jobs_to_flush)

    def start_kv_transfers(self, metadata: OffloadingConnectorMetadata):
        for job_id, transfer_spec in self._unsubmitted_store_jobs:
            success = self.worker.transfer_async(job_id, transfer_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        for job_id, entry in metadata.load_jobs.items():
            self._load_jobs[job_id] = entry.req_id
            success = self.worker.transfer_async(job_id, entry.transfer_spec)
            assert success

    def prepare_store_kv(self, metadata: OffloadingConnectorMetadata):
        for job_id, entry in metadata.store_jobs.items():
            # NOTE(orozery): defer the store to the beginning of the next
            # engine step, so that offloading starts AFTER transfers related
            # to token sampling, thereby avoiding delays to token generation.
            self._unsubmitted_store_jobs.append((job_id, entry.transfer_spec))

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
        finished_recving: set[str] = set()
        for transfer_result in self.worker.get_finished():
            # we currently do not support job failures
            job_id = transfer_result.job_id
            assert transfer_result.success
            if (
                transfer_result.transfer_time
                and transfer_result.transfer_size is not None
                and transfer_result.transfer_type is not None
            ):
                self.kv_connector_stats.record_transfer(
                    num_bytes=transfer_result.transfer_size,
                    time=transfer_result.transfer_time,
                    transfer_type=transfer_result.transfer_type,
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

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """
        Get the KV transfer stats for the connector.
        """

        if self.kv_connector_stats.is_empty():
            return None
        # Clear stats for next iteration
        kv_connector_stats = self.kv_connector_stats
        self.kv_connector_stats = OffloadingConnectorStats()
        return kv_connector_stats

    def shutdown(self) -> None:
        self._unsubmitted_store_jobs.clear()
        self._load_jobs.clear()
        self._connector_worker_meta = OffloadingWorkerMetadata()
        self.worker.shutdown()
