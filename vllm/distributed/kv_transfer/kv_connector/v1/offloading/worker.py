# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict

import torch

from vllm.config import get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    ReqId,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_offload.spec import OffloadingSpec
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

        self._job_counter = 0

        self.kv_connector_stats = OffloadingConnectorStats()
        # req_id -> (job_id, store)
        self._jobs: dict[int, tuple[ReqId, bool]] = {}
        # req_id -> active job IDs
        self._load_job: dict[ReqId, int] = {}
        # req_id -> set(active job IDs)
        self._store_jobs = defaultdict[ReqId, set[int]](set)
        # list of store jobs pending submission (job_id, transfer_spec)
        self._unsubmitted_store_jobs: list[tuple[int, TransferSpec]] = []

        self._finished_reqs_waiting_for_store: set[ReqId] = set()

    def _generate_job_id(self) -> int:
        job_id = self._job_counter
        self._job_counter = job_id + 1
        return job_id

    def _register_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ):
        for src_cls, dst_cls, handler in self.spec.get_handlers(
            kv_caches, attn_backends
        ):
            self.worker.register_handler(src_cls, dst_cls, handler)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        layer_names = list(kv_caches.keys())
        layers = get_layers_from_vllm_config(
            self.spec.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
            layer_names,
        )
        attn_backends = {
            layer_name: layers[layer_name].get_attn_backend()
            for layer_name in layer_names
        }
        self._register_handlers(kv_caches, attn_backends)

    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ):
        cross_layer_name = "ALL_LAYERS"
        kv_caches = {cross_layer_name: kv_cache}
        attn_backends = {cross_layer_name: attn_backend}
        self._register_handlers(kv_caches, attn_backends)

    def handle_preemptions(self, kv_connector_metadata: OffloadingConnectorMetadata):
        for job_id, transfer_spec in self._unsubmitted_store_jobs:
            success = self.worker.transfer_async(job_id, transfer_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        for req_id in kv_connector_metadata.reqs_to_flush or ():
            job_ids = self._store_jobs.get(req_id)
            if job_ids:
                self.worker.wait(job_ids)

    def start_kv_transfers(self, metadata: OffloadingConnectorMetadata):
        for job_id, transfer_spec in self._unsubmitted_store_jobs:
            success = self.worker.transfer_async(job_id, transfer_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        for req_id, transfer_spec in metadata.reqs_to_load.items():
            job_id = self._generate_job_id()
            self._jobs[job_id] = (req_id, False)
            assert req_id not in self._load_job
            self._load_job[req_id] = job_id
            success = self.worker.transfer_async(job_id, transfer_spec)
            assert success

    def prepare_store_kv(self, metadata: OffloadingConnectorMetadata):
        for req_id, transfer_spec in metadata.reqs_to_store.items():
            job_id = self._generate_job_id()
            self._jobs[job_id] = (req_id, True)
            self._store_jobs[req_id].add(job_id)
            # NOTE(orozery): defer the store to the beginning of the next engine step,
            # so that offloading starts AFTER transfers related to token sampling,
            # thereby avoiding delays to token generation due to offloading.
            self._unsubmitted_store_jobs.append((job_id, transfer_spec))

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.
        Returns a list of request IDs that finished loading or storing.

        Returns:
            ids of requests that have finished asynchronous transfer
            tuple of (sending/saving ids, recving/loading ids).
        """
        finished_sending = set()
        finished_recving = set()
        for transfer_result in self.worker.get_finished():
            # we currently do not support job failures
            job_id = transfer_result.job_id
            assert transfer_result.success
            req_id, store = self._jobs.pop(job_id)
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
            if store:
                req_jobs = self._store_jobs[req_id]
                req_jobs.remove(job_id)
                if req_jobs:
                    continue

                if req_id in self._finished_reqs_waiting_for_store:
                    self._finished_reqs_waiting_for_store.remove(req_id)
                    finished_sending.add(req_id)
                    del self._store_jobs[req_id]
            else:
                req_job = self._load_job[req_id]
                assert job_id == req_job
                del self._load_job[req_id]
                finished_recving.add(req_id)

        for req_id in finished_req_ids:
            pending_req_jobs = self._store_jobs.get(req_id)
            if pending_req_jobs:
                self._finished_reqs_waiting_for_store.add(req_id)
            elif pending_req_jobs is not None:
                finished_sending.add(req_id)
                del self._store_jobs[req_id]

        return finished_sending, finished_recving

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
