# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict

import torch

from vllm.attention.layer import Attention
from vllm.config import get_layers_from_vllm_config
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.worker import OffloadingWorker, TransferSpec

from .metadata import ReqId, LoomConnectorMetadata


def _get_forward_context_arg(_forward_context: object) -> None:
    # Avoid importing ForwardContext at runtime.
    return None


class LoomConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, spec: OffloadingSpec):
        self.spec = spec
        self.worker = OffloadingWorker()

        loom_cfg = getattr(spec, "loom_config", None)
        self._loom_load_only: bool = bool(getattr(loom_cfg, "loom_load_only", False))

        self._job_counter = 0

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
            self.spec.vllm_config, Attention, layer_names
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

    def handle_preemptions(self, preempted_req_ids: set[str]):
        if self._loom_load_only:
            return
        for job_id, transfer_spec in self._unsubmitted_store_jobs:
            success = self.worker.transfer_async(job_id, transfer_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        for req_id in preempted_req_ids:
            job_ids = self._store_jobs.get(req_id)
            if job_ids:
                self.worker.wait(job_ids)

    def start_kv_transfers(self, metadata: LoomConnectorMetadata, forward_context: object):
        _get_forward_context_arg(forward_context)
        if not self._loom_load_only:
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

    def prepare_store_kv(self, metadata: LoomConnectorMetadata):
        if self._loom_load_only:
            return
        for req_id, transfer_spec in metadata.reqs_to_store.items():
            job_id = self._generate_job_id()
            self._jobs[job_id] = (req_id, True)
            self._store_jobs[req_id].add(job_id)
            # NOTE(orozery): defer the store to the beginning of the next engine step,
            # so that offloading starts AFTER transfers related to token sampling,
            # thereby avoiding delays to token generation due to offloading.
            self._unsubmitted_store_jobs.append((job_id, transfer_spec))

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        finished_sending = set()
        finished_recving = set()
        if self._loom_load_only:
            finished_req_ids = set()
        for job_id, success in self.worker.get_finished():
            # we currently do not support job failures
            assert success
            req_id, store = self._jobs.pop(job_id)
            if store:
                if self._loom_load_only:
                    continue
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
