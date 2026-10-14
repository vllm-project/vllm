"""
Worker-side hooks for vLLM's CPU offloading connectors.

There is no data to move, so stores and loads complete immediately and the
scheduler side sees host-cache hits normally.  Covers SimpleCPUOffloadWorker
(VLLM_USE_SIMPLE_KV_OFFLOAD=1) and OffloadingConnectorWorker.
"""

from vllm_simulator.hook import BaseHook
from vllm_simulator.utils import get_logger

logger = get_logger()


class C_VLLMSimpleCPUOffloadWorkerHook(BaseHook):
    """Hook SimpleCPUOffloadWorker to skip CUDA and auto-complete transfers."""

    HOOK_CLASS_NAME = "SimpleCPUOffloadWorker"
    HOOK_MODULE_NAME = "vllm.v1.simple_kv_offload.worker"

    @classmethod
    def hook(cls, target):
        def override_init(self, *args, **kwargs):
            """Skip pinned memory and CUDA streams."""
            self._connector_metadata = None
            self._completed_store_events: dict[int, int] = {}
            logger.info(
                "[vLLM Hijack] SimpleCPUOffloadWorker.__init__: "
                "skipped CUDA allocation (simulation mode)"
            )

        def override_register_kv_caches(self, kv_caches):
            """No-op: no actual KV cache tensors to register."""
            pass

        def override_bind_connector_metadata(self, metadata):
            """Keep the metadata without tracking CUDA events."""
            self._connector_metadata = metadata

        def override_handle_preemptions(self, kv_connector_metadata):
            """No-op: no in-flight transfers to flush."""
            pass

        def override_start_load_kv(self, *args, **kwargs):
            """No-op: loads auto-complete via override_get_finished."""
            pass

        def override_wait_for_save(self, *args, **kwargs):
            """No-op: stores auto-complete via override_get_finished."""
            pass

        def override_get_finished(self, finished_req_ids):
            """Report all loads finished and the store event completed.

            Returns (finished_sending, finished_recving).
            """
            meta = self._connector_metadata
            finished_recving = set()

            if meta is not None and meta.load_event >= 0:
                for reqs in meta.load_event_to_reqs.values():
                    finished_recving.update(reqs)

            if meta is not None and meta.store_event >= 0:
                self._completed_store_events[meta.store_event] = 1

            return None, finished_recving or None

        target.__init__ = override_init
        target.register_kv_caches = override_register_kv_caches
        target.bind_connector_metadata = override_bind_connector_metadata
        target.handle_preemptions = override_handle_preemptions
        target.start_load_kv = override_start_load_kv
        target.wait_for_save = override_wait_for_save
        target.get_finished = override_get_finished


class C_VLLMOffloadingConnectorWorkerHook(BaseHook):
    """Hook OffloadingConnectorWorker to auto-complete transfers.

    The native __init__ creates no CUDA resources; the transfer worker is
    built in register_kv_caches, which is skipped here.
    """

    HOOK_CLASS_NAME = "OffloadingConnectorWorker"
    HOOK_MODULE_NAME = "vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker"

    @classmethod
    def hook(cls, target):
        def override_register_kv_caches(self, kv_caches):
            """No-op: keeps the transfer worker unbuilt."""
            pass

        def override_handle_preemptions(self, kv_connector_metadata):
            """No-op: no in-flight transfers to flush."""
            pass

        def override_start_kv_transfers(self, metadata):
            """Record load jobs; get_finished completes them."""
            for job_id, entry in metadata.load_jobs.items():
                self._load_jobs[job_id] = entry.req_id

        def override_prepare_store_kv(self, metadata):
            """Mark store jobs completed immediately."""
            for job_id in metadata.store_jobs:
                self._connector_worker_meta.mark_completed(job_id)

        def override_get_finished(self, finished_req_ids):
            """Report all loads as immediately finished."""
            finished_recving: set[str] = set()
            for job_id, req_id in self._load_jobs.items():
                self._connector_worker_meta.mark_completed(job_id)
                finished_recving.add(req_id)
            self._load_jobs.clear()
            return set(), finished_recving

        target.register_kv_caches = override_register_kv_caches
        target.handle_preemptions = override_handle_preemptions
        target.start_kv_transfers = override_start_kv_transfers
        target.prepare_store_kv = override_prepare_store_kv
        target.get_finished = override_get_finished
