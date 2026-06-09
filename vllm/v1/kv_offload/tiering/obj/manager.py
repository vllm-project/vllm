# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Object store secondary tier implementation."""

import ctypes
from collections.abc import Iterable
from typing import TYPE_CHECKING, NamedTuple

from vllm.distributed.nixl_utils import NixlWrapper as nixl_agent
from vllm.distributed.nixl_utils import nixl_agent_config
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.base import (
    JobMetadata,
    JobResult,
    RequestOffloadingContext,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.obj.config import ObjStoreConfig

if TYPE_CHECKING:
    from nixl._api import nixl_prepped_dlist_handle, nixl_xfer_handle

    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)

NIXL_WRITE = "WRITE"
NIXL_READ = "READ"
NIXL_PROC = "PROC"
NIXL_DONE = "DONE"

# Device ID for CPU DRAM descriptors. DRAM is not a multi-device resource so
# the device ID is always 0.
NIXL_DEV_ID: int = 0

# Fields for NIXL OBJ descriptors: (addr, len, dev_id, obj_key).
# For existence probes addr and len are placeholders — no data is read.
# dev_id=0 is reserved for probes; transfers start from 1.
_PROBE_ADDR: int = 0
_PROBE_LEN: int = 1
_PROBE_DEV_ID: int = 0


class TransferEntry(NamedTuple):
    xfer_handle: "nixl_xfer_handle"
    files_desc: object
    obj_handle: "nixl_prepped_dlist_handle"


class ObjectStoreSecondaryTierManager(SecondaryTierManager):
    """Secondary tier that offloads KV cache blocks to an S3-compatible store.

    Handles CPU DRAM <-> S3 transfers only. GPU <-> CPU is managed by the
    primary tier. Object keys are formed as ``{prefix}/{hash_shard}/{hash}.bin``.
    """

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        store_config: dict,
        prefix: str = "",
        io_threads: int = 4,
    ):
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        agent_config = nixl_agent_config(backends=[])
        self._agent = nixl_agent("ObjAgent", agent_config)
        obj_config = ObjStoreConfig(**store_config)
        params = {**obj_config.to_nixl_params(), "num_threads": str(io_threads)}
        self._agent.create_backend("OBJ", params)
        self._transfers: dict[int, TransferEntry] = {}
        self._failed_jobs: list[JobResult] = []
        self._primary_reg = None
        self._block_size_bytes: int = 0
        root_dir = f"{prefix}/" if prefix else ""
        self._file_mapper = FileMapper.from_offloading_spec(root_dir, offloading_spec)
        self._next_obj_dev_id: int = 1  # dev_id=0 is reserved for _exists() probes

        self._probe_connectivity()

        base_addr = ctypes.addressof(ctypes.c_char.from_buffer(primary_kv_view))
        assert primary_kv_view.strides is not None
        stride = primary_kv_view.strides[0]
        self._primary_reg = self._agent.register_memory(
            [(base_addr, primary_kv_view.nbytes, NIXL_DEV_ID, "")], "DRAM"
        )
        self._block_size_bytes = stride
        all_blocks = [
            (base_addr + i * stride, stride, NIXL_DEV_ID)
            for i in range(len(primary_kv_view))
        ]
        # NIXL_INIT_AGENT marks this as the local side; make_prepped_xfer requires
        # local_xfer_side tagged with NIXL_INIT_AGENT and remote_xfer_side tagged
        # with the peer agent name ("ObjAgent").
        self._dram_prepped_handle: nixl_prepped_dlist_handle = (
            self._agent.prep_xfer_dlist("NIXL_INIT_AGENT", all_blocks, "DRAM")
        )

    def _probe_connectivity(self) -> None:
        """Verify object store connectivity at startup via a NIXL lookup probe.

        Performs a single exists() check against a synthetic key that will
        never exist. A True/False result confirms the bucket is reachable;
        an exception indicates misconfigured obj store params and raises RuntimeError.
        """
        probe_key = "__nixl_probe__/connectivity_test"
        try:
            self._exists(probe_key)
            logger.info("Object store tier connectivity probe succeeded")
        except Exception as e:
            raise RuntimeError(
                f"Object store tier connectivity probe failed — check bucket, "
                f"endpoint_override, access_key, secret_key, and scheme. "
                f"Error: {e}"
            ) from e

    def _exists(self, obj_key: str) -> bool:
        results = self._agent.query_memory(
            [(_PROBE_ADDR, _PROBE_LEN, _PROBE_DEV_ID, obj_key)], "OBJ", "OBJ"
        )
        return results[0] is not None

    def _submit_transfer(
        self,
        job_id: int,
        block_ids: Iterable[int],
        obj_keys: Iterable[str],
        op: str,
    ) -> None:
        """Submit an async transfer. op is 'WRITE' (store) or 'READ' (load)."""
        block_ids_list = [int(bid) for bid in block_ids]
        # The OBJ backend maps devId -> obj_key. All descriptors must have
        # unique devIds or later registrations overwrite earlier ones.
        nixl_files = [
            (0, self._block_size_bytes, dev_id, key)
            for dev_id, key in enumerate(obj_keys, self._next_obj_dev_id)
        ]
        self._next_obj_dev_id += len(nixl_files)

        files_desc = self._agent.register_memory(nixl_files, "OBJ")
        if files_desc is None:
            logger.warning("register_memory (OBJ) failed for job %d", job_id)
            self._failed_jobs.append(JobResult(job_id=job_id, success=False))
            return

        obj_handle = self._agent.prep_xfer_dlist("ObjAgent", files_desc.trim())
        if not obj_handle:
            logger.warning("prep_xfer_dlist (OBJ) failed for job %d", job_id)
            self._agent.deregister_memory(files_desc)
            self._failed_jobs.append(JobResult(job_id=job_id, success=False))
            return

        xfer_handle = self._agent.make_prepped_xfer(
            op,
            self._dram_prepped_handle,
            block_ids_list,
            obj_handle,
            list(range(len(nixl_files))),
        )
        if not xfer_handle:
            logger.warning("make_prepped_xfer failed for job %d", job_id)
            self._agent.release_dlist_handle(obj_handle)
            self._agent.deregister_memory(files_desc)
            self._failed_jobs.append(JobResult(job_id=job_id, success=False))
            return

        state = self._agent.transfer(xfer_handle)
        if state == "ERR":
            logger.warning("agent.transfer failed for job %d", job_id)
            self._agent.release_dlist_handle(obj_handle)
            self._agent.deregister_memory(files_desc)
            self._agent.release_xfer_handle(xfer_handle)
            self._failed_jobs.append(JobResult(job_id=job_id, success=False))
            return

        self._transfers[job_id] = TransferEntry(xfer_handle, files_desc, obj_handle)

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        try:
            return self._exists(self._file_mapper.get_file_name(key))
        except Exception as e:
            logger.warning("lookup failed for key %s: %s", key, e)
            return False

    def submit_store(self, job_metadata: JobMetadata) -> None:
        obj_keys = (self._file_mapper.get_file_name(k) for k in job_metadata.keys)
        self._submit_transfer(
            job_metadata.job_id, job_metadata.block_ids, obj_keys, NIXL_WRITE
        )

    def submit_load(self, job_metadata: JobMetadata) -> None:
        obj_keys = (self._file_mapper.get_file_name(k) for k in job_metadata.keys)
        self._submit_transfer(
            job_metadata.job_id, job_metadata.block_ids, obj_keys, NIXL_READ
        )

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def get_finished_jobs(self) -> Iterable[JobResult]:
        """Poll in-flight transfers; return completed (job_id, success) pairs."""
        results: list[JobResult] = self._failed_jobs
        self._failed_jobs = []
        for job_id, entry in list(self._transfers.items()):
            try:
                state = self._agent.check_xfer_state(entry.xfer_handle)
            except Exception as exc:
                success = False
                logger.warning("check_xfer_state raised for job %d: %s", job_id, exc)
            else:
                if state == NIXL_PROC:
                    continue
                elif state == NIXL_DONE:
                    success = True
                else:
                    success = False
                    logger.warning("transfer failed job=%d state=%s", job_id, state)
            del self._transfers[job_id]
            self._agent.release_xfer_handle(entry.xfer_handle)
            self._agent.release_dlist_handle(entry.obj_handle)
            self._agent.deregister_memory(entry.files_desc)
            results.append(JobResult(job_id=job_id, success=success))
        return results

    def shutdown(self) -> None:
        for job_id, entry in self._transfers.items():
            try:
                self._agent.release_xfer_handle(entry.xfer_handle)
            except Exception as exc:
                logger.warning("release_xfer_handle failed for job %d: %s", job_id, exc)
            try:
                self._agent.release_dlist_handle(entry.obj_handle)
            except Exception as exc:
                logger.warning(
                    "release_dlist_handle failed for job %d: %s", job_id, exc
                )
            try:
                self._agent.deregister_memory(entry.files_desc)
            except Exception as exc:
                logger.warning("deregister_memory failed for job %d: %s", job_id, exc)
        self._transfers.clear()
        if self._dram_prepped_handle is not None:
            try:
                self._agent.release_dlist_handle(self._dram_prepped_handle)
            except Exception as exc:
                logger.warning("failed to release DRAM prepped handle: %s", exc)
            self._dram_prepped_handle = None
        if self._primary_reg is not None:
            try:
                self._agent.deregister_memory(self._primary_reg)
            except Exception as exc:
                logger.warning("failed to deregister primary buffer: %s", exc)
            self._primary_reg = None
