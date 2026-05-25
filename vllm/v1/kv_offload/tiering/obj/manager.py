# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Object store secondary tier implementation."""

from collections.abc import Iterable
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
from nixl._api import nixl_agent, nixl_agent_config, nixl_xfer_handle

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.base import JobMetadata, JobResult, SecondaryTierManager
from vllm.v1.kv_offload.tiering.obj.config import ObjStoreConfig

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)

NIXL_WRITE = "WRITE"
NIXL_READ = "READ"
NIXL_PROC = "PROC"
NIXL_DONE = "DONE"

class TransferEntry(NamedTuple):
    xfer_handle: nixl_xfer_handle
    files_desc: object


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
        self._primary_tensor: torch.Tensor | None = None
        self._primary_reg = None
        self._base_addr: int = 0
        self._stride: int = 0
        root_dir = f"{prefix}/" if prefix else ""
        self._file_mapper = FileMapper.from_offloading_spec(root_dir, offloading_spec)
        self._next_obj_dev_id: int = 0  # unique devId for each OBJ registration

        self._probe_connectivity()
        self.set_primary_view(primary_kv_view)

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

    def set_primary_view(self, view: memoryview) -> None:
        """Register the entire primary CPU buffer with NIXL once."""
        np_arr = np.asarray(view)
        self._primary_tensor = torch.as_tensor(np_arr)
        self._primary_reg = self._agent.register_memory([self._primary_tensor])
        self._base_addr = self._primary_tensor.data_ptr()
        self._stride = view.strides[0]

    def _exists(self, obj_key: str) -> bool:
        return self._agent.query_memory([(0, 1, 0, obj_key)], "OBJ", "OBJ")[0] is not None

    def _get_obj_key(self, key: OffloadKey) -> str:
        return self._file_mapper.get_file_name(key)

    def _submit_transfer(
        self,
        job_id: int,
        block_ids: Iterable[int],
        obj_keys: Iterable[str],
        op: str,
    ) -> bool:
        """Submit an async transfer. op is 'WRITE' (store) or 'READ' (load)."""
        blocks_data = [
            (self._base_addr + int(bid) * self._stride, self._stride, 0)
            for bid in block_ids
        ]
        # The OBJ backend maps devId -> obj_key. All descriptors must have
        # unique devIds or later registrations overwrite earlier ones.
        dev_id_base = self._next_obj_dev_id
        obj_keys_list = list(obj_keys)
        self._next_obj_dev_id += len(obj_keys_list)
        nixl_files = [(0, self._stride, dev_id, key)
                      for dev_id, key in enumerate(obj_keys_list, dev_id_base)]

        xfer_desc = self._agent.get_xfer_descs(blocks_data, "DRAM")
        if xfer_desc is None:
            logger.warning("get_xfer_descs failed for job %d", job_id)
            return False

        files_desc = self._agent.register_memory(nixl_files, "OBJ")
        if files_desc is None:
            logger.warning("register_memory (OBJ) failed for job %d", job_id)
            return False

        xfer_handle = self._agent.initialize_xfer(
            op, xfer_desc, files_desc.trim(), "ObjAgent"
        )
        if not xfer_handle:
            logger.warning("initialize_xfer failed for job %d", job_id)
            self._agent.deregister_memory(files_desc)
            return False

        state = self._agent.transfer(xfer_handle)
        if state == "ERR":
            logger.warning("agent.transfer failed for job %d", job_id)
            self._agent.deregister_memory(files_desc)
            self._agent.release_xfer_handle(xfer_handle)
            return False

        self._transfers[job_id] = TransferEntry(xfer_handle, files_desc)
        return True

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        try:
            return self._exists(self._get_obj_key(key))
        except Exception as e:
            logger.warning("lookup failed for key %s: %s", key, e)
            return False

    def submit_store(self, job_metadata: JobMetadata) -> None:
        obj_keys = (self._get_obj_key(k) for k in job_metadata.keys)
        self._submit_transfer(job_metadata.job_id, job_metadata.block_ids, obj_keys, NIXL_WRITE)

    def submit_load(self, job_metadata: JobMetadata) -> None:
        obj_keys = (self._get_obj_key(k) for k in job_metadata.keys)
        self._submit_transfer(job_metadata.job_id, job_metadata.block_ids, obj_keys, NIXL_READ)

    def get_finished(self) -> Iterable[JobResult]:
        """Poll in-flight transfers; return completed (job_id, success) pairs."""
        results: list[JobResult] = []
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
                self._agent.deregister_memory(entry.files_desc)
            except Exception as exc:
                logger.warning("deregister_memory failed for job %d: %s", job_id, exc)
        self._transfers.clear()
        if self._primary_reg is not None:
            try:
                self._agent.deregister_memory(self._primary_reg)
            except Exception as exc:
                logger.warning("failed to deregister primary buffer: %s", exc)
            self._primary_reg = None
        self._primary_tensor = None

    @staticmethod
    def get_tier_type() -> str:
        return "obj"
