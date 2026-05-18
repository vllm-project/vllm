# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Object store secondary tier implementation."""

import hashlib
import json
from collections.abc import Iterable
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
from nixl._api import nixl_agent, nixl_agent_config, nixl_xfer_handle

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    OffloadKey,
    ReqContext,
    get_offload_block_hash,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.tiering.base import JobMetadata, JobResult, SecondaryTierManager
from vllm.v1.kv_offload.tiering.obj.config import ObjStoreConfig

if TYPE_CHECKING:
    from vllm.config import VllmConfig

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
        vllm_config: "VllmConfig",
        primary_kv_view: memoryview,
        store_config: dict,
        prefix: str = "",
        io_threads: int = 4,
    ):
        super().__init__(vllm_config, primary_kv_view)
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
        self._root_dir = f"{prefix}/" if prefix else ""
        self._base_path = self._compute_base_path(vllm_config, self._root_dir)
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

    @staticmethod
    def _compute_base_path(vllm_config: "VllmConfig", root_dir: str) -> str | None:
        """Compute a FileMapper-compatible base path from vllm_config.

        Uses parallel_agnostic convention (all TP/PP sizes=1, rank=0) since
        this tier stores the full interleaved CPU buffer, not per-rank slices.
        Returns None when vllm_config is unavailable (e.g. in tests).
        """
        if vllm_config is None:
            return None
        try:
            fields = {
                "model_name": vllm_config.model_config.model,
                "hash_block_size": vllm_config.cache_config.block_size,
                "gpu_blocks_per_file": 1,
                "tp_size": 1,
                "pp_size": 1,
                "pcp_size": 1,
                "dcp_size": 1,
                "dtype": str(vllm_config.cache_config.cache_dtype).replace("torch.", ""),
                "kv_cache_groups": [],
                "inference_engine": "vllm",
            }
            canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
            digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
            safe_model = fields["model_name"].replace("/", "_")
            return f"{root_dir}{safe_model}_{digest}"
        except Exception as e:
            logger.warning("Failed to compute base path from vllm_config: %s", e)
            return None

    def _get_obj_key(self, key: OffloadKey) -> str:
        hash_hex = get_offload_block_hash(key).hex()
        g = get_offload_group_idx(key)
        if self._base_path is not None:
            return f"{self._base_path}_r0/{hash_hex[:3]}/{hash_hex[3:5]}_g{g}/{hash_hex}.bin"
        # fallback when vllm_config is unavailable (tests)
        return f"{self._root_dir}group_{g}/{hash_hex[:3]}/{hash_hex[3:5]}/{hash_hex}.bin"

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
