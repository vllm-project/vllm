# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OBJ (S3) secondary tier implementation."""

from collections.abc import Collection, Iterable

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext, get_offload_block_hash
from vllm.v1.kv_offload.tiering.base import JobMetadata, JobResult, SecondaryTierManager
from vllm.v1.kv_offload.tiering.obj.nixl_engine import NixlEngine, READ, WRITE
from vllm.v1.kv_offload.tiering.obj.nixl_lookup import NixlLookup
from vllm.v1.kv_offload.tiering.obj.obj_store_config import ObjStoreConfig

logger = init_logger(__name__)


class ObjectStoreSecondaryTierManager(SecondaryTierManager):
    """Secondary tier that offloads KV cache blocks to an S3-compatible store.

    Handles CPU DRAM <-> S3 transfers only. GPU <-> CPU is managed by the
    primary tier. Object keys are formed as ``{prefix}/{hash_shard}/{hash}.bin``.
    """

    def __init__(
        self,
        obj_config: ObjStoreConfig,
        prefix: str = "",
        io_threads: int = 4,
    ):
        self._engine = NixlEngine(obj_config, io_threads=io_threads)
        self._nixl_lookup = NixlLookup(obj_config)
        self._prefix = f"{prefix}/" if prefix else ""

        self._pending_jobs: set[int] = set()

        self._probe_connectivity()

    def _probe_connectivity(self) -> None:
        """Verify S3 connectivity at startup via a NIXL lookup probe.

        Performs a single exists() check against a synthetic key that will
        never exist. A True/False result confirms the bucket is reachable;
        an exception indicates misconfigured S3 params and raises RuntimeError.
        """
        probe_key = "__nixl_probe__/connectivity_test"
        try:
            self._nixl_lookup.exists(probe_key)
            logger.info("OBJ tier S3 connectivity probe succeeded")
        except Exception as e:
            raise RuntimeError(
                f"OBJ tier S3 connectivity probe failed — check bucket, "
                f"endpoint_override, access_key, secret_key, and scheme. "
                f"Error: {e}"
            ) from e

    def set_primary_view(self, view: memoryview) -> None:
        self._engine.set_primary_view(view)

    def _get_obj_key(self, block_hash: bytes) -> str:
        h = block_hash[-8:].hex()
        return f"{self._prefix}{h[:3]}/{h[3:5]}/{h}.bin"

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        s3_key = self._get_obj_key(get_offload_block_hash(key))
        return self._nixl_lookup.exists(s3_key)

    def submit_store(self, job_metadata: JobMetadata) -> None:
        s3_keys = (
            self._get_obj_key(get_offload_block_hash(k))
            for k in job_metadata.keys
        )
        ok = self._engine.submit_transfer(
            job_metadata.job_id, job_metadata.block_ids, s3_keys, WRITE
        )
        if ok:
            self._pending_jobs.add(job_metadata.job_id)

    def submit_load(self, job_metadata: JobMetadata) -> None:
        s3_keys = (
            self._get_obj_key(get_offload_block_hash(k))
            for k in job_metadata.keys
        )
        ok = self._engine.submit_transfer(
            job_metadata.job_id, job_metadata.block_ids, s3_keys, READ
        )
        if ok:
            self._pending_jobs.add(job_metadata.job_id)

    def get_finished(self) -> Iterable[JobResult]:
        for job_id, success in self._engine.get_finished():
            if job_id not in self._pending_jobs:
                continue
            self._pending_jobs.discard(job_id)
            yield JobResult(job_id=job_id, success=success)

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        pass

    def shutdown(self) -> None:
        self._engine.shutdown()

    @classmethod
    def from_config(cls, config: dict) -> "ObjectStoreSecondaryTierManager":
        config = config.copy()
        obj_config = ObjStoreConfig(
            bucket=config.pop("bucket"),
            endpoint_override=config.pop("endpoint_override"),
            access_key=config.pop("access_key"),
            secret_key=config.pop("secret_key"),
            scheme=config.pop("scheme", "http"),
            ca_bundle=config.pop("ca_bundle", ""),
        )
        prefix = config.pop("prefix", "")
        io_threads = config.pop("io_threads", 4)
        return cls(obj_config=obj_config, prefix=prefix, io_threads=io_threads)

    @staticmethod
    def get_tier_type() -> str:
        return "obj"
