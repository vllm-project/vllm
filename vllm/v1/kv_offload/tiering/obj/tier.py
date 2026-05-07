# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OBJ (S3) secondary tier implementation."""

from collections.abc import Collection, Iterable

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext, get_offload_block_hash
from vllm.v1.kv_offload.tiering.base import JobMetadata, JobResult, SecondaryTierManager
from vllm.v1.kv_offload.tiering.obj.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.obj.nixl_engine import NixlEngine
from vllm.v1.kv_offload.tiering.obj.nixl_lookup import NixlLookup

logger = init_logger(__name__)


class ObjSecondaryTier(SecondaryTierManager):
    """Secondary tier that offloads KV cache blocks to an S3-compatible store.

    Handles CPU DRAM <-> S3 transfers only. GPU <-> CPU is managed by the
    primary tier (CpuGpuOffloadingHandlers from PR 40020).
    """

    def __init__(
        self,
        model_name: str,
        gpu_block_size: int,
        tp_size: int,
        pp_size: int,
        pcp_size: int,
        rank: int,
        dtype: str,
        key_prefix: str = "",
        io_threads: int = 4,
        *,
        bucket: str,
        endpoint_override: str,
        access_key: str,
        secret_key: str,
        scheme: str = "http",
        ca_bundle: str = "",
    ):
        s3_params = dict(
            bucket=bucket,
            endpoint_override=endpoint_override,
            access_key=access_key,
            secret_key=secret_key,
            scheme=scheme,
            ca_bundle=ca_bundle,
        )
        self._engine = NixlEngine(**s3_params, io_threads=io_threads)
        self._file_mapper = FileMapper(
            model_name=model_name,
            gpu_block_size=gpu_block_size,
            tp_size=tp_size,
            pp_size=pp_size,
            pcp_size=pcp_size,
            rank=rank,
            dtype=dtype,
            key_prefix=key_prefix,
        )
        self._nixl_lookup = NixlLookup(**s3_params)

        # Keys currently being promoted S3 -> CPU (submit_load in flight)
        self._load_in_flight: set[OffloadKey] = set()
        # Store jobs: only need to know they exist, not their keys
        self._pending_stores: set[int] = set()
        # Load jobs: keys needed to clear _load_in_flight on completion
        self._pending_loads: dict[int, list[OffloadKey]] = {}

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

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        if key in self._load_in_flight:
            return None
        s3_key = self._file_mapper.get_key(get_offload_block_hash(key))
        return self._nixl_lookup.exists(s3_key)

    def submit_store(self, job_metadata: JobMetadata) -> None:
        s3_keys = [
            self._file_mapper.get_key(get_offload_block_hash(k))
            for k in job_metadata.keys
        ]
        ok = self._engine.submit_transfer(
            job_metadata.job_id, job_metadata.block_ids, s3_keys, "WRITE"
        )
        if ok:
            self._pending_stores.add(job_metadata.job_id)

    def submit_load(self, job_metadata: JobMetadata) -> None:
        keys = list(job_metadata.keys)
        s3_keys = [
            self._file_mapper.get_key(get_offload_block_hash(k)) for k in keys
        ]
        ok = self._engine.submit_transfer(
            job_metadata.job_id, job_metadata.block_ids, s3_keys, "READ"
        )
        if ok:
            self._load_in_flight.update(keys)
            self._pending_loads[job_metadata.job_id] = keys

    def get_finished(self) -> Iterable[JobResult]:
        for job_id, success in self._engine.get_finished():
            if job_id in self._pending_stores:
                self._pending_stores.remove(job_id)
            elif job_id in self._pending_loads:
                keys = self._pending_loads.pop(job_id)
                self._load_in_flight.difference_update(keys)
            else:
                continue
            yield JobResult(job_id=job_id, success=success)

    def touch(self, keys: Collection[OffloadKey]) -> None:
        pass

    def shutdown(self) -> None:
        self._engine.shutdown()

    @staticmethod
    def get_tier_type() -> str:
        return "obj"
