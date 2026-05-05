# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OBJ (S3) secondary tier implementation."""

from collections.abc import Collection, Iterable

from vllm.v1.kv_offload.base import OffloadKey, ReqContext, get_offload_block_hash
from vllm.v1.kv_offload.tiering.base import JobMetadata, JobResult, SecondaryTierManager
from vllm.v1.kv_offload.tiering.obj.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.obj.nixl_engine import NixlEngine
from vllm.v1.kv_offload.tiering.obj.nixl_lookup import NixlLookup


class ObjSecondaryTier(SecondaryTierManager):
    """Secondary tier that offloads KV cache blocks to an S3-compatible store.

    Handles CPU DRAM <-> S3 transfers only. GPU <-> CPU is managed by the
    primary tier (CpuGpuOffloadingHandlers from PR 40020).
    """

    def __init__(
        self,
        bucket: str,
        endpoint_override: str,
        access_key: str,
        secret_key: str,
        model_name: str,
        gpu_block_size: int,
        tp_size: int,
        pp_size: int,
        pcp_size: int,
        rank: int,
        dtype: str,
        lookup_mode: str = "nixl_query",
        scheme: str = "http",
        ca_bundle: str = "",
        key_prefix: str = "",
        io_threads: int = 4,
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

        self._lookup_mode = lookup_mode
        if lookup_mode == "nixl_query":
            self._nixl_lookup: NixlLookup | None = NixlLookup(**s3_params)
        else:
            self._nixl_lookup = None
            self._stored_keys: set[OffloadKey] = set()

        # Keys currently being promoted S3 -> CPU (submit_load in flight)
        self._load_in_flight: set[OffloadKey] = set()
        # job_id -> (is_store, keys) for completion bookkeeping
        self._pending_jobs: dict[int, tuple[bool, list[OffloadKey]]] = {}

    def set_primary_view(self, view: memoryview) -> None:
        self._engine.set_primary_view(view)

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        if key in self._load_in_flight:
            return None
        if self._lookup_mode == "dict":
            return key in self._stored_keys
        s3_key = self._file_mapper.get_key(get_offload_block_hash(key))
        return self._nixl_lookup.exists(s3_key)  # type: ignore[union-attr]

    def submit_store(self, job_metadata: JobMetadata) -> None:
        keys = list(job_metadata.keys)
        s3_keys = [
            self._file_mapper.get_key(get_offload_block_hash(k)) for k in keys
        ]
        ok = self._engine.submit_transfer(
            job_metadata.job_id, job_metadata.block_ids, s3_keys, "WRITE"
        )
        if ok:
            self._pending_jobs[job_metadata.job_id] = (True, keys)

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
            self._pending_jobs[job_metadata.job_id] = (False, keys)

    def get_finished(self) -> Iterable[JobResult]:
        for job_id, success in self._engine.get_finished():
            job_info = self._pending_jobs.pop(job_id, None)
            if job_info is None:
                continue
            is_store, keys = job_info
            if is_store:
                if success and self._lookup_mode == "dict":
                    self._stored_keys.update(keys)
            else:
                self._load_in_flight.difference_update(keys)
            yield JobResult(job_id=job_id, success=success)

    def touch(self, keys: Collection[OffloadKey]) -> None:
        pass

    def shutdown(self) -> None:
        self._engine.shutdown()

    @staticmethod
    def get_tier_type() -> str:
        return "obj"
