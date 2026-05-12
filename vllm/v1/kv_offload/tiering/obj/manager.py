# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OBJ (S3) secondary tier implementation."""

from collections.abc import Collection, Iterable

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext, get_offload_block_hash, get_offload_group_idx
from vllm.v1.kv_offload.tiering.base import JobMetadata, JobResult, SecondaryTierManager
from nixl._api import nixl_agent, nixl_agent_config

from vllm.v1.kv_offload.tiering.obj.nixl_engine import NixlEngine, READ, WRITE
from vllm.v1.kv_offload.tiering.obj.config import ObjStoreConfig

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
        lookup_agent_config = nixl_agent_config(backends=[])
        self._lookup_agent = nixl_agent("ObjLookup", lookup_agent_config)
        self._lookup_agent.create_backend("OBJ", obj_config.to_nixl_params())
        self._prefix = f"{prefix}/" if prefix else ""

        self._probe_connectivity()

    def _probe_connectivity(self) -> None:
        """Verify S3 connectivity at startup via a NIXL lookup probe.

        Performs a single exists() check against a synthetic key that will
        never exist. A True/False result confirms the bucket is reachable;
        an exception indicates misconfigured S3 params and raises RuntimeError.
        """
        probe_key = "__nixl_probe__/connectivity_test"
        try:
            self._exists(probe_key)
            logger.info("OBJ tier S3 connectivity probe succeeded")
        except Exception as e:
            raise RuntimeError(
                f"OBJ tier S3 connectivity probe failed — check bucket, "
                f"endpoint_override, access_key, secret_key, and scheme. "
                f"Error: {e}"
            ) from e

    def set_primary_view(self, view: memoryview) -> None:
        self._engine.set_primary_view(view)

    def _exists(self, s3_key: str) -> bool:
        return self._lookup_agent.query_memory([(0, 1, 0, s3_key)], "OBJ", "OBJ")[0] is not None

    def _get_obj_key(self, key: OffloadKey) -> str:
        h = get_offload_block_hash(key)[-8:].hex()
        g = get_offload_group_idx(key)
        return f"{self._prefix}group_{g}/{h[:3]}/{h[3:5]}/{h}.bin"

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        return self._exists(self._get_obj_key(key))

    def submit_store(self, job_metadata: JobMetadata) -> None:
        s3_keys = (self._get_obj_key(k) for k in job_metadata.keys)
        self._engine.submit_transfer(
            job_metadata.job_id, job_metadata.block_ids, s3_keys, WRITE
        )

    def submit_load(self, job_metadata: JobMetadata) -> None:
        s3_keys = (self._get_obj_key(k) for k in job_metadata.keys)
        self._engine.submit_transfer(
            job_metadata.job_id, job_metadata.block_ids, s3_keys, READ
        )

    def get_finished(self) -> Iterable[JobResult]:
        return self._engine.get_finished()

    def shutdown(self) -> None:
        self._engine.shutdown()

    @classmethod
    def from_config(cls, config: dict) -> "ObjectStoreSecondaryTierManager":
        return cls(
            obj_config=ObjStoreConfig(**config["store_config"]),
            prefix=config.get("prefix", ""),
            io_threads=config.get("io_threads", 4),
        )

    @staticmethod
    def get_tier_type() -> str:
        return "obj"
