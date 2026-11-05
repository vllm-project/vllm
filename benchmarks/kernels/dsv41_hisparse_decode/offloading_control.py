# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the frozen agentic request loop with the standard OffloadingConnector.

The benchmark-only factory supplies connector configuration before engine creation.
The original request loop, prompts, scheduling, and timing boundaries are reused.
The wrapper's separate hash is recorded alongside the common source manifest.
"""

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
from pathlib import Path

import agentic

_native_inventory = agentic.memory_inventory


class TransferCounts:
    def __init__(self, connector):
        self.connector = connector
        self.original = connector.build_connector_worker_meta
        self.reported = dict(load_bytes=0, store_bytes=0, load_jobs=0, store_jobs=0)
        connector.build_connector_worker_meta = self.collect

    @staticmethod
    def add(totals, stats):
        for direction in ("load", "store"):
            value = getattr(stats, direction)
            totals[f"{direction}_bytes"] += value.bytes
            totals[f"{direction}_jobs"] += len(value.sizes)

    def collect(self):
        metadata = self.original()
        if metadata is not None:
            self.add(self.reported, metadata.transfer_stats)
        return metadata

    def read(self):
        totals = dict(self.reported)
        self.add(totals, self.connector._connector_worker_meta.transfer_stats)
        return totals


def install(worker):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
        OffloadingConnectorWorker,
    )
    from vllm.distributed.kv_transfer.kv_transfer_state import get_kv_transfer_group

    connector = get_kv_transfer_group().connector_worker
    assert isinstance(connector, OffloadingConnectorWorker)
    worker.agentic_traffic = None
    worker.standard_offload_traffic = TransferCounts(connector)
    return dict(rank=worker.rank, standard_offloading=True)


def memory_inventory(worker, reset=False):
    from vllm.distributed.kv_transfer.kv_transfer_state import get_kv_transfer_group

    result = _native_inventory(worker, reset=reset)
    connector = get_kv_transfer_group().connector_worker
    spec = connector.spec
    region = connector.worker._mmap_region
    assert region is not None and region.is_pinned
    regions = 1 if spec.replicated_layout else spec.config.parallel.world_size
    node_bytes = regions * region.total_size_bytes
    result["standard_offloading"] = dict(
        replicated_layout=spec.replicated_layout,
        worker_region_bytes=region.total_size_bytes,
        node_cpu_bytes=node_bytes,
        node_cpu_chunk_bytes=node_bytes // spec.num_chunks,
        configured_cpu_bytes=spec.num_chunks * spec.kv_bytes_per_chunk,
        cpu_chunks=spec.num_chunks,
        cpu_chunk_bytes=spec.kv_bytes_per_chunk,
        host_memory_is_pinned=region.is_pinned,
        completed_transfer_reports=worker.standard_offload_traffic.read(),
    )
    return result


async def main(options):
    from vllm.config import KVTransferConfig
    from vllm.v1.engine.async_llm import AsyncLLM

    wrapper_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert wrapper_sha == options.wrapper_sha256
    connector_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "cpu_bytes_to_use": options.host_gib * 2**30,
            "blocks_per_chunk": 1,
            "offload_prompt_only": False,
        },
    )
    original_factory = AsyncLLM.from_engine_args

    def create(engine_args, *args, **kwargs):
        assert engine_args.kv_transfer_config is None
        engine_args.kv_transfer_config = connector_config
        receipt = options.output / "run.json"
        record = json.loads(receipt.read_text())
        record["engine_kwargs"]["kv_transfer_config"] = dataclasses.asdict(
            connector_config
        )
        record["control"] = dict(
            connector="OffloadingConnector",
            wrapper_sha256=wrapper_sha,
            note="Common request loop and production source manifest are unchanged",
        )
        receipt.write_text(json.dumps(record, indent=2) + "\n")
        return original_factory(engine_args, *args, **kwargs)

    AsyncLLM.from_engine_args = staticmethod(create)
    agentic.install_traffic = install
    agentic.memory_inventory = memory_inventory
    options.mode = "resident"
    options.sources = []
    options.validate = False
    options.fail_fast = False
    options.bounded_replay = True
    await agentic.main(options)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--wrapper-sha256", required=True)
    parser.add_argument("--kv-bytes", type=int)
    parser.add_argument("--gpu-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=1048576)
    parser.add_argument("--max-seqs", type=int, default=32)
    parser.add_argument("--max-batch", type=int, default=8192)
    parser.add_argument("--host-gib", type=int, default=64)
    parser.add_argument("--case-timeout", type=int, default=7200)
    os.environ.update(
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS="8192",
    )
    asyncio.run(main(parser.parse_args()))
