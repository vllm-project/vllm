# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in synchronous EC-format SSD probe against an external Mooncake Store."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any


def require_disk_only(store: Any, keys: list[str]) -> dict[str, list[str]]:
    """Reject missing, memory-backed or unrecognized replicas as SSD evidence."""
    descriptors = store.batch_get_replica_desc(keys)
    tiers: dict[str, list[str]] = {}
    for key in keys:
        replicas = descriptors[key]
        kinds = []
        for replica in replicas:
            if replica.is_memory_replica():
                kinds.append("memory")
            elif replica.is_local_disk_replica():
                kinds.append("local_disk")
            elif replica.is_disk_replica():
                kinds.append("disk")
            else:
                kinds.append("unknown")
        tiers[key] = kinds
    if any(
        not kinds or any(kind != "local_disk" for kind in kinds)
        for kinds in tiers.values()
    ):
        raise ValueError(f"Expected only LOCAL_DISK SSD replicas: {tiers}")
    return tiers


def verify_read(client: Any, tensors: dict[str, Any], device: str) -> dict[str, Any]:
    """Use the real EC resolver's read/codec path and verify every tensor."""
    import torch

    from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding import (
        data,
    )

    before = require_disk_only(client.store, list(tensors))
    expected = {
        key: data.TensorSpec(tuple(t.shape), str(t.dtype), t.nbytes)
        for key, t in tensors.items()
    }
    start = time.perf_counter()
    loaded = client.load_tensors(expected, device)
    elapsed = time.perf_counter() - start
    if set(loaded) != set(tensors):
        raise AssertionError(
            f"GET did not restore all objects: {set(tensors) - set(loaded)}"
        )
    for key, tensor in tensors.items():
        if not torch.equal(loaded[key].cpu(), tensor):
            raise AssertionError(f"Embedding mismatch: {key}")
    after = require_disk_only(client.store, list(tensors))
    return {
        "phase": "read",
        "verified": True,
        "destination": device,
        "objects": len(tensors),
        "payload_bytes": sum(t.nbytes for t in tensors.values()),
        "get_and_copy_seconds": elapsed,
        "replicas_before": before,
        "replicas_after": after,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("publish", "read"))
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--namespace", required=True, help="Unique experiment key prefix"
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=[16, 256, 4096])
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--read-buffer-bytes", type=int, default=128 * 1024**2)
    parser.add_argument("--disk-budget-bytes", type=int, required=True)
    parser.add_argument("--preferred-segment")
    args = parser.parse_args()
    if (
        min(*args.tokens, args.width, args.read_buffer_bytes, args.disk_budget_bytes)
        <= 0
    ):
        parser.error("Sizes and budgets must be positive")

    import importlib.metadata

    import torch
    from mooncake.store import MooncakeDistributedStore, ObjectDataType, ReplicateConfig

    from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding import (
        store_client,
    )
    from vllm.distributed.mooncake_store import (
        MooncakeStoreConfig,
        setup_mooncake_store,
    )
    from vllm.utils.network_utils import get_ip

    config = MooncakeStoreConfig.from_file(args.config)
    if config.mode != "standalone-store" or not config.enable_offload:
        parser.error("Probe requires standalone-store with enable_offload=true")
    dtype = getattr(torch, args.dtype)
    tensors = {}
    for index, tokens in enumerate(args.tokens):
        key = (
            f"{args.namespace}@ssd-probe-v1@{args.dtype}@{tokens}x{args.width}@{index}"
        )
        tensors[key] = (
            ((torch.arange(tokens * args.width, device="cpu") + index) % 257 - 128)
            .to(dtype)
            .reshape(tokens, args.width)
        )
    store = MooncakeDistributedStore()
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.rdma_utils import (
        get_requester_local_hostname,
    )

    setup_mooncake_store(store, config, get_requester_local_hostname(get_ip()))
    replicate = ReplicateConfig()
    replicate.data_type = ObjectDataType.TENSOR
    if args.preferred_segment:
        replicate.preferred_segment = args.preferred_segment
    # The probe uses the same client without constructing an Encoder worker.
    client = store_client.MooncakeEmbeddingStoreClient(
        store,
        replicate_config=replicate,
        read_buffer_bytes=args.read_buffer_bytes,
        disk_staging_budget_bytes=args.disk_budget_bytes,
    )
    try:
        if args.phase == "publish":
            if any(client.batch_exists(list(tensors))):
                raise ValueError("Probe keys already exist; use a fresh namespace")
            for key, tensor in tensors.items():
                client.put_tensor(key, tensor.to(args.device))
            result = {
                "phase": "publish",
                "put_completed": True,
                "ssd_persistence_confirmed": False,
                "keys": list(tensors),
            }
        else:
            result = verify_read(client, tensors, args.device)
        result["torch_version"] = torch.__version__
        result["vllm_version"] = importlib.metadata.version("vllm")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        # Unknown native completion must retain registered owners until exit.
        if not client._poisoned:
            client.close()


if __name__ == "__main__":
    main()
