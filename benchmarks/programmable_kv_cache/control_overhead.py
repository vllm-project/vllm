# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure CPU result polling and cached policy guidance; no GPU serving work."""

import argparse
import json
import statistics
import time
from functools import partial
from pathlib import Path

import torch

from vllm.config.kv_control import KVControlConfig, KVPolicyConfig
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.kv_hints.control import ControlContext, OperationRegistry, OperationStatus
from vllm.v1.kv_hints.controller import NativeKVController
from vllm.v1.kv_hints.policy_runtime import KVPolicyRuntime
from vllm.v1.kv_hints.protocol import KvHintAction, KvHintsEnvelope
from vllm.v1.kv_hints.retention import RetentionDeclarations
from vllm.v1.kv_hints.state import SessionPrefixIndex


def measure(function, iterations, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            function()
        samples.append((time.perf_counter_ns() - start) / iterations / 1000)
    return {"median_us": statistics.median(samples), "samples_us": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    report = {
        "scope": "cpu_only_no_requests",
        "repeats": args.repeats,
        "measurements": {},
    }
    results = report["measurements"]
    context = ControlContext("engine", "epoch", "native")
    for records in (64, 4096):
        registry = OperationRegistry("engine", engine_epoch="epoch", clock=lambda: 0.0)
        for i in range(records):
            result = registry.submit(
                KvHintsEnvelope(
                    "0.1",
                    str(i),
                    [
                        KvHintAction("a", "signal", "1", {}),
                    ],
                ),
                context,
            ).results[0]
            registry.update(result, OperationStatus.COMPLETED)
        results[f"poll_{records}_results"] = measure(
            partial(registry.get, "0", "a"),
            3000,
            args.repeats,
        )
    index = SessionPrefixIndex("namespace", "epoch")
    controller = NativeKVController(registry, index, RetentionDeclarations(index))
    settings = KVControlConfig(
        policies=[KVPolicyConfig(name="quota", implementation="quota")]
    )
    for blocks in (10000, 100000):
        spec = FullAttentionSpec(
            block_size=16, num_kv_heads=1, head_size=16, dtype=torch.float16
        )
        manager = KVCacheManager(
            KVCacheConfig(blocks, [], [KVCacheGroupSpec(["layer"], spec)]),
            max_model_len=128,
            scheduler_block_size=16,
            hash_block_size=16,
        )
        runtime = KVPolicyRuntime(
            settings,
            controller,
            manager,
            lambda: (),
            clock=lambda: 0.0,
            wall_clock=lambda: 0.0,
        )
        runtime.observe_requests()
        results[f"cached_guidance_{blocks}_blocks"] = measure(
            runtime.observe_requests, 100, args.repeats
        )
        results[f"fresh_observation_{blocks}_blocks"] = measure(
            runtime.observation, 100, args.repeats
        )
        pool = manager.block_pool
        allocated = pool.get_new_blocks(blocks - 1)
        for block in allocated:
            key = make_block_hash_with_group_id(
                BlockHash(block.block_id.to_bytes(8, "big")), 0
            )
            pool.cache_transferred_block(block, key, 16)
        pool.free_blocks(allocated)
        results[f"populated_observation_{blocks}_blocks"] = measure(
            runtime.observation, 100, args.repeats
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
