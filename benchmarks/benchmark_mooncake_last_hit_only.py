# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU lookup comparison; optionally use an isolated real TCP Master."""

import argparse
import contextlib
import json
import statistics
import time
from uuid import uuid4

from tests.v1.kv_connector.unit.test_mooncake_store_worker import (
    _make_selective_worker,
    _selected_mask,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    ChunkedTokenDatabase,
    KeyMetadata,
    PoolKey,
)
from vllm.v1.core.kv_cache_utils import BlockHash


class FakeStore:
    def __init__(self, present: set[str]) -> None:
        self.present = present

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        return [int(key in self.present) for key in keys]

    def batch_probe_key(
        self, keys: list[str], policy: str, candidate_size: int
    ) -> list[int]:
        assert policy == "LastHitOnly"
        return _selected_mask(keys, candidate_size, self.present)

    def close(self) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--kv-divisor", type=int, choices=(1, 8), default=1)
    parser.add_argument("--master")
    parser.add_argument("--selective-first", action="store_true")
    args = parser.parse_args()
    worker = _make_selective_worker((4,))
    model = f"lookup-bench-{uuid4().hex}"
    worker.token_dbs = [
        ChunkedTokenDatabase(
            KeyMetadata(model, 0, 0, 0, 0, group_id=gid), block_size=16
        )
        for gid in range(2)
    ]
    worker._init_lookup_key_prefixes()
    hashes = [BlockHash(i.to_bytes(4, "big")) for i in range(args.checkpoints)]
    cap = args.checkpoints // args.kv_divisor
    present = {
        PoolKey.build_key_string(prefix, block_hash.hex())
        for gid, prefixes in enumerate(worker._lookup_key_prefixes)
        for block_hash in (hashes[:cap] if gid == 0 else hashes)
        for prefix in prefixes
    }
    with contextlib.ExitStack() as cleanup:
        worker.store = FakeStore(present)
        if args.master:
            from mooncake.store import MooncakeDistributedStore

            store = MooncakeDistributedStore()
            cleanup.callback(store.close)
            assert (
                store.setup(
                    "127.0.0.1:0",
                    "P2PHANDSHAKE",
                    32 << 20,
                    16 << 20,
                    "tcp",
                    "",
                    args.master,
                )
                == 0
            )
            for key in present:
                assert store.put(key, b"checkpoint") == 0
                cleanup.callback(store.remove, key, True)
            worker.store = store
        rows = []
        for enabled in (True, False) if args.selective_first else (False, True):
            worker.mooncake_kda_last_hit_only = enabled
            samples = []
            for iteration in range(args.iterations + 20):
                started = time.perf_counter_ns()
                result = worker.lookup(args.checkpoints * 16 + 1, hashes)
                elapsed = (time.perf_counter_ns() - started) / 1000
                assert result.hit_length == cap * 16
                worker.get_kv_connector_stats()
                if iteration >= 20:
                    samples.append(elapsed)
            samples.sort()
            rows.append(
                {
                    "selective": enabled,
                    "p50_us": statistics.median(samples),
                    "p95_us": samples[int(len(samples) * 0.95) - 1],
                }
            )
        print(
            json.dumps(
                {
                    "checkpoints": args.checkpoints,
                    "kv_cap": cap,
                    "transport": "tcp" if args.master else "fake",
                    "rows": rows,
                }
            )
        )


if __name__ == "__main__":
    main()
