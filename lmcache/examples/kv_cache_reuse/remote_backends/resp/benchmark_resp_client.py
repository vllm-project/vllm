# SPDX-License-Identifier: Apache-2.0
"""
Benchmark and example usage of the RESPClient.

This script demonstrates how to use the RESPClient for high-throughput
batch operations with Redis using the RESP protocol.
"""

# Standard
import argparse
import asyncio
import time

# First Party
from lmcache.v1.storage_backend.native_clients.resp_client import RESPClient


async def run_benchmark(
    host: str,
    port: int,
    chunk_mb: float,
    num_workers: int,
    num_keys: int,
    username: str = "",
    password: str = "",
):
    batch_chunk_num_bytes = int(chunk_mb * 1024 * 1024)

    client = RESPClient(host, port, num_workers, username=username, password=password)

    try:
        print("Redis RESP Client Benchmark")
        print(f"Server: {host}:{port}, Workers: {num_workers}")
        print(f"Chunk size: {batch_chunk_num_bytes / 1024:.0f}KB, Keys: {num_keys}")
        print("-" * 60)

        # Prepare test data
        print("starting buffer initialization (this might take a while)")
        keys = [f"bench:key:{i}" for i in range(num_keys)]
        buffers = [bytearray(batch_chunk_num_bytes) for _ in range(num_keys)]
        for i, buf in enumerate(buffers):
            for j in range(batch_chunk_num_bytes):
                buf[j] = (i + j) % 256

        print("buffer initialization complete")
        print("starting throughput benchmarks")

        # Batch SET
        t0 = time.perf_counter()
        await client.batch_set(keys, [memoryview(b) for b in buffers])
        t1 = time.perf_counter()
        elapsed_set = t1 - t0
        total_bytes_set = num_keys * batch_chunk_num_bytes
        throughput_set = total_bytes_set / elapsed_set / (1024**3)
        print(
            f"Batch SET:    {throughput_set:6.2f} GB/s  "
            f"({total_bytes_set / (1024**3):.2f} GB written)"
        )

        # Batch GET
        read_bufs = [bytearray(batch_chunk_num_bytes) for _ in range(num_keys)]
        t0 = time.perf_counter()
        await client.batch_get(keys, [memoryview(b) for b in read_bufs])
        t1 = time.perf_counter()
        elapsed_get = t1 - t0
        total_bytes_get = num_keys * batch_chunk_num_bytes
        throughput_get = total_bytes_get / elapsed_get / (1024**3)
        print(
            f"Batch GET:    {throughput_get:6.2f} GB/s  "
            f"({total_bytes_get / (1024**3):.2f} GB read)"
        )

        # Verify data
        assert all(read_bufs[i] == buffers[i] for i in range(num_keys)), "Data mismatch"

        # Batch EXISTS
        t0 = time.perf_counter()
        exists_results = await client.batch_exists(keys)
        t1 = time.perf_counter()
        elapsed_exists = t1 - t0
        ops_per_sec = num_keys / elapsed_exists
        hits = sum(exists_results)
        print(f"Batch EXISTS: {ops_per_sec:6.0f} ops/s  ({hits}/{num_keys} hits)")

        # Test batched_exists alias
        results = await client.batched_exists(keys[:10])
        assert results == exists_results[:10], "batched_exists mismatch"

        print("-" * 60)
        print("All tests passed")

    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark RESPClient with configurable parameters"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Redis server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6379,
        help="Redis server port (default: 6379)",
    )
    parser.add_argument(
        "--chunk-mb",
        type=float,
        default=4.0,
        help="Chunk size in MB (default: 4.0)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of worker threads (default: 8)",
    )
    parser.add_argument(
        "--num-keys",
        type=int,
        default=500,
        help="Number of keys to benchmark (default: 500)",
    )
    parser.add_argument(
        "--username",
        type=str,
        default="",
        help="Redis username for authentication (default: empty, no auth)",
    )
    parser.add_argument(
        "--password",
        type=str,
        default="",
        help="Redis password for authentication (default: empty, no auth)",
    )

    args = parser.parse_args()
    asyncio.run(
        run_benchmark(
            host=args.host,
            port=args.port,
            chunk_mb=args.chunk_mb,
            num_workers=args.num_workers,
            num_keys=args.num_keys,
            username=args.username,
            password=args.password,
        )
    )
