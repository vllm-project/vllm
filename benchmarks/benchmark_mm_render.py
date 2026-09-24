# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure HTTP render latency, including returned multimodal tensor payloads.

Run against one or more warmed ``vllm launch render`` servers. This does not
measure GPU inference, TTFT, or the ``vllm serve --api-server-count`` deployment.
"""

import argparse
import asyncio
import io
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import psutil
import pybase64 as base64
from PIL import Image


def make_payloads(
    model: str, width: int, height: int, count: int, seed: int
) -> tuple[list[bytes], list[int]]:
    rng = np.random.default_rng(seed)
    payloads = []
    image_sizes = []
    for _ in range(count):
        pixels = rng.integers(
            0, 256, size=(max(1, height // 8), max(1, width // 8), 3), dtype=np.uint8
        )
        image = Image.fromarray(pixels).resize(
            (width, height), Image.Resampling.NEAREST
        )
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            png = buffer.getvalue()
        image_sizes.append(len(png))
        payloads.append(
            json.dumps(
                {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this image."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "data:image/png;base64,"
                                        + base64.b64encode(png).decode("ascii")
                                    },
                                },
                            ],
                        }
                    ],
                    "max_tokens": 16,
                    "temperature": 0,
                    "stream": False,
                }
            ).encode("utf-8")
        )
    return payloads, image_sizes


async def send_request(
    session: aiohttp.ClientSession, url: str, payload: bytes
) -> tuple[float, bytes]:
    start = time.perf_counter()
    async with session.post(
        url, data=payload, headers={"Content-Type": "application/json"}
    ) as response:
        body = await response.read()
        if response.status != 200:
            raise RuntimeError(
                f"{url} returned HTTP {response.status}: "
                f"{body[:1000].decode('utf-8', errors='replace')}"
            )
    return time.perf_counter() - start, body


async def run_requests(
    session: aiohttp.ClientSession,
    urls: list[str],
    payloads: list[bytes],
    count: int,
    concurrency: int,
) -> tuple[list[float], list[int]]:
    latencies = []
    response_sizes = []

    async def worker(worker_index: int) -> None:
        for index in range(worker_index, count, concurrency):
            latency, body = await send_request(
                session, urls[index % len(urls)], payloads[index % len(payloads)]
            )
            latencies.append(latency)
            response_sizes.append(len(body))
            del body

    await asyncio.gather(*(worker(index) for index in range(min(concurrency, count))))
    return latencies, response_sizes


def server_processes(pids: list[int]) -> list[psutil.Process]:
    processes = {}
    for pid in pids:
        parent = psutil.Process(pid)
        processes[parent.pid] = parent
        for child in parent.children(recursive=True):
            processes[child.pid] = child
    return list(processes.values())


def cpu_seconds(processes: list[psutil.Process]) -> float:
    times = [process.cpu_times() for process in processes]
    return sum(item.user + item.system for item in times)


async def measure(
    session: aiohttp.ClientSession,
    urls: list[str],
    payloads: list[bytes],
    args: argparse.Namespace,
) -> dict[str, Any]:
    processes = server_processes(args.server_pids)
    initial_cpu = cpu_seconds(processes)
    peak_rss = 0
    peak_shm = 0
    done = asyncio.Event()

    async def monitor() -> None:
        nonlocal peak_rss, peak_shm
        while not done.is_set():
            peak_rss = max(
                peak_rss, sum(process.memory_info().rss for process in processes)
            )
            if processes:
                peak_shm = max(peak_shm, psutil.disk_usage("/dev/shm").used)
            await asyncio.sleep(0.1)

    monitor_task = asyncio.create_task(monitor())
    start = time.perf_counter()
    try:
        latencies, response_sizes = await run_requests(
            session, urls, payloads, args.num_prompts, args.concurrency
        )
        elapsed = time.perf_counter() - start
        used_cpu = cpu_seconds(processes) - initial_cpu
    finally:
        done.set()
        await monitor_task

    return {
        "completed": len(latencies),
        "elapsed_seconds": elapsed,
        "requests_per_second": len(latencies) / elapsed,
        "latency_p50_ms": float(np.percentile(latencies, 50) * 1000),
        "latency_p99_ms": float(np.percentile(latencies, 99) * 1000),
        "mean_response_bytes": float(np.mean(response_sizes)),
        "server_cpu_seconds": used_cpu if processes else None,
        "server_average_cpu_cores": used_cpu / elapsed if processes else None,
        "server_peak_summed_rss_bytes": peak_rss if processes else None,
        "peak_shm_used_bytes": peak_shm if processes else None,
    }


async def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    urls = [url.rstrip("/") + "/v1/chat/completions/render" for url in args.base_urls]
    payloads, image_sizes = make_payloads(
        args.model, args.width, args.height, args.unique_images, args.seed
    )
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for url in urls:
            _, body = await send_request(session, url, payloads[0])
            rendered = json.loads(body)
            if not rendered.get("token_ids") or not rendered.get("features"):
                raise RuntimeError(f"{url} did not return a multimodal render result")
        await run_requests(session, urls, payloads, args.num_warmups, args.concurrency)
        trials = []
        for index in range(args.repetitions):
            result = await measure(session, urls, payloads, args)
            trials.append(result)
            print(json.dumps({"trial": index + 1, **result}), flush=True)

    return {
        "scope": "HTTP render-only; not GPU inference or API-server-count scaling",
        "label": args.label,
        "model": args.model,
        "base_urls": args.base_urls,
        "width": args.width,
        "height": args.height,
        "unique_images": args.unique_images,
        "seed": args.seed,
        "mean_png_bytes": float(np.mean(image_sizes)),
        "num_prompts": args.num_prompts,
        "num_warmups": args.num_warmups,
        "concurrency": args.concurrency,
        "client_cpu_affinity": psutil.Process().cpu_affinity(),
        "server_cpu_affinities": {
            str(pid): psutil.Process(pid).cpu_affinity() for pid in args.server_pids
        },
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("torch", "torchvision", "transformers", "vllm")
        },
        "trials": trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-urls", nargs="+", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--server-pids", nargs="*", type=int, default=[])
    parser.add_argument("--width", type=int, default=1275)
    parser.add_argument("--height", type=int, default=1650)
    parser.add_argument("--unique-images", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=96)
    parser.add_argument("--num-warmups", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    for name in (
        "width",
        "height",
        "unique_images",
        "num_prompts",
        "num_warmups",
        "concurrency",
        "repetitions",
        "timeout",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if len(set(args.base_urls)) != len(args.base_urls):
        parser.error("--base-urls must identify distinct render servers")
    result = asyncio.run(benchmark(args))
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
