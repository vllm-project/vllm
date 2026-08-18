#!/usr/bin/env python3
"""
Benchmark client for vLLM serving with real H.264 video files.

Supports two modes:
  --use-file-url   Send a file:// URL (tiny JSON payload, server reads from disk)
  (default)        Send base64-encoded video over HTTP (legacy, large payload)

Measures:
  - TTFT  (Time To First Token)
  - TPOT  (Time Per Output Token)
  - ITL   (Inter-Token Latency)
  - E2EL  (End-to-End Latency)
  - Request throughput, output-token throughput

Unlike `vllm bench serve --dataset-name random-mm`, this script uses a
*real* H.264 file so that NVDEC hardware decoding is exercised when the
server's video backend is set to DeepStream.

Usage:
    # File-URL mode (recommended — tiny payload, server reads from disk):
    python3 bench_h264_client.py \
        --video /data/video/drivesim.mp4 --use-file-url \
        --num-prompts 30 --request-rate inf \
        --base-url http://localhost:8000 \
        --model bench-model

    # Base64 mode (legacy — 31MB payload per request):
    python3 bench_h264_client.py \
        --video /path/to/1080p.mp4 \
        --num-prompts 30 --request-rate inf \
        --base-url http://localhost:8000 \
        --model bench-model
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import numpy as np


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class RequestResult:
    success: bool = False
    ttft: float = 0.0          # seconds
    e2el: float = 0.0          # seconds
    token_timestamps: list[float] = field(default_factory=list)
    output_tokens: int = 0
    error: str = ""
    generated_text: str = ""
    request_id: int = -1


@dataclass
class BenchmarkMetrics:
    completed: int = 0
    failed: int = 0
    total_duration: float = 0.0

    mean_ttft_ms: float = 0.0
    median_ttft_ms: float = 0.0
    std_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0

    mean_tpot_ms: float = 0.0
    median_tpot_ms: float = 0.0
    std_tpot_ms: float = 0.0
    p99_tpot_ms: float = 0.0

    mean_itl_ms: float = 0.0
    median_itl_ms: float = 0.0
    std_itl_ms: float = 0.0
    p99_itl_ms: float = 0.0

    mean_e2el_ms: float = 0.0
    median_e2el_ms: float = 0.0
    std_e2el_ms: float = 0.0
    p99_e2el_ms: float = 0.0

    request_throughput: float = 0.0
    output_throughput: float = 0.0


def compute_metrics(
    results: list[RequestResult], total_duration: float
) -> BenchmarkMetrics:
    m = BenchmarkMetrics()
    good = [r for r in results if r.success]
    m.completed = len(good)
    m.failed = len(results) - len(good)
    m.total_duration = total_duration

    if not good:
        return m

    ttfts = np.array([r.ttft * 1000 for r in good])
    e2els = np.array([r.e2el * 1000 for r in good])

    m.mean_ttft_ms = float(np.mean(ttfts))
    m.median_ttft_ms = float(np.median(ttfts))
    m.std_ttft_ms = float(np.std(ttfts))
    m.p99_ttft_ms = float(np.percentile(ttfts, 99))

    m.mean_e2el_ms = float(np.mean(e2els))
    m.median_e2el_ms = float(np.median(e2els))
    m.std_e2el_ms = float(np.std(e2els))
    m.p99_e2el_ms = float(np.percentile(e2els, 99))

    # TPOT = (e2el - ttft) / (output_tokens - 1) for each request
    tpots: list[float] = []
    for r in good:
        if r.output_tokens > 1:
            tpots.append((r.e2el - r.ttft) / (r.output_tokens - 1) * 1000)
    if tpots:
        tpot_arr = np.array(tpots)
        m.mean_tpot_ms = float(np.mean(tpot_arr))
        m.median_tpot_ms = float(np.median(tpot_arr))
        m.std_tpot_ms = float(np.std(tpot_arr))
        m.p99_tpot_ms = float(np.percentile(tpot_arr, 99))

    # ITL from per-token timestamps
    all_itls: list[float] = []
    for r in good:
        ts = r.token_timestamps
        for i in range(1, len(ts)):
            all_itls.append((ts[i] - ts[i - 1]) * 1000)
    if all_itls:
        itl_arr = np.array(all_itls)
        m.mean_itl_ms = float(np.mean(itl_arr))
        m.median_itl_ms = float(np.median(itl_arr))
        m.std_itl_ms = float(np.std(itl_arr))
        m.p99_itl_ms = float(np.percentile(itl_arr, 99))

    total_tokens = sum(r.output_tokens for r in good)
    m.request_throughput = m.completed / total_duration if total_duration > 0 else 0
    m.output_throughput = total_tokens / total_duration if total_duration > 0 else 0

    return m


# ── Single-request sender ────────────────────────────────────────────

async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    video_url: str,
    prompt: str,
    max_tokens: int,
    request_id: int,
) -> RequestResult:
    result = RequestResult()

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": video_url},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    result.request_id = request_id
    t_start = time.monotonic()
    first_token_time = None
    text_parts: list[str] = []

    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                result.error = f"HTTP {resp.status}: {body[:300]}"
                return result

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content is not None and len(content) > 0:
                    now = time.monotonic()
                    if first_token_time is None:
                        first_token_time = now
                    result.token_timestamps.append(now)
                    result.output_tokens += 1
                    text_parts.append(content)

    except Exception as exc:
        result.error = str(exc)
        return result

    t_end = time.monotonic()
    result.generated_text = "".join(text_parts)

    if first_token_time is not None:
        result.success = True
        result.ttft = first_token_time - t_start
        result.e2el = t_end - t_start

        tpot = 0.0
        if result.output_tokens > 1:
            tpot = (result.e2el - result.ttft) / (result.output_tokens - 1) * 1000
        print(f"  [req {request_id:>3}] TTFT={result.ttft*1000:7.1f}ms  "
              f"E2EL={result.e2el*1000:7.1f}ms  "
              f"TPOT={tpot:6.1f}ms  "
              f"tokens={result.output_tokens}")
    else:
        print(f"  [req {request_id:>3}] FAILED: {result.error[:80]}")

    return result


# ── Request dispatcher ────────────────────────────────────────────────

def _resolve_video_paths(video_args: list[str]) -> list[Path]:
    """Expand directories and glob patterns into a flat list of video files."""
    paths: list[Path] = []
    for v in video_args:
        p = Path(v)
        if p.is_dir():
            for ext in ("*.mp4", "*.mkv", "*.avi", "*.ts", "*.mov"):
                paths.extend(sorted(p.glob(ext)))
        elif p.exists():
            paths.append(p)
        else:
            print(f"WARNING: Video path not found, skipping: {v}",
                  file=sys.stderr)
    return paths


async def run_benchmark(
    args: argparse.Namespace,
) -> tuple[BenchmarkMetrics, list[RequestResult]]:
    video_paths = _resolve_video_paths(args.video)
    if not video_paths:
        print("ERROR: No video files found.", file=sys.stderr)
        sys.exit(1)

    video_urls: list[str] = []
    for vp in video_paths:
        if args.use_file_url:
            abs_path = str(vp.resolve())
            video_urls.append(f"file://{abs_path}")
            print(f"  Video file  : {abs_path} "
                  f"({vp.stat().st_size / 1024:.0f} KB)")
        else:
            video_b64 = base64.b64encode(vp.read_bytes()).decode("ascii")
            video_urls.append(f"data:video/mp4;base64,{video_b64}")
            print(f"  Video file  : {vp} "
                  f"({vp.stat().st_size / 1024:.0f} KB, base64)")

    n_videos = len(video_urls)
    print(f"  Mode        : {'file:// URL' if args.use_file_url else 'base64 inline'}")
    print(f"  Videos      : {n_videos} file(s)")

    url = f"{args.base_url}/v1/chat/completions"
    prompt = args.prompt
    num = args.num_prompts
    rate = args.request_rate

    reqs_per_video = num // n_videos
    remainder = num % n_videos
    print(f"  Target URL  : {url}")
    print(f"  Model       : {args.model}")
    print(f"  Num prompts : {num} ({reqs_per_video} per video"
          f"{f', +1 for first {remainder}' if remainder else ''})")
    print(f"  Request rate: {'inf (all at once)' if rate == float('inf') else f'{rate} req/s'}")
    print(f"  Max tokens  : {args.max_tokens}")
    print()

    timeout = aiohttp.ClientTimeout(total=600)
    conn = aiohttp.TCPConnector(limit=num + 10)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        if not args.skip_warmup:
            print("  Sending warm-up request...")
            warm = await send_request(
                session, url, args.model, video_urls[0], prompt, 5, -1
            )
            if warm.success:
                print(f"  Warm-up done (TTFT={warm.ttft*1000:.0f}ms, "
                      f"E2EL={warm.e2el*1000:.0f}ms)")
            else:
                print(f"  Warm-up FAILED: {warm.error}")
                if not args.force:
                    print("  Aborting. Use --force to run anyway.")
                    sys.exit(1)
            print()

        tasks: list[asyncio.Task] = []
        print(f"  Launching {num} requests...")
        bench_start = time.monotonic()

        for i in range(num):
            vid_url = video_urls[i % n_videos]
            task = asyncio.create_task(
                send_request(
                    session, url, args.model, vid_url,
                    prompt, args.max_tokens, i,
                )
            )
            tasks.append(task)
            if rate != float("inf") and i < num - 1:
                await asyncio.sleep(1.0 / rate)

        results = await asyncio.gather(*tasks)
        bench_end = time.monotonic()

    total_duration = bench_end - bench_start
    result_list = list(results)
    metrics = compute_metrics(result_list, total_duration)

    return metrics, result_list


# ── Display & save ────────────────────────────────────────────────────

def print_metrics(m: BenchmarkMetrics) -> None:
    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║              Benchmark Results                      ║")
    print("  ╠══════════════════════════════════════════════════════╣")
    print(f"  ║  Completed requests    : {m.completed:>8}                 ║")
    print(f"  ║  Failed requests       : {m.failed:>8}                 ║")
    print(f"  ║  Benchmark duration    : {m.total_duration:>8.2f}s                ║")
    print(f"  ║  Request throughput    : {m.request_throughput:>8.2f} req/s           ║")
    print(f"  ║  Output throughput     : {m.output_throughput:>8.2f} tok/s           ║")
    print("  ╠══════════════════════════════════════════════════════╣")
    print(f"  ║  Mean   TTFT           : {m.mean_ttft_ms:>10.2f} ms             ║")
    print(f"  ║  Median TTFT           : {m.median_ttft_ms:>10.2f} ms             ║")
    print(f"  ║  P99    TTFT           : {m.p99_ttft_ms:>10.2f} ms             ║")
    print("  ╠══════════════════════════════════════r═══════════════╣")
    print(f"  ║  Mean   TPOT           : {m.mean_tpot_ms:>10.2f} ms             ║")
    print(f"  ║  Median TPOT           : {m.median_tpot_ms:>10.2f} ms             ║")
    print(f"  ║  P99    TPOT           : {m.p99_tpot_ms:>10.2f} ms             ║")
    print("  ╠══════════════════════════════════════════════════════╣")
    print(f"  ║  Mean   ITL            : {m.mean_itl_ms:>10.2f} ms             ║")
    print(f"  ║  Median ITL            : {m.median_itl_ms:>10.2f} ms             ║")
    print(f"  ║  P99    ITL            : {m.p99_itl_ms:>10.2f} ms             ║")
    print("  ╠══════════════════════════════════════════════════════╣")
    print(f"  ║  Mean   E2EL           : {m.mean_e2el_ms:>10.2f} ms             ║")
    print(f"  ║  Median E2EL           : {m.median_e2el_ms:>10.2f} ms             ║")
    print(f"  ║  P99    E2EL           : {m.p99_e2el_ms:>10.2f} ms             ║")
    print("  ╚══════════════════════════════════════════════════════╝")


def print_vlm_outputs(results: list[RequestResult], max_show: int = 5) -> None:
    good = [r for r in results if r.success]
    good.sort(key=lambda r: r.request_id)
    print()
    print(f"  ── VLM Text Output (showing {min(max_show, len(good))}"
          f" of {len(good)} responses) ──")
    for r in good[:max_show]:
        tokens = r.output_tokens
        text = r.generated_text.replace("\n", " ").strip()
        print(f"  [req {r.request_id:>3}] ({tokens:>3} tokens) {text}")
    if len(good) > max_show:
        lengths = [r.output_tokens for r in good]
        print(f"  ...")
        print(f"  Token counts: min={min(lengths)}, max={max(lengths)}, "
              f"mean={np.mean(lengths):.1f}, total={sum(lengths)}")
    print()


def save_results(m: BenchmarkMetrics, path: str,
                 results: list[RequestResult] | None = None) -> None:
    d = {k: v for k, v in m.__dict__.items()}
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"  Results saved to: {path}")

    if results:
        txt_path = path.rsplit(".", 1)[0] + "_vlm_outputs.txt"
        with open(txt_path, "w") as f:
            for r in sorted(results, key=lambda r: r.request_id):
                if r.success:
                    f.write(f"[req {r.request_id}] ({r.output_tokens} tokens) "
                            f"{r.generated_text}\n")
        print(f"  VLM outputs saved to: {txt_path}")


# ── Main ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark vLLM serving with real H.264 video"
    )
    p.add_argument("--video", required=True, nargs="+",
                   help="Path(s) to H.264 MP4 video file(s), or a directory. "
                        "Requests are distributed round-robin across files.")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", default="bench-model")
    p.add_argument("--num-prompts", type=int, default=30)
    p.add_argument("--request-rate", type=float, default=float("inf"),
                   help="Requests per second (default: inf = all at once)")
    p.add_argument("--max-tokens", type=int, default=50)
    p.add_argument("--prompt", default="Describe what is happening in this video in detail.")
    p.add_argument("--save-result", action="store_true")
    p.add_argument("--result-filename", default="bench_result.json")
    p.add_argument("--skip-warmup", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Continue even if warm-up fails")
    p.add_argument("--chunk-duration", type=float, default=10.0,
                   help="Seconds of video per chunk (default: 10)")
    p.add_argument("--use-file-url", action="store_true",
                   help="Send file:// URL instead of base64. "
                        "Server must have --allowed-local-media-path set.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.request_rate <= 0:
        print("ERROR: --request-rate must be positive or 'inf'",
              file=sys.stderr)
        sys.exit(1)

    metrics, results = asyncio.run(run_benchmark(args))
    print_metrics(metrics)
    print_vlm_outputs(results)

    if args.save_result:
        save_results(metrics, args.result_filename, results)


if __name__ == "__main__":
    main()
