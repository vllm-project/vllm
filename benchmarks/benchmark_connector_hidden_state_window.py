# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare native engine capture and separately time file consumption on CUDA."""

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import time
from pathlib import Path


def worker_metrics(worker, reset=False):
    import psutil
    import torch

    from vllm.distributed.kv_transfer import (
        get_kv_transfer_group,
        has_kv_transfer_group,
    )

    connector = get_kv_transfer_group() if has_kv_transfer_group() else None
    if connector is not None:
        for future in list(getattr(connector, "_req_futures", {}).values()):
            future.result()
    if reset:
        import threading

        sampler = dict(stop=threading.Event(), peak=psutil.Process().memory_info().rss)

        def sample():
            while not sampler["stop"].wait(0.01):
                sampler["peak"] = max(
                    sampler["peak"], psutil.Process().memory_info().rss
                )

        sampler["thread"] = threading.Thread(target=sample, daemon=True)
        sampler["thread"].start()
        worker._hs_benchmark_memory_sampler = sampler
        reset_host = getattr(torch.cuda.memory, "reset_peak_host_memory_stats", None)
        if reset_host:
            reset_host()
        torch.accelerator.memory.reset_peak_memory_stats()
        if connector is not None and hasattr(connector, "_capture_metrics"):
            for key in connector._capture_metrics:
                connector._capture_metrics[key] = 0
    result = dict(
        gpu_allocated_bytes=torch.accelerator.memory.memory_allocated(),
        gpu_reserved_bytes=torch.accelerator.memory.memory_reserved(),
        gpu_allocated_peak_bytes=torch.accelerator.memory.max_memory_allocated(),
        gpu_reserved_peak_bytes=torch.accelerator.memory.max_memory_reserved(),
        worker_rss_bytes=psutil.Process().memory_info().rss,
    )
    sampler = getattr(worker, "_hs_benchmark_memory_sampler", None)
    if sampler is not None and not reset:
        sampler["stop"].set()
        sampler["thread"].join()
        result["sampled_worker_rss_peak_bytes"] = max(
            sampler["peak"], psutil.Process().memory_info().rss
        )
    host_stats = getattr(torch.cuda.memory, "host_memory_stats", None)
    result["host_allocator_stats"] = host_stats() if host_stats else None
    cache = getattr(connector, "_kv_cache", None)
    result["hidden_cache_allocated_bytes"] = (
        cache.numel() * cache.element_size() if cache is not None else 0
    )
    if connector is not None and hasattr(connector, "get_window_capture_metrics"):
        result["connector"] = connector.get_window_capture_metrics()
    return result


def distribution(values):
    ordered = sorted(values)
    return {
        "count": len(values),
        "median": statistics.median(values) if values else None,
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
        if values
        else None,
    }


async def request(engine, args, index):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    prompt = [1] + [3 + (index + i) % 100 for i in range(args.input_len - 1)]
    params = SamplingParams(
        max_tokens=args.output_len,
        temperature=0,
        ignore_eos=True,
        detokenize=False,
        output_kind=RequestOutputKind.DELTA,
    )
    if args.mode in ("full", "window"):
        transfer = {"include_output_tokens": True}
        if args.mode == "window":
            transfer["hidden_state_window"] = {
                "start": args.window_start,
                "end": args.window_start + args.window_size,
            }
        params.extra_args = {"kv_transfer_params": transfer}
    request_id = f"bench-{index}"
    started = time.perf_counter()
    first = previous = None
    itl = []
    tokens = []
    final = None
    async for output in engine.generate(
        {"prompt_token_ids": prompt}, params, request_id
    ):
        now = time.perf_counter()
        new = list(output.outputs[0].token_ids) if output.outputs else []
        if new:
            if first is None:
                first = now
            else:
                itl.append((now - previous) * 1000 / len(new))
            previous = now
            tokens.extend(new)
        final = output
    assert first is not None and final.finished
    return dict(
        index=index,
        ttft_ms=(first - started) * 1000,
        itl_ms=itl,
        tokens=len(tokens),
        token_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        path=(final.kv_transfer_params or {}).get("hidden_states_path"),
    )


async def batch(engine, args, offset=0):
    indices = iter(range(offset, offset + args.requests))

    async def worker():
        return [await request(engine, args, index) for index in indices]

    groups = await asyncio.gather(*(worker() for _ in range(args.concurrency)))
    return sorted(
        (result for group in groups for result in group),
        key=lambda result: result["index"],
    )


def consume(results):
    from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (  # noqa: E501
        cleanup_hidden_states,
        load_hidden_states,
    )

    returned = file_bytes = 0
    for result in results:
        if result["path"]:
            values = load_hidden_states(result["path"])
            returned += (
                values["hidden_states"].numel() * values["hidden_states"].element_size()
            )
            file_bytes += Path(result["path"]).stat().st_size
            cleanup_hidden_states(result["path"])
    return dict(returned_hidden_bytes=returned, serialized_file_bytes=file_bytes)


async def run(args):
    import torch

    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA and a Linux vLLM installation")
    kwargs = dict(
        model=args.model,
        dtype=args.dtype,
        skip_tokenizer_init=True,
        max_model_len=args.input_len + args.output_len + 8,
        max_num_seqs=args.concurrency,
        max_num_batched_tokens=args.batch_tokens,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        enforce_eager=args.eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    if args.mode in ("full", "window"):
        kwargs["speculative_config"] = {
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": args.layers,
                    "hidden_states_layout": args.layout,
                }
            },
        }
        kwargs["kv_transfer_config"] = {
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": str(args.storage.resolve()),
                "enable_window_capture": args.mode == "window",
                "max_inflight_export_bytes": args.export_budget,
            },
        }
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))
    try:
        warm = await batch(engine, args, -args.requests)
        if args.mode != "baseline":
            await asyncio.to_thread(consume, warm)
        before = await engine.collective_rpc(worker_metrics, kwargs={"reset": True})
        started = time.perf_counter()
        results = await batch(engine, args)
        generation_seconds = time.perf_counter() - started
        read_started = time.perf_counter()
        payload = (
            await asyncio.to_thread(consume, results) if args.mode != "baseline" else {}
        )
        consume_seconds = time.perf_counter() - read_started
        after = await engine.collective_rpc(worker_metrics)
        return {
            "config": vars(args)
            | {"storage": str(args.storage), "report": str(args.report)},
            "torch_version": torch.__version__,
            "generation_seconds": generation_seconds,
            "tokens_per_second": sum(r["tokens"] for r in results) / generation_seconds,
            "ttft_ms": distribution([r["ttft_ms"] for r in results]),
            "event_amortized_itl_ms": distribution(
                [v for r in results for v in r["itl_ms"]]
            ),
            "file_wait_read_cleanup_seconds": consume_seconds,
            "worker_before": before,
            "worker_after": after,
            "payload": payload,
            "token_hashes": [r["token_sha256"] for r in results],
        }
    finally:
        engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["baseline", "full", "window"], required=True)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--window-start", type=int, default=16)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=8256)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--batch-tokens", type=int, default=1024)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--layout",
        choices=["auxiliary", "final", "auxiliary_plus_final"],
        default="auxiliary",
    )
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--export-budget", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument(
        "--storage", type=Path, default=Path("/dev/shm/window-features")
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.concurrency < 1 or args.requests < 1 or args.input_len < 1:
        parser.error("concurrency, requests and input-len must be positive")
    if args.window_size < 1 or args.window_start < 0:
        parser.error("window-size must be positive and window-start nonnegative")
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    result = asyncio.run(run(args))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
