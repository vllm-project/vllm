# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay timed mixed-length requests through AsyncLLM's native scheduler."""

import argparse
import asyncio
import dataclasses
import hashlib
import json
import math
import os
import statistics
import time
import traceback
from pathlib import Path

from capture_model import build_prompt, write_json
from execution import execution_counts
from validate import inventory


def worker_inventory(worker, reset=False):
    model = worker.model_runner.get_model()
    result = inventory(model, reset=reset)
    result["hisparse_row_stats"] = {}
    storages = {}
    for layer in model.modules():
        tensor = getattr(layer, "kv_cache", None)
        if hasattr(tensor, "untyped_storage"):
            storage = tensor.untyped_storage()
            storages[str(tensor.device), storage.data_ptr()] = storage.nbytes()
        cache = getattr(layer, "hisparse_cache", None)
        if cache is not None:
            counters = cache.runtime.index_group.swap_stats
            if reset:
                counters.zero_()
            hits, misses = counters.cpu().tolist()
            result["hisparse_row_stats"][layer.layer_id] = dict(
                hits=hits, host_miss_rows=misses, raw_fetch_bytes=misses * 584
            )
    result["unique_kv_storage_bytes"] = {
        device: sum(size for (where, _), size in storages.items() if where == device)
        for device, _ in storages
    }
    config = worker.model_runner.kv_cache_config
    result.update(
        gpu_blocks=config.num_blocks,
        host_blocks=config.hisparse_host_num_blocks,
        cache_groups=[
            dict(
                layers=g.layer_names,
                block_tokens=g.kv_cache_spec.block_size,
                page_bytes=g.kv_cache_spec.page_size_bytes,
                host_resident=g.host_resident,
                role=str(g.role),
            )
            for g in config.kv_cache_groups
        ],
    )
    return result


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(requests):
    intervals = [v for r in requests for v in r["engine_intertoken_ms"]]
    return dict(
        requests=len(requests),
        output_tokens=sum(r["received_tokens"] for r in requests),
        mean_ttft_seconds=statistics.fmean(r["ttft_seconds"] for r in requests),
        mean_tpot_ms=statistics.fmean(
            1000
            * (r["finished_seconds"] - r["first_token_seconds"])
            / (r["received_tokens"] - 1)
            for r in requests
            if r["received_tokens"] > 1
        ),
        median_ttft_seconds=percentile([r["ttft_seconds"] for r in requests], 0.5),
        p95_ttft_seconds=percentile([r["ttft_seconds"] for r in requests], 0.95),
        median_latency_seconds=percentile(
            [r["latency_seconds"] for r in requests], 0.5
        ),
        p95_latency_seconds=percentile([r["latency_seconds"] for r in requests], 0.95),
        median_intertoken_ms=percentile(intervals, 0.5),
        p95_intertoken_ms=percentile(intervals, 0.95),
        p99_intertoken_ms=percentile(intervals, 0.99),
        max_intertoken_ms=max(intervals, default=None),
        preemptions=sum(r["preemptions"] for r in requests),
    )


async def main(args):
    from vllm import SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    args.output.mkdir(parents=True, exist_ok=False)
    sources = json.loads(args.source_manifest.read_text())
    for name, expected in sources.items():
        assert (
            hashlib.sha256(
                (args.source_manifest.parent / name).read_bytes()
            ).hexdigest()
            == expected
        ), name
    jobs = json.loads(args.jobs.read_text())
    kwargs = dict(
        model="/model-cache/rits-models/models/deepseek-ai/DeepSeek-V4.1-Flash-1017",
        tensor_parallel_size=8,
        enable_expert_parallel=True,
        enable_ep_weight_filter=True,
        engram_config={"cpu_offload": True},
        tokenizer_mode="deepseek_v41",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_seqs,
        max_num_batched_tokens=args.max_batch,
        kv_cache_memory_bytes=args.kv_bytes,
        safetensors_load_strategy="prefetch",
        block_size=64,
        async_scheduling=False,
        enable_prefix_caching=False,
        swa_bounded_replay=False,
        kernel_config={"enable_jit_warmup": False},
        disable_log_stats=False,
        enable_logging_iteration_details=True,
        limit_mm_per_prompt={"image": 0},
        seed=0,
        compilation_config=dict(
            mode=0,
            cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=list(range(1, args.max_seqs + 1)),
            cudagraph_num_of_warmups=1,
        ),
    )
    if args.mode != "resident":
        kwargs.update(
            attention_config={"hisparse_config": {"device_buffer_size": args.buffer}},
            kv_transfer_config=KVTransferConfig(
                kv_connector="HiSparseConnector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "host_pool_gib": args.host_gib,
                    "dsv41_sources": args.sources,
                },
            ),
        )
    if args.validate:
        assert args.mode != "resident"
        kwargs["worker_cls"] = "graph_validate.ValidatingWorker"
    if args.mode == "candidate":
        assert args.buffer >= 16384
        kwargs["worker_cls"] = (
            "candidate_prefetch.CandidateValidatingWorker"
            if args.validate
            else "candidate_prefetch.CandidateWorker"
        )
    write_json(
        args.output / "run.json",
        dict(
            arguments={
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            engine_kwargs={
                k: dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v
                for k, v in kwargs.items()
            },
            jobs_sha256=hashlib.sha256(args.jobs.read_bytes()).hexdigest(),
            source_sha256=sources,
            isolated_library_sha256=hashlib.sha256(
                Path(os.environ["VLLM_DSV41_HISPARSE_OPS_LIB"]).read_bytes()
            ).hexdigest(),
            scope=(
                "Timed arrivals, native scheduler, no HTTP transport "
                "or tokenization time"
            ),
        ),
    )

    def progress(state, **fields):
        record = dict(state=state, utc_seconds=time.time(), **fields)
        write_json(args.output / "status.json", record)
        print(json.dumps(record), flush=True)

    engine = None
    try:
        progress("loading")
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))
        await engine.collective_rpc(execution_counts, args=("install",))
        initial = await engine.collective_rpc(worker_inventory)
        for rank in initial:
            assert set(rank["owners"]) == {2, 8, 14, 20}
            assert len(rank["indexers"]) == 4
            assert all(
                s["device"].startswith("cuda") for s in rank["indexers"].values()
            )
            for owner, state in rank["owners"].items():
                assert state["device"].startswith(
                    "cpu"
                    if args.mode != "resident" and owner in args.sources
                    else "cuda"
                )
        write_json(args.output / "initial-inventory.json", initial)

        async def warmup(i):
            async for _ in engine.generate(
                {"prompt_token_ids": [1] * 2048},
                SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
                f"warmup-{i}",
            ):
                pass

        await asyncio.gather(*(warmup(i) for i in range(4)))
        tokenizer = engine.get_tokenizer()
        bases = {}
        for length in sorted(
            {r["context_tokens"] for j in jobs for r in j["requests"]}
        ):
            bases[length] = build_prompt(tokenizer, "/opt/vllm-source", "code", length)
        original_receive = engine.engine_core.get_output_async
        payloads = []

        async def receive():
            payload = await original_receive()
            stats = payload.scheduler_stats
            payloads.append(
                dict(
                    timestamp=payload.timestamp,
                    tokens=[
                        dict(
                            request_id=r.request_id,
                            count=len(r.new_token_ids),
                            corrupt=r.num_nans_in_logits,
                            events=[(int(e.type), e.timestamp) for e in r.events or []],
                        )
                        for r in payload.outputs
                    ],
                    running=stats.num_running_reqs if stats else None,
                    waiting=stats.num_waiting_reqs if stats else None,
                    kv_usage=stats.kv_cache_usage if stats else None,
                    details=dataclasses.asdict(stats.iteration_details)
                    if stats and stats.iteration_details
                    else None,
                )
            )
            return payload

        engine.engine_core.get_output_async = receive
        for job in jobs:
            name = job["name"]
            output = args.output / name
            output.mkdir()
            prepared = []
            for i, request in enumerate(job["requests"]):
                context = request["context_tokens"]
                assert context + request["output_tokens"] <= args.max_model_len
                base, corpus = bases[context]
                shift = (i + 1) * 8191 % context
                tokens = base[shift:] + base[:shift]
                prepared.append((request, tokens, f"{name}-{i:03d}"))
            write_json(
                output / "prompts.json",
                [
                    dict(
                        **request,
                        request_id=rid,
                        prompt_sha256=hashlib.sha256(
                            json.dumps(tokens, separators=(",", ":")).encode()
                        ).hexdigest(),
                        corpus_manifest=bases[request["context_tokens"]][1],
                    )
                    for request, tokens, rid in prepared
                ],
            )
            before = await engine.collective_rpc(
                worker_inventory, kwargs={"reset": True}
            )
            await engine.collective_rpc(execution_counts, args=("reset",))
            if args.validate:
                from graph_validate import control

                await engine.collective_rpc(control, args=("reset",))
            if args.mode == "candidate":
                from candidate_prefetch import control as candidate_control

                await engine.collective_rpc(candidate_control, args=("reset",))
            payloads.clear()
            results = []
            progress("generating", case=name)
            started = time.perf_counter()
            semaphore = asyncio.Semaphore(job.get("client_concurrency", len(prepared)))

            async def request_one(
                request, tokens, rid, started=started, results=results, output=output
            ):
                await asyncio.sleep(
                    max(0, started + request["arrival_seconds"] - time.perf_counter())
                )
                submitted = time.perf_counter()
                first = None
                received = []
                chunks = []
                final = None
                async for response in engine.generate(
                    {"prompt_token_ids": tokens, "cache_salt": rid},
                    SamplingParams(
                        temperature=0,
                        max_tokens=request["output_tokens"],
                        ignore_eos=True,
                        seed=0,
                        output_kind=RequestOutputKind.DELTA,
                    ),
                    rid,
                ):
                    now = time.perf_counter()
                    count = sum(len(c.token_ids) for c in response.outputs)
                    if count:
                        first = now if first is None else first
                        chunks.append((now - started, count))
                        received.extend(
                            t for c in response.outputs for t in c.token_ids
                        )
                    if response.finished:
                        final = response
                finished = time.perf_counter()
                assert final is not None and first is not None
                assert len(received) == request["output_tokens"]
                record = dict(
                    **request,
                    request_id=rid,
                    submitted_seconds=submitted - started,
                    arrival_lag_seconds=submitted
                    - started
                    - request["arrival_seconds"],
                    first_token_seconds=first - started,
                    finished_seconds=finished - started,
                    ttft_seconds=first - submitted,
                    latency_seconds=finished - submitted,
                    received_tokens=len(received),
                    output_token_ids=received,
                    client_chunks=chunks,
                    metrics=dataclasses.asdict(final.metrics)
                    if final.metrics
                    else None,
                )
                results.append(record)
                write_json(output / f"{rid}.json", record)

            async def bounded_request(
                prepared_request, request_one=request_one, semaphore=semaphore
            ):
                async with semaphore:
                    await request_one(*prepared_request)

            await asyncio.wait_for(
                asyncio.gather(*(bounded_request(p) for p in prepared)),
                timeout=args.case_timeout,
            )
            wall = time.perf_counter() - started
            steps = list(payloads)
            write_json(output / "steps.json", steps)
            for result in results:
                rid = result["request_id"]
                internal_ids = {
                    r["request_id"]
                    for p in steps
                    for r in p["tokens"]
                    if r["request_id"] == rid or r["request_id"].startswith(rid + "-")
                }
                assert len(internal_ids) == 1, (rid, internal_ids)
                internal_id = internal_ids.pop()
                result["engine_request_id"] = internal_id
                times = [
                    p["timestamp"]
                    for p in steps
                    for r in p["tokens"]
                    if r["request_id"] == internal_id and r["count"]
                ]
                events = [
                    e
                    for p in steps
                    for r in p["tokens"]
                    if r["request_id"] == internal_id
                    for e in r["events"]
                ]
                result["engine_intertoken_ms"] = [
                    1000 * (b - a) for a, b in zip(times, times[1:])
                ]
                assert len(times) == result["received_tokens"], rid
                result["preemptions"] = sum(e[0] == 3 for e in events)
                queued = next((t for kind, t in events if kind == 1), None)
                scheduled = next((t for kind, t in events if kind == 2), None)
                result["initial_queue_seconds"] = (
                    scheduled - queued
                    if queued is not None and scheduled is not None
                    else None
                )
            assert not any(r["corrupt"] for p in steps for r in p["tokens"])
            assert all(r["count"] <= 1 for p in steps for r in p["tokens"])
            assert sum(r["count"] for p in steps for r in p["tokens"]) == sum(
                r["received_tokens"] for r in results
            )
            details = [p["details"] for p in steps if p["details"]]
            mixed_steps = [
                d for d in details if d["num_ctx_tokens"] and d["num_generation_tokens"]
            ]
            assert mixed_steps, "The trace did not actually mix prefill with decode"
            validation = None
            if args.validate:
                validation = await engine.collective_rpc(control, args=("read",))
                assert all(v["host_miss_rows"] > 0 for v in validation)
                assert all(v["prefill_staging_checks"] > 0 for v in validation)
                assert all(v["prefill_reuse_checks"] > 0 for v in validation)
                if job.get("require_all_source_misses"):
                    assert all(
                        all(s["host_miss_rows"] > 0 for s in v["sources"].values())
                        for v in validation
                    )
            candidate_validation = None
            if args.mode == "candidate":
                candidate_validation = await engine.collective_rpc(
                    candidate_control, args=("read",)
                )
            after = await engine.collective_rpc(worker_inventory)
            execution = await engine.collective_rpc(execution_counts, args=("read",))
            summary = dict(
                job=job,
                mode=args.mode,
                validate=args.validate,
                complete=True,
                wall_seconds=wall,
                output_tokens_per_second=sum(r["received_tokens"] for r in results)
                / wall,
                all_requests=summarize(results),
                by_class={
                    kind: summarize([r for r in results if r["class"] == kind])
                    for kind in sorted({r["class"] for r in results})
                },
                mixed_steps=len(mixed_steps),
                total_steps=len(details),
                max_running=max((p["running"] or 0 for p in steps), default=0),
                max_waiting=max((p["waiting"] or 0 for p in steps), default=0),
                max_active_decodes=max(
                    (d["num_generation_requests"] for d in details), default=0
                ),
                max_arrival_lag_seconds=max(r["arrival_lag_seconds"] for r in results),
                requests=sorted(results, key=lambda r: r["request_id"]),
                before=before,
                after=after,
                validation=validation,
                candidate_validation=candidate_validation,
                execution=execution,
            )
            write_json(output / "result.json", summary)
            progress(
                "case_complete",
                case=name,
                wall_seconds=wall,
                output_tokens_per_second=summary["output_tokens_per_second"],
            )
        progress("complete")
    except BaseException:
        write_json(args.output / "error.json", dict(traceback=traceback.format_exc()))
        progress("failed")
        raise
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["resident", "hisparse", "candidate"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--kv-bytes", type=int, default=1342177280)
    parser.add_argument("--max-model-len", type=int, default=139264)
    parser.add_argument("--max-seqs", type=int, default=8)
    parser.add_argument("--max-batch", type=int, default=2048)
    parser.add_argument(
        "--sources", type=lambda s: list(map(int, s.split(","))), default=[20]
    )
    parser.add_argument("--buffer", type=int, default=2048)
    parser.add_argument("--host-gib", type=int, default=2)
    parser.add_argument("--case-timeout", type=int, default=1800)
    parser.add_argument("--validate", action="store_true")
    options = parser.parse_args()
    os.environ.update(
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS="8192",
    )
    asyncio.run(main(options))
