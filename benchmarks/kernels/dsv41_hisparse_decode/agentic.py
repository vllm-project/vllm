# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure cold starts and cached, multi-turn continuations through AsyncLLM."""

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

from capture_model import build_prompt, write_json
from execution import execution_counts
from mixed import summarize, worker_inventory
from traffic import install as install_traffic
from traffic import snapshot as traffic_snapshot


def token_hash(tokens):
    return hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode()
    ).hexdigest()


class Transcript:
    def __init__(self, tokens):
        assert tokens
        self.tokens = tokens
        self.state = hashlib.sha256(
            json.dumps(tokens, separators=(",", ":")).encode()[:-1]
        )

    def append(self, tokens):
        if tokens:
            self.state.update(
                b"," + json.dumps(tokens, separators=(",", ":")).encode()[1:-1]
            )
            self.tokens.extend(tokens)

    def digest(self):
        state = self.state.copy()
        state.update(b"]")
        return state.hexdigest()


def memory_inventory(worker, reset=False):
    import torch

    if reset:
        torch.accelerator.reset_peak_memory_stats()
    result = worker_inventory(worker)
    result.pop("hisparse_row_stats")
    result["hot_traffic"] = traffic_snapshot(worker)
    result["gpu_memory"] = dict(
        reserved_bytes=torch.accelerator.memory_reserved(),
        peak_reserved_bytes=torch.accelerator.max_memory_reserved(),
    )
    return result


def attach_engine_metrics(requests, steps):
    for result in requests:
        rid = result["request_id"]
        internal_ids = {
            r["request_id"]
            for p in steps
            for r in p["tokens"]
            if r["request_id"] == rid or r["request_id"].startswith(rid + "-")
        }
        assert len(internal_ids) == 1, (rid, internal_ids)
        internal_id = internal_ids.pop()
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
        assert len(times) == result["received_tokens"], rid
        result["engine_request_id"] = internal_id
        result["engine_intertoken_ms"] = [
            1000 * (b - a) for a, b in zip(times, times[1:])
        ]
        result["preemptions"] = sum(kind == 3 for kind, _ in events)
    assert not any(r["corrupt"] for p in steps for r in p["tokens"])
    assert all(r["count"] <= 1 for p in steps for r in p["tokens"])
    assert sum(r["count"] for p in steps for r in p["tokens"]) == sum(
        r["received_tokens"] for r in requests
    )


async def main(args):
    from vllm import SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    args.output.mkdir(parents=True, exist_ok=False)
    source_manifest = json.loads(args.source_manifest.read_text())
    for name, expected in source_manifest.items():
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
        gpu_memory_utilization=args.gpu_utilization,
        kv_cache_memory_bytes=args.kv_bytes,
        safetensors_load_strategy="prefetch",
        block_size=64,
        async_scheduling=False,
        enable_prefix_caching=True,
        swa_bounded_replay=args.bounded_replay,
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
    if args.mode == "hisparse":
        kwargs.update(
            attention_config={"hisparse_config": {"device_buffer_size": 2048}},
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
        assert args.mode == "hisparse"
        kwargs["worker_cls"] = "graph_validate.ValidatingWorker"
        if args.fail_fast:
            os.environ["DSV41_VALIDATION_DUMPS"] = str(args.output / "mismatches")
    write_json(
        args.output / "run.json",
        dict(
            arguments={
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            engine_kwargs={
                key: dataclasses.asdict(value)
                if dataclasses.is_dataclass(value)
                else value
                for key, value in kwargs.items()
            },
            jobs_sha256=hashlib.sha256(args.jobs.read_bytes()).hexdigest(),
            source_sha256=source_manifest,
            isolated_library_sha256=hashlib.sha256(
                Path(os.environ["VLLM_DSV41_HISPARSE_OPS_LIB"]).read_bytes()
            ).hexdigest(),
            scope="Native scheduler; excludes HTTP, tokenization, and model startup",
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
        await engine.collective_rpc(install_traffic)
        initial = await engine.collective_rpc(memory_inventory)
        for rank in initial:
            assert set(rank["owners"]) == {2, 8, 14, 20}
            assert len(rank["indexers"]) == 4
            assert all(
                item["device"].startswith("cuda") for item in rank["indexers"].values()
            )
            for owner, state in rank["owners"].items():
                host = args.mode == "hisparse" and owner in args.sources
                assert state["device"].startswith("cpu" if host else "cuda")
        write_json(args.output / "initial-inventory.json", initial)
        tokenizer = engine.get_tokenizer()
        lengths = sorted({job["context_tokens"] for job in jobs})
        bases = {
            length: build_prompt(tokenizer, "/opt/vllm-source", "code", length)
            for length in lengths
        }
        tool_size = max(job["tool_tokens"] * job["warm_turns"] for job in jobs)
        tool_base, tool_manifest = build_prompt(
            tokenizer, "/opt/vllm-source", "code", tool_size + 4096
        )
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
                            request_id=item.request_id,
                            count=len(item.new_token_ids),
                            corrupt=item.num_nans_in_logits,
                            events=[
                                (int(event.type), event.timestamp)
                                for event in item.events or []
                            ],
                        )
                        for item in payload.outputs
                    ],
                    running=stats.num_running_reqs if stats else None,
                    waiting=stats.num_waiting_reqs if stats else None,
                    kv_usage=stats.kv_cache_usage if stats else None,
                    prefix_cache=dataclasses.asdict(stats.prefix_cache_stats)
                    if stats
                    else None,
                    details=dataclasses.asdict(stats.iteration_details)
                    if stats and stats.iteration_details
                    else None,
                )
            )
            return payload

        engine.engine_core.get_output_async = receive
        transcripts = {}
        for job in jobs:
            name = job["name"]
            output = args.output / name
            output.mkdir()
            if not job.get("carry_sessions"):
                assert await engine.reset_prefix_cache(reset_connector=True)
                transcripts.clear()
            base, corpus = bases[job["context_tokens"]]
            new_sessions = []
            prompt_receipts = []
            for index in range(job["sessions"]):
                sid = f"{job.get('cohort', name)}-s{index:03d}"
                if sid not in transcripts:
                    shift = (index + 1) * 8191 % len(base)
                    transcripts[sid] = Transcript(base[shift:] + base[:shift])
                    new_sessions.append(sid)
                prompt_receipts.append(
                    dict(
                        session=sid,
                        context_tokens=len(transcripts[sid].tokens),
                        prompt_sha256=transcripts[sid].digest(),
                    )
                )
            assert len(transcripts) == job["sessions"] and new_sessions
            write_json(
                output / "prompts.json",
                dict(
                    sessions=prompt_receipts,
                    corpus_manifest=corpus,
                    tool_corpus_manifest=tool_manifest,
                    tool_sha256=token_hash(tool_base),
                ),
            )
            phases = []
            for phase in ("cold", "warm"):
                before = await engine.collective_rpc(
                    memory_inventory, kwargs={"reset": True}
                )
                await engine.collective_rpc(execution_counts, args=("reset",))
                if args.validate:
                    from graph_validate import control

                    await engine.collective_rpc(control, args=("reset",))
                payloads.clear()
                results = []
                progress("generating", case=name, phase=phase)
                started = time.perf_counter()

                async def session(
                    sid,
                    phase=phase,
                    started=started,
                    results=results,
                    job=job,
                    transcripts=transcripts,
                    output=output,
                    name=name,
                ):
                    turns = 1 if phase == "cold" else job["warm_turns"]
                    for turn in range(turns):
                        if phase == "warm":
                            await asyncio.sleep(job["think_seconds"])
                            begin = turn * job["tool_tokens"]
                            transcripts[sid].append(
                                tool_base[begin : begin + job["tool_tokens"]]
                            )
                        tokens = transcripts[sid].tokens
                        requested = job[
                            "cold_output_tokens" if phase == "cold" else "output_tokens"
                        ]
                        assert len(tokens) + requested <= args.max_model_len
                        rid = f"{name}-{sid}-{phase}-{turn}"
                        submitted = time.perf_counter()
                        first = None
                        received = []
                        cached = 0
                        final = None
                        prompt_sha = transcripts[sid].digest()
                        async for response in engine.generate(
                            {"prompt_token_ids": tokens, "cache_salt": sid},
                            SamplingParams(
                                temperature=0,
                                max_tokens=requested,
                                ignore_eos=True,
                                seed=0,
                                output_kind=RequestOutputKind.DELTA,
                            ),
                            rid,
                        ):
                            chunk = [
                                token
                                for completion in response.outputs
                                for token in completion.token_ids
                            ]
                            if chunk:
                                first = time.perf_counter() if first is None else first
                                received.extend(chunk)
                            cached = max(cached, response.num_cached_tokens or 0)
                            if response.finished:
                                final = response
                        finished = time.perf_counter()
                        assert final is not None and first is not None
                        assert len(received) == requested
                        record = dict(
                            session=sid,
                            request_id=rid,
                            phase=phase,
                            turn=turn,
                            context_tokens=len(tokens),
                            prompt_sha256=prompt_sha,
                            output_tokens=requested,
                            received_tokens=len(received),
                            num_cached_tokens=cached,
                            submitted_seconds=submitted - started,
                            first_token_seconds=first - started,
                            finished_seconds=finished - started,
                            ttft_seconds=first - submitted,
                            latency_seconds=finished - submitted,
                            output_token_ids=received,
                            metrics=dataclasses.asdict(final.metrics)
                            if final.metrics
                            else None,
                        )
                        results.append(record)
                        write_json(output / f"{rid}.json", record)
                        transcripts[sid].append(received)

                await asyncio.wait_for(
                    asyncio.gather(
                        *(
                            session(sid)
                            for sid in (
                                new_sessions if phase == "cold" else transcripts
                            )
                        )
                    ),
                    timeout=args.case_timeout,
                )
                wall = time.perf_counter() - started
                assert all(
                    transcript.digest() == token_hash(transcript.tokens)
                    for transcript in transcripts.values()
                )
                steps = list(payloads)
                write_json(output / f"{phase}-steps.json", steps)
                attach_engine_metrics(results, steps)
                validation = None
                if args.validate:
                    validation = await engine.collective_rpc(control, args=("read",))
                after = await engine.collective_rpc(memory_inventory)
                execution = await engine.collective_rpc(
                    execution_counts, args=("read",)
                )
                details = [step["details"] for step in steps if step["details"]]
                pure = [
                    item
                    for item in details
                    if item["num_ctx_tokens"] == 0
                    and item["num_generation_tokens"]
                    == item["num_generation_requests"]
                    > 0
                ]
                summary = dict(
                    phase=phase,
                    wall_seconds=wall,
                    output_tokens_per_second=sum(
                        item["received_tokens"] for item in results
                    )
                    / wall,
                    all_requests=summarize(results),
                    cache_hit_fraction=sum(
                        item["num_cached_tokens"] for item in results
                    )
                    / sum(item["context_tokens"] for item in results),
                    peak_pure_decode_batch=max(
                        (item["num_generation_requests"] for item in pure), default=0
                    ),
                    requests=sorted(results, key=lambda item: item["request_id"]),
                    before=before,
                    after=after,
                    validation=validation,
                    execution=execution,
                )
                write_json(output / f"{phase}-result.json", summary)
                phases.append(summary)
                progress(
                    "phase_complete",
                    case=name,
                    phase=phase,
                    wall_seconds=wall,
                    output_tokens_per_second=summary["output_tokens_per_second"],
                    cache_hit_fraction=summary["cache_hit_fraction"],
                )
            if job.get("require_prefix_hits"):
                assert all(
                    item["num_cached_tokens"] > 0 for item in phases[1]["requests"]
                ), "A continuation failed to reuse any prefix"
            write_json(
                output / "result.json", dict(job=job, phases=phases, complete=True)
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
    parser.add_argument("--mode", choices=["resident", "hisparse"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--kv-bytes", type=int)
    parser.add_argument("--gpu-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=1048576)
    parser.add_argument("--max-seqs", type=int, default=32)
    parser.add_argument("--max-batch", type=int, default=8192)
    parser.add_argument(
        "--sources", type=lambda value: list(map(int, value.split(","))), default=[20]
    )
    parser.add_argument("--host-gib", type=int, default=32)
    parser.add_argument("--case-timeout", type=int, default=5400)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--bounded-replay", action=argparse.BooleanOptionalAction, default=True
    )
    options = parser.parse_args()
    os.environ.update(
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS="8192",
    )
    asyncio.run(main(options))
