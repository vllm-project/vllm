# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Closed-cohort decode comparison with native scheduler iteration receipts."""

import argparse
import dataclasses
import functools
import hashlib
import json
import os
import statistics
import time
import traceback
from pathlib import Path

from capture_model import build_prompt, write_json
from execution import execution_counts
from validate import inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["resident", "hisparse"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validate-graphs", action="store_true")
    parser.add_argument("--kv-bytes", type=int, default=20 * 2**30)
    parser.add_argument("--max-model-len", type=int, default=1048576)
    parser.add_argument("--max-seqs", type=int, default=64)
    parser.add_argument("--max-batch", type=int, default=4096)
    parser.add_argument("--buffer", type=int, default=4096)
    parser.add_argument(
        "--sources", type=lambda s: list(map(int, s.split(","))), default=[20]
    )
    parser.add_argument("--host-gib", type=int, default=32)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--graphs", action="store_true")
    args = parser.parse_args()
    assert not (args.eager and args.graphs)
    assert not (args.validate and args.validate_graphs)
    if args.validate_graphs:
        assert args.graphs and args.mode == "hisparse"
    validation_run = args.validate or args.validate_graphs
    args.output.mkdir(parents=True, exist_ok=False)
    sources = None
    if args.source_manifest:
        sources = json.loads(args.source_manifest.read_text())
        for relative, expected in sources.items():
            assert (
                hashlib.sha256(
                    (args.source_manifest.parent / relative).read_bytes()
                ).hexdigest()
                == expected
            ), relative
    os.environ.update(
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS="8192",
    )
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

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
        enforce_eager=args.eager,
        enable_prefix_caching=False,
        swa_bounded_replay=False,
        kernel_config={"enable_jit_warmup": False},
        disable_log_stats=False,
        enable_logging_iteration_details=True,
        limit_mm_per_prompt={"image": 0},
        seed=0,
        scheduler_cls="decode_scheduler.DecodeBarrierScheduler",
    )
    if args.mode == "hisparse":
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
    if args.graphs:
        kwargs["compilation_config"] = dict(
            mode=0,
            cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=list(range(1, args.max_seqs + 1)),
            cudagraph_num_of_warmups=1,
        )
    if args.validate_graphs:
        kwargs["worker_cls"] = "graph_validate.ValidatingWorker"
    write_json(
        args.output / "run.json",
        dict(
            arguments=vars(args)
            | {
                "output": str(args.output),
                "jobs": str(args.jobs),
                "source_manifest": str(args.source_manifest)
                if args.source_manifest
                else None,
            },
            source_sha256=sources,
            isolated_library_sha256=hashlib.sha256(
                Path(os.environ["VLLM_DSV41_HISPARSE_OPS_LIB"]).read_bytes()
            ).hexdigest()
            if os.environ.get("VLLM_DSV41_HISPARSE_OPS_LIB")
            else None,
            engine_kwargs={
                k: dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v
                for k, v in kwargs.items()
            },
            scope=(
                "Closed cohorts; pure decode steps reported separately from "
                "full-run throughput. Validation shadows excluded from timings."
            ),
        ),
    )

    def progress(state, **fields):
        record = dict(state=state, utc_seconds=time.time(), **fields)
        write_json(args.output / "status.json", record)
        print(json.dumps(record), flush=True)

    llm = None
    try:
        progress("loading")
        llm = LLM(**kwargs)
        llm.collective_rpc(execution_counts, args=("install",))
        initial = llm.apply_model(inventory)
        for rank in initial:
            assert set(rank["owners"]) == {2, 8, 14, 20}
            assert len(rank["indexers"]) == 4 and all(
                state["device"].startswith("cuda")
                for state in rank["indexers"].values()
            )
            for owner, state in rank["owners"].items():
                assert state["device"].startswith(
                    "cpu"
                    if args.mode == "hisparse" and owner in args.sources
                    else "cuda"
                )
        write_json(args.output / "initial-inventory.json", initial)
        llm.generate(
            {"prompt_token_ids": [1] * 2048},
            SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
            use_tqdm=False,
        )
        for index, job in enumerate(json.loads(args.jobs.read_text())):
            name = job.get("name", f"case-{index:03d}")
            output = args.output / name
            output.mkdir()
            progress("preparing", case=name, **job)
            count, context, generation = (
                job["concurrency"],
                job["context_tokens"],
                job["output_tokens"],
            )
            assert context + generation <= args.max_model_len
            tokens, corpus = build_prompt(
                llm.get_tokenizer(), "/opt/vllm-source", "code", context
            )
            prompts = []
            hashes = []
            for request in range(count):
                shift = request * 8191 % context
                prompt = tokens[shift:] + tokens[:shift]
                prompts.append(
                    dict(prompt_token_ids=prompt, cache_salt=f"decode:{name}:{request}")
                )
                hashes.append(
                    hashlib.sha256(
                        json.dumps(prompt, separators=(",", ":")).encode()
                    ).hexdigest()
                )
            assert len(set(hashes)) == count
            write_json(
                output / "prompt.json",
                dict(
                    corpus_manifest=corpus,
                    prompt_sha256=hashes,
                    rotations=[i * 8191 % context for i in range(count)],
                    context_tokens=context,
                ),
            )
            if args.validate:
                assert args.mode == "hisparse"
                from validate import control

                llm.apply_model(
                    functools.partial(
                        control,
                        action="install",
                        output=str(output),
                        max_tokens=context + generation,
                        concurrency=count,
                    )
                )
            before = llm.apply_model(functools.partial(inventory, reset=True))
            if args.validate_graphs:
                from graph_validate import control as graph_control

                llm.collective_rpc(graph_control, args=("reset",))
            llm.collective_rpc(execution_counts, args=("reset",))
            payloads = []
            original = llm.llm_engine.engine_core.get_output

            def receive(_get_output=original, _payloads=payloads):
                payload = _get_output()
                stats = payload.scheduler_stats
                _payloads.append(
                    dict(
                        timestamp=payload.timestamp,
                        tokens=[
                            dict(
                                request_id=r.request_id,
                                count=len(r.new_token_ids),
                                corrupt=r.num_nans_in_logits,
                            )
                            for r in payload.outputs
                        ],
                        running=stats.num_running_reqs if stats else None,
                        waiting=stats.num_waiting_reqs if stats else None,
                        details=dataclasses.asdict(stats.iteration_details)
                        if stats and stats.iteration_details
                        else None,
                    )
                )
                return payload

            progress("generating", case=name, **job)
            llm.llm_engine.engine_core.get_output = receive
            started = time.monotonic()
            try:
                results = llm.generate(
                    prompts,
                    SamplingParams(
                        temperature=0,
                        max_tokens=generation,
                        ignore_eos=True,
                        seed=0,
                        extra_args={"benchmark_decode_cohort": count},
                    ),
                    use_tqdm=False,
                )
            finally:
                wall = time.monotonic() - started
                llm.llm_engine.engine_core.get_output = original
            assert len(results) == count
            assert all(len(r.outputs[0].token_ids) == generation for r in results)
            assert not any(r["corrupt"] for p in payloads for r in p["tokens"])
            after = llm.apply_model(inventory)
            execution = llm.collective_rpc(execution_counts, args=("read",))
            for rank in execution:
                decodes = [s for s in rank["steps"] if s[1] == s[2] == count]
                assert decodes
                if args.graphs or args.eager:
                    assert all(s[0] == args.graphs for s in decodes), rank
            validation = None
            if args.validate:
                validation = llm.apply_model(
                    functools.partial(control, action="finish", output=str(output))
                )
            if args.validate_graphs:
                validation = llm.collective_rpc(graph_control, args=("read",))
                for rank in validation:
                    write_json(output / f"validation-rank{rank['rank']:02d}.json", rank)
            if job.get("require_host_misses"):
                assert validation is not None and all(
                    v["host_miss_rows"] > 0 for v in validation
                )
            if job.get("require_prefill_reuse"):
                assert validation is not None and all(
                    v["prefill_reuse_checks"] > 0
                    and v["prefill_staging_checks"] > 0
                    and v["prefill_bad_bytes"] == 0
                    for v in validation
                )
            if job.get("require_prefill_attention"):
                assert validation is not None and all(
                    v["prefill_attention_checks"] > 0 for v in validation
                )
            pure = [
                p["details"]
                for p in payloads
                if p["details"]
                and not p["details"]["num_ctx_requests"]
                and p["details"]["num_generation_tokens"]
            ]
            cohort_start = next(
                i for i, p in enumerate(pure) if p["num_generation_requests"] == count
            )
            preparation_decodes = pure[:cohort_start]
            pure = pure[cohort_start:]
            measured = pure[16:]
            assert pure and all(p["num_generation_requests"] == count for p in pure), (
                "Decode cohort did not run together",
                [p["num_generation_requests"] for p in pure],
            )
            times = {}
            for payload in payloads:
                for request in payload["tokens"]:
                    if request["count"]:
                        assert request["count"] == 1
                        times.setdefault(request["request_id"], []).append(
                            payload["timestamp"]
                        )
            intervals = [
                1000 * (b - a)
                for values in times.values()
                for a, b in zip(values[16:], values[17:])
            ]
            requests = [
                dict(
                    request_id=r.request_id,
                    token_ids=list(r.outputs[0].token_ids),
                    metrics=dataclasses.asdict(r.metrics),
                )
                for r in results
            ]
            summary = dict(
                job=job,
                mode=args.mode,
                validate=validation_run,
                complete=True,
                wall_seconds=wall,
                full_run_tokens_per_second=count * generation / wall,
                pure_decode_tokens_per_second=sum(
                    p["num_generation_tokens"] for p in measured
                )
                / (sum(p["elapsed_ms"] for p in measured) / 1000)
                if measured
                else None,
                pure_decode_steps=len(measured),
                preparation_decode_steps=len(preparation_decodes),
                max_active_decodes=max(
                    (p["num_generation_requests"] for p in pure), default=0
                ),
                median_intertoken_ms=statistics.median(intervals)
                if intervals
                else None,
                p95_intertoken_ms=sorted(intervals)[int(0.95 * (len(intervals) - 1))]
                if intervals
                else None,
                p99_intertoken_ms=sorted(intervals)[int(0.99 * (len(intervals) - 1))]
                if intervals
                else None,
                requests=requests,
                before=before,
                after=after,
                validation=validation,
                execution=execution,
            )
            write_json(output / "steps.json", payloads)
            write_json(output / "result.json", summary)
            progress("case_complete", case=name)
        progress("complete")
    except BaseException:
        write_json(args.output / "error.json", dict(traceback=traceback.format_exc()))
        progress("failed")
        raise
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
