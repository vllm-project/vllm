# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from vllm.benchmarks.serve import add_cli_args
from vllm.benchmarks.serve import main as bench_main
from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase
from vllm.eval.runner import (
    _collect_environment,
    _sanitize_model_name,
    preload_lm_eval_requests,
    score_lm_eval_offline,
)


class BenchmarkServingSubcommand(BenchmarkSubcommandBase):
    """The `serve` subcommand for `vllm bench`."""

    name = "serve"
    help = "Benchmark the online serving throughput."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        if args.eval:
            _run_eval(args)
        else:
            bench_main(args)


def _run_eval(args: argparse.Namespace) -> None:
    """Run bench serve with lm_eval accuracy scoring in a single pass."""
    if not args.eval_tasks:
        raise SystemExit("--eval requires --eval-tasks, e.g. --eval-tasks gsm8k")
    if not args.model:
        raise SystemExit("--eval requires --model to be specified explicitly")

    _BACKEND_DEFAULT_ENDPOINT = {
        "openai-chat": "/v1/chat/completions",
    }
    if args.endpoint == "/v1/completions":
        corrected = _BACKEND_DEFAULT_ENDPOINT.get(args.backend)
        if corrected:
            args.endpoint = corrected
            print(
                f"[eval] Auto-set --endpoint to {corrected} for {args.backend} backend"
            )

    if args.backend == "openai-chat":
        print(
            "[eval] WARNING: --backend openai-chat uses chat messages. "
            "lm_eval tasks expect the completions API (--backend openai). "
            "Accuracy will differ from canonical lm_eval output."
        )

    # Preload lm_eval prompts before the benchmark timer starts
    print("[eval] Preloading lm_eval task prompts ...")
    requests, instances, task_dict = preload_lm_eval_requests(
        task_names=args.eval_tasks,
        model=args.model,
        limit=args.eval_limit,
        num_fewshot=getattr(args, "eval_num_fewshot", None),
    )
    print(f"[eval] Loaded {len(requests)} prompts from {args.eval_tasks}")
    args.num_prompts = len(requests)

    # Collect stop sequences from all instances so multi-task runs
    # with different stop tokens are handled correctly.
    if instances:
        all_stops: set[str] = set()
        for inst in instances:
            until = inst.args[1].get("until", [])
            if isinstance(until, str):
                all_stops.add(until)
            else:
                all_stops.update(until)
        if all_stops:
            extra = dict(getattr(args, "extra_body", None) or {})
            if "stop" not in extra:
                extra["stop"] = list(all_stops)
                args.extra_body = extra

    result = bench_main(args, prebuilt_requests=requests)

    generated_texts = result.pop("_lm_eval_generated_texts", [])
    accuracy = score_lm_eval_offline(instances, task_dict, generated_texts)

    _write_eval_record(args, result, accuracy)


def _write_eval_record(
    args: argparse.Namespace,
    result: dict,
    accuracy: dict,
) -> None:
    """Write a merged accuracy + performance JSONL record."""
    base_url = (
        args.base_url
        if args.base_url is not None
        else f"http://{args.host}:{args.port}"
    )
    model = result.get("model_id") or args.model
    record = {
        "metadata": {
            "run_id": str(uuid.uuid4()),
            "model": model,
            "tasks": args.eval_tasks,
            "bench_type": "serve",
            "base_url": base_url,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "accuracy": accuracy,
        "performance": result,
        "environment": _collect_environment(),
    }
    output = args.eval_output or str(
        Path("./eval_results") / f"{_sanitize_model_name(model)}.jsonl"
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\n[eval] merged record written to {output}")
