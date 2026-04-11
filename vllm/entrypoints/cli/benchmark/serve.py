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
    EvalConfig,
    _collect_environment,
    _sanitize_model_name,
    score_gpt_oss_offline,
    score_lm_eval_offline,
)

_VALID_EVAL_BACKENDS = ("lm_eval", "gpt_oss")


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
            eval_cfg = _parse_eval_config(args.eval)
            _run_eval(args, eval_cfg)
        else:
            bench_main(args)


def _parse_eval_config(raw: dict) -> EvalConfig:
    """Validate the raw --eval dict and return a typed EvalConfig."""
    backend = raw.get("backend", "lm_eval")
    if backend not in _VALID_EVAL_BACKENDS:
        raise SystemExit(
            f"--eval.backend must be one of {_VALID_EVAL_BACKENDS}, got '{backend}'."
        )
    # num_samples is a friendly alias for limit; num_samples wins if both set.
    raw_limit = raw.get("num_samples", raw.get("limit"))
    return EvalConfig(
        backend=backend,
        tasks=raw.get("tasks"),
        limit=float(raw_limit) if raw_limit is not None else None,
        num_samples=(
            float(raw["num_samples"]) if raw.get("num_samples") is not None else None
        ),
        num_fewshot=(
            int(raw["num_fewshot"]) if raw.get("num_fewshot") is not None else None
        ),
        max_tokens=(
            int(raw["max_tokens"]) if raw.get("max_tokens") is not None else None
        ),
        reasoning_effort=raw.get("reasoning_effort"),
        output=raw.get("output"),
    )


def _run_eval(args: argparse.Namespace, eval_cfg: EvalConfig) -> None:
    """Dispatch `--eval.*` to the right backend shim."""
    if not eval_cfg.tasks:
        raise SystemExit("--eval requires tasks, e.g. --eval.tasks gsm8k")
    if not args.model:
        raise SystemExit("--eval requires --model to be specified explicitly")

    if eval_cfg.backend == "lm_eval":
        _run_eval_lm_eval(args, eval_cfg)
    elif eval_cfg.backend == "gpt_oss":
        _run_eval_gpt_oss(args, eval_cfg)
    else:
        raise SystemExit(f"Unknown eval backend: {eval_cfg.backend}")


def _run_eval_lm_eval(args: argparse.Namespace, eval_cfg: EvalConfig) -> None:
    if args.backend != "openai":
        raise SystemExit(
            f"--eval.backend lm_eval requires --backend openai (got '{args.backend}')."
        )

    if getattr(args, "temperature", None) is None:
        args.temperature = 0
        print("[eval] Auto-set --temperature 0 for greedy decoding")

    args.dataset_name = "lm_eval"
    args.lm_eval_tasks = eval_cfg.tasks
    args.lm_eval_num_fewshot = eval_cfg.num_fewshot
    args.lm_eval_limit = int(eval_cfg.limit) if eval_cfg.limit is not None else None
    args.lm_eval_max_tokens = eval_cfg.max_tokens
    # The scorer needs per-request generations, which bench_main strips
    # from the result unless --save-detailed is set.
    args.save_detailed = True

    result = bench_main(args)
    dataset = args._lm_eval_dataset
    accuracy = score_lm_eval_offline(
        dataset.instances, dataset.task_dict, result.get("generated_texts", [])
    )
    _write_eval_record(args, eval_cfg, result, accuracy)


def _run_eval_gpt_oss(args: argparse.Namespace, eval_cfg: EvalConfig) -> None:
    # gpt_oss reasoning evals are chat-shaped and need /v1/chat/completions.
    # Promote the default openai backend; reject anything else.
    if args.backend == "openai":
        args.backend = "openai-chat"
        args.endpoint = "/v1/chat/completions"
        print("[eval] Auto-set --backend openai-chat for gpt_oss")
    elif args.backend != "openai-chat":
        raise SystemExit(
            f"--eval.backend gpt_oss requires --backend openai-chat "
            f"(got '{args.backend}')."
        )

    # Match `python -m gpt_oss.evals` default sampling temperature.
    if getattr(args, "temperature", None) is None:
        args.temperature = 1.0
        print("[eval] Auto-set --temperature 1.0 (gpt_oss default)")

    args.dataset_name = "gpt_oss_eval"
    args.gpt_oss_eval_tasks = eval_cfg.tasks
    args.gpt_oss_eval_limit = (
        int(eval_cfg.limit) if eval_cfg.limit is not None else None
    )
    args.gpt_oss_eval_max_tokens = eval_cfg.max_tokens
    args.gpt_oss_eval_reasoning_effort = eval_cfg.reasoning_effort
    args.save_detailed = True

    result = bench_main(args)
    dataset = args._gpt_oss_eval_dataset
    accuracy = score_gpt_oss_offline(
        dataset.task_evals, dataset.row_index, result.get("generated_texts", [])
    )
    _write_eval_record(args, eval_cfg, result, accuracy)


def _write_eval_record(
    args: argparse.Namespace,
    eval_cfg: EvalConfig,
    result: dict,
    accuracy: dict,
) -> None:
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
            "tasks": eval_cfg.tasks,
            "bench_type": "serve",
            "base_url": base_url,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "accuracy": accuracy,
        "performance": result,
        "environment": _collect_environment(),
    }
    output = eval_cfg.output or str(
        Path("./eval_results") / f"{_sanitize_model_name(model)}.jsonl"
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\n[eval] merged record written to {output}")
