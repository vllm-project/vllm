# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Live verl tool rollouts and fixed-work replay against a real vLLM server."""

import argparse
import ast
import asyncio
import gzip
import hashlib
import json
import operator
import random
import statistics
import subprocess
import time
from collections import Counter
from contextlib import suppress
from pathlib import Path

import pyarrow.parquet as pq
from prometheus_client.parser import text_string_to_metric_families

from benchmarks.programmable_kv_cache.client import answer_number
from benchmarks.programmable_kv_cache.rollout import (
    RolloutClient,
    ToolDurationEstimator,
)
from benchmarks.programmable_kv_cache.run_rollout_matrix import (
    dataset_identity,
    source_hashes,
)


def arithmetic(expression: str) -> str:
    """Evaluate bounded arithmetic supplied by an actual model tool call."""
    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
    }
    if len(expression) > 512:
        raise ValueError("Expression is too long")

    def visit(node):
        if (
            isinstance(node, ast.Constant)
            and type(node.value) in (int, float)
            and abs(node.value) <= 1e12
        ):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            value = operations[type(node.op)](visit(node.left), visit(node.right))
            if abs(value) <= 1e15:
                return value
        raise ValueError("Use only numeric constants and basic arithmetic")

    return str(visit(ast.parse(expression, mode="eval").body))


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


async def run_episodes(episodes):
    tasks = [asyncio.create_task(episode) for episode in episodes]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def summarize(report):
    rows, episodes = report["requests"], report["episodes"]
    seconds = report["seconds"]
    tokens = sum(len(row["token_ids"]) for row in rows)
    usage = [row["usage"] for row in rows]
    ttft = [row["ttft"] for row in rows]

    def counters(text):
        values = {}
        for family in text_string_to_metric_families(text):
            for sample in family.samples:
                values[sample.name] = values.get(sample.name, 0) + sample.value
        return values

    before = counters(report.get("metrics_before", ""))
    after = counters(report.get("metrics_after", ""))
    queue_sum = after.get("vllm:request_queue_time_seconds_sum", 0) - before.get(
        "vllm:request_queue_time_seconds_sum", 0
    )
    queue_count = after.get("vllm:request_queue_time_seconds_count", 0) - before.get(
        "vllm:request_queue_time_seconds_count", 0
    )
    transfer_before = report.get("control_before", {}).get("transfers", {})
    transfer_after = report.get("control_capacity", {}).get("transfers", {})
    phases = {}
    for phase in ("queue", "prefill", "decode", "inference"):
        prefix = f"vllm:request_{phase}_time_seconds"
        elapsed = after.get(prefix + "_sum", 0) - before.get(prefix + "_sum", 0)
        count = after.get(prefix + "_count", 0) - before.get(prefix + "_count", 0)
        phases[phase] = {
            "seconds_sum": elapsed,
            "count": count,
            "mean_seconds": elapsed / count if count else None,
        }
    workload = sorted(
        (row["request_id"], row["turn"], row["prompt_ids"], len(row["token_ids"]))
        for row in rows
    )
    families = {}
    for episode in episodes:
        if "family" in episode:
            families.setdefault(episode["family"], []).append(episode)
    return {
        "completed_rollouts": len(episodes),
        "rollouts_per_second": len(episodes) / seconds,
        "output_tokens_per_second": tokens / seconds,
        "generated_tokens": tokens,
        "requests": len(rows),
        "cached_tokens": sum(
            (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
            for u in usage
        ),
        "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage),
        "ttft_p50": percentile(ttft, 0.5),
        "ttft_p95": percentile(ttft, 0.95),
        "ttft_p99": percentile(ttft, 0.99),
        "request_latency_p95": percentile([r["seconds"] for r in rows], 0.95),
        "prepare_wait_p50": percentile([r["prepare_wait_seconds"] for r in rows], 0.5),
        "prepare_wait_p99": percentile([r["prepare_wait_seconds"] for r in rows], 0.99),
        "control_rpc": report.get("control_rpc", {}),
        "driver_cpu_seconds": report.get("driver_cpu_seconds"),
        "control_status_counts": dict(Counter(r["status"] for r in report["controls"])),
        "policy_result_counts": dict(
            Counter(
                f"{r['origin']['policy_name']}:{r['action_type']}:{r['status']}"
                for r in report["controls"]
                if r.get("origin")
            )
        ),
        "server_request_phases": phases,
        "workload_sha256": hashlib.sha256(json.dumps(workload).encode()).hexdigest(),
        "rollout_latency_p95": percentile([e["seconds"] for e in episodes], 0.95),
        "client_queue_p95": percentile(
            [e["client_queue_seconds"] for e in episodes], 0.95
        ),
        "family_latency": {
            str(family): {
                "completed": len(items),
                "mean_seconds": statistics.mean(e["seconds"] for e in items),
                "p95_seconds": percentile([e["seconds"] for e in items], 0.95),
                "max_client_queue_seconds": max(
                    e["client_queue_seconds"] for e in items
                ),
            }
            for family, items in sorted(families.items())
        },
        "reward_mean": statistics.mean(e["reward"] for e in episodes)
        if episodes and all("reward" in e for e in episodes)
        else None,
        "length_limited_requests": sum(r["finish_reason"] == "length" for r in rows),
        "server_preemptions": after.get("vllm:num_preemptions_total", 0)
        - before.get("vllm:num_preemptions_total", 0),
        "transfers": {
            key: value - transfer_before.get(key, 0)
            for key, value in transfer_after.items()
        },
        "mean_server_queue_seconds": queue_sum / queue_count if queue_count else None,
        "peak_kv_usage": max(
            (
                value
                for sample in report.get("metrics_samples", [])
                for key, value in sample["values"].items()
                if "kv_cache_usage_perc" in key
            ),
            default=None,
        ),
        "peak_waiting_requests": max(
            (
                value
                for sample in report.get("metrics_samples", [])
                for key, value in sample["values"].items()
                if "num_requests_waiting{" in key
            ),
            default=None,
        ),
    }


async def metrics(client, report, stop):
    while not stop.is_set():
        response = await client.http.get("/metrics")
        response.raise_for_status()
        text = response.text
        report.setdefault("metrics_before", text)
        report["metrics_after"] = text
        wanted = (
            "num_requests_running",
            "num_requests_waiting",
            "kv_cache_usage",
            "num_preemptions",
            "request_queue_time_seconds",
            "prefix_cache_hits",
            "prefix_cache_queries",
        )
        samples = {}
        for family in text_string_to_metric_families(text):
            for sample in family.samples:
                if any(key in sample.name for key in wanted):
                    name = sample.name + json.dumps(sample.labels, sort_keys=True)
                    samples[name] = sample.value
        report.setdefault("metrics_samples", []).append(
            {
                "time": time.perf_counter(),
                "values": samples,
            }
        )
        with suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=1)


async def live(args, client, report):
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
    from verl.tools.function_tool import FunctionTool
    from verl.tools.schemas import OpenAIFunctionToolSchema
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.workers.rollout.llm_server import LLMServerClient
    from verl.workers.rollout.replica import TokenOutput

    dataset = pq.read_table(
        args.dataset / "main/test-00000-of-00001.parquet"
    ).to_pylist()
    train = pq.read_table(
        args.dataset / "main/train-00000-of-00001.parquet"
    ).to_pylist()
    indices = random.Random(args.seed).sample(range(len(dataset)), args.prompts)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": args.max_context,
                    "response_length": args.response_length,
                    "full_determinism": True,
                    "multi_turn": {
                        "max_user_turns": 8,
                        "max_assistant_turns": 6,
                        "max_parallel_calls": 2,
                        "max_tool_response_length": 512,
                        "tool_response_truncate_side": "right",
                        "format": "qwen3_coder",
                    },
                }
            }
        }
    )
    tool_rows = report["tools"] = []

    async def lookup_problem(problem_id: int):
        started = time.perf_counter()
        if not 0 <= problem_id < len(dataset):
            return "Unknown problem identifier"
        # The optional delay models an external service; execution/rewards remain real.
        if args.tool_delay:
            await asyncio.sleep(args.tool_delay)
        result = dataset[problem_id]["question"]
        tool_rows.append(
            {
                "name": "lookup_problem",
                "problem_id": problem_id,
                "seconds": time.perf_counter() - started,
                "result": result,
            }
        )
        return result

    async def calculate(expression: str):
        started = time.perf_counter()
        try:
            result = await asyncio.to_thread(arithmetic, expression)
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as error:
            result = f"Calculation error: {error}"
        if args.tool_delay:
            await asyncio.sleep(args.tool_delay)
        tool_rows.append(
            {
                "name": "calculate",
                "expression": expression,
                "seconds": time.perf_counter() - started,
                "result": result,
            }
        )
        return result

    tools = []
    for name, function, description, field, kind in (
        (
            "lookup_problem",
            lookup_problem,
            "Retrieve the math problem by identifier.",
            "problem_id",
            "integer",
        ),
        (
            "calculate",
            calculate,
            "Compute a numeric arithmetic expression.",
            "expression",
            "string",
        ),
    ):
        schema = OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            field: {"type": kind, "description": description}
                        },
                        "required": [field],
                    },
                },
            }
        )
        tools.append(FunctionTool(name, function, schema, True))

    class HTTPServerClient(LLMServerClient):
        async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
            row = await client.generate(request_id, prompt_ids, sampling_params)
            return TokenOutput(
                token_ids=row["token_ids"],
                log_probs=row["log_probs"],
                stop_reason="completed",
            )

    estimator = ToolDurationEstimator(args.prediction_seconds)

    class ControlledToolLoop(ToolAgentLoop):
        async def _handle_processing_tools_state(self, agent_data):
            names = tuple(
                call.name for call in agent_data.tool_calls[: self.max_parallel_calls]
            )
            predicted = (
                estimator.predict(names)
                if args.prediction_mode == "adaptive"
                else args.prediction_seconds
            )
            await client.pause(agent_data.request_id, predicted)
            started = time.monotonic()
            state = await super()._handle_processing_tools_state(agent_data)
            elapsed = time.monotonic() - started
            estimator.observe(names, elapsed)
            report["tool_waits"].append(
                {
                    "request_id": agent_data.request_id,
                    "turn": client.turns[agent_data.request_id] - 1,
                    "tools": names,
                    "predicted_seconds": predicted,
                    "actual_seconds": elapsed,
                }
            )
            return state

    manager = HTTPServerClient(config)
    semaphore = asyncio.Semaphore(args.concurrency)
    report["indices"] = indices
    report["verl_revision"] = subprocess.check_output(
        ["git", "-C", args.verl_root, "rev-parse", "HEAD"], text=True
    ).strip()

    async def episode(offset, index, sample):
        identifier = offset * args.samples + sample
        family = offset % args.families
        examples = train[family * args.fewshot : (family + 1) * args.fewshot]
        context = "\n\n".join(
            f"Question: {e['question']}\nAnswer: {e['answer']}" for e in examples
        )
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are solving a math task in study group {family}. "
                    "First call lookup_problem with the requested identifier. "
                    "Then use calculate to check the arithmetic before answering. "
                    "Give a brief explanation followed by #### and the final number. "
                    "Do not answer a reference example. Reference worked examples:\n"
                    + context
                ),
            },
            {
                "role": "user",
                "content": f"Solve problem identifier {index} using the tools.",
            },
        ]
        queued = time.perf_counter()
        async with semaphore:
            started = time.perf_counter()
            loop = ControlledToolLoop(
                trainer_config=DictConfigWrap(config),
                server_manager=manager,
                tokenizer=tokenizer,
                processor=None,
                dataset_cls=RLHFDataset,
                data_config=DictConfigWrap(
                    OmegaConf.create(
                        {"apply_chat_template_kwargs": {"enable_thinking": False}}
                    )
                ),
                hf_model_type="qwen3_5",
                tools=ToolListWrap(tools),
            )
            output = await loop.run(
                {"temperature": args.temperature, "max_tokens": args.max_tokens},
                priority=identifier,
                raw_prompt=messages,
            )
            request_id = f"det-{identifier}"
            rows = [r for r in client.rows if r["request_id"] == request_id]
            final = rows[-1]["text"]
            expected = answer_number(dataset[index]["answer"])
            predicted = answer_number(final)
            assert len(output.response_ids) == len(output.response_mask)
            if output.response_logprobs:
                assert len(output.response_ids) == len(output.response_logprobs)
            await client.finish(request_id)
            report["episodes"].append(
                {
                    "request_id": request_id,
                    "index": index,
                    "sample": sample,
                    "family": family,
                    "seconds": time.perf_counter() - started,
                    "client_queue_seconds": started - queued,
                    "expected": expected,
                    "predicted": predicted,
                    "reward": int(expected is not None and expected == predicted),
                    "num_turns": output.num_turns,
                    "response_mask": output.response_mask,
                    "response_ids": output.response_ids,
                    "response_logprobs": output.response_logprobs,
                    "metrics": output.metrics.model_dump(),
                }
            )

    report["workload_started"] = time.perf_counter()
    report["workload_cpu_started"] = time.process_time()
    await run_episodes(
        (
            episode(offset, index, sample)
            for sample in range(args.samples)
            for offset, index in enumerate(indices)
        )
    )


async def replay(args, client, report):
    with gzip.open(args.trace, "rt") as file:
        trace = json.load(file)
    trajectories = {}
    episode_metadata = {e["request_id"]: e for e in trace["episodes"]}
    for row in trace["requests"]:
        trajectories.setdefault(row["request_id"], []).append(row)
    semaphore = asyncio.Semaphore(args.concurrency)
    estimator = ToolDurationEstimator(args.prediction_seconds)
    tool_names = {
        (row["request_id"], row["turn"]): tuple(row["tools"])
        for row in trace.get("tool_waits", [])
    }

    async def episode(identifier, rows):
        rows.sort(key=lambda r: r["turn"])
        queued = time.perf_counter()
        async with semaphore:
            started = time.perf_counter()
            for turn, row in enumerate(rows):
                if turn:
                    pause = max(
                        args.replay_pause_floor,
                        row["started"] - rows[turn - 1]["finished"],
                    )
                    names = tool_names.get((identifier, turn - 1), ("recorded_tool",))
                    predicted = (
                        pause
                        if args.oracle
                        else estimator.predict(names)
                        if args.prediction_mode == "adaptive"
                        else args.prediction_seconds
                    )
                    await client.pause(identifier, predicted)
                    pause_started = time.monotonic()
                    await asyncio.sleep(pause)
                    elapsed = time.monotonic() - pause_started
                    estimator.observe(names, elapsed)
                    report["tool_waits"].append(
                        {
                            "request_id": identifier,
                            "turn": turn - 1,
                            "tools": names,
                            "predicted_seconds": predicted,
                            "actual_seconds": elapsed,
                        }
                    )
                generated = await client.generate(
                    identifier,
                    row["prompt_ids"],
                    {"temperature": 0},
                    fixed_tokens=len(row["token_ids"]),
                )
                assert len(generated["token_ids"]) == len(row["token_ids"])
            await client.finish(identifier)
            report["episodes"].append(
                {
                    **{
                        k: v
                        for k, v in episode_metadata[identifier].items()
                        if k in ("family", "index", "sample")
                    },
                    "request_id": identifier,
                    "seconds": time.perf_counter() - started,
                    "client_queue_seconds": started - queued,
                    "num_turns": len(rows),
                }
            )
            if len(report["episodes"]) % 32 == 0:
                print(
                    json.dumps({"completed_rollouts": len(report["episodes"])}),
                    flush=True,
                )

    report["workload_started"] = time.perf_counter()
    report["workload_cpu_started"] = time.process_time()
    await run_episodes(episode(key, rows) for key, rows in trajectories.items())


async def run(args):
    client = RolloutClient(
        args.base_url,
        args.mode,
        seed=args.seed,
        trace_rpc=args.trace_control_rpc,
        control_timeout=args.control_timeout,
    )
    report = {
        "clock": {"wall_time": time.time(), "monotonic": time.perf_counter()},
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "requests": client.rows,
        "controls": client.controls,
        "control_rpc": client.rpc_stats,
        "control_rpc_requests": client.rpc_rows,
        "episodes": [],
        "tool_waits": [],
        "passed": False,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_hashes": source_hashes(),
        "dataset_identity": dataset_identity(args.dataset),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cache_salt": client.run_id,
    }
    stop = asyncio.Event()
    sampler = None
    try:
        await client.initialize()
        report["capabilities"] = client.capabilities
        if client.invocation:
            report["control_before"] = await client.call("capacity", {})
        sampler = asyncio.create_task(metrics(client, report, stop))
        started = time.perf_counter()
        if args.trace:
            await replay(args, client, report)
        else:
            await live(args, client, report)
        report["seconds"] = time.perf_counter() - report.get(
            "workload_started", started
        )
        report["driver_cpu_seconds"] = (
            time.process_time() - report["workload_cpu_started"]
        )
        response = await client.http.get("/metrics")
        response.raise_for_status()
        report["metrics_after"] = response.text
        if client.invocation:
            report["control_capacity"] = await client.call("capacity", {})
            assert not report["control_capacity"].get("policy_errors"), report[
                "control_capacity"
            ]
            assert not report["control_capacity"]["records"], report["control_capacity"]
            assert not any(r["status"] == "client_error" for r in client.controls)
            report["logical_after"] = await client.call("associations", {})
            assert not report["logical_after"]["items"], report["logical_after"]
        report["summary"] = summarize(report)
        report["passed"] = True
        print(json.dumps(report["summary"]), flush=True)
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        stop.set()
        if sampler is not None:
            try:
                await sampler
            except Exception as error:
                report["metrics_error"] = str(error)
                report["passed"] = False
        try:
            await client.close()
        except Exception as error:
            report["cleanup_error"] = str(error)
            report["passed"] = False
        finally:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(args.output, "wt") as file:
                json.dump(report, file)
    if not report["passed"]:
        raise RuntimeError("Benchmark cleanup or metrics collection failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--verl-root", type=Path, default=Path("../verl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument(
        "--mode",
        choices=[
            "native",
            "idle",
            "noop",
            "quota",
            "retention",
            "priority",
            "movement",
            "composed",
            "policy",
        ],
        default="native",
    )
    parser.add_argument("--prompts", type=int, default=16)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--families", type=int, default=4)
    parser.add_argument("--fewshot", type=int, default=24)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--response-length", type=int, default=2048)
    parser.add_argument("--tool-delay", type=float, default=0)
    parser.add_argument("--prediction-seconds", type=float, default=0.25)
    parser.add_argument(
        "--prediction-mode", choices=("fixed", "adaptive"), default="adaptive"
    )
    parser.add_argument(
        "--replay-pause-floor",
        type=float,
        default=0,
        help="Replay-only modeled external-tool wait floor in seconds",
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Replay-only oracle using recorded pause durations",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace-control-rpc", action="store_true")
    parser.add_argument("--control-timeout", type=float, default=30)
    args = parser.parse_args()
    if not 0 <= args.replay_pause_floor <= 60:
        parser.error("replay-pause-floor must be between zero and 60 seconds")
    if args.replay_pause_floor and not args.trace:
        parser.error("replay-pause-floor requires a recorded trace")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
