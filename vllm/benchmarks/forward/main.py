# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the model forward of exact batches as one-shot CUDA graphs.

Each case is a list of requests `(query_len, kv_len, num_draft_tokens)`
per DP rank. The bench builds the batch without the scheduler, runs it through
the real `execute_model` (input prep, attention metadata, DP sync), and times
the target forward and configured drafter captured as a CUDA graph. Target
sampling, rejection sampling and CPU step overhead aren't measured.

Single GPU: `vllm bench forward --model M --case decode=32*1@8192`.
Multi-GPU: `torchrun --nproc-per-node=N -m vllm.entrypoints.cli.main bench
forward ...`, one process per GPU.
"""

import argparse
import json
import os
import socket
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from vllm.benchmarks.forward.shapes import (
    BenchCase,
    jitter_kv_lens,
    load_cases,
    parse_case,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

logger = init_logger(__name__)


def add_cli_args(parser: FlexibleArgumentParser) -> None:
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="`NAME=REQS`, where REQS is a comma list of `[N*]Q[@K][c][:D]` "
        "(query length Q, KV length K including the query (default Q), D draft "
        "tokens, N requests). A prefill's prefix is a prefix-cache hit unless c "
        "marks a chunked prefill. E.g. `decode=32*1@8192`, "
        "`verify=16*4@4096:3`, `prefill=8192`, `hit=2048@10240`, "
        "`mix=8192,32*1@8192`. Repeatable.",
    )
    parser.add_argument("--shapes", help="JSON or YAML file with cases.")
    parser.add_argument(
        "--kv-len-jitter",
        type=float,
        default=0.5,
        help="Draw each request's prefix (KV length minus query length) "
        "uniformly within +-this fraction of its value, like the varied lengths "
        "of a real batch, capped at the max model length. Seeded by --seed only, "
        "so every DP rank gets the same lengths. 0 keeps the exact lengths.",
    )
    parser.add_argument(
        "--kv-init",
        choices=["zeros"],
        default="zeros",
        help="Initial KV cache contents of each case's blocks.",
    )
    parser.add_argument(
        "--routing",
        choices=["auto", "real", "balanced"],
        default="auto",
        help="MoE routing. balanced: fixed routing with equal load per EP rank. "
        "auto: balanced with dummy weights, else real.",
    )
    parser.add_argument(
        "--exclude-drafter",
        action="store_true",
        help="Time only the target model. By default, a configured MTP, EAGLE, "
        "DSpark, or DFlash drafter is included, assuming all draft tokens "
        "are accepted. Target sampling and rejection sampling are excluded.",
    )
    parser.add_argument("--iters", type=int, default=20, help="Replays per sample.")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output-json", help="Write results to this file.")
    parser.add_argument(
        "--profile-dir",
        help="After timing, trace a few replays of each case and write "
        "`<case>.rank<r>.trace.json.gz` (Perfetto) and `<case>.rank<r>.kernels.json` "
        "(GPU time per kernel per forward) here. Summarize or compare with "
        "`python -m vllm.benchmarks.forward.profile A.kernels.json [B.kernels.json]`.",
    )
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(load_format="dummy", enable_prefix_caching=False)


def _setup_env() -> None:
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # Each case's eager pass compiles exactly the shapes it needs.
    os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
    if "RANK" in os.environ:
        return  # Launched by torchrun.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    os.environ.update(
        RANK="0",
        LOCAL_RANK="0",
        WORLD_SIZE="1",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )


def _make_engine_args(args: argparse.Namespace, cases: list[BenchCase]):
    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    engine_args = EngineArgs.from_cli_args(args)
    engine_args.distributed_executor_backend = "external_launcher"
    if engine_args.pipeline_parallel_size > 1:
        from vllm.benchmarks.forward.pipeline import ForwardPipelineExecutor

        engine_args.distributed_executor_backend = ForwardPipelineExecutor
    engine_args.enable_prefix_caching = False
    engine_args.compilation_config.mode = CompilationMode.NONE
    engine_args.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    if engine_args.max_num_seqs is None:
        engine_args.max_num_seqs = max(len(c.requests) for c in cases)
    if engine_args.max_num_batched_tokens is None:
        # Prefix hits may replay up to a window more per request.
        engine_args.max_num_batched_tokens = max(
            8192,
            *(
                c.num_tokens + 1024 * sum(r.prefix_hit for r in c.requests)
                for c in cases
            ),
        )
    return engine_args


def run_case(runner, case: BenchCase, args: argparse.Namespace) -> dict[str, Any]:
    required_drafts = max((r.num_draft_tokens for r in case.requests), default=0)
    if required_drafts > runner.num_speculative_steps:
        raise ValueError(
            f"Case {case.name!r} needs {required_drafts} draft tokens per request, "
            f"but the runner supports {runner.num_speculative_steps}. "
            "Set --speculative-config with a sufficient num_speculative_tokens."
        )

    from vllm.benchmarks.forward.batch import build_batch
    from vllm.benchmarks.forward.capture import ForwardBench
    from vllm.benchmarks.forward.kv_init import touched_kv_views, zero_kv_blocks
    from vllm.benchmarks.forward.pipeline import PipelineBench
    from vllm.benchmarks.forward.routing import install_routing

    context = runner.vllm_config.compilation_config.static_forward_context
    kv_config = runner.kv_cache_config
    dcp_size = runner.parallel_config.decode_context_parallel_size
    rng = np.random.default_rng([args.seed, runner.dp_rank])
    include_drafter = not args.exclude_drafter
    lookahead = (
        runner.num_speculative_steps + 1
        if include_drafter and runner.vllm_config.speculative_config is not None
        else 0
    )
    if lookahead and any(
        r.kv_len + lookahead > runner.model_config.max_model_len for r in case.requests
    ):
        raise ValueError("Increase --max-model-len to leave room for draft lookahead.")
    batch = build_batch(
        case,
        kv_config.kv_cache_groups,
        kv_config.num_blocks,
        runner.model_config.get_vocab_size(),
        rng,
        dcp_size,
        lookahead,
    )
    zero_kv_blocks(context, kv_config.kv_cache_groups, batch, kv_config.num_blocks)
    profile_prefix = None
    if args.profile_dir:
        from vllm.distributed.parallel_state import get_world_group

        os.makedirs(args.profile_dir, exist_ok=True)
        rank = get_world_group().rank
        profile_prefix = os.path.join(args.profile_dir, f"{case.name}.rank{rank}")
    bench_cls = (
        PipelineBench
        if runner.parallel_config.pipeline_parallel_size > 1
        else ForwardBench
    )
    bench = bench_cls(
        runner,
        batch,
        args.iters,
        args.samples,
        touched_kv_views(
            context,
            kv_config.kv_cache_groups,
            batch,
            kv_config.num_blocks,
            dcp_size,
            lookahead,
        ),
        case.name,
        profile_prefix,
        include_drafter=include_drafter,
    )
    routing, balanced = install_routing(
        runner, args.routing, args.load_format == "dummy", batch.num_tokens, args.seed
    )
    try:
        result = bench.run()
        if balanced is not None:
            balanced.check_applied()
    finally:
        runner.execute_model(batch.finish())
        if balanced is not None:
            balanced.remove()
    return {
        "num_tokens": batch.num_tokens,
        "kv_blocks": batch.num_blocks,
        "routing": routing,
        **result,
    }


def main(args: argparse.Namespace) -> None:
    cases = [parse_case(c) for c in args.case]
    if args.shapes:
        cases += load_cases(args.shapes)
    if not cases:
        raise ValueError("Pass at least one --case or --shapes.")
    if not 0 <= args.kv_len_jitter < 1:
        raise ValueError("--kv-len-jitter must be in [0, 1).")
    _setup_env()

    from vllm import LLM
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import get_world_group

    llm = LLM.from_engine_args(_make_engine_args(args, cases))
    runner = llm.llm_engine.model_executor.driver_worker.worker.model_runner
    # Same draw on every rank: shapes are balanced across DP ranks.
    max_kv_len = runner.model_config.max_model_len
    cases = [
        jitter_kv_lens(c, args.kv_len_jitter, max_kv_len, args.seed) for c in cases
    ]
    world = get_world_group()
    results = []
    with torch.inference_mode(), set_current_vllm_config(runner.vllm_config):
        for case in cases:
            result = run_case(runner, case, args)
            gathered: list[Any] = [None] * world.world_size
            torch.distributed.all_gather_object(gathered, result, group=world.cpu_group)
            if world.rank != 0:
                continue
            rank0 = gathered[0]
            latency = rank0["latency_us"]  # Each replay is the max over ranks.
            results.append(
                {
                    "case": case.name,
                    "requests": [asdict(r) for r in case.requests],
                    "median_us": latency["median"],
                    "ranks": gathered,
                }
            )
            print(
                f"[forward] {case.name}: {len(case.requests)} reqs, "
                f"{rank0['num_tokens']} tokens, median {latency['median']:.1f} us, "
                f"sample spread {latency['sample_spread_pct']:.2f}%, graph vs "
                f"eager rel rms {rank0['graph_vs_eager']['relative_rms']:.1e}",
                flush=True,
            )
            if profile := rank0.get("profile"):
                from vllm.benchmarks.forward.profile import format_summary

                print(
                    f"[forward] {case.name} rank 0 kernels ({profile['summary']}):\n"
                    + format_summary({**profile, "kernels": profile["top_kernels"]}),
                    flush=True,
                )
            if latency["sample_spread_pct"] > 0.5:
                logger.warning(
                    "%s: unstable timing (sample medians spread %.2f%%).",
                    case.name,
                    latency["sample_spread_pct"],
                )
    if world.rank == 0 and args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
