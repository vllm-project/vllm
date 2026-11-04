# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure a Dynamic SD table (``num_speculative_tokens_per_batch_size``) for
a target model and drafter on this GPU, in minutes, with no engine changes.
See the Dynamic SD docs for the method; implements steps 2-3 of #62740.

Example::

    vllm bench sweep tune --target Qwen/Qwen3-8B \\
        --drafter Qwen/Qwen3-8B-speculator.eagle3 --prompts-file prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.spec_decode.dynamic.schedule import (
    build_schedule,
    compress,
    goodput_table,
    load_costs,
    tokens_per_step,
)

logger = logging.getLogger(__name__)

DType = Literal["auto", "half", "float16", "bfloat16", "float", "float32"]

EPS = 0.005  # rule 1: change K only if another K is >0.5% better
MIN_WIDTH_FRAC = 0.25  # rule 2: a range must span >=25% of its start batch size
N_DECODE = 64  # decode steps timed per (batch size, K)
REPEATS = 3  # timed runs per (batch size, K), median kept


def speculative_config(schedule: list[list[int]], drafter: str, method: str) -> dict:
    """The --speculative-config dict vLLM expects for Dynamic SD."""
    k_max = max(1, max(k for _, _, k in schedule))
    return {
        "method": method,
        "model": drafter,
        "num_speculative_tokens": k_max,
        "num_speculative_tokens_per_batch_size": schedule,
    }


# ---------------------------------------------------------------------------
# Measurement stages: one process each, so every stage gets the whole GPU.
# ---------------------------------------------------------------------------


def _spec_config_costs(
    k: int, k_max: int, max_num_seqs: int, method: str, drafter: str
) -> dict:
    """Dynamic SD table pinned to K with every guess rejected."""
    return {
        "method": method,
        "model": drafter,
        "num_speculative_tokens": k_max,
        "num_speculative_tokens_per_batch_size": [[1, max_num_seqs, k]],
        "rejection_sample_method": "synthetic",
        "synthetic_acceptance_rates": [0.0] * k_max,
    }


def _measure_costs(args: TuneArgs, k: int) -> None:
    """Time the decode step for K over the batch grid; write costs_k{k}.json."""
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    from vllm.entrypoints.llm import LLM
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import SamplingParams

    llm = LLM(
        model=args.target,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=False,
        seed=0,
        speculative_config=_spec_config_costs(
            k, args.k_max, args.max_num_seqs, args.method, args.drafter
        ),
    )
    vocab = llm.get_tokenizer().vocab_size
    rng = random.Random(k)

    def prompts(b: int) -> list[TokensPrompt]:
        # Random tokens: with every guess rejected the prompt content does not
        # matter, only its length (attention work per step).
        return [
            TokensPrompt(
                prompt_token_ids=[
                    rng.randrange(1000, vocab - 1000) for _ in range(args.isl)
                ]
            )
            for _ in range(b)
        ]

    def timed(b: int, max_tokens: int) -> float:
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        t0 = time.perf_counter()
        outs = llm.generate(prompts(b), sp, use_tqdm=False)
        assert all(len(o.outputs[0].token_ids) >= max_tokens for o in outs)
        return time.perf_counter() - t0

    timed(4, 16)  # warm-up (compile caches, allocator)
    rows = []
    for b in args.batch_sizes:
        samples = []
        for _ in range(args.repeats):
            t_prefill = timed(b, 1)
            t_full = timed(b, args.n_decode + 1)
            samples.append((t_full - t_prefill) / args.n_decode * 1000)
        step_ms = statistics.median(samples)
        rows.append(
            {
                "batch_size": b,
                "k": k,
                "step_ms": round(step_ms, 3),
                "samples_ms": [round(s, 3) for s in samples],
            }
        )
        logger.info("[costs] K=%d B=%3d step=%7.2f ms", k, b, step_ms)

    out = args.output_dir / f"costs_k{k}.json"
    out.write_text(
        json.dumps(
            {
                "target": args.target,
                "method": args.method,
                "drafter": args.drafter,
                "k": k,
                "k_max": args.k_max,
                "isl": args.isl,
                "max_num_seqs": args.max_num_seqs,
                "n_decode": args.n_decode,
                "rows": rows,
            },
            indent=2,
        )
    )
    logger.info("[costs] wrote %s", out)


def _measure_acceptance(args: TuneArgs) -> None:
    """Per-position acceptance at the longest K on calibration prompts."""
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    from vllm.entrypoints.llm import LLM
    from vllm.sampling_params import SamplingParams

    with open(args.prompts_file) as f:
        prompts = [json.loads(line)["prompt"] for line in f]
    llm = LLM(
        model=args.target,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=64,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=0,
        disable_log_stats=False,
        speculative_config={
            "method": args.method,
            "model": args.drafter,
            "num_speculative_tokens": args.k_max,
        },
    )
    chats = [[{"role": "user", "content": p}] for p in prompts]
    outs = llm.chat(
        chats,  # type: ignore[arg-type]
        SamplingParams(
            temperature=args.temperature, max_tokens=args.max_tokens, seed=0
        ),
        chat_template_kwargs=args.chat_template_kwargs,
        use_tqdm=False,
    )

    c = {m.name: m for m in llm.get_metrics() if m.name.startswith("vllm:spec_decode")}
    drafts = c["vllm:spec_decode_num_drafts"].value  # type: ignore[attr-defined]
    per_pos = c["vllm:spec_decode_num_accepted_tokens_per_pos"].values  # type: ignore[attr-defined]
    rates = [round(x / drafts, 4) for x in per_pos]
    osl = [len(o.outputs[0].token_ids) for o in outs]
    steps = sum(osl) - sum(per_pos) - len(outs)  # request-steps
    result = {
        "target": args.target,
        "method": args.method,
        "drafter": args.drafter,
        "k_max": args.k_max,
        "temperature": args.temperature,
        "calibration_prompts": len(prompts),
        "drafts": drafts,
        "decode_steps": steps,
        "proposal_rate": round(drafts / steps, 4),
        "accepted_per_pos": per_pos,
        "accept_by_pos": rates,
    }
    out = args.output_dir / "acceptance.json"
    out.write_text(json.dumps(result, indent=2))
    logger.info(
        "[accept] by position %s -> tokens/step %s",
        rates,
        tokens_per_step(rates, result["proposal_rate"]),
    )


# ---------------------------------------------------------------------------
# CLI (same conventions as the other sweep subcommands).
# ---------------------------------------------------------------------------


@dataclass
class TuneArgs:
    target: str
    drafter: str
    method: str = "eagle3"
    k_max: int = 3
    include_k0: bool = False
    max_num_seqs: int = 128
    batch_sizes: list[int] = field(default_factory=list)
    isl: int = 320
    prompts_file: str = ""
    temperature: float = 1.0
    max_tokens: int = 256
    chat_template_kwargs: dict | None = None
    dtype: DType = "bfloat16"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    n_decode: int = N_DECODE
    repeats: int = REPEATS
    eps: float = EPS
    min_width_frac: float = MIN_WIDTH_FRAC
    output_dir: Path = Path("dsd-tune")

    parser_name: ClassVar[str] = "tune"
    parser_help: ClassVar[str] = (
        "Measure a Dynamic SD table (num_speculative_tokens_per_batch_size) "
        "for a target model and drafter on this GPU."
    )

    @classmethod
    def add_cli_args(cls, parser: FlexibleArgumentParser) -> None:
        parser.add_argument("--target", required=True, help="target model to serve")
        parser.add_argument("--drafter", required=True, help="draft model")
        parser.add_argument(
            "--method",
            default="eagle3",
            help="speculative decoding method (eagle, eagle3, mtp, ...)",
        )
        parser.add_argument(
            "--k-max", type=int, default=3, help="longest draft length to consider"
        )
        parser.add_argument(
            "--include-k0",
            action="store_true",
            help="also measure K=0 (speculation off). Off by default: a K=0 "
            "step still pays a drafter forward pass on current builds, "
            "so its measured cost understates the savings",
        )
        parser.add_argument(
            "--max-num-seqs",
            type=int,
            default=128,
            help="largest batch size to profile; the table covers 1..this",
        )
        parser.add_argument(
            "--batch-sizes",
            default="",
            help="comma-separated batch sizes (default: 1-16 dense, then "
            "every 8 up to --max-num-seqs)",
        )
        parser.add_argument(
            "--isl",
            type=int,
            default=320,
            help="prompt length in tokens; match your workload, it sets the "
            "attention work per step",
        )
        parser.add_argument(
            "--prompts-file",
            default="",
            help='JSONL with one {"prompt": ...} per line, for acceptance; '
            "use prompts like your traffic",
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=1.0,
            help="sampling temperature for the acceptance run; acceptance "
            "drops with sampling",
        )
        parser.add_argument(
            "--max-tokens",
            type=int,
            default=256,
            help="answer length for the acceptance run",
        )
        parser.add_argument(
            "--chat-template-kwargs",
            default="{}",
            help="JSON dict passed to the chat template (e.g. "
            "'{\"enable_thinking\": false}' for Qwen3)",
        )
        parser.add_argument("--dtype", default="bfloat16")
        parser.add_argument("--max-model-len", type=int, default=4096)
        parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
        parser.add_argument("--tensor-parallel-size", type=int, default=1)
        parser.add_argument(
            "--n-decode",
            type=int,
            default=N_DECODE,
            help="decode steps timed per (batch size, K)",
        )
        parser.add_argument(
            "--repeats",
            type=int,
            default=REPEATS,
            help="timed runs per (batch size, K); the median is kept",
        )
        parser.add_argument(
            "--eps",
            type=float,
            default=EPS,
            help="switch K only if another K is more than this fraction better",
        )
        parser.add_argument(
            "--min-width-frac",
            type=float,
            default=MIN_WIDTH_FRAC,
            help="merge a range narrower than this fraction of its start batch size",
        )
        parser.add_argument("-o", "--output-dir", default="dsd-tune")

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> TuneArgs:
        batch_sizes = (
            [int(x) for x in args.batch_sizes.split(",")]
            if args.batch_sizes
            else list(range(1, 17)) + list(range(24, args.max_num_seqs + 1, 8))
        )
        if args.prompts_file == "":
            raise ValueError(
                "--prompts-file is required: acceptance must be measured on "
                "prompts like your traffic"
            )
        if not args.include_k0 and 0 in batch_sizes:
            raise ValueError("batch sizes must be >= 1")
        return cls(
            target=args.target,
            drafter=args.drafter,
            method=args.method,
            k_max=args.k_max,
            include_k0=args.include_k0,
            max_num_seqs=args.max_num_seqs,
            batch_sizes=batch_sizes,
            isl=args.isl,
            prompts_file=args.prompts_file,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            chat_template_kwargs=json.loads(args.chat_template_kwargs),
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            n_decode=args.n_decode,
            repeats=args.repeats,
            eps=args.eps,
            min_width_frac=args.min_width_frac,
            output_dir=Path(args.output_dir),
        )


def main(args: argparse.Namespace) -> None:
    run_main(TuneArgs.from_cli_args(args))


def run_main(args: TuneArgs) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    import multiprocessing as mp

    ctx = mp.get_context("spawn")  # each stage gets the whole GPU

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ks = ([0] if args.include_k0 else []) + list(range(1, args.k_max + 1))

    for k in ks:
        logger.info("[tune] measuring step costs for K=%d", k)
        p = ctx.Process(target=_measure_costs, args=(args, k))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"cost measurement for K={k} failed (exit {p.exitcode})")

    logger.info("[tune] measuring acceptance at K=%d", args.k_max)
    p = ctx.Process(target=_measure_acceptance, args=(args,))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"acceptance measurement failed (exit {p.exitcode})")

    costs = load_costs(str(args.output_dir / "costs_k*.json"))
    acc = json.loads((args.output_dir / "acceptance.json").read_text())
    tps = tokens_per_step(acc["accept_by_pos"], acc.get("proposal_rate", 1.0))
    goodput = goodput_table(costs, tps, args.max_num_seqs)
    schedule = compress(build_schedule(goodput, args.eps, args.min_width_frac))

    (args.output_dir / "schedule.json").write_text(json.dumps(schedule))
    print(f"\ntokens/step by K: {tps}")
    print(f"num_speculative_tokens_per_batch_size: {schedule}")
    print("ready to paste into --speculative-config:")
    print(json.dumps(speculative_config(schedule, args.drafter, args.method), indent=2))
    print(f"\nwrote {args.output_dir / 'schedule.json'}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    TuneArgs.add_cli_args(parser)
    main(parser.parse_args())
