"""
evaluation/evaluate.py

Evaluation pipeline for CUDA-Agent on the CUDA-Agent-Ops-6K dataset.

Reproduces the evaluation methodology from the paper:
  - Run the agent on each task in the dataset (or KernelBench)
  - Report: Pass Rate, Faster Rate, Speedup (geometric mean)
  - Compare against baselines: torch.compile, eager, other LLMs

Paper results (KernelBench, 250 tasks):
  Level 1 (100): Pass=100%, Faster-than-compile=99%, Speedup=1.87x
  Level 2 (100): Pass=100%, Faster-than-compile=100%, Speedup=2.80x
  Level 3  (50): Pass=94%, Faster-than-compile=90%, Speedup=1.52x
  Overall:       Pass=98.8%, Faster-than-compile=96.8%, Speedup=2.11x

Usage:
    python -m cuda_agent.evaluation.evaluate \
        --model BytedTsinghua-SIA/Seed1.6 \
        --dataset CUDA-Agent-Ops-6K \
        --max-samples 100 \
        --output-dir ./eval_results
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from cuda_agent.agent.environment import CUDAAgentEnvironment, Reward
from cuda_agent.agent.react_agent import CUDAReActAgent, Trajectory
from cuda_agent.data.dataset_loader import (
    CUDAAgentSample,
    load_cuda_agent_dataset,
    sort_by_difficulty,
)
from cuda_agent.data.task_generator import TaskGenerator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class EvalMetrics:
    total: int = 0
    passed: int = 0           # compiled + correct
    faster_than_eager: int = 0
    faster_than_compile: int = 0
    speedups: list[float] = field(default_factory=list)   # only for passed tasks
    compile_errors: int = 0
    correctness_failures: int = 0
    per_sample: list[dict] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / max(self.total, 1)

    @property
    def faster_compile_rate(self) -> float:
        return self.faster_than_compile / max(self.total, 1)

    @property
    def geometric_mean_speedup(self) -> float:
        """Geometric mean speedup (paper metric)."""
        if not self.speedups:
            return 0.0
        log_sum = sum(math.log(max(s, 1e-9)) for s in self.speedups)
        return math.exp(log_sum / len(self.speedups))

    def summary(self) -> dict:
        return {
            "total": self.total,
            "pass_rate": self.pass_rate,
            "faster_compile_rate": self.faster_compile_rate,
            "geometric_mean_speedup": self.geometric_mean_speedup,
            "passed": self.passed,
            "faster_than_compile": self.faster_than_compile,
            "compile_errors": self.compile_errors,
            "correctness_failures": self.correctness_failures,
        }

    def to_paper_table_row(self, model_name: str) -> str:
        """Format results matching the paper's Table 3."""
        return (
            f"| {model_name:<30} | "
            f"{self.pass_rate*100:.1f}% | "
            f"{self.faster_compile_rate*100:.1f}% | "
            f"{self.geometric_mean_speedup:.2f}x |"
        )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class CUDAAgentEvaluator:
    """
    Evaluates a CUDA-Agent on the Ops-6K dataset or KernelBench tasks.

    Args:
        agent:          CUDAReActAgent to evaluate.
        env:            CUDAAgentEnvironment for sandboxed execution.
        task_generator: TaskGenerator for creating task directories.
        output_dir:     Where to save detailed results.
        max_turns:      Max agent turns per episode (paper: 200 at eval time).
    """

    MAX_TURNS_EVAL = 200  # paper: 200 turns at evaluation

    def __init__(
        self,
        agent: CUDAReActAgent,
        env: CUDAAgentEnvironment,
        task_generator: TaskGenerator,
        output_dir: str | Path,
        max_turns: int = MAX_TURNS_EVAL,
    ):
        self.agent = agent
        self.env   = env
        self.task_gen = task_generator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_turns = max_turns

    def evaluate(
        self,
        samples: list[CUDAAgentSample],
        split_name: str = "eval",
    ) -> EvalMetrics:
        """
        Evaluate the agent on a list of samples.

        Returns aggregate EvalMetrics.
        """
        metrics = EvalMetrics()

        for i, sample in enumerate(samples):
            logger.info(
                "Evaluating sample %d/%d (id=%s, ops=%s) …",
                i + 1, len(samples), sample.sample_id, sample.ops_str,
            )
            t0 = time.perf_counter()

            task_dir = self.task_gen.generate_task(
                sample, task_id=f"eval_{split_name}_{i:05d}"
            )
            try:
                traj = self.agent.run_episode(sample, self.env, task_dir)
            except Exception as exc:  # noqa: BLE001
                logger.error("Sample %d failed with exception: %s", i, exc)
                traj = _failed_trajectory(sample)
            finally:
                try:
                    self.env.cleanup(self.env.reset(task_dir))
                except Exception:  # noqa: BLE001
                    pass

            elapsed = time.perf_counter() - t0
            metrics.total += 1
            sample_result = self._record_trajectory(traj, elapsed)
            metrics.per_sample.append(sample_result)

            if traj.final_reward >= Reward.CORRECT_NO_SPEEDUP:
                metrics.passed += 1
            if traj.final_reward >= Reward.FASTER_THAN_EAGER:
                metrics.faster_than_eager += 1
            if traj.final_reward == Reward.FASTER_THAN_COMPILE:
                metrics.faster_than_compile += 1
                if traj.speedup_vs_compile > 0:
                    metrics.speedups.append(traj.speedup_vs_compile)
            if traj.final_reward == Reward.CORRECTNESS_FAILURE:
                if traj.steps and any("compile" in s.action_input if isinstance(s.action_input, str)
                                      else False for s in traj.steps):
                    metrics.compile_errors += 1
                else:
                    metrics.correctness_failures += 1

            # Print running metrics every 10 samples
            if (i + 1) % 10 == 0:
                logger.info(
                    "Progress %d/%d | pass=%.1f%% | faster_compile=%.1f%% | speedup=%.2fx",
                    i + 1, len(samples),
                    metrics.pass_rate * 100,
                    metrics.faster_compile_rate * 100,
                    metrics.geometric_mean_speedup,
                )

        self._save_results(metrics, split_name)
        return metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _record_trajectory(self, traj: Trajectory, elapsed_s: float) -> dict:
        return {
            "sample_id": traj.sample_id,
            "final_reward": int(traj.final_reward),
            "speedup_vs_compile": traj.speedup_vs_compile,
            "num_turns": traj.total_turns,
            "elapsed_s": elapsed_s,
            "passed": traj.final_reward >= Reward.CORRECT_NO_SPEEDUP,
            "faster_than_compile": traj.final_reward == Reward.FASTER_THAN_COMPILE,
        }

    def _save_results(self, metrics: EvalMetrics, split_name: str) -> None:
        # Summary JSON
        summary_path = self.output_dir / f"{split_name}_summary.json"
        summary_path.write_text(
            json.dumps(metrics.summary(), indent=2), encoding="utf-8"
        )
        # Per-sample JSONL
        per_sample_path = self.output_dir / f"{split_name}_per_sample.jsonl"
        with per_sample_path.open("w", encoding="utf-8") as f:
            for row in metrics.per_sample:
                f.write(json.dumps(row) + "\n")

        logger.info("Results saved to %s", self.output_dir)
        logger.info("\n%s", _format_summary_table(metrics))


def _format_summary_table(m: EvalMetrics) -> str:
    return (
        "\n" + "=" * 60 + "\n"
        "CUDA-Agent Evaluation Summary\n"
        "=" * 60 + "\n"
        f"  Total samples:          {m.total}\n"
        f"  Pass rate:              {m.pass_rate*100:.1f}%\n"
        f"  Faster than compile:    {m.faster_compile_rate*100:.1f}%\n"
        f"  Geo-mean speedup:       {m.geometric_mean_speedup:.2f}x\n"
        f"  Compile errors:         {m.compile_errors}\n"
        f"  Correctness failures:   {m.correctness_failures}\n"
        "=" * 60
    )


def _failed_trajectory(sample: CUDAAgentSample) -> Trajectory:
    from cuda_agent.agent.react_agent import Trajectory
    return Trajectory(
        sample_id=sample.sample_id,
        final_reward=Reward.CORRECTNESS_FAILURE,
        speedup_vs_compile=0.0,
        done=True,
    )


# ---------------------------------------------------------------------------
# Comparison table (paper Table 3)
# ---------------------------------------------------------------------------

PAPER_RESULTS = {
    "CUDA-Agent (ours)": {
        "pass_rate": 0.988, "faster_compile_rate": 0.968, "speedup": 2.11
    },
    "Claude Opus 4.5": {
        "pass_rate": 0.952, "faster_compile_rate": 0.60, "speedup": 1.46
    },
    "Gemini 3 Pro": {
        "pass_rate": 0.912, "faster_compile_rate": 0.58, "speedup": 1.42
    },
    "Seed1.6 (base)": {
        "pass_rate": 0.740, "faster_compile_rate": 0.14, "speedup": 0.69
    },
    "GLM 4.6": {
        "pass_rate": 0.756, "faster_compile_rate": None, "speedup": 0.57
    },
    "Kimi K2": {
        "pass_rate": 0.668, "faster_compile_rate": None, "speedup": 0.66
    },
}


def print_paper_comparison_table() -> None:
    """Print the paper's Table 3 for reference."""
    header = f"{'Model':<32} {'Pass Rate':>10} {'Faster/Compile':>16} {'Speedup':>10}"
    print("=" * len(header))
    print("KernelBench Results (Paper Table 3)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for model, r in PAPER_RESULTS.items():
        faster = f"{r['faster_compile_rate']*100:.1f}%" if r["faster_compile_rate"] else "N/A"
        print(f"{model:<32} {r['pass_rate']*100:>9.1f}% {faster:>16} {r['speedup']:>9.2f}x")
    print("=" * len(header))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CUDA-Agent on the Ops-6K dataset.")
    p.add_argument("--model",        default="BytedTsinghua-SIA/Seed1.6")
    p.add_argument("--vllm-host",    default="http://localhost:8000")
    p.add_argument("--template-dir", default="cuda_agent/agent_workdir")
    p.add_argument("--output-dir",   default="./eval_results")
    p.add_argument("--max-samples",  type=int, default=100)
    p.add_argument("--max-turns",    type=int, default=200)
    p.add_argument("--cuda-arch",    default="9.0")
    p.add_argument("--sort-by-difficulty", action="store_true",
                   help="Sort samples by difficulty (curriculum order)")
    p.add_argument("--print-paper-table", action="store_true",
                   help="Print paper comparison table and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.print_paper_table:
        print_paper_comparison_table()
        return

    # Load dataset
    samples = load_cuda_agent_dataset(max_samples=args.max_samples)
    if args.sort_by_difficulty:
        samples = sort_by_difficulty(samples)

    # Build components
    agent = CUDAReActAgent(
        model=args.model,
        max_turns=args.max_turns,
        vllm_host=args.vllm_host,
    )
    template_dir = Path(args.template_dir)
    env = CUDAAgentEnvironment(
        template_dir=template_dir,
        cuda_arch=args.cuda_arch,
        max_turns=args.max_turns,
    )
    task_gen = TaskGenerator(
        template_dir=template_dir,
        output_dir=Path(args.output_dir) / "tasks",
    )
    evaluator = CUDAAgentEvaluator(
        agent=agent,
        env=env,
        task_generator=task_gen,
        output_dir=args.output_dir,
        max_turns=args.max_turns,
    )

    # Run evaluation
    metrics = evaluator.evaluate(samples)
    print_paper_comparison_table()
    print("\nYour model results:")
    print(metrics.to_paper_table_row(args.model))


if __name__ == "__main__":
    main()
