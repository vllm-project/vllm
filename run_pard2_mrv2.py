"""Scratch driver: PARD-2 acceptance without pytest fixtures, so it can run
against either model runner. Not part of the change; deleted before commit."""

import os
import sys

from tests.evals.gsm8k.gsm8k_eval import _build_gsm8k_prompts
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig
from vllm.v1.metrics.reader import Metric

TARGET = os.environ.get("TARGET", "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8")
DRAFT = os.environ.get("DRAFT", "amd/PARD2-Llama-3.1-8B")
K = int(os.environ.get("K", "5"))
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "50"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))


def _metric(metrics: list[Metric], name: str) -> float:
    for m in metrics:
        if m.name == name:
            return float(m.value)
    return 0.0


def main() -> int:
    print(f"V2 runner: {os.environ.get('VLLM_USE_V2_MODEL_RUNNER')}", flush=True)
    prompts = _build_gsm8k_prompts(num_questions=NUM_PROMPTS, num_shots=5)[0]
    print(f"prompts: {len(prompts)}  target: {TARGET}  draft: {DRAFT}  K={K}")

    llm = LLM(
        model=TARGET,
        speculative_config={"model": DRAFT, "num_speculative_tokens": K},
        max_model_len=4096,
        enforce_eager=True,
        disable_log_stats=False,
        compilation_config=CompilationConfig(),
    )
    llm.generate(prompts, SamplingParams(temperature=0, max_tokens=MAX_TOKENS))

    metrics = llm.get_metrics()
    n_drafts = _metric(metrics, "vllm:spec_decode_num_drafts")
    n_accepted = _metric(metrics, "vllm:spec_decode_num_accepted_tokens")
    acceptance = 1.0 if n_drafts == 0 else 1 + n_accepted / n_drafts
    print(f"\nnum_drafts={n_drafts:.0f} accepted={n_accepted:.0f}")
    print(f"ACCEPTANCE_LENGTH={acceptance:.4f}")
    return 0 if acceptance > 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
