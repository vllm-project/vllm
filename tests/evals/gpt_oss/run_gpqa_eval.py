# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the gpt-oss GPQA evaluator with an explicit output-token budget."""

import argparse

from gpt_oss.evals.gpqa_eval import GPQAEval
from gpt_oss.evals.responses_sampler import ResponsesSampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--n-threads", required=True, type=int)
    parser.add_argument("--max-output-tokens", required=True, type=int)
    args = parser.parse_args()

    sampler = ResponsesSampler(
        model=args.model,
        reasoning_model=True,
        reasoning_effort=args.reasoning_effort,
        base_url=args.base_url,
        max_tokens=args.max_output_tokens,
    )
    result = GPQAEval(n_repeats=8, n_threads=args.n_threads)(sampler)
    # gpt-oss returns a NumPy scalar. Convert it so the parent process gets a
    # stable, dependency-independent representation that its regex can parse.
    print({"metric": float(result.score)})


if __name__ == "__main__":
    main()
