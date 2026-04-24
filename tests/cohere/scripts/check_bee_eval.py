# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
VLLM_DIR = SCRIPT_DIR.parent.parent.parent


def main(results_file, ground_truths_file):
    """
    Compare if the results match ground truths
    """
    with open(results_file) as f1, open(ground_truths_file) as f2:
        results = json.load(f1)
        ground_truths = json.load(f2)

    mismatches = []

    for result in results:
        model = result["model"]
        if ground_truths.get(model):
            for eval_task in ground_truths[model]:
                if eval_task in result["eval_results"]:
                    eval_result = result["eval_results"][eval_task]
                    print(
                        f"Evaluating model {model} with bee eval task "
                        f"{eval_task}, result: {eval_result}"
                    )
                    ground_truth = ground_truths[model][eval_task][0]
                    atol = ground_truths[model][eval_task][1]
                    if not np.isclose(
                        eval_result,
                        ground_truth,
                        atol=atol,
                    ):
                        mismatch_msg = (
                            f"{model} failed with bee eval task "
                            f"{eval_task}, result {eval_result} "
                            f"doesn't match ground truth {ground_truth} "
                            f"(atol={atol})"
                        )
                        mismatches.append(mismatch_msg)

    if mismatches:
        error_msg = "\n".join(
            ["Found the following mismatches:", *[f"  - {msg}" for msg in mismatches]]
        )
        print(error_msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare specific parts of two JSON files."
    )
    parser.add_argument(
        "--results-file",
        default=VLLM_DIR / "results" / "eval_results_summary.json",
        required=False,
        help="Path to the results_file.",
    )
    parser.add_argument(
        "--ground-truths-file",
        default=VLLM_DIR
        / "tests"
        / "cohere"
        / "configs"
        / "bee_tasks"
        / "ground_truths.json",
        required=False,
        help="Path to the ground_truths_file.",
    )

    args = parser.parse_args()
    main(args.results_file, args.ground_truths_file)
