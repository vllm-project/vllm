# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import asyncio
import inspect
import sys

from test_dynamic_gg import run_tool_validation_tests
from test_gg_json import run_json_validation_tests
from test_gg_long_context import run_long_context_tests
from test_utils import RunMode


class TestGuidedGenerations:
    def __init__(self, args):
        self.args = args
        self.tests = {
            "JSON Validation Test": run_json_validation_tests,
            "Dynamic Guided Generation Test": run_tool_validation_tests,
            "Long Context Test": run_long_context_tests,
        }

    def num_args(self, func):
        return len(inspect.signature(func).parameters)

    def run_all_guided_generation_test(self):
        print("Running all model tests...\n")
        for name, test_fn in self.tests.items():
            try:
                print(f"Running {name}...")
                if self.num_args(test_fn) > 0:
                    asyncio.run(test_fn(self.args))
                else:
                    test_fn()
                print(f"{name} ✅ PASSED\n")
            except Exception as e:
                print(f"{name} ❌ FAILED with error: {e}\n")
                raise RuntimeError(f"Test {name} failed") from e


def parse_args():
    parser = argparse.ArgumentParser(description="Run Guided Generation tests")
    parser.add_argument("--model", type=str, default="CohereForAI/c4ai-command-r-v01")
    parser.add_argument("--tensor_parallel_size", type=int, default=2)

    # Mode selection
    parser.add_argument(
        "--mode",
        type=RunMode,
        choices=list(RunMode),
        default=RunMode.BOTH,
        help="Choose run mode: non-speculative, speculative, or both",
    )

    # Speculative decoding args
    parser.add_argument("--method", type=str, default="eagle")
    parser.add_argument(
        "--draft_model",
        type=str,
        default=None,
        help="Draft model for speculative decoding",
    )
    parser.add_argument(
        "--num_spec_tokens", type=int, default=4, help="Number of speculative tokens"
    )
    parser.add_argument(
        "--draft_tp",
        type=int,
        default=1,
        help="Tensor parallel size for speculative model",
    )
    parser.add_argument("--max_model_len", type=int, default=32_000)
    parser.add_argument(
        "--engine-args",
        type=str,
        default=None,
        help=(
            "CLI-style engine args to pass to AsyncLLM (e.g., '--max-model-len 32768 "
            "--enable-chunked-prefill'). "
            "If not provided, uses VLLM_HARDWARE_PROFILE_ARGS "
            "environment variable."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        runner = TestGuidedGenerations(args)
        runner.run_all_guided_generation_test()
    except Exception as e:
        print(f"\n❌ Guided Generation test failed: {e}")
        sys.exit(1)
