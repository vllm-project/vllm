# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import runpy
import sys

from offline_inference_tt import check_tt_model_supported, register_tt_models

register_tt_models()  # Import and register models from tt-metal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-70B-Instruct",
        help="Model name",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=32,
        help="Maximum number of sequences to be processed in a single iteration",
    )
    parser.add_argument(
        "--num_scheduler_steps", type=int, default=10, help="Number of scheduler steps"
    )
    parser.add_argument(
        "--block_size", type=int, default=64, help="KV cache block size"
    )
    args, _ = parser.parse_known_args()

    check_tt_model_supported(args.model)

    sys.argv.extend(
        [
            "--model",
            args.model,
            "--block_size",
            str(args.block_size),
            "--max_num_seqs",
            str(args.max_num_seqs),
            "--num_scheduler_steps",
            str(args.num_scheduler_steps),
        ]
    )
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
