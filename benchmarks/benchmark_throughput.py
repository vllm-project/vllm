# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark offline inference throughput script.
"""

from vllm.benchmarks.throughput import add_cli_args, main
from vllm.utils.argparse_utils import FlexibleArgumentParser

if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark offline inference throughput.")
    add_cli_args(parser)
    args = parser.parse_args()
    main(args)
