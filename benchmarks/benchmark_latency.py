# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys

if __name__ == "__main__":
    print("""DEPRECATED: This script has been moved to the vLLM CLI.

Please use the following command instead:
    vllm bench latency

For help with the new command, run:
    vllm bench latency --help

Alternatively, you can run the new command directly with:
    python -m vllm.entrypoints.cli.main bench latency --help
""")
    sys.exit(1)
