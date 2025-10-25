# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Legacy entrypoint for the standalone encoder benchmark."""

import sys


def main() -> None:
    print(
        """DEPRECATED: This script has been moved to the vLLM CLI.

Please use the following command instead:
    vllm bench encoder

For help with the new command, run:
    vllm bench encoder --help

Alternatively, you can run the new command directly with:
    python -m vllm.entrypoints.cli.main bench encoder --help
"""
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
