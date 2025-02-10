# SPDX-License-Identifier: Apache-2.0

from vllm.entrypoints.cli.main import main as vllm_main


# Backwards compatibility for the move from vllm.scripts to
# vllm.entrypoints.cli.main
def main():
    vllm_main()
