# SPDX-License-Identifier: Apache-2.0

from vllm.cmd.main import main as vllm_main


# Backwards compatibility for the move from vllm.scripts to vllm.cmd.main
def main():
    vllm_main()
