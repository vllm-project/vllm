# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["snapshot"]:
        from vllm_cli.snapshot import capture_snapshot_environment

        capture_snapshot_environment(os.environ)
    if args[:2] == ["snapshot", "restore"]:
        from vllm_cli.snapshot import main as snapshot_main

        snapshot_main(args[1:])
        return

    from vllm.entrypoints.cli.main import main as vllm_main

    if argv is None:
        vllm_main()
        return

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *args]
        vllm_main()
    finally:
        sys.argv = original_argv
