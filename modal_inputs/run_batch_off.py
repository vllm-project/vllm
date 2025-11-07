# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ABOUTME: Minimal run_batch helper for CPU Modal reproduction runs.
# ABOUTME: Forces CpuPlatform and prints EPS metrics artifacts for inspection.

import asyncio
from pathlib import Path

from vllm import platforms  # type: ignore
from vllm.entrypoints.openai.run_batch import main as run_batch_main
from vllm.entrypoints.openai.run_batch import parse_args
from vllm.platforms.cpu import CpuPlatform  # type: ignore

platforms.current_platform = CpuPlatform()

args = parse_args()
asyncio.run(run_batch_main(args))

metrics_path = (
    Path(args.eps_config.metrics_path) if getattr(args, "eps_config", None) else None
)
if metrics_path and metrics_path.exists():
    print(metrics_path.read_text(), end="")
else:
    print("(no metrics recorded)")
