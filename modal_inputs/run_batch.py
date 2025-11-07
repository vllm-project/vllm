# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ABOUTME: Entrypoint for running OpenAI batch jobs with EPS instrumentation on CPU.
# ABOUTME: Forces CpuPlatform and dumps EPS metrics path details for debugging.

import asyncio
from pathlib import Path

from vllm import platforms  # type: ignore
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.run_batch import main as run_batch_main
from vllm.entrypoints.openai.run_batch import parse_args
from vllm.platforms.cpu import CpuPlatform  # type: ignore


def _resolve_metrics_path(args, engine_args) -> Path | None:
    path = getattr(args, "eps_metrics_path", None)
    if path:
        return Path(path)
    cfg = getattr(args, "eps_config", None)
    if cfg and getattr(cfg, "metrics_path", None):
        return Path(cfg.metrics_path)
    if engine_args and getattr(engine_args.eps_config, "metrics_path", None):
        return Path(engine_args.eps_config.metrics_path)
    return None


def _run() -> None:
    platforms.current_platform = CpuPlatform()
    args = parse_args()
    engine_args = AsyncEngineArgs.from_cli_args(args)
    print(f"eps_metrics_path arg: {getattr(args, 'eps_metrics_path', None)}")
    print(f"eps namespace keys: {[k for k in vars(args) if k.startswith('eps')]}")
    print(f"engine_args.eps_config: {engine_args.eps_config}")
    cfg = getattr(args, "eps_config", None)
    if cfg is not None:
        print(f"eps_config.metrics_path: {getattr(cfg, 'metrics_path', None)}")
    asyncio.run(run_batch_main(args))
    metrics_path = _resolve_metrics_path(args, engine_args)
    if metrics_path and metrics_path.exists():
        print(metrics_path.read_text(), end="")
    elif metrics_path:
        print(f"(metrics path {metrics_path} missing)")
    else:
        print("(metrics path not set)")


if __name__ == "__main__":
    _run()
