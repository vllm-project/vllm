# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write the explicitly safe subset of the Ascend benchmark environment."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

# Keep this list deliberately small and exact. In particular, do not replace it
# with a prefix match: credentials can be carried by GitHub, HF, publication,
# proxy, SSH, or future VLLM_* environment variables.
SAFE_ENVIRONMENT_KEYS = (
    "ASCEND_DEVICE_SELECTION_ATTEMPT",
    "ASCEND_GLOBAL_LOG_LEVEL",
    "ASCEND_HOME_PATH",
    "ASCEND_LAUNCH_BLOCKING",
    "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_SLOG_PRINT_TO_STDOUT",
    "ASCEND_VISIBLE_DEVICES",
    "HARDWARE_CHIP_MODEL",
    "HCCL_OP_EXPANSION_MODE",
    "LD_LIBRARY_PATH",
    "NPU_SMI_BIN",
    "PYTHONNOUSERSITE",
    "TASK_QUEUE_ENABLE",
    "VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION",
    "VLLM_ASCEND_FORCE_CPU_SLOT_MAPPING",
    "VLLM_ASCEND_FORCE_NATIVE_ROPE",
    "VLLM_ENABLE_MC2",
    "VLLM_ENABLE_V1_MULTIPROCESSING",
    "VLLM_PLUGINS",
    "VLLM_TARGET_DEVICE",
)


def collect_safe_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """Return only explicitly allowlisted diagnostic environment entries."""
    return {key: environ[key] for key in SAFE_ENVIRONMENT_KEYS if key in environ}


def write_safe_environment(output: Path, environ: Mapping[str, str]) -> None:
    output.write_text(
        json.dumps(
            collect_safe_environment(environ),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    write_safe_environment(args.output, os.environ)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
