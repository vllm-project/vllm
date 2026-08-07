# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Mapping

from vllm_cli.snapshot.cli import main

_invocation_environment: dict[str, str] | None = None


def capture_snapshot_environment(environment: Mapping[str, str]) -> None:
    global _invocation_environment
    _invocation_environment = dict(environment)


def snapshot_environment() -> dict[str, str]:
    return dict(
        os.environ if _invocation_environment is None else _invocation_environment
    )


__all__ = ["capture_snapshot_environment", "main", "snapshot_environment"]
