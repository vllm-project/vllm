# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Utility functions for environment variable handling."""

import os
from pathlib import Path


def parse_path(value: str) -> Path:
    """Expand ~ and env vars, return a Path."""
    return Path(os.path.expanduser(os.path.expandvars(value)))


def parse_list(value: str, separator: str = ",") -> list[str]:
    """Split a separated string into a list of stripped values."""
    if not value.strip():
        return []
    return [item.strip() for item in value.split(separator) if item.strip()]
