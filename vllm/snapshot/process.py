# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import sys


def detach_snapshot_streams() -> None:
    """Remove launch-log descriptors from the reusable process image."""
    sys.stdout.flush()
    sys.stderr.flush()
    sink = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(sink, sys.stdout.fileno())
        os.dup2(sink, sys.stderr.fileno())
    finally:
        os.close(sink)
