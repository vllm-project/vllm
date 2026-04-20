# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Argv/env/cwd/stdio handover between the client-side `vllm serve` call
and the restored snapshot-helper process.

Protocol:
  1. Client writes a JSON payload to /tmp/vllm-resume-<uuid>.json.
  2. Client sets VLLM_RESUME_PAYLOAD=<path> in the restored process's env.
     (In practice this env lives in the snapshot's environ via
      criu's --override-env hook; prototype writes to a well-known
      location and the helper reads it at resume time.)
  3. Client kicks SIGUSR2 on the restored pid.
  4. Helper's _resume() handler reads the payload and calls main().

FD handover (stdin/stdout/stderr) is not implemented in the prototype;
in v1 it will go via SCM_RIGHTS on a socketpair set up before criu
restore, with fd numbers baked into the payload JSON.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import uuid
from pathlib import Path


def _default_payload_path() -> Path:
    return Path(tempfile.gettempdir()) / f"vllm-resume-{uuid.uuid4().hex}.json"


def write_payload(
    argv: list[str] | None = None,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    stdio_fds: tuple[int, int, int] | None = None,
) -> Path:
    """Serialize the resume payload to a tmpfile. Returns its path."""
    payload: dict = {
        "argv": list(argv if argv is not None else sys.argv),
        "env": dict(env if env is not None else os.environ),
        "cwd": cwd if cwd is not None else os.getcwd(),
    }
    if stdio_fds is not None:
        payload["stdin_fd"], payload["stdout_fd"], payload["stderr_fd"] = stdio_fds
    path = _default_payload_path()
    with open(path, "w") as f:
        json.dump(payload, f)
    return path
