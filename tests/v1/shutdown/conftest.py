# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Iterable
from pathlib import Path

import pytest

from vllm.platforms import current_platform


@pytest.fixture
def spawn_sitecustomize_factory(monkeypatch, tmp_path: Path):
    """Return a function that installs a given sitecustomize payload.

    Both ROCm and XPU force the `spawn` multiprocessing start method (see
    `vllm.platforms.rocm`/`vllm.platforms.xpu`), which re-imports modules in
    child processes instead of inheriting them via `fork`'s copy-on-write
    memory. A plain `monkeypatch.setattr` in the parent test process is
    therefore invisible to spawned EngineCore/worker subprocesses on these
    platforms, so the payload must be installed via `sitecustomize.py` on
    `PYTHONPATH` instead.
    """
    if not (current_platform.is_rocm() or current_platform.is_xpu()):
        return lambda _: None

    def install(lines: Iterable[str]) -> None:
        sc = tmp_path / "sitecustomize.py"
        sc.write_text("\n".join(lines) + "\n")
        monkeypatch.setenv(
            "PYTHONPATH",
            os.pathsep.join(filter(None, [str(tmp_path), os.getenv("PYTHONPATH")])),
        )

    return install
