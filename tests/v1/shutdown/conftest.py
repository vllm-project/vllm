# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Iterable
from pathlib import Path

import pytest

from vllm.platforms import current_platform


@pytest.fixture
def rocm_sitecustomize_factory(monkeypatch, tmp_path: Path):
    """Return a function that installs a given sitecustomize payload."""
    if not current_platform.is_rocm():
        return lambda _: None

    def install(lines: Iterable[str]) -> None:
        sc = tmp_path / "sitecustomize.py"
        sc.write_text("\n".join(lines) + "\n")
        monkeypatch.setenv(
            "PYTHONPATH",
            os.pathsep.join(filter(None, [str(tmp_path), os.getenv("PYTHONPATH")])),
        )

    return install
