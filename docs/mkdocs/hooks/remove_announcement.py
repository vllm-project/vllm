# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from pathlib import Path
from typing import Literal


def on_startup(command: Literal["build", "gh-deploy", "serve"], dirty: bool):
    # see https://docs.readthedocs.io/en/stable/reference/environment-variables.html # noqa
    if os.getenv('READTHEDOCS_VERSION_TYPE') == "tag":
        # remove the warning banner if the version is a tagged release
        mkdocs_dir = Path(__file__).parent.parent
        announcement_path = mkdocs_dir / "overrides/main.html"
        # The file might be removed already if the build is triggered multiple
        # times (readthedocs build both HTML and PDF versions separately)
        if announcement_path.exists():
            os.remove(announcement_path)
