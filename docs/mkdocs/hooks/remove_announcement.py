# SPDX-License-Identifier: Apache-2.0
import os
from typing import Literal


def on_startup(command: Literal["build", "gh-deploy", "serve"], dirty: bool):
    # see https://docs.readthedocs.io/en/stable/reference/environment-variables.html # noqa
    if os.getenv('READTHEDOCS_VERSION_TYPE') == "tag":
        # remove the warning banner if the version is a tagged release
        docs_dir = os.path.dirname(__file__)
        announcement_path = os.path.join(docs_dir,
                                         "mkdocs/overrides/main.html")
        # The file might be removed already if the build is triggered multiple
        # times (readthedocs build both HTML and PDF versions separately)
        if os.path.exists(announcement_path):
            os.remove(announcement_path)
