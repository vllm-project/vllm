# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os


def cli_env_setup() -> None:
    """Set process defaults shared by all CLI runtime commands."""
    if "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ:
        logging.getLogger(__name__).debug(
            "Setting VLLM_WORKER_MULTIPROC_METHOD to 'spawn'"
        )
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
