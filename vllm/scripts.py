# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.foundation.observability.logger import init_logger
from vllm.frontend.entrypoints.cli.main import main as vllm_main

logger = init_logger(__name__)


# Backwards compatibility for the move from vllm.scripts to
# vllm.frontend.entrypoints.cli.main
def main():
    logger.warning(
        "vllm.scripts.main() is deprecated. Please re-install "
        "vllm or use vllm.frontend.entrypoints.cli.main.main() instead."
    )
    vllm_main()
