# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio

import pytest

from vllm.benchmarks.serve import add_cli_args, main_async
from vllm.utils.argparse_utils import FlexibleArgumentParser


@pytest.mark.parametrize("strategy", ["linear", "exponential"])
def test_zero_ramp_up_start_rps_is_rejected(strategy: str) -> None:
    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args(
        [
            "--ramp-up-strategy",
            strategy,
            "--ramp-up-start-rps",
            "0",
            "--ramp-up-end-rps",
            "10",
        ]
    )

    with pytest.raises(ValueError, match="^--ramp-up-start-rps must be > 0$"):
        asyncio.run(main_async(args))
