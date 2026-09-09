# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""moe_expert_pool_rows: config field, CLI -> EngineArgs, hash."""

import pytest
from pydantic import ValidationError

from vllm.config.offload import OffloadConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


def test_default_is_off():
    assert OffloadConfig().moe_expert_pool_rows == 0


def test_hash_distinguishes_the_pool_size():
    assert (
        OffloadConfig().compute_hash()
        != OffloadConfig(moe_expert_pool_rows=8).compute_hash()
    )


def test_negative_rows_are_rejected():
    with pytest.raises(ValidationError):
        OffloadConfig(moe_expert_pool_rows=-1)


def test_cli_reaches_engine_args_and_offload_config():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(["--moe-expert-pool-rows", "16"])
    engine_args = EngineArgs.from_cli_args(args)
    assert engine_args.moe_expert_pool_rows == 16
    offload = OffloadConfig(moe_expert_pool_rows=engine_args.moe_expert_pool_rows)
    assert offload.moe_expert_pool_rows == 16
