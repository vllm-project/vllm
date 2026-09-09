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


def test_cli_flag_and_field():
    # Checks the CLI flag and the OffloadConfig field only. EngineArgs
    # construction and create_engine_config resolve a model (network or a
    # local snapshot), so they are out of scope here.
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(["--moe-expert-pool-rows", "16"])
    assert args.moe_expert_pool_rows == 16
    assert EngineArgs.moe_expert_pool_rows == 0
    assert (
        OffloadConfig(
            moe_expert_pool_rows=args.moe_expert_pool_rows
        ).moe_expert_pool_rows
        == 16
    )
