# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


def test_encoder_budget_defaults_to_max_num_batched_tokens():
    """When not set, both encoder fields default to max_num_batched_tokens."""
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args([])
    engine_args = EngineArgs.from_cli_args(args)
    vllm_config = engine_args.create_engine_config()
    sc = vllm_config.scheduler_config
    assert sc.max_num_encoder_input_tokens == sc.max_num_batched_tokens
    assert sc.encoder_cache_size == sc.max_num_batched_tokens


@pytest.mark.parametrize(
    ("encoder_input_tokens", "cache_size"),
    [
        (100000, 200000),
        (500000, 600000),
    ],
)
def test_encoder_budget_cli_args_and_config(encoder_input_tokens, cache_size):
    """CLI args propagate to EngineArgs and flow through to SchedulerConfig."""
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(
        [
            "--max-num-encoder-input-tokens",
            str(encoder_input_tokens),
            "--encoder-cache-size",
            str(cache_size),
        ]
    )
    engine_args = EngineArgs.from_cli_args(args)
    assert engine_args.max_num_encoder_input_tokens == encoder_input_tokens
    assert engine_args.encoder_cache_size == cache_size

    vllm_config = engine_args.create_engine_config()
    sc = vllm_config.scheduler_config
    assert sc.max_num_encoder_input_tokens == encoder_input_tokens
    assert sc.encoder_cache_size == cache_size


def test_encoder_budget_partial_override():
    """Setting only one field leaves the other at default."""
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(["--max-num-encoder-input-tokens", "100000"])
    engine_args = EngineArgs.from_cli_args(args)
    vllm_config = engine_args.create_engine_config()
    sc = vllm_config.scheduler_config
    assert sc.max_num_encoder_input_tokens == 100000
    assert sc.encoder_cache_size == sc.max_num_batched_tokens


def test_encoder_budget_not_in_default_args():
    """Default CLI parse produces None for both encoder fields."""
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args([])
    engine_args = EngineArgs.from_cli_args(args)
    assert engine_args.max_num_encoder_input_tokens is None
    assert engine_args.encoder_cache_size is None
