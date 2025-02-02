# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm import envs
from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser

if not envs.VLLM_USE_V1:
    pytest.skip(
        "Skipping V1 tests. Rerun with `VLLM_USE_V1=1` to test.",
        allow_module_level=True,
    )


def test_prefix_caching_from_cli():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args([])
    engine_args = EngineArgs.from_cli_args(args=args)
    assert (engine_args.enable_prefix_caching
            ), "V1 turns on prefix caching by default."

    # Turn it off possible with flag.
    args = parser.parse_args(["--no-enable-prefix-caching"])
    engine_args = EngineArgs.from_cli_args(args=args)
    assert not engine_args.enable_prefix_caching

    # Turn it on with flag.
    args = parser.parse_args(["--enable-prefix-caching"])
    engine_args = EngineArgs.from_cli_args(args=args)
    assert engine_args.enable_prefix_caching


def test_defaults_with_usage_context():
    engine_args = EngineArgs(model="facebook/opt-125m")
    vllm_config: VllmConfig = engine_args.create_engine_config(
        UsageContext.LLM_CLASS)

    assert vllm_config.scheduler_config.max_num_seqs == 1024
    assert vllm_config.scheduler_config.max_num_batched_tokens == 8192

    engine_args = EngineArgs(model="facebook/opt-125m")
    vllm_config = engine_args.create_engine_config(
        UsageContext.OPENAI_API_SERVER)
    assert vllm_config.scheduler_config.max_num_seqs == 1024
    assert vllm_config.scheduler_config.max_num_batched_tokens == 2048
