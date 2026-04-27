# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import ArgumentError

import pytest

from vllm.config import ModelConfig, MoEOffloadConfig, VllmConfig, replace
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.engine import moe_offload_cli
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.hashing import _xxhash


def test_prefix_caching_from_cli():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args([])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.enable_prefix_caching, (
        "V1 turns on prefix caching by default."
    )

    # Turn it off possible with flag.
    args = parser.parse_args(["--no-enable-prefix-caching"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert not vllm_config.cache_config.enable_prefix_caching

    # Turn it on with flag.
    args = parser.parse_args(["--enable-prefix-caching"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.enable_prefix_caching

    # default hash algorithm is "builtin"
    assert vllm_config.cache_config.prefix_caching_hash_algo == "sha256"

    # set hash algorithm to sha256_cbor
    args = parser.parse_args(["--prefix-caching-hash-algo", "sha256_cbor"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_caching_hash_algo == "sha256_cbor"

    # set hash algorithm to sha256
    args = parser.parse_args(["--prefix-caching-hash-algo", "sha256"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_caching_hash_algo == "sha256"

    # an invalid hash algorithm raises an error
    parser.exit_on_error = False
    with pytest.raises(ArgumentError):
        args = parser.parse_args(["--prefix-caching-hash-algo", "invalid"])


@pytest.mark.skipif(_xxhash is None, reason="xxhash not installed")
def test_prefix_caching_xxhash_from_cli():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())

    # set hash algorithm to xxhash (pickle)
    args = parser.parse_args(["--prefix-caching-hash-algo", "xxhash"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_caching_hash_algo == "xxhash"

    # set hash algorithm to xxhash_cbor
    args = parser.parse_args(["--prefix-caching-hash-algo", "xxhash_cbor"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_caching_hash_algo == "xxhash_cbor"


def test_moe_cpu_offload_flags_visible_and_defaulted():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert "--moe-cpu-offload" in option_strings
    assert "--moe-gpu-limit" not in option_strings
    assert "--moe-active-expert-budget" not in option_strings
    assert "--moe-active-expert-cache" not in option_strings
    assert "--moe-max-pipeline-depth" not in option_strings

    args = parser.parse_args([])
    engine_args = EngineArgs.from_cli_args(args=args)
    assert engine_args.moe_cpu_offload is False

    config = engine_args.create_engine_config()
    assert isinstance(config.moe_offload_config, MoEOffloadConfig)
    assert config.moe_offload_config.enabled is False


def test_moe_cpu_offload_flag_parses():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(["--moe-cpu-offload"])
    engine_args = EngineArgs.from_cli_args(args=args)
    config = engine_args.create_engine_config()

    assert engine_args.moe_cpu_offload is True
    assert config.moe_offload_config.enabled is False


def test_moe_cpu_offload_ignores_dense_model(monkeypatch):
    log_messages = []
    monkeypatch.setattr(
        moe_offload_cli.logger,
        "info",
        lambda message, *args: log_messages.append(message % args if args else message),
    )

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(["--model", "facebook/opt-125m", "--moe-cpu-offload"])
    engine_args = EngineArgs.from_cli_args(args=args)
    config = engine_args.create_engine_config()

    assert config.model_config.is_moe is False
    assert config.model_config.enforce_eager is False
    assert config.moe_offload_config.enabled is False
    assert log_messages == [
        "MoE CPU offload ignored: --moe-cpu-offload was set, "
        "but the model is not a MoE model."
    ]


def test_moe_cpu_offload_enables_for_moe_model(monkeypatch):
    monkeypatch.setattr(ModelConfig, "is_moe", property(lambda self: True))
    monkeypatch.setattr(moe_offload_cli, "_get_active_expert_count", lambda _: 8)
    log_messages = []
    monkeypatch.setattr(
        moe_offload_cli.logger,
        "info",
        lambda message, *args: log_messages.append(message % args if args else message),
    )

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(["--model", "facebook/opt-125m", "--moe-cpu-offload"])
    engine_args = EngineArgs.from_cli_args(args=args)
    config = engine_args.create_engine_config()

    assert config.model_config.enforce_eager is True
    assert config.moe_offload_config.enabled is True
    assert log_messages == [
        "MoE CPU offload enabled: total experts=0, active experts=8, "
        "active expert transfer=passive."
    ]


def test_moe_cpu_offload_cli_preserves_async_engine_args():
    parser = AsyncEngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(["--enable-log-requests"])

    engine_args = AsyncEngineArgs.from_cli_args(args=args)

    assert isinstance(engine_args, AsyncEngineArgs)
    assert engine_args.enable_log_requests is True
    assert engine_args.moe_cpu_offload is False


def test_moe_offload_config_survives_vllm_config_replace():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(["--moe-cpu-offload"])
    config = EngineArgs.from_cli_args(args=args).create_engine_config()

    replaced = replace(config, performance_mode="throughput")

    assert replaced.performance_mode == "throughput"
    assert replaced.moe_offload_config.enabled is False


def test_defaults_with_usage_context():
    engine_args = EngineArgs(model="facebook/opt-125m")
    vllm_config: VllmConfig = engine_args.create_engine_config(UsageContext.LLM_CLASS)

    from vllm.platforms import current_platform
    from vllm.utils.mem_constants import GiB_bytes

    device_memory = current_platform.get_device_total_memory()
    device_name = current_platform.get_device_name().lower()
    if device_memory >= 70 * GiB_bytes and "a100" not in device_name:
        # For GPUs like H100, H200, and MI300x with >= 70GB memory
        default_llm_tokens = 16384
        default_server_tokens = 8192
        default_max_num_seqs = 1024
    else:
        default_llm_tokens = 8192
        default_server_tokens = 2048
        default_max_num_seqs = 256

    assert vllm_config.scheduler_config.max_num_seqs == default_max_num_seqs
    assert vllm_config.scheduler_config.max_num_batched_tokens == default_llm_tokens  # noqa: E501

    engine_args = EngineArgs(model="facebook/opt-125m")
    vllm_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)
    assert vllm_config.scheduler_config.max_num_seqs == default_max_num_seqs
    assert vllm_config.scheduler_config.max_num_batched_tokens == default_server_tokens  # noqa: E501
