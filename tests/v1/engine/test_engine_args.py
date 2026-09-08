# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import ArgumentError

import pytest

from vllm.config import ModelConfig, SpeculativeConfig
from vllm.engine.arg_utils import EngineArgs
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
    assert vllm_config.cache_config.prefix_cache_retention_interval == 0

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

    args = parser.parse_args(["--prefix-cache-retention-interval", "64"])
    vllm_config = EngineArgs.from_cli_args(args=args).create_engine_config()
    assert vllm_config.cache_config.prefix_cache_retention_interval == 64


@pytest.mark.parametrize(
    ("is_hybrid", "has_inner_state", "use_eagle", "explicit", "expected"),
    [
        pytest.param(True, False, True, "unset", None, id="hybrid-eagle-dense"),
        pytest.param(True, True, True, "unset", None, id="hybrid-with-inner-state"),
        pytest.param(True, False, True, 0, 0, id="hybrid-eagle-explicit-zero"),
        pytest.param(True, False, True, None, None, id="hybrid-eagle-explicit-none"),
        pytest.param(True, False, True, 64, 64, id="hybrid-eagle-explicit-interval"),
        pytest.param(True, False, False, "unset", 0, id="hybrid-without-eagle"),
        pytest.param(False, True, True, "unset", 0, id="non-hybrid-inner-state"),
        pytest.param(False, False, True, "unset", 0, id="eagle-without-hybrid"),
        pytest.param(False, False, False, "unset", 0, id="plain-model"),
    ],
)
def test_prefix_cache_retention_interval_default_resolution(
    monkeypatch, is_hybrid, has_inner_state, use_eagle, explicit, expected
):
    """Default to dense for hybrid + EAGLE.

    Non-hybrid defaults and explicit retention values must remain unchanged.
    """
    monkeypatch.setattr(ModelConfig, "is_hybrid", property(lambda self: is_hybrid))
    monkeypatch.setattr(
        ModelConfig, "has_inner_state", property(lambda self: has_inner_state)
    )
    if use_eagle:
        spec_config = SpeculativeConfig(model="ngram", num_speculative_tokens=1)
        spec_config.method = "eagle"
        monkeypatch.setattr(
            EngineArgs,
            "create_speculative_config",
            lambda self, **kwargs: spec_config,
        )
    engine_kwargs = (
        {} if explicit == "unset" else {"prefix_cache_retention_interval": explicit}
    )
    vllm_config = EngineArgs(
        model="Qwen/Qwen3-0.6B", **engine_kwargs
    ).create_engine_config()
    assert vllm_config.cache_config.prefix_cache_retention_interval == expected


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


def test_mm_prefix_lm_raises_batched_tokens_floor():
    """Verify that prefix-LM multimodal models auto-raise
    max_num_batched_tokens to fit at least one multimodal item.

    Regression test for https://github.com/vllm-project/vllm/issues/42687
    """
    from unittest.mock import patch

    # Simulate a prefix-LM multimodal model whose largest modality
    # (video) requires 2496 tokens — more than the 2048 default.
    fake_mm_min = (2496, "video")

    engine_args = EngineArgs(
        model="facebook/opt-125m",
        max_model_len=2048,
        enforce_eager=True,
    )

    with (
        patch.object(
            type(engine_args),
            "_get_min_mm_batched_tokens",
            staticmethod(lambda _mc: fake_mm_min),
        ),
        patch(
            "vllm.config.ModelConfig.is_multimodal_model",
            new_callable=lambda: property(lambda self: True),
        ),
        patch(
            "vllm.config.ModelConfig.is_mm_prefix_lm",
            new_callable=lambda: property(lambda self: True),
        ),
    ):
        vllm_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)

    assert vllm_config.scheduler_config.max_num_batched_tokens >= 2496


def test_data_parallel_start_rank_zero_infers_hybrid_lb():
    """An explicit --data-parallel-start-rank 0 must be treated the same as
    any other explicit start rank when inferring hybrid LB mode, not as
    "unset" (regression test for a truthiness-vs-`is not None` bug).
    """
    engine_args = EngineArgs(
        model="facebook/opt-125m",
        data_parallel_size=4,
        data_parallel_size_local=2,
        data_parallel_start_rank=0,
    )
    vllm_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)

    assert vllm_config.parallel_config.data_parallel_hybrid_lb is True
    assert vllm_config.parallel_config.data_parallel_rank == 0
