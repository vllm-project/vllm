# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import ArgumentError
from unittest.mock import patch

import pytest

from vllm.config import ModelConfig
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


def _mock_eagle_spec_config(monkeypatch):
    from unittest.mock import MagicMock

    from vllm.config import SpeculativeConfig

    try:
        spec_config = SpeculativeConfig(model="ngram", num_speculative_tokens=1)
        spec_config.method = "eagle"
    except (ValueError, TypeError):
        spec_config = MagicMock()
        spec_config.use_eagle.return_value = True
    monkeypatch.setattr(
        EngineArgs, "create_speculative_config", lambda self, **kwargs: spec_config
    )


@pytest.mark.parametrize(
    ("is_hybrid", "use_eagle", "explicit", "expected"),
    [
        pytest.param(True, True, "unset", "k_block", id="hybrid-eagle-unset"),
        pytest.param(True, True, 0, 0, id="hybrid-eagle-explicit-zero"),
        pytest.param(True, True, None, None, id="hybrid-eagle-explicit-none"),
        pytest.param(True, True, 64, 64, id="hybrid-eagle-explicit-interval"),
        pytest.param(True, False, "unset", 0, id="hybrid-without-eagle"),
        pytest.param(False, True, "unset", 0, id="eagle-without-hybrid"),
        pytest.param(False, False, "unset", 0, id="plain-model"),
    ],
)
def test_prefix_cache_retention_interval_default_resolution(
    monkeypatch, is_hybrid, use_eagle, explicit, expected
):
    """Default hybrid+EAGLE retention to k * scheduler block_size.

    Explicit 0/None/N and non-hybrid / non-EAGLE unset stay unchanged.
    """
    from vllm.config.cache import (
        HYBRID_EAGLE_PREFIX_CACHE_RETENTION_BLOCKS,
        CacheConfig,
    )

    monkeypatch.setattr(ModelConfig, "is_hybrid", property(lambda self: is_hybrid))
    if use_eagle:
        _mock_eagle_spec_config(monkeypatch)
    engine_kwargs = (
        {} if explicit == "unset" else {"prefix_cache_retention_interval": explicit}
    )
    vllm_config = EngineArgs(
        model="facebook/opt-125m", **engine_kwargs
    ).create_engine_config()
    if expected == "k_block":
        # Applied after create_speculative_config using DEFAULT_BLOCK_SIZE
        # unless --block-size was set. Some platforms (CPU) then bump
        # block_size in check_and_update_config without re-applying.
        applied_block_size = (
            vllm_config.cache_config.block_size
            if vllm_config.cache_config.user_specified_block_size
            else CacheConfig.DEFAULT_BLOCK_SIZE
        )
        expected = HYBRID_EAGLE_PREFIX_CACHE_RETENTION_BLOCKS * applied_block_size
    assert vllm_config.cache_config.prefix_cache_retention_interval == expected


def test_prefix_cache_retention_interval_hybrid_eagle_uses_block_size(monkeypatch):
    from vllm.config.cache import HYBRID_EAGLE_PREFIX_CACHE_RETENTION_BLOCKS

    monkeypatch.setattr(ModelConfig, "is_hybrid", property(lambda self: True))
    _mock_eagle_spec_config(monkeypatch)
    vllm_config = EngineArgs(
        model="facebook/opt-125m",
        block_size=32,
    ).create_engine_config()
    assert (
        vllm_config.cache_config.prefix_cache_retention_interval
        == HYBRID_EAGLE_PREFIX_CACHE_RETENTION_BLOCKS * 32
    )


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


def test_external_lb_preserves_explicit_rank_when_dp_exceeds_nodes():
    engine_args = EngineArgs(
        model="facebook/opt-125m",
        data_parallel_size=4,
        data_parallel_rank=3,
        data_parallel_external_lb=True,
        nnodes=2,
        node_rank=1,
    )

    with patch.object(ModelConfig, "is_moe", new=property(lambda self: True)):
        vllm_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)

    assert vllm_config.parallel_config.data_parallel_rank == 3


@pytest.mark.parametrize(
    ("data_parallel_size", "nnodes", "tensor_parallel_size"),
    [(4, 2, 1), (2, 3, 3)],
)
def test_external_lb_requires_explicit_rank_when_nodes_are_not_evenly_partitioned(
    data_parallel_size, nnodes, tensor_parallel_size
):
    engine_args = EngineArgs(
        model="facebook/opt-125m",
        data_parallel_size=data_parallel_size,
        data_parallel_external_lb=True,
        tensor_parallel_size=tensor_parallel_size,
        nnodes=nnodes,
        node_rank=1,
    )

    with (
        patch.object(ModelConfig, "is_moe", new=property(lambda self: True)),
        pytest.raises(
            ValueError,
            match="Set a unique `--data-parallel-rank`",
        ),
    ):
        engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)


def test_external_lb_infers_rank_when_dp_does_not_exceed_nodes():
    engine_args = EngineArgs(
        model="facebook/opt-125m",
        data_parallel_size=2,
        data_parallel_external_lb=True,
        nnodes=2,
        node_rank=1,
    )

    with patch.object(ModelConfig, "is_moe", new=property(lambda self: True)):
        vllm_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)

    assert vllm_config.parallel_config.data_parallel_rank == 1


def test_external_lb_infers_rank_for_multinode_replicas():
    engine_args = EngineArgs(
        model="facebook/opt-125m",
        data_parallel_size=2,
        data_parallel_external_lb=True,
        tensor_parallel_size=12,
        nnodes=4,
        node_rank=1,
    )

    with patch.object(ModelConfig, "is_moe", new=property(lambda self: True)):
        vllm_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)

    assert vllm_config.parallel_config.data_parallel_rank == 0
