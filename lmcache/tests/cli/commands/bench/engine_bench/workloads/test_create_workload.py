# SPDX-License-Identifier: Apache-2.0
"""Tests for the workload factory."""

# Standard
from unittest.mock import MagicMock
import argparse

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.config import EngineBenchConfig
from lmcache.cli.commands.bench.engine_bench.workloads import (
    BaseWorkload,
    create_workload,
    validate_max_output_length_supported,
)
from lmcache.cli.commands.bench.engine_bench.workloads.long_doc_qa import (
    LongDocQAWorkload,
)
from lmcache.cli.commands.bench.engine_bench.workloads.multi_round_chat import (
    MultiRoundChatWorkload,
)
from lmcache.cli.commands.bench.engine_bench.workloads.prefix_suffix_tuner import (
    PrefixSuffixTunerWorkload,
)
from lmcache.cli.commands.bench.engine_bench.workloads.random_prefill import (
    RandomPrefillWorkload,
)


def _make_config(**overrides) -> EngineBenchConfig:
    defaults = dict(
        engine_url="http://localhost:8000",
        model="test-model",
        workload="long-doc-qa",
        kv_cache_volume_gb=100.0,
        tokens_per_gb_kvcache=50000,
        seed=42,
        output_dir=".",
        export_csv=True,
        export_json=False,
        quiet=False,
    )
    defaults.update(overrides)
    return EngineBenchConfig(**defaults)  # type: ignore[arg-type]


def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        # long-doc-qa defaults
        ldqa_document_length=10000,
        ldqa_query_per_document=2,
        ldqa_shuffle_policy="random",
        ldqa_num_inflight_requests=3,
        ldqa_max_output_length=128,
        # random-prefill defaults
        rp_request_length=10000,
        rp_num_requests=50,
        # multi-round-chat defaults
        mrc_shared_prompt_length=2000,
        mrc_chat_history_length=10000,
        mrc_user_input_length=50,
        mrc_output_length=200,
        mrc_qps=1.0,
        mrc_duration=60.0,
        # prefix-suffix-tuner defaults
        psf_context_length=8000,
        psf_prefix_ratio=0.8,
        psf_thrash=20.0,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_deps() -> tuple[MagicMock, MagicMock, MagicMock]:
    sender = MagicMock()
    collector = MagicMock()
    monitor = MagicMock()
    return sender, collector, monitor


class TestValidateMaxOutputLengthSupported:
    @pytest.mark.parametrize("workload", ["long-doc-qa", "multi-round-chat"])
    def test_allowed_for_workloads_with_the_parameter(self, workload: str) -> None:
        validate_max_output_length_supported(workload)

    @pytest.mark.parametrize(
        "workload",
        ["long-doc-permutator", "random-prefill", "prefix-suffix-tuner"],
    )
    def test_rejected_for_workloads_without_the_parameter(self, workload: str) -> None:
        with pytest.raises(ValueError, match="max output length cannot be specified"):
            validate_max_output_length_supported(workload)


class TestCreateWorkload:
    def test_long_doc_qa(self) -> None:
        config = _make_config(workload="long-doc-qa")
        args = _make_args()
        sender, collector, monitor = _make_deps()
        result = create_workload(
            config,
            args,
            sender,
            collector,
            monitor,
        )
        assert isinstance(result, BaseWorkload)
        assert isinstance(result, LongDocQAWorkload)
        # 100 * 50000 / 10000 = 500
        assert result._config.num_documents == 500

    def test_long_doc_qa_custom_args(self) -> None:
        config = _make_config(
            workload="long-doc-qa",
            kv_cache_volume_gb=10.0,
            tokens_per_gb_kvcache=10000,
        )
        args = _make_args(
            ldqa_document_length=5000,
            ldqa_query_per_document=4,
            ldqa_shuffle_policy="tile",
            ldqa_max_output_length=1024,
        )
        sender, collector, monitor = _make_deps()
        result = create_workload(
            config,
            args,
            sender,
            collector,
            monitor,
        )
        assert isinstance(result, LongDocQAWorkload)
        assert result._config.document_length == 5000
        assert result._config.query_per_document == 4
        assert result._config.shuffle_policy == "tile"
        assert result._config.max_output_length == 1024
        assert result._config.num_documents == 20  # 10 * 10000 / 5000

    def test_multi_round_chat(self) -> None:
        config = _make_config(workload="multi-round-chat")
        args = _make_args()
        sender, collector, monitor = _make_deps()
        result = create_workload(
            config,
            args,
            sender,
            collector,
            monitor,
        )
        assert isinstance(result, BaseWorkload)
        assert isinstance(result, MultiRoundChatWorkload)
        # 100 * 50000 / (2000 + 10000) = 416
        assert result._config.num_concurrent_users == 416

    def test_multi_round_chat_custom_args(self) -> None:
        config = _make_config(
            workload="multi-round-chat",
            kv_cache_volume_gb=10.0,
            tokens_per_gb_kvcache=10000,
        )
        args = _make_args(
            mrc_shared_prompt_length=500,
            mrc_chat_history_length=5000,
            mrc_qps=5.0,
            mrc_duration=30.0,
        )
        sender, collector, monitor = _make_deps()
        result = create_workload(
            config,
            args,
            sender,
            collector,
            monitor,
        )
        assert isinstance(result, MultiRoundChatWorkload)
        assert result._config.shared_prompt_length == 500
        assert result._config.chat_history_length == 5000
        assert result._config.qps == 5.0
        assert result._config.duration == 30.0
        # 10 * 10000 / (500 + 5000) = 18
        assert result._config.num_concurrent_users == 18

    def test_random_prefill(self) -> None:
        config = _make_config(workload="random-prefill")
        args = _make_args()
        sender, collector, monitor = _make_deps()
        result = create_workload(
            config,
            args,
            sender,
            collector,
            monitor,
        )
        assert isinstance(result, BaseWorkload)
        assert isinstance(result, RandomPrefillWorkload)
        assert result._config.request_length == 10000
        assert result._config.num_requests == 50

    def test_random_prefill_custom_args(self) -> None:
        config = _make_config(workload="random-prefill")
        args = _make_args(rp_request_length=5000, rp_num_requests=20)
        sender, collector, monitor = _make_deps()
        result = create_workload(
            config,
            args,
            sender,
            collector,
            monitor,
        )
        assert isinstance(result, RandomPrefillWorkload)
        assert result._config.request_length == 5000
        assert result._config.num_requests == 20

    def test_prefix_suffix_tuner(self) -> None:
        # Defaults: psf_thrash=20.0 (target tier in GB).  The workload does
        # NOT consume kv_cache_volume_gb (its sizing is independent of the
        # general kv_cache_volume flag), so this test asserts only on the
        # psf-* args + tokens-per-gb.
        config = _make_config(
            workload="prefix-suffix-tuner",
            tokens_per_gb_kvcache=10000,
        )
        args = _make_args()
        sender, collector, monitor = _make_deps()
        result = create_workload(
            config,
            args,
            sender,
            collector,
            monitor,
        )
        assert isinstance(result, BaseWorkload)
        assert isinstance(result, PrefixSuffixTunerWorkload)
        # context=8000, ratio=0.8 → prefix=6400; suffix=8000-6400-32=1568
        assert result._config.prefix_tokens == 6400
        assert result._config.suffix_tokens == 1568
        # thrash=20.0 GB; pool_gb=20.0 * 1.05 = 21.0 GB
        # Sized by context_length=8000 (full per-request KV footprint):
        # num_prefixes = 21.0 * 10000 / 8000 = 26.25 → 26
        assert result._config.num_prefixes == 26

    def test_prefix_suffix_tuner_custom_args(self) -> None:
        config = _make_config(
            workload="prefix-suffix-tuner",
            tokens_per_gb_kvcache=10000,
        )
        args = _make_args(
            psf_context_length=4000,
            psf_prefix_ratio=0.5,
            psf_thrash=50.0,
        )
        sender, collector, monitor = _make_deps()
        result = create_workload(
            config,
            args,
            sender,
            collector,
            monitor,
        )
        assert isinstance(result, PrefixSuffixTunerWorkload)
        assert result._config.context_length == 4000
        assert result._config.prefix_ratio == 0.5
        assert result._config.thrash == 50.0
        assert result._config.prefix_tokens == 2000
        # thrash=50 GB; pool_gb=50 * 1.05 = 52.5 GB
        # Sized by context_length=4000:
        # num_prefixes = 52.5 * 10000 / 4000 = 131.25 → 131
        assert result._config.num_prefixes == 131

    def test_unknown_workload_raises(self) -> None:
        config = _make_config(workload="unknown-workload")
        args = _make_args()
        sender, collector, monitor = _make_deps()
        with pytest.raises(ValueError, match="Unknown workload"):
            create_workload(
                config,
                args,
                sender,
                collector,
                monitor,
            )
