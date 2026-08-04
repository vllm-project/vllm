# SPDX-License-Identifier: Apache-2.0
"""Tests for bench engine config module."""

# Standard
from unittest.mock import MagicMock, patch
import argparse

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.engine_bench.config import (
    EngineBenchConfig,
    _find_model_meta,
    auto_detect_model,
    parse_args_to_config,
    resolve_tokens_per_gb,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_namespace() -> argparse.Namespace:
    """Namespace with all required fields for parse_args_to_config."""
    return argparse.Namespace(
        engine_url="http://localhost:8000",
        lmcache_url=None,
        model="test-model",
        workload="long-doc-qa",
        kv_cache_volume=100.0,
        tokens_per_gb_kvcache=50000,
        seed=42,
        output_dir=".",
        no_csv=False,
        json=False,
        quiet=False,
        ignore_eos=False,
    )


# ---------------------------------------------------------------------------
# EngineBenchConfig
# ---------------------------------------------------------------------------


class TestEngineBenchConfig:
    def _make_config(self, **overrides) -> EngineBenchConfig:
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

    def test_valid_construction(self) -> None:
        cfg = self._make_config()
        assert cfg.engine_url == "http://localhost:8000"
        assert cfg.model == "test-model"
        assert cfg.workload == "long-doc-qa"
        assert cfg.tokens_per_gb_kvcache == 50000

    def test_ignore_eos_defaults_false_and_overridable(self) -> None:
        assert self._make_config().ignore_eos is False
        assert self._make_config(ignore_eos=True).ignore_eos is True

    def test_empty_engine_url(self) -> None:
        with pytest.raises(ValueError, match="engine_url must be non-empty"):
            self._make_config(engine_url="")

    def test_invalid_kv_cache_volume(self) -> None:
        with pytest.raises(ValueError, match="kv_cache_volume_gb must be positive"):
            self._make_config(kv_cache_volume_gb=0)

    def test_invalid_tokens_per_gb(self) -> None:
        with pytest.raises(ValueError, match="tokens_per_gb_kvcache must be positive"):
            self._make_config(tokens_per_gb_kvcache=0)


# ---------------------------------------------------------------------------
# auto_detect_model
# ---------------------------------------------------------------------------


class TestAutoDetectModel:
    @patch(
        "lmcache.cli.commands.bench.engine_bench.config.OpenAI",
    )
    def test_returns_first_model_id(self, mock_openai_cls) -> None:
        mock_model = MagicMock()
        mock_model.id = "Qwen/Qwen3-14B"
        mock_client = MagicMock()
        mock_client.models.list.return_value = MagicMock(data=[mock_model])
        mock_openai_cls.return_value = mock_client

        result = auto_detect_model("http://localhost:8000")
        assert result == "Qwen/Qwen3-14B"

    @patch(
        "lmcache.cli.commands.bench.engine_bench.config.OpenAI",
    )
    def test_empty_data_raises(self, mock_openai_cls) -> None:
        mock_client = MagicMock()
        mock_client.models.list.return_value = MagicMock(data=[])
        mock_openai_cls.return_value = mock_client

        with pytest.raises(RuntimeError, match="No models returned"):
            auto_detect_model("http://localhost:8000")

    @patch(
        "lmcache.cli.commands.bench.engine_bench.config.OpenAI",
    )
    def test_connection_error_raises(self, mock_openai_cls) -> None:
        mock_openai_cls.side_effect = ConnectionError("refused")

        with pytest.raises(RuntimeError, match="Failed to fetch models"):
            auto_detect_model("http://localhost:8000")


# ---------------------------------------------------------------------------
# parse_args_to_config
# ---------------------------------------------------------------------------


class TestParseArgsToConfig:
    def test_basic_parsing(self, base_namespace) -> None:
        cfg = parse_args_to_config(base_namespace)
        assert cfg.engine_url == "http://localhost:8000"
        assert cfg.model == "test-model"
        assert cfg.workload == "long-doc-qa"
        assert cfg.kv_cache_volume_gb == 100.0
        assert cfg.seed == 42
        assert cfg.export_csv is True
        assert cfg.export_json is False

    @patch(
        "lmcache.cli.commands.bench.engine_bench.config.auto_detect_model",
        return_value="auto-detected-model",
    )
    def test_model_auto_detect(self, mock_auto_detect, base_namespace) -> None:
        base_namespace.model = None
        cfg = parse_args_to_config(base_namespace)
        assert cfg.model == "auto-detected-model"
        mock_auto_detect.assert_called_once_with("http://localhost:8000")

    def test_no_tokens_per_gb_no_lmcache_url_raises(
        self,
        base_namespace,
    ) -> None:
        base_namespace.tokens_per_gb_kvcache = None
        base_namespace.lmcache_url = None
        with pytest.raises(ValueError, match="--tokens-per-gb-kvcache"):
            parse_args_to_config(base_namespace)

    @patch(
        "lmcache.cli.commands.bench.engine_bench.config.resolve_tokens_per_gb",
        return_value=6553,
    )
    def test_lmcache_url_resolves_tokens_per_gb(
        self,
        mock_resolve,
        base_namespace,
    ) -> None:
        base_namespace.tokens_per_gb_kvcache = None
        base_namespace.lmcache_url = "http://localhost:8080"
        cfg = parse_args_to_config(base_namespace)
        assert cfg.tokens_per_gb_kvcache == 6553
        mock_resolve.assert_called_once_with(
            "http://localhost:8080",
            "test-model",
        )

    def test_export_flags(self, base_namespace) -> None:
        base_namespace.no_csv = True
        base_namespace.json = True
        cfg = parse_args_to_config(base_namespace)
        assert cfg.export_csv is False
        assert cfg.export_json is True


# ---------------------------------------------------------------------------
# _find_model_meta
# ---------------------------------------------------------------------------


class TestFindModelMeta:
    def _gpu_meta(self) -> dict:
        return {
            "gpu_0": {
                "model_name": "Qwen/Qwen3-14B",
                "world_size": 1,
                "kv_cache_layout": {"cache_size_per_token": 163840},
            },
            "gpu_1": {
                "model_name": "meta-llama/Llama-3.1-70B",
                "world_size": 4,
                "kv_cache_layout": {"cache_size_per_token": 327680},
            },
        }

    def test_finds_matching_model(self) -> None:
        meta = _find_model_meta(self._gpu_meta(), "Qwen/Qwen3-14B")
        assert meta["model_name"] == "Qwen/Qwen3-14B"

    def test_finds_second_model(self) -> None:
        meta = _find_model_meta(
            self._gpu_meta(),
            "meta-llama/Llama-3.1-70B",
        )
        assert meta["world_size"] == 4

    def test_missing_model_raises(self) -> None:
        with pytest.raises(RuntimeError, match="not found"):
            _find_model_meta(self._gpu_meta(), "nonexistent-model")

    def test_error_lists_available(self) -> None:
        with pytest.raises(RuntimeError, match="Qwen/Qwen3-14B"):
            _find_model_meta(self._gpu_meta(), "nonexistent")


# ---------------------------------------------------------------------------
# resolve_tokens_per_gb
# ---------------------------------------------------------------------------


class TestResolveTokensPerGb:
    def _status_response(
        self,
        cache_size_per_token: int = 163840,
        world_size: int = 1,
        model_name: str = "Qwen/Qwen3-14B",
    ) -> dict:
        return {
            "cache_context_meta": {
                "gpu_0": {
                    "model_name": model_name,
                    "world_size": world_size,
                    "kv_cache_layout": {
                        "num_layers": 40,
                        "hidden_dim_size": 1024,
                        "dtype": "torch.bfloat16",
                        "cache_size_per_token": cache_size_per_token,
                    },
                },
            },
        }

    @patch(
        "lmcache.cli.commands.bench.engine_bench.config._fetch_lmcache_status",
    )
    def test_basic_calculation(self, mock_fetch) -> None:
        # 163840 bytes/token, world_size=1
        # 1 GB = 1073741824 bytes
        # 1073741824 // 163840 = 6553
        mock_fetch.return_value = self._status_response(
            cache_size_per_token=163840,
            world_size=1,
        )
        result = resolve_tokens_per_gb(
            "http://localhost:8080",
            "Qwen/Qwen3-14B",
        )
        assert result == 6553

    @patch(
        "lmcache.cli.commands.bench.engine_bench.config._fetch_lmcache_status",
    )
    def test_world_size_multiplier(self, mock_fetch) -> None:
        # 163840 bytes/token (rank-local), world_size=4
        # global = 163840 * 4 = 655360
        # 1073741824 // 655360 = 1638
        mock_fetch.return_value = self._status_response(
            cache_size_per_token=163840,
            world_size=4,
        )
        result = resolve_tokens_per_gb(
            "http://localhost:8080",
            "Qwen/Qwen3-14B",
        )
        assert result == 1638

    @patch(
        "lmcache.cli.commands.bench.engine_bench.config._fetch_lmcache_status",
    )
    def test_no_gpu_meta_raises(self, mock_fetch) -> None:
        mock_fetch.return_value = {"cache_context_meta": {}}
        with pytest.raises(RuntimeError, match="No model info"):
            resolve_tokens_per_gb(
                "http://localhost:8080",
                "Qwen/Qwen3-14B",
            )

    @patch(
        "lmcache.cli.commands.bench.engine_bench.config._fetch_lmcache_status",
    )
    def test_model_not_found_raises(self, mock_fetch) -> None:
        mock_fetch.return_value = self._status_response()
        with pytest.raises(RuntimeError, match="not found"):
            resolve_tokens_per_gb(
                "http://localhost:8080",
                "nonexistent-model",
            )

    @patch(
        "lmcache.cli.commands.bench.engine_bench.config._fetch_lmcache_status",
    )
    def test_no_cache_size_raises(self, mock_fetch) -> None:
        data = self._status_response()
        del data["cache_context_meta"]["gpu_0"]["kv_cache_layout"][
            "cache_size_per_token"
        ]
        mock_fetch.return_value = data
        with pytest.raises(RuntimeError, match="cache_size_per_token"):
            resolve_tokens_per_gb(
                "http://localhost:8080",
                "Qwen/Qwen3-14B",
            )

    @patch(
        "lmcache.cli.commands.bench.engine_bench.config._fetch_lmcache_status",
    )
    def test_no_layout_raises(self, mock_fetch) -> None:
        data = self._status_response()
        del data["cache_context_meta"]["gpu_0"]["kv_cache_layout"]
        mock_fetch.return_value = data
        with pytest.raises(RuntimeError, match="kv_cache_layout"):
            resolve_tokens_per_gb(
                "http://localhost:8080",
                "Qwen/Qwen3-14B",
            )
