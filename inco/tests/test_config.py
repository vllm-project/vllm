# SPDX-License-Identifier: Apache-2.0
"""Config is the single source of truth for both sides of the benchmark, so
parsing and derived values (request counts, artifact paths) must be exact."""

import pytest
from bench.config import (
    DEFAULT_CONCURRENCIES,
    RunManifest,
    SweepConfig,
    Workload,
    _as_bool,
    _as_int_tuple,
    _coerce,
)


class TestPrimitiveParsing:
    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", " yes ", "on"])
    def test_truthy(self, raw):
        assert _as_bool(raw) is True

    @pytest.mark.parametrize("raw", ["0", "false", "No", "off"])
    def test_falsy(self, raw):
        assert _as_bool(raw) is False

    def test_bad_bool(self):
        with pytest.raises(ValueError, match="cannot parse"):
            _as_bool("maybe")

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1,2,4", (1, 2, 4)),
            ("1 2 4", (1, 2, 4)),
            (" 8 , 16 ", (8, 16)),
            ("3", (3,)),
        ],
    )
    def test_int_tuple(self, raw, expected):
        assert _as_int_tuple(raw) == expected

    def test_empty_int_tuple(self):
        with pytest.raises(ValueError, match="at least one"):
            _as_int_tuple("  ")

    def test_optional_none(self):
        assert _coerce("none", "float | None") is None
        assert _coerce("2.5", "float | None") == 2.5

    def test_unsupported_annotation(self):
        with pytest.raises(TypeError, match="unsupported config annotation"):
            _coerce("x", "dict[str, int]")


class TestWorkload:
    def test_tokenizer_defaults_to_model(self):
        assert Workload(model="Qwen/Qwen3-4B").tokenizer_id == "Qwen/Qwen3-4B"

    def test_explicit_tokenizer_wins(self):
        assert Workload(tokenizer="Qwen/Qwen3-4B").tokenizer_id == "Qwen/Qwen3-4B"

    def test_slug_is_filesystem_safe(self):
        slug = Workload(
            model="Qwen/Qwen3-30B-A3B-Instruct-2507", isl=512, osl=64, num_gpus=2
        ).slug
        assert slug == "Qwen_Qwen3-30B-A3B-Instruct-2507-isl512-osl64-tp2"
        assert "/" not in slug

    @pytest.mark.parametrize("field", ["isl", "osl", "num_gpus"])
    def test_rejects_nonpositive(self, field):
        with pytest.raises(ValueError, match=">= 1"):
            Workload(**{field: 0})

    @pytest.mark.parametrize("field", ["isl_stddev", "osl_stddev"])
    def test_rejects_negative_stddev(self, field):
        with pytest.raises(ValueError, match=">= 0"):
            Workload(**{field: -1})

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("INCO_MODEL", "Qwen/Qwen3-4B")
        monkeypatch.setenv("INCO_OSL", "512")
        monkeypatch.setenv("INCO_STREAMING", "false")
        monkeypatch.setenv("INCO_TOKENIZER", "none")
        workload = Workload.from_env()
        assert (workload.model, workload.osl, workload.streaming) == (
            "Qwen/Qwen3-4B",
            512,
            False,
        )
        assert workload.tokenizer is None

    def test_empty_env_var_is_ignored(self, monkeypatch):
        monkeypatch.setenv("INCO_MODEL", "")
        assert Workload.from_env().model == Workload().model


class TestSweepConfig:
    def test_request_count_scales_then_clamps(self):
        sweep = SweepConfig(
            requests_per_concurrency=8, min_requests=24, max_requests=512
        )
        assert sweep.request_count(1) == 24  # 8 -> floored
        assert sweep.request_count(8) == 64  # scaled
        assert sweep.request_count(256) == 512  # 2048 -> capped

    def test_request_count_boundaries(self):
        sweep = SweepConfig(requests_per_concurrency=1, min_requests=4, max_requests=4)
        assert sweep.request_count(1) == 4
        assert sweep.request_count(1000) == 4

    def test_artifact_dir_is_sorted_by_name(self):
        sweep, workload = SweepConfig(artifact_root="/r", label="base"), Workload()
        dirs = [str(sweep.artifact_dir(workload, c)) for c in (2, 16, 128)]
        assert dirs == sorted(dirs), "zero padding must keep lexical == numeric order"
        assert dirs[0].startswith(f"/r/base/{workload.slug}/concurrency")

    def test_run_dir(self):
        assert str(SweepConfig(artifact_root="/r", label="opt").run_dir) == "/r/opt"

    def test_defaults_cover_single_user_to_saturation(self):
        assert SweepConfig().concurrencies == DEFAULT_CONCURRENCIES
        assert min(DEFAULT_CONCURRENCIES) == 1

    def test_rejects_empty_concurrencies(self):
        with pytest.raises(ValueError, match="must not be empty"):
            SweepConfig(concurrencies=())

    def test_rejects_nonpositive_concurrency(self):
        with pytest.raises(ValueError, match=">= 1"):
            SweepConfig(concurrencies=(1, 0))

    def test_rejects_inverted_request_bounds(self):
        with pytest.raises(ValueError, match="min_requests must be <="):
            SweepConfig(min_requests=100, max_requests=10)

    def test_rejects_zero_requests_per_concurrency(self):
        with pytest.raises(ValueError, match="requests_per_concurrency"):
            SweepConfig(requests_per_concurrency=0)

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("INCO_CONCURRENCIES", "1,4,16")
        monkeypatch.setenv("INCO_BENCHMARK_DURATION", "30")
        monkeypatch.setenv("INCO_RESET_PREFIX_CACHE", "0")
        sweep = SweepConfig.from_env()
        assert sweep.concurrencies == (1, 4, 16)
        assert sweep.benchmark_duration == 30.0
        assert sweep.reset_prefix_cache is False


class TestRunManifest:
    def test_roundtrip_is_reproducible(self, tmp_path):
        manifest = RunManifest.create(Workload(), SweepConfig())
        manifest.commands.append(["aiperf", "profile", "--concurrency", "1"])
        manifest.server_info = {"vllm_config": {"model_config": {}}}
        path = manifest.write(tmp_path / "nested" / "manifest.json")

        import json

        payload = json.loads(path.read_text())
        assert payload["workload"]["model"] == Workload().model
        assert payload["sweep"]["concurrencies"] == list(DEFAULT_CONCURRENCIES)
        assert payload["commands"][0][0] == "aiperf"
        assert payload["server_info"]["vllm_config"] == {"model_config": {}}


class TestWarmupCount:
    """Warmup must cover one full wave at the target width: it runs *at* the
    concurrency it warms, so a smaller count warms a narrower CUDA graph and
    leaves the target width's first-touch cost in the measurement."""

    def test_scales_to_one_full_wave(self):
        sweep = SweepConfig(warmup_requests=16)
        assert [sweep.warmup_count(c) for c in (32, 96, 448)] == [32, 96, 448]

    def test_configured_value_is_a_floor(self):
        sweep = SweepConfig(warmup_requests=16)
        assert [sweep.warmup_count(c) for c in (1, 8, 16)] == [16, 16, 16]

    def test_zero_disables_warmup_entirely(self):
        assert SweepConfig(warmup_requests=0).warmup_count(48) == 0

    def test_negative_is_clamped_to_zero(self):
        assert SweepConfig(warmup_requests=-5).warmup_count(48) == 0
