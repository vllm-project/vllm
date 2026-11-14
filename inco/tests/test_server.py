# SPDX-License-Identifier: Apache-2.0
"""The perf-feature audit is what makes the baseline credible: a server that
quietly fell back to eager mode must fail the audit, not produce a curve."""

import pytest
from bench import server as server_mod
from bench.server import (
    ServerNotReady,
    _cudagraphs_disabled,
    _dig,
    _first_gpu_model,
    _get_status,
    audit_perf_features,
    fetch_server_info,
    reset_prefix_cache,
    wait_for_server,
)
from conftest import FakeClock
from conftest import make_server_info as server_info


class TestWaitForServer:
    def test_returns_immediately_when_healthy(self):
        clock = FakeClock()
        elapsed = wait_for_server(
            "http://h", sleep=clock.sleep, clock=clock, getter=lambda *a, **k: {}
        )
        assert elapsed == 0.0

    def test_retries_until_healthy_and_reports_elapsed(self):
        clock = FakeClock()
        attempts = []

        def getter(url, timeout=5.0):
            attempts.append(url)
            if len(attempts) < 4:
                raise ConnectionRefusedError
            return {}

        elapsed = wait_for_server(
            "http://h/",
            timeout_s=100,
            interval_s=5,
            sleep=clock.sleep,
            clock=clock,
            getter=getter,
        )
        assert len(attempts) == 4
        assert attempts[0] == "http://h/health", "trailing slash must not double up"
        assert elapsed == 15.0

    def test_raises_after_timeout(self):
        clock = FakeClock()

        def getter(url, timeout=5.0):
            raise ConnectionRefusedError

        with pytest.raises(ServerNotReady, match="did not become healthy"):
            wait_for_server(
                "http://h",
                timeout_s=10,
                interval_s=5,
                sleep=clock.sleep,
                clock=clock,
                getter=getter,
            )


class TestFetchServerInfo:
    def test_returns_payload(self):
        payload = server_info()
        got = fetch_server_info("http://h", getter=lambda url, **k: payload)
        assert got is payload

    def test_requests_json_format(self):
        seen = {}

        def getter(url, **kwargs):
            seen["url"] = url
            return {}

        fetch_server_info("http://h/", getter=getter)
        assert seen["url"] == "http://h/server_info?config_format=json"

    def test_dev_mode_off_is_not_fatal(self):
        def getter(url, **kwargs):
            raise OSError("404")

        assert fetch_server_info("http://h", getter=getter) is None


class TestResetPrefixCache:
    def test_success(self):
        assert reset_prefix_cache("http://h", poster=lambda url, **k: 200) is True

    def test_failure_is_reported_not_raised(self):
        def poster(url, **kwargs):
            raise OSError("404")

        assert reset_prefix_cache("http://h", poster=poster) is False

    def test_posts_to_the_dev_route(self):
        seen = {}
        reset_prefix_cache(
            "http://h/", poster=lambda url, **k: seen.setdefault("url", url)
        )
        assert seen["url"] == "http://h/reset_prefix_cache"


class TestAudit:
    def test_healthy_server_has_no_problems(self):
        audit = audit_perf_features(server_info())
        assert audit.problems() == []
        assert audit.cudagraph_mode == "FULL_AND_PIECEWISE"
        assert audit.async_scheduling is True
        assert audit.max_num_batched_tokens == 8192
        assert audit.attention_backend == "FLASH_ATTN"
        assert audit.expert_parallel is False
        assert audit.kv_cache_size_tokens == 76_000

    def test_enforce_eager_is_caught(self):
        audit = audit_perf_features(server_info(model={"enforce_eager": True}))
        assert any("CUDA graphs are disabled" in w for w in audit.problems())

    @pytest.mark.parametrize("mode", ["NONE", "none", "0"])
    def test_cudagraphs_off_is_caught(self, mode):
        audit = audit_perf_features(server_info(compilation={"cudagraph_mode": mode}))
        assert any("no CUDA graphs" in w for w in audit.problems())

    def test_async_scheduling_off_is_caught(self):
        audit = audit_perf_features(server_info(scheduler={"async_scheduling": False}))
        assert any("overlap scheduler off" in w for w in audit.problems())

    def test_unknown_async_scheduling_is_not_flagged(self):
        """None means the server did not report it; do not fail the run on that."""
        audit = audit_perf_features(server_info(scheduler={"async_scheduling": None}))
        assert audit.problems() == []

    def test_batch_cap_below_the_sweep_is_caught(self):
        audit = audit_perf_features(server_info(scheduler={"max_num_seqs": 32}))
        assert any("queue-bound" in w for w in audit.problems(max_concurrency=48))

    def test_batch_cap_is_not_judged_without_a_sweep_width(self):
        """A small max_num_seqs is only a problem relative to the sweep."""
        audit = audit_perf_features(server_info(scheduler={"max_num_seqs": 32}))
        assert audit.problems() == []

    def test_batch_cap_matching_the_sweep_is_fine(self):
        audit = audit_perf_features(server_info(scheduler={"max_num_seqs": 48}))
        assert audit.problems(max_concurrency=48) == []

    def test_multiple_problems_all_reported(self):
        audit = audit_perf_features(
            server_info(
                model={"enforce_eager": True},
                scheduler={"async_scheduling": False, "max_num_seqs": 32},
            )
        )
        assert len(audit.problems(max_concurrency=48)) == 3

    def test_missing_server_info_yields_empty_audit(self):
        import dataclasses

        audit = audit_perf_features(None)
        assert all(v is None for v in dataclasses.asdict(audit).values())
        assert audit.problems() == []

    def test_text_format_config_is_tolerated(self):
        """/server_info returns a repr string unless config_format=json."""
        audit = audit_perf_features({"vllm_config": "VllmConfig(model=...)"})
        assert audit.cudagraph_mode is None
        assert audit.problems() == []

    def test_partial_config_does_not_raise(self):
        audit = audit_perf_features({"vllm_config": {"scheduler_config": {}}})
        assert audit.max_num_seqs is None


class TestDig:
    def test_walks_nested_keys(self):
        assert _dig({"a": {"b": {"c": 1}}}, "a", "b", "c") == 1

    def test_missing_key(self):
        assert _dig({"a": {}}, "a", "b") is None

    def test_stops_at_a_non_dict_mid_path(self):
        assert _dig({"a": "VllmConfig(...)"}, "a", "b") is None

    def test_empty_path_returns_input(self):
        assert _dig({"a": 1}) == {"a": 1}


class TestServableConcurrency:
    """The 30B MoE leaves little KV cache, so capacity decides how wide the
    sweep can go. The engine reports the real number after profiling."""

    def test_capacity_is_tokens_over_sequence_length(self):
        audit = audit_perf_features(server_info())
        assert audit.servable_concurrency(1280) == pytest.approx(76_000 / 1280)

    def test_shorter_sequences_fit_more_requests(self):
        audit = audit_perf_features(server_info())
        assert audit.servable_concurrency(640) > audit.servable_concurrency(1280)

    def test_unreported_capacity_is_none(self):
        audit = audit_perf_features(server_info(cache={"kv_cache_size_tokens": None}))
        assert audit.servable_concurrency(1280) is None

    def test_missing_server_info_has_no_capacity(self):
        assert audit_perf_features(None).servable_concurrency(1280) is None

    @pytest.mark.parametrize("seq_len", [0, -1])
    def test_nonpositive_sequence_length_does_not_divide_by_zero(self, seq_len):
        assert audit_perf_features(server_info()).servable_concurrency(seq_len) is None


class TestWaitHeartbeat:
    """A silent wait is indistinguishable from a hang, so callers get a hook."""

    def test_on_wait_is_called_with_elapsed_seconds(self):
        clock = FakeClock()
        beats = []
        attempts = []

        def getter(url, timeout=5.0):
            attempts.append(url)
            if len(attempts) < 3:
                raise ConnectionRefusedError
            return {}

        wait_for_server(
            "http://h",
            timeout_s=100,
            interval_s=5,
            sleep=clock.sleep,
            clock=clock,
            getter=getter,
            on_wait=beats.append,
        )
        assert beats == [0.0, 5.0]

    def test_no_heartbeat_when_already_healthy(self):
        beats = []
        wait_for_server("http://h", getter=lambda *a, **k: {}, on_wait=beats.append)
        assert beats == []


class TestHealthProbeUsesStatusNotBody:
    """vLLM's /health answers 200 with an EMPTY body.

    The probe must therefore look at the status code only. Parsing the body as
    JSON raises, the retry loop swallows it as "not up yet", and the sweep
    spins until its 30-minute timeout against a server that is already up --
    a silent failure that looks exactly like a slow model load.
    """

    def test_empty_body_is_healthy(self, monkeypatch):
        import io

        class EmptyResponse(io.BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(
            server_mod.urllib.request,
            "urlopen",
            lambda url, timeout=0: EmptyResponse(b""),
        )
        assert _get_status("http://h/health") == 200

    def test_wait_for_server_accepts_an_empty_health_body(self, monkeypatch):
        import io

        class EmptyResponse(io.BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(
            server_mod.urllib.request,
            "urlopen",
            lambda url, timeout=0: EmptyResponse(b""),
        )
        clock = FakeClock()
        # Default getter: the regression is that this used to never return.
        assert (
            wait_for_server("http://h", timeout_s=30, sleep=clock.sleep, clock=clock)
            == 0.0
        )


class TestGpuModel:
    """The GPU must be recorded per run: a provider can serve variants with
    different memory for the same request (H100 80GB HBM3 vs H100 NVL), which
    silently doubles KV capacity and makes two curves incomparable."""

    def test_strips_collect_env_prefix(self):
        env = {"nvidia_gpu_models": "GPU 0: NVIDIA H100 NVL"}
        assert _first_gpu_model(env) == "NVIDIA H100 NVL"

    def test_bare_name_passes_through(self):
        env = {"nvidia_gpu_models": "NVIDIA H100 80GB HBM3"}
        assert _first_gpu_model(env) == "NVIDIA H100 80GB HBM3"

    def test_first_of_several_gpus(self):
        env = {"nvidia_gpu_models": "GPU 0: NVIDIA H100 NVL\nGPU 1: NVIDIA H100 NVL"}
        assert _first_gpu_model(env) == "NVIDIA H100 NVL"

    @pytest.mark.parametrize(
        "env",
        [
            None,
            {},
            "not-a-dict",
            {"nvidia_gpu_models": ""},
            {"nvidia_gpu_models": None},
        ],
    )
    def test_missing_or_malformed_is_none(self, env):
        assert _first_gpu_model(env) is None

    def test_surfaced_on_the_audit(self):
        audit = audit_perf_features(
            server_info(env={})
            | {"system_env": {"nvidia_gpu_models": "GPU 0: NVIDIA H100 NVL"}}
        )
        assert audit.gpu_model == "NVIDIA H100 NVL"


class TestCudagraphDetection:
    """cudagraph_mode arrives as the enum's *value*: an int for NONE/PIECEWISE/
    FULL (0/1/2) or a pair for composites (FULL_AND_PIECEWISE -> [2, 1]). A
    string comparison never matches a disabled server, so the gate would pass
    a server running eager."""

    @pytest.mark.parametrize("mode", ["NONE", "none", "0", 0, [0, 0], (0, 0)])
    def test_disabled_forms_are_caught(self, mode):
        assert _cudagraphs_disabled(mode) is True

    @pytest.mark.parametrize(
        "mode", [None, 1, 2, [2, 1], [2, 0], [0, 1], [], "FULL_AND_PIECEWISE"]
    )
    def test_enabled_or_unknown_forms_pass(self, mode):
        assert _cudagraphs_disabled(mode) is False

    def test_audit_flags_the_real_json_none_form(self):
        audit = audit_perf_features(server_info(compilation={"cudagraph_mode": 0}))
        assert any("no CUDA graphs" in w for w in audit.problems())

    def test_audit_accepts_the_real_json_full_and_piecewise_form(self):
        audit = audit_perf_features(server_info(compilation={"cudagraph_mode": [2, 1]}))
        assert audit.problems() == []
