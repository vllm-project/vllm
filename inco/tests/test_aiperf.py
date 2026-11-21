# SPDX-License-Identifier: Apache-2.0
"""The aiperf command *is* the benchmark definition, so it is asserted flag by
flag: a silently dropped --ignore-eos or --streaming changes what is measured
without failing anything."""

from pathlib import Path

import pytest
from bench.aiperf import (
    AiperfNotInstalled,
    AiperfResult,
    aiperf_help,
    build_aiperf_command,
    find_export_json,
    resolve_flag,
    run_concurrency_point,
    streaming_runner,
)
from bench.config import SweepConfig, Workload
from conftest import EXPORT_NAME, completed, flag_value


@pytest.fixture
def workload():
    return Workload(model="Qwen/Qwen3-30B-A3B-Instruct-2507", isl=1024, osl=256)


@pytest.fixture
def sweep(tmp_path):
    return SweepConfig(artifact_root=str(tmp_path), label="baseline")


class TestResolveFlag:
    def test_prefers_modern_name_when_help_unavailable(self):
        assert resolve_flag("isl", None) == "--prompt-input-tokens-mean"

    def test_picks_modern_name_when_offered(self):
        help_text = "--prompt-input-tokens-mean INT  --synthetic-input-tokens-mean INT"
        assert resolve_flag("isl", help_text) == "--prompt-input-tokens-mean"

    def test_falls_back_to_legacy_alias(self):
        help_text = "--synthetic-input-tokens-mean INT"
        assert resolve_flag("isl", help_text) == "--synthetic-input-tokens-mean"

    def test_falls_back_to_short_alias(self):
        assert resolve_flag("osl", "--osl INT") == "--osl"

    def test_raises_when_no_alias_matches(self):
        with pytest.raises(RuntimeError, match="accepts none of"):
            resolve_flag("isl", "--totally-different INT")


class TestBuildCommand:
    def test_workload_shape_is_pinned(self, workload, sweep):
        cmd = build_aiperf_command(workload, sweep, 32)
        assert cmd[:2] == ["aiperf", "profile"]
        assert flag_value(cmd, "--model") == "Qwen/Qwen3-30B-A3B-Instruct-2507"
        assert flag_value(cmd, "--tokenizer") == "Qwen/Qwen3-30B-A3B-Instruct-2507"
        assert flag_value(cmd, "--prompt-input-tokens-mean") == "1024"
        assert flag_value(cmd, "--prompt-output-tokens-mean") == "256"
        assert flag_value(cmd, "--prompt-input-tokens-stddev") == "0"
        assert flag_value(cmd, "--prompt-output-tokens-stddev") == "0"
        assert flag_value(cmd, "--concurrency") == "32"
        assert flag_value(cmd, "--endpoint-type") == "chat"

    def test_seed_and_request_count_track_config(self, workload, sweep):
        cmd = build_aiperf_command(workload, sweep, 8)
        assert flag_value(cmd, "--random-seed") == "100"
        assert flag_value(cmd, "--request-count") == str(sweep.request_count(8))

    def test_streaming_enabled_for_token_level_metrics(self, workload, sweep):
        assert "--streaming" in build_aiperf_command(workload, sweep, 1)

    def test_streaming_omitted_when_disabled(self, sweep):
        cmd = build_aiperf_command(Workload(streaming=False), sweep, 1)
        assert "--streaming" not in cmd

    def test_ignore_eos_pins_output_length(self, workload, sweep):
        cmd = build_aiperf_command(workload, sweep, 1)
        assert flag_value(cmd, "--extra-inputs") == "ignore_eos:true"

    def test_ignore_eos_omitted_when_disabled(self, sweep):
        cmd = build_aiperf_command(Workload(ignore_eos=False), sweep, 1)
        assert "--extra-inputs" not in cmd

    def test_warmup_omitted_when_zero(self, workload, sweep):
        cmd = build_aiperf_command(workload, SweepConfig(warmup_requests=0), 1)
        assert "--warmup-request-count" not in cmd
        assert "--warmup-request-count" in build_aiperf_command(workload, sweep, 1)

    def test_warmup_covers_one_full_wave(self, workload):
        sweep = SweepConfig(warmup_requests=16)
        cmd = build_aiperf_command(workload, sweep, 4)
        assert flag_value(cmd, "--warmup-request-count") == "16"
        cmd = build_aiperf_command(workload, sweep, 448)
        assert flag_value(cmd, "--warmup-request-count") == "448"

    def test_benchmark_duration_optional(self, workload, sweep):
        assert "--benchmark-duration" not in build_aiperf_command(workload, sweep, 1)
        timed = SweepConfig(benchmark_duration=45.0)
        cmd = build_aiperf_command(workload, timed, 1)
        assert flag_value(cmd, "--benchmark-duration") == "45.0"

    def test_artifact_dir_is_per_concurrency(self, workload, sweep):
        low = flag_value(build_aiperf_command(workload, sweep, 1), "--artifact-dir")
        high = flag_value(build_aiperf_command(workload, sweep, 64), "--artifact-dir")
        assert low != high
        assert low.endswith("concurrency0001")

    def test_extra_args_are_shell_split(self, workload):
        sweep = SweepConfig(extra_aiperf_args="--conversation-num 4 --api-key tok")
        cmd = build_aiperf_command(workload, sweep, 1)
        assert cmd[-4:] == ["--conversation-num", "4", "--api-key", "tok"]

    def test_legacy_help_rewrites_every_renamed_flag(self, workload, sweep):
        legacy = (
            "--synthetic-input-tokens-mean --synthetic-input-tokens-stddev "
            "--output-tokens-mean --output-tokens-stddev --num-requests "
            "--num-warmup-requests --ui --output-artifact-dir"
        )
        cmd = build_aiperf_command(workload, sweep, 4, help_text=legacy)
        assert "--synthetic-input-tokens-mean" in cmd
        assert "--num-requests" in cmd
        assert "--output-artifact-dir" in cmd
        assert "--prompt-input-tokens-mean" not in cmd


class TestFindExport:
    def test_missing_dir(self, tmp_path):
        assert find_export_json(tmp_path / "absent") is None

    def test_empty_dir(self, tmp_path):
        assert find_export_json(tmp_path) is None

    def test_direct_hit(self, tmp_path):
        (tmp_path / EXPORT_NAME).write_text("{}")
        assert find_export_json(tmp_path) == tmp_path / EXPORT_NAME

    def test_nested_hit(self, tmp_path):
        nested = tmp_path / "Qwen_Qwen3-30B-A3B-Instruct-2507-openai-chat-concurrency8"
        nested.mkdir()
        (nested / EXPORT_NAME).write_text("{}")
        assert find_export_json(tmp_path) == nested / EXPORT_NAME

    def test_prefers_last_when_ambiguous(self, tmp_path):
        for name in ("a-run", "b-run"):
            sub = tmp_path / name
            sub.mkdir()
            (sub / EXPORT_NAME).write_text("{}")
        assert find_export_json(tmp_path) == tmp_path / "b-run" / EXPORT_NAME


class TestRunConcurrencyPoint:
    @staticmethod
    def fake_aiperf(returncode=0, write_export=True, seen=None):
        """Stand in for the aiperf subprocess, recording what it saw."""

        def runner(cmd, check=False):
            artifact_dir = Path(flag_value(cmd, "--artifact-dir"))
            if seen is not None:
                seen["artifact_dir_existed"] = artifact_dir.is_dir()
            if write_export:
                (artifact_dir / EXPORT_NAME).write_text("{}")
            return completed(returncode)

        return runner

    def test_success_reports_export(self, workload, sweep):
        result = run_concurrency_point(workload, sweep, 4, runner=self.fake_aiperf())
        assert result.ok
        assert result.concurrency == 4
        assert result.export_json.name == EXPORT_NAME

    def test_nonzero_exit_is_not_ok(self, workload, sweep):
        result = run_concurrency_point(
            workload, sweep, 4, runner=self.fake_aiperf(returncode=2)
        )
        assert not result.ok
        assert result.returncode == 2

    def test_missing_export_is_not_ok_despite_clean_exit(self, workload, sweep):
        result = run_concurrency_point(
            workload, sweep, 4, runner=self.fake_aiperf(write_export=False)
        )
        assert not result.ok, "clean exit without results must still fail the point"

    def test_artifact_dir_created_before_run(self, workload, sweep):
        seen = {}
        run_concurrency_point(workload, sweep, 1, runner=self.fake_aiperf(seen=seen))
        assert seen["artifact_dir_existed"]


def test_aiperf_help_raises_when_binary_absent():
    with pytest.raises(AiperfNotInstalled, match="not found on PATH"):
        aiperf_help("aiperf-does-not-exist-12345")


def test_result_ok_requires_both_conditions(tmp_path):
    base = dict(concurrency=1, artifact_dir=tmp_path, command=["aiperf"])
    assert AiperfResult(returncode=0, export_json=tmp_path / "x", **base).ok
    assert not AiperfResult(returncode=0, export_json=None, **base).ok
    assert not AiperfResult(returncode=1, export_json=tmp_path / "x", **base).ok


class TestStreamingRunner:
    """aiperf output must reach the log live; a quiet point must still prove
    liveness, otherwise a working sweep is indistinguishable from a hang."""

    def run(self, script, **kwargs):
        lines = []
        runner = streaming_runner(lines.append, **kwargs)
        result = runner(["sh", "-c", script])
        return lines, result

    def test_streams_each_line(self):
        lines, result = self.run("echo one; echo two")
        assert lines == ["one", "two"]
        assert result.returncode == 0

    def test_strips_ansi_escapes(self):
        lines, _ = self.run(r"printf '\033[1;32mgreen\033[0m\n'")
        assert lines == ["green"]

    def test_drops_blank_lines(self):
        lines, _ = self.run("echo; echo real; echo")
        assert lines == ["real"]

    def test_captures_stderr_too(self):
        lines, _ = self.run("echo out; echo err >&2")
        assert set(lines) == {"out", "err"}

    def test_propagates_nonzero_exit(self):
        _, result = self.run("exit 3")
        assert result.returncode == 3

    def test_heartbeat_fires_while_silent(self):
        lines, result = self.run("sleep 0.35", heartbeat_s=0.1)
        assert result.returncode == 0
        assert any("aiperf running" in line for line in lines), (
            "a silent command must still emit liveness lines"
        )

    def test_no_heartbeat_when_command_is_quick(self):
        lines, _ = self.run("echo fast", heartbeat_s=30.0)
        assert lines == ["fast"]
