# SPDX-License-Identifier: Apache-2.0
"""End-to-end sweep with the GPU replaced by fakes.

Covers the paths that only run on a real benchmarking host: flag resolution
against an installed binary, the measurement loop, per-point prefix-cache
resets, and failure handling.
"""

import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from bench import aiperf as aiperf_mod
from bench import server as server_mod
from bench import sweep as sweep_mod
from bench.aiperf import AiperfResult, aiperf_help
from conftest import FakeHttpResponse, make_server_info, write_export


@dataclass
class FakeGpu:
    """A healthy vLLM server with a fake aiperf in front of it.

    Mutate the fields to inject the conditions a real host would produce:
    failing concurrency points, request errors, or a server without dev mode.
    """

    root: Path
    server_info: dict | None = field(default_factory=make_server_info)
    failures: set[int] = field(default_factory=set)
    errors: float = 0.0
    reset_ok: bool = True
    resets: list[str] = field(default_factory=list)

    def run(self, *extra_args) -> int:
        return sweep_mod.main(["--artifact-root", str(self.root), *extra_args])

    @property
    def run_dir(self) -> Path:
        return self.root / "baseline"


@pytest.fixture
def gpu(monkeypatch, tmp_path, export_factory):
    fake = FakeGpu(root=tmp_path)

    def run_point(workload, sweep, concurrency, help_text=None, runner=None):
        """Write the export a real aiperf run would have produced."""
        artifact_dir = sweep.artifact_dir(workload, concurrency)
        cmd = aiperf_mod.build_aiperf_command(workload, sweep, concurrency, help_text)
        if concurrency in fake.failures:
            return AiperfResult(concurrency, 1, artifact_dir, cmd, None)
        export = export_factory(
            total_tps=80.0 * concurrency**0.8,
            per_user=90.0 / concurrency**0.2,
            request_count=float(sweep.request_count(concurrency)),
            errors=fake.errors,
        )
        path = write_export(artifact_dir, export)
        return AiperfResult(concurrency, 0, artifact_dir, cmd, path)

    def reset(url):
        fake.resets.append(url)
        return fake.reset_ok

    monkeypatch.setattr(
        sweep_mod, "wait_for_server", lambda url, timeout_s=0, on_wait=None: 1.5
    )
    monkeypatch.setattr(sweep_mod, "fetch_server_info", lambda url: fake.server_info)
    monkeypatch.setattr(sweep_mod, "reset_prefix_cache", reset)
    monkeypatch.setattr(sweep_mod, "aiperf_help", lambda binary: None)
    monkeypatch.setattr(sweep_mod, "run_concurrency_point", run_point)
    return fake


@pytest.fixture
def fake_binary(tmp_path, monkeypatch):
    """Put an executable shell script named `aiperf` at the front of PATH."""

    def _install(body):
        path = tmp_path / "aiperf"
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
        aiperf_help.cache_clear()
        return str(path)

    yield _install
    aiperf_help.cache_clear()


class TestFullSweep:
    def test_produces_curve_manifest_and_report(self, gpu, capsys):
        assert gpu.run("--concurrency", "1", "8", "64") == 0

        rows = (gpu.run_dir / "pareto.csv").read_text().splitlines()
        assert len(rows) == 4, "header + one row per concurrency"
        assert "interactivity SLO" in (gpu.run_dir / "summary.md").read_text()

        manifest = json.loads((gpu.run_dir / "manifest.json").read_text())
        assert len(manifest["commands"]) == 3
        assert manifest["workload"]["osl"] == 256
        assert (
            manifest["server_info"]["vllm_config"]["scheduler_config"][
                "async_scheduling"
            ]
            is True
        )

        out = capsys.readouterr().out
        assert "audit passed" in out
        assert out.count("[ok]") == 3

    def test_prefix_cache_is_flushed_before_every_point(self, gpu):
        gpu.run("--concurrency", "1", "8")
        assert len(gpu.resets) == 2

    def test_keep_prefix_cache_skips_the_reset(self, gpu):
        gpu.run("--concurrency", "1", "--keep-prefix-cache")
        assert gpu.resets == []

    def test_failed_prefix_cache_reset_only_warns(self, gpu, capsys):
        gpu.reset_ok = False
        assert gpu.run("--concurrency", "1") == 0
        assert "could not reset prefix cache" in capsys.readouterr().err

    def test_failed_point_is_reported_and_exits_nonzero(self, gpu, capsys):
        gpu.failures = {8}
        assert gpu.run("--concurrency", "1", "8") == 1
        assert "concurrencies failed: [8]" in capsys.readouterr().err
        # The surviving point is still reported rather than thrown away.
        assert len((gpu.run_dir / "pareto.csv").read_text().splitlines()) == 2

    def test_error_rate_is_surfaced(self, gpu, capsys):
        gpu.errors = 5.0
        gpu.run("--concurrency", "8")
        assert "error rate" in capsys.readouterr().err

    def test_analyze_only_reuses_artifacts_without_a_server(
        self, gpu, monkeypatch, capsys
    ):
        gpu.run("--concurrency", "1", "8")
        capsys.readouterr()

        def explode(*args, **kwargs):
            raise AssertionError("analyze-only must not touch the server")

        monkeypatch.setattr(sweep_mod, "wait_for_server", explode)
        assert gpu.run("--analyze-only", "--concurrency", "1") == 0
        assert "| Conc |" in capsys.readouterr().out

    def test_missing_aiperf_is_fatal_for_a_real_run(self, gpu, monkeypatch):
        def missing(binary):
            raise aiperf_mod.AiperfNotInstalled("nope")

        monkeypatch.setattr(sweep_mod, "aiperf_help", missing)
        with pytest.raises(aiperf_mod.AiperfNotInstalled):
            gpu.run("--concurrency", "1")


class TestAiperfHelpAgainstABinary:
    def test_help_output_is_captured_and_cached(self, fake_binary):
        binary = fake_binary('echo "--prompt-input-tokens-mean INT --request-count"')
        help_text = aiperf_help(binary)
        assert aiperf_mod.resolve_flag("isl", help_text) == "--prompt-input-tokens-mean"
        assert aiperf_help(binary) is help_text, "help output must be cached"

    def test_stderr_only_help_is_still_usable(self, fake_binary):
        binary = fake_binary('echo "--osl INT" >&2')
        assert aiperf_mod.resolve_flag("osl", aiperf_help(binary)) == "--osl"


class TestHttpHelpers:
    def test_get_json_decodes_a_response(self, monkeypatch):
        monkeypatch.setattr(
            server_mod.urllib.request,
            "urlopen",
            lambda url, timeout=0: FakeHttpResponse(body=b'{"ok": true}'),
        )
        assert server_mod._get_json("http://h/health") == {"ok": True}

    def test_post_returns_status_and_uses_the_post_method(self, monkeypatch):
        seen = {}

        def urlopen(request, timeout=0):
            seen["method"] = request.method
            return FakeHttpResponse(status=200)

        monkeypatch.setattr(server_mod.urllib.request, "urlopen", urlopen)
        assert server_mod._post("http://h/reset_prefix_cache") == 200
        assert seen["method"] == "POST"


def test_compare_without_matplotlib(
    tmp_path, run_tree, export_factory, monkeypatch, capsys
):
    from bench import compare, report

    for label in ("baseline", "opt"):
        run_tree(label, {32: export_factory()})
    monkeypatch.setattr(report, "plot_pareto", lambda *a, **k: None)
    assert compare.main(["baseline", "opt", "--artifact-root", str(tmp_path)]) == 0
    assert "matplotlib missing" in capsys.readouterr().err
