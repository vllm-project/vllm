# SPDX-License-Identifier: Apache-2.0
"""CLI plumbing and the refuse-to-benchmark-a-bad-server gate."""

import pytest
from bench import sweep as sweep_mod
from bench.config import Workload
from conftest import make_server_info, write_export


def parse(argv):
    return sweep_mod.build_parser().parse_args(argv)


@pytest.fixture
def audit_against(monkeypatch, capsys):
    """Stub `/server_info` with `info`, then run the audit gate against it."""

    def _audit(info, allow_degraded=False, argv=()):
        monkeypatch.setattr(sweep_mod, "fetch_server_info", lambda url: info)
        workload, sweep = sweep_mod.configs_from_args(parse(list(argv)))
        return sweep_mod.audit_or_exit(workload, sweep, allow_degraded)

    return _audit


class TestCli:
    def test_defaults_match_config_defaults(self):
        workload, sweep = sweep_mod.configs_from_args(parse([]))
        assert workload.model == Workload().model
        assert workload.ignore_eos is True
        assert sweep.label == "baseline"
        assert sweep.reset_prefix_cache is True

    def test_overrides_flow_into_configs(self):
        args = parse(
            [
                "--model",
                "Qwen/Qwen3-4B",
                "--isl",
                "512",
                "--osl",
                "128",
                "--num-gpus",
                "2",
                "--label",
                "fp8",
                "--concurrency",
                "1",
                "16",
                "--url",
                "http://gpu:9000",
            ]
        )
        workload, sweep = sweep_mod.configs_from_args(args)
        assert (workload.model, workload.isl, workload.osl) == (
            "Qwen/Qwen3-4B",
            512,
            128,
        )
        assert workload.num_gpus == 2
        assert sweep.concurrencies == (1, 16)
        assert sweep.url == "http://gpu:9000"
        assert sweep.label == "fp8"

    def test_negative_switches(self):
        args = parse(["--no-ignore-eos", "--keep-prefix-cache"])
        workload, sweep = sweep_mod.configs_from_args(args)
        assert workload.ignore_eos is False
        assert sweep.reset_prefix_cache is False

    def test_env_seeds_cli_defaults(self, monkeypatch):
        monkeypatch.setenv("INCO_LABEL", "from-env")
        monkeypatch.setenv("INCO_MODEL", "Qwen/Qwen3-32B")
        _, sweep = sweep_mod.configs_from_args(parse([]))
        assert sweep.label == "from-env"

    def test_dry_run_prints_commands_without_a_server(self, capsys, tmp_path):
        rc = sweep_mod.main(
            [
                "--dry-run",
                "--concurrency",
                "1",
                "64",
                "--artifact-root",
                str(tmp_path),
                "--aiperf-bin",
                "aiperf-not-installed-xyz",
            ]
        )
        out = capsys.readouterr().out
        assert rc == 0
        assert out.count("aiperf-not-installed-xyz profile") == 2
        assert "--concurrency 64" in out

    def test_invalid_concurrency_is_rejected(self):
        with pytest.raises(ValueError, match=">= 1"):
            sweep_mod.configs_from_args(parse(["--concurrency", "0"]))


class TestAuditGate:
    def test_missing_dev_mode_warns_and_continues(self, audit_against, capsys):
        assert audit_against(None) is None
        assert "server_info unavailable" in capsys.readouterr().err

    def test_healthy_server_passes(self, audit_against, capsys):
        info = make_server_info()
        assert audit_against(info) is info
        assert "audit passed" in capsys.readouterr().out

    def test_degraded_server_aborts_the_sweep(self, audit_against):
        eager = make_server_info(model={"enforce_eager": True})
        with pytest.raises(SystemExit, match="refusing to benchmark"):
            audit_against(eager)

    def test_degraded_server_allowed_with_opt_in(self, audit_against, capsys):
        eager = make_server_info(model={"enforce_eager": True})
        assert audit_against(eager, allow_degraded=True) is eager
        assert "CUDA graphs are disabled" in capsys.readouterr().err


class TestRenderReport:
    def test_writes_csv_summary_and_plot(self, tmp_path, export_factory, capsys):
        workload = Workload()
        args = parse(["--artifact-root", str(tmp_path), "--label", "baseline"])
        _, sweep = sweep_mod.configs_from_args(args)
        for concurrency, tps in ((1, 95.0), (32, 2240.0)):
            write_export(
                sweep.artifact_dir(workload, concurrency),
                export_factory(total_tps=tps, per_user=tps / concurrency),
            )

        sweep_mod.render_report(workload, sweep)
        out = capsys.readouterr().out
        assert (sweep.run_dir / "pareto.csv").exists()
        assert (sweep.run_dir / "summary.md").exists()
        assert workload.model in (sweep.run_dir / "summary.md").read_text()
        assert "| Conc |" in out

    def test_no_results_warns_instead_of_crashing(self, tmp_path, capsys):
        args = parse(["--artifact-root", str(tmp_path)])
        workload, sweep = sweep_mod.configs_from_args(args)
        sweep_mod.render_report(workload, sweep)
        assert "no results under" in capsys.readouterr().err


class TestKvCapacityReport:
    """Qwen3-30B-A3B's bf16 weights leave little KV cache, so the sweep can
    outrun what the engine can hold resident. That has to be said out loud."""

    def test_capacity_is_reported_in_requests(self, audit_against, capsys):
        assert audit_against(make_server_info()) is not None
        out = capsys.readouterr().out
        assert "76,000 tokens" in out
        assert "59 resident requests at ISL+OSL=1280" in out

    def test_sweep_beyond_capacity_is_flagged(self, audit_against, capsys):
        audit_against(
            make_server_info(scheduler={"max_num_seqs": 256}),
            argv=["--concurrency", "1", "48", "128"],
        )
        err = capsys.readouterr().err
        assert "concurrency [128] exceeds KV capacity" in err
        assert "measure the queue, not the engine" in err

    def test_sweep_within_capacity_is_not_flagged(self, audit_against, capsys):
        audit_against(make_server_info(), argv=["--concurrency", "1", "48"])
        assert "exceeds KV capacity" not in capsys.readouterr().err

    def test_shorter_sequences_raise_the_usable_concurrency(
        self, audit_against, capsys
    ):
        audit_against(
            make_server_info(scheduler={"max_num_seqs": 256}),
            argv=["--isl", "256", "--osl", "64", "--concurrency", "128"],
        )
        assert "exceeds KV capacity" not in capsys.readouterr().err

    def test_unreported_capacity_is_silent(self, audit_against, capsys):
        audit_against(make_server_info(cache={"kv_cache_size_tokens": None}))
        assert "KV cache holds" not in capsys.readouterr().out


class TestIntegrityWarningsAreSurfaced:
    def test_impossible_ordering_is_logged_to_stderr(
        self, tmp_path, run_tree, export_factory, capsys
    ):
        """A curve where tokens/s/user rises with concurrency must not pass
        silently; it invalidates the run rather than merely adding noise."""
        args = parse(["--artifact-root", str(tmp_path), "--label", "bad"])
        workload, sweep = sweep_mod.configs_from_args(args)
        for concurrency, per_user in ((320, 37.6), (336, 38.5)):
            write_export(
                sweep.artifact_dir(workload, concurrency),
                export_factory(total_tps=per_user * concurrency, per_user=per_user),
            )
        sweep_mod.render_report(workload, sweep)
        assert "impossible" in capsys.readouterr().err
