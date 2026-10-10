# SPDX-License-Identifier: Apache-2.0
"""Shared fakes for the harness tests.

Everything the suite needs to stand in for a GPU lives here: aiperf exports,
sweep points, `/server_info` payloads, and on-disk artifact trees.
"""

import json
import sys
from pathlib import Path

import pytest

INCO_DIR = Path(__file__).resolve().parent.parent
if str(INCO_DIR) not in sys.path:
    sys.path.insert(0, str(INCO_DIR))

EXPORT_NAME = "profile_export_aiperf.json"


def _scalar(avg, unit):
    return {"unit": unit, "avg": avg}


def _latency(avg):
    """Latency distribution block: the tail is slower than the mean."""
    return {
        "unit": "ms",
        "avg": avg,
        "p50": avg,
        "p90": avg * 1.5,
        "p99": avg * 2,
        "min": avg * 0.5,
        "max": avg * 3,
        "std": avg * 0.1,
    }


def _rate(avg):
    """Rate distribution block: the tail is slower, i.e. fewer tokens/sec."""
    return {
        "unit": "tokens/sec",
        "avg": avg,
        "p50": avg,
        "p90": avg * 0.9,
        "p99": avg * 0.8,
        "min": avg * 0.7,
        "max": avg * 1.1,
        "std": avg * 0.05,
    }


def make_export(
    *,
    total_tps=4800.0,
    per_user=75.0,
    ttft=120.0,
    itl=13.3,
    osl=256.0,
    latency=3500.0,
    request_count=256.0,
    errors=0.0,
):
    """A realistically shaped ``profile_export_aiperf.json`` payload.

    A metric passed as None is omitted, which is how aiperf represents metrics
    that do not apply to a run (no ITL without ``--streaming``, for instance).
    """
    metrics = {
        "output_token_throughput": _scalar(total_tps, "tokens/sec"),
        "output_token_throughput_per_user": _rate(per_user) if per_user else None,
        "time_to_first_token": _latency(ttft) if ttft else None,
        "inter_token_latency": _latency(itl) if itl else None,
        "request_latency": _latency(latency) if latency else None,
        "output_sequence_length": _scalar(osl, "tokens") if osl else None,
        "input_sequence_length": _scalar(1024.0, "tokens") if osl else None,
        "request_count": _scalar(request_count, "count") if request_count else None,
        "request_throughput": (
            _scalar(total_tps / (osl or 256.0), "requests/sec")
            if request_count
            else None
        ),
        "benchmark_duration": _scalar(60.0, "seconds") if request_count else None,
        "error_request_count": (
            _scalar(errors, "count") if errors is not None else None
        ),
    }
    return {
        "schema_version": "1.5",
        "input_config": {
            "endpoint": {"model_names": ["Qwen/Qwen3-30B-A3B-Instruct-2507"]}
        },
        **{tag: block for tag, block in metrics.items() if block is not None},
    }


def make_point(concurrency, per_user, per_gpu, label="baseline", **kwargs):
    """A :class:`SweepPoint` with only the fields a test cares about."""
    from bench.collect import SweepPoint

    return SweepPoint(
        concurrency=concurrency,
        label=label,
        tokens_per_s_per_user=per_user,
        tokens_per_s_per_gpu=per_gpu,
        output_token_throughput=per_gpu,
        num_gpus=1,
        **kwargs,
    )


def make_server_info(**sections):
    """A ``/server_info`` payload for a fully-featured server.

    Override a section by keyword: ``make_server_info(model={"enforce_eager": True})``.
    """
    defaults = {
        "compilation": {"cudagraph_mode": "FULL_AND_PIECEWISE"},
        "scheduler": {
            "async_scheduling": True,
            "max_num_seqs": 256,
            "max_num_batched_tokens": 8192,
        },
        "cache": {
            "enable_prefix_caching": True,
            "cache_dtype": "auto",
            # Set by the engine after it profiles free memory.
            "kv_cache_size_tokens": 76_000,
            "kv_cache_max_concurrency": 18.5,
        },
        "model": {"enforce_eager": False, "dtype": "torch.bfloat16"},
        "parallel": {"tensor_parallel_size": 1, "enable_expert_parallel": False},
    }
    env = sections.pop("env", {"VLLM_ATTENTION_BACKEND": "FLASH_ATTN"})
    merged = {name: {**default} for name, default in defaults.items()}
    for name, override in sections.items():
        merged[name].update(override)
    return {
        "vllm_config": {f"{name}_config": block for name, block in merged.items()},
        "vllm_env": env,
    }


def write_export(point_dir, export):
    """Write one export into an aiperf-style nested artifact directory."""
    nested = Path(point_dir) / "Qwen-openai-chat"
    nested.mkdir(parents=True, exist_ok=True)
    path = nested / EXPORT_NAME
    path.write_text(json.dumps(export))
    return path


def flag_value(cmd, flag):
    """The value following ``flag`` in a rendered command line."""
    return cmd[cmd.index(flag) + 1]


def completed(returncode):
    """Stand-in for a ``subprocess.CompletedProcess``."""
    return type("Completed", (), {"returncode": returncode})()


class FakeHttpResponse:
    """Minimal `urlopen()` return value: a context manager with body + status."""

    def __init__(self, body=b"", status=200):
        self.body = body
        self.status = status

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeClock:
    """Monotonic clock whose only advance is an explicit sleep."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def export_factory():
    return make_export


@pytest.fixture
def run_tree(tmp_path):
    """Write ``{concurrency: export}`` under ``<root>/<label>/<slug>/``.

    Returns the run directory; ``tmp_path`` is the artifact root that
    ``--artifact-root`` and ``load_runs`` expect.
    """

    def _build(label, points, slug="model-isl-osl"):
        run_dir = tmp_path / label / slug
        for concurrency, export in points.items():
            write_export(run_dir / f"concurrency{concurrency:04d}", export)
        return run_dir

    return _build
