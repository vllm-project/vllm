# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark and regression-test pinned (page-locked) CPU memory for vLLM.

Verifies that enabling pinned memory does not regress throughput or latency
compared to unpinned memory.  Each condition runs in an isolated ``spawn``
subprocess so both start from a cold CUDA context, giving an unbiased
comparison.

Usage
-----
Run all tests with the default model::

    python benchmarks/benchmark_pin_memory.py -v

Override the model and optional max-model-len::

    python benchmarks/benchmark_pin_memory.py --model unsloth/Qwen3-1.7B -v
    python benchmarks/benchmark_pin_memory.py --model unsloth/Qwen3-1.7B \
        --max-model-len 8192 -v

Run only throughput or latency tests::

    python benchmarks/benchmark_pin_memory.py -v -k test_throughput
    python benchmarks/benchmark_pin_memory.py -v -k test_latency

Run only the v1 or v2 runner variant::

    python benchmarks/benchmark_pin_memory.py -v -k v1
    python benchmarks/benchmark_pin_memory.py -v -k v2

Note: on WSL2, v1 runner tests are skipped because pin memory is not available
for the v1 runner without cpu_offload_gb.  Run on other platforms to exercise v1.
"""

import argparse
import json
import multiprocessing
import sys
import tempfile

import pytest

# Allow up to 2% degradation.  Both benchmark runs start from an identical
# cold CUDA context (separate spawn subprocesses), so the measured difference
# reflects the genuine pin_memory overhead rather than cold/warm ordering bias.
_THROUGHPUT_TOLERANCE = 0.98
_THROUGHPUT_NUM_REQUESTS = 200
_THROUGHPUT_INPUT_LEN = 128
_THROUGHPUT_OUTPUT_LEN = 512
_THROUGHPUT_MAX_NUM_SEQS = 128

# Latency benchmark constants — match latency.py defaults.
_LATENCY_TOLERANCE = 1.02  # Allow up to 2% latency regression.
_LATENCY_BATCH_SIZE = 64
_LATENCY_INPUT_LEN = 32
_LATENCY_OUTPUT_LEN = 128
_LATENCY_WARMUP_ITERS = 5
_LATENCY_BENCH_ITERS = 15

_DEFAULT_MODEL = "unsloth/Qwen3-1.7B"
_DEFAULT_MAX_MODEL_LEN = 16384


def _benchmark_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--max-model-len", type=int, default=_DEFAULT_MAX_MODEL_LEN)
    args, _ = parser.parse_known_args()
    return args


@pytest.fixture
def model() -> str:
    return _benchmark_args().model


@pytest.fixture
def max_model_len() -> int:
    return _benchmark_args().max_model_len


def _skip_if_pin_memory_not_available(engine_args_kwargs: dict) -> None:
    """Skip the current pytest test if pin_memory is unavailable for this config."""
    import vllm.utils.platform_utils as pu
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs

    vllm_config = EngineArgs(**engine_args_kwargs).create_engine_config()
    with set_current_vllm_config(vllm_config):
        pu.is_pin_memory_available.cache_clear()
        if not pu.is_pin_memory_available():
            import os

            runner = "v2" if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == "1" else "v1"
            model = engine_args_kwargs.get("model", "unknown")
            print(
                f"\033[33mSKIP: pin_memory not available for "
                f"{runner} runner, model={model}\033[0m"
            )
            pytest.skip("pin_memory not available for this configuration")


def _throughput_worker(
    pin: bool,
    engine_args_kwargs: dict,
    q: "multiprocessing.Queue[float]",
    v2_mode: bool = False,
) -> None:
    """Run throughput benchmark in a fresh spawn subprocess.

    Delegates to vllm/benchmarks/throughput.py main() using the random dataset,
    so the methodology matches the official benchmark.  Results are written to a
    temp JSON file and forwarded through the queue as tokens/s.

    v2_mode: when True, monkeypatches is_uva_available() to always return True
    so the v2 model runner's UVA buffers remain functional even when pin=False.
    This isolates the non-UVA pin_memory paths in v2.
    """
    import vllm.utils.platform_utils as pu
    from vllm.platforms import current_platform

    pu.is_pin_memory_available.cache_clear()
    pu.is_uva_available.cache_clear()
    type(current_platform).is_pin_memory_available = classmethod(lambda cls: pin)
    if v2_mode:
        pu.is_uva_available = lambda: True

    from vllm.benchmarks.throughput import add_cli_args
    from vllm.benchmarks.throughput import main as throughput_main

    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args([])

    for key, val in engine_args_kwargs.items():
        setattr(args, key, val)
    args.max_num_seqs = _THROUGHPUT_MAX_NUM_SEQS
    args.dataset_name = "random"
    args.input_len = _THROUGHPUT_INPUT_LEN
    args.output_len = _THROUGHPUT_OUTPUT_LEN
    # Nullify defaults that conflict with explicit input/output_len.
    args.random_input_len = None
    args.random_output_len = None
    args.random_prefix_len = None
    args.num_prompts = _THROUGHPUT_NUM_REQUESTS
    args.seed = 0
    args.disable_detokenize = True

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        tmp_path = f.name
    args.output_json = tmp_path

    throughput_main(args)

    with open(tmp_path) as f:
        results = json.load(f)
    q.put(results["tokens_per_second"])


def _run_throughput_benchmark(
    pin: bool,
    engine_args_kwargs: dict,
    v2_mode: bool = False,
) -> float:
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=_throughput_worker,
        args=(pin, engine_args_kwargs, q, v2_mode),
    )
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(
            f"Throughput benchmark subprocess (pin={pin}) exited with code {p.exitcode}"
        )
    return q.get()


def _latency_worker(
    pin: bool,
    engine_args_kwargs: dict,
    q: "multiprocessing.Queue[dict]",
    v2_mode: bool = False,
) -> None:
    """Run latency benchmark in a fresh spawn subprocess.

    Follows latency.py methodology: fixed batch of dummy token IDs, warmup
    iterations to reach steady state, then timed iterations reduced to avg
    and percentiles.  Results are written to a temp JSON file by latency_main
    and forwarded through the queue.
    """
    import vllm.utils.platform_utils as pu
    from vllm.platforms import current_platform

    pu.is_pin_memory_available.cache_clear()
    pu.is_uva_available.cache_clear()
    type(current_platform).is_pin_memory_available = classmethod(lambda cls: pin)
    if v2_mode:
        pu.is_uva_available = lambda: True

    from vllm.benchmarks.latency import add_cli_args
    from vllm.benchmarks.latency import main as latency_main

    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args([])

    for key, val in engine_args_kwargs.items():
        setattr(args, key, val)
    args.input_len = _LATENCY_INPUT_LEN
    args.output_len = _LATENCY_OUTPUT_LEN
    args.batch_size = _LATENCY_BATCH_SIZE
    args.num_iters_warmup = _LATENCY_WARMUP_ITERS
    args.num_iters = _LATENCY_BENCH_ITERS
    args.profile = False
    args.disable_detokenize = True

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        tmp_path = f.name
    args.output_json = tmp_path

    latency_main(args)

    with open(tmp_path) as f:
        results = json.load(f)
    q.put(results)


def _run_latency_benchmark(
    pin: bool,
    engine_args_kwargs: dict,
    v2_mode: bool = False,
) -> dict:
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=_latency_worker,
        args=(pin, engine_args_kwargs, q, v2_mode),
    )
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(
            f"Latency benchmark subprocess (pin={pin}) exited with code {p.exitcode}"
        )
    return q.get()


@pytest.mark.parametrize(
    "test_v2_runner",
    [
        pytest.param(False, id="v1"),
        pytest.param(True, id="v2"),
    ],
)
class TestPinnedMemory:
    """Verify pinned memory yields >= throughput vs unpinned via real vLLM inference."""

    def test_throughput(self, monkeypatch, test_v2_runner, model, max_model_len):
        """Benchmark throughput with pin_memory forced on then off.

        Delegates to vllm/benchmarks/throughput.py main() with the random
        dataset.  Each condition runs in an isolated spawn subprocess so both
        start from a cold CUDA context, giving an unbiased comparison.
        """
        monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1" if test_v2_runner else "0")

        engine_args_kwargs = dict(
            model=model,
            gpu_memory_utilization=0.88,
            max_model_len=max_model_len,
            enable_prefix_caching=False,
        )

        _skip_if_pin_memory_not_available(engine_args_kwargs)

        unpinned_tps = _run_throughput_benchmark(
            False, engine_args_kwargs, v2_mode=test_v2_runner
        )
        pinned_tps = _run_throughput_benchmark(
            True, engine_args_kwargs, v2_mode=test_v2_runner
        )

        pct_diff = (pinned_tps - unpinned_tps) / unpinned_tps * 100
        runner = "v2" if test_v2_runner else "v1"
        print(
            f"\n=== Throughput results ({runner} runner, {model}) ==="
            f"\npin_memory=True:  {pinned_tps:.1f} tok/s"
            f"\npin_memory=False: {unpinned_tps:.1f} tok/s"
            f"\nDifference: {pct_diff:+.1f}% (pinned vs unpinned)"
        )

        assert pinned_tps >= unpinned_tps * _THROUGHPUT_TOLERANCE, (
            f"Pinned throughput ({pinned_tps:.1f} tok/s) fell more than "
            f"{(1.0 - _THROUGHPUT_TOLERANCE) * 100:.1f}% below "
            f"unpinned ({unpinned_tps:.1f} tok/s)."
        )

    def test_latency(self, monkeypatch, test_v2_runner, model, max_model_len):
        """Benchmark per-batch latency with pin_memory forced on then off.

        Follows vllm/benchmarks/latency.py: fixed dummy-token batch, warmup
        iterations to reach steady state, then timed iterations reduced to avg
        and percentiles.  Subprocesses run serially so each gets a cold CUDA
        context without GPU memory pressure from the other run.
        """
        monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1" if test_v2_runner else "0")

        engine_args_kwargs = dict(
            model=model,
            gpu_memory_utilization=0.88,
            max_model_len=max_model_len,
            enable_prefix_caching=False,
        )

        _skip_if_pin_memory_not_available(engine_args_kwargs)

        unpinned = _run_latency_benchmark(
            False, engine_args_kwargs, v2_mode=test_v2_runner
        )
        pinned = _run_latency_benchmark(
            True, engine_args_kwargs, v2_mode=test_v2_runner
        )

        pct_diff = (
            (pinned["avg_latency"] - unpinned["avg_latency"])
            / unpinned["avg_latency"]
            * 100
        )
        runner = "v2" if test_v2_runner else "v1"
        print(
            f"\n=== Latency results ({runner} runner, {model}) ==="
            f"\npin_memory=True:  avg={pinned['avg_latency']:.3f}s"
            f"  p50={pinned['percentiles']['50']:.3f}s"
            f"  p99={pinned['percentiles']['99']:.3f}s"
            f"\npin_memory=False: avg={unpinned['avg_latency']:.3f}s"
            f"  p50={unpinned['percentiles']['50']:.3f}s"
            f"  p99={unpinned['percentiles']['99']:.3f}s"
            f"\nDifference: {pct_diff:+.1f}% (pinned vs unpinned)"
        )

        assert pinned["avg_latency"] <= unpinned["avg_latency"] * _LATENCY_TOLERANCE, (
            f"Pinned avg latency ({pinned['avg_latency']:.3f}s) exceeded "
            f"unpinned ({unpinned['avg_latency']:.3f}s) by more than "
            f"{(_LATENCY_TOLERANCE - 1.0) * 100:.1f}%."
        )


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--model", default=_DEFAULT_MODEL)
    _parser.add_argument("--max-model-len", type=int, default=_DEFAULT_MAX_MODEL_LEN)
    _, _remaining = _parser.parse_known_args()
    sys.exit(pytest.main([__file__] + _remaining))
