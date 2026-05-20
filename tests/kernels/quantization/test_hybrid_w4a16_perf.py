# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Performance regression tests for the hybrid W4A16 GEMM kernel.

Compares measured TFLOP/s against golden baselines stored in per-GPU JSON
files under ``golden/``.  A two-sided tolerance band catches both regressions
and unexpected improvements.

Usage::

    # rep=20 for testing, rep=50 for --write-golden.
    .venv/bin/python -m pytest tests/kernels/quantization/test_hybrid_w4a16_perf.py \\
        -v -s

    # Measure new baselines (overwrites golden/ in place):
    .venv/bin/python -m pytest tests/kernels/quantization/test_hybrid_w4a16_perf.py \\
        --write-golden -s

    # Include noisy/intermittent cases:
    .venv/bin/python -m pytest tests/kernels/quantization/test_hybrid_w4a16_perf.py \\
        --intermittent -v -s

Golden JSON schema is documented in ``golden/README.md``.
"""

from __future__ import annotations

import json
import math
import pathlib
import time
from typing import Any

import pytest
import torch

from tests.kernels.quantization.conftest import preload_golden
from vllm.platforms import current_platform

# ---------------------------------------------------------------------------
# GPU temperature reading
# ---------------------------------------------------------------------------


def _read_gpu_temp() -> float:
    """Return GPU edge temperature in degrees C, or NaN if unavailable."""
    try:
        import amdsmi

        amdsmi.amdsmi_init()
        devices = amdsmi.amdsmi_get_processor_handles()
        if not devices:
            return float("nan")
        return float(
            amdsmi.amdsmi_get_temp_metric(
                devices[0],
                amdsmi.AmdSmiTemperatureType.EDGE,
                amdsmi.AmdSmiTemperatureMetric.CURRENT,
            )
        )
    except Exception:
        return float("nan")


def _log_temp(config: Any, label: str) -> float:
    """Read and log temperature with a label. Returns temp in C."""
    from tests.kernels.quantization.conftest import get_temp_log

    t = _read_gpu_temp()
    get_temp_log(config).append((time.monotonic(), label, t))
    return t


# ---------------------------------------------------------------------------
# Constants -- single source of truth
# ---------------------------------------------------------------------------

SHAPES: list[dict[str, Any]] = [
    # google/gemma-2b-AWQ
    {
        "in_features": 2048,
        "out_features": 32768,
        "group_size": 128,
        "comment": "gemma-2b gate_up_proj",
    },
    {
        "in_features": 16384,
        "out_features": 2048,
        "group_size": 128,
        "comment": "gemma-2b down_proj",
    },
    {
        "in_features": 2048,
        "out_features": 2560,
        "group_size": 128,
        "comment": "gemma-2b qkv_proj",
    },
    {
        "in_features": 2048,
        "out_features": 2048,
        "group_size": 128,
        "comment": "gemma-2b o_proj",
    },
    # hf-kernel: shapes with K > 32768 that dispatch to wvSplitK_int4_hf_
    # instead of wvSplitK_int4_hf_sml_ at batch=1.
    {
        "in_features": 38912,
        "out_features": 2048,
        "group_size": 128,
        "comment": "hf-kernel, K_packed % 4096 != 0",
    },
    {
        "in_features": 49152,
        "out_features": 2048,
        "group_size": 128,
        "comment": "hf-kernel, K_packed % 4096 == 0",
    },
    {
        "in_features": 49152,
        "out_features": 4096,
        "group_size": 128,
        "comment": "hf-kernel, K_packed % 4096 == 0",
    },
    # Qwen/Qwen3-4B
    {
        "in_features": 2560,
        "out_features": 3840,
        "group_size": 128,
        "comment": "Qwen3-4B qkv_proj",
    },
    {
        "in_features": 2560,
        "out_features": 2560,
        "group_size": 128,
        "comment": "Qwen3-4B o_proj",
    },
    {
        "in_features": 2560,
        "out_features": 19456,
        "group_size": 128,
        "comment": "Qwen3-4B gate_up_proj",
    },
    {
        "in_features": 9728,
        "out_features": 2560,
        "group_size": 128,
        "comment": "Qwen3-4B down_proj",
    },
    # Intel/Qwen3.5-35B-A3B-int4-AutoRound -- exercises the gfx11 K=4096 N=1
    # wvSplitK_int4 dispatch branch added by this PR (W=16, AC=32, YT=1, UN=4).
    {
        "in_features": 4096,
        "out_features": 2048,
        "group_size": 128,
        "comment": "Qwen3.5-35B-A3B GDN out_proj",
    },
    # Qwen/Qwen2.5-7B-Instruct
    {
        "in_features": 3584,
        "out_features": 4608,
        "group_size": 128,
        "comment": "Qwen2.5-7B qkv_proj",
    },
    {
        "in_features": 3584,
        "out_features": 3584,
        "group_size": 128,
        "comment": "Qwen2.5-7B o_proj",
    },
    {
        "in_features": 3584,
        "out_features": 37888,
        "group_size": 128,
        "comment": "Qwen2.5-7B gate_up_proj",
    },
    {
        "in_features": 18944,
        "out_features": 3584,
        "group_size": 128,
        "comment": "Qwen2.5-7B down_proj",
    },
    # W4-L2-cache-boundary
    {
        "in_features": 8192,
        "out_features": 512,
        "group_size": 128,
        "comment": "L2 2MiB at",
    },
    {
        "in_features": 8320,
        "out_features": 512,
        "group_size": 128,
        "comment": "L2 2MiB above",
    },
]

PROVIDERS = ["hybrid-w4a16", "hybrid-w4a16-zp"]
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
TFLOPS_TOLERANCE_PCT = {  # [low, high] allowed deviation from golden
    "default": [-8, 8],
    "hybrid_triton_w4a16": [-80, 80],  # TODO: Why does it vary so much?
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = pathlib.Path(__file__).parent
_GOLDEN_DIR = _HERE / "golden"

# ---------------------------------------------------------------------------
# Shape key helper
# ---------------------------------------------------------------------------

ShapeKey = tuple[int, int, int]  # (in_features, out_features, group_size)


def _shape_key(s: dict[str, Any]) -> ShapeKey:
    return (s["in_features"], s["out_features"], s["group_size"])


assert len({_shape_key(s) for s in SHAPES}) == len(SHAPES), "duplicate in SHAPES"
_SHAPE_KEY_SET: set[ShapeKey] = {_shape_key(s) for s in SHAPES}
_BATCH_SIZE_SET: set[int] = set(BATCH_SIZES)

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------


def _get_gcn_arch() -> str:
    """Return the GCN architecture string, or '' on non-ROCm."""
    if not current_platform.is_rocm():
        return ""
    try:
        from vllm.platforms.rocm import _GCN_ARCH

        return _GCN_ARCH
    except ImportError:
        return ""


# ---------------------------------------------------------------------------
# Golden-data loading
# ---------------------------------------------------------------------------


def _load_golden(gcn_arch: str) -> tuple[str | None, dict[str, Any] | None]:
    """Find and load the golden JSON for *gcn_arch*.

    Returns ``(filename, data)`` or ``(None, None)`` when no match.
    """
    for path in sorted(_GOLDEN_DIR.glob("hybrid_w4a16_*.json")):
        data = json.loads(path.read_text())
        if gcn_arch.startswith(data.get("gpu", "")):
            return path.name, data
    return None, None


def _validate_golden(data: dict[str, Any]) -> None:
    """Raise on rogue shapes or batch sizes in the golden file."""
    for shape in data.get("shapes", []):
        sk = _shape_key(shape)
        if "skip" not in shape and sk not in _SHAPE_KEY_SET:
            raise ValueError(
                f"Golden file contains shape {sk} not in SHAPES. "
                "Remove it or add it to SHAPES in the test file."
            )
        for prov in shape.get("providers", []):
            for bl in prov.get("baselines", []):
                bs = bl.get("batch_size")
                if bs is not None and bs not in _BATCH_SIZE_SET:
                    raise ValueError(
                        f"Golden file contains batch_size={bs} "
                        f"(shape {sk}) not in BATCH_SIZES."
                    )


# ---------------------------------------------------------------------------
# Weight preparation (ported from benchmark script)
# ---------------------------------------------------------------------------


def prepare_hybrid_weights(
    K: int, N: int, group_size: int, device: str = "cuda"
) -> dict[str, torch.Tensor]:
    """Create random packed weights for benchmarking."""
    num_groups = K // group_size

    w_q_skinny_i32 = torch.randint(
        0, 2**31, (N, K // 8), dtype=torch.int32, device=device
    )
    w_q_skinny = w_q_skinny_i32.view(torch.int8).contiguous()
    w_s_skinny = torch.randn(N, num_groups, dtype=torch.float16, device=device) * 0.01
    w_zp = torch.randint(0, 16, (N, num_groups), dtype=torch.int32, device=device).to(
        torch.float16
    )

    return {
        "w_q_skinny": w_q_skinny,
        "w_s_skinny": w_s_skinny,
        "w_q_skinny_i32": w_q_skinny_i32,
        "w_zp": w_zp,
    }


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------


COOL_DOWN_DELAY_S = 2.0
STEADY_STATE_TEMP_C = 60.0
COOL_DOWN_POLL_INTERVAL_S = 0.25


def _cool_down(config: Any, test_id: str) -> None:
    """Wait until the GPU is cool enough, bounded by COOL_DOWN_DELAY_S."""
    temp = _log_temp(config, f"{test_id}:pre-sleep")
    deadline = time.monotonic() + COOL_DOWN_DELAY_S
    while math.isfinite(temp) and temp > STEADY_STATE_TEMP_C:
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            break
        time.sleep(min(COOL_DOWN_POLL_INTERVAL_S, remaining_s))
        temp = _read_gpu_temp()
    _log_temp(config, f"{test_id}:post-sleep")


def measure_tflops(
    M: int,
    weights: dict[str, torch.Tensor],
    K: int,
    N: int,
    group_size: int,
    provider: str,
) -> tuple[str, float]:
    """Run the kernel and return (kernel label, median TFLOP/s)."""
    from vllm.model_executor.kernels.linear.mixed_precision.hybrid_w4a16 import (
        _hybrid_w4a16_apply_impl,
    )
    from vllm.triton_utils import triton
    from vllm.utils.platform_utils import num_compute_units

    device = "cuda"
    dtype = torch.float16
    a = torch.randn((M, K), device=device, dtype=dtype)

    cu_count = num_compute_units()
    use_zp = provider == "hybrid-w4a16-zp"

    def run():
        return _hybrid_w4a16_apply_impl(
            a,
            weights["w_q_skinny"],
            weights["w_s_skinny"],
            weights["w_q_skinny_i32"],
            weights["w_zp"] if use_zp else None,
            None,  # bias
            cu_count,
            group_size,
        )

    ms = triton.testing.do_bench_cudagraph(run, quantiles=[0.5])
    tflops = (2 * M * N * K) * 1e-12 / (ms * 1e-3)

    # Detect which kernel path was taken by observing record_function labels.
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        run()
    labels = [
        e.name
        for e in prof.events()
        if e.name.startswith(("wvsplitk_int4", "hybrid_triton_w4a16"))
    ]
    kernel = labels[0].split()[0] if labels else "unknown"

    return kernel, tflops


# ---------------------------------------------------------------------------
# Parametrize helpers
# ---------------------------------------------------------------------------


def _make_params() -> list[pytest.param]:
    """Build the SHAPES x PROVIDERS parameter list."""
    params = []
    for shape in sorted(SHAPES, key=_shape_key):
        for prov in PROVIDERS:
            k = shape["in_features"]
            n = shape["out_features"]
            g = shape["group_size"]
            test_id = f"i{k}-o{n}-g{g}-{prov}"
            params.append(pytest.param(shape, prov, id=test_id))
    return params


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def _find_shape_in_golden(
    golden: dict[str, Any], key: ShapeKey
) -> dict[str, Any] | None:
    for s in golden.get("shapes", []):
        if _shape_key(s) == key:
            return s
    return None


def _find_provider_in_shape(
    shape_data: dict[str, Any], provider: str
) -> dict[str, Any] | None:
    for p in shape_data.get("providers", []):
        if p.get("provider") == provider:
            return p
    return None


def _find_baseline(
    provider_data: dict[str, Any], batch_size: int
) -> dict[str, Any] | None:
    for bl in provider_data.get("baselines", []):
        if bl.get("batch_size") == batch_size:
            return bl
    return None


# ---------------------------------------------------------------------------
# Measured-results collector for --write-golden
# ---------------------------------------------------------------------------


def _record_measurement(
    config: Any,
    gpu: str,
    shape: dict[str, Any],
    provider: str,
    batch_size: int,
    kernel: str,
    tflops: float,
    annotations: dict[str, Any] | None = None,
) -> None:
    """Append a measurement into the session-scoped collector."""
    from tests.kernels.quantization.conftest import get_measured_results

    results = get_measured_results(config)
    shapes_list = results.setdefault(gpu, [])
    sk = _shape_key(shape)

    # Find or create shape entry
    shape_entry = None
    for s in shapes_list:
        if _shape_key(s) == sk:
            shape_entry = s
            break
    if shape_entry is None:
        shape_entry = {
            "in_features": sk[0],
            "out_features": sk[1],
            "group_size": sk[2],
            "comment": shape.get("comment", ""),
            "providers": [],
        }
        shapes_list.append(shape_entry)
        # Keep sorted
        shapes_list.sort(key=_shape_key)

    # Find or create provider entry
    prov_entry = _find_provider_in_shape(shape_entry, provider)
    if prov_entry is None:
        prov_entry = {"provider": provider, "baselines": []}
        shape_entry["providers"].append(prov_entry)

    # Replace existing baseline for this batch_size, if any.
    for i, bl_existing in list(enumerate(prov_entry["baselines"])):
        if bl_existing.get("batch_size") == batch_size:
            prov_entry["baselines"].pop(i)

    # Build baseline entry
    bl: dict[str, Any] = {
        "batch_size": batch_size,
        "kernel": kernel,
        "tflops": round(tflops, 4),
    }
    if annotations:
        bl.update(annotations)
    prov_entry["baselines"].append(bl)
    # Keep batch_sizes sorted
    prov_entry["baselines"].sort(key=lambda b: b["batch_size"])


def _record_skip(
    config: Any,
    gpu: str,
    shape: dict[str, Any],
    provider: str,
    batch_size: int,
    reason: str,
) -> None:
    """Record a skipped entry in measured results."""
    _record_measurement(
        config,
        gpu,
        shape,
        provider,
        batch_size,
        "",
        0.0,
        annotations={"skip": reason},
    )
    # Remove kernel and tflops fields since they're not meaningful
    from tests.kernels.quantization.conftest import get_measured_results

    shapes_list = get_measured_results(config)[gpu]
    for s in shapes_list:
        if _shape_key(s) == _shape_key(shape):
            for p in s["providers"]:
                if p["provider"] == provider:
                    for bl in p["baselines"]:
                        if bl["batch_size"] == batch_size and "skip" in bl:
                            bl.pop("kernel", None)
                            bl.pop("tflops", None)


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _warm_up_gpu():
    """Run a throwaway measurement pass to bring the GPU to steady-state temp."""
    if current_platform.is_rocm():
        temp = _read_gpu_temp()
        print(f"GPU temperature: {temp:.0f}\u00b0C")
        if temp < STEADY_STATE_TEMP_C:
            print("Warming up GPU...")
            shape = sorted(SHAPES, key=_shape_key)[0]
            K, N, gs = shape["in_features"], shape["out_features"], shape["group_size"]
            weights = prepare_hybrid_weights(K, N, gs)
            for bs in BATCH_SIZES:
                measure_tflops(
                    bs, weights, K, N, gs, PROVIDERS[0]
                )  # warmup; ignore return
            del weights
            time.sleep(COOL_DOWN_DELAY_S)
    yield


@pytest.mark.benchmark
@pytest.mark.parametrize("shape,provider", _make_params())
def test_hybrid_w4a16_perf(
    shape: dict[str, Any],
    provider: str,
    request: pytest.FixtureRequest,
    _warm_up_gpu: None,
) -> None:
    # ---- gate ----
    gcn_arch = _get_gcn_arch()
    if not current_platform.is_rocm() or not gcn_arch.startswith("gfx1151"):
        pytest.skip("ROCm gfx1151 only")

    # Pre-load existing golden so subset runs do not drop unmeasured entries.
    preload_golden(request.config, gcn_arch)

    measure_mode = request.config.getoption("--write-golden", default=False)
    intermittent_mode = (
        request.config.getoption("--intermittent", default=False) or measure_mode
    )

    # ---- load golden ----
    golden_fname, golden = _load_golden(gcn_arch)
    if golden is not None:
        _validate_golden(golden)

    if golden is None and not measure_mode:
        pytest.skip(f"No golden baselines for {gcn_arch}")

    sk = _shape_key(shape)
    K, N, group_size = sk

    # ---- shape-level skip ----
    if golden is not None:
        shape_data = _find_shape_in_golden(golden, sk)
        if shape_data is not None and "skip" in shape_data:
            pytest.skip(shape_data["skip"])
    else:
        shape_data = None

    # ---- provider-level skip ----
    if shape_data is not None:
        prov_data = _find_provider_in_shape(shape_data, provider)
        if prov_data is not None and "skip" in prov_data:
            pytest.skip(prov_data["skip"])
    else:
        prov_data = None

    # If shape not in golden, skip in normal mode
    if golden is not None and shape_data is None and not measure_mode:
        pytest.skip(f"Shape {sk} not yet measured on {gcn_arch}")
    if (
        golden is not None
        and shape_data is not None
        and prov_data is None
        and not measure_mode
    ):
        pytest.skip(f"Provider {provider} not yet measured for shape {sk}")

    # ---- cooldown + temperature log ----
    test_id = f"i{K}-o{N}-g{group_size}-{provider}"
    _cool_down(request.config, test_id)

    # ---- allocate weights once ----
    weights = prepare_hybrid_weights(K, N, group_size)

    # ---- iterate batch sizes ----
    failures: list[str] = []
    xpass_list: list[str] = []
    exit_first = getattr(request.config.option, "exitfirst", False)

    for bs in BATCH_SIZES:
        # Look up baseline
        bl_entry: dict[str, Any] | None = None
        if prov_data is not None:
            bl_entry = _find_baseline(prov_data, bs)

        # ---- skip annotations ----
        if bl_entry is not None and "skip" in bl_entry:
            reason = bl_entry["skip"]
            print(f"  batch_size={bs}: SKIP ({reason})")
            if measure_mode:
                _record_skip(request.config, gcn_arch, shape, provider, bs, reason)
            continue

        # ---- intermittent handling ----
        is_intermittent = bl_entry is not None and bl_entry.get("intermittent", False)
        if is_intermittent and not intermittent_mode:
            print(f"  batch_size={bs}: SKIP (intermittent)")
            continue

        # ---- measure ----
        _log_temp(request.config, f"{test_id}:bs{bs}:pre")
        kernel, tflops = measure_tflops(bs, weights, K, N, group_size, provider)
        post_temp = _log_temp(request.config, f"{test_id}:bs{bs}:post")
        temp_tag = f" [{post_temp:.0f}\u00b0C]"

        if measure_mode:
            # Carry forward annotations
            annot: dict[str, Any] = {}
            if bl_entry is not None:
                for key in ("expected_failure", "intermittent"):
                    if key in bl_entry:
                        annot[key] = bl_entry[key]
            _record_measurement(
                request.config,
                gcn_arch,
                shape,
                provider,
                bs,
                kernel,
                tflops,
                annot,
            )
            print(
                f"  batch_size={bs}: {tflops:.2f} TFLOP/s"
                f" (measured) [{kernel}]{temp_tag}"
            )
            continue

        if bl_entry is None:
            print(
                f"  batch_size={bs}: {tflops:.2f} TFLOP/s "
                f"(no golden value, skipping assertion) [{kernel}]{temp_tag}"
            )
            continue

        expected_tflops = bl_entry["tflops"]
        expected_kernel = bl_entry.get("kernel")
        assert expected_kernel, (
            f"Golden kernel= for batch_size={bs} is missing. "
            "Run --write-golden to populate."
        )

        # ---- sanity check golden value ----
        assert expected_tflops > 0 and math.isfinite(expected_tflops), (
            f"Golden tflops={expected_tflops} for batch_size={bs} is invalid. "
            "Run --write-golden to populate."
        )

        # ---- kernel mismatch check ----
        if kernel != expected_kernel:
            direction = "regression" if tflops < expected_tflops else "improvement"
            delta_pct = (tflops - expected_tflops) / expected_tflops * 100
            msg = (
                f"  batch_size={bs}: kernel mismatch: "
                f"expected {expected_kernel}, got {kernel}; "
                f"{tflops:.2f} TFLOP/s "
                f"(expected {expected_tflops:.2f}, {delta_pct:+.1f}% "
                f"{direction}){temp_tag}"
            )
            print(msg)
            failures.append(msg)
            continue

        # ---- tolerance band ----
        tflops_tolerance_pct = TFLOPS_TOLERANCE_PCT.get(expected_kernel)
        if tflops_tolerance_pct is None:
            tflops_tolerance_pct = TFLOPS_TOLERANCE_PCT["default"]
        lo = expected_tflops * (1 + tflops_tolerance_pct[0] / 100)
        hi = expected_tflops * (1 + tflops_tolerance_pct[1] / 100)
        in_band = lo <= tflops <= hi

        has_xfail = "expected_failure" in bl_entry

        if has_xfail:
            if in_band:
                xpass_list.append(
                    f"  batch_size={bs}: {tflops:.2f} TFLOP/s is now within "
                    f"band [{lo:.2f}, {hi:.2f}] -- remove expected_failure "
                    f"annotation: {bl_entry['expected_failure']}"
                )
                print(
                    f"  batch_size={bs}: {tflops:.2f} TFLOP/s "
                    f"(expected {expected_tflops:.2f} "
                    f"+ {tflops_tolerance_pct}%) XPASS [{kernel}]{temp_tag}"
                )
            else:
                print(
                    f"  batch_size={bs}: {tflops:.2f} TFLOP/s "
                    f"(expected {expected_tflops:.2f} "
                    f"+ {tflops_tolerance_pct}%) "
                    f"XFAIL: {bl_entry['expected_failure']} [{kernel}]{temp_tag}"
                )
            continue

        if in_band:
            print(
                f"  batch_size={bs}: {tflops:.2f} TFLOP/s "
                f"(expected {expected_tflops:.2f} "
                f"+ {tflops_tolerance_pct}%) PASS [{kernel}]{temp_tag}"
            )
        else:
            direction = "regression" if tflops < lo else "improvement"
            delta_pct = (tflops - expected_tflops) / expected_tflops * 100
            msg = (
                f"  batch_size={bs}: {tflops:.2f} TFLOP/s "
                f"(expected {expected_tflops:.2f} "
                f"+ {tflops_tolerance_pct}%) "
                f"FAIL ({direction}, {delta_pct:+.1f}%) [{kernel}]{temp_tag}"
            )
            print(msg)
            if exit_first:
                if direction == "regression":
                    pytest.fail(
                        f"Performance regression at batch_size={bs}: "
                        f"{tflops:.2f} < {lo:.2f} TFLOP/s. "
                        "Run --write-golden to update."
                    )
                else:
                    pytest.fail(
                        f"Performance improved at batch_size={bs}: "
                        f"{tflops:.2f} > {hi:.2f} TFLOP/s. "
                        "Run --write-golden to update baselines."
                    )
            failures.append(msg)

    # ---- report xpass ----
    if xpass_list:
        raise AssertionError(
            "Unexpected passes -- remove expected_failure annotations:\n"
            + "\n".join(xpass_list)
        )

    # ---- report failures ----
    if failures:
        raise AssertionError(
            f"{len(failures)} batch size(s) out of tolerance band. "
            "Run --write-golden to update.\n" + "\n".join(failures)
        )
