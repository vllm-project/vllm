# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Real-checkpoint smoke tests for Nano-Nemotron-VL video batching.

Keep this file small. Pure unit coverage in
``tests/models/multimodal/processing/`` already pins the same-resolution
segment packer, per-video output reconstruction, and the processor
``BatchFeature`` boundary. These tests only load a real
``NemotronH_Nano_VL_V2`` checkpoint and verify the two runtime paths that
the scheduler can exercise:

* packed same-resolution video tensors,
* list-shaped varying-resolution video tensors.

Set ``VLLM_NEMOTRON_VL_V2_PATH`` to a local checkpoint directory and have
at least ``VLLM_TEST_TP`` (default 8) accelerator devices available.
Otherwise these tests skip cleanly.
"""

from __future__ import annotations

import os

import pytest
import torch

MODEL_PATH_ENV = "VLLM_NEMOTRON_VL_V2_PATH"
MODEL_PATH = os.environ.get(MODEL_PATH_ENV, "")
TP = int(os.environ.get("VLLM_TEST_TP", "8"))

SEED = 42
REL_TOL_SAME_RES = 3e-2
REL_TOL_DYNAMIC = 5e-3


def _accelerator_device_count() -> int:
    """Safe probe; never raises at import time."""
    try:
        return torch.accelerator.device_count()
    except Exception:
        return 0


needs_model = pytest.mark.skipif(
    not MODEL_PATH or not os.path.isdir(MODEL_PATH),
    reason=(
        f"Set {MODEL_PATH_ENV} to a local NemotronH_Nano_VL_V2 checkpoint "
        "directory to enable this integration test."
    ),
)
needs_gpus = pytest.mark.skipif(
    _accelerator_device_count() < TP,
    reason=(
        f"Need at least TP={TP} accelerator devices (override with VLLM_TEST_TP=<n>)."
    ),
)


# ---------------------------------------------------------------------------
# Module-level helpers. Must be picklable so `collective_rpc` can ship the
# per-test worker functions into each TP worker process.
# ---------------------------------------------------------------------------


def _sanity_check_model(model) -> list[str]:
    failures: list[str] = []
    if type(model).__name__ != "NemotronH_Nano_VL_V2":
        failures.append(f"wrong model class {type(model).__name__}")
    if not getattr(model, "handles_video_batching_internally", False):
        failures.append(
            "model missing handles_video_batching_internally=True; the "
            "scheduler would not exercise the batched encoder path"
        )
    return failures


def _model_ctx(worker):
    """Return ((model, device, T, patch_size, H, W), []) or (None, failures)."""
    model = worker.model_runner.get_model()
    failures = _sanity_check_model(model)
    if failures:
        return None, failures
    device = next(model.parameters()).device
    T = int(model.video_temporal_patch_size)
    patch_size = int(model.patch_size)
    H = W = patch_size * 16
    return (model, device, T, patch_size, H, W), []


def _make_video(
    device: torch.device, nf: int, H: int, W: int, seed: int
) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(nf, 3, H, W, dtype=torch.bfloat16, device=device, generator=g)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max-abs relative difference; 1e-9 guards against a zero reference."""
    return ((a.float() - b.float()).abs().max() / (a.float().abs().max() + 1e-9)).item()


def _same_res_input(videos: list[torch.Tensor]) -> dict:
    """Packed same-resolution input, matching the scheduler shape."""
    device = videos[0].device
    return {
        "pixel_values_flat": torch.cat(videos, dim=0),
        "num_patches": torch.tensor(
            [v.shape[0] for v in videos], dtype=torch.long, device=device
        ),
    }


def _varying_res_input(videos: list[torch.Tensor]) -> dict:
    """List-shaped input that triggers the varying-resolution path."""
    device = videos[0].device
    return {
        "pixel_values_flat": list(videos),
        "num_patches": torch.tensor(
            [v.shape[0] for v in videos], dtype=torch.long, device=device
        ),
    }


def _full_same_res_input(videos: list[torch.Tensor]) -> dict:
    """Full video input accepted by `_process_video_input`."""
    device = videos[0].device
    nfs = [v.shape[0] for v in videos]
    return {
        "type": "pixel_values_videos",
        "pixel_values_flat": torch.cat(videos, dim=0),
        "num_patches": torch.tensor(nfs, dtype=torch.long, device=device),
        "frames_indices": torch.cat(
            [torch.arange(nf, dtype=torch.long, device=device) for nf in nfs]
        ),
        "frame_duration_ms": torch.full(
            (len(videos),), 500.0, dtype=torch.float32, device=device
        ),
    }


def _compare_tensor_lists(
    label: str,
    refs: list[torch.Tensor],
    actual,
    tol: float,
    failures: list[str],
    info: list[str],
) -> None:
    if len(actual) != len(refs):
        failures.append(f"{label}: expected {len(refs)} outputs, got {len(actual)}")
        return

    for i, ref in enumerate(refs):
        if ref.shape != actual[i].shape:
            failures.append(
                f"{label}[{i}]: shape mismatch ref={tuple(ref.shape)} "
                f"actual={tuple(actual[i].shape)}"
            )
            continue

        rel_diff = _rel(ref, actual[i])
        info.append(f"{label}[{i}] rel_diff={rel_diff:.4g}")
        if rel_diff > tol:
            failures.append(f"{label}[{i}]: rel_diff {rel_diff:.4g} > {tol}")


def _check_bit_exact(label: str, ref, actual, idx: int, failures: list[str]) -> None:
    if len(ref) <= idx or len(actual) <= idx:
        failures.append(
            f"{label}: expected output index {idx}, got {len(ref)}/{len(actual)}"
        )
        return
    if ref[idx].shape != actual[idx].shape:
        failures.append(
            f"{label}: output {idx} shape changed across companion swap: "
            f"{tuple(ref[idx].shape)} vs {tuple(actual[idx].shape)}"
        )
        return
    if not torch.equal(ref[idx], actual[idx]):
        diff = (ref[idx].float() - actual[idx].float()).abs().max().item()
        failures.append(
            f"{label}: output {idx} changed when only the companion pixels changed "
            f"(max abs diff {diff:.4g})"
        )


# ---------------------------------------------------------------------------
# Per-test worker functions.
# ---------------------------------------------------------------------------


def _w_same_resolution_video_batching(worker):
    failures: list[str] = []
    info: list[str] = []
    ctx, fs = _model_ctx(worker)
    if fs:
        return False, fs, info
    model, device, T, _patch_size, H, W = ctx
    if T <= 1:
        info.append("skipped: video_temporal_patch_size <= 1")
        return True, failures, info

    videos = [
        _make_video(device, 8, H, W, seed=SEED),
        _make_video(device, 8, H, W, seed=SEED + 1),
    ]
    videos_alt = [
        _make_video(device, 8, H, W, seed=SEED + 100),
        _make_video(device, 8, H, W, seed=SEED + 101),
    ]

    with torch.no_grad():
        solos = [
            model._extract_video_embeddings_temporal(_same_res_input([v]))[0]
            for v in videos
        ]
        batched = model._extract_video_embeddings_temporal(_same_res_input(videos))

        process_solos = [
            model._process_video_input(_full_same_res_input([v]))[0] for v in videos
        ]
        process_batched = model._process_video_input(_full_same_res_input(videos))

        ab = model._extract_video_embeddings_temporal(_same_res_input(videos))
        same_a_alt_b = model._extract_video_embeddings_temporal(
            _same_res_input([videos[0], videos_alt[1]])
        )
        alt_a_same_b = model._extract_video_embeddings_temporal(
            _same_res_input([videos_alt[0], videos[1]])
        )

    _compare_tensor_lists(
        "same_res_extract", solos, batched, REL_TOL_SAME_RES, failures, info
    )
    _compare_tensor_lists(
        "same_res_process",
        process_solos,
        process_batched,
        REL_TOL_SAME_RES,
        failures,
        info,
    )
    _check_bit_exact("same_res_companion", ab, same_a_alt_b, 0, failures)
    _check_bit_exact("same_res_companion", ab, alt_a_same_b, 1, failures)
    return not failures, failures, info


def _w_varying_resolution_video_batching(worker):
    failures: list[str] = []
    info: list[str] = []
    ctx, fs = _model_ctx(worker)
    if fs:
        return False, fs, info
    model, device, T, patch_size, _H, _W = ctx
    if T <= 1:
        info.append("skipped: video_temporal_patch_size <= 1")
        return True, failures, info

    hidden_size = model.config.text_config.hidden_size

    def run_dynamic(videos: list[torch.Tensor]):
        return model._extract_video_embeddings_temporal_dynamic(
            pixel_values=videos,
            num_frames_per_video=[v.shape[0] for v in videos],
            hidden_size=hidden_size,
            T=T,
            patch_size=patch_size,
        )

    def run_dispatch(videos: list[torch.Tensor]):
        return model._extract_video_embeddings_temporal(_varying_res_input(videos))

    # These two videos are individually under the 32768-patch dynamic
    # budget, but together exceed it. That keeps the integration test
    # compact while still proving the whole-video microbatch split.
    H1, W1 = patch_size * 16, patch_size * 16
    H2, W2 = patch_size * 20, patch_size * 16
    boundary_videos = [
        _make_video(device, 96, H1, W1, seed=SEED + 10),
        _make_video(device, 40, H2, W2, seed=SEED + 11),
    ]

    # A small fixed-shape pair is the exact leakage probe: changing only
    # companion pixels must not perturb the queried video's output.
    companion_a = _make_video(device, 8, H1, W1, seed=SEED + 20)
    companion_a_alt = _make_video(device, 8, H1, W1, seed=SEED + 21)
    companion_b = _make_video(device, 6, H2, W2, seed=SEED + 22)
    companion_b_alt = _make_video(device, 6, H2, W2, seed=SEED + 23)

    with torch.no_grad():
        solos = [run_dynamic([v])[0] for v in boundary_videos]
        batched_direct = run_dynamic(boundary_videos)
        batched_dispatch = run_dispatch(boundary_videos)

        ab = run_dispatch([companion_a, companion_b])
        same_a_alt_b = run_dispatch([companion_a, companion_b_alt])
        alt_a_same_b = run_dispatch([companion_a_alt, companion_b])

    _compare_tensor_lists(
        "dynamic_direct", solos, batched_direct, REL_TOL_DYNAMIC, failures, info
    )
    _compare_tensor_lists(
        "dynamic_dispatch", solos, batched_dispatch, REL_TOL_DYNAMIC, failures, info
    )
    _check_bit_exact("dynamic_companion", ab, same_a_alt_b, 0, failures)
    _check_bit_exact("dynamic_companion", ab, alt_a_same_b, 1, failures)
    return not failures, failures, info


# ---------------------------------------------------------------------------
# Fixture + shared assertion helper
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nemotron_nano_vl_v2_llm():
    from vllm import LLM

    # collective_rpc needs to ship a callable into the worker; vLLM only
    # allows pickle-based serialization when this env var is set.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=TP,
        dtype="bfloat16",
        max_model_len=32768,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"video": 4, "audio": 1, "image": 16},
        mamba_ssm_cache_dtype="float32",
    )
    yield llm
    del llm


def _assert_all_workers_pass(results, label: str) -> None:
    """Fail if any TP worker reports failures."""
    aggregated: list[str] = []
    for rank, result in enumerate(results):
        ok, failures, info = result
        for line in info:
            print(f"[{label} rank={rank}] {line}")
        if not ok:
            aggregated.append(f"rank {rank}: " + " | ".join(failures))

    assert not aggregated, (
        f"{label} failed on {len(aggregated)}/{len(results)} worker(s):\n  "
        + "\n  ".join(aggregated)
    )


# ---------------------------------------------------------------------------
# Public tests -- GPU + checkpoint required
# ---------------------------------------------------------------------------


@needs_model
@needs_gpus
def test_video_same_resolution_batching(nemotron_nano_vl_v2_llm):
    """Packed same-resolution videos match solo references on real weights."""
    _assert_all_workers_pass(
        nemotron_nano_vl_v2_llm.collective_rpc(_w_same_resolution_video_batching),
        "same_resolution_video_batching",
    )


@needs_model
@needs_gpus
def test_video_varying_resolution_batching(nemotron_nano_vl_v2_llm):
    """List-shaped varying-resolution videos route and batch correctly."""
    _assert_all_workers_pass(
        nemotron_nano_vl_v2_llm.collective_rpc(_w_varying_resolution_video_batching),
        "varying_resolution_video_batching",
    )
