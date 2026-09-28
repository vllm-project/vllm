# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the tensor-descriptor (TD) load path in the Mamba2 SSD
prefill kernels _chunk_scan_fwd_kernel and _chunk_state_fwd_kernel, gated on
VLLM_TRITON_USE_TD.

Every test forces TD on or off explicitly and compares against the pointer path
rather than the naive torch reference, which is tight enough to catch a wrong
tile offset or transpose; accuracy against the reference is covered by
tests/kernels/mamba/test_mamba_ssm_ssd.py. Launches are spied on, so a silent
fallback to the pointer path fails the test instead of trivially matching.

Inputs follow the mamba_mixer2 prefill call: x, B and C are views into one xBC
buffer (row stride wider than the tile), D is a per-head scalar, z is None.
Ragged sequence lengths put partial chunks and in-chunk sequence boundaries into
every case, which is where the descriptor zero-fill replaces the masks.
"""

import pytest
import torch

import vllm.model_executor.layers.mamba.ops.ssd_chunk_scan as ssd_chunk_scan
import vllm.model_executor.layers.mamba.ops.ssd_chunk_state as ssd_chunk_state
from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined_varlen,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl
from vllm.v1.attention.backends.mamba2_attn import compute_varlen_chunk_metadata

DEVICE = current_platform.device_type

pytestmark = pytest.mark.skipif(
    not hasattr(tl, "make_tensor_descriptor") or not current_platform.is_xpu(),
    reason="The SSD TD path needs tl.make_tensor_descriptor (Triton >= 3.6) and "
    "is validated on XPU only; elsewhere it is opt-in and untested",
)

# (nheads, headdim, dstate, ngroups, chunk_size): Nemotron-3-Nano-4B at TP=1,
# Nemotron-3-Ultra at TP=8 (per rank), and dstate > 128, which takes the
# K-looped prev_states branch of _chunk_scan_fwd_kernel.
SHAPES = [
    (8, 80, 128, 8, 256),
    (8, 64, 128, 1, 128),
    (4, 64, 256, 1, 128),
]
SEQLENS = [(300, 17, 500), (1, 127, 129, 256)]


def _spy_use_td(monkeypatch) -> dict[str, list[bool]]:
    """Record the USE_TD constexpr of every launch of both kernels."""
    calls: dict[str, list[bool]] = {"scan": [], "state": []}
    for name, kernel in (
        ("scan", ssd_chunk_scan._chunk_scan_fwd_kernel),
        ("state", ssd_chunk_state._chunk_state_fwd_kernel),
    ):

        def spy(*args, _run=kernel.run, _calls=calls[name], **kwargs):
            _calls.append(kwargs["USE_TD"])
            return _run(*args, **kwargs)

        monkeypatch.setattr(kernel, "run", spy)
    return calls


def _make_inputs(nheads, headdim, dstate, ngroups, seqlens, dtype):
    torch.manual_seed(0)
    T = sum(seqlens)
    xBC = torch.randn(
        T, nheads * headdim + 2 * ngroups * dstate, device=DEVICE, dtype=dtype
    )
    x, B, C = torch.split(
        xBC, [nheads * headdim, ngroups * dstate, ngroups * dstate], dim=-1
    )
    return dict(
        x=x.view(T, nheads, headdim),
        B=B.view(T, ngroups, dstate),
        C=C.view(T, ngroups, dstate),
        dt=torch.randn(T, nheads, device=DEVICE, dtype=dtype) * 0.5,
        A=-torch.exp(torch.rand(nheads, device=DEVICE, dtype=torch.float32)),
        D=torch.rand(nheads, device=DEVICE, dtype=torch.float32),
        dt_bias=torch.rand(nheads, device=DEVICE, dtype=torch.float32),
    )


def _run_ssd(inputs, seqlens, chunk_size, initial_states):
    cu_seqlens = torch.tensor((0, *seqlens), device=DEVICE).cumsum(0).to(torch.int32)
    cu_chunk_seqlens, last_chunk_indices, seq_idx = compute_varlen_chunk_metadata(
        cu_seqlens, chunk_size
    )
    out = torch.empty_like(inputs["x"])
    final_states = mamba_chunk_scan_combined_varlen(
        inputs["x"],
        inputs["dt"],
        inputs["A"],
        inputs["B"],
        inputs["C"],
        chunk_size,
        cu_seqlens=cu_seqlens,
        cu_chunk_seqlens=cu_chunk_seqlens,
        last_chunk_indices=last_chunk_indices,
        seq_idx=seq_idx,
        out=out,
        D=inputs["D"],
        dt_bias=inputs["dt_bias"],
        initial_states=initial_states,
        dt_softplus=True,
        state_dtype=torch.float32,
    )
    return out, final_states


def _run_both_paths(monkeypatch, inputs, seqlens, chunk_size, initial_states):
    results, calls = {}, {}
    for use_td in (False, True):
        monkeypatch.setenv("VLLM_TRITON_USE_TD", "1" if use_td else "0")
        with monkeypatch.context() as m:
            calls[use_td] = _spy_use_td(m)
            results[use_td] = _run_ssd(inputs, seqlens, chunk_size, initial_states)
    return results, calls


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seqlens", SEQLENS)
@pytest.mark.parametrize("has_initial_states", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_ssd_td_matches_pointer(shape, seqlens, has_initial_states, dtype, monkeypatch):
    """TD and pointer paths do the same arithmetic on the same tiles, with and
    without initial states (chunked prefill / prefix-cache hits). bf16 must be
    bit-identical; fp32 chunk_scan output may differ by an ULP, since the two
    paths compile to different instruction schedules even at the same config."""
    nheads, headdim, dstate, ngroups, chunk_size = shape
    inputs = _make_inputs(nheads, headdim, dstate, ngroups, seqlens, dtype)
    initial_states = (
        torch.randn(
            len(seqlens), nheads, headdim, dstate, device=DEVICE, dtype=torch.float32
        )
        * 0.1
        if has_initial_states
        else None
    )

    results, calls = _run_both_paths(
        monkeypatch, inputs, seqlens, chunk_size, initial_states
    )

    for use_td in (False, True):
        for name in ("scan", "state"):
            assert calls[use_td][name] and all(
                v is use_td for v in calls[use_td][name]
            ), f"{name} kernel launched with USE_TD={calls[use_td][name]}"
    (out_ptr, st_ptr), (out_td, st_td) = results[False], results[True]
    # Exact for bf16; torch's default fp32 tolerance for fp32.
    tol = dict(atol=0, rtol=0) if dtype == torch.bfloat16 else {}
    torch.testing.assert_close(out_td, out_ptr, **tol)
    torch.testing.assert_close(st_td, st_ptr, **tol)


def test_ssd_td_falls_back_on_unaligned_layout(monkeypatch):
    """A headdim whose rows are not 16-byte aligned cannot back a descriptor:
    with TD forced on, both kernels must launch on the pointer path and still
    produce the pointer-path result."""
    seqlens = (300, 17, 500)
    inputs = _make_inputs(4, 5, 128, 1, seqlens, torch.bfloat16)

    results, calls = _run_both_paths(monkeypatch, inputs, seqlens, 128, None)

    for name in ("scan", "state"):
        assert calls[True][name] and not any(calls[True][name])
    torch.testing.assert_close(results[True][0], results[False][0], atol=0, rtol=0)
    torch.testing.assert_close(results[True][1], results[False][1], atol=0, rtol=0)
