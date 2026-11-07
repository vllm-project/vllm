# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The gfx942 GDN prefill configs change only launch parameters, so output
and final state must stay bit-identical to the stock autotune space."""

from __future__ import annotations

import contextlib

import pytest
import torch

from vllm.third_party.flash_linear_attention.ops import (
    chunk_delta_h,
    chunk_o,
    chunk_scaled_dot_kkt,
    solve_tril,
    wy_fast,
)
from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule
from vllm.third_party.flash_linear_attention.ops.utils import is_gfx942

# kernel label -> (autotuned kernel, gfx942 config list, stock config list)
KERNELS = {
    "chunk_o": (
        lambda: chunk_o.chunk_fwd_kernel_o,
        lambda: chunk_o._FWD_O_CONFIGS,
        lambda: chunk_o._FWD_O_STOCK,
    ),
    "chunk_delta_h": (
        lambda: chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
        lambda: chunk_delta_h._FWD_H_CONFIGS,
        lambda: chunk_delta_h._FWD_H_STOCK,
    ),
    "chunk_scaled_dot_kkt": (
        lambda: chunk_scaled_dot_kkt.chunk_scaled_dot_kkt_fwd_kernel,
        lambda: chunk_scaled_dot_kkt._KKT_CONFIGS,
        lambda: chunk_scaled_dot_kkt._KKT_STOCK,
    ),
    "wy_fast": (
        lambda: wy_fast.recompute_w_u_fwd_kernel,
        lambda: wy_fast._WU_CONFIGS,
        lambda: wy_fast._WU_STOCK,
    ),
    "solve_tril_merge64": (
        lambda: solve_tril.merge_16x16_to_64x64_inverse_kernel,
        lambda: solve_tril._MERGE_64_CONFIGS,
        lambda: solve_tril._MERGE_64_STOCK,
    ),
}

# Ragged and not a multiple of the 64-token chunk, so masked tails run.
SEQ_LENS = [137, 64, 391, 432]
SHAPES = {"16x48": (16, 48), "2x8": (2, 8)}
HEAD_DIM = 128


def _autotuner(kernel):
    seen = kernel
    while not hasattr(kernel, "configs"):
        kernel = getattr(kernel, "fn", None)
        if kernel is None:
            raise AssertionError(f"no triton.Autotuner under {seen!r}")
    return kernel


@contextlib.contextmanager
def _pinned(kernel, configs):
    tuner = _autotuner(kernel)
    saved_configs, saved_cache = tuner.configs, dict(tuner.cache)
    tuner.configs = list(configs)
    tuner.cache.clear()
    try:
        yield
    finally:
        tuner.configs = saved_configs
        tuner.cache.clear()
        tuner.cache.update(saved_cache)


def _inputs(hg, h, device="cuda"):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    total = sum(SEQ_LENS)
    cu = torch.zeros(len(SEQ_LENS) + 1, dtype=torch.int32)
    cu[1:] = torch.tensor(SEQ_LENS, dtype=torch.int32).cumsum(0)
    return dict(
        q=torch.randn(1, total, hg, HEAD_DIM, dtype=dtype, device=device),
        k=torch.nn.functional.normalize(
            torch.randn(1, total, hg, HEAD_DIM, dtype=dtype, device=device),
            p=2,
            dim=-1,
        ),
        v=torch.randn(1, total, h, HEAD_DIM, dtype=dtype, device=device),
        g=torch.nn.functional.logsigmoid(
            torch.rand(1, total, h, dtype=torch.float32, device=device)
        ),
        beta=torch.rand(1, total, h, dtype=dtype, device=device).sigmoid(),
        initial_state=torch.randn(
            len(SEQ_LENS), h, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device
        )
        * 0.05,
        output_final_state=True,
        cu_seqlens=cu.to(device),
        use_qk_l2norm_in_kernel=True,
    )


def _run(inputs, pins):
    was_on = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    # NaN-fills torch.empty so an unwritten tile cannot pass on stale memory.
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with contextlib.ExitStack() as stack:
            for kernel, configs in pins:
                stack.enter_context(_pinned(kernel, configs))
            out, state = chunk_gated_delta_rule(**inputs)
    finally:
        torch.use_deterministic_algorithms(was_on, warn_only=warn_only)
    return out.clone(), state.clone()


@pytest.mark.skipif(not is_gfx942, reason="gfx942-specific configs")
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("kernel_name", [*KERNELS, "all"])
@torch.inference_mode()
def test_gfx942_config_matches_stock(kernel_name, shape):
    """Each gfx942 config, and all of them together, reproduce stock output."""
    hg, h = SHAPES[shape]
    inputs = _inputs(hg, h)

    names = list(KERNELS) if kernel_name == "all" else [kernel_name]
    stock_pins, gfx_pins = [], []
    for n in KERNELS:
        kernel, gfx, stock = KERNELS[n]
        stock_pins.append((kernel(), stock()[:1]))
        gfx_pins.append((kernel(), gfx() if n in names else stock()[:1]))

    ref_out, ref_state = _run(inputs, stock_pins)
    out, state = _run(inputs, gfx_pins)

    assert torch.equal(ref_out, out), f"{kernel_name}/{shape}: output changed"
    assert torch.equal(ref_state, state), f"{kernel_name}/{shape}: final state changed"


_KDA_SHARED = ["chunk_delta_h", "solve_tril_merge64"]


def _kda_inputs(h, device="cuda"):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    total = sum(SEQ_LENS)
    cu = torch.zeros(len(SEQ_LENS) + 1, dtype=torch.int32)
    cu[1:] = torch.tensor(SEQ_LENS, dtype=torch.int32).cumsum(0)
    return dict(
        q=torch.randn(1, total, h, HEAD_DIM, dtype=dtype, device=device),
        k=torch.randn(1, total, h, HEAD_DIM, dtype=dtype, device=device),
        v=torch.randn(1, total, h, HEAD_DIM, dtype=dtype, device=device),
        # Per-key-dim gate; this is what drives the kernel's USE_GK branch.
        g=torch.nn.functional.logsigmoid(
            torch.rand(1, total, h, HEAD_DIM, dtype=torch.float32, device=device)
        ),
        beta=torch.rand(1, total, h, dtype=dtype, device=device).sigmoid(),
        initial_state=torch.randn(
            len(SEQ_LENS), h, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device
        )
        * 0.05,
        output_final_state=True,
        cu_seqlens=cu.to(device),
        use_qk_l2norm_in_kernel=True,
    )


@pytest.mark.skipif(not is_gfx942, reason="gfx942-specific configs")
@pytest.mark.parametrize("kernel_name", [*_KDA_SHARED, "all"])
@torch.inference_mode()
def test_gfx942_config_matches_stock_on_kda(kernel_name):
    """Kimi Linear / Kimi-K3 / GLM-5-Next reach two of these kernels too."""
    from vllm.third_party.flash_linear_attention.ops.kda import chunk_kda

    def run(pins):
        # chunk_kda consumes v in place, so every call needs fresh inputs.
        inputs = _kda_inputs(8)
        with contextlib.ExitStack() as stack:
            for kernel, configs in pins:
                stack.enter_context(_pinned(kernel, configs))
            out, state = chunk_kda(**inputs)
        return out.clone(), state.clone()

    stock_pins, gfx_pins = [], []
    for n in _KDA_SHARED:
        kernel, gfx, stock = KERNELS[n]
        stock_pins.append((kernel(), stock()[:1]))
        gfx_pins.append((kernel(), gfx() if kernel_name in (n, "all") else stock()[:1]))

    ref_out, ref_state = run(stock_pins)
    out, state = run(gfx_pins)

    assert torch.equal(ref_out, out), f"kda/{kernel_name}: output changed"
    assert torch.equal(ref_state, state), f"kda/{kernel_name}: final state changed"


def test_only_gfx942_gets_the_specialised_configs():
    """A future edit must not hand another GPU the gfx942-only configs."""
    for name, (_kernel, gfx, stock) in KERNELS.items():
        if is_gfx942:
            assert len(gfx()) == 1, f"{name}: expected one static gfx942 config"
        else:
            assert gfx() is stock(), f"{name}: non-gfx942 must use the stock list"
