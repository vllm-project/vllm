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
from vllm.third_party.flash_linear_attention.ops.utils import is_cdna3
from vllm.triton_utils import triton
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import set_random_seed

# kernel label -> (kernel, gfx942 configs, stock configs)
KERNELS = {
    "chunk_o": (
        lambda: chunk_o.chunk_fwd_kernel_o,
        lambda: list(chunk_o._FWD_O_CDNA3),
        lambda: chunk_o._FWD_O_STOCK,
    ),
    "chunk_delta_h": (
        lambda: chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
        lambda: [chunk_delta_h._FWD_H_CDNA3],
        lambda: chunk_delta_h._FWD_H_STOCK,
    ),
    "chunk_scaled_dot_kkt": (
        lambda: chunk_scaled_dot_kkt.chunk_scaled_dot_kkt_fwd_kernel,
        lambda: list(chunk_scaled_dot_kkt._KKT_CDNA3),
        lambda: chunk_scaled_dot_kkt._KKT_STOCK,
    ),
    "wy_fast": (
        lambda: wy_fast.recompute_w_u_fwd_kernel,
        lambda: list(wy_fast._WU_CDNA3),
        lambda: wy_fast._WU_STOCK,
    ),
    "solve_tril_merge64": (
        lambda: solve_tril.merge_16x16_to_64x64_inverse_kernel,
        lambda: solve_tril._MERGE_64_CONFIGS,
        lambda: solve_tril._MERGE_64_STOCK,
    ),
}
# Kernels that pick their gfx942 config at launch -> (module, picker).
PICKERS = {
    "chunk_o": (chunk_o, "_fwd_o_cdna3_config"),
    "chunk_delta_h": (chunk_delta_h, "_fwd_h_cdna3_config"),
    "chunk_scaled_dot_kkt": (chunk_scaled_dot_kkt, "_kkt_cdna3_config"),
    "wy_fast": (wy_fast, "_wu_cdna3_config"),
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
def _pinned(name, configs):
    if name in PICKERS:
        # Picked at launch on gfx942: pin one config, or keep the real choice.
        module, attr = PICKERS[name]
        pick = getattr(module, attr)
        if len(configs) == 1:
            setattr(module, attr, lambda *_: configs[0])
        try:
            yield
        finally:
            setattr(module, attr, pick)
        return
    tuner = _autotuner(KERNELS[name][0]())
    saved_configs, saved_cache = tuner.configs, dict(tuner.cache)
    tuner.configs = list(configs)
    tuner.cache.clear()
    try:
        yield
    finally:
        tuner.configs = saved_configs
        tuner.cache.clear()
        tuner.cache.update(saved_cache)


def _inputs(hg, h, dtype, device="cuda"):
    set_random_seed(0)
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


def _run(inputs, pins, op=chunk_gated_delta_rule):
    was_on = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    # NaN-fills torch.empty so an unwritten tile cannot pass on stale memory.
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with contextlib.ExitStack() as stack:
            for name, configs in pins:
                stack.enter_context(_pinned(name, configs))
            out, state = op(**inputs)
    finally:
        torch.use_deterministic_algorithms(was_on, warn_only=warn_only)
    return out.clone(), state.clone()


def _stock_pins(names):
    return [(n, KERNELS[n][2]()[:1]) for n in names]


def _gfx_pin_sets(names, kernel_name):
    """kernel_name's gfx942 configs one at a time, or for "all" every launch choice."""
    if kernel_name == "all":
        return [[(n, KERNELS[n][1]()) for n in names]]
    stock = dict(_stock_pins(names))
    return [
        [(n, [cfg] if n == kernel_name else stock[n]) for n in names]
        for cfg in KERNELS[kernel_name][1]()
    ]


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


@pytest.mark.skipif(not is_cdna3, reason="gfx942-specific configs")
@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("kernel_name", [*KERNELS, "all"])
@torch.inference_mode()
def test_gfx942_config_matches_stock(kernel_name, shape, dtype):
    """Each gfx942 config, and every launch-time choice, reproduces stock output."""
    hg, h = SHAPES[shape]
    inputs = _inputs(hg, h, DTYPES[dtype])
    ref_out, ref_state = _run(inputs, _stock_pins(KERNELS))
    for i, pins in enumerate(_gfx_pin_sets(list(KERNELS), kernel_name)):
        out, state = _run(inputs, pins)
        tag = f"{kernel_name}[{i}]/{shape}/{dtype}"
        assert torch.equal(ref_out, out), f"{tag}: output changed"
        assert torch.equal(ref_state, state), f"{tag}: final state changed"


_KDA_SHARED = ["chunk_delta_h", "solve_tril_merge64"]


def _kda_inputs(h, dtype, device="cuda"):
    set_random_seed(0)
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


@pytest.mark.skipif(not is_cdna3, reason="gfx942-specific configs")
@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("h", [8, 32])
@pytest.mark.parametrize("kernel_name", [*_KDA_SHARED, "all"])
@torch.inference_mode()
def test_gfx942_config_matches_stock_on_kda(kernel_name, h, dtype):
    """The two configs shared with KDA reproduce stock output on its path too."""
    from vllm.third_party.flash_linear_attention.ops.kda import chunk_kda

    td = DTYPES[dtype]
    # chunk_kda consumes v in place, so every call needs fresh inputs.
    ref_out, ref_state = _run(_kda_inputs(h, td), _stock_pins(_KDA_SHARED), chunk_kda)
    for i, pins in enumerate(_gfx_pin_sets(_KDA_SHARED, kernel_name)):
        out, state = _run(_kda_inputs(h, td), pins, chunk_kda)
        tag = f"kda/{kernel_name}[{i}]/h{h}/{dtype}"
        assert torch.equal(ref_out, out), f"{tag}: output changed"
        assert torch.equal(ref_state, state), f"{tag}: final state changed"


TUNED_BV = chunk_delta_h._FWD_H_CDNA3.kwargs["BV"]
PICK_KEY = ("test",)
STOCK_BV64 = [c for c in chunk_delta_h._FWD_H_STOCK if c.kwargs["BV"] == 64]
STOCK_BV32 = [c for c in chunk_delta_h._FWD_H_STOCK if c.kwargs["BV"] == TUNED_BV]


@contextlib.contextmanager
def _fwd_h_pick(cfg):
    saved = dict(chunk_delta_h._FWD_H_PICK)
    chunk_delta_h._FWD_H_PICK.clear()
    if cfg is not None:
        chunk_delta_h._FWD_H_PICK[PICK_KEY] = cfg
    try:
        yield
    finally:
        chunk_delta_h._FWD_H_PICK.clear()
        chunk_delta_h._FWD_H_PICK.update(saved)


def _gate(V, num_seq_heads):
    return chunk_delta_h._fwd_h_cdna3_config(V, num_seq_heads, 0, PICK_KEY)


@pytest.mark.skipif(not is_cdna3, reason="gfx942-specific configs")
@pytest.mark.parametrize("V", [32, 64, 96, 128, 192])
def test_fwd_h_gate_defers_before_the_key_is_tuned(V):
    """With no cached pick the gate returns None, so the stock autotuner runs."""
    cu = num_compute_units(0)
    with _fwd_h_pick(None):
        for num_seq_heads in (1, 8, cu, 4 * cu):
            assert _gate(V, num_seq_heads) is None


@pytest.mark.skipif(not is_cdna3, reason="gfx942-specific configs")
@pytest.mark.parametrize("V", [32, 64, 96, 128, 192])
@pytest.mark.parametrize("stock", STOCK_BV32, ids=lambda c: f"w{c.num_warps}")
def test_fwd_h_gate_keeps_tuned_config_for_a_bv32_pick(V, stock):
    """A stock BV=32 pick takes the tuned config at every grid size."""
    cu = num_compute_units(0)
    with _fwd_h_pick(stock):
        for num_seq_heads in (1, 8, cu - 1, cu, cu + 1, 4 * cu):
            cfg = _gate(V, num_seq_heads)
            assert cfg is chunk_delta_h._FWD_H_CDNA3, (V, num_seq_heads, cfg)


@pytest.mark.skipif(not is_cdna3, reason="gfx942-specific configs")
@pytest.mark.parametrize("V", [32, 64, 96, 128, 192])
@pytest.mark.parametrize("stock", STOCK_BV64, ids=lambda c: f"w{c.num_warps}")
def test_fwd_h_gate_falls_back_to_stock_above_the_cu_count(V, stock):
    """A stock BV=64 pick is kept once the tuned BV=32 grid exceeds the CUs."""
    cu = num_compute_units(0)
    per_seq_head = triton.cdiv(V, TUNED_BV)
    boundary = cu // per_seq_head
    probes = {1, boundary - 1, boundary, boundary + 1, 4 * cu}
    with _fwd_h_pick(stock):
        for num_seq_heads in sorted(n for n in probes if n >= 1):
            cfg = _gate(V, num_seq_heads)
            fits = per_seq_head * num_seq_heads <= cu
            assert cfg is (chunk_delta_h._FWD_H_CDNA3 if fits else stock), (
                V,
                num_seq_heads,
                cu,
                cfg,
            )


@pytest.mark.skipif(not is_cdna3, reason="gfx942-specific configs")
@pytest.mark.parametrize("V", [33, 65, 97, 129])
def test_fwd_h_gate_counts_a_partial_tile_as_a_program(V):
    """V not divisible by BV rounds the grid up, so the switch happens earlier."""
    cu = num_compute_units(0)
    per_seq_head = triton.cdiv(V, TUNED_BV)
    assert per_seq_head == -(-V // TUNED_BV)
    stock = STOCK_BV64[0]
    with _fwd_h_pick(stock):
        fits = cu // per_seq_head
        assert _gate(V, fits) is chunk_delta_h._FWD_H_CDNA3
        assert _gate(V, fits + 1) is stock
