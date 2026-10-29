# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib

import pytest
import torch

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import chunk_delta_h, chunk_o
from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule

B, T, HG, H, K, V = 1, 1024, 2, 4, 128, 128

SELECT_T, SELECT_HG, SELECT_H = 8192, 16, 48

_ON_GFX942 = False
if current_platform.is_rocm():
    from vllm.platforms.rocm import get_cdna_version

    _ON_GFX942 = get_cdna_version() == 3


def _autotuner(kernel):
    seen = kernel
    while not hasattr(kernel, "configs"):
        kernel = getattr(kernel, "fn", None)
        if kernel is None:
            raise AssertionError(f"no triton.Autotuner under {seen!r}")
    return kernel


@contextlib.contextmanager
def _configs(kernel, configs):
    tuner = _autotuner(kernel)
    saved_configs, saved_cache = tuner.configs, dict(tuner.cache)
    tuner.configs = configs
    tuner.cache.clear()
    try:
        yield
    finally:
        tuner.configs = saved_configs
        tuner.cache.clear()
        tuner.cache.update(saved_cache)


@contextlib.contextmanager
def _fresh_cache(kernel):
    tuner = _autotuner(kernel)
    saved_cache = dict(tuner.cache)
    tuner.cache.clear()
    try:
        yield tuner
    finally:
        tuner.cache.clear()
        tuner.cache.update(saved_cache)


def _added(kernel):
    return [c for c in _autotuner(kernel).configs if "matrix_instr_nonkdim" in c.kwargs]


def _stock(kernel):
    return [
        c for c in _autotuner(kernel).configs if "matrix_instr_nonkdim" not in c.kwargs
    ]


def test_configs_are_added_only_on_gfx942():
    for module in (chunk_o, chunk_delta_h):
        assert bool(module._GFX942_CONFIGS) == _ON_GFX942


def _inputs(device, t=T, hg=HG, h=H):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    return dict(
        q=torch.randn(B, t, hg, K, dtype=dtype, device=device),
        k=torch.nn.functional.normalize(
            torch.randn(B, t, hg, K, dtype=dtype, device=device), p=2, dim=-1
        ),
        v=torch.randn(B, t, h, V, dtype=dtype, device=device),
        g=torch.nn.functional.logsigmoid(
            torch.rand(B, t, h, dtype=torch.float32, device=device)
        ),
        beta=torch.rand(B, t, h, dtype=dtype, device=device).sigmoid(),
        initial_state=torch.zeros(B, h, K, V, dtype=torch.float32, device=device),
        output_final_state=True,
        cu_seqlens=torch.tensor([0, t], dtype=torch.int32, device=device),
        use_qk_l2norm_in_kernel=True,
    )


_KERNEL_O = "chunk_fwd_kernel_o"
_KERNEL_H = "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"


def _kernels():
    return (
        chunk_o.chunk_fwd_kernel_o,
        getattr(chunk_delta_h, _KERNEL_H),
    )


def _pinned(inputs, o_config, h_config):
    kernel_o, kernel_h = _kernels()
    was_on = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    # NaN-fills torch.empty so a skipped write can't pass on leftover memory.
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with _configs(kernel_o, [o_config]), _configs(kernel_h, [h_config]):
            out, state = chunk_gated_delta_rule(**inputs)
    finally:
        torch.use_deterministic_algorithms(was_on, warn_only=was_warn_only)
    return out.clone(), state.clone()


@pytest.fixture(scope="module")
def stock_reference():
    inputs = _inputs("cuda")
    kernel_o, kernel_h = _kernels()
    out, state = _pinned(inputs, _stock(kernel_o)[0], _stock(kernel_h)[0])
    return inputs, out, state


def _added_cases():
    if not _ON_GFX942:
        return []
    kernel_o, kernel_h = _kernels()
    return [(_KERNEL_O, c) for c in _added(kernel_o)] + [
        (_KERNEL_H, c) for c in _added(kernel_h)
    ]


@pytest.mark.skipif(not _ON_GFX942, reason="gfx942-specific configs")
@pytest.mark.parametrize(
    "kernel_name,config", _added_cases(), ids=lambda v: str(v).replace(" ", "")
)
def test_added_config_matches_the_stock_result(kernel_name, config, stock_reference):
    inputs, ref_out, ref_state = stock_reference
    kernel_o, kernel_h = _kernels()
    if kernel_name == _KERNEL_O:
        out, state = _pinned(inputs, config, _stock(kernel_h)[0])
    else:
        out, state = _pinned(inputs, _stock(kernel_o)[0], config)

    assert torch.equal(ref_state, state), f"final state changed under {config}"
    assert torch.equal(ref_out, out), f"output changed under {config}"


@pytest.mark.optional
@pytest.mark.skipif(not _ON_GFX942, reason="gfx942-specific configs")
@pytest.mark.parametrize(
    "kernel_name",
    ["chunk_fwd_kernel_o", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"],
)
def test_autotuner_selects_an_added_config(kernel_name):
    module = chunk_o if kernel_name == "chunk_fwd_kernel_o" else chunk_delta_h
    kernel = getattr(module, kernel_name)

    with _fresh_cache(kernel) as tuner:
        chunk_gated_delta_rule(**_inputs("cuda", t=SELECT_T, hg=SELECT_HG, h=SELECT_H))
        assert "matrix_instr_nonkdim" in tuner.best_config.kwargs, (
            f"{kernel_name} picked a stock config: {tuner.best_config}"
        )
