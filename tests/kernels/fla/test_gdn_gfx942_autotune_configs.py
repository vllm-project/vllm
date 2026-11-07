# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib

import pytest
import torch

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import chunk_delta_h, chunk_o
from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule
from vllm.triton_utils import triton

B, T, HG, H, K, V = 1, 1024, 2, 4, 128, 128

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


def _added(kernel):
    return [c for c in _autotuner(kernel).configs if "matrix_instr_nonkdim" in c.kwargs]


def _inputs(device):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    return dict(
        q=torch.randn(B, T, HG, K, dtype=dtype, device=device),
        k=torch.nn.functional.normalize(
            torch.randn(B, T, HG, K, dtype=dtype, device=device), p=2, dim=-1
        ),
        v=torch.randn(B, T, H, V, dtype=dtype, device=device),
        g=torch.nn.functional.logsigmoid(
            torch.rand(B, T, H, dtype=torch.float32, device=device)
        ),
        beta=torch.rand(B, T, H, dtype=dtype, device=device).sigmoid(),
        initial_state=torch.zeros(B, H, K, V, dtype=torch.float32, device=device),
        output_final_state=True,
        cu_seqlens=torch.tensor([0, T], dtype=torch.int32, device=device),
        use_qk_l2norm_in_kernel=True,
    )


_KERNEL_O = "chunk_fwd_kernel_o"
_KERNEL_H = "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"
_STOCK_O = triton.Config({"BK": 32, "BV": 32}, num_warps=4, num_stages=2)
_STOCK_H = triton.Config({"BV": 32}, num_warps=4, num_stages=2)


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
    out, state = _pinned(inputs, _STOCK_O, _STOCK_H)
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
    if kernel_name == _KERNEL_O:
        out, state = _pinned(inputs, config, _STOCK_H)
    else:
        out, state = _pinned(inputs, _STOCK_O, config)

    assert torch.equal(ref_state, state), f"final state changed under {config}"
    assert torch.equal(ref_out, out), f"output changed under {config}"
