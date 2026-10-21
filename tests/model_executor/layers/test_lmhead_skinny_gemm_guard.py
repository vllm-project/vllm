# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""lm_head skinny-GEMM weight-size guard: core contracts.

wvSplitK_hf_sml re-reads the weight matrix once per K-shard, so its
effective bandwidth collapses on weights far larger than L2. At the
DSV4.1-Flash TP=4 serving point the draft vocab-parallel lm_head shard
([32320, 5120] bf16, ~315 MiB, one call per decode step) hit this
collapse. The guard keeps every weight above
VLLM_SKINNY_GEMM_MAX_WEIGHT_BYTES (default 256 MiB) on the generic GEMM
fall-through at 0 < n <= 5. The swap is NOT bit-exact (different GEMM
kernel, different accumulation order) -- these tests check the decision
table, dispatch selection, env behavior, and numerical agreement within
bf16 GEMM tolerance. VLLM_DISABLE_LMHEAD_SKINNY_GEMM_GUARD=1 restores
the stock dispatch exactly.
"""

import os

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.layers.utils import (
    _skinny_guard_warned,
    lmhead_skinny_guard_permits,
    rocm_unquantized_gemm_impl,
)

try:
    from vllm import _custom_ops as ops

    WVSPLITK_AVAILABLE = hasattr(ops, "wvSplitK")
except ImportError:
    WVSPLITK_AVAILABLE = False

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GEMM kernels run on GPU"),
    pytest.mark.skipif(
        not WVSPLITK_AVAILABLE,
        reason="requires the ROCm skinny-GEMM custom ops",
    ),
]

_DISABLE_VAR = "VLLM_DISABLE_LMHEAD_SKINNY_GEMM_GUARD"
_MAX_BYTES_VAR = "VLLM_SKINNY_GEMM_MAX_WEIGHT_BYTES"

# The serving-site shape: the DSpark draft vocab-parallel lm_head shard at
# TP=4 (draft hidden size 5120), above the 256 MiB default cap.
BIG_SHAPE = (32320, 5120)  # 315.5 MiB bf16
# A dense-projection-class weight: far below the cap.
SMALL_SHAPE = (2048, 7168)  # 29 MiB bf16


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(_DISABLE_VAR, raising=False)
    monkeypatch.delenv(_MAX_BYTES_VAR, raising=False)
    _skinny_guard_warned.clear()
    envs.disable_envs_cache()
    yield
    _skinny_guard_warned.clear()
    envs.disable_envs_cache()


def _weight(shape, seed=0):
    torch.manual_seed(seed)
    return torch.randn(*shape, dtype=torch.bfloat16, device="cuda")


def _x(n, k, seed=1):
    torch.manual_seed(seed)
    return torch.randn(n, k, dtype=torch.bfloat16, device="cuda")


def _set_guard(state):
    if state:
        os.environ[_DISABLE_VAR] = "1"
    else:
        os.environ.pop(_DISABLE_VAR, None)
    envs.disable_envs_cache()


def _launched_kernels(fn, iters=3):
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.accelerator.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.accelerator.synchronize()
    return [e.key for e in prof.key_averages()]


def _has_wvsplitk_hf_sml(names):
    return any("wvSplitK_hf_sml" in k for k in names)


def _has_cijk(names):
    return any("Cijk" in k for k in names)


def test_guard_decision():
    """Decision table: bytes-gated at 0 < n <= 5; n=1 (no pathology) and
    the kill switch always permit; the cap is tunable."""
    small = torch.zeros(SMALL_SHAPE, dtype=torch.bfloat16)
    big = torch.zeros(BIG_SHAPE, dtype=torch.bfloat16)
    assert lmhead_skinny_guard_permits(small, 5)
    for n in (2, 3, 4, 5):
        assert not lmhead_skinny_guard_permits(big, n)
    assert lmhead_skinny_guard_permits(big, 1), "n=1 carve-out missing"

    os.environ[_DISABLE_VAR] = "1"
    envs.disable_envs_cache()
    assert lmhead_skinny_guard_permits(big, 5), "kill switch must permit"

    os.environ[_DISABLE_VAR] = "0"
    os.environ[_MAX_BYTES_VAR] = str(1024 * 1024 * 1024)
    envs.disable_envs_cache()
    assert lmhead_skinny_guard_permits(big, 5), "raised cap must permit"

    os.environ.pop(_DISABLE_VAR, None)
    os.environ.pop(_MAX_BYTES_VAR, None)
    envs.disable_envs_cache()


def test_census_warning_fires_once_per_site(caplog):
    w = torch.zeros(BIG_SHAPE, dtype=torch.bfloat16)
    with caplog.at_level("WARNING", logger="vllm.model_executor.layers.utils"):
        lmhead_skinny_guard_permits(w, 5)
        lmhead_skinny_guard_permits(w, 5)  # same (shape, n): once
        lmhead_skinny_guard_permits(w, 4)  # new n: second line
    warns = [r for r in caplog.records if "skinny-GEMM guard engaged" in r.message]
    assert len(warns) == 2
    assert "32320" in warns[0].message and "n=5" in warns[0].message


@pytest.mark.parametrize("disabled,expect_skinny", [(False, False), (True, True)])
def test_n5_big_weight_dispatch(disabled, expect_skinny):
    """The draft-lm_head shape (n=5, >256 MiB): the guard redirects to the
    generic hipBLASLt GEMM (the same kernel class its n=6 verify twin
    dispatches to); the kill switch restores the stock wvSplitK path."""
    _set_guard("1" if disabled else None)
    w = _weight(BIG_SHAPE)
    x = _x(5, BIG_SHAPE[1])
    ref = torch.nn.functional.linear(x, w)
    kernels = _launched_kernels(lambda: rocm_unquantized_gemm_impl(x, w, None))
    assert _has_wvsplitk_hf_sml(kernels) == expect_skinny
    if not expect_skinny:
        assert _has_cijk(kernels), "fall-through did not reach the hipBLASLt GEMM"
    out = rocm_unquantized_gemm_impl(x, w, None)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-1)


@pytest.mark.parametrize(
    "shape,n,expect_skinny",
    [
        (BIG_SHAPE, 6, False),  # n=6 never matches the skinny branch
        (SMALL_SHAPE, 5, True),  # small weight: bytes-gated only
    ],
)
def test_guard_no_op_where_it_must_be(shape, n, expect_skinny):
    """The guard must not touch dispatch outside its window, in BOTH guard
    states: n=6 (above the skinny branch) and small weights (below the
    byte cap)."""
    w = _weight(shape)
    x = _x(n, shape[1])
    ref = torch.nn.functional.linear(x, w)
    for guard_state in (None, "1"):
        _set_guard(guard_state)
        _skinny_guard_warned.clear()
        kernels = _launched_kernels(lambda: rocm_unquantized_gemm_impl(x, w, None))
        assert _has_wvsplitk_hf_sml(kernels) == expect_skinny, (
            f"guard_state={guard_state}: skinny dispatch changed for "
            f"shape={shape} n={n}"
        )
    out = rocm_unquantized_gemm_impl(x, w, None)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-1)


def test_n1_big_weight_dispatch_unchanged():
    """n=1 must keep the stock wvSplitK_hf_sml<..., 2, 1> kernel under the
    guard (default ON) -- the carve-out keeps n=1 bit-identical to stock."""
    w = _weight(BIG_SHAPE)
    x = _x(1, BIG_SHAPE[1])
    kernels = _launched_kernels(lambda: rocm_unquantized_gemm_impl(x, w, None))
    assert _has_wvsplitk_hf_sml(kernels), (
        "n=1 was redirected by the guard; the carve-out is missing"
    )
