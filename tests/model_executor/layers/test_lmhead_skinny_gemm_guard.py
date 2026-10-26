# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""lm_head skinny-GEMM weight-size guard: core contracts.

wvSplitK_hf_sml's cost at n activation rows scales as ~n passes over the
weight, so past the last-level (memory-side) cache its effective
bandwidth collapses. At the DSV4.1-Flash TP=4 serving point the draft
vocab-parallel lm_head shard ([32320, 5120] bf16, ~315 MiB, one call per
decode step) hit this collapse. The guard keeps every weight above
VLLM_SKINNY_GEMM_MAX_WEIGHT_BYTES on the generic GEMM fall-through at
0 < n <= 5; the default cap is the memory-side cache size, read once
from the hardware (256 MiB on MI355X, 128 MiB on MI300X-class parts),
with a 256 MiB fallback where that cannot be read. The swap is NOT
bit-exact (different GEMM kernel, different accumulation order) -- these
tests check the decision table, dispatch selection, cap tuning and
hardware-derivation, and numerical agreement within bf16 GEMM tolerance.
"""

import os

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.layers.utils import (
    _amd_l3_cache_bytes,
    _skinny_gemm_max_weight_bytes,
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

_MAX_BYTES_VAR = "VLLM_SKINNY_GEMM_MAX_WEIGHT_BYTES"

# The serving-site shape: the DSpark draft vocab-parallel lm_head shard at
# TP=4 (draft hidden size 5120), above the 256 MiB default cap.
BIG_SHAPE = (32320, 5120)  # 315.5 MiB bf16
# A dense-projection-class weight: far below the cap.
SMALL_SHAPE = (2048, 7168)  # 29 MiB bf16


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # These tests assert exact dispatch selection, so pin every env var
    # that feeds it: state leaked by unrelated tests earlier in a long
    # suite run must not change the outcome here.
    monkeypatch.delenv(_MAX_BYTES_VAR, raising=False)
    monkeypatch.setenv("VLLM_ROCM_USE_SKINNY_GEMM", "1")
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    _skinny_guard_warned.clear()
    envs.disable_envs_cache()
    _skinny_gemm_max_weight_bytes.cache_clear()
    yield
    _skinny_guard_warned.clear()
    envs.disable_envs_cache()
    _skinny_gemm_max_weight_bytes.cache_clear()


def _weight(shape, seed=0):
    torch.manual_seed(seed)
    return torch.randn(*shape, dtype=torch.bfloat16, device="cuda")


def _x(n, k, seed=1):
    torch.manual_seed(seed)
    return torch.randn(n, k, dtype=torch.bfloat16, device="cuda")


def _set_cap(max_bytes):
    # None -> the hardware-derived default (memory-side cache size, with
    # the 256 MiB fallback); an int -> that exact cap. The accessor caches
    # its one evaluation, so clear it after every change.
    if max_bytes is None:
        os.environ.pop(_MAX_BYTES_VAR, None)
    else:
        os.environ[_MAX_BYTES_VAR] = str(max_bytes)
    envs.disable_envs_cache()
    _skinny_gemm_max_weight_bytes.cache_clear()


# Event names that are not compute-kernel records: transfers, runtime API
# wrappers, framework overhead. Every fn profiled here is a GEMM, so a
# capture holding only these (in the observed failure: nothing at all)
# recorded no kernels.
_NOT_KERNEL = ("Memcpy", "Memset", "cuda", "hip", "aten::", "at::", "ProfilerStep")


def _launched_kernels(fn, iters=3, attempts=3):
    """Kernel names recorded for fn(), retrying failed captures.

    Mid-suite, the ROCm activity tracer can drop every kernel record of a
    capture while the neighboring captures in the same test stay healthy
    (seen once in the full model_executor suite, next to a "ROCTracer
    produced duplicate flow start" warning). A GEMM always launches
    device kernels, so a capture without any kernel record is a failed
    capture -- retry it rather than read it as dispatch evidence.
    """
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.accelerator.synchronize()
    kernels = []
    for _ in range(attempts):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                fn()
            torch.accelerator.synchronize()
        kernels = [e.key for e in prof.key_averages()]
        if any(not k.startswith(_NOT_KERNEL) for k in kernels):
            break
    return kernels


def _has_wvsplitk_hf_sml(names):
    return any("wvSplitK_hf_sml" in k for k in names)


def _has_cijk(names):
    return any("Cijk" in k for k in names)


def test_guard_decision():
    """Decision table: bytes-gated at 0 < n <= 5; n=1 (no pathology)
    always permits; the cap is tunable. The default (auto) cap gates the
    315.5 MiB site shape on every part it can run on: the memory-side
    cache is 256 MiB (MI355X) / 128 MiB (MI300X-class), and the fallback
    is 256 MiB."""
    small = torch.zeros(SMALL_SHAPE, dtype=torch.bfloat16)
    big = torch.zeros(BIG_SHAPE, dtype=torch.bfloat16)
    assert lmhead_skinny_guard_permits(small, 5)
    for n in (2, 3, 4, 5):
        assert not lmhead_skinny_guard_permits(big, n)
    assert lmhead_skinny_guard_permits(big, 1), "n=1 carve-out missing"

    _set_cap(1024 * 1024 * 1024)
    assert lmhead_skinny_guard_permits(big, 5), "raised cap must permit"

    _set_cap(128 * 1024 * 1024)
    assert not lmhead_skinny_guard_permits(big, 5), "lowered cap must gate"

    _set_cap(None)
    assert not lmhead_skinny_guard_permits(big, 5), "auto cap must gate"


# Canned amdsmi cache lists (amdsmi_get_gpu_cache_info(...)["cache"]),
# trimmed to the fields the parse keys on; sizes are KB as amdsmi
# reports them.
_AMD_SMI_MI355X = [
    # L1 data / L1 inst / L2: the levels below the boundary.
    {
        "cache_properties": ["DATA_CACHE", "SIMD_CACHE"],
        "cache_level": 1,
        "cache_size": 32,
    },
    {
        "cache_properties": ["INST_CACHE", "SIMD_CACHE"],
        "cache_level": 1,
        "cache_size": 64,
    },
    {
        "cache_properties": ["DATA_CACHE", "SIMD_CACHE"],
        "cache_level": 2,
        "cache_size": 4096,
    },
    # The memory-side cache (MALL): 256 MiB, the level the guard keys on.
    {
        "cache_properties": ["DATA_CACHE", "SIMD_CACHE"],
        "cache_level": 3,
        "cache_size": 262144,
    },
]
_AMD_SMI_MI300X_CLASS = [
    {"cache_properties": ["DATA_CACHE"], "cache_level": 2, "cache_size": 8192},
    {"cache_properties": ["DATA_CACHE"], "cache_level": 3, "cache_size": 131072},
]


def test_l3_parse_selects_level3_data_cache():
    """The parse takes the level-3 DATA_CACHE (the memory-side cache),
    not the L2 below it, in amdsmi's KB units."""
    assert _amd_l3_cache_bytes(_AMD_SMI_MI355X) == 256 * 1024 * 1024
    assert _amd_l3_cache_bytes(_AMD_SMI_MI300X_CLASS) == 128 * 1024 * 1024
    # Empty list, or instruction caches only: no L3 to key on.
    assert _amd_l3_cache_bytes([]) is None
    assert (
        _amd_l3_cache_bytes(
            [{"cache_properties": ["INST_CACHE"], "cache_level": 3, "cache_size": 64}]
        )
        is None
    )


def test_cap_env_hardware_fallback_precedence(monkeypatch):
    """Unset env: the cap is the L3 size the library reports; an
    unreadable L3 falls back to the documented 256 MiB; an explicit env
    value overrides the hardware."""
    _set_cap(None)
    monkeypatch.setattr(
        "vllm.model_executor.layers.utils._amd_l3_cache_bytes",
        lambda cache_info=None: 128 * 1024 * 1024,
    )
    _skinny_gemm_max_weight_bytes.cache_clear()
    assert _skinny_gemm_max_weight_bytes() == 128 * 1024 * 1024

    monkeypatch.setattr(
        "vllm.model_executor.layers.utils._amd_l3_cache_bytes",
        lambda cache_info=None: None,
    )
    _skinny_gemm_max_weight_bytes.cache_clear()
    assert _skinny_gemm_max_weight_bytes() == 256 * 1024 * 1024

    _set_cap(1024 * 1024 * 1024)
    assert _skinny_gemm_max_weight_bytes() == 1024 * 1024 * 1024


def test_census_warning_fires_once_per_site(caplog):
    w = torch.zeros(BIG_SHAPE, dtype=torch.bfloat16)
    with caplog.at_level("WARNING", logger="vllm.model_executor.layers.utils"):
        lmhead_skinny_guard_permits(w, 5)
        lmhead_skinny_guard_permits(w, 5)  # same (shape, n): once
        lmhead_skinny_guard_permits(w, 4)  # new n: second line
    warns = [r for r in caplog.records if "skinny-GEMM guard engaged" in r.message]
    assert len(warns) == 2
    assert "32320" in warns[0].message and "n=5" in warns[0].message


@pytest.mark.parametrize("cap_gib,expect_skinny", [(None, False), (1, True)])
def test_n5_big_weight_dispatch(cap_gib, expect_skinny):
    """The draft-lm_head shape (n=5, >256 MiB): under the default cap the
    guard redirects to the generic hipBLASLt GEMM (the same kernel class
    its n=6 verify twin dispatches to); a raised cap keeps the stock
    wvSplitK path."""
    _set_cap(cap_gib * 1024 * 1024 * 1024 if cap_gib else None)
    w = _weight(BIG_SHAPE)
    x = _x(5, BIG_SHAPE[1])
    ref = torch.nn.functional.linear(x, w)
    kernels = _launched_kernels(lambda: rocm_unquantized_gemm_impl(x, w, None))
    assert _has_wvsplitk_hf_sml(kernels) == expect_skinny, f"kernels={kernels}"
    if not expect_skinny:
        assert _has_cijk(kernels), (
            f"fall-through did not reach the hipBLASLt GEMM; kernels={kernels}"
        )
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
    """The guard must not touch dispatch outside its window, under the
    default cap AND a raised one: n=6 (above the skinny branch) and small
    weights (below the byte cap)."""
    w = _weight(shape)
    x = _x(n, shape[1])
    ref = torch.nn.functional.linear(x, w)
    for cap_gib in (None, 1):
        _set_cap(cap_gib * 1024 * 1024 * 1024 if cap_gib else None)
        _skinny_guard_warned.clear()
        kernels = _launched_kernels(lambda: rocm_unquantized_gemm_impl(x, w, None))
        assert _has_wvsplitk_hf_sml(kernels) == expect_skinny, (
            f"cap={cap_gib} GiB: skinny dispatch changed for "
            f"shape={shape} n={n}; kernels={kernels}"
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
        f"n=1 was redirected by the guard; the carve-out is missing; kernels={kernels}"
    )
