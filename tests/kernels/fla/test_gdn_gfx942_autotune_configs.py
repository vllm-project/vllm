# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The gfx942 autotune configs added to ``chunk_o`` and ``chunk_delta_h``.

The stock space cannot reach the fastest configuration here: it never sets
``matrix_instr_nonkdim``, ``chunk_o`` has no ``num_stages=1``, and
``check_shared_mem()`` compares the device's LDS against a threshold read off a
table with only NVIDIA entries, so it is ``False`` on gfx942 and the BK/BV=128
tiles are dropped. These tests pin the properties that make the added configs
safe: they appear only on gfx942, the installed Triton accepts the backend
option they rely on, and each one is checked individually against the stock
result rather than only the one the autotuner happens to choose today.
"""

from __future__ import annotations

import contextlib

import pytest
import torch
import triton

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import chunk_delta_h, chunk_o
from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule
from vllm.third_party.flash_linear_attention.ops.utils import check_shared_mem

# A small shape, enough to compile every config and compare results.
B, T, HG, H, K, V = 1, 1024, 2, 4, 128, 128

# Whether an added config *wins* depends on the head count far more than on the
# sequence length: at HG=2/H=4 the grid is a few workgroups, the kernels are
# latency-bound and the whole space is flat, so the autotuner's pick there is
# noise. Measured on MI325X, stock vs. the full space:
#
#     T=1024 HG=2  H=4    1.1526 -> 1.3715 ms   0.840x   no added config won
#     T=4096 HG=2  H=4    1.8668 -> 2.3300 ms   0.801x   no added config won
#     T=8192 HG=2  H=4    4.6301 -> 5.4293 ms   0.853x   no added config won
#     T=4096 HG=16 H=48   2.4262 -> 2.2095 ms   1.098x   added config won
#     T=8192 HG=16 H=48   6.5200 -> 4.0948 ms   1.592x   added config won
#
# So the selection test uses Qwen3-Next's head counts. At T=8192 the margin is
# 1.592x, far outside the run-to-run spread, which is what makes that assertion
# safe to make in a test.
SELECT_T, SELECT_HG, SELECT_H = 8192, 16, 48


def _on_gfx942() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx942

    return on_gfx942()


def _autotuner(kernel):
    """Unwrap the triton.heuristics layers down to the Autotuner."""
    seen = kernel
    while not hasattr(kernel, "configs"):
        kernel = getattr(kernel, "fn", None)
        if kernel is None:
            raise AssertionError(
                f"no triton.Autotuner under {seen!r}; the decorator stack on this "
                "kernel changed and these tests need updating"
            )
    return kernel


@contextlib.contextmanager
def _configs(kernel, configs):
    """Run with the autotuner restricted to ``configs``, then put it back."""
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
    """Autotune from scratch over the full space, then put the cache back."""
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
        assert bool(module._GFX942_CONFIGS) == _on_gfx942()


@pytest.mark.skipif(not _on_gfx942(), reason="gfx942-specific configs")
def test_triton_accepts_the_backend_option_the_configs_rely_on():
    """``matrix_instr_nonkdim`` is a backend option, not a kernel argument.

    Triton's AMD backend lifts it out of the config kwargs at compile time. A
    Triton that does not recognize it raises rather than being skipped -- the
    autotuner only swallows ``OutOfResources`` -- so every GDN prefill on gfx942
    would fail at the first autotune. This test names that dependency, so the
    failure arrives here with an explanation instead of at a model launch.
    """
    kernel = chunk_o.chunk_fwd_kernel_o
    one_added = _added(kernel)[:1]
    assert one_added, "no added configs to compile"

    # Restricting the space to a single added config makes this a compile of that
    # config and nothing else, so a failure here is about the option.
    with _configs(kernel, one_added):
        chunk_gated_delta_rule(**_inputs("cuda"))


@pytest.mark.skipif(not _on_gfx942(), reason="gfx942-specific configs")
def test_added_configs_restore_the_tile_the_shared_memory_gate_hides():
    # The premise of the change: check_shared_mem() measures the device's LDS
    # against a threshold taken from a table with only NVIDIA entries, so it is
    # False on gfx942 and the stock space tops out below 128. These tiles win the
    # isolated long-sequence benchmark; at a served chunk size the autotuner
    # prefers a stock tile with the MFMA width.
    #
    # The gate's value is upstream's to change -- a fix there would be an
    # improvement, not a regression -- so it is reported, not asserted. What is
    # asserted is the part this change owns: the wide tile is reachable only
    # through the configs added here.
    if check_shared_mem():
        pytest.skip(
            "check_shared_mem() is now True on gfx942: upstream's threshold "
            "changed and the stock space already carries the 128-wide tiles, so "
            "this test's premise is stale"
        )
    assert chunk_o.BKV_LIST == [32, 64]

    stock_tiles = {
        (c.kwargs["BK"], c.kwargs["BV"]) for c in _stock(chunk_o.chunk_fwd_kernel_o)
    }
    added_tiles = {
        (c.kwargs["BK"], c.kwargs["BV"]) for c in _added(chunk_o.chunk_fwd_kernel_o)
    }
    assert (64, 128) not in stock_tiles
    assert (64, 128) in added_tiles

    # chunk_delta_h keeps its tile and stage count. What the stock space cannot
    # reach is the MFMA width and, only in combination with it, num_warps=8 --
    # the warp extension on its own measures 0.982x, slightly slower.
    kernel = chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    assert max(c.num_warps for c in _stock(kernel)) == 4
    assert max(c.num_warps for c in _added(kernel)) == 8


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
    """Run with exactly one config pinned per kernel.

    Returns ``None`` when the config does not fit in LDS. The autotuner swallows
    exactly that error, so such a config is unreachable in production and is not
    this test's business; pinning it is what makes it visible.
    """
    kernel_o, kernel_h = _kernels()
    try:
        with _configs(kernel_o, [o_config]), _configs(kernel_h, [h_config]):
            out, state = chunk_gated_delta_rule(**inputs)
    except triton.runtime.errors.OutOfResources:
        return None
    return out.clone(), state.clone()


@pytest.fixture(scope="module")
def stock_reference():
    """Inputs and the stock space's result, as the baseline for every config."""
    inputs = _inputs("cuda")
    kernel_o, kernel_h = _kernels()
    got = _pinned(inputs, _stock(kernel_o)[0], _stock(kernel_h)[0])
    assert got is not None, "the stock reference config does not fit in LDS"
    return inputs, got[0], got[1]


def _added_cases():
    """(kernel, config) for every added config, or [] off gfx942."""
    if not _on_gfx942():
        return []
    kernel_o, kernel_h = _kernels()
    return [(_KERNEL_O, c) for c in _added(kernel_o)] + [
        (_KERNEL_H, c) for c in _added(kernel_h)
    ]


@pytest.mark.skipif(not _on_gfx942(), reason="gfx942-specific configs")
@pytest.mark.parametrize(
    "kernel_name,config", _added_cases(), ids=lambda v: str(v).replace(" ", "")
)
def test_added_config_matches_the_stock_result(kernel_name, config, stock_reference):
    """Every added config, checked individually against the stock result.

    Bit-identity is measured, not guaranteed: a config is a scheduling choice,
    but BK is the K-loop tile, so changing it would re-associate an fp32
    reduction. Every config added here keeps the stock BK, and all of them are
    bit-identical on MI325X. A BK=128 family was measured and left out for
    exactly this reason -- it differs in the last bits of a handful of elements.

    Asserting it per config is the point: whether the end-to-end result is
    bit-identical depends otherwise on which config the autotuner picks for the
    shape under test, which is not a property of this change.
    """
    inputs, ref_out, ref_state = stock_reference
    kernel_o, kernel_h = _kernels()
    if kernel_name == _KERNEL_O:
        got = _pinned(inputs, config, _stock(kernel_h)[0])
    else:
        got = _pinned(inputs, _stock(kernel_o)[0], config)
    if got is None:
        pytest.skip(f"{config} does not fit in LDS; the autotuner skips it too")
    out, state = got

    # The recurrent state is carried across chunks, so it must match exactly for
    # every config -- a difference there would compound down the sequence.
    assert torch.equal(ref_state, state), f"final state changed under {config}"
    assert torch.equal(ref_out, out), f"output changed under {config}"


@pytest.mark.skipif(not _on_gfx942(), reason="gfx942-specific configs")
@pytest.mark.parametrize(
    "kernel_name",
    ["chunk_fwd_kernel_o", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"],
)
def test_autotuner_selects_an_added_config(kernel_name):
    """If the added configs never win there is no reason to carry them.

    Run at Qwen3-Next's head counts, not at this file's small shape -- see the
    note by SELECT_T. At HG=2/H=4 the space is flat and the autotuner's pick is
    noise, so the same assertion there fails about as often as it passes and says
    nothing about the change.

    This is the one assertion here that is timed rather than structural, so it can
    still fail for reasons outside this change: a contended GPU has been seen to
    move the stock arm by 13%.
    """
    module = chunk_o if kernel_name == "chunk_fwd_kernel_o" else chunk_delta_h
    kernel = getattr(module, kernel_name)

    with _fresh_cache(kernel) as tuner:
        chunk_gated_delta_rule(**_inputs("cuda", t=SELECT_T, hg=SELECT_HG, h=SELECT_H))
        assert "matrix_instr_nonkdim" in tuner.best_config.kwargs, (
            f"the autotuner preferred a stock config for {kernel_name}: "
            f"{tuner.best_config}. Check whether the GPU is contended before "
            "reading this as the added configs not being worth carrying."
        )
