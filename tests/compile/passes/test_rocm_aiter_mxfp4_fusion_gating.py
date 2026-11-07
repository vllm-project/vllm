# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gating and registration of the ROCm AITER MXFP4 activation-quant fusion pass.

Covers the opt-in env flag, the gate that says whether the pass is registered
(``act_mxfp4_pass_registered`` in vllm/_aiter_ops.py: AITER enabled, the flag,
``fuse_act_quant`` and gfx950), the pass manager's rule for which AITER platforms
register it, and the compile ranges the pass is limited to: the ranges split at
``MXFP4_PRESHUFFLE_MAX_M`` tokens exactly when the pass is registered, and the pass
applies only above that limit. The pass manager tests stand in for the passes it names,
because the real ones import only where ROCm and AITER are present. The fusion pass
itself is tested in test_rocm_aiter_act_mul_mxfp4_fusion.py.
"""

import itertools

import pytest
import torch

from vllm import envs
from vllm.compilation.passes.inductor_pass import pass_context
from vllm.compilation.passes.pass_manager import PostGradPassManager
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    PassConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.utils import Range, hash_factors
from vllm.platforms import current_platform

ACT_FLAG = "VLLM_ROCM_USE_AITER_FP4_ACT_QUANT_FUSION"
FLAGS = [ACT_FLAG]


@pytest.fixture
def gfx950(monkeypatch, rocm_aiter_fusion):
    """Sets the GPU architecture the fusion gate sees; returns the setter."""
    import vllm.platforms.rocm as rocm_platform

    def set_arch(is_gfx950: bool) -> None:
        monkeypatch.setattr(rocm_platform, "on_gfx950", lambda: is_gfx950)

    set_arch(True)
    return set_arch


@pytest.fixture
def aiter_on(monkeypatch):
    """Sets what rocm_aiter_ops.is_enabled() answers; returns the setter.

    The class attribute is patched, so the gate, the pass manager and the compile range
    split all see the same answer for the rest of the test.
    """
    from vllm._aiter_ops import rocm_aiter_ops

    def set_enabled(enabled: bool) -> None:
        monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: enabled)

    set_enabled(True)
    return set_enabled


# --- the opt-in flags ---------------------------------------------------------


@pytest.mark.parametrize("flag", FLAGS)
def test_flag_defaults_off(flag, monkeypatch):
    # Input: the variable is not set. Expected: False, a default start is unchanged.
    monkeypatch.delenv(flag, raising=False)
    assert getattr(envs, flag) is False


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1", True), ("true", True), ("TRUE", True), ("0", False), ("yes", False)],
)
def test_flag_parses_value(flag, raw, expected, monkeypatch):
    # Only 1 or true (any case) is on, like the other VLLM_ROCM_USE_AITER_* flags, so
    # a typo never switches a fused path on silently.
    monkeypatch.setenv(flag, raw)
    assert getattr(envs, flag) is expected


@pytest.mark.parametrize("flag", FLAGS)
def test_flag_changes_the_compile_cache_key(flag, monkeypatch):
    # Input: the compile factors with the flag off, then on. Expected: the flag is a
    # factor, the only one that changes, and the hash changes. Why: a graph compiled
    # without the fusion must never be replayed from the cache with the flag on.
    for other in FLAGS:
        monkeypatch.delenv(other, raising=False)
    monkeypatch.setenv(flag, "0")
    off = envs.compile_factors()
    monkeypatch.setenv(flag, "1")
    on = envs.compile_factors()

    assert flag in off, f"{flag} is not a compile cache factor"
    assert {name for name in on if on[name] != off.get(name)} == {flag}
    assert hash_factors(off) != hash_factors(on)


# --- when a pass is registered ------------------------------------------------


@pytest.mark.parametrize(
    ("flag_on", "fuse", "on_gfx950", "enabled"),
    [
        (False, True, True, False),  # the default optimization level sets fuse
        (True, False, True, False),  # the flag adds to the pass config, not replaces
        (True, True, False, False),  # the asm MXFP4 GEMM exists only on gfx950
        (True, True, True, True),
    ],
)
def test_pass_needs_flag_pass_config_and_gfx950(
    gfx950,
    aiter_on,
    monkeypatch,
    flag_on,
    fuse,
    on_gfx950,
    enabled,
):
    # Input: the flag, fuse_act_quant and gfx950 each on or off, with AITER enabled.
    # Expected: the gate is open only when all three are on. Why: the pass rewrites the
    # asm MXFP4 GEMM, which exists only on gfx950, and the default level sets fuse.
    from vllm._aiter_ops import act_mxfp4_pass_registered

    monkeypatch.delenv(ACT_FLAG, raising=False)
    if flag_on:
        monkeypatch.setenv(ACT_FLAG, "1")
    gfx950(on_gfx950)
    assert act_mxfp4_pass_registered(fuse) is enabled


# Every fusion switch of PassConfig that PostGradPassManager.configure reads.
FUSION_SWITCHES = [
    "fuse_act_quant",
    "fuse_norm_quant",
    "fuse_attn_quant",
    "eliminate_noops",
    "fuse_allreduce_rms",
    "enable_qk_norm_rope_fusion",
    "fuse_rope_kvcache_cat_mla",
    "fuse_act_padding",
    "fuse_mla_dual_rms_norm",
    "fuse_rope_kvcache",
    "fuse_qk_norm_rope_kvcache",
]


def _pass_config_with_only(**switches: bool) -> PassConfig:
    """A PassConfig with every fusion switch off except the ones given."""
    return PassConfig(**{**dict.fromkeys(FUSION_SWITCHES, False), **switches})


class _StandInPass:
    """A pass the manager registers in place of the real one, which needs ROCm."""

    def __init__(self, config) -> None:
        self.config = config


class _ActMulMxfp4Pass(_StandInPass):
    """Stands in for the act MXFP4 pass; the tests only count its registration.

    Its constructor is the inherited one, so it takes the config alone, as the real
    pass does: a call that passes more arguments fails here as it would there.
    """


@pytest.fixture
def configure_manager(monkeypatch, gfx950, rocm_aiter_fusion):
    """Returns configure(*, aiter, rdna, **switches) -> the registered passes.

    ``aiter`` and ``rdna`` answer rocm_aiter_ops.is_enabled() and
    is_rdna_aiter_enabled(). They are set on the rocm_aiter_ops class, so the pass
    manager and the act pass gate see the same answers. The gfx950 fixture always
    applies (this fixture depends on it), so the modelled GPU is a gfx950 whatever the
    two answers; gfx12 is not modelled here. pass_manager imports the ROCm passes only
    on a platform that has them, so the passes configure() names are stand-ins; the
    gate functions are the real ones.
    """
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.compilation.passes import pass_manager

    for name, stand_in in (
        ("RocmAiterActMulMxfp4GemmFusionPass", _ActMulMxfp4Pass),
        ("ActivationQuantFusionPass", _StandInPass),
        ("RocmAiterSiluMulFp8GroupQuantFusionPass", _StandInPass),
    ):
        monkeypatch.setattr(pass_manager, name, stand_in, raising=False)

    def configure(*, aiter: bool = True, rdna: bool = False, **switches: bool):
        monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: aiter)
        monkeypatch.setattr(rocm_aiter_ops, "is_rdna_aiter_enabled", lambda: rdna)
        config = VllmConfig(
            compilation_config=CompilationConfig(
                mode=CompilationMode.VLLM_COMPILE,
                pass_config=_pass_config_with_only(**switches),
            )
        )
        manager = PostGradPassManager()
        with set_current_vllm_config(config):
            manager.configure(config)
        return manager.passes

    return configure


def test_pass_manager_registers_a_pass_only_for_its_flag_and_switch(
    configure_manager, monkeypatch
):
    # Input: the flag unset, then set with its pass config switch off, then set with
    # the switch on. Expected: the pass is registered only in the last case. Why: with
    # every flag off the pass list must be the one a build without this fusion has.
    for other in FLAGS:
        monkeypatch.delenv(other, raising=False)

    def mxfp4_passes(**switches):
        passes = configure_manager(**switches)
        return [p for p in passes if isinstance(p, _ActMulMxfp4Pass)]

    assert mxfp4_passes(fuse_act_quant=True) == []
    monkeypatch.setenv(ACT_FLAG, "1")
    assert mxfp4_passes(fuse_act_quant=False) == []
    (registered,) = mxfp4_passes(fuse_act_quant=True)
    assert type(registered) is _ActMulMxfp4Pass


def test_act_mxfp4_pass_is_not_registered_on_rdna_aiter(configure_manager, monkeypatch):
    # Input: fuse_act_quant on and the opt-in flag on, with the gate open on gfx950 (the
    # gfx950 fixture sets it) and AITER reporting RDNA only: is_enabled() False and
    # is_rdna_aiter_enabled() True. Expected: pass_manager does not register the act
    # MXFP4 pass, while the same switches register it where is_enabled() is True. Why:
    # gfx12 is the motivation, since its asm MXFP4 GEMM path does not exist, so the pass
    # must register only under is_enabled().
    monkeypatch.setenv(ACT_FLAG, "1")

    def mxfp4_passes(**platform):
        passes = configure_manager(fuse_act_quant=True, **platform)
        return [p for p in passes if isinstance(p, _ActMulMxfp4Pass)]

    assert len(mxfp4_passes(aiter=True)) == 1
    assert mxfp4_passes(aiter=False, rdna=True) == []


# --- the compile ranges the pass is limited to ---------------------------------


def _vllm_config(
    max_num_batched_tokens: int,
    user_endpoints: list[int] | None = None,
    **switches: bool,
) -> VllmConfig:
    """A VllmConfig with the given batch limit and fusion switches.

    Its compile ranges are computed when it is built, so the tests read them from it.
    """
    return VllmConfig(
        scheduler_config=SchedulerConfig.default_factory(
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=min(max_num_batched_tokens, 128),
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            pass_config=_pass_config_with_only(**switches),
            compile_ranges_endpoints=user_endpoints,
        ),
    )


@pytest.mark.parametrize(
    ("flag_on", "fuse", "on_gfx950", "aiter_enabled"),
    list(itertools.product([False, True], repeat=4)),
)
def test_compile_range_split_exists_exactly_when_the_pass_is_registered(
    gfx950,
    configure_manager,
    monkeypatch,
    flag_on,
    fuse,
    on_gfx950,
    aiter_enabled,
):
    # Input: each combination of the flag, fuse_act_quant, gfx950 and AITER, with a
    # batch limit of 8192 tokens. Expected: the compile ranges split at
    # MXFP4_PRESHUFFLE_MAX_M exactly when pass_manager registers the act pass.
    # Why: the split and the registration come from one rule, so a split with no
    # pass, or a pass with no split, can never happen. configure_manager sets AITER
    # on the class for the whole test.
    from vllm._aiter_ops import MXFP4_PRESHUFFLE_MAX_M

    monkeypatch.delenv(ACT_FLAG, raising=False)
    if flag_on:
        monkeypatch.setenv(ACT_FLAG, "1")
    gfx950(on_gfx950)
    passes = configure_manager(aiter=aiter_enabled, fuse_act_quant=fuse)
    registered = any(isinstance(p, _ActMulMxfp4Pass) for p in passes)

    config = _vllm_config(8192, fuse_act_quant=fuse)
    endpoints = config.compilation_config.compile_ranges_endpoints
    assert endpoints is not None, "_set_compile_ranges always sets the endpoints"
    assert (MXFP4_PRESHUFFLE_MAX_M in endpoints) is registered


@pytest.mark.parametrize(
    ("max_num_batched_tokens", "expected"),
    [(8192, [64, 8192]), (65, [64, 65]), (64, [64]), (32, [32])],
)
def test_compile_ranges_split_only_when_the_batch_limit_is_above_the_preshuffle_limit(
    gfx950, aiter_on, monkeypatch, max_num_batched_tokens, expected
):
    # Input: the flag on, gfx950, AITER enabled, and a batch limit of 8192, 65, 64 or 32
    # tokens. Expected: the limit is the last endpoint, and 64 comes before it only when
    # the limit is above 64. Why: when the limit is 64 or less no batch is above the
    # split, so an extra 64 would only duplicate the limit or add a range that never
    # applies.
    monkeypatch.setenv(ACT_FLAG, "1")
    config = _vllm_config(max_num_batched_tokens, fuse_act_quant=True)

    assert config.compilation_config.compile_ranges_endpoints == expected


def test_user_endpoint_at_the_preshuffle_limit_does_not_duplicate_the_split(
    gfx950, aiter_on, monkeypatch
):
    # Input: the pass on, a batch limit of 8192, and a user compile_ranges_endpoints of
    # [64], the same number as the split. Expected: the compile ranges are (1, 64) and
    # (65, 8192), with no empty or repeated range. Why: get_compile_ranges must treat a
    # repeated endpoint as one endpoint.
    monkeypatch.setenv(ACT_FLAG, "1")
    config = _vllm_config(8192, user_endpoints=[64], fuse_act_quant=True)

    assert config.compilation_config.get_compile_ranges() == [
        Range(1, 64),
        Range(65, 8192),
    ]


def test_act_mxfp4_pass_applies_only_to_ranges_above_the_preshuffle_limit(
    rocm_aiter_fusion,
):
    # Input: compile ranges that end at, start at, or sit just above the 64-token limit,
    # for a pass built with fuse_act_quant on. Expected: a range gets the fusion only
    # when every batch in it is above 64 tokens. Why: at 64 tokens or fewer the
    # rewritten GEMM takes the preshuffled path and only splits the activation
    # back out, so those ranges must compile the unfused graph.
    config = _vllm_config(8192, fuse_act_quant=True)
    with set_current_vllm_config(config):
        fusion_pass = rocm_aiter_fusion.RocmAiterActMulMxfp4GemmFusionPass(config)

    assert not fusion_pass.is_applicable_for_range(Range(1, 64))
    assert not fusion_pass.is_applicable_for_range(Range(64, 64))
    assert fusion_pass.is_applicable_for_range(Range(65, 65))
    assert fusion_pass.is_applicable_for_range(Range(65, 8192))


def test_pass_manager_runs_the_act_pass_only_for_ranges_above_the_limit(
    gfx950, aiter_on, rocm_aiter_fusion, monkeypatch
):
    # Input: the real act MXFP4 pass inside the real pass manager, with the flag on and
    # a batch limit of 8192, run for compile range (1, 64) and then for (65, 8192).
    # Expected: the pass body runs only for (65, 8192). Why: the range rule must reach
    # the pipeline, not only the is_applicable_for_range method. Without the manager's
    # per-range check, batches of 64 tokens or fewer get the rewrite again.
    monkeypatch.setenv(ACT_FLAG, "1")
    config = _vllm_config(8192, fuse_act_quant=True)
    calls = []
    monkeypatch.setattr(
        rocm_aiter_fusion.RocmAiterActMulMxfp4GemmFusionPass,
        "__call__",
        lambda self, graph: calls.append(graph),
    )
    with set_current_vllm_config(config):
        fusion_pass = rocm_aiter_fusion.RocmAiterActMulMxfp4GemmFusionPass(config)
    # The passes after the act pass need a compiled graph, so the manager's other steps
    # are stubbed out and the test reads only what the act pass saw.
    manager = PostGradPassManager()
    manager.passes = [fusion_pass]
    manager.post_cleanup = lambda graph: None
    manager.ir_lowering = lambda graph: None
    manager.clone_elimination = lambda graph: None
    manager.fix_functionalization = lambda graph: None
    graph = torch.fx.Graph()

    with pass_context(Range(1, 64)):
        manager(graph)
    assert calls == []

    with pass_context(Range(65, 8192)):
        manager(graph)
    assert len(calls) == 1


@pytest.mark.skipif(
    current_platform.is_rocm(),
    reason="checks the real AITER state of a non-ROCm host; ROCm can have AITER on",
)
def test_real_aiter_state_on_a_plain_host_gives_no_split_and_no_pass(monkeypatch):
    # Input: the flag and fuse_act_quant on, with the real rocm_aiter_ops.is_enabled()
    # (off on CPU and CUDA hosts) and a batch limit of 8192. Expected: the pass is not
    # registered and the compile ranges keep only the batch limit. Why: a CPU or CUDA
    # host must not split the ranges or register the act MXFP4 pass.
    from vllm._aiter_ops import act_mxfp4_pass_registered

    monkeypatch.setenv(ACT_FLAG, "1")
    assert not act_mxfp4_pass_registered(True)
    config = _vllm_config(8192, fuse_act_quant=True)
    assert config.compilation_config.compile_ranges_endpoints == [8192]


def test_aiter_forced_off_gives_no_split_and_no_pass_without_a_gfx950_patch(
    monkeypatch,
):
    # Input: the flag and fuse_act_quant on, AITER forced off, batch limit of 8192.
    # Expected: the predicate is False and no 64 endpoint is added. Why: the predicate
    # returns before it imports vllm.platforms.rocm, which queries the GPU at import, so
    # this holds on a plain CPU host with no gfx950 patch.
    from vllm._aiter_ops import (
        MXFP4_PRESHUFFLE_MAX_M,
        act_mxfp4_pass_registered,
        rocm_aiter_ops,
    )

    monkeypatch.setenv(ACT_FLAG, "1")
    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: False)
    assert act_mxfp4_pass_registered(True) is False
    config = _vllm_config(8192, fuse_act_quant=True)
    endpoints = config.compilation_config.compile_ranges_endpoints
    assert endpoints is not None, "_set_compile_ranges always sets the endpoints"
    assert MXFP4_PRESHUFFLE_MAX_M not in endpoints
