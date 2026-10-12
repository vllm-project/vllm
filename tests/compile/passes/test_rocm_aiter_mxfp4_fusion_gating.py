# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gating and start-up checks of the ROCm AITER MXFP4 activation-quant fusion passes.

Covers what the two passes share: the opt-in env flags and the rule that registers a
pass, ``is_target_model_graph``, ``TargetGraphSiteCheckMixin`` (which counts rewritten
sites per compile range and judges them once the whole graph is compiled), and the
``PostGradPassManager`` and ``VllmBackend`` hooks that feed it.

The env flag, pass manager and backend tests run on any platform. The tests of the
fusion module need ROCm with AITER, because its patterns name AITER custom ops; they
stand in for the GPU architecture query and the other passes. The last tests run the
real act+quant pass on a small graph and need a gfx950.
"""

import types
from pathlib import Path
from typing import Any

import pytest
import regex as re
import torch
from torch import fx

# This import automatically registers `torch.ops.silly.attention`
import tests.compile.silly_attention  # noqa: F401
import vllm
from vllm import envs
from vllm.compilation.decorators import support_torch_compile
from vllm.compilation.passes.inductor_pass import (
    InductorPass,
    get_pass_context,
    pass_context,
)
from vllm.compilation.passes.pass_manager import PostGradPassManager
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.utils import Range, hash_factors
from vllm.forward_context import set_forward_context
from vllm.platforms import current_platform

ACT_FLAG = "VLLM_ROCM_AITER_FP4_ACT_QUANT_FUSION"
POSTNORM_FLAG = "VLLM_ROCM_AITER_FP4_POSTNORM_QUANT_FUSION"
FLAGS = [ACT_FLAG, POSTNORM_FLAG]

RANGE_A = Range(1, 8)
RANGE_B = Range(9, 64)


@pytest.fixture
def gfx950(monkeypatch, rocm_aiter_fusion):
    """Sets the GPU architecture the fusion gate sees; returns the setter."""
    import vllm.platforms.rocm as rocm_platform

    def set_arch(is_gfx950: bool) -> None:
        monkeypatch.setattr(rocm_platform, "on_gfx950", lambda: is_gfx950)

    set_arch(True)
    return set_arch


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

GATES = [
    pytest.param("act_mxfp4_pass_enabled", ACT_FLAG, id="act"),
    pytest.param("norm_mxfp4_pass_enabled", POSTNORM_FLAG, id="norm"),
]


@pytest.mark.parametrize(("gate", "flag"), GATES)
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
    rocm_aiter_fusion,
    gfx950,
    monkeypatch,
    gate,
    flag,
    flag_on,
    fuse,
    on_gfx950,
    enabled,
):
    monkeypatch.delenv(flag, raising=False)
    if flag_on:
        monkeypatch.setenv(flag, "1")
    gfx950(on_gfx950)
    assert getattr(rocm_aiter_fusion, gate)(fuse) is enabled


# --- which compile prefixes are the target model --------------------------------
#
# VllmBackend.prefix is `prefix or model_tag`; model_tag is "backbone" unless a loader
# wraps the model construction in set_model_tag(...). MTP and EAGLE heads are built
# under set_model_tag("eagle_head"), so their graph compiles with that prefix.

# Parts of one target model that have their own tag.
MODEL_INTERNAL_TAGS = frozenset({"self_decoder", "cross_decoder"})


def test_only_speculative_decoding_graphs_are_not_target_graphs(rocm_aiter_fusion):
    # Input: the prefix of the model, of its internal parts, and of draft graphs.
    # Expected: only draft graphs are not target graphs. Why: the MTP head is bf16, so
    # finding no site there is correct and must not stop the start.
    is_target = rocm_aiter_fusion.is_target_model_graph
    for prefix in ("backbone", "", "language_model", *MODEL_INTERNAL_TAGS):
        assert is_target(prefix), prefix
    for prefix in rocm_aiter_fusion.DRAFT_GRAPH_PREFIXES:
        assert not is_target(prefix), prefix
    assert is_target("eagle_head") is False


def test_every_set_model_tag_in_the_source_tree_is_classified(rocm_aiter_fusion):
    # Input: every set_model_tag("...") literal in the vllm package. Expected: exactly
    # the draft prefixes plus the model-internal tags. Why: a new tag nobody classified
    # would count as a target graph, and a bf16 draft graph carrying it would stop the
    # start; adding a tag has to force a decision here.
    pattern = re.compile(r'set_model_tag\(\s*"([^"]+)"')
    found = {
        tag
        for path in Path(vllm.__file__).parent.rglob("*.py")
        for tag in pattern.findall(path.read_text(encoding="utf-8"))
    }
    assert found == rocm_aiter_fusion.DRAFT_GRAPH_PREFIXES | MODEL_INTERNAL_TAGS


# --- the site check mixin -----------------------------------------------------


class _FakeMatcherPass:
    """Stands in for VllmFusionPatternMatcherPass: reports a preset number of matches
    on every call, the way ``matched_count`` does after ``pm_pass.apply``.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.matched_count = 0
        self.matches_per_call = 0

    def __call__(self, graph) -> None:
        self.matched_count = self.matches_per_call


class _LogRecorder:
    """Replaces the module logger and keeps the formatted messages by level."""

    def __init__(self) -> None:
        self.info_lines: list[str] = []
        self.warning_lines: list[str] = []

    def info(self, msg, *args) -> None:
        self.info_lines.append(msg % args)

    def warning(self, msg, *args) -> None:
        self.warning_lines.append(msg % args)

    def fused_lines(self) -> list[str]:
        return [line for line in self.info_lines if line.startswith("Fused ")]


@pytest.fixture
def log(rocm_aiter_fusion, monkeypatch):
    recorder = _LogRecorder()
    monkeypatch.setattr(rocm_aiter_fusion, "logger", recorder)
    return recorder


@pytest.fixture
def make_pass(rocm_aiter_fusion):
    """Builds a mixin pass for a graph prefix; the test sets its matches per call."""
    checked_pass = type(
        "_CheckedPass",
        (rocm_aiter_fusion.TargetGraphSiteCheckMixin, _FakeMatcherPass),
        {"flag_name": "VLLM_TEST_FUSION_FLAG", "site_label": "test fusion"},
    )
    return lambda prefix=None: checked_pass(compile_prefix=prefix)


def _run_piece(fusion_pass, sites: int, compile_range: Range = RANGE_A) -> None:
    """One call of the pass, as the pass manager makes it for one piece and range."""
    fusion_pass.matches_per_call = sites
    with pass_context(compile_range):
        fusion_pass(fx.Graph())


def test_sites_are_logged_per_compile_range_over_all_pieces(make_pass, log):
    # Input: a backbone graph; range A has a piece without a site (embedding, layer 0)
    # and one with 2, range B one with 5. Expected: nothing raises per piece, and each
    # range is logged once with its own total. Why: the same sites are rewritten once
    # per range, and a model is judged as a whole graph, not piece by piece.
    fusion_pass = make_pass()  # no prefix: vLLM's default model tag, "backbone"
    _run_piece(fusion_pass, 0, RANGE_A)
    _run_piece(fusion_pass, 2, RANGE_A)
    _run_piece(fusion_pass, 5, RANGE_B)
    fusion_pass.finish_compile()
    assert log.fused_lines() == [
        f"Fused test fusion at 2 sites (graph=backbone) for compile range {RANGE_A}",
        f"Fused test fusion at 5 sites (graph=backbone) for compile range {RANGE_B}",
    ]


def test_target_graph_with_a_range_without_a_site_fails_naming_the_flag(make_pass, log):
    # Input: range A rewrote 2 sites, range B ran and rewrote none. Expected:
    # finish_compile raises, names the flag and range B. Why: the flag is on, so a
    # start that leaves a range unfused must not pass silently.
    fusion_pass = make_pass("backbone")
    _run_piece(fusion_pass, 2, RANGE_A)
    _run_piece(fusion_pass, 0, RANGE_B)
    with pytest.raises(RuntimeError, match=r"VLLM_TEST_FUSION_FLAG is on but 0 sites"):
        fusion_pass.finish_compile()
    with pytest.raises(RuntimeError, match=re.escape(str(RANGE_B))):
        fusion_pass.finish_compile()


def test_draft_graph_without_a_site_only_logs(make_pass, log):
    # Input: the eagle_head graph, the pass ran and rewrote nothing. Expected: a log
    # line with 0 sites, no error. Why: the MTP head is bf16, no site is correct.
    fusion_pass = make_pass("eagle_head")
    _run_piece(fusion_pass, 0)
    fusion_pass.finish_compile()
    assert log.fused_lines() == [
        f"Fused test fusion at 0 sites (graph=eagle_head) for compile range {RANGE_A}"
    ]


def test_target_graph_the_pass_never_ran_on_only_warns(make_pass, log, monkeypatch):
    # Input: a backbone graph on which the pass never ran (Inductor's cache served
    # every piece). Expected: no error, one warning that names the flag and the cache
    # directory. Why: nothing was rewritten or checked, but a restart that reuses the
    # cache is legitimate, so it cannot stop the start.
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", "/tmp/vllm-test-cache")
    make_pass("backbone").finish_compile()
    (warning,) = log.warning_lines
    assert "VLLM_TEST_FUSION_FLAG is on" in warning
    assert "/tmp/vllm-test-cache/torch_compile_cache" in warning
    assert log.fused_lines() == []


# --- the pass manager and the backend -----------------------------------------


def test_pass_manager_finish_compile_calls_the_passes_that_have_it():
    # Input: a pass manager holding one pass with finish_compile and one without.
    # Expected: the first is called once, the second is skipped without an error.
    calls = []
    manager = PostGradPassManager()
    passes: list[Any] = [
        types.SimpleNamespace(finish_compile=lambda: calls.append("called")),
        types.SimpleNamespace(),
    ]
    manager.passes = passes
    manager.finish_compile()
    assert calls == ["called"]


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
    """A pass the manager registers: remembers the prefix it was built with."""

    def __init__(self, config, compile_prefix=None) -> None:
        self.compile_prefix = compile_prefix


class _ActMulMxfp4Pass(_StandInPass):
    pass


class _NormMxfp4Pass(_StandInPass):
    pass


@pytest.fixture
def configure_manager(monkeypatch, gfx950, rocm_aiter_fusion):
    """Returns configure(prefix, **switches) -> the passes the manager registers.

    pass_manager imports the ROCm passes only on a platform that has them, so the
    passes configure() names are stand-ins; the gate functions are the real ones.
    """
    from vllm.compilation.passes import pass_manager

    for name in ("act_mxfp4_pass_enabled", "norm_mxfp4_pass_enabled"):
        monkeypatch.setattr(
            pass_manager, name, getattr(rocm_aiter_fusion, name), raising=False
        )
    for name, stand_in in (
        ("RocmAiterActMulMxfp4GemmFusionPass", _ActMulMxfp4Pass),
        ("RocmAiterRMSNormMxfp4GemmFusionPass", _NormMxfp4Pass),
        ("ActivationQuantFusionPass", _StandInPass),
        ("RocmAiterSiluMulFp8GroupQuantFusionPass", _StandInPass),
        ("RMSNormQuantFusionPass", _StandInPass),
        ("RocmAiterRMSNormQuantFusionPass", _StandInPass),
    ):
        monkeypatch.setattr(pass_manager, name, stand_in, raising=False)
    monkeypatch.setattr(
        pass_manager,
        "rocm_aiter_ops",
        types.SimpleNamespace(
            is_enabled=lambda: True, is_rdna_aiter_enabled=lambda: False
        ),
    )

    def configure(prefix: str | None = None, **switches: bool):
        config = VllmConfig(
            compilation_config=CompilationConfig(
                mode=CompilationMode.VLLM_COMPILE,
                pass_config=_pass_config_with_only(**switches),
            )
        )
        manager = PostGradPassManager()
        manager.compile_prefix = prefix
        with set_current_vllm_config(config):
            manager.configure(config)
        return manager.passes

    return configure


@pytest.mark.parametrize(
    ("flag", "switch", "pass_cls"),
    [
        (ACT_FLAG, "fuse_act_quant", _ActMulMxfp4Pass),
        (POSTNORM_FLAG, "fuse_norm_quant", _NormMxfp4Pass),
    ],
    ids=["act", "norm"],
)
def test_pass_manager_registers_a_pass_only_for_its_flag_and_switch(
    configure_manager, monkeypatch, flag, switch, pass_cls
):
    # Input: the flag of one pass unset, then set with its pass config switch off,
    # then set with the switch on. Expected: the pass is registered, with the compile
    # prefix of the graph, only in the last case, and the other pass never is. Why:
    # with every flag off the pass list must be the one a build without this fusion
    # has.
    for other in FLAGS:
        monkeypatch.delenv(other, raising=False)

    def mxfp4_passes(prefix=None, **switches):
        passes = configure_manager(prefix, **switches)
        return [p for p in passes if isinstance(p, (_ActMulMxfp4Pass, _NormMxfp4Pass))]

    assert mxfp4_passes(**{switch: True}) == []
    monkeypatch.setenv(flag, "1")
    assert mxfp4_passes(**{switch: False}) == []
    (registered,) = mxfp4_passes("backbone", **{switch: True})
    assert type(registered) is pass_cls
    assert registered.compile_prefix == "backbone"


@pytest.mark.parametrize("model_tag", ["backbone", "eagle_head"])
def test_backend_hands_its_compile_prefix_to_the_pass_manager(model_tag, monkeypatch):
    # Input: a VllmBackend built the way the loaders build it: inside
    # set_model_tag(...) for a draft head, outside it for the backbone. Expected: the
    # pass manager has the backend's prefix, which is the tag, when it is configured.
    # Why: the passes tell the draft graph from the target graph by that prefix.
    from contextlib import nullcontext

    from vllm.compilation.backends import VllmBackend, set_model_tag

    seen: list[str | None] = []
    monkeypatch.setattr(
        PostGradPassManager,
        "configure",
        lambda self, config: seen.append(self.compile_prefix),
    )
    config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    )
    tag = nullcontext() if model_tag == "backbone" else set_model_tag(model_tag)
    with set_current_vllm_config(config), tag:
        backend = VllmBackend(config)
    with set_current_vllm_config(config):
        backend.configure_post_pass()
    assert backend.prefix == model_tag
    assert seen == [model_tag]


class _PlatformPassManager:
    """What a platform plugin's pass manager (get_pass_manager_cls) can look like: it
    knows neither the compile prefix nor finish_compile.
    """

    def __init__(self) -> None:
        self.configured = False

    def configure(self, config: VllmConfig) -> None:
        self.configured = True

    def add(self, pass_) -> None:
        pass


def test_backend_still_configures_a_platform_pass_manager_without_the_new_hooks(
    monkeypatch,
):
    # Input: a platform whose pass manager class has configure(config) only. Expected:
    # the backend configures it without a TypeError. Why: the prefix is handed over as
    # an attribute and finish_compile is optional, so out-of-tree pass managers keep
    # working.
    from vllm.compilation.backends import VllmBackend

    monkeypatch.setattr(
        type(current_platform),
        "get_pass_manager_cls",
        classmethod(lambda cls: f"{__name__}._PlatformPassManager"),
    )
    config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    )
    with set_current_vllm_config(config):
        backend = VllmBackend(config)
        backend.configure_post_pass()
    assert isinstance(backend.pass_manager, _PlatformPassManager)
    assert backend.pass_manager.configured


class _RecordingPass(InductorPass):
    """A custom post-grad pass that records when it runs and when it is finished."""

    def __init__(self, events: list) -> None:
        self.events = events

    def __call__(self, graph: fx.Graph) -> None:
        self.events.append(("pass", str(get_pass_context().compile_range)))

    def finish_compile(self) -> None:
        self.events.append("finish_compile")

    def uuid(self) -> str:
        return InductorPass.hash_dict({})


@support_torch_compile
class _TwoPieceModel(torch.nn.Module):
    """Two compiled pieces: the part before the attention op and the part after."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", **kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + x
        attn_output = torch.empty_like(x)
        torch.ops.silly.attention(x, x, x, attn_output)
        return attn_output * 3


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="the CPU platform compiles with stock torch.compile, not VllmBackend",
)
def test_finish_compile_runs_once_after_every_piece_and_before_the_cache_is_saved(
    disable_vllm_compile_cache, monkeypatch
):
    # Input: a model that compiles as two pieces, split at an attention op, with a
    # custom post-grad pass that records its calls. Expected: the pass sees both
    # pieces, then finish_compile runs exactly once, and only then is the compile
    # cache index saved. Why: a pass can judge the whole graph only after the last
    # piece, and a start that fails that judgement must not leave a cache entry that
    # the next start would take without running the pass.
    from vllm.compilation.backends import CompilerManager

    events: list = []
    save_to_file = CompilerManager.save_to_file

    def recording_save(self) -> None:
        events.append("save_to_file")
        save_to_file(self)

    monkeypatch.setattr(CompilerManager, "save_to_file", recording_save)

    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            splitting_ops=["silly::attention"],
            pass_config=_pass_config_with_only(),
            inductor_compile_config={
                "post_grad_custom_post_pass": _RecordingPass(events)
            },
        )
    )
    with set_current_vllm_config(config), torch.device(current_platform.device_type):
        model = _TwoPieceModel(vllm_config=config, prefix="").eval()
        with set_forward_context({}, vllm_config=config):
            model(torch.randn(64, 128))

    last_pass = max(i for i, e in enumerate(events) if isinstance(e, tuple))
    assert events.count("finish_compile") == 1
    assert last_pass > 0, "the pass should run on both pieces"
    assert last_pass < events.index("finish_compile") < events.index("save_to_file")


# --- the real act+quant pass on a small graph (gfx950) ---------------------------


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


requires_gfx950_aiter = pytest.mark.skipif(
    not _on_gfx950(), reason="ROCm with AITER on gfx950 only"
)


@pytest.fixture
def make_act_pass(monkeypatch, rocm_aiter_fusion):
    """Builds the real act pass for a graph prefix, as a bf16 model gets it.

    The pattern is traced with the default dtype (there is no model config here), so
    the default is bf16 while the pass is built; otherwise the traced SiLU lacks the
    fp32 round trip bf16 graphs have and nothing matches.
    """
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    rocm_aiter_ops.refresh_env_variables()
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["none"],
            backend="eager",
            pass_config=PassConfig(fuse_act_quant=True, eliminate_noops=True),
        )
    )
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with set_current_vllm_config(config):
            yield lambda prefix: rocm_aiter_fusion.RocmAiterActMulMxfp4GemmFusionPass(
                config, compile_prefix=prefix
            )
    finally:
        torch.set_default_dtype(previous_dtype)
        monkeypatch.undo()
        rocm_aiter_ops.refresh_env_variables()


def _act_mul_gemm_piece(with_act_and_mul: bool) -> fx.GraphModule:
    """One graph piece, as aten ops: [SiLU-and-mul ->] gemm_with_dynamic_quant."""
    from vllm.compilation.passes.inductor_pass import enable_fake_mode
    from vllm.compilation.passes.vllm_inductor_pass import VllmFusionPatternMatcherPass
    from vllm.model_executor.layers.activation import SiluAndMul

    @enable_fake_mode
    def trace() -> fx.GraphModule:
        m, d, n = 4, 64, 32
        width = 2 * d if with_act_and_mul else d
        device = current_platform.device_type
        x = torch.empty(m, width, dtype=torch.bfloat16, device=device)
        weight = torch.empty(n, d // 2, dtype=torch.uint8, device=device)
        weight_scale = torch.empty(n, d // 32, dtype=torch.uint8, device=device)

        def piece(x, weight, weight_scale):
            y = SiluAndMul().forward_native(x) if with_act_and_mul else x
            return torch.ops.vllm.gemm_with_dynamic_quant(
                y, weight, weight_scale, True, torch.bfloat16
            )

        return VllmFusionPatternMatcherPass._trace_fn(piece, [x, weight, weight_scale])

    return trace()


@requires_gfx950_aiter
def test_real_act_pass_logs_the_site_it_rewrote(make_act_pass, log):
    # Input: a piece with SiLU-and-mul feeding an asm MXFP4 GEMM, in a backbone graph.
    # Expected: the GEMM takes the gate/up tensor and the activation name, and the
    # pass logs 1 site for the range.
    act_pass = make_act_pass("backbone")
    gm = _act_mul_gemm_piece(with_act_and_mul=True)
    with pass_context(RANGE_A):
        act_pass(gm.graph)
    (gemm,) = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.vllm.gemm_with_dynamic_quant.default
    ]
    assert gemm.args[-1] == "silu"
    act_pass.finish_compile()
    assert log.fused_lines() == [
        f"Fused act+mxfp4 quant at 1 sites (graph=backbone) for compile range {RANGE_A}"
    ]


@requires_gfx950_aiter
def test_real_act_pass_fails_a_target_graph_without_a_site(make_act_pass, log):
    # Input: a backbone graph whose only piece has a GEMM but no SiLU-and-mul.
    # Expected: finish_compile raises and names the flag.
    act_pass = make_act_pass("backbone")
    gm = _act_mul_gemm_piece(with_act_and_mul=False)
    with pass_context(RANGE_A):
        act_pass(gm.graph)
    with pytest.raises(RuntimeError, match=f"{ACT_FLAG} is on but 0 sites"):
        act_pass.finish_compile()
