# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests of RocmAiterRMSNormMxfp4GemmFusionPass, the post-attention norm fusion.

The pass rewrites ``fused_add_rms_norm -> gemm_with_dynamic_quant`` (the Gemma
post-attention norm feeding the MLP's gate_up projection) into the custom op
``rocm_aiter_rmsnorm_mxfp4_quant``, whose codes and scales feed the GEMM through its
``x_scales`` argument.

The toy graphs are traced the way the pass sees a compiled piece (an aten graph with
fake tensor metadata). They need the AITER custom ops, so the file needs ROCm with
AITER; only the last test, which runs the rewritten graph, needs a gfx950 GPU.
"""

import inspect
import operator
import types

import pytest
import torch
from torch import fx

from vllm._aiter_ops import ROCM_AITER_MXFP4_NORM_QUANT_MAX_K, rocm_aiter_ops
from vllm.compilation.passes.inductor_pass import enable_fake_mode, pass_context
from vllm.compilation.passes.vllm_inductor_pass import VllmFusionPatternMatcherPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.utils import Range
from vllm.platforms import current_platform

pytestmark = pytest.mark.usefixtures("rocm_aiter_fusion")

POSTNORM_FLAG = "VLLM_ROCM_AITER_FP4_POSTNORM_QUANT_FUSION"
EPS = 1e-6
RANGE = Range(1, 8)
DEVICE = current_platform.device_type

# Toy sizes: K is a multiple of 32, which the op's fake requires.
TOY_M, TOY_K, TOY_N = 4, 64, 32


def _norm_op():
    return torch.ops.vllm_ir.fused_add_rms_norm.default


def _fused_op():
    return torch.ops.vllm.rocm_aiter_rmsnorm_mxfp4_quant.default


def _gemm_op():
    return torch.ops.vllm.gemm_with_dynamic_quant.default


# --- toy graphs: the stretches of a decoder layer the pass looks at ---------------
# A compiled piece starts with the previous layer's output projection, then comes the
# post-attention norm (its residual is a piece input), the gate_up GEMM, the down
# GEMM, and the NEXT layer's input norm, whose residual is the post-attention norm's
# residual output. The functions are those stretches, written with the same calls as
# the model; their argument names pick the tensors _toy_args builds.


def _norm(x, residual, w, eps=EPS):
    # GemmaRMSNorm.forward_native: the weight reaches the IR op as fp32 w + 1.0
    return torch.ops.vllm_ir.fused_add_rms_norm(x, residual, w.float() + 1.0, eps)


def _gemm(x, wq, ws, asm=True):
    return torch.ops.vllm.gemm_with_dynamic_quant(x, wq, ws, asm, torch.bfloat16)


def _post_site(asm=True, eps=EPS):
    """norm(x, residual) -> gate_up GEMM; the residual output is returned too."""

    def piece(x, residual, w, wq, ws):
        y, residual_out = _norm(x, residual, w, eps)
        return _gemm(y, wq, ws, asm), residual_out

    return piece


def _post_then_input_site(second_consumer=False):
    """The post-attention site, then the next layer's input norm and its GEMM(s).

    second_consumer=True is a gated-delta-net input norm, whose output feeds two GEMMs.
    """

    def piece(x, residual, w, wq, ws, wd, wsd, w2, wq2, ws2, wq3, ws3):
        y, residual_out = _norm(x, residual, w)
        down = _gemm(_gemm(y, wq, ws), wd, wsd)
        y2, residual_out2 = _norm(down, residual_out, w2)
        if second_consumer:
            return _gemm(y2, wq2, ws2), _gemm(y2, wq3, ws3), residual_out2
        return _gemm(y2, wq2, ws2), residual_out2

    return piece


def _post_with_two_consumers(x, residual, w, wq, ws, wq3, ws3):
    y, residual_out = _norm(x, residual, w)
    return _gemm(y, wq, ws), _gemm(y, wq3, ws3), residual_out


def _gemm_only(x, wq, ws):
    """The first piece of a model: a GEMM and no norm."""
    return _gemm(x, wq, ws)


def _toy_args(names, k, n, dtype, batch):
    """Fake tensors for the argument names of a toy function, shaped for K and N."""
    u8 = torch.uint8
    shapes = {
        "x": ((*batch, k), dtype),
        "residual": ((*batch, k), dtype),
        "w": ((k,), dtype),
        "w2": ((k,), dtype),
        "wq": ((n, k // 2), u8),  # gate_up-like weights; "down" maps back to k
        "ws": ((n, k // 32), u8),
        "wd": ((k, n // 2), u8),
        "wsd": ((k, n // 32), u8),
        "wq2": ((n, k // 2), u8),
        "ws2": ((n, k // 32), u8),
        "wq3": ((n, k // 2), u8),
        "ws3": ((n, k // 32), u8),
    }
    return [
        torch.empty(shapes[name][0], dtype=shapes[name][1], device=DEVICE)
        for name in names
    ]


@enable_fake_mode
def _trace(fn, k=TOY_K, n=TOY_N, dtype=torch.bfloat16, batch=(TOY_M,)):
    """The aten graph of a toy function with fake tensor metadata (what a pass sees)."""
    names = list(inspect.signature(fn).parameters)
    args = _toy_args(names, k, n, dtype, batch)
    return VllmFusionPatternMatcherPass._trace_fn(fn, args)


def _targets(graph, op):
    return [node for node in graph.nodes if node.target == op]


def _run_pass(norm_pass, gm, compile_range=RANGE):
    """Run the pass on one piece the way PostGradPassManager does; return the graph."""
    with pass_context(compile_range):
        norm_pass(gm.graph)
    return gm.graph


def _gemm_x_scales(gemm_node):
    """The x_scales argument of a gemm_with_dynamic_quant node (None when absent)."""
    if "x_scales" in gemm_node.kwargs:
        return gemm_node.kwargs["x_scales"]
    return gemm_node.args[5] if len(gemm_node.args) > 5 else None


@pytest.fixture
def make_norm_pass(rocm_aiter_fusion):
    """Builds the norm pass under a serving-like config.

    make_norm_pass(prefix, tp_size=1, graph_partition=False, splitting_ops=None) is the
    pass of the given compile prefix. The config stays current for the test body (the
    pass traces its patterns under it).
    """

    def config_with(tp_size, graph_partition=False, splitting_ops=None):
        config = VllmConfig(
            compilation_config=CompilationConfig(
                mode=CompilationMode.VLLM_COMPILE,
                custom_ops=["none"],
                backend="eager",
                splitting_ops=["vllm::unified_attention"],
                pass_config=PassConfig(fuse_norm_quant=True, eliminate_noops=True),
            )
        )
        config.parallel_config.tensor_parallel_size = tp_size
        # set after construction, so the config's own validation does not interfere
        config.compilation_config.use_inductor_graph_partition = graph_partition
        if splitting_ops is not None:
            config.compilation_config.splitting_ops = splitting_ops
        return config

    def build(prefix, tp_size=1, **compilation):
        config = config_with(tp_size, **compilation)
        with set_current_vllm_config(config):
            return rocm_aiter_fusion.RocmAiterRMSNormMxfp4GemmFusionPass(
                config, compile_prefix=prefix
            )

    with set_current_vllm_config(config_with(1)):
        yield build


def test_toy_site_is_what_gemma_rms_norm_traces_to(make_norm_pass):
    # Input: the toy post-attention site (norm -> gate_up GEMM), and the same stretch
    # traced from the model's own GemmaRMSNorm.forward_native followed by the GEMM op.
    # Expected: the same sequence of call targets. Why: the pattern is written against
    # the graph the model produces, so a toy that drifts from it would pass tests for a
    # graph the model never makes.
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    def targets(graph):
        return [n.target for n in graph.nodes if n.op == "call_function"]

    @enable_fake_mode
    def traced_from_the_model():
        norm = GemmaRMSNorm(TOY_K)
        norm.weight = torch.nn.Parameter(
            torch.empty(TOY_K, dtype=torch.bfloat16, device=DEVICE),
            requires_grad=False,
        )
        x, residual, wq, ws = _toy_args(
            ["x", "residual", "wq", "ws"], TOY_K, TOY_N, torch.bfloat16, (TOY_M,)
        )

        def piece(x, residual):
            y, residual_out = norm.forward_native(x, residual)
            return _gemm(y, wq, ws), residual_out

        return VllmFusionPatternMatcherPass._trace_fn(piece, [x, residual])

    assert targets(_trace(_post_site()).graph) == targets(traced_from_the_model().graph)


# --- the rewrite ------------------------------------------------------------------


@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_pass_rewrites_the_post_attention_site(make_norm_pass, eps):
    # Input: norm(x, residual, w) -> gemm_with_dynamic_quant on the asm path, with the
    # two epsilons models use. Expected: one site rewritten; the fused op takes the
    # RAW bf16 weight placeholder (not the fp32 w + 1.0 node: the kernel adds the 1)
    # and the epsilon; the GEMM takes the op's codes and gets the op's scales as
    # x_scales; the residual output is the op's third result; neither the IR norm nor
    # the weight convert and add nodes remain.
    norm_pass = make_norm_pass("backbone")
    graph = _run_pass(norm_pass, _trace(_post_site(eps=eps)))
    assert norm_pass.matched_count == 1
    (fused,) = _targets(graph, _fused_op())
    x, residual, weight, fused_eps = fused.args
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    assert (x, residual, weight) == tuple(placeholders[:3])  # x, residual, raw weight
    assert fused_eps == eps
    (gemm_node,) = _targets(graph, _gemm_op())
    assert gemm_node.args[0].args == (fused, 0)  # getitem(fused, 0): the codes
    assert _gemm_x_scales(gemm_node).args == (fused, 1)  # the scales
    (output,) = [n for n in graph.nodes if n.op == "output"]
    assert output.args[0][1].args == (fused, 2)  # the residual output
    assert not _targets(graph, _norm_op())
    assert not _targets(graph, torch.ops.aten.add.Tensor)
    assert not _targets(graph, torch.ops.prims.convert_element_type.default)


def test_pass_leaves_a_gemm_off_the_asm_path_alone(make_norm_pass):
    # Input: the site with use_asm_gemm False in the GEMM call (the Triton GEMM, whose
    # scale layout differs). Expected: no site, norm and GEMM untouched. Why: the fused
    # op's scales are in the asm layout.
    norm_pass = make_norm_pass("backbone")
    graph = _run_pass(norm_pass, _trace(_post_site(asm=False)))
    assert norm_pass.matched_count == 0
    assert _targets(graph, _norm_op())
    assert not _targets(graph, _fused_op())


# --- which norms count as post-attention sites --------------------------------------


@pytest.mark.parametrize("second_consumer", [False, True], ids=["qkv", "gdn"])
def test_only_the_post_attention_norm_is_rewritten(make_norm_pass, second_consumer):
    # Input: a piece as the model splits it: the post-attention norm and its gate_up
    # GEMM, then the next layer's input norm (its residual is the first norm's residual
    # output) feeding one GEMM (a full-attention qkv_proj), or two (a gated delta net).
    # Expected: only the post-attention site is rewritten; the input norm keeps its IR
    # norm and its GEMM(s) keep four arguments. Why: by shape alone the sites look the
    # same, and the input norms are not sites of this pass.
    norm_pass = make_norm_pass("backbone")
    gm = _trace(_post_then_input_site(second_consumer))
    graph = _run_pass(norm_pass, gm)
    assert norm_pass.matched_count == 1
    (remaining_norm,) = _targets(graph, _norm_op())
    assert remaining_norm.args[0].target == _gemm_op()  # the down GEMM's output
    untouched = [
        g for g in _targets(graph, _gemm_op()) if g.args[0].args[0].target == _norm_op()
    ]
    assert len(untouched) == (2 if second_consumer else 1)
    assert all(len(g.args) == 4 for g in untouched)

    # A second application matches nothing more: the input norm now takes its residual
    # from the fused op, and must still be recognised as an input norm.
    _run_pass(norm_pass, gm)
    assert norm_pass.matched_count == 0
    assert len(_targets(gm.graph, _fused_op())) == 1


def test_a_norm_output_with_two_consumers_is_left_alone(make_norm_pass):
    # Input: a norm taking a piece input as its residual whose output feeds two GEMMs.
    # Expected: no rewrite. Why: the pattern covers one consumer (the pattern matcher
    # refuses a match that leaves another user of the normed tensor behind).
    norm_pass = make_norm_pass("backbone")
    graph = _run_pass(norm_pass, _trace(_post_with_two_consumers))
    assert norm_pass.matched_count == 0
    assert _targets(graph, _norm_op())


def _norm_residual(kind, g, a, b):
    """The residual operand of a norm, built in graph g from the inputs a and b."""
    item = {"ir_norm": 1, "fused_op": 2, "norm_output": 0}
    if kind == "graph_input":
        return a
    if kind == "plain_op":
        return g.call_function(torch.ops.aten.add.Tensor, (a, b))
    if kind == "split_item":
        return g.call_function(
            operator.getitem, (g.call_function(torch.ops.aten.split.Tensor, (a, 2)), 0)
        )
    norm = _fused_op() if kind == "fused_op" else _norm_op()
    return g.call_function(
        operator.getitem, (g.call_function(norm, (a, b, b, EPS)), item[kind])
    )


@pytest.mark.parametrize(
    ("kind", "site_kind"),
    [
        ("graph_input", "post"),
        ("ir_norm", "input"),
        ("fused_op", "input"),
        ("norm_output", "input"),  # the item index is deliberately not checked
        ("split_item", "post"),
        ("plain_op", "post"),
    ],
)
def test_norm_site_kind_follows_where_the_residual_comes_from(
    rocm_aiter_fusion, kind, site_kind
):
    # Input: the residual operand of a norm that is a graph input, an item of the IR
    # norm or of the fused op, or something else. Expected: "input" exactly when it is
    # an item of one of the two norm ops, "post" otherwise. Why: this rule tells a
    # post-attention norm (the residual comes in with the piece) from the next layer's
    # input norm (the residual is the previous norm's output). It is purely syntactic.
    graph = fx.Graph()
    a, b = graph.placeholder("a"), graph.placeholder("b")
    residual = _norm_residual(kind, graph, a, b)
    pattern_cls = rocm_aiter_fusion.AiterAddRMSNormMxfp4GemmPattern
    assert pattern_cls._norm_site_kind(residual, (_norm_op(), _fused_op())) == site_kind


# --- the eligibility guard ----------------------------------------------------


@pytest.mark.parametrize(
    "trace_kwargs",
    [
        {"k": 48},
        {"k": ROCM_AITER_MXFP4_NORM_QUANT_MAX_K + 32},
        {"dtype": torch.float16},
        {"batch": (2, 64)},
    ],
    ids=["k-not-a-multiple-of-32", "k-above-the-max", "fp16", "3d-activations"],
)
def test_ineligible_site_is_left_alone(make_norm_pass, trace_kwargs):
    # Input: a site the op cannot take. Expected: no rewrite and no exception. Why: the
    # op's fake raises for these, so a match would crash the compile; the guard filters
    # them at match time.
    norm_pass = make_norm_pass("backbone")
    graph = _run_pass(norm_pass, _trace(_post_site(), **trace_kwargs))
    assert norm_pass.matched_count == 0
    assert _targets(graph, _norm_op())
    assert not _targets(graph, _fused_op())


def test_gemm_with_a_tuned_preshuffle_config_is_left_alone(make_norm_pass, monkeypatch):
    # Input: the site, with the Triton preshuffle lookup answering True. Expected: no
    # rewrite, and the lookup was asked with the GEMM weight's (N, K bytes). Why: such
    # a GEMM may take its small-M branch, which wants UNshuffled scales for M < 32,
    # while the fused op writes the asm (shuffled) layout.
    asked = []

    def tuned(n, k_bytes):
        asked.append((n, k_bytes))
        return True

    monkeypatch.setattr(rocm_aiter_ops, "is_triton_gemm_afp4wfp4_presh_ws_tuned", tuned)
    norm_pass = make_norm_pass("backbone")
    _run_pass(norm_pass, _trace(_post_site()))
    assert norm_pass.matched_count == 0
    assert asked == [(TOY_N, TOY_K // 2)]


def test_tensor_parallel_run_is_left_alone(make_norm_pass):
    # Input: the pass built for tensor parallel size 2. Expected: no site rewritten.
    # Why: with TP the norm input comes out of an all-reduce, whose fusion pass owns
    # the norm; TP=1 only is supported.
    norm_pass = make_norm_pass("backbone", tp_size=2)
    _run_pass(norm_pass, _trace(_post_site()))
    assert norm_pass.matched_count == 0


@pytest.mark.parametrize(
    ("compilation", "setting"),
    [
        ({"graph_partition": True}, "use_inductor_graph_partition"),
        ({"splitting_ops": []}, "splitting_ops"),
    ],
)
def test_pass_refuses_a_graph_that_is_not_split_at_the_attention_ops(
    make_norm_pass, compilation, setting
):
    # Input: the pass built with inductor graph partition on, or with no splitting ops.
    # Expected: a RuntimeError naming the flag and the setting. Why: the graph is then
    # not split at the attention ops, the residual chain of every layer runs through
    # one graph, and the site classifier would call every norm but the first an input
    # norm: one site, the start-up check satisfied, the rest silently unfused.
    with pytest.raises(RuntimeError) as raised:
        make_norm_pass("backbone", **compilation)
    assert POSTNORM_FLAG in str(raised.value)
    assert setting in str(raised.value)


# --- extra_check on hand-made matches: what traced graphs cannot show ---------------


def _stub_match(**overrides):
    """A match holding the four pattern arguments the check reads, as nodes.

    The tensors are meta tensors (shape, dtype and strides only). An argument may be
    replaced by another value, or by None for a node without metadata.
    """

    def node(value):
        meta = {} if value is None else {"val": value}
        return types.SimpleNamespace(meta=meta, target="placeholder", args=())

    def meta_tensor(*shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")

    values = {
        "x": meta_tensor(TOY_M, TOY_K, dtype=torch.bfloat16),
        "residual": meta_tensor(TOY_M, TOY_K, dtype=torch.bfloat16),
        "weight": meta_tensor(TOY_K, dtype=torch.bfloat16),
        "gemm_weight": meta_tensor(TOY_N, TOY_K // 2, dtype=torch.uint8),
    }
    values.update(overrides)
    return types.SimpleNamespace(kwargs={k: node(v) for k, v in values.items()})


class _Symbolic:
    """A tensor-like object whose sizes are not plain ints (a symbolic shape)."""

    dtype = torch.bfloat16

    def __init__(self, shape):
        self.shape = shape

    def dim(self):
        return len(self.shape)

    def stride(self, dim):
        return 1


@pytest.fixture
def pattern(rocm_aiter_fusion, monkeypatch):
    # no tuned preshuffle config for any shape, so only the guard under test decides
    monkeypatch.setattr(
        rocm_aiter_ops, "is_triton_gemm_afp4wfp4_presh_ws_tuned", lambda n, k: False
    )
    return rocm_aiter_fusion.AiterAddRMSNormMxfp4GemmPattern(
        EPS, torch.bfloat16, tp_size=1
    )


def test_hand_made_match_of_a_good_site_is_accepted(pattern):
    # The control for the refusals below, which each change one thing.
    assert pattern.extra_check(_stub_match()) is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"weight": None},  # no fake tensor value: shapes and dtypes are unknown
        {"x": _Symbolic((TOY_M, object()))},  # a symbolic hidden size
        {"gemm_weight": _Symbolic((object(), object()))},  # a symbolic GEMM shape
    ],
    ids=["no-metadata", "symbolic-hidden-size", "symbolic-gemm-shape"],
)
def test_site_whose_sizes_are_unknown_is_left_alone(pattern, overrides):
    # Input: a match whose weight has no tensor value, whose x has a symbolic hidden
    # size, or whose GEMM weight has symbolic sizes. Expected: refused, no exception.
    # Why: the op's limits (K a multiple of 32 up to 8192) and the tuned-config lookup
    # need plain ints, and testing a symbol would add a shape guard.
    assert pattern.extra_check(_stub_match(**overrides)) is False


# --- the pass and the op agree on what the op takes ---------------------------------

BF16, FP16 = torch.bfloat16, torch.float16


def _sized(rows, k, dtype=BF16, row_stride=None, inner_stride=1):
    """(size, stride, dtype) of a 2D tensor, for _op_and_pass_answers."""
    return ((rows, k), (row_stride or k, inner_stride), dtype)


# (case, x, residual, weight), each tensor as (size, stride, dtype)
OP_ARGUMENT_CASES = [
    ("ok", _sized(4, 64), _sized(4, 64), ((64,), (1,), BF16)),
    ("row-strided", _sized(4, 64, row_stride=128), _sized(4, 64), ((64,), (1,), BF16)),
    ("inner-stride", _sized(4, 64, inner_stride=2), _sized(4, 64), ((64,), (1,), BF16)),
    ("residual-width", _sized(4, 64), _sized(4, 96), ((64,), (1,), BF16)),
    ("weight-stride", _sized(4, 64), _sized(4, 64), ((64,), (2,), BF16)),
    ("fp16", _sized(4, 64, FP16), _sized(4, 64, FP16), ((64,), (1,), FP16)),
    ("k-8192", _sized(4, 8192), _sized(4, 8192), ((8192,), (1,), BF16)),
    ("k-8224", _sized(4, 8224), _sized(4, 8224), ((8224,), (1,), BF16)),
    ("k-0", _sized(4, 0), _sized(4, 0), ((0,), (1,), BF16)),
]


def _op_and_pass_answers(rocm_aiter_fusion, x, residual, weight):
    """(whether the op's fake takes the arguments, whether the pass's check does)."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        args = [
            torch.empty_strided(size, stride, dtype=dtype, device=DEVICE)
            for size, stride, dtype in (x, residual, weight)
        ]
        try:
            torch.ops.vllm.rocm_aiter_rmsnorm_mxfp4_quant(*args, EPS)
            op_takes_it = True
        except ValueError:
            op_takes_it = False
        pass_cls = rocm_aiter_fusion.AiterAddRMSNormMxfp4GemmPattern
        return op_takes_it, pass_cls.norm_args_supported(*args)


@pytest.mark.parametrize(
    ("case", "x", "residual", "weight"),
    OP_ARGUMENT_CASES,
    ids=[case[0] for case in OP_ARGUMENT_CASES],
)
def test_pass_and_op_agree_on_what_the_op_takes(
    rocm_aiter_fusion, case, x, residual, weight
):
    # Input: argument triples that are fine, row-strided, or break one rule of the op.
    # Expected: the pass's check says "supported" exactly when the op's own fake
    # accepts them. Why: a pass that accepted what the op refuses would crash the
    # compile, and one that refused what the op takes would leave sites unfused.
    op_takes_it, pass_says = _op_and_pass_answers(
        rocm_aiter_fusion, x, residual, weight
    )
    assert pass_says is op_takes_it, case


def test_the_op_argument_cases_include_accepted_and_refused_ones(rocm_aiter_fusion):
    # If every case got the same answer, agreeing with the op would prove nothing.
    answers = {
        _op_and_pass_answers(rocm_aiter_fusion, *case[1:])[0]
        for case in OP_ARGUMENT_CASES
    }
    assert answers == {True, False}


# --- the start-up check, with the real pass -------------------------------------------


class _LogRecorder:
    """Replaces the fusion module's logger and keeps the "Fused ..." info lines."""

    def __init__(self) -> None:
        self.fused: list[str] = []

    def info(self, msg, *args) -> None:
        if msg.startswith("Fused "):
            self.fused.append(msg % args)

    def warning(self, msg, *args) -> None:
        pass


@pytest.fixture
def log(rocm_aiter_fusion, monkeypatch):
    recorder = _LogRecorder()
    monkeypatch.setattr(rocm_aiter_fusion, "logger", recorder)
    return recorder


def test_target_graph_with_a_site_logs_it_and_a_piece_without_one_is_fine(
    make_norm_pass, log
):
    # Input: the pieces of a backbone graph in order: a GEMM without a norm (the first
    # piece of a model), then one with the site. Expected: the first piece raises
    # nothing, finish_compile logs the site with its label, graph and compile range.
    norm_pass = make_norm_pass("backbone")
    _run_pass(norm_pass, _trace(_gemm_only))
    _run_pass(norm_pass, _trace(_post_site()))
    norm_pass.finish_compile()
    (line,) = log.fused
    assert line == (
        "Fused post-attention add+rmsnorm+mxfp4 quant at 1 sites (graph=backbone) "
        f"for compile range {RANGE}"
    )


def test_target_graph_without_a_site_fails_naming_the_flag(make_norm_pass, log):
    # Input: a backbone graph on which the pass ran and found no site. Expected:
    # finish_compile raises and names the flag. Why: the flag is on, so a start that
    # leaves the model unfused must not pass silently.
    norm_pass = make_norm_pass("backbone")
    _run_pass(norm_pass, _trace(_gemm_only))
    with pytest.raises(RuntimeError, match=f"{POSTNORM_FLAG} is on but 0 sites"):
        norm_pass.finish_compile()


# --- the rewritten graph computes the same thing (gfx950 GPU) ---------------------


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


@pytest.mark.skipif(
    not (_on_gfx950() and torch.cuda.is_available()), reason="needs a gfx950 GPU"
)
@pytest.mark.parametrize("num_tokens", [1, 64])
@pytest.mark.parametrize("n", [256, 2048])
def test_rewritten_graph_is_as_accurate_as_the_unrewritten_one(
    make_norm_pass, num_tokens, n
):
    # Input: the post-attention site at hidden size 1024 traced once, run as traced
    # (the IR norm in front of the asm MXFP4 GEMM) and after the pass (the fused op
    # feeding the GEMM through x_scales), on the same activations, norm weight and
    # MXFP4 weight. Expected: the rewritten graph's error against the fp32 reference
    # (fp32 norm output times the unquantized weight) is within 5 percent of the
    # unrewritten graph's, and its residual output equals the sum the unrewritten
    # graph returns. Why: the kernel tests pin the op; this pins the wiring of the
    # rewrite. A wrong output (codes and scales swapped, the weight's +1 applied
    # twice, a stale residual) shows here as an error of order 1.
    import aiter
    from aiter.ops.shuffle import shuffle_weight

    k = 1024
    torch.manual_seed(0)
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device=DEVICE)
    residual = torch.randn(num_tokens, k, dtype=torch.bfloat16, device=DEVICE)
    norm_weight = torch.randn(k, dtype=torch.bfloat16, device=DEVICE) * 0.1
    w = torch.randn(n, k, dtype=torch.bfloat16, device=DEVICE) * 0.02
    w_q, w_s = aiter.per_1x32_f4_quant_hip(w, shuffle=False)
    w_q, w_s = w_q.view(torch.uint8), w_s.view(torch.uint8)
    # the layouts AiterMxfp4LinearKernel.process_weights_after_loading produces for the
    # asm GEMM
    sm, sn = w_s.shape
    w_s = (
        w_s.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
        .permute(0, 3, 5, 2, 4, 1, 6)
        .contiguous()
        .view(sm, sn)
    )
    w_q = shuffle_weight(w_q, layout=(16, 16))

    summed = x.float() + residual.float()
    rms = (summed.pow(2).mean(-1, keepdim=True) + EPS).rsqrt()
    reference = (summed * rms * (norm_weight.float() + 1.0)) @ w.float().T

    norm_pass = make_norm_pass("backbone")
    unrewritten = _trace(_post_site(), k=k, n=n)
    rewritten = _trace(_post_site(), k=k, n=n)
    _run_pass(norm_pass, rewritten)
    assert norm_pass.matched_count == 1
    rewritten.recompile()

    out_before, residual_before = unrewritten(x, residual, norm_weight, w_q, w_s)
    out_after, residual_after = rewritten(x, residual, norm_weight, w_q, w_s)

    def rel_err(y):
        return ((y.float() - reference).norm() / reference.norm()).item()

    assert rel_err(out_after) <= rel_err(out_before) * 1.05 + 1e-3
    assert torch.equal(residual_after, residual_before)
