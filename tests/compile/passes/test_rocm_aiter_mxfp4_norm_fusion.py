# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests of RocmAiterRMSNormMxfp4GemmFusionPass, the Gemma norm + MXFP4 quant fusion.

The pass rewrites ``fused_add_rms_norm -> gemm_with_dynamic_quant`` into the custom op
``rocm_aiter_rmsnorm_mxfp4_quant``, whose codes and scales feed the GEMM through its
``x_scales`` argument. It comes in two site kinds, one flag each: the post-attention
norm feeding the MLP's gate_up projection ("post"), and the input norm that starts a
decoder layer ("input"), whose output feeds one GEMM (qkv_proj) or two (a gated delta
net's in_proj_qkvz and in_proj_ba) that then share one quantization.

The toy graphs are traced the way the pass sees a compiled piece (an aten graph with
fake tensor metadata). They need the AITER custom ops, so the file needs ROCm with
AITER; only the last tests, which run the rewritten graph, need a gfx950 GPU.
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
INPUTNORM_FLAG = "VLLM_ROCM_AITER_FP4_INPUTNORM_QUANT_FUSION"
POST_LABEL = "post-attention add+rmsnorm+mxfp4 quant"
INPUT_LABEL = "input add+rmsnorm+mxfp4 quant"
EPS = 1e-6
RANGE = Range(1, 8)
DEVICE = current_platform.device_type

# Toy sizes: K is a multiple of 32, which the op's fake requires.
TOY_M, TOY_K, TOY_N = 4, 64, 32

# GEMM widths behind the input norms of a hybrid model: a gated delta net layer's
# in_proj_qkvz and in_proj_ba, and a full-attention layer's qkv_proj.
GDN_N_QKVZ, GDN_N_BA, ATTENTION_N_QKV = 16384, 96, 14336


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
        # the narrow second consumer of an input norm (in_proj_ba of a gated delta net)
        "wq_ba": ((GDN_N_BA, k // 2), u8),
        "ws_ba": ((GDN_N_BA, k // 32), u8),
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

    make_norm_pass(prefix, tp_size=1, sites="post", graph_partition=False,
    splitting_ops=None) is the pass of the given compile prefix and site kind ("post" or
    "input"). The config stays current for the test body (the pass traces its patterns
    under it).
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

    def build(prefix, tp_size=1, sites="post", **compilation):
        config = config_with(tp_size, **compilation)
        with set_current_vllm_config(config):
            return rocm_aiter_fusion.RocmAiterRMSNormMxfp4GemmFusionPass(
                config, compile_prefix=prefix, sites=sites
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


# --- the input site kind: the norm that starts a decoder layer ---------------------
#
# An input norm takes the residual the previous layer's post-attention norm returned
# (see _norm_site_kind). A full-attention layer has ONE GEMM on its output (qkv_proj),
# a gated delta net layer TWO (in_proj_qkvz and in_proj_ba), and the input pass
# quantizes once and gives the codes and scales to every GEMM. Layer 0's norm is a
# plain rms_norm with no residual and is never a site.


def _input_site(x, residual, w, w2, wq, ws):
    """The next layer's input norm and ONE GEMM (a full-attention layer's qkv_proj).

    A norm with no GEMM after it stands in for the post-attention norm: the input norm
    takes its residual output, which is what makes the second norm an input norm.
    """
    y, residual_out = _norm(x, residual, w)
    y2, residual_out2 = _norm(y, residual_out, w2)
    return _gemm(y2, wq, ws), residual_out2


def _input_pair_site(x, residual, w, w2, wq, ws, wq_ba, ws_ba):
    """An input norm and TWO GEMMs: in_proj_qkvz (the n of _trace), in_proj_ba (96)."""
    y, residual_out = _norm(x, residual, w)
    y2, residual_out2 = _norm(y, residual_out, w2)
    return _gemm(y2, wq, ws), _gemm(y2, wq_ba, ws_ba), residual_out2


def _input_site_also_read(x, residual, w, w2, wq, ws):
    """An input norm whose output feeds a GEMM and something that is not a GEMM."""
    y, residual_out = _norm(x, residual, w)
    y2, residual_out2 = _norm(y, residual_out, w2)
    return _gemm(y2, wq, ws), y2 * 2.0, residual_out2


def _input_pair_site_also_read(x, residual, w, w2, wq, ws, wq_ba, ws_ba):
    """The pair of GEMMs of _input_pair_site and a third reader that is not a GEMM."""
    y, residual_out = _norm(x, residual, w)
    y2, residual_out2 = _norm(y, residual_out, w2)
    return _gemm(y2, wq, ws), _gemm(y2, wq_ba, ws_ba), y2 * 2.0, residual_out2


def _input_pair_site_residual_reused(x, residual, w, w2, wq, ws, wq_ba, ws_ba):
    """The pair site whose residual output has two users: an add and an output."""
    y, residual_out = _norm(x, residual, w)
    y2, residual_out2 = _norm(y, residual_out, w2)
    shifted = residual_out2 + 1.0
    return _gemm(y2, wq, ws), _gemm(y2, wq_ba, ws_ba), shifted, residual_out2


def _input_triple_site(x, residual, w, w2, wq, ws, wq_ba, ws_ba, wq3, ws3):
    """An input norm with THREE GEMMs: more than the pass has a pattern for."""
    y, residual_out = _norm(x, residual, w)
    y2, residual_out2 = _norm(y, residual_out, w2)
    return (
        _gemm(y2, wq, ws),
        _gemm(y2, wq_ba, ws_ba),
        _gemm(y2, wq3, ws3),
        residual_out2,
    )


def _plain_norm_site(x, w, wq, ws, wq_ba, ws_ba):
    """Layer 0: a norm with no residual (the plain rms_norm op) and two GEMMs."""
    y = torch.ops.vllm_ir.rms_norm(x, w.float() + 1.0, EPS)
    return _gemm(y, wq, ws), _gemm(y, wq_ba, ws_ba)


def _gemm_weight_widths(graph):
    """The N of every GEMM's weight, ascending."""
    return sorted(g.args[1].meta["val"].shape[0] for g in _targets(graph, _gemm_op()))


def _output_gemm_widths(graph, count=2):
    """The N of the weight of each of the first `count` outputs of a piece, in order.

    The outputs must be GEMM nodes. This is what pins which GEMM's result goes to which
    output of the piece: a rewrite that returned the right values in the wrong slots
    would leave the set of GEMMs in the graph unchanged.
    """
    (output,) = [n for n in graph.nodes if n.op == "output"]
    assert all(n.target == _gemm_op() for n in output.args[0][:count])
    return [n.args[1].meta["val"].shape[0] for n in output.args[0][:count]]


def _reverse_users(node):
    """Reverse the order in which the graph lists the users of a node.

    The pattern matcher tries the users of the normed tensor in this order, so a graph
    whose users are listed the other way round (an earlier pass can do that) is a
    different input for the pattern of two GEMMs.
    """
    node.users = dict(reversed(list(node.users.items())))


def test_input_norm_with_two_gemms_becomes_one_fused_op(make_norm_pass):
    # Input: a gated-delta-net input norm whose output feeds in_proj_qkvz (N=16384) and
    # in_proj_ba (N=96), given to the input pass. Expected: one site; ONE fused op whose
    # codes (item 0) and scales (item 1) feed BOTH GEMMs (each keeps its own weight and
    # weight scale and still takes the asm path), the residual output is the op's third
    # result, the input norm is gone (the stand-in norm before it stays), and the
    # piece's first two outputs are still the GEMMs of N=16384 and N=96 in that order.
    # Why: this is the point of the pair pattern, one quantization instead of two, and
    # the two results must not trade places on the way.
    norm_pass = make_norm_pass("backbone", sites="input")
    graph = _run_pass(norm_pass, _trace(_input_pair_site, n=GDN_N_QKVZ))
    assert norm_pass.matched_count == 1
    (fused,) = _targets(graph, _fused_op())
    gemms = _targets(graph, _gemm_op())
    assert len(gemms) == 2
    for gemm_node in gemms:
        assert gemm_node.args[0].args == (fused, 0)  # the shared codes
        assert _gemm_x_scales(gemm_node).args == (fused, 1)  # the shared scales
        assert gemm_node.args[3] is True
        weight, weight_scale = (gemm_node.args[i].meta["val"] for i in (1, 2))
        assert weight.shape[0] == weight_scale.shape[0]  # its own weight pair
    assert _gemm_weight_widths(graph) == [GDN_N_BA, GDN_N_QKVZ]
    # each result is still the output it was before: in_proj_qkvz first, then
    # in_proj_ba, the order the toy piece returns them in
    assert _output_gemm_widths(graph) == [GDN_N_QKVZ, GDN_N_BA]
    (output,) = [n for n in graph.nodes if n.op == "output"]
    assert output.args[0][-1].args == (fused, 2)  # the residual output
    (stand_in,) = _targets(graph, _norm_op())
    assert stand_in.args[0].op == "placeholder"  # the stand-in is the only norm left


@pytest.mark.parametrize("reverse_users", [False, True], ids=["listed", "reversed"])
def test_pair_is_found_whichever_way_the_norm_outputs_users_are_listed(
    make_norm_pass, reverse_users
):
    # Input: the same input norm with two GEMMs, with the users of its output listed in
    # graph order and in the opposite order. Expected: rewritten either way, both GEMMs
    # take the fused op's codes and scales, each GEMM once, results in their places.
    # Why: the pattern has two GEMM slots; with the users listed the other way round the
    # matcher first binds both slots to the SAME GEMM. That match must be refused (see
    # the extra_check tests below) and the other anchor must still produce the real
    # pair, or the site is silently lost.
    norm_pass = make_norm_pass("backbone", sites="input")
    gm = _trace(_input_pair_site, n=GDN_N_QKVZ)
    first_gemm = _targets(gm.graph, _gemm_op())[0]
    if reverse_users:
        _reverse_users(first_gemm.args[0])
    graph = _run_pass(norm_pass, gm)
    assert norm_pass.matched_count == 1
    (fused,) = _targets(graph, _fused_op())
    gemms = _targets(graph, _gemm_op())
    assert [g.args[0].args for g in gemms] == [(fused, 0)] * 2
    assert [_gemm_x_scales(g).args for g in gemms] == [(fused, 1)] * 2
    assert _gemm_weight_widths(graph) == [GDN_N_BA, GDN_N_QKVZ]  # each GEMM once
    assert _output_gemm_widths(graph) == [GDN_N_QKVZ, GDN_N_BA]  # results in place


def test_every_user_of_the_residual_output_follows_the_fused_op(make_norm_pass):
    # Input: the pair site whose residual output is used by an add and is also a piece
    # output. Expected: both users now read the fused op's third result, and the graph
    # stays valid (every node defined before its users). Why: the old norm's residual
    # output disappears with the rewrite, so a user left on it would read a deleted
    # node or a stale tensor.
    norm_pass = make_norm_pass("backbone", sites="input")
    graph = _run_pass(norm_pass, _trace(_input_pair_site_residual_reused, n=GDN_N_QKVZ))
    assert norm_pass.matched_count == 1
    (fused,) = _targets(graph, _fused_op())
    # the adds of the graph: the stand-in norm's weight + 1 and the user of the residual
    (shift,) = [
        n
        for n in _targets(graph, torch.ops.aten.add.Tensor)
        if n.args[0].target is operator.getitem
    ]
    assert shift.args[0].args == (fused, 2)
    (output,) = [n for n in graph.nodes if n.op == "output"]
    assert output.args[0][-1].args == (fused, 2)
    graph.lint()


def test_input_norm_with_one_gemm_is_rewritten(make_norm_pass):
    # Input: a full-attention layer's input norm with its single qkv_proj GEMM
    # (N=14336). Expected: one site, one fused op feeding the GEMM, the residual output
    # wired to the op's third result, the input norm gone. Why: the single pattern
    # serves the input kind too.
    norm_pass = make_norm_pass("backbone", sites="input")
    graph = _run_pass(norm_pass, _trace(_input_site, n=ATTENTION_N_QKV))
    assert norm_pass.matched_count == 1
    (fused,) = _targets(graph, _fused_op())
    (gemm_node,) = _targets(graph, _gemm_op())
    assert gemm_node.args[0].args == (fused, 0)
    assert _gemm_x_scales(gemm_node).args == (fused, 1)
    (output,) = [n for n in graph.nodes if n.op == "output"]
    assert output.args[0][-1].args == (fused, 2)
    assert len(_targets(graph, _norm_op())) == 1  # the stand-in only


@pytest.mark.parametrize(
    "tuned_n", [GDN_N_QKVZ, GDN_N_BA], ids=["qkvz-tuned", "ba-tuned"]
)
def test_input_norm_with_one_ineligible_gemm_is_left_whole(
    make_norm_pass, monkeypatch, tuned_n
):
    # Input: the pair site with the Triton preshuffle lookup answering True for ONE of
    # the two GEMMs (N=96, then N=16384). Expected: no rewrite at all: both IR norms
    # stay, no fused op, both GEMMs keep their four arguments. Why: a site is fused for
    # all of its consumers or for none, and the GEMM that may take its small-M Triton
    # branch cannot take the op's asm scales.
    monkeypatch.setattr(
        rocm_aiter_ops,
        "is_triton_gemm_afp4wfp4_presh_ws_tuned",
        lambda n, k_bytes: n == tuned_n,
    )
    norm_pass = make_norm_pass("backbone", sites="input")
    graph = _run_pass(norm_pass, _trace(_input_pair_site, n=GDN_N_QKVZ))
    assert norm_pass.matched_count == 0
    assert len(_targets(graph, _norm_op())) == 2
    assert not _targets(graph, _fused_op())
    assert [len(g.args) for g in _targets(graph, _gemm_op())] == [4, 4]


@pytest.mark.parametrize(
    "piece",
    [_input_site_also_read, _input_pair_site_also_read, _input_triple_site],
    ids=["gemm-and-a-reader", "two-gemms-and-a-reader", "three-gemms"],
)
def test_input_norm_with_a_consumer_the_pass_cannot_take_is_left_whole(
    make_norm_pass, piece
):
    # Input: an input norm whose output also feeds something that is not a GEMM (next
    # to one GEMM, next to two), and one with three GEMMs. Expected: no rewrite, both
    # norms stay, every GEMM keeps four arguments. Why: the fused op replaces the
    # normed tensor, so a reader that still needs the bf16 tensor would be left without
    # it, and a norm with more GEMMs than the pass has a pattern for must not be fused
    # for some of them only.
    norm_pass = make_norm_pass("backbone", sites="input")
    graph = _run_pass(norm_pass, _trace(piece, n=ATTENTION_N_QKV))
    assert norm_pass.matched_count == 0
    assert len(_targets(graph, _norm_op())) == 2
    assert not _targets(graph, _fused_op())
    assert {len(g.args) for g in _targets(graph, _gemm_op())} == {4}


@pytest.mark.parametrize("sites", ["post", "input"])
def test_plain_norm_without_a_residual_is_never_a_site(make_norm_pass, sites):
    # Input: layer 0's norm, the plain rms_norm op with no residual, feeding two GEMMs,
    # given to the pass of either site kind. Expected: no rewrite. Why: Inductor fuses
    # that norm with the embedding lookup; the op has no residual to add, and the fused
    # op needs one.
    norm_pass = make_norm_pass("backbone", sites=sites)
    graph = _run_pass(norm_pass, _trace(_plain_norm_site, n=GDN_N_QKVZ))
    assert norm_pass.matched_count == 0
    assert _targets(graph, torch.ops.vllm_ir.rms_norm.default)
    assert not _targets(graph, _fused_op())


@pytest.mark.parametrize("n", [GDN_N_QKVZ, ATTENTION_N_QKV])
def test_input_pass_refuses_what_the_op_cannot_take(make_norm_pass, n):
    # Input: input sites at a K the op cannot take (not a multiple of 32). Expected: no
    # rewrite and no exception, for the single GEMM and for the pair. Why: the op's fake
    # raises for it, so the guard must filter the site at match time.
    norm_pass = make_norm_pass("backbone", sites="input")
    piece = _input_pair_site if n == GDN_N_QKVZ else _input_site
    graph = _run_pass(norm_pass, _trace(piece, k=48, n=n))
    assert norm_pass.matched_count == 0
    assert len(_targets(graph, _norm_op())) == 2


def test_tensor_parallel_run_is_left_alone_by_the_input_pass(make_norm_pass):
    # Input: the input pass built for tensor parallel size 2. Expected: nothing is
    # rewritten. Why: same as the post pass, an all-reduce sits in front of the norm.
    norm_pass = make_norm_pass("backbone", tp_size=2, sites="input")
    _run_pass(norm_pass, _trace(_input_pair_site, n=GDN_N_QKVZ))
    assert norm_pass.matched_count == 0


# --- which pass takes which norm ----------------------------------------------------


@pytest.mark.parametrize("second_consumer", [False, True], ids=["attention", "gdn"])
def test_input_pass_leaves_the_post_attention_site_alone(
    make_norm_pass, second_consumer
):
    # Input: a piece as the model splits it (post-attention norm and gate_up, down GEMM,
    # the next layer's input norm and its GEMM or GEMMs) given to the input pass.
    # Expected: only the input norm is rewritten; the post-attention norm and its GEMM
    # keep their form. Why: each site kind belongs to its own flag; the input flag must
    # not also switch the post-attention fusion on.
    norm_pass = make_norm_pass("backbone", sites="input")
    graph = _run_pass(norm_pass, _trace(_post_then_input_site(second_consumer)))
    assert norm_pass.matched_count == 1
    (fused,) = _targets(graph, _fused_op())
    (remaining_norm,) = _targets(graph, _norm_op())
    assert remaining_norm.args[1].op == "placeholder"  # the post norm: piece residual
    gemms = _targets(graph, _gemm_op())
    fed_by_fused = [g for g in gemms if g.args[0].args[0] == fused]
    assert len(fed_by_fused) == (2 if second_consumer else 1)
    assert all(_gemm_x_scales(g) is not None for g in fed_by_fused)
    others = [g for g in gemms if g not in fed_by_fused]
    assert [len(g.args) for g in others] == [4, 4]  # gate_up and down, untouched


@pytest.mark.parametrize("order", [("post", "input"), ("input", "post")], ids="-".join)
@pytest.mark.parametrize("second_consumer", [False, True], ids=["attention", "gdn"])
def test_both_passes_rewrite_their_own_sites_in_either_order(
    make_norm_pass, order, second_consumer
):
    # Input: the same piece given to both passes, post then input and input then post.
    # Expected: each pass rewrites exactly its site (matched_count 1 each); two fused
    # ops and no IR norm remain; the gate_up GEMM and every input-norm GEMM take scales
    # and the down GEMM, which follows no norm, keeps four arguments. Why: with both
    # flags on the passes share a graph; neither may hide the other's site (the input
    # norm's residual comes from the post norm's output, a fused op once that was
    # rewritten) or rewrite it twice.
    gm = _trace(_post_then_input_site(second_consumer))
    passes = [make_norm_pass("backbone", sites=sites) for sites in order]
    for norm_pass in passes:
        _run_pass(norm_pass, gm)
    assert [norm_pass.matched_count for norm_pass in passes] == [1, 1]
    assert len(_targets(gm.graph, _fused_op())) == 2
    assert not _targets(gm.graph, _norm_op())
    gemms = _targets(gm.graph, _gemm_op())
    scaled = [g for g in gemms if _gemm_x_scales(g) is not None]
    assert len(scaled) == 1 + (2 if second_consumer else 1)
    assert len(gemms) == len(scaled) + 1


def test_input_pass_does_not_run_again_on_its_own_output(make_norm_pass):
    # Input: a piece rewritten by the input pass, given to it again. Expected: the
    # second run matches nothing and the graph keeps its one fused op.
    norm_pass = make_norm_pass("backbone", sites="input")
    gm = _trace(_input_pair_site, n=GDN_N_QKVZ)
    _run_pass(norm_pass, gm)
    assert norm_pass.matched_count == 1
    _run_pass(norm_pass, gm)
    assert norm_pass.matched_count == 0
    assert len(_targets(gm.graph, _fused_op())) == 1


# --- extra_check of the input kind and of the pair pattern --------------------------


def _input_norm_residual():
    """A residual operand that is an item of another norm's output: an input norm's."""
    graph = fx.Graph()
    operand = graph.placeholder("operand")
    norm = graph.call_function(_norm_op(), (operand, operand, operand, EPS))
    residual = graph.call_function(operator.getitem, (norm, 1))
    residual.meta["val"] = torch.empty(
        TOY_M, TOY_K, dtype=torch.bfloat16, device="meta"
    )
    return residual


def _with_second_gemm(match, outputs, second_weight_n=GDN_N_BA):
    """Makes a hand-made match one of the pair pattern: a second GEMM weight, and what
    match.output_nodes() answers (the two GEMM nodes and the residual)."""
    match.kwargs["gemm_weight_2"] = types.SimpleNamespace(
        meta={
            "val": torch.empty(
                second_weight_n, TOY_K // 2, dtype=torch.uint8, device="meta"
            )
        },
        target="placeholder",
        args=(),
    )
    match.output_nodes = lambda: outputs
    return match


def _input_stub_match():
    """_stub_match of an input norm: its residual comes from another norm's output."""
    match = _stub_match()
    match.kwargs["residual"] = _input_norm_residual()
    return match


@pytest.fixture
def no_tuned_preshuffle_config(monkeypatch):
    # no tuned preshuffle config for any shape, so only the guard under test decides
    monkeypatch.setattr(
        rocm_aiter_ops, "is_triton_gemm_afp4wfp4_presh_ws_tuned", lambda n, k: False
    )


@pytest.fixture
def input_pattern(rocm_aiter_fusion, no_tuned_preshuffle_config):
    return rocm_aiter_fusion.AiterAddRMSNormMxfp4GemmPattern(
        EPS, torch.bfloat16, tp_size=1, sites="input"
    )


@pytest.fixture
def pair_pattern(rocm_aiter_fusion, no_tuned_preshuffle_config):
    return rocm_aiter_fusion.AiterAddRMSNormMxfp4GemmPairPattern(
        EPS, torch.bfloat16, tp_size=1, sites="input"
    )


def test_each_pattern_takes_only_the_norms_of_its_site_kind(
    rocm_aiter_fusion, input_pattern, pattern
):
    # Input: a hand-made match of an input norm (the residual is an item of another
    # norm) and one of a post-attention norm (the residual is a graph input), given to
    # the pattern of each site kind. Expected: each pattern accepts its own kind and
    # refuses the other. Why: one pass per flag, and a flag must only switch on its own
    # sites.
    assert input_pattern.extra_check(_input_stub_match()) is True
    assert input_pattern.extra_check(_stub_match()) is False
    assert pattern.extra_check(_stub_match()) is True
    assert pattern.extra_check(_input_stub_match()) is False


def test_pair_match_of_two_different_gemms_is_accepted(pair_pattern):
    # Input: a hand-made match whose two GEMM slots hold two different nodes and whose
    # residual operand is an item of another norm's output (an input norm). Expected:
    # accepted. The control for the refusals below.
    match = _with_second_gemm(_input_stub_match(), ["a", "b", "r"])
    assert pair_pattern.extra_check(match) is True


def test_pair_match_with_both_slots_on_one_gemm_is_refused(pair_pattern):
    # Input: a match that binds both GEMM slots to the same node. Expected: refused.
    # Why: it would replace one GEMM twice and leave the other reading the normed
    # tensor that the rewrite removes.
    match = _with_second_gemm(_input_stub_match(), ["a", "a", "r"])
    assert pair_pattern.extra_check(match) is False


def test_pair_match_of_a_post_attention_norm_is_refused_by_the_input_pattern(
    pair_pattern,
):
    # Input: a pair match whose residual operand is a graph input (the post-attention
    # norm's kind of residual) given to the input pattern. Expected: refused.
    match = _with_second_gemm(_stub_match(), ["a", "b", "r"])
    assert pair_pattern.extra_check(match) is False


def test_pair_match_is_refused_when_the_second_gemm_is_tuned(pair_pattern, monkeypatch):
    # Input: a pair match with the preshuffle lookup True for the second GEMM's width
    # only. Expected: refused. Why: BOTH GEMMs must be eligible.
    monkeypatch.setattr(
        rocm_aiter_ops,
        "is_triton_gemm_afp4wfp4_presh_ws_tuned",
        lambda n, k_bytes: n == GDN_N_BA,
    )
    match = _with_second_gemm(_input_stub_match(), ["a", "b", "r"])
    assert pair_pattern.extra_check(match) is False


def test_pair_match_without_a_value_for_the_second_gemm_is_refused(pair_pattern):
    # Input: a pair match whose second GEMM weight has no tensor value. Expected:
    # refused, no exception. Why: shapes of every GEMM are needed to judge it.
    match = _with_second_gemm(_input_stub_match(), ["a", "b", "r"])
    match.kwargs["gemm_weight_2"].meta = {}
    assert pair_pattern.extra_check(match) is False


# --- the pass: site kinds, flags, labels --------------------------------------------


def test_each_pass_names_its_own_flag_and_label(make_norm_pass):
    # Input: the pass built for each site kind. Expected: it quotes its own flag and
    # names its own sites in the startup log; the default kind is the post-attention
    # one. Why: with both flags on, each pass is judged on its own and says which flag
    # is at fault.
    post, input_ = make_norm_pass("backbone"), make_norm_pass("backbone", sites="input")
    assert (post.sites, post.flag_name, post.site_label) == (
        "post",
        POSTNORM_FLAG,
        POST_LABEL,
    )
    assert (input_.sites, input_.flag_name, input_.site_label) == (
        "input",
        INPUTNORM_FLAG,
        INPUT_LABEL,
    )


def test_unknown_site_kind_is_refused(make_norm_pass, rocm_aiter_fusion):
    # Input: the pass and the pattern built for the site kind "both". Expected:
    # ValueError naming the kinds there are and the one asked for. Why: two flags mean
    # two passes, one per kind; a kind that is not one would build a pass that matches
    # nothing and fails only at the start-up check.
    with pytest.raises(ValueError) as raised:
        make_norm_pass("backbone", sites="both")
    for named in ("'post'", "'input'", "'both'"):
        assert named in str(raised.value)
    with pytest.raises(ValueError, match="sites must be one of"):
        rocm_aiter_fusion.AiterAddRMSNormMxfp4GemmPattern(
            EPS, torch.bfloat16, tp_size=1, sites="both"
        )


def test_each_pass_has_its_own_pattern_matcher_name(make_norm_pass):
    # Input: the pass of each site kind. Expected: two different names, the post one
    # the name the post pass always had. Why: the name keys the pattern matcher's match
    # table and the pattern dump files; two passes under one name would add their
    # counts and overwrite each other's dump.
    post, input_ = make_norm_pass("backbone"), make_norm_pass("backbone", sites="input")
    assert post.pass_name == "rocm_aiter_rmsnorm_mxfp4_gemm_fusion_pass"
    assert input_.pass_name == "rocm_aiter_input_rmsnorm_mxfp4_gemm_fusion_pass"


def test_pass_uuid_tells_the_site_kinds_apart_even_with_the_same_patterns(
    make_norm_pass, rocm_aiter_fusion, monkeypatch
):
    # Input: the passes of both kinds built with the same pattern classes (the table of
    # patterns changed so both kinds list the same). Expected: different uuids, and the
    # same kind built twice has the same uuid. Why: the uuid hashes the source of the
    # pass and pattern classes, and the kind is only a constructor argument of them;
    # the two kinds rewrite different norms, so a shared uuid would let one kind's
    # compiled graph be served for the other.
    pass_cls = rocm_aiter_fusion.RocmAiterRMSNormMxfp4GemmFusionPass
    monkeypatch.setitem(pass_cls.PATTERNS, "input", pass_cls.PATTERNS["post"])
    post, input_ = make_norm_pass("backbone"), make_norm_pass("backbone", sites="input")
    assert post.uuid() != input_.uuid()
    assert post.uuid() == make_norm_pass("backbone").uuid()


def test_input_pass_registers_a_single_and_a_pair_pattern_per_epsilon(make_norm_pass):
    # Input: the input pass as built for serving. Expected: four patterns, the single
    # and the pair for each of the two epsilons; the post pass keeps its two (single
    # only). Why: a post-attention norm with two consumers is not a site of the post
    # kind.
    patterns = make_norm_pass("backbone", sites="input")._pattern_replacements
    assert sorted(type(p).__name__ for p in patterns) == [
        "AiterAddRMSNormMxfp4GemmPairPattern",
        "AiterAddRMSNormMxfp4GemmPairPattern",
        "AiterAddRMSNormMxfp4GemmPattern",
        "AiterAddRMSNormMxfp4GemmPattern",
    ]
    assert len(make_norm_pass("backbone")._pattern_replacements) == 2


def test_input_pass_refuses_a_run_that_is_not_split_and_names_its_flag(make_norm_pass):
    # Input: the input pass built with inductor graph partition on, and with no
    # splitting ops. Expected: RuntimeError naming the INPUT flag and the setting. Why:
    # the input/post classification needs the split; the message must name the flag of
    # the pass that refuses.
    with pytest.raises(RuntimeError) as raised:
        make_norm_pass("backbone", sites="input", graph_partition=True)
    assert INPUTNORM_FLAG in str(raised.value)
    assert POSTNORM_FLAG not in str(raised.value)
    assert "use_inductor_graph_partition" in str(raised.value)
    with pytest.raises(RuntimeError) as raised:
        make_norm_pass("backbone", sites="input", splitting_ops=[])
    assert INPUTNORM_FLAG in str(raised.value)
    assert POSTNORM_FLAG not in str(raised.value)


# --- the site count, judged per flag --------------------------------------------------


def test_input_pass_with_a_site_logs_it_under_its_own_label(make_norm_pass, log):
    # Input: a target graph whose only piece has an input site with two consumers.
    # Expected: finish_compile logs the count (the pair counts as ONE site) with the
    # input label, the graph name and the compile range, and does not raise.
    norm_pass = make_norm_pass("backbone", sites="input")
    _run_pass(norm_pass, _trace(_input_pair_site, n=GDN_N_QKVZ))
    norm_pass.finish_compile()
    assert log.fused == [
        f"Fused {INPUT_LABEL} at 1 sites (graph=backbone) for compile range {RANGE}"
    ]


def test_input_pass_with_no_site_raises_naming_its_flag(make_norm_pass, log):
    # Input: a target graph where the input pass found no site (only a post-attention
    # site is in it). Expected: the 0 is logged under the input label, then a
    # RuntimeError naming the INPUT flag. Why: the flag is on but nothing was fused for
    # it.
    norm_pass = make_norm_pass("backbone", sites="input")
    _run_pass(norm_pass, _trace(_post_site()))
    with pytest.raises(RuntimeError) as raised:
        norm_pass.finish_compile()
    assert f"{INPUTNORM_FLAG} is on but 0 sites" in str(raised.value)
    assert POSTNORM_FLAG not in str(raised.value)
    assert log.fused == [
        f"Fused {INPUT_LABEL} at 0 sites (graph=backbone) for compile range {RANGE}"
    ]


def test_each_flag_is_judged_on_its_own_sites(make_norm_pass, log):
    # Input: both passes over a graph that has a post-attention site and no input
    # site. Expected: the post pass logs its site and passes; the input pass logs 0 and
    # raises with its own flag. Why: one flag's success must not cover for the other
    # flag's failure.
    post = make_norm_pass("backbone")
    input_ = make_norm_pass("backbone", sites="input")
    gm = _trace(_post_site())
    _run_pass(post, gm)
    _run_pass(input_, gm)
    post.finish_compile()
    with pytest.raises(RuntimeError, match=f"{INPUTNORM_FLAG} is on but 0 sites"):
        input_.finish_compile()
    assert log.fused == [
        f"Fused {POST_LABEL} at 1 sites (graph=backbone) for compile range {RANGE}",
        f"Fused {INPUT_LABEL} at 0 sites (graph=backbone) for compile range {RANGE}",
    ]


def test_input_pass_on_a_draft_graph_only_logs(make_norm_pass, log):
    # Input: the MTP head with no input site (its correct result: bf16 layers).
    # Expected: the 0 is logged under the draft graph's name and nothing raises.
    norm_pass = make_norm_pass("eagle_head", sites="input")
    _run_pass(norm_pass, _trace(_gemm_only))
    norm_pass.finish_compile()
    assert log.fused == [
        f"Fused {INPUT_LABEL} at 0 sites (graph=eagle_head) for compile range {RANGE}"
    ]


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


def _rel_err(y, reference):
    return ((y.float() - reference).norm() / reference.norm()).item()


def _asm_mxfp4_weight(n, k, scale=0.02):
    """A random bf16 weight and its MXFP4 form in the layouts the asm GEMM takes.

    The layouts are those AiterMxfp4LinearKernel.process_weights_after_loading makes.
    Returns (bf16 weight, packed weight, shuffled scales).
    """
    import aiter
    from aiter.ops.shuffle import shuffle_weight

    w = torch.randn(n, k, dtype=torch.bfloat16, device=DEVICE) * scale
    w_q, w_s = aiter.per_1x32_f4_quant_hip(w, shuffle=False)
    w_q, w_s = w_q.view(torch.uint8), w_s.view(torch.uint8)
    sm, sn = w_s.shape
    w_s = (
        w_s.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
        .permute(0, 3, 5, 2, 4, 1, 6)
        .contiguous()
        .view(sm, sn)
    )
    return w, shuffle_weight(w_q, layout=(16, 16)), w_s


@pytest.mark.skipif(
    not (_on_gfx950() and torch.cuda.is_available()), reason="needs a gfx950 GPU"
)
@pytest.mark.parametrize("num_tokens", [1, 64])
def test_rewritten_input_pair_is_as_accurate_as_the_unrewritten_one(
    make_norm_pass, num_tokens
):
    # Input: a gated-delta-net-style input norm at hidden size 1024 feeding two GEMMs
    # (N=2048 and N=256), traced once, run as traced (the IR norm in front of two GEMMs
    # that each quantize the normed tensor) and after the input pass (ONE fused op
    # feeding both GEMMs through x_scales), on the same activations, norm weights and
    # MXFP4 weights. Expected: for EACH GEMM the rewritten graph's error against the
    # fp32 reference is within 5 percent of the unrewritten graph's, and its residual
    # output equals the unrewritten graph's. Why: the shared quantization hands the
    # same codes and scales to a wide and a narrow GEMM; a layout only one of them
    # reads, or a wrong wiring, shows as an error of order 1 on one of the two.
    k, widths = 1024, (2048, 256)
    torch.manual_seed(0)
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device=DEVICE)
    residual = torch.randn(num_tokens, k, dtype=torch.bfloat16, device=DEVICE)
    norm_weight = torch.randn(k, dtype=torch.bfloat16, device=DEVICE) * 0.1
    input_norm_weight = torch.randn(k, dtype=torch.bfloat16, device=DEVICE) * 0.1
    weights = [_asm_mxfp4_weight(n, k) for n in widths]
    gemm_args = [t for _, w_q, w_s in weights for t in (w_q, w_s)]

    # the stand-in norm run eagerly gives the input norm's operands for the reference
    norm_in, residual_in = _norm(x, residual, norm_weight)
    summed = norm_in.float() + residual_in.float()
    rms = (summed.pow(2).mean(-1, keepdim=True) + EPS).rsqrt()
    normed = summed * rms * (input_norm_weight.float() + 1.0)

    norm_pass = make_norm_pass("backbone", sites="input")
    unrewritten = _trace(_input_pair_site, k=k, n=widths[0])
    rewritten = _trace(_input_pair_site, k=k, n=widths[0])
    _run_pass(norm_pass, rewritten)
    assert norm_pass.matched_count == 1
    rewritten.recompile()

    args = (x, residual, norm_weight, input_norm_weight, *gemm_args)
    *outs_before, residual_before = unrewritten(*args)
    *outs_after, residual_after = rewritten(*args)

    for (w, _, _), before, after in zip(weights, outs_before, outs_after):
        reference = normed @ w.float().T
        assert _rel_err(after, reference) <= _rel_err(before, reference) * 1.05 + 1e-3
    assert torch.equal(residual_after, residual_before)
