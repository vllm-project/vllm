# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest
import torch

from tests.compile.backend import TestBackend
from vllm._aiter_ops import IS_AITER_FOUND, rocm_aiter_ops
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.activation import GeluAndMul, SiluAndMul
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
        RocmAiterActMulMxfp4GemmFusionPass,
    )

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and IS_AITER_FOUND),
    reason="ROCm with AITER only",
)

DTYPE = torch.bfloat16
HIDDEN_SIZE = 1024


def _is_gfx950() -> bool:
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


def _skip_unless_gfx950() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")
    if not _is_gfx950():
        pytest.skip("AITER MXFP4 GEMM needs gfx950")


@contextmanager
def _fusion_pass_under_test(
    custom_ops: list[str],
    monkeypatch: pytest.MonkeyPatch,
    presh_ws_tuned: bool = False,
) -> Iterator[tuple["RocmAiterActMulMxfp4GemmFusionPass", TestBackend]]:
    """Yield the act fusion pass and a TestBackend that runs it.

    Compiles with the given custom ops, bfloat16 on the default CUDA/HIP device, and
    a fixed seed. AITER is on and the tuned preshuffled lookup is pinned to
    ``presh_ws_tuned`` (off by default) until the block exits.
    """
    torch.set_default_device("cuda")
    torch.set_default_dtype(DTYPE)
    torch.manual_seed(0)
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=custom_ops,
            backend="eager",
            pass_config=PassConfig(fuse_act_quant=True, eliminate_noops=True),
        ),
    )

    with set_current_vllm_config(config), monkeypatch.context() as m:
        m.setenv("VLLM_ROCM_USE_AITER", "1")
        rocm_aiter_ops.refresh_env_variables()
        # Off keeps the fused path under test at every M. On sends M <= 64 down the
        # small-M branch, which runs the activation as a separate kernel call.
        m.setattr(
            rocm_aiter_ops,
            "is_triton_gemm_afp4wfp4_presh_ws_tuned",
            lambda n, k: presh_ws_tuned,
        )
        from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
            RocmAiterActMulMxfp4GemmFusionPass,
        )

        fusion_pass = RocmAiterActMulMxfp4GemmFusionPass(config)
        backend = TestBackend(
            NoOpEliminationPass(config), fusion_pass, PostCleanupPass(config)
        )
        yield fusion_pass, backend


class ActMulMxfp4Model(torch.nn.Module):
    """act_and_mul feeding an MXFP4 weight through the AITER ASM GEMM op."""

    def __init__(
        self,
        act: torch.nn.Module,
        hidden_size: int,
        dtype: torch.dtype,
        use_asm_gemm: bool,
    ):
        super().__init__()
        import aiter
        from aiter.ops.shuffle import shuffle_weight

        self.act = act
        self.dtype = dtype
        self.use_asm_gemm = use_asm_gemm
        w = torch.randn(hidden_size, hidden_size, dtype=dtype) * 0.02
        self.w_ref = w
        w_q, w_s = aiter.per_1x32_f4_quant_hip(w, shuffle=False)
        w_q, w_s = w_q.view(torch.uint8), w_s.view(torch.uint8)
        # The layouts AiterMxfp4LinearKernel.process_weights_after_loading
        # produces for each GEMM path.
        if use_asm_gemm:
            sm, sn = w_s.shape
            w_s = (
                w_s.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
                .permute(0, 3, 5, 2, 4, 1, 6)
                .contiguous()
                .view(sm, sn)
            )
            w_q = shuffle_weight(w_q, layout=(16, 16))
        else:
            w_s = w_s.T.contiguous()
        self.weight = torch.nn.Parameter(w_q, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(w_s, requires_grad=False)

    def exact(self, x: torch.Tensor, act_name: str) -> torch.Tensor:
        """act(gate) * up in fp32 against the unquantized weight."""
        d = x.shape[-1] // 2
        gate, up = x[..., :d].float(), x[..., d:].float()
        if act_name == "silu":
            h = torch.nn.functional.silu(gate) * up
        else:
            approximate = "tanh" if act_name == "gelu_tanh" else "none"
            h = torch.nn.functional.gelu(gate, approximate=approximate) * up
        return h @ self.w_ref.float().T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(x)
        return torch.ops.vllm.gemm_with_dynamic_quant(
            y, self.weight, self.weight_scale, self.use_asm_gemm, self.dtype
        )


# Input: num_tokens rows. Expected: one fused GEMM, and a fused error against
# the exact product within 1.05x of the eager path's. 72 is a decode capture
# size above 64 whose last 32-row tile is partial (24 rows masked): the only
# parametrized M with several tiles and a partial one.
@pytest.mark.parametrize("use_asm_gemm", [True, False])
@pytest.mark.parametrize("num_tokens", [16, 72, 256])
@pytest.mark.parametrize("act_name", ["silu", "gelu", "gelu_tanh"])
@pytest.mark.parametrize("enable_act_custom_op", [True, False])
def test_rocm_aiter_act_mul_mxfp4_gemm_fusion(
    use_asm_gemm: bool,
    num_tokens: int,
    act_name: str,
    enable_act_custom_op: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    _skip_unless_gfx950()
    custom_op = "gelu_and_mul" if act_name.startswith("gelu") else "silu_and_mul"
    custom_ops = ["none"] + ([f"+{custom_op}"] if enable_act_custom_op else [])
    with _fusion_pass_under_test(custom_ops, monkeypatch) as (fusion_pass, backend):
        if act_name == "silu":
            act = SiluAndMul()
        else:
            act = GeluAndMul("tanh" if act_name == "gelu_tanh" else "none")
        model = ActMulMxfp4Model(act, HIDDEN_SIZE, DTYPE, use_asm_gemm)

        x = torch.randn(num_tokens, 2 * HIDDEN_SIZE)
        torch._dynamo.mark_dynamic(x, 0)
        ref = model(x)
        out = torch.compile(model, backend=backend)(x)

        if not use_asm_gemm:
            # The Triton GEMM path keeps Inductor's act_and_mul: measured on
            # MI355X, aiter's fused act + quant was slower there.
            assert fusion_pass.matched_count == 0
            return
        assert fusion_pass.matched_count == 1
        gemm_nodes = [
            n
            for n in backend.graph_post_pass.nodes
            if n.target == torch.ops.vllm.gemm_with_dynamic_quant.default
        ]
        assert len(gemm_nodes) == 1
        # The fused op must compute what the matched graph computed: on ROCm the
        # native GeluAndMul drops the tanh approximation (see forward_native).
        if act_name == "silu":
            expected = "silu"
        elif act_name == "gelu_tanh" and (
            enable_act_custom_op or not current_platform.is_rocm()
        ):
            expected = "gelu_tanh"
        else:
            expected = "gelu"
        assert gemm_nodes[0].args[-1] == expected

        # The fused path quantizes act(gate) * up with aiter's Triton kernel
        # instead of the HIP quant, so individual MXFP4 roundings differ. Hold it
        # to the unfused path's accuracy against the exact product.
        exact = model.exact(x, act_name)

        def rel(y: torch.Tensor) -> float:
            return ((y.float() - exact).norm() / exact.norm()).item()

        assert rel(out) <= rel(ref) * 1.05 + 1e-3, (rel(out), rel(ref))


@pytest.mark.parametrize("act_name", ["silu", "gelu", "gelu_tanh"])
def test_rocm_aiter_act_mul_mxfp4_small_m_tuned_path_matches_eager(
    act_name: str, monkeypatch: pytest.MonkeyPatch
):
    # Input: 16 rows of gate and up with the tuned preshuffled lookup pinned on, so the
    # fused op takes the M <= 64 branch that runs _act_and_mul as a separate
    # kernel call.
    # Expected: the pass still matches the one act-and-GEMM pair, and the compiled
    # output equals the eager model's output within bfloat16 tolerance.
    # Why: this branch had no test, so a change to _act_and_mul could break the
    # small-M path without any test noticing.
    _skip_unless_gfx950()
    custom_op = "gelu_and_mul" if act_name.startswith("gelu") else "silu_and_mul"
    with _fusion_pass_under_test(
        [f"+{custom_op}"], monkeypatch, presh_ws_tuned=True
    ) as (fusion_pass, backend):
        if act_name == "silu":
            act = SiluAndMul()
        else:
            act = GeluAndMul("tanh" if act_name == "gelu_tanh" else "none")
        model = ActMulMxfp4Model(act, HIDDEN_SIZE, DTYPE, use_asm_gemm=True)

        x = torch.randn(16, 2 * HIDDEN_SIZE)
        torch._dynamo.mark_dynamic(x, 0)
        ref = model(x)
        out = torch.compile(model, backend=backend)(x)

        assert fusion_pass.matched_count == 1
        torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("num_tokens", [16, 256])
def test_asm_mxfp4_gemm_on_a_plain_input_is_not_rewritten(
    num_tokens: int, monkeypatch: pytest.MonkeyPatch
):
    # Input: an asm MXFP4 GEMM fed by a plain tensor, as qkv_proj is, with no SiLU or
    # GELU-and-mul in front. Expected: the pass matches nothing, the graph still calls
    # gemm_with_dynamic_quant with no activation, and the output equals the eager one.
    # Why: a loosened pattern would apply an activation to a qkv_proj input and give
    # wrong output silently.
    _skip_unless_gfx950()
    with _fusion_pass_under_test(["none"], monkeypatch) as (fusion_pass, backend):
        # Identity in front: the GEMM's input is the plain tensor itself.
        model = ActMulMxfp4Model(
            torch.nn.Identity(), HIDDEN_SIZE, DTYPE, use_asm_gemm=True
        )

        x = torch.randn(num_tokens, HIDDEN_SIZE)
        torch._dynamo.mark_dynamic(x, 0)
        ref = model(x)
        out = torch.compile(model, backend=backend)(x)

        assert fusion_pass.matched_count == 0
        gemm_nodes = [
            n
            for n in backend.graph_post_pass.nodes
            if n.target == torch.ops.vllm.gemm_with_dynamic_quant.default
        ]
        assert len(gemm_nodes) == 1
        # The activation is the op's seventh argument or keyword. The plain call
        # passes neither, so the activation is None.
        gemm_node = gemm_nodes[0]
        activation = gemm_node.args[6] if len(gemm_node.args) > 6 else None
        assert activation is None and "activation" not in gemm_node.kwargs
        torch.testing.assert_close(out, ref)


@contextmanager
def _act_mul_pass_built_with(
    custom_ops: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator["RocmAiterActMulMxfp4GemmFusionPass"]:
    """Yield the act fusion pass built with the given custom ops.

    Unlike _fusion_pass_under_test this leaves the torch default device and dtype
    alone and builds no TestBackend, so the pass can be inspected without a GPU
    device.
    """
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=custom_ops,
            backend="eager",
            pass_config=PassConfig(fuse_act_quant=True, eliminate_noops=True),
        ),
    )

    with set_current_vllm_config(config), monkeypatch.context() as m:
        m.setenv("VLLM_ROCM_USE_AITER", "1")
        rocm_aiter_ops.refresh_env_variables()
        from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
            RocmAiterActMulMxfp4GemmFusionPass,
        )

        yield RocmAiterActMulMxfp4GemmFusionPass(config)


def _record_gelu_builds(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Return a list collecting the approximate mode of every GeluAndMul built.

    Install the spy before the pass is built: the pass constructs its matchers, and
    with them the activation layers, in its own __init__.
    """
    built: list[str] = []
    original_init = GeluAndMul.__init__

    def recording_init(self: GeluAndMul, approximate: str = "none") -> None:
        built.append(approximate)
        original_init(self, approximate)

    monkeypatch.setattr(GeluAndMul, "__init__", recording_init)
    return built


def _registered_activations(
    fusion_pass: "RocmAiterActMulMxfp4GemmFusionPass",
) -> list[str]:
    """The activation names the pass registered patterns for, sorted.

    The pass keeps every pattern replacement it registered in _pattern_replacements,
    and the isinstance check gives the type checker the activation attribute.
    """
    from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
        AiterActMulMxfp4GemmPattern,
    )

    return sorted(
        pr.activation
        for pr in fusion_pass._pattern_replacements
        if isinstance(pr, AiterActMulMxfp4GemmPattern)
    )


# Input: custom_ops = ["none"], so the gelu_and_mul custom op is off and only the
# native GELU graph can ever be matched. Expected: no GeluAndMul is built with the
# tanh approximation. Why: the pass built MatcherGeluAndMul("tanh") anyway, which
# constructs GeluAndMul("tanh") and logs ROCm's tanh warning, even though the matcher
# is later dropped when its key collapses onto "gelu".
def test_act_mul_pass_builds_no_tanh_gelu_when_gelu_custom_op_is_off(
    monkeypatch: pytest.MonkeyPatch,
):
    built = _record_gelu_builds(monkeypatch)
    with _act_mul_pass_built_with(["none"], monkeypatch):
        assert "tanh" not in built


# Input: custom_ops = ["none"], so both GELU matchers trace to the same native graph.
# Expected: the pass registers exactly the "gelu" and "silu" patterns. Why: guard for
# the fix, dropping the tanh build must not drop the exact GELU fusion, the only GELU
# one ROCm can match.
def test_act_mul_pass_registers_only_silu_and_gelu_when_gelu_custom_op_is_off(
    monkeypatch: pytest.MonkeyPatch,
):
    with _act_mul_pass_built_with(["none"], monkeypatch) as fusion_pass:
        assert _registered_activations(fusion_pass) == ["gelu", "silu"]


# Input: custom_ops = ["none", "+gelu_and_mul"], so the custom-op form of GELU is
# matched and tanh is a graph of its own. Expected: a GeluAndMul with the tanh
# approximation is built and the pass registers "gelu", "gelu_tanh" and "silu".
# Why: guard for the fix, the tanh build is only wrong while the custom op is off.
def test_act_mul_pass_registers_gelu_tanh_when_gelu_custom_op_is_on(
    monkeypatch: pytest.MonkeyPatch,
):
    built = _record_gelu_builds(monkeypatch)
    with _act_mul_pass_built_with(
        ["none", "+gelu_and_mul"], monkeypatch
    ) as fusion_pass:
        assert "tanh" in built
        assert _registered_activations(fusion_pass) == ["gelu", "gelu_tanh", "silu"]


# Input: the pass built with custom_ops ["none"], and the base __call__ reporting 3
# rewritten sites.
# Expected: exactly one INFO record from the pass logger, with the text "AITER act-mul
# + MXFP4 GEMM fusion: replaced 3 pattern(s)".
# Why: the count is the pass's only log line. A level change back to debug would hide
# it on every start, and this test must fail then.
def test_act_mul_pass_call_logs_rewritten_count_at_info(
    monkeypatch: pytest.MonkeyPatch, caplog_vllm: pytest.LogCaptureFixture
):
    from vllm.compilation.passes.vllm_inductor_pass import (
        VllmFusionPatternMatcherPass,
    )

    def count_three_sites(
        self: VllmFusionPatternMatcherPass, graph: torch.fx.Graph
    ) -> None:
        self.matched_count = 3

    monkeypatch.setattr(VllmFusionPatternMatcherPass, "__call__", count_three_sites)

    pass_logger = "vllm.compilation.passes.fusion.rocm_aiter_fusion"
    with (
        _act_mul_pass_built_with(["none"], monkeypatch) as fusion_pass,
        caplog_vllm.at_level(logging.INFO, logger=pass_logger),
    ):
        fusion_pass(torch.fx.Graph())

    records = [r for r in caplog_vllm.records if r.name == pass_logger]
    assert [(r.levelno, r.getMessage()) for r in records] == [
        (logging.INFO, "AITER act-mul + MXFP4 GEMM fusion: replaced 3 pattern(s)")
    ]
