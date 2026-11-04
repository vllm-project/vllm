# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse a zentorch gate/up linear with SiluAndMul on AMD Zen CPUs.

    gate_up = zentorch_linear_unary(x, w13)            # bf16
            | zentorch_dynamic_qlinear(x, w13, s13)    # DA8W8 / DA8W4
    out = SiluAndMul(gate_up)

becomes `zentorch_linear_gated_act.out(x, w13, s13, activation="silu", out=out)`.
"""

import functools
from collections.abc import Callable

import torch
import torch._inductor.pattern_matcher as pm
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.logger import init_logger

from ..inductor_pass import InductorPass, enable_fake_mode
from ..vllm_inductor_pass import VllmFusionPatternMatcherPass, VllmPatternReplacement
from .matcher_utils import MatcherSiluAndMul

logger = init_logger(__name__)

GATED_ACT_OP = torch.ops.zentorch.zentorch_linear_gated_act.out
GATE_UP_OPS = (
    torch.ops.zentorch.zentorch_linear_unary.default,
    torch.ops.zentorch.zentorch_dynamic_qlinear.default,
)


def _silu_and_mul_matcher() -> MatcherSiluAndMul:
    # SiluAndMul.forward_cpu runs forward_native on x86, so the custom-op form
    # never reaches a Zen graph, even with +silu_and_mul.
    return MatcherSiluAndMul(enabled=False)


def _fused_gated_act(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor | None,
    is_weight_prepacked: bool,
) -> torch.Tensor:
    output = torch.empty(
        (x.shape[0], weight.shape[0] // 2), dtype=x.dtype, device=x.device
    )
    at = auto_functionalized(
        GATED_ACT_OP,
        out=output,
        input=x,
        weight=weight,
        weight_scales=weight_scales,
        activation="silu",
        is_weight_prepacked=is_weight_prepacked,
    )
    return at[1]


class ZentorchLinearSiluMulPattern(VllmPatternReplacement):
    """bf16 zentorch_linear_unary gate/up (no bias) + SiluAndMul, as emitted by
    dispatch_cpu_unquantized_gemm."""

    def __init__(self, is_weight_prepacked: bool) -> None:
        self.is_weight_prepacked = is_weight_prepacked
        self.silu_and_mul_matcher = _silu_and_mul_matcher()

    def get_inputs(self) -> list[torch.Tensor]:
        return [self.empty_bf16(5, 16), self.empty_bf16(32, 16)]

    @property
    def pattern(self) -> Callable[..., torch.Tensor]:
        def _pattern(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            gate_up = torch.ops.zentorch.zentorch_linear_unary(
                x, weight, None, is_weight_prepacked=self.is_weight_prepacked
            )
            return self.silu_and_mul_matcher(gate_up)

        return _pattern

    @property
    def replacement(self) -> Callable[..., torch.Tensor]:
        def _replacement(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return _fused_gated_act(x, weight, None, self.is_weight_prepacked)

        return _replacement


class ZentorchQLinearSiluMulPattern(VllmPatternReplacement):
    """DA8W8 / DA8W4 zentorch_dynamic_qlinear gate/up (no bias) + SiluAndMul,
    as emitted by ZentorchInt8ScaledMMLinearKernel and
    ZentorchWNA16LinearKernel. Tracing drops arguments equal to their default
    (the None bias, the default zentorch_op_name), so both calls give this one
    pattern."""

    is_weight_prepacked = False

    def __init__(self) -> None:
        self.silu_and_mul_matcher = _silu_and_mul_matcher()

    def get_inputs(self) -> list[torch.Tensor]:
        return [
            self.empty_bf16(5, 16),
            self.empty(32, 16, dtype=torch.int8),
            self.empty_fp32(32),
        ]

    @property
    def pattern(self) -> Callable[..., torch.Tensor]:
        def _pattern(
            x: torch.Tensor, weight: torch.Tensor, weight_scales: torch.Tensor
        ) -> torch.Tensor:
            gate_up = torch.ops.zentorch.zentorch_dynamic_qlinear.default(
                x, weight, weight_scales, None
            )
            return self.silu_and_mul_matcher(gate_up)

        return _pattern

    @property
    def replacement(self) -> Callable[..., torch.Tensor]:
        def _replacement(
            x: torch.Tensor, weight: torch.Tensor, weight_scales: torch.Tensor
        ) -> torch.Tensor:
            return _fused_gated_act(x, weight, weight_scales, False)

        return _replacement


def _is_fusable(match: pm.Match, is_weight_prepacked: bool) -> bool:
    # zentorch_linear_gated_act takes a 2-D bf16 input only.
    x = match.kwargs["x"].meta["val"]
    if x.dtype != torch.bfloat16 or x.dim() != 2:
        return False
    # A pattern traced with the default is_weight_prepacked=False omits the
    # kwarg and so also matches a prepacked call; check the matched value.
    gate_up = next(n for n in match.nodes if n.target in GATE_UP_OPS)
    if gate_up.kwargs.get("is_weight_prepacked", False) != is_weight_prepacked:
        return False
    # The pattern matcher ignores slice bounds, so check that SiluAndMul reads
    # silu(gate_up[:, :I]) * gate_up[:, I:].
    up = match.output_node().args[1]
    gate = next(
        n
        for n in match.nodes
        if n.target == torch.ops.aten.slice.Tensor and n is not up
    )
    width = gate_up.meta["val"].shape[-1]
    _, gate_dim, gate_start, gate_end = gate.args
    _, up_dim, up_start, _ = up.args
    return (
        gate_dim in (1, -1)
        and up_dim in (1, -1)
        and gate_start == 0
        and up_start == gate_end
        and 2 * gate.meta["val"].shape[-1] == width
        and 2 * up.meta["val"].shape[-1] == width
    )


class ZentorchGatedActFusionPass(VllmFusionPatternMatcherPass):
    """Fuses a zentorch gate/up linear with SiluAndMul into
    zentorch_linear_gated_act, which applies the activation in the GEMM
    epilogue and writes only the [M, I] output."""

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config, "zentorch_gated_act_fusion_pass")

        for is_weight_prepacked in (False, True):
            self.register(ZentorchLinearSiluMulPattern(is_weight_prepacked))
        self.register(ZentorchQLinearSiluMulPattern())

        self.dump_patterns(config, self.pm_pass)

    @enable_fake_mode
    def register(
        self, pr: ZentorchLinearSiluMulPattern | ZentorchQLinearSiluMulPattern
    ) -> None:
        pm.register_replacement(
            pr.pattern,
            pr.replacement,
            pr.get_inputs(),
            self._trace_fn,
            self.pm_pass,
            extra_check=functools.partial(
                _is_fusable, is_weight_prepacked=pr.is_weight_prepacked
            ),
        )
        self._pattern_replacements.append(pr)


class ZenCpuPostGradPass(InductorPass):
    """Runs the Zen CPU fusion passes from Inductor's post_grad_custom_post_pass.

    The CPU platform compiles in DYNAMO_TRACE_ONCE mode with the plain Inductor
    backend, so PostGradPassManager never runs there; ZenCpuPlatform installs
    this pass in inductor_compile_config instead.
    """

    def __init__(self, config: VllmConfig) -> None:
        self.config = config
        self._passes: list[VllmFusionPatternMatcherPass] | None = None

    def __getstate__(self) -> dict:
        # The registered patterns do not pickle; they are rebuilt on first use.
        return {"config": self.config, "_passes": None}

    def __call__(self, graph: torch.fx.Graph) -> None:
        if self._passes is None:
            # The CPU worker compiles outside a set_current_vllm_config context.
            with set_current_vllm_config(self.config, check_compile=False):
                self._passes = [ZentorchGatedActFusionPass(self.config)]
        for pass_ in self._passes:
            pass_(graph)
            logger.debug("%s matched %d patterns", pass_.pass_name, pass_.matched_count)

    def uuid(self) -> str:
        pass_config = self.config.compilation_config.pass_config
        return InductorPass.hash_dict(
            {
                "source": InductorPass.hash_source(
                    ZenCpuPostGradPass,
                    ZentorchGatedActFusionPass,
                    ZentorchLinearSiluMulPattern,
                    ZentorchQLinearSiluMulPattern,
                    _is_fusable,
                ),
                "pass_config": pass_config.compile_factors(),
            }
        )
