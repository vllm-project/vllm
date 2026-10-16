# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import fx
from torch._inductor.pattern_matcher import (
    CallFunctionVarArgs,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)

from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.logger import init_logger
from vllm.model_executor.layers.utils import FLASHINFER_CUTEDSL_BF16_MAX_M

from ..vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)


class FlashInferBF16GemmFallbackPass(VllmInductorPass):
    """Replace vllm.cuda_flashinfer_bf16_gemm with aten linear in compile ranges
    above FLASHINFER_CUTEDSL_BF16_MAX_M tokens, where FlashInfer does not use its
    CuTeDSL kernel. Inductor then emits its extern mm instead of the opaque op,
    whose dispatch costs host time on every call in steps without CUDA graphs.
    """

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.patterns = PatternMatcherPass(self.pass_name)
        register_graph_pattern(
            CallFunctionVarArgs(torch.ops.vllm.cuda_flashinfer_bf16_gemm.default),
            pass_dict=self.patterns,
        )(self.replace_matched_op)

    @staticmethod
    def replace_matched_op(
        match: Match,
        x: fx.Node,
        weight: fx.Node,
        bias: fx.Node | None,
        pdl: bool,
        vllm_backend: str,
    ) -> None:
        match.replace_by_example(torch.nn.functional.linear, [x, weight, bias])

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        count = self.patterns.apply(graph)
        logger.debug("Replaced %d cuda_flashinfer_bf16_gemm nodes", count)

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return compile_range.start > FLASHINFER_CUTEDSL_BF16_MAX_M
