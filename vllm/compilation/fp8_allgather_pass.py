# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger

from .fp8_collective_ops import vllm_all_gather_fp8
from .inductor_pass import enable_fake_mode
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)


class AllGatherFP8Pattern:
    """Optimize AllGather + FP8 quantization by quantizing before AllGather

    Matches: AllGather(BF16) -> input_to_float8()
    Where input_to_float8 decomposes into:
        aminmax -> abs -> max -> clamp -> div -> mul -> clamp -> to(fp8)
    """

    def __init__(self, device: str, dtype: torch.dtype, tp_size: int,
                 tp_group_name: str):
        self.device = device
        self.dtype = dtype
        self.tp_size = tp_size
        self.tp_group_name = tp_group_name
        self.fp8_dtype = torch.float8_e4m3fn

    def get_inputs(self):
        # BF16 tensor that will be all-gathered, then quantized to FP8
        x = torch.empty([8, 16], device=self.device, dtype=self.dtype)
        # Precomputed FP8 scale (scalar)
        scale = torch.empty([], device=self.device, dtype=torch.float32)
        return [x, scale]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            # Match: AllGather(BF16) -> modelopt FP8 quantization
            # This matches what's in the FX graph from modelopt quant
            gathered_bf16 = torch.ops.vllm.all_gather.default(
                x,
                dim=0,  # Actual dimension used in the graph
                world_size=self.tp_size,
                group_name=self.tp_group_name,
            )

            # Modelopt quantization pattern (uses precomputed scale):
            # convert to fp32 -> multiply by 1/scale -> clamp -> convert to fp8
            x_f32 = gathered_bf16.to(torch.float32)
            scale_inv = scale.reciprocal()
            x_scaled = x_f32 * scale_inv
            x_clamped = x_scaled.clamp(min=-448.0, max=448.0)
            gathered_fp8 = x_clamped.to(self.fp8_dtype)

            return gathered_fp8

        def replacement(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            # Step 1: Quantize to FP8 locally BEFORE AllGather
            # Use the same modelopt quantization logic
            x_f32 = x.to(torch.float32)
            scale_inv = scale.reciprocal()
            x_scaled = x_f32 * scale_inv
            x_clamped = x_scaled.clamp(min=-448.0, max=448.0)
            x_fp8 = x_clamped.to(self.fp8_dtype)

            # Step 2: AllGather FP8 tensors (2x less bandwidth!)
            gathered_fp8 = vllm_all_gather_fp8(
                x_fp8,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp_group_name,
            )

            return gathered_fp8

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class FP8AllGatherOptPass(VllmPatternMatcherPass):
    """Optimize AllGather by quantizing to FP8 first (2x bandwidth reduction)"""

    @enable_fake_mode
    def __init__(self, config: VllmConfig):
        super().__init__(config)

        self.disabled = False  # Initialize disabled flag
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size <= 1:
            self.disabled = True
            logger.info(
                "FP8 AllGather optimization disabled: TP size = %d "
                "(no communication needed)", self.tp_size)
            return

        from vllm.distributed import get_tp_group
        self.tp_group_name = get_tp_group().unique_name

        self.patterns = PatternMatcherPass(pass_name="fp8_allgather_opt_pass")

        # Only apply to BF16 models (FP8 requires BF16 output dtype)
        if self.model_dtype == torch.bfloat16:
            AllGatherFP8Pattern(
                self.device,
                self.model_dtype,
                self.tp_size,
                self.tp_group_name,
            ).register(self.patterns)
            logger.info(
                "FP8 AllGather optimization enabled: "
                "TP size = %d, dtype = %s", self.tp_size, self.model_dtype)
        else:
            self.disabled = True
            logger.info(
                "FP8 AllGather optimization disabled: "
                "model dtype = %s (requires BF16)", self.model_dtype)

        if not self.disabled:
            self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph):
        if getattr(self, 'disabled', False):
            return

        self.matched_count = self.patterns.apply(graph)
        if self.matched_count > 0:
            logger.info(
                "FP8 AllGather optimization: replaced %d AllGather "
                "operation(s) with FP8 quantized versions",
                self.matched_count)
        else:
            logger.debug(
                "FP8 AllGather optimization: "
                "no matching patterns found in graph")
