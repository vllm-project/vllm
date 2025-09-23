# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Callable

import torch
from torch import fx
from torch._inductor.pattern_matcher import (PatternMatcherPass, fwd_only,
                                             register_replacement)

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import round_up

from .inductor_pass import enable_fake_mode
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)


class PadPattern:
    """Fuse mul + constant_pad_nd for last-dim zero padding.
    eg.
        x = a * b
        y = F.pad(x, (0, 192), 0.0)
    """

    def __init__(self, roundup: Callable, device: str):
        self.roundup = roundup
        self.device = device

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(a, b):
            mul = torch.ops.aten.mul.Tensor(a, b)
            return torch.ops.aten.constant_pad_nd.default(
                self=mul,
                pad=[0, self.roundup(mul.shape[-1]) - mul.shape[-1]],
                value=0.0,
            )

        def replacement(a, b):
            new_shape = list(a.shape)
            shape = [0, self.roundup(a.shape[-1]) - a.shape[-1]]
            new_shape[-1] += shape[-1]

            result = torch.full(new_shape,
                                fill_value=0.0,
                                dtype=a.dtype,
                                device=a.device)
            result = torch.slice_scatter(result,
                                         a * b,
                                         dim=-1,
                                         start=0,
                                         end=a.shape[-1])

            return result

        inputs = [
            torch.empty(3, 12, device=self.device, dtype=torch.bfloat16),
            torch.empty(1, 12, device=self.device, dtype=torch.bfloat16),
        ]
        register_replacement(pattern, replacement, inputs, fwd_only, pm_pass)


class MulPadFusionPass(VllmPatternMatcherPass):
    """
    Fuse mul + pad into a single op to avoid memory copy.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig):
        super().__init__(config)
        device = config.device_config.device_type

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="mul_pad_fusion_pass")

        for i in [64, 128, 256]:
            PadPattern(roundup=lambda x, i=i: round_up(x, i),
                       device=device).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph):
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self) -> Any:
        return self.hash_source(self, PadPattern)
