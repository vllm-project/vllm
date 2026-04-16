# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.config
from tests.compile.backend import TestBackend
from vllm.platforms import current_platform
from vllm.compilation.passes.vllm_inductor_pass import (
    VllmFusionPatternMatcherPass,
    VllmPatternMatcherPass,
    VllmPatternReplacement,
)
from vllm.config import CompilationConfig, CompilationMode, VllmConfig



class ReluToAbsPattern(VllmPatternReplacement):
    """Replaces relu(x) with abs(x) — a minimal test fixture."""

    @property
    def pattern(self):
        def _pattern(x: torch.Tensor) -> torch.Tensor:
            return torch.ops.aten.relu.default(x)

        return _pattern

    @property
    def replacement(self):
        def _replacement(x: torch.Tensor) -> torch.Tensor:
            return torch.ops.aten.abs.default(x)

        return _replacement

    def get_inputs(self) -> list[torch.Tensor]:
        return [self.empty_fp32(4)]


class ExpToSqrtPattern(VllmPatternReplacement):
    """A second distinct pattern type — used to test uuid differentiation."""

    @property
    def pattern(self):
        def _pattern(x: torch.Tensor) -> torch.Tensor:
            return torch.ops.aten.exp.default(x)

        return _pattern

    @property
    def replacement(self):
        def _replacement(x: torch.Tensor) -> torch.Tensor:
            return torch.ops.aten.sqrt.default(x)

        return _replacement

    def get_inputs(self) -> list[torch.Tensor]:
        return [self.empty_fp32(4)]



class ReluFusionPass(VllmFusionPatternMatcherPass):
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config, "test_relu_fusion")
        self.register(ReluToAbsPattern())


class TwoPatternFusionPass(VllmFusionPatternMatcherPass):
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config, "test_two_pattern_fusion")
        self.register(ReluToAbsPattern())
        self.register(ExpToSqrtPattern())



@pytest.fixture
def vllm_config():
    return VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE),
    )

@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="Requires CUDA")
def test_register_tracks_patterns(vllm_config):
    """register() appends each VllmPatternReplacement to _pattern_replacements."""
    with vllm.config.set_current_vllm_config(vllm_config):
        single = ReluFusionPass(vllm_config)
        two = TwoPatternFusionPass(vllm_config)

    assert len(single._pattern_replacements) == 1
    assert len(two._pattern_replacements) == 2


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="Requires CUDA")
def test_uuid_stable(vllm_config):
    """Two instances of the same pass class produce identical uuids."""
    with vllm.config.set_current_vllm_config(vllm_config):
        p1 = ReluFusionPass(vllm_config)
        p2 = ReluFusionPass(vllm_config)
        p3= TwoPatternFusionPass(vllm_config)

    assert p1.uuid() == p2.uuid()
    assert p1.uuid() != p3.uuid()
    assert p2.uuid() != p3.uuid()


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="Requires CUDA")
@pytest.mark.parametrize("N", [1, 2, 4])
def test_matched_count_and_match_table(vllm_config, N):
    """matched_count and match_table reflect the number of matched patterns."""

    class Model(torch.nn.Module):
        def forward(self, *inputs):
            # N independent relus
            return sum(torch.relu(x) for x in inputs)

    with vllm.config.set_current_vllm_config(vllm_config):
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.float32)

        fusion_pass = ReluFusionPass(vllm_config)
        backend = TestBackend(fusion_pass)
        model = torch.compile(Model(), backend=backend)

        inputs = [torch.rand(8) for _ in range(N)]
        model(*inputs)

    assert fusion_pass.matched_count == N
    assert VllmPatternMatcherPass.match_table["test_relu_fusion"] >= N
