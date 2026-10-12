# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch._inductor import pattern_matcher as pm
from torch.fx.experimental.proxy_tensor import make_fx

import vllm.config
from tests.compile.backend import TestBackend
from vllm.compilation.passes.vllm_inductor_pass import (
    VllmFusionPatternMatcherPass,
    VllmPatternMatcherPass,
    VllmPatternReplacement,
)
from vllm.config import CompilationConfig, CompilationMode, VllmConfig
from vllm.platforms import current_platform


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
        p3 = TwoPatternFusionPass(vllm_config)

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


class SelectiveReluToAbsPattern(ReluToAbsPattern):
    """Replaces relu(x) with abs(x) where `accept` allows, and records the matches."""

    def __init__(self, accept: bool) -> None:
        self.accept = accept
        self.seen: list[pm.Match] = []

    def extra_check(self, match: pm.Match) -> bool:
        self.seen.append(match)
        return self.accept


def _apply_to_two_relus(vllm_config, replacement) -> tuple[int, torch.fx.GraphModule]:
    """Registers the replacement in a pass and applies it to a graph with two relus."""
    with vllm.config.set_current_vllm_config(vllm_config):
        fusion_pass = VllmFusionPatternMatcherPass(vllm_config, "test_extra_check")
        fusion_pass.register(replacement)
    gm = make_fx(lambda x: torch.relu(torch.relu(x) + 1))(torch.randn(4))
    return fusion_pass.pm_pass.apply(gm.graph), gm


@pytest.mark.parametrize("accept", [True, False])
def test_extra_check_decides_which_matches_are_replaced(vllm_config, accept):
    """The pattern matcher asks a replacement's extra_check about every match, and
    replaces the matches it accepts.

    The nodes of a match are in `match.kwargs`, keyed by the pattern's parameter names.
    """
    replacement = SelectiveReluToAbsPattern(accept)
    replaced, gm = _apply_to_two_relus(vllm_config, replacement)

    assert len(replacement.seen) == 2
    assert {m.kwargs["x"].op for m in replacement.seen} == {
        "placeholder",
        "call_function",
    }
    assert replaced == (2 if accept else 0)
    ops = [node.target for node in gm.graph.nodes]
    assert ops.count(torch.ops.aten.abs.default) == replaced
    assert ops.count(torch.ops.aten.relu.default) == 2 - replaced


def test_default_extra_check_accepts_every_match(vllm_config):
    """A replacement without an extra_check override is replaced everywhere."""
    replaced, gm = _apply_to_two_relus(vllm_config, ReluToAbsPattern())
    assert replaced == 2
