# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for the VllmBackendWithCache class.
"""

import torch

from vllm.compilation.backends import VllmBackendWithCache
from vllm.compilation.caching import VllmSerializableFunction
from vllm.config import CompilationConfig, CompilationMode, VllmConfig
from vllm.config.vllm import get_current_vllm_config

InductorCompiledArtifacts = VllmSerializableFunction.InductorCompiledArtifacts


def make_vllm_config() -> VllmConfig:
    """Create a test VllmConfig."""
    return VllmConfig(
        compilation_config=CompilationConfig(
            level=CompilationMode.VLLM_COMPILE,
        )
    )


class TestVllmBackendWithCache:
    """Test the VllmBackendWithCache class."""

    def test_init(self):
        """Test initialization of VllmBackendWithCache."""
        inductor_compiled_artifacts = (
            VllmSerializableFunction.InductorCompiledArtifacts()
        )
        vllm_config = make_vllm_config()

        backend = VllmBackendWithCache(
            inductor_compiled_artifacts=inductor_compiled_artifacts,
            vllm_config=vllm_config,
            prefix="test",
            submod_names=["submod_0", "submod_1"],
        )

        assert backend.inductor_compiled_artifacts == inductor_compiled_artifacts
        assert backend.vllm_config == vllm_config
        assert backend.prefix == "test"
        assert backend.submod_names == ["submod_0", "submod_1"]
        assert backend.compiled_callables == {"submod_0": {}, "submod_1": {}}

    def test_build_dispatch_callable_empty_cache(self):
        """Test building dispatch callable with an empty cache."""
        inductor_compiled_artifacts = (
            VllmSerializableFunction.InductorCompiledArtifacts()
        )
        vllm_config = make_vllm_config()

        backend = VllmBackendWithCache(
            inductor_compiled_artifacts=inductor_compiled_artifacts,
            vllm_config=vllm_config,
            prefix="test",
            submod_names=["submod_0"],
        )

        assert backend.compiled_callables == {"submod_0": {}}

    def test_call_empty_submodules(self):
        """Test creating split_gm with empty submod_names."""
        backend = VllmBackendWithCache(
            inductor_compiled_artifacts=InductorCompiledArtifacts(),
            vllm_config=get_current_vllm_config(),
            prefix="test",
            submod_names=[],
            sym_shape_indices_map={},
            returns_tuple_map={},
        )

        import torch.fx as fx

        gm = fx.GraphModule(torch.nn.Module(), fx.Graph())

        assert backend.create_split_gm_from_cache(gm) is gm
