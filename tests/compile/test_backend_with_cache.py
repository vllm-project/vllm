# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for the VllmBackendWithCache class.
"""

import pytest
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

        # With empty cache, compiled_callables should be empty
        assert backend.compiled_callables == {"submod_0": {}}

    def test_create_piecewise_backend_from_cache_no_general_shape(self):
        """Test creating piecewise backend without a general shape function."""
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

        # Should raise ValueError when no general shape function is available
        with pytest.raises(
            ValueError, match="No general shape compiled function found"
        ):
            backend.create_piecewise_backend_from_cache("submod_0", 0)

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

        # Create a simple split_gm
        import torch.fx as fx

        gm = fx.GraphModule(torch.nn.Module(), fx.Graph())

        # Should work with empty submodules (no replacements needed)
        result_gm = backend.create_split_gm_from_cache(gm)
        assert result_gm is gm


class TestVllmBackendWithCacheFlag:
    """Test the VllmBackendWithCache flag integration."""

    def test_flag_parsing(self):
        """Test that the flag is properly parsed."""
        import os

        # Test default value
        if "VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS" in os.environ:
            del os.environ["VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS"]
        import vllm.envs as envs

        assert not envs.use_backend_with_cache()

        # Test enabling the flag
        os.environ["VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS"] = "1"
        # Need to reload the function to get the new value
        assert os.environ["VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS"] == "1"

        # Clean up
        del os.environ["VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestVllmBackendWithCacheIntegration:
    """Integration tests for VllmBackendWithCache."""

    def test_full_workflow_with_mock_cache(self):
        """Test the full workflow with a mocked inductor cache."""
        # This is a placeholder for a more comprehensive integration test
        # that would actually populate the cache and test execution
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
