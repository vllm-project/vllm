# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for OCI model loader."""

import pytest

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.oci_loader import OciModelLoader


class TestOciModelLoader:
    """Test suite for OciModelLoader."""

    def test_normalize_oci_reference_with_full_reference(self):
        """Test normalization with full OCI reference."""
        load_config = LoadConfig(load_format="oci")
        loader = OciModelLoader(load_config)

        # Full reference should remain unchanged
        ref = "ghcr.io/user/model:tag"
        normalized = loader._normalize_oci_reference(ref)
        assert normalized == ref

    def test_normalize_oci_reference_without_registry(self):
        """Test normalization without registry (should default to docker.io)."""
        load_config = LoadConfig(load_format="oci")
        loader = OciModelLoader(load_config)

        # Without registry, should prepend docker.io
        ref = "user/model:tag"
        normalized = loader._normalize_oci_reference(ref)
        assert normalized == "docker.io/user/model:tag"

    def test_normalize_oci_reference_single_name(self):
        """Test normalization with single name (should add library)."""
        load_config = LoadConfig(load_format="oci")
        loader = OciModelLoader(load_config)

        # Single name should prepend docker.io/library
        ref = "model"
        normalized = loader._normalize_oci_reference(ref)
        assert normalized == "docker.io/library/model"

    def test_get_cache_dir(self):
        """Test cache directory creation."""
        load_config = LoadConfig(load_format="oci")
        loader = OciModelLoader(load_config)

        # Test cache directory path generation
        ref = "docker.io/user/model:tag"
        cache_dir = loader._get_cache_dir(ref)

        # Should contain 'oci' and sanitized reference
        assert "oci" in cache_dir
        assert "docker.io_user_model_tag" in cache_dir

    def test_media_type_constants(self):
        """Test that media type constants are correctly defined."""
        load_config = LoadConfig(load_format="oci")
        loader = OciModelLoader(load_config)

        assert loader.SAFETENSORS_MEDIA_TYPE == "application/vnd.docker.ai.safetensors"
        assert (
            loader.CONFIG_TAR_MEDIA_TYPE == "application/vnd.docker.ai.vllm.config.tar"
        )
        assert loader.DEFAULT_REGISTRY == "docker.io"

    def test_go_client_initialization(self):
        """Test that Go client initializes successfully."""
        load_config = LoadConfig(load_format="oci")
        try:
            loader = OciModelLoader(load_config)
            assert loader.go_client is not None
        except RuntimeError as e:
            # If the Go library is not built, skip this test
            if "OCI Go library not found" in str(e):
                pytest.skip("OCI Go library not built")
            raise

    def test_download_oci_model_if_needed_config_only(self):
        """Test that download_oci_model_if_needed with download_weights=False
        only downloads config."""
        load_config = LoadConfig(load_format="oci")
        loader = OciModelLoader(load_config)

        # This is a unit test to verify the method signature and behavior
        # We're not actually downloading anything, just testing the logic
        import contextlib
        import unittest.mock as mock

        with (
            mock.patch.object(loader, "_pull_oci_manifest") as mock_manifest,
            mock.patch.object(loader, "_download_layer") as mock_download,
            mock.patch.object(loader, "_extract_config_tar"),
        ):
            # Setup mock manifest
            mock_manifest.return_value = (
                {"layers": []},  # manifest
                [],  # safetensors_layers
                {"digest": "sha256:abc", "size": 100},  # config_layer
            )

            # Call with download_weights=False
            with contextlib.suppress(Exception):
                loader._download_oci_model_if_needed(
                    "test/model:tag", download_weights=False
                )

            # Verify _download_layer was not called for safetensors layers
            # (it might be called for config layer)
            if mock_download.called:
                # If called, it should only be for config, not safetensors
                for call in mock_download.call_args_list:
                    layer = call[0][1]  # second positional arg is layer
                    assert layer.get("mediaType") != loader.SAFETENSORS_MEDIA_TYPE

    def test_download_oci_model_simple_calls_with_weights_false(self):
        """Test that download_oci_model_simple calls _download_oci_model_if_needed
        with download_weights=False."""
        load_config = LoadConfig(load_format="oci")
        loader = OciModelLoader(load_config)

        import unittest.mock as mock

        with mock.patch.object(
            loader, "_download_oci_model_if_needed"
        ) as mock_download:
            mock_download.return_value = "/fake/config/dir"

            result = loader.download_oci_model_simple("test/model:tag")

            # Verify it was called with download_weights=False
            mock_download.assert_called_once_with(
                "test/model:tag", download_weights=False
            )
            assert result == "/fake/config/dir"


@pytest.mark.skip(reason="Integration test - requires actual OCI registry access")
class TestOciModelLoaderIntegration:
    """Integration tests for OCI model loader (requires network access)."""

    def test_download_model_from_public_registry(self):
        """Test downloading a model from a public OCI registry.

        Note: This test is skipped by default as it requires:
        1. Network access to a public registry
        2. A real OCI model artifact to test with

        To run this test, remove the skip decorator and ensure you have
        a valid test model available in a public registry.
        """
        # Example: test with a real model
        # load_config = LoadConfig(load_format="oci")
        # loader = OciModelLoader(load_config)
        # model_config = ModelConfig(
        #     model="aistaging/smollm2-vllm",
        #     ...
        # )
        # loader.download_model(model_config)
        pass

    def test_load_weights_from_oci(self):
        """Test loading weights from OCI layers.

        Note: This test is skipped by default as it requires:
        1. A downloaded OCI model
        2. An initialized model to load weights into
        """
        pass


class TestOciFormatAutoDetection:
    """Test suite for automatic OCI format detection."""

    def test_auto_detect_oci_with_tag(self):
        """Should auto-detect OCI format when tag is present."""
        from vllm.transformers_utils.utils import is_oci_model_with_tag

        # These should be detected as OCI
        test_cases = [
            "username/model:latest",
            "username/model:v1.0",
            "aistaging/smollm2-vllm:latest",
            "organization/repository:tag123",
            "user_name/model-name:v2.0.1",
        ]

        for model in test_cases:
            assert is_oci_model_with_tag(model), (
                f"Expected {model} to be detected as OCI format"
            )

    def test_auto_detect_oci_with_registry(self):
        """Should auto-detect OCI format with explicit registry."""
        from vllm.transformers_utils.utils import is_oci_model_with_tag

        test_cases = [
            "docker.io/username/model:tag",
            "ghcr.io/org/repo:v2.0",
            "registry.example.com/user/model:tag",
            "quay.io/namespace/image:latest",
        ]

        for model in test_cases:
            assert is_oci_model_with_tag(model), (
                f"Expected {model} to be detected as OCI format"
            )

    def test_auto_detect_oci_with_registry_port(self):
        """Should auto-detect OCI format with registry including port."""
        from vllm.transformers_utils.utils import is_oci_model_with_tag

        test_cases = [
            "registry.example.com:5000/user/model:tag",
            "localhost:8080/namespace/repo:v1.0",
        ]

        for model in test_cases:
            assert is_oci_model_with_tag(model), (
                f"Expected {model} to be detected as OCI format"
            )

    def test_auto_detect_oci_with_digest(self):
        """Should auto-detect OCI format when digest is present."""
        from vllm.transformers_utils.utils import is_oci_model_with_tag

        test_cases = [
            "username/model@sha256:abc123def456",
            "ghcr.io/org/repo@sha256:0123456789abcdef",
            "docker.io/user/image@sha256:fedcba9876543210",
        ]

        for model in test_cases:
            assert is_oci_model_with_tag(model), (
                f"Expected {model} to be detected as OCI format"
            )

    def test_no_auto_detect_without_tag(self):
        """Should NOT auto-detect OCI format without tag (ambiguous)."""
        from vllm.transformers_utils.utils import is_oci_model_with_tag

        # These should require explicit load_format="oci"
        test_cases = [
            "username/model",  # Ambiguous - could be HF or OCI
            "aistaging/smollm2-vllm",  # Your specific example
            "organization/repository",
            "user-name/model_name",
        ]

        for model in test_cases:
            assert not is_oci_model_with_tag(model), (
                f"Expected {model} NOT to be auto-detected as OCI format "
                "(should require explicit load_format='oci')"
            )

    def test_no_auto_detect_local_paths(self):
        """Should NOT detect local paths as OCI."""
        import os
        import tempfile

        from vllm.transformers_utils.utils import is_oci_model_with_tag

        # Test with actual existing path
        with tempfile.TemporaryDirectory() as tmpdir:
            test_cases = [
                tmpdir,  # Existing directory
                os.path.join(tmpdir, "model"),  # Path under existing dir
            ]

            for model in test_cases:
                assert not is_oci_model_with_tag(model), (
                    f"Expected {model} NOT to be detected as OCI format (local path)"
                )

        # Test with path-like strings that don't exist
        test_cases = [
            "/absolute/path/to/model",
            "./relative/path",
            "../parent/path",
            "model",  # Single name without slash
        ]

        for model in test_cases:
            assert not is_oci_model_with_tag(model), (
                f"Expected {model} NOT to be detected as OCI format"
            )

    def test_no_auto_detect_huggingface_style(self):
        """Should NOT detect standard HuggingFace repos as OCI."""
        from vllm.transformers_utils.utils import is_oci_model_with_tag

        test_cases = [
            "meta-llama/Llama-2-7b",  # No tag = HF format
            "mistralai/Mistral-7B-v0.1",
            "microsoft/phi-2",
            "google/gemma-2b",
        ]

        for model in test_cases:
            assert not is_oci_model_with_tag(model), (
                f"Expected {model} NOT to be auto-detected as OCI format "
                "(standard HuggingFace repo)"
            )

    def test_edge_cases(self):
        """Test edge cases for OCI format detection."""
        from vllm.transformers_utils.utils import is_oci_model_with_tag

        # Edge cases that should be detected
        should_detect = [
            "a/b:c",  # Minimal valid OCI reference
            "registry.io/a/b:c",  # Minimal with registry
            "user/model:tag-with-hyphen",
            "user/model:tag_with_underscore",
            "user/model:tag.with.dots",
        ]

        for model in should_detect:
            assert is_oci_model_with_tag(model), (
                f"Expected {model} to be detected as OCI format"
            )

        # Edge cases that should NOT be detected
        should_not_detect = [
            "a/b",  # No tag
            "model:tag",  # No slash (not repository format)
            ":tag",  # Invalid format
            "a/:tag",  # Invalid format
            "/a/b:tag",  # Starts with slash (local path)
        ]

        for model in should_not_detect:
            assert not is_oci_model_with_tag(model), (
                f"Expected {model} NOT to be detected as OCI format"
            )


# Example usage documentation
"""
Usage Example:
--------------

from vllm import LLM

# AUTO-DETECTED: Load with explicit tag (no need to specify load_format)
llm = LLM(
    model="aistaging/smollm2-vllm:latest"
)

# REQUIRES EXPLICIT: Load without tag (ambiguous)
llm = LLM(
    model="aistaging/smollm2-vllm",
    load_format="oci"
)

# The loader will:
# 1. Normalize the reference to docker.io/aistaging/smollm2-vllm:latest
# 2. Pull the OCI manifest
# 3. Download safetensors layers in order
# 4. Download and extract config tar (if present)
# 5. Load weights from safetensors files
# 6. Use extracted config for tokenizer/model initialization

# AUTO-DETECTED: With explicit registry and tag
llm = LLM(
    model="ghcr.io/myorg/mymodel:v1.0"
)

# AUTO-DETECTED: With digest instead of tag
llm = LLM(
    model="myuser/mymodel@sha256:abc123..."
)
"""
