# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

from vllm.transformers_utils.utils import (
    is_cloud_storage,
    is_gcs,
    is_gguf,
    is_remote_gguf,
    is_s3,
)


def test_is_gcs():
    assert is_gcs("gs://model-path")
    assert not is_gcs("s3://model-path/path-to-model")
    assert not is_gcs("/unix/local/path")
    assert not is_gcs("nfs://nfs-fqdn.local")


def test_is_s3():
    assert is_s3("s3://model-path/path-to-model")
    assert not is_s3("gs://model-path")
    assert not is_s3("/unix/local/path")
    assert not is_s3("nfs://nfs-fqdn.local")


def test_is_cloud_storage():
    assert is_cloud_storage("gs://model-path")
    assert is_cloud_storage("s3://model-path/path-to-model")
    assert not is_cloud_storage("/unix/local/path")
    assert not is_cloud_storage("nfs://nfs-fqdn.local")


class TestIsRemoteGGUF:
    """Test is_remote_gguf utility function."""

    def test_is_remote_gguf_with_colon_and_slash(self):
        """Test is_remote_gguf with repo_id:quant_type format."""
        assert is_remote_gguf("unsloth/Qwen3-0.6B-GGUF:IQ1_S")
        assert is_remote_gguf("repo/model:quant")
        assert is_remote_gguf("user/repo:Q2_K")

    def test_is_remote_gguf_without_colon(self):
        """Test is_remote_gguf without colon."""
        assert not is_remote_gguf("repo/model")
        assert not is_remote_gguf("unsloth/Qwen3-0.6B-GGUF")

    def test_is_remote_gguf_without_slash(self):
        """Test is_remote_gguf without slash."""
        assert not is_remote_gguf("model.gguf")
        assert not is_remote_gguf("model:quant")

    def test_is_remote_gguf_local_path(self):
        """Test is_remote_gguf with local file path."""
        assert not is_remote_gguf("/path/to/model.gguf")
        assert not is_remote_gguf("./model.gguf")


class TestIsGGUF:
    """Test is_gguf utility function."""

    def test_is_gguf_with_quant_type(self):
        """Test is_gguf with quant_type parameter."""
        # With quant_type set, should return True regardless of model name
        assert is_gguf("unsloth/Qwen3-0.6B-GGUF", "IQ1_S")
        assert is_gguf("any-model", "Q2_K")
        assert is_gguf("model", "Q4_K_M")

    @patch("vllm.transformers_utils.utils.check_gguf_file", return_value=True)
    def test_is_gguf_with_local_file(self, mock_check_gguf):
        """Test is_gguf with local GGUF file."""
        assert is_gguf("/path/to/model.gguf", None)
        assert is_gguf("./model.gguf", None)

    def test_is_gguf_with_remote_gguf(self):
        """Test is_gguf with remote GGUF format."""
        assert is_gguf("unsloth/Qwen3-0.6B-GGUF:IQ1_S", None)
        assert is_gguf("repo/model:quant", None)

    @patch("vllm.transformers_utils.utils.check_gguf_file", return_value=False)
    def test_is_gguf_false(self, mock_check_gguf):
        """Test is_gguf returns False for non-GGUF models."""
        assert not is_gguf("unsloth/Qwen3-0.6B", None)
        assert not is_gguf("repo/model", None)
        assert not is_gguf("model", None)

    def test_is_gguf_edge_cases(self):
        """Test is_gguf with edge cases."""
        # Empty string
        assert not is_gguf("", None)

        # Only colon, no slash
        assert not is_gguf("model:quant", None)

        # Only slash, no colon
        assert not is_gguf("repo/model", None)
