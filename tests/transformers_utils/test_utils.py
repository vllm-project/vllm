# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path
from unittest.mock import patch

import pytest

from vllm.transformers_utils.gguf_utils import (
    is_gguf,
    is_remote_gguf,
    split_remote_gguf,
)
from vllm.transformers_utils.utils import (
    is_cloud_storage,
    is_gcs,
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
        # Valid quant types
        assert is_remote_gguf("unsloth/Qwen3-0.6B-GGUF:IQ1_S")
        assert is_remote_gguf("user/repo:Q2_K")
        assert is_remote_gguf("repo/model:Q4_K")
        assert is_remote_gguf("repo/model:Q8_0")

        # Invalid quant types should return False
        assert not is_remote_gguf("repo/model:quant")
        assert not is_remote_gguf("repo/model:INVALID")
        assert not is_remote_gguf("repo/model:invalid_type")

    def test_is_remote_gguf_without_colon(self):
        """Test is_remote_gguf without colon."""
        assert not is_remote_gguf("repo/model")
        assert not is_remote_gguf("unsloth/Qwen3-0.6B-GGUF")

    def test_is_remote_gguf_without_slash(self):
        """Test is_remote_gguf without slash."""
        assert not is_remote_gguf("model.gguf")
        # Even with valid quant_type, no slash means not remote GGUF
        assert not is_remote_gguf("model:IQ1_S")
        assert not is_remote_gguf("model:quant")

    def test_is_remote_gguf_local_path(self):
        """Test is_remote_gguf with local file path."""
        assert not is_remote_gguf("/path/to/model.gguf")
        assert not is_remote_gguf("./model.gguf")

    def test_is_remote_gguf_with_path_object(self):
        """Test is_remote_gguf with Path object."""
        assert is_remote_gguf(Path("unsloth/Qwen3-0.6B-GGUF:IQ1_S"))
        assert not is_remote_gguf(Path("repo/model"))

    def test_is_remote_gguf_with_http_https(self):
        """Test is_remote_gguf with HTTP/HTTPS URLs."""
        # HTTP/HTTPS URLs should return False even with valid quant_type
        assert not is_remote_gguf("http://example.com/repo/model:IQ1_S")
        assert not is_remote_gguf("https://huggingface.co/repo/model:Q2_K")
        assert not is_remote_gguf("http://repo/model:Q4_K")
        assert not is_remote_gguf("https://repo/model:Q8_0")

    def test_is_remote_gguf_with_cloud_storage(self):
        """Test is_remote_gguf with cloud storage paths."""
        # Cloud storage paths should return False even with valid quant_type
        assert not is_remote_gguf("s3://bucket/repo/model:IQ1_S")
        assert not is_remote_gguf("gs://bucket/repo/model:Q2_K")
        assert not is_remote_gguf("s3://repo/model:Q4_K")
        assert not is_remote_gguf("gs://repo/model:Q8_0")


class TestSplitRemoteGGUF:
    """Test split_remote_gguf utility function."""

    def test_split_remote_gguf_valid(self):
        """Test split_remote_gguf with valid repo_id:quant_type format."""
        repo_id, quant_type = split_remote_gguf("unsloth/Qwen3-0.6B-GGUF:IQ1_S")
        assert repo_id == "unsloth/Qwen3-0.6B-GGUF"
        assert quant_type == "IQ1_S"

        repo_id, quant_type = split_remote_gguf("repo/model:Q2_K")
        assert repo_id == "repo/model"
        assert quant_type == "Q2_K"

    def test_split_remote_gguf_with_path_object(self):
        """Test split_remote_gguf with Path object."""
        repo_id, quant_type = split_remote_gguf(Path("unsloth/Qwen3-0.6B-GGUF:IQ1_S"))
        assert repo_id == "unsloth/Qwen3-0.6B-GGUF"
        assert quant_type == "IQ1_S"

    def test_split_remote_gguf_invalid(self):
        """Test split_remote_gguf with invalid format."""
        # Invalid format (no colon) - is_remote_gguf returns False
        with pytest.raises(ValueError, match="Wrong GGUF model"):
            split_remote_gguf("repo/model")

        # Invalid quant type - is_remote_gguf returns False
        with pytest.raises(ValueError, match="Wrong GGUF model"):
            split_remote_gguf("repo/model:INVALID_TYPE")

        # HTTP URL - is_remote_gguf returns False
        with pytest.raises(ValueError, match="Wrong GGUF model"):
            split_remote_gguf("http://repo/model:IQ1_S")

        # Cloud storage - is_remote_gguf returns False
        with pytest.raises(ValueError, match="Wrong GGUF model"):
            split_remote_gguf("s3://bucket/repo/model:Q2_K")


class TestIsGGUF:
    """Test is_gguf utility function."""

    @patch("vllm.transformers_utils.gguf_utils.check_gguf_file", return_value=True)
    def test_is_gguf_with_local_file(self, mock_check_gguf):
        """Test is_gguf with local GGUF file."""
        assert is_gguf("/path/to/model.gguf")
        assert is_gguf("./model.gguf")

    def test_is_gguf_with_remote_gguf(self):
        """Test is_gguf with remote GGUF format."""
        # Valid remote GGUF format (repo_id:quant_type with valid quant_type)
        assert is_gguf("unsloth/Qwen3-0.6B-GGUF:IQ1_S")
        assert is_gguf("repo/model:Q2_K")
        assert is_gguf("repo/model:Q4_K")

        # Invalid quant_type should return False
        assert not is_gguf("repo/model:quant")
        assert not is_gguf("repo/model:INVALID")

    @patch("vllm.transformers_utils.gguf_utils.check_gguf_file", return_value=False)
    def test_is_gguf_false(self, mock_check_gguf):
        """Test is_gguf returns False for non-GGUF models."""
        assert not is_gguf("unsloth/Qwen3-0.6B")
        assert not is_gguf("repo/model")
        assert not is_gguf("model")

    def test_is_gguf_edge_cases(self):
        """Test is_gguf with edge cases."""
        # Empty string
        assert not is_gguf("")

        # Only colon, no slash (even with valid quant_type)
        assert not is_gguf("model:IQ1_S")

        # Only slash, no colon
        assert not is_gguf("repo/model")

        # HTTP/HTTPS URLs
        assert not is_gguf("http://repo/model:IQ1_S")
        assert not is_gguf("https://repo/model:Q2_K")

        # Cloud storage
        assert not is_gguf("s3://bucket/repo/model:IQ1_S")
        assert not is_gguf("gs://bucket/repo/model:Q2_K")
