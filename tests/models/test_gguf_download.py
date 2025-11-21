# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.gguf_loader import GGUFModelLoader


class TestGGUFDownload:
    """Test GGUF model downloading functionality."""

    @patch("vllm.model_executor.model_loader.weight_utils.download_weights_from_hf")
    def test_download_gguf_single_file(self, mock_download):
        """Test downloading a single GGUF file."""
        # Setup mock
        mock_folder = "/tmp/mock_cache"
        mock_download.return_value = mock_folder

        # Mock glob to return a single file
        with patch("glob.glob") as mock_glob:
            mock_glob.side_effect = lambda pattern, **kwargs: (
                [f"{mock_folder}/model-IQ1_S.gguf"] if "IQ1_S" in pattern else []
            )

            result = GGUFModelLoader.download_gguf("unsloth/Qwen3-0.6B-GGUF", "IQ1_S")

            # Verify download_weights_from_hf was called with correct patterns
            mock_download.assert_called_once_with(
                model_name_or_path="unsloth/Qwen3-0.6B-GGUF",
                cache_dir=None,
                allow_patterns=[
                    "*-IQ1_S.gguf",
                    "*-IQ1_S-*.gguf",
                    "*/*-IQ1_S.gguf",
                    "*/*-IQ1_S-*.gguf",
                ],
                revision=None,
                ignore_patterns=None,
            )

            # Verify result is the file path, not folder
            assert result == f"{mock_folder}/model-IQ1_S.gguf"

    @patch("vllm.model_executor.model_loader.weight_utils.download_weights_from_hf")
    def test_download_gguf_sharded_files(self, mock_download):
        """Test downloading sharded GGUF files."""
        mock_folder = "/tmp/mock_cache"
        mock_download.return_value = mock_folder

        # Mock glob to return sharded files
        with patch("glob.glob") as mock_glob:
            mock_glob.side_effect = lambda pattern, **kwargs: (
                [
                    f"{mock_folder}/model-Q2_K-00001-of-00002.gguf",
                    f"{mock_folder}/model-Q2_K-00002-of-00002.gguf",
                ]
                if "Q2_K" in pattern
                else []
            )

            result = GGUFModelLoader.download_gguf("unsloth/gpt-oss-120b-GGUF", "Q2_K")

            # Should return the first file after sorting
            assert result == f"{mock_folder}/model-Q2_K-00001-of-00002.gguf"

    @patch("vllm.model_executor.model_loader.weight_utils.download_weights_from_hf")
    def test_download_gguf_subdir(self, mock_download):
        """Test downloading GGUF files from subdirectory."""
        mock_folder = "/tmp/mock_cache"
        mock_download.return_value = mock_folder

        with patch("glob.glob") as mock_glob:
            mock_glob.side_effect = lambda pattern, **kwargs: (
                [f"{mock_folder}/Q2_K/model-Q2_K.gguf"]
                if "Q2_K" in pattern or "**/*.gguf" in pattern
                else []
            )

            result = GGUFModelLoader.download_gguf("unsloth/gpt-oss-120b-GGUF", "Q2_K")

            assert result == f"{mock_folder}/Q2_K/model-Q2_K.gguf"

    @patch("vllm.model_executor.model_loader.weight_utils.download_weights_from_hf")
    def test_download_gguf_fallback_search(self, mock_download):
        """Test fallback search when pattern matching fails."""
        mock_folder = "/tmp/mock_cache"
        mock_download.return_value = mock_folder

        with patch("glob.glob") as mock_glob:
            # First glob calls return empty (pattern match fails)
            # Then recursive search finds the file
            def glob_side_effect(pattern, **kwargs):
                if "**/*.gguf" in pattern:
                    # Fallback recursive search
                    return [f"{mock_folder}/custom-name-IQ1_S.gguf"]
                return []

            mock_glob.side_effect = glob_side_effect

            result = GGUFModelLoader.download_gguf("unsloth/Qwen3-0.6B-GGUF", "IQ1_S")

            assert result == f"{mock_folder}/custom-name-IQ1_S.gguf"

    @patch("vllm.model_executor.model_loader.weight_utils.download_weights_from_hf")
    @patch("glob.glob", return_value=[])
    def test_download_gguf_no_files_found(self, mock_glob, mock_download):
        """Test error when no GGUF files are found."""
        mock_folder = "/tmp/mock_cache"
        mock_download.return_value = mock_folder

        with pytest.raises(ValueError, match="Downloaded GGUF files not found"):
            GGUFModelLoader.download_gguf("unsloth/Qwen3-0.6B-GGUF", "IQ1_S")


class TestGGUFModelLoader:
    """Test GGUFModelLoader class methods."""

    @patch("vllm.config.model.get_config")
    @patch("vllm.config.model.is_gguf")
    @patch(
        "vllm.model_executor.models.registry.ModelRegistry.is_text_generation_model",
        return_value=True,
    )
    def test_get_model_path_for_download_with_quant_type(
        self, mock_is_text_gen, mock_is_gguf, mock_get_config
    ):
        """Test _get_model_path_for_download with quant_type."""
        mock_is_gguf.return_value = True
        mock_hf_config = MagicMock()
        mock_hf_config.architectures = ["Qwen3ForCausalLM"]

        # Setup get_text_config to return a config with max_position_embeddings
        # Use a simple object with only the attributes we need
        class MockTextConfig:
            max_position_embeddings = 4096
            sliding_window = None
            # Add model_type to avoid errors in _get_and_verify_max_len
            model_type = "qwen3"
            # Add num_attention_heads for get_hf_text_config assertion
            num_attention_heads = 32

        mock_text_config = MockTextConfig()
        mock_hf_config.get_text_config.return_value = mock_text_config
        mock_hf_config.dtype = "bfloat16"
        mock_get_config.return_value = mock_hf_config

        load_config = LoadConfig(load_format="gguf")
        loader = GGUFModelLoader(load_config)

        model_config = ModelConfig(
            model="unsloth/Qwen3-0.6B-GGUF",
            gguf_quant_type="IQ1_S",
            tokenizer="Qwen/Qwen3-0.6B",
        )

        path = loader._get_model_path_for_download(model_config)
        assert path == "unsloth/Qwen3-0.6B-GGUF:IQ1_S"

    def test_get_model_path_for_download_without_quant_type(self):
        """Test _get_model_path_for_download without quant_type."""
        load_config = LoadConfig(load_format="gguf")
        loader = GGUFModelLoader(load_config)

        model_config = ModelConfig(
            model="unsloth/Qwen3-0.6B-GGUF",
            gguf_quant_type=None,
        )

        path = loader._get_model_path_for_download(model_config)
        assert path == "unsloth/Qwen3-0.6B-GGUF"

    @patch("os.path.isfile", return_value=True)
    def test_prepare_weights_local_file(self, mock_isfile):
        """Test _prepare_weights with local file."""
        load_config = LoadConfig(load_format="gguf")
        loader = GGUFModelLoader(load_config)

        result = loader._prepare_weights("/path/to/model.gguf")
        assert result == "/path/to/model.gguf"
        mock_isfile.assert_called_once_with("/path/to/model.gguf")

    @patch("vllm.model_executor.model_loader.gguf_loader.hf_hub_download")
    @patch("os.path.isfile", return_value=False)
    def test_prepare_weights_https_url(self, mock_isfile, mock_hf_download):
        """Test _prepare_weights with HTTPS URL."""
        load_config = LoadConfig(load_format="gguf")
        loader = GGUFModelLoader(load_config)

        mock_hf_download.return_value = "/downloaded/model.gguf"

        result = loader._prepare_weights("https://huggingface.co/model.gguf")
        assert result == "/downloaded/model.gguf"
        mock_hf_download.assert_called_once_with(
            url="https://huggingface.co/model.gguf"
        )

    @patch("vllm.model_executor.model_loader.gguf_loader.hf_hub_download")
    @patch("os.path.isfile", return_value=False)
    def test_prepare_weights_repo_filename(self, mock_isfile, mock_hf_download):
        """Test _prepare_weights with repo_id/filename.gguf format."""
        load_config = LoadConfig(load_format="gguf")
        loader = GGUFModelLoader(load_config)

        mock_hf_download.return_value = "/downloaded/model.gguf"

        result = loader._prepare_weights("unsloth/Qwen3-0.6B-GGUF/model.gguf")
        assert result == "/downloaded/model.gguf"
        mock_hf_download.assert_called_once_with(
            repo_id="unsloth/Qwen3-0.6B-GGUF", filename="model.gguf"
        )

    @patch.object(GGUFModelLoader, "download_gguf")
    @patch("os.path.isfile", return_value=False)
    def test_prepare_weights_repo_quant_type(self, mock_isfile, mock_download_gguf):
        """Test _prepare_weights with repo_id:quant_type format."""
        load_config = LoadConfig(load_format="gguf")
        loader = GGUFModelLoader(load_config)

        mock_download_gguf.return_value = "/downloaded/model-IQ1_S.gguf"

        result = loader._prepare_weights("unsloth/Qwen3-0.6B-GGUF:IQ1_S")
        assert result == "/downloaded/model-IQ1_S.gguf"
        mock_download_gguf.assert_called_once_with("unsloth/Qwen3-0.6B-GGUF", "IQ1_S")

    @patch("os.path.isfile", return_value=False)
    def test_prepare_weights_invalid_format(self, mock_isfile):
        """Test _prepare_weights with invalid format."""
        load_config = LoadConfig(load_format="gguf")
        loader = GGUFModelLoader(load_config)

        with pytest.raises(ValueError, match="Unrecognised GGUF reference"):
            loader._prepare_weights("invalid-format")
