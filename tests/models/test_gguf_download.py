# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.gguf_loader import (
    GGUFModelLoader,
    _add_gemma4_gguf_mappings,
    _gguf_name_with_suffix,
)
from vllm.model_executor.model_loader.weight_utils import download_gguf
from vllm.model_executor.models.gemma4 import (
    _load_gemma4_gguf_fused_moe_qweight_type,
)
from vllm.model_executor.models.gemma4_mm import _gemma4_patch_embed_weight_loader


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

            result = download_gguf("unsloth/Qwen3-0.6B-GGUF", "IQ1_S")

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

            result = download_gguf("unsloth/gpt-oss-120b-GGUF", "Q2_K")

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

            result = download_gguf("unsloth/gpt-oss-120b-GGUF", "Q2_K")

            assert result == f"{mock_folder}/Q2_K/model-Q2_K.gguf"

    @patch("vllm.model_executor.model_loader.weight_utils.download_weights_from_hf")
    @patch("glob.glob", return_value=[])
    def test_download_gguf_no_files_found(self, mock_glob, mock_download):
        """Test error when no GGUF files are found."""
        mock_folder = "/tmp/mock_cache"
        mock_download.return_value = mock_folder

        with pytest.raises(ValueError, match="Downloaded GGUF files not found"):
            download_gguf("unsloth/Qwen3-0.6B-GGUF", "IQ1_S")


class TestGGUFModelLoader:
    """Test GGUFModelLoader class methods."""

    @patch("os.path.isfile", return_value=True)
    def test_prepare_weights_local_file(self, mock_isfile):
        """Test _prepare_weights with local file."""
        load_config = LoadConfig(load_format="gguf")
        loader = GGUFModelLoader(load_config)

        # Create a simple mock ModelConfig with only the model attribute
        model_config = MagicMock()
        model_config.model = "/path/to/model.gguf"

        result = loader._prepare_weights(model_config)
        assert result == "/path/to/model.gguf"
        mock_isfile.assert_called_once_with("/path/to/model.gguf")

    @patch("vllm.model_executor.model_loader.gguf_loader.hf_hub_download")
    @patch("os.path.isfile", return_value=False)
    def test_prepare_weights_repo_filename(self, mock_isfile, mock_hf_download):
        """Test _prepare_weights with repo_id/filename.gguf format."""
        load_config = LoadConfig(load_format="gguf")
        loader = GGUFModelLoader(load_config)

        mock_hf_download.return_value = "/downloaded/model.gguf"

        model_config = MagicMock()
        model_config.model = "unsloth/Qwen3-0.6B-GGUF/model.gguf"
        model_config.revision = "abc123"

        result = loader._prepare_weights(model_config)
        assert result == "/downloaded/model.gguf"
        mock_hf_download.assert_called_once_with(
            repo_id="unsloth/Qwen3-0.6B-GGUF",
            filename="model.gguf",
            revision="abc123",
            cache_dir=None,
        )

    @patch("vllm.config.model.get_hf_image_processor_config", return_value=None)
    @patch("vllm.transformers_utils.config.file_or_path_exists", return_value=True)
    @patch("vllm.config.model.get_config")
    @patch("vllm.config.model.is_gguf", return_value=True)
    @patch("vllm.model_executor.model_loader.gguf_loader.download_gguf")
    @patch("os.path.isfile", return_value=False)
    def test_prepare_weights_repo_quant_type(
        self,
        mock_isfile,
        mock_download_gguf,
        mock_is_gguf,
        mock_get_config,
        mock_file_exists,
        mock_get_image_config,
    ):
        """Test _prepare_weights with repo_id:quant_type format."""
        mock_hf_config = MagicMock()
        mock_hf_config.architectures = ["Qwen3ForCausalLM"]

        class MockTextConfig:
            max_position_embeddings = 4096
            sliding_window = None
            model_type = "qwen3"
            num_attention_heads = 32

        mock_text_config = MockTextConfig()
        mock_hf_config.get_text_config.return_value = mock_text_config
        mock_hf_config.dtype = "bfloat16"
        mock_get_config.return_value = mock_hf_config

        load_config = LoadConfig(load_format="gguf")
        loader = GGUFModelLoader(load_config)

        mock_download_gguf.return_value = "/downloaded/model-IQ1_S.gguf"

        model_config = ModelConfig(
            model="unsloth/Qwen3-0.6B-GGUF:IQ1_S", tokenizer="Qwen/Qwen3-0.6B"
        )
        result = loader._prepare_weights(model_config)
        # The actual result will be the downloaded file path from mock
        assert result == "/downloaded/model-IQ1_S.gguf"
        mock_download_gguf.assert_called_once_with(
            "unsloth/Qwen3-0.6B-GGUF",
            "IQ1_S",
            cache_dir=None,
            revision=None,
            ignore_patterns=["original/**/*"],
        )

    @patch("vllm.config.model.get_hf_image_processor_config", return_value=None)
    @patch("vllm.config.model.get_config")
    @patch("vllm.config.model.is_gguf", return_value=False)
    @patch("vllm.transformers_utils.gguf_utils.check_gguf_file", return_value=False)
    @patch("os.path.isfile", return_value=False)
    def test_prepare_weights_invalid_format(
        self,
        mock_isfile,
        mock_check_gguf,
        mock_is_gguf,
        mock_get_config,
        mock_get_image_config,
    ):
        """Test _prepare_weights with invalid format."""
        mock_hf_config = MagicMock()
        mock_hf_config.architectures = ["Qwen3ForCausalLM"]

        class MockTextConfig:
            max_position_embeddings = 4096
            sliding_window = None
            model_type = "qwen3"
            num_attention_heads = 32

        mock_text_config = MockTextConfig()
        mock_hf_config.get_text_config.return_value = mock_text_config
        mock_hf_config.dtype = "bfloat16"
        mock_get_config.return_value = mock_hf_config

        load_config = LoadConfig(load_format="gguf")
        loader = GGUFModelLoader(load_config)

        # Create ModelConfig with a valid repo_id to avoid validation errors
        # Then test _prepare_weights with invalid format
        model_config = ModelConfig(model="unsloth/Qwen3-0.6B")
        # Manually set model to invalid format after creation
        model_config.model = "invalid-format"
        with pytest.raises(ValueError, match="Unrecognised GGUF reference"):
            loader._prepare_weights(model_config)

    def test_suffixless_gguf_name_has_no_trailing_dot(self):
        assert _gguf_name_with_suffix("blk.0.ssm_a", "") == "blk.0.ssm_a"
        assert _gguf_name_with_suffix("blk.0.attn_q", "weight") == "blk.0.attn_q.weight"

    def test_gemma4_manual_gguf_mappings(self):
        text_config = MagicMock()
        text_config.num_hidden_layers = 2
        vision_config = MagicMock()
        vision_config.num_hidden_layers = 3
        gguf_to_hf_name_map: dict[str, str] = {}

        _add_gemma4_gguf_mappings(
            gguf_to_hf_name_map,
            text_config,
            vision_config,
        )

        assert gguf_to_hf_name_map["blk.1.layer_output_scale.weight"] == (
            "model.language_model.layers.1.layer_scalar"
        )
        assert gguf_to_hf_name_map["blk.1.ffn_gate_inp.scale"] == (
            "model.language_model.layers.1.router.scale"
        )
        assert gguf_to_hf_name_map["blk.1.ffn_down_exps.scale"] == (
            "model.language_model.layers.1.router.per_expert_scale"
        )
        assert gguf_to_hf_name_map["blk.1.ffn_gate_up_exps.weight"] == (
            "model.language_model.layers.1.experts.gate_up_proj.weight"
        )
        assert gguf_to_hf_name_map["v.blk.2.attn_q.weight"] == (
            "model.vision_tower.encoder.layers.2.self_attn.q_proj.linear.weight"
        )
        assert gguf_to_hf_name_map["v.blk.2.ffn_down.weight"] == (
            "model.vision_tower.encoder.layers.2.mlp.down_proj.linear.weight"
        )
        assert gguf_to_hf_name_map["v.patch_embd.weight"] == (
            "model.vision_tower.patch_embedder.input_proj.weight"
        )
        assert gguf_to_hf_name_map["mm.input_projection.weight"] == (
            "model.embed_vision.embedding_projection.weight"
        )

    def test_gemma4_patch_embedder_weight_transform(self):
        param = torch.nn.Parameter(torch.empty(2, 60), requires_grad=False)
        loaded_weight = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)

        _gemma4_patch_embed_weight_loader(param, loaded_weight)

        assert torch.equal(param, loaded_weight.flatten(1))

    def test_gemma4_patch_embedder_loader_keeps_flat_weight(self):
        param = torch.nn.Parameter(torch.empty(2, 60), requires_grad=False)
        loaded_weight = torch.arange(2 * 60).reshape(2, 60)

        _gemma4_patch_embed_weight_loader(param, loaded_weight)

        assert torch.equal(param, loaded_weight)

    def test_gemma4_fused_moe_gguf_qweight_type_remap(self):
        w13_type = torch.nn.Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        w13_type.is_gguf_weight_type = True
        w13_type.weight_type = 0
        w2_type = torch.nn.Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        w2_type.is_gguf_weight_type = True
        w2_type.weight_type = 0
        params_dict = {
            "layers.0.moe.experts.w13_qweight_type": w13_type,
            "layers.0.moe.experts.w2_qweight_type": w2_type,
        }

        remapped = _load_gemma4_gguf_fused_moe_qweight_type(
            "layers.0.moe.gate_up_proj.qweight_type",
            torch.tensor(12),
            params_dict,
        )

        assert remapped == "layers.0.moe.experts.w13_qweight_type"
        assert w13_type.weight_type == 12
        assert w13_type.item() == 12

        remapped = _load_gemma4_gguf_fused_moe_qweight_type(
            "layers.0.moe.down_proj.qweight_type",
            torch.tensor(13),
            params_dict,
        )

        assert remapped == "layers.0.moe.experts.w2_qweight_type"
        assert w2_type.weight_type == 13
        assert w2_type.item() == 13
