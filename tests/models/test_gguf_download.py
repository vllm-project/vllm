# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.gguf_loader import GGUFModelLoader
from vllm.model_executor.model_loader.weight_utils import download_gguf
from vllm.model_executor.models.qwen3_5 import (
    _load_gguf_tuple_shard,
    _maybe_unsqueeze_single_output_weight,
)


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
                    "*.IQ1_S.gguf",
                    "*-IQ1_S-*.gguf",
                    "*.IQ1_S-*.gguf",
                    "*/*-IQ1_S.gguf",
                    "*/*.IQ1_S.gguf",
                    "*/*-IQ1_S-*.gguf",
                    "*/*.IQ1_S-*.gguf",
                ],
                revision=None,
                ignore_patterns=None,
            )

            # Verify result is the file path, not folder
            assert result == f"{mock_folder}/model-IQ1_S.gguf"

    @patch("vllm.model_executor.model_loader.weight_utils.download_weights_from_hf")
    def test_download_gguf_dot_quant_suffix(self, mock_download):
        """Test GGUF repos whose quant suffix is separated by a dot."""
        mock_folder = "/tmp/mock_cache"
        mock_download.return_value = mock_folder

        with patch("glob.glob") as mock_glob:
            mock_glob.side_effect = lambda pattern, **kwargs: (
                [f"{mock_folder}/model.Q4_K_M.gguf"] if ".Q4_K_M" in pattern else []
            )

            result = download_gguf("hesamation/model-GGUF", "Q4_K_M")

            assert result == f"{mock_folder}/model.Q4_K_M.gguf"

    @patch("vllm.model_executor.model_loader.weight_utils.download_weights_from_hf")
    @patch(
        "vllm.model_executor.model_loader.weight_utils.maybe_download_from_modelscope"
    )
    def test_download_gguf_modelscope_uses_selected_patterns(
        self,
        mock_modelscope_download,
        mock_hf_download,
        monkeypatch,
    ):
        """Test ModelScope downloads only the selected GGUF quant files."""
        monkeypatch.setenv("VLLM_USE_MODELSCOPE", "True")
        mock_folder = "/tmp/mock_cache"
        mock_modelscope_download.return_value = mock_folder

        with patch("glob.glob") as mock_glob:
            mock_glob.side_effect = lambda pattern, **kwargs: (
                [f"{mock_folder}/model.Q4_K_M.gguf"] if ".Q4_K_M" in pattern else []
            )

            result = download_gguf(
                "hesamation/model-GGUF",
                "Q4_K_M",
                cache_dir="/cache",
                revision="master",
                ignore_patterns=["original/**/*"],
            )

            assert result == f"{mock_folder}/model.Q4_K_M.gguf"
            mock_modelscope_download.assert_called_once_with(
                model="hesamation/model-GGUF",
                revision="master",
                download_dir="/cache",
                ignore_patterns=["original/**/*"],
                allow_patterns=[
                    "*-Q4_K_M.gguf",
                    "*.Q4_K_M.gguf",
                    "*-Q4_K_M-*.gguf",
                    "*.Q4_K_M-*.gguf",
                    "*/*-Q4_K_M.gguf",
                    "*/*.Q4_K_M.gguf",
                    "*/*-Q4_K_M-*.gguf",
                    "*/*.Q4_K_M-*.gguf",
                ],
            )
            mock_hf_download.assert_not_called()

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

    @patch("vllm.model_executor.model_loader.gguf_loader.AutoModelForImageTextToText")
    def test_qwen35_moe_gguf_mapping(self, mock_auto_model):
        """Test Qwen3.5-MoE maps GGUF names to loadable expert tensors."""

        class DummyTensorNameMap:
            def get_name(self, name: str):
                if name == "model.norm":
                    return "output_norm"
                return None

        class DummyModel:
            _checkpoint_conversion_mapping = None

            def state_dict(self):
                return {
                    "model.language_model.layers.0.linear_attn.dt_bias": torch.empty(
                        32,
                        device="meta",
                    ),
                    "model.language_model.layers.0.mlp.experts.gate_up_proj": (
                        torch.empty(256, 1024, 2048, device="meta")
                    ),
                }

        mock_auto_model.from_config.return_value = DummyModel()

        hf_config = MagicMock()
        hf_config.model_type = "qwen3_5_moe"
        hf_config.vision_config = SimpleNamespace(num_hidden_layers=0)
        hf_config.get_text_config.return_value = SimpleNamespace(num_hidden_layers=1)

        model_config = MagicMock()
        model_config.hf_config = hf_config
        model_config.trust_remote_code = False
        model_config.multimodal_config = SimpleNamespace(language_model_only=False)

        load_config = LoadConfig(load_format="gguf")
        loader = GGUFModelLoader(load_config)

        with patch(
            "vllm.model_executor.model_loader.gguf_loader.gguf.get_tensor_name_map",
            return_value=DummyTensorNameMap(),
        ):
            mapping = loader._get_gguf_weights_map(model_config)

        assert mapping["blk.0.ffn_gate_exps.weight"] == (
            "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"
        )
        assert mapping["blk.0.ffn_up_exps.weight"] == (
            "model.language_model.layers.0.mlp.experts.0.up_proj.weight"
        )
        assert mapping["blk.0.ffn_down_exps.weight"] == (
            "model.language_model.layers.0.mlp.experts.0.down_proj.weight"
        )
        assert mapping["blk.0.ssm_dt.bias"] == (
            "model.language_model.layers.0.linear_attn.dt_bias"
        )

    @patch("vllm.model_executor.model_loader.gguf_loader.AutoModelForCausalLM")
    def test_qwen35_moe_gguf_language_model_only_mapping(self, mock_auto_model):
        """Test Qwen3.5-MoE text-only GGUF names avoid multimodal prefixes."""

        class DummyTensorNameMap:
            def get_name(self, name: str):
                if name == "model.norm":
                    return "output_norm"
                return None

        class DummyModel:
            _checkpoint_conversion_mapping = None

            def state_dict(self):
                return {
                    "model.layers.0.linear_attn.dt_bias": torch.empty(
                        32,
                        device="meta",
                    ),
                    "model.layers.0.mlp.experts.gate_up_proj": torch.empty(
                        256,
                        1024,
                        2048,
                        device="meta",
                    ),
                    "model.norm.weight": torch.empty(2048, device="meta"),
                }

        mock_auto_model.from_config.return_value = DummyModel()

        hf_config = MagicMock()
        hf_config.model_type = "qwen3_5_moe"
        hf_config.vision_config = SimpleNamespace(depth=27)
        hf_config.get_text_config.return_value = SimpleNamespace(num_hidden_layers=1)

        model_config = MagicMock()
        model_config.hf_config = hf_config
        model_config.trust_remote_code = False
        model_config.multimodal_config = SimpleNamespace(language_model_only=True)

        load_config = LoadConfig(load_format="gguf")
        loader = GGUFModelLoader(load_config)

        with patch(
            "vllm.model_executor.model_loader.gguf_loader.gguf.get_tensor_name_map",
            return_value=DummyTensorNameMap(),
        ):
            mapping = loader._get_gguf_weights_map(model_config)

        assert mapping["blk.0.ffn_gate_exps.weight"] == (
            "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"
        )
        assert mapping["blk.0.ssm_dt.bias"] == (
            "model.language_model.layers.0.linear_attn.dt_bias"
        )
        assert mapping["output_norm.weight"] == "model.language_model.norm.weight"

    @patch("vllm.model_executor.model_loader.gguf_loader.detect_gguf_multimodal")
    @patch(
        "vllm.model_executor.model_loader.gguf_loader.get_gguf_weight_type_map",
        return_value={"model.layers.0.mlp.experts.0.gate_proj.weight": "Q4_K"},
    )
    def test_language_model_only_weight_types_skip_mmproj(
        self, mock_get_weight_type_map, mock_detect_mm
    ):
        """Test text-only multimodal configs do not require an mm_proj file."""

        hf_config = MagicMock()
        hf_config.vision_config = SimpleNamespace(depth=27)

        model_config = MagicMock()
        model_config.hf_config = hf_config
        model_config.multimodal_config = SimpleNamespace(language_model_only=True)

        load_config = LoadConfig(load_format="gguf")
        loader = GGUFModelLoader(load_config)

        weight_type_map = loader._get_gguf_weight_type(
            model_config,
            "/tmp/model.gguf",
            {
                "blk.0.ffn_gate_exps.weight": (
                    "model.layers.0.mlp.experts.0.gate_proj.weight"
                )
            },
        )

        assert weight_type_map == {
            "model.layers.0.mlp.experts.0.gate_proj.weight": "Q4_K"
        }
        mock_get_weight_type_map.assert_called_once()
        mock_detect_mm.assert_not_called()

    @patch("vllm.model_executor.model_loader.gguf_loader.detect_gguf_multimodal")
    @patch(
        "vllm.model_executor.model_loader.gguf_loader.gguf_quant_weights_iterator",
        return_value=iter([("model.layers.0.linear_attn.dt_bias", torch.empty(1))]),
    )
    def test_language_model_only_weights_iterator_skips_mmproj(
        self, mock_weight_iter, mock_detect_mm
    ):
        """Test text-only multimodal configs load only the main GGUF file."""

        hf_config = MagicMock()
        hf_config.vision_config = SimpleNamespace(depth=27)

        model_config = MagicMock()
        model_config.hf_config = hf_config
        model_config.multimodal_config = SimpleNamespace(language_model_only=True)

        load_config = LoadConfig(load_format="gguf")
        loader = GGUFModelLoader(load_config)

        weights = list(
            loader._get_weights_iterator(
                model_config,
                "/tmp/model.gguf",
                {"blk.0.ssm_dt.bias": "model.layers.0.linear_attn.dt_bias"},
            )
        )

        assert weights[0][0] == "model.layers.0.linear_attn.dt_bias"
        mock_weight_iter.assert_called_once()
        mock_detect_mm.assert_not_called()

    def test_qwen35_gguf_tuple_shard_weight_splits(self):
        """Test Qwen3.5 GGUF qkv tuple shards are loaded one shard at a time."""

        calls = []

        class DummyLoader:
            output_sizes = [2, 3, 4]

            def weight_loader(self, param, loaded_weight, shard_id):
                calls.append((loaded_weight.clone(), shard_id))

        param = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        param.is_gguf_weight = True
        param.output_dim = 0
        loaded_weight = torch.arange(18).view(9, 2)

        handled = _load_gguf_tuple_shard(
            param,
            loaded_weight,
            (0, 1, 2),
            DummyLoader().weight_loader,
        )

        assert handled
        assert [shard_id for _, shard_id in calls] == [0, 1, 2]
        assert [weight.shape[0] for weight, _ in calls] == [2, 3, 4]
        assert torch.equal(calls[1][0], loaded_weight[2:5])

    def test_qwen35_gguf_tuple_shard_weight_type_reuses_scalar(self):
        """Test tuple qweight_type shards reuse the same GGUF type scalar."""

        calls = []

        class DummyLoader:
            output_sizes = [2, 3, 4]

            def weight_loader(self, param, loaded_weight, shard_id):
                calls.append((loaded_weight, shard_id))

        param = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        param.is_gguf_weight_type = True
        loaded_weight = torch.tensor(12)

        handled = _load_gguf_tuple_shard(
            param,
            loaded_weight,
            (0, 1, 2),
            DummyLoader().weight_loader,
        )

        assert handled
        assert calls == [(loaded_weight, 0), (loaded_weight, 1), (loaded_weight, 2)]

    def test_qwen35_gguf_single_output_weight_unsqueeze(self):
        """Test single-output GGUF tensors match vLLM's 2D gate params."""

        param = torch.nn.Parameter(torch.empty(1, 2048), requires_grad=False)
        loaded_weight = torch.empty(2048)

        reshaped = _maybe_unsqueeze_single_output_weight(param, loaded_weight)

        assert reshaped.shape == torch.Size([1, 2048])
        assert reshaped.data_ptr() == loaded_weight.data_ptr()

    def test_qwen35_gguf_conv_weight_unsqueeze(self):
        """Test GGUF conv tensors match vLLM's singleton channel dim."""

        param = torch.nn.Parameter(torch.empty(2048, 1, 4), requires_grad=False)
        loaded_weight = torch.empty(2048, 4)

        reshaped = _maybe_unsqueeze_single_output_weight(param, loaded_weight)

        assert reshaped.shape == torch.Size([2048, 1, 4])
        assert reshaped.data_ptr() == loaded_weight.data_ptr()

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

        # Create a simple mock ModelConfig with only the model attribute
        model_config = MagicMock()
        model_config.model = "unsloth/Qwen3-0.6B-GGUF/model.gguf"

        result = loader._prepare_weights(model_config)
        assert result == "/downloaded/model.gguf"
        mock_hf_download.assert_called_once_with(
            repo_id="unsloth/Qwen3-0.6B-GGUF", filename="model.gguf"
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
