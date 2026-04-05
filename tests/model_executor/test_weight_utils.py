# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile

import huggingface_hub.constants
import pytest
from huggingface_hub.utils import LocalEntryNotFoundError

from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    enable_hf_transfer,
    maybe_remap_kv_scale_name,
)


def test_hf_transfer_auto_activation():
    if "HF_HUB_ENABLE_HF_TRANSFER" in os.environ:
        # in case it is already set, we can't test the auto activation
        pytest.skip("HF_HUB_ENABLE_HF_TRANSFER is set, can't test auto activation")
    enable_hf_transfer()
    try:
        # enable hf hub transfer if available
        import hf_transfer  # type: ignore # noqa

        HF_TRANSFER_ACTIVE = True
    except ImportError:
        HF_TRANSFER_ACTIVE = False
    assert huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER == HF_TRANSFER_ACTIVE


def test_download_weights_from_hf():
    with tempfile.TemporaryDirectory() as tmpdir:
        # assert LocalEntryNotFoundError error is thrown
        # if offline is set and model is not cached
        huggingface_hub.constants.HF_HUB_OFFLINE = True
        with pytest.raises(LocalEntryNotFoundError):
            download_weights_from_hf(
                "facebook/opt-125m",
                allow_patterns=["*.safetensors", "*.bin"],
                cache_dir=tmpdir,
            )

        # download the model
        huggingface_hub.constants.HF_HUB_OFFLINE = False
        download_weights_from_hf(
            "facebook/opt-125m",
            allow_patterns=["*.safetensors", "*.bin"],
            cache_dir=tmpdir,
        )

        # now it should work offline
        huggingface_hub.constants.HF_HUB_OFFLINE = True
        assert (
            download_weights_from_hf(
                "facebook/opt-125m",
                allow_patterns=["*.safetensors", "*.bin"],
                cache_dir=tmpdir,
            )
            is not None
        )


class TestDownloadWeightsPatternSelection:
    """Tests for pattern selection in download_weights_from_hf.

    Regression test for https://github.com/vllm-project/vllm/issues/38829
    Unsharded model.safetensors was not found because fnmatch does not
    match '*' across '/' in full repo-relative paths from HfFileSystem.ls().
    """

    def test_unsharded_safetensors_pattern_selected(self):
        """When HfFileSystem.ls() returns full paths containing a single
        model.safetensors, the *.safetensors pattern must still be selected
        over *.bin even if spurious .bin files exist."""
        from unittest.mock import MagicMock, patch

        file_list = [
            "test-org/test-model/model.safetensors",
            "test-org/test-model/training_args.bin",
            "test-org/test-model/config.json",
        ]

        mock_fs = MagicMock()
        mock_fs.ls.return_value = file_list

        with (
            patch(
                "vllm.model_executor.model_loader.weight_utils.HfFileSystem",
                return_value=mock_fs,
            ),
            patch(
                "vllm.model_executor.model_loader.weight_utils.snapshot_download",
                return_value="/tmp/fake_dir",
            ) as mock_download,
            patch(
                "pathlib.Path.glob",
                return_value=["/tmp/fake_dir/model.safetensors"],
            ),
        ):
            download_weights_from_hf(
                "test-org/test-model",
                allow_patterns=["*.safetensors", "*.bin"],
                cache_dir="/tmp/cache",
            )

            call_kwargs = mock_download.call_args
            assert call_kwargs is not None
            used_pattern = call_kwargs.kwargs.get("allow_patterns") or call_kwargs[
                1
            ].get("allow_patterns")
            assert used_pattern == "*.safetensors", (
                f"Expected *.safetensors but got {used_pattern}"
            )

    def test_sharded_safetensors_pattern_selected(self):
        """Sharded safetensors with full repo paths should also match."""
        from unittest.mock import MagicMock, patch

        file_list = [
            "test-org/test-model/model-00001-of-00002.safetensors",
            "test-org/test-model/model-00002-of-00002.safetensors",
            "test-org/test-model/model.safetensors.index.json",
            "test-org/test-model/config.json",
        ]

        mock_fs = MagicMock()
        mock_fs.ls.return_value = file_list

        with (
            patch(
                "vllm.model_executor.model_loader.weight_utils.HfFileSystem",
                return_value=mock_fs,
            ),
            patch(
                "vllm.model_executor.model_loader.weight_utils.hf_hub_download",
            ) as mock_hf_download,
            patch(
                "vllm.model_executor.model_loader.weight_utils.snapshot_download",
                return_value="/tmp/fake_dir",
            ),
            patch("pathlib.Path.glob", return_value=[]),
            patch("builtins.open", create=True),
            patch(
                "json.load",
                return_value={
                    "weight_map": {
                        "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                        "lm_head.weight": "model-00002-of-00002.safetensors",
                    }
                },
            ),
        ):
            mock_hf_download.return_value = "/tmp/fake_index.json"
            download_weights_from_hf(
                "test-org/test-model",
                allow_patterns=["*.safetensors", "*.bin"],
                cache_dir="/tmp/cache",
            )
            mock_hf_download.assert_called_once()

    def test_only_bin_files_selects_bin(self):
        """When repo has only .bin weight files, *.bin pattern should be used."""
        from unittest.mock import MagicMock, patch

        file_list = [
            "test-org/test-model/pytorch_model.bin",
            "test-org/test-model/config.json",
        ]

        mock_fs = MagicMock()
        mock_fs.ls.return_value = file_list

        with (
            patch(
                "vllm.model_executor.model_loader.weight_utils.HfFileSystem",
                return_value=mock_fs,
            ),
            patch(
                "vllm.model_executor.model_loader.weight_utils.snapshot_download",
                return_value="/tmp/fake_dir",
            ) as mock_download,
            patch(
                "pathlib.Path.glob",
                return_value=["/tmp/fake_dir/pytorch_model.bin"],
            ),
        ):
            download_weights_from_hf(
                "test-org/test-model",
                allow_patterns=["*.safetensors", "*.bin"],
                cache_dir="/tmp/cache",
            )

            call_kwargs = mock_download.call_args
            assert call_kwargs is not None
            used_pattern = call_kwargs.kwargs.get("allow_patterns") or call_kwargs[
                1
            ].get("allow_patterns")
            assert used_pattern == "*.bin", f"Expected *.bin but got {used_pattern}"


class TestMaybeRemapKvScaleName:
    """Tests for maybe_remap_kv_scale_name covering all checkpoint formats."""

    PARAMS_DICT = {
        "model.layers.0.self_attn.attn.k_scale": None,
        "model.layers.0.self_attn.attn.v_scale": None,
        "model.layers.0.self_attn.attn.q_scale": None,
        "model.layers.0.self_attn.qkv_proj.weight": None,
    }

    def test_qkv_proj_k_scale(self):
        """Qwen3-MoE / llm-compressor format: qkv_proj.k_scale -> attn.k_scale
        Regression test for https://github.com/vllm-project/vllm/issues/25047"""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.qkv_proj.k_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_qkv_proj_v_scale(self):
        """Qwen3-MoE / llm-compressor format: qkv_proj.v_scale -> attn.v_scale
        Regression test for https://github.com/vllm-project/vllm/issues/25047"""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.qkv_proj.v_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.v_scale"

    def test_modelopt_k_proj_k_scale(self):
        """ModelOpt format: k_proj.k_scale -> attn.k_scale"""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.k_proj.k_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_modelopt_v_proj_v_scale(self):
        """ModelOpt format: v_proj.v_scale -> attn.v_scale"""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.v_proj.v_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.v_scale"

    def test_deprecated_kv_scale(self):
        """Old format: kv_scale -> attn.k_scale (deprecated)"""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.kv_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_default_bare_k_scale(self):
        """Default format: .k_scale -> .attn.k_scale"""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.k_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_non_scale_name_unchanged(self):
        """Non-scale names should be returned unchanged."""
        name = "model.layers.0.self_attn.qkv_proj.weight"
        result = maybe_remap_kv_scale_name(name, self.PARAMS_DICT)
        assert result == name

    def test_nvfp4_modelopt_k_proj_k_scale(self):
        """ModelOpt NVFP4 format (e.g. nvidia/Qwen3-30B-A3B-NVFP4):
        k_proj.k_scale -> attn.k_scale.
        Validates that NVFP4 checkpoints are not broken by this change."""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.k_proj.k_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_nvfp4_modelopt_v_proj_v_scale(self):
        """ModelOpt NVFP4 format (e.g. nvidia/Qwen3-30B-A3B-NVFP4):
        v_proj.v_scale -> attn.v_scale.
        Validates that NVFP4 checkpoints are not broken by this change."""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.v_proj.v_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.v_scale"

    def test_qwen3_vl_moe_qkv_proj_k_scale(self):
        """Qwen3-VL-MoE uses the same fused qkv_proj naming as Qwen3-MoE.
        Regression test for qwen3_vl_moe.py fix (same bug as #25047)."""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.qkv_proj.k_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_qwen3_vl_moe_qkv_proj_v_scale(self):
        """Qwen3-VL-MoE uses the same fused qkv_proj naming as Qwen3-MoE.
        Regression test for qwen3_vl_moe.py fix (same bug as #25047)."""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.qkv_proj.v_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.v_scale"

    def test_nvfp4_weight_scale_not_remapped(self):
        """NVFP4 weight_scale should not be touched by remap (not a kv scale)."""
        name = "model.layers.0.self_attn.k_proj.weight_scale"
        result = maybe_remap_kv_scale_name(name, self.PARAMS_DICT)
        assert result == name

    def test_nvfp4_input_scale_not_remapped(self):
        """NVFP4 input_scale should not be touched by remap (not a kv scale)."""
        name = "model.layers.0.self_attn.k_proj.input_scale"
        result = maybe_remap_kv_scale_name(name, self.PARAMS_DICT)
        assert result == name

    def test_missing_target_returns_none(self):
        """If remapped name not in params_dict, return None."""
        empty_params: dict[str, None] = {}
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.qkv_proj.k_scale", empty_params
        )
        assert result is None


if __name__ == "__main__":
    test_hf_transfer_auto_activation()
    test_download_weights_from_hf()
