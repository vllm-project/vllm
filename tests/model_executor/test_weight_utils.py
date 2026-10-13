# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile
from unittest.mock import Mock

import huggingface_hub.constants
import pytest
import torch
from huggingface_hub.utils import LocalEntryNotFoundError

from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    drop_checkpoint_cache,
    initialize_single_dummy_weight,
    maybe_remap_kv_scale_name,
)
from vllm.platforms import current_platform


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
        """ModelOpt format: k_proj.k_scale -> attn.k_scale."""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.k_proj.k_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_modelopt_v_proj_v_scale(self):
        """ModelOpt format: v_proj.v_scale -> attn.v_scale."""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.v_proj.v_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.v_scale"

    def test_deprecated_kv_scale(self):
        """Old format: kv_scale -> attn.k_scale (deprecated)."""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.kv_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_default_bare_k_scale(self):
        """Default format: .k_scale -> .attn.k_scale."""
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


def test_checkpoint_weight_mapper_discards_serialized_g_idx():
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

    mapper = QuantizationConfig.get_checkpoint_weight_mapper()

    assert mapper._map_name("model.layers.0.g_idx") is None
    assert mapper._map_name("model.layers.0.qweight") == "model.layers.0.qweight"


class TestKvCacheScaleMapper:
    """The `WeightsMapper` returned by `get_cache_scale_mapper` replaces the
    per-model `maybe_remap_kv_scale_name` calls. It must remap the same set of
    checkpoint formats (the non-`params_dict`-dependent ones) and be idempotent
    so it composes safely with a model's own qkv/gate_up `hf_to_vllm_mapper`."""

    def _mapper(self):
        # `get_cache_scale_mapper` does not use `self`; call it on the base
        # class to get the default (non-config-specific) mapper.
        from vllm.model_executor.layers.quantization.base_config import (
            QuantizationConfig,
        )

        return QuantizationConfig.get_cache_scale_mapper()

    def _map(self, name: str) -> str | None:
        return self._mapper()._map_name(name)

    @pytest.mark.parametrize(
        "name,expected",
        [
            # Qwen3-MoE / llm-compressor fused qkv_proj
            (
                "model.layers.0.self_attn.qkv_proj.k_scale",
                "model.layers.0.self_attn.attn.k_scale",
            ),
            (
                "model.layers.0.self_attn.qkv_proj.v_scale",
                "model.layers.0.self_attn.attn.v_scale",
            ),
            # ModelOpt / NVFP4 k_proj/v_proj
            (
                "model.layers.0.self_attn.k_proj.k_scale",
                "model.layers.0.self_attn.attn.k_scale",
            ),
            (
                "model.layers.0.self_attn.v_proj.v_scale",
                "model.layers.0.self_attn.attn.v_scale",
            ),
            # deprecated fused kv_scale and bare scales
            (
                "model.layers.0.self_attn.kv_scale",
                "model.layers.0.self_attn.attn.k_scale",
            ),
            (
                "model.layers.0.self_attn.k_scale",
                "model.layers.0.self_attn.attn.k_scale",
            ),
            # NemotronH mixer
            (
                "model.layers.0.mixer.k_proj.k_scale",
                "model.layers.0.mixer.attn.k_scale",
            ),
            # already in vLLM form -> unchanged (idempotent)
            (
                "model.layers.0.self_attn.attn.k_scale",
                "model.layers.0.self_attn.attn.k_scale",
            ),
            # non-kv scales must not be touched
            (
                "model.layers.0.self_attn.k_proj.weight_scale",
                "model.layers.0.self_attn.k_proj.weight_scale",
            ),
            (
                "model.layers.0.self_attn.k_proj.input_scale",
                "model.layers.0.self_attn.k_proj.input_scale",
            ),
            # regular weights untouched
            (
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.q_proj.weight",
            ),
        ],
    )
    def test_remap(self, name, expected):
        assert self._map(name) == expected

    @pytest.mark.parametrize(
        "name",
        [
            "model.layers.0.self_attn.k_scale",
            "model.layers.0.self_attn.k_proj.k_scale",
            "model.layers.0.self_attn.qkv_proj.v_scale",
            "model.layers.0.mixer.k_proj.k_scale",
        ],
    )
    def test_idempotent(self, name):
        once = self._map(name)
        assert once is not None
        assert self._map(once) == once

    def test_composes_with_qkv_mapper(self):
        """Applied together with a model's qkv/gate_up mapper, the regex scale
        rules run before the substr rename, so scales are normalized to `.attn.`
        and regular projections are still fused correctly."""
        from vllm.model_executor.models.utils import WeightsMapper

        model_mapper = WeightsMapper(
            orig_to_new_substr={
                ".q_proj": ".qkv_proj.q",
                ".k_proj": ".qkv_proj.k",
                ".v_proj": ".qkv_proj.v",
            }
        )
        # AutoWeightsLoader does `mapper |= cache_scale_mapper`
        combined = model_mapper | self._mapper()

        assert (
            combined._map_name("model.layers.0.self_attn.q_proj.weight")
            == "model.layers.0.self_attn.qkv_proj.q.weight"
        )
        assert (
            combined._map_name("model.layers.0.self_attn.k_proj.k_scale")
            == "model.layers.0.self_attn.attn.k_scale"
        )
        assert (
            combined._map_name("model.layers.0.self_attn.k_scale")
            == "model.layers.0.self_attn.attn.k_scale"
        )


@pytest.mark.parametrize("fails", [False, True])
def test_drop_checkpoint_cache_releases_local_weight_files(
    tmp_path, monkeypatch, fails
):
    """Evict only weights; cache eviction errors must not prevent loading."""
    for name in ("weights.safetensors", "config.json"):
        (tmp_path / name).touch()
    fadvise = Mock(side_effect=OSError("cache eviction failed") if fails else None)
    monkeypatch.setattr("os.posix_fadvise", fadvise)
    drop_checkpoint_cache(str(tmp_path))
    fadvise.assert_called_once()


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_dummy_fp8_weight_is_not_degenerate(dtype):
    """+-1e-3 underflows in FP8, so 8-bit floats default to +-1."""
    param = torch.empty(4096, dtype=dtype)
    initialize_single_dummy_weight(param)
    values = param.float()
    assert values.isfinite().all()
    assert values.abs().max() <= 1.0
    assert (values != 0).float().mean() > 0.99


@pytest.mark.parametrize(
    ("dtype", "name"),
    [(torch.float8_e8m0fnu, "weight_scale_inv"), (torch.uint8, "w13_weight_scale")],
)
def test_dummy_e8m0_scale_is_fixed_exponent(dtype, name):
    """E8M0 scales, including uint8-stored ones, get a finite 2**-7."""
    param = torch.empty(64, dtype=dtype)
    initialize_single_dummy_weight(param, name=name)
    assert (param.view(torch.uint8) == 120).all()


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8, torch.float4_e2m1fn_x2])
def test_dummy_packed_weight_gets_random_bytes(dtype):
    """Packed weights are filled deterministically instead of left uninitialized."""
    # Some backends can't fill packed FP4 directly, so view zeroed bytes.
    params = [torch.zeros(4096, dtype=torch.uint8).view(dtype) for _ in range(2)]
    for param in params:
        initialize_single_dummy_weight(param, name="w13_weight")
    raw = params[0].view(torch.uint8)
    assert raw.unique().numel() > 250
    assert torch.equal(raw, params[1].view(torch.uint8))


@pytest.mark.skipif(current_platform.is_rocm(), reason="ROCm zeroes integer params")
def test_dummy_wide_integer_param_is_untouched():
    """Wider integer params may be indices, so random values are unsafe."""
    param = torch.arange(16, dtype=torch.int32)
    initialize_single_dummy_weight(param, name="g_idx")
    assert torch.equal(param, torch.arange(16, dtype=torch.int32))


if __name__ == "__main__":
    test_download_weights_from_hf()
