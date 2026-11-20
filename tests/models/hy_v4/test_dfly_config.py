# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.platforms.rocm as rocm_platform
from vllm.models.hy_v4.dfly_config import HYV4DFlySpeculatorConfig
from vllm.platforms import current_platform
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.worker.gpu.spec_decode.speculator_config import (
    get_method_config_class,
)


def _spec(
    model_type: str = "hy_v4", architecture: str = "HYV4VLForConditionalGeneration"
):
    target = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type=model_type),
        architectures=[architecture],
    )
    parallel = SimpleNamespace(
        data_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    return SimpleNamespace(
        custom_method=None,
        method="dfly",
        target_model_config=target,
        target_parallel_config=parallel,
        kv_cache_dtype=None,
    )


@pytest.mark.parametrize("model_type", ["hy_v4", "hy_v4_flash"])
def test_hyv4_dfly_speculator_matches_platform(model_type):
    """ROCm targets use the AMD speculator, whose draft packs its top-k rows."""
    config_cls = get_method_config_class(_spec(model_type))
    vendor = "amd" if current_platform.is_rocm() else "nvidia"
    assert config_cls.speculator_class == (
        f"vllm.models.hy_v4.{vendor}.dfly_speculator.HYV4DFlySpeculator"
    )
    if current_platform.is_cuda_alike():
        assert resolve_obj_by_qualname(config_cls.speculator_class)


def test_hyv4_dfly_rocm_accepts_vl_targets(monkeypatch):
    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(rocm_platform, "on_gfx950", lambda: True)
    spec = _spec()
    HYV4DFlySpeculatorConfig.verify_target_parallel_config(spec)

    spec.target_parallel_config.pipeline_parallel_size = 2
    with pytest.raises(ValueError, match="pipeline_parallel_size=1"):
        HYV4DFlySpeculatorConfig.verify_target_parallel_config(spec)


def test_hyv4_dfly_rocm_rejects_text_only_targets(monkeypatch):
    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(rocm_platform, "on_gfx950", lambda: True)
    with pytest.raises(NotImplementedError, match="HYV4 VL or HYV4 Flash target"):
        HYV4DFlySpeculatorConfig.verify_target_parallel_config(
            _spec(architecture="HYV4ForCausalLM")
        )


def test_hyv4_dfly_rocm_requires_gfx950(monkeypatch):
    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(rocm_platform, "on_gfx950", lambda: False)
    with pytest.raises(NotImplementedError, match="gfx950"):
        HYV4DFlySpeculatorConfig.verify_target_parallel_config(_spec())


def test_hyv4_dfly_rocm_rejects_fp8_ds_mla_draft_kv(monkeypatch):
    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)
    spec = _spec()
    spec.kv_cache_dtype = "fp8_ds_mla"
    with pytest.raises(ValueError, match="BF16 draft MLA KV cache"):
        HYV4DFlySpeculatorConfig.apply_options(spec)
