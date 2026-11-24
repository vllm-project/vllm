# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.platforms.rocm as rocm_platform
from vllm.models.hy_v4.common.dfly_speculator import HYV4DFlySpeculator
from vllm.models.hy_v4.dfly_config import HYV4DFlySpeculatorConfig
from vllm.models.hy_v4_flash.dfly_config import HYV4FlashDFlySpeculatorConfig
from vllm.platforms import current_platform
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.worker.gpu.spec_decode.speculator_config import (
    get_method_config_class,
)

CONFIG_CLASSES = pytest.mark.parametrize(
    "config_cls",
    [HYV4DFlySpeculatorConfig, HYV4FlashDFlySpeculatorConfig],
    ids=["hy_v4", "hy_v4_flash"],
)


def _spec(
    model_type: str = "hy_v4",
    architecture: str = "HYV4VLForConditionalGeneration",
    kv_cache_dtype: str | None = None,
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
        kv_cache_dtype=kv_cache_dtype,
        attention_backend=None,
        enable_adaptive_verification=False,
        rejection_sample_method="standard",
    )


def _set_platform(monkeypatch, *, rocm: bool, gfx950: bool) -> None:
    monkeypatch.setattr(current_platform, "is_rocm", lambda: rocm)
    monkeypatch.setattr(rocm_platform, "on_gfx950", lambda: gfx950)


@pytest.mark.parametrize("model_type", ["hy_v4", "hy_v4_flash"])
def test_hyv4_dfly_speculator_matches_platform(model_type):
    """ROCm targets use the AMD speculator, whose draft packs its top-k rows."""
    config_cls = get_method_config_class(_spec(model_type))
    assert config_cls.speculator_class == "vllm.models.hy_v4.HYV4DFlySpeculator"
    if current_platform.is_cuda_alike():
        speculator = resolve_obj_by_qualname(config_cls.speculator_class)
        vendor = "amd" if current_platform.is_rocm() else "nvidia"
        assert speculator.__module__ == f"vllm.models.hy_v4.{vendor}.dfly_speculator"
        assert issubclass(speculator, HYV4DFlySpeculator)


@CONFIG_CLASSES
@pytest.mark.parametrize(
    "architecture", ["HYV4VLForConditionalGeneration", "HYV4ForCausalLM"]
)
def test_hyv4_dfly_rocm_accepts_vl_and_text_only_targets(
    monkeypatch, config_cls, architecture
):
    """The ROCm gate must not restrict DFly to VL targets."""
    _set_platform(monkeypatch, rocm=True, gfx950=True)
    config_cls.apply_options(_spec(architecture=architecture))


def test_hyv4_dfly_requires_single_pipeline_stage():
    spec = _spec()
    HYV4DFlySpeculatorConfig.verify_target_parallel_config(spec)
    spec.target_parallel_config.pipeline_parallel_size = 2
    with pytest.raises(ValueError, match="pipeline_parallel_size=1"):
        HYV4DFlySpeculatorConfig.verify_target_parallel_config(spec)


@CONFIG_CLASSES
def test_hyv4_dfly_rocm_requires_gfx950(monkeypatch, config_cls):
    _set_platform(monkeypatch, rocm=True, gfx950=False)
    with pytest.raises(NotImplementedError, match="gfx950"):
        config_cls.apply_options(_spec())


@CONFIG_CLASSES
def test_hyv4_dfly_gfx950_gate_is_rocm_only(monkeypatch, config_cls):
    _set_platform(monkeypatch, rocm=False, gfx950=False)
    config_cls.apply_options(_spec(kv_cache_dtype="fp8_ds_mla"))


@CONFIG_CLASSES
@pytest.mark.parametrize("kv_cache_dtype", ["fp8_ds_mla", "fp8"])
def test_hyv4_dfly_rocm_rejects_fp8_kv(monkeypatch, config_cls, kv_cache_dtype):
    _set_platform(monkeypatch, rocm=True, gfx950=True)
    with pytest.raises(ValueError, match="requires a BF16 KV cache"):
        config_cls.apply_options(_spec(kv_cache_dtype=kv_cache_dtype))
