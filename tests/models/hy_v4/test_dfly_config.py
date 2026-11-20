# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.models.hy_v4.dfly_config import HYV4DFlySpeculatorConfig
from vllm.models.hy_v4_flash.dfly_config import HYV4FlashDFlySpeculatorConfig
from vllm.platforms import current_platform


@pytest.mark.parametrize(
    "config_cls", [HYV4DFlySpeculatorConfig, HYV4FlashDFlySpeculatorConfig]
)
def test_hyv4_dfly_speculator_matches_platform(config_cls):
    """ROCm targets use the AMD speculator, whose draft packs its top-k rows."""
    vendor = "amd" if current_platform.is_rocm() else "nvidia"
    assert config_cls.speculator_class == (
        f"vllm.models.hy_v4.{vendor}.dfly_speculator.HYV4DFlySpeculator"
    )


@pytest.mark.parametrize("is_rocm", [False, True])
def test_hyv4_dfly_accepts_hyv4_targets(monkeypatch, is_rocm):
    monkeypatch.setattr(current_platform, "is_rocm", lambda: is_rocm)
    parallel = SimpleNamespace(
        data_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    spec = SimpleNamespace(target_parallel_config=parallel)
    HYV4DFlySpeculatorConfig.verify_target_parallel_config(spec)

    parallel.pipeline_parallel_size = 2
    with pytest.raises(ValueError, match="pipeline_parallel_size=1"):
        HYV4DFlySpeculatorConfig.verify_target_parallel_config(spec)
