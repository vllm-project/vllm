# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.models.hy_v4.dfly_config import HYV4DFlySpeculatorConfig
from vllm.platforms import current_platform


@pytest.mark.parametrize("is_rocm", [False, True])
def test_hyv4_dfly_rejects_non_flash_targets_on_rocm(monkeypatch, is_rocm):
    """Only Flash targets swap in the ROCm packed-row DFly speculator."""
    monkeypatch.setattr(current_platform, "is_rocm", lambda: is_rocm)
    parallel = SimpleNamespace(
        data_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    spec = SimpleNamespace(target_parallel_config=parallel)

    if is_rocm:
        with pytest.raises(NotImplementedError, match="hy_v4_flash target"):
            HYV4DFlySpeculatorConfig.verify_target_parallel_config(spec)
    else:
        HYV4DFlySpeculatorConfig.verify_target_parallel_config(spec)
