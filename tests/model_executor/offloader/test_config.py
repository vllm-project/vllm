# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from vllm.config import (
    DeviceConfig,
    OffloadConfig,
    ParallelConfig,
    PrefetchOffloadConfig,
    VllmConfig,
)
from vllm.config.compilation import PassConfig
from vllm.platforms import current_platform


@pytest.mark.parametrize(
    "offload_config",
    [
        pytest.param(
            OffloadConfig(
                prefetch=PrefetchOffloadConfig(offload_group_size=1),
            ),
            id="auto_prefetch",
        ),
        pytest.param(
            OffloadConfig(
                offload_backend="prefetch",
                prefetch=PrefetchOffloadConfig(offload_group_size=1),
            ),
            id="explicit_prefetch",
        ),
    ],
)
def test_prefetch_offload_rejects_eplb(offload_config: OffloadConfig):
    # NOTE: PassConfig.flashinfer_max_size is patched to return None so that
    # VllmConfig._set_compile_ranges() skips the
    # `assert isinstance(self.model_config.dtype, torch.dtype)` branch. The
    # test does not exercise model loading, so model_config is left as None;
    # without this patch the unrelated allreduce-rms-fusion threshold compute
    # would dereference model_config and raise AttributeError. The behaviour
    # under test (OffloadConfig + EPLB validation) is orthogonal to flashinfer
    # fusion sizing.
    with (
        patch.object(current_platform, "is_cuda_alike", return_value=True),
        patch.object(current_platform, "device_count", return_value=2),
        patch.object(PassConfig, "flashinfer_max_size", return_value=None),
    ):
        parallel_config = ParallelConfig(
            enable_expert_parallel=True,
            enable_eplb=True,
            tensor_parallel_size=2,
        )

        with pytest.raises(
            (ValueError, ValidationError),
            match="Prefetch weight offloading does not support EPLB yet",
        ):
            VllmConfig(
                device_config=DeviceConfig(device="cpu"),
                parallel_config=parallel_config,
                offload_config=offload_config,
            )


def test_uva_offload_allows_eplb():
    offload_config = OffloadConfig(
        offload_backend="uva",
        uva={"cpu_offload_gb": 1},
    )

    # See note in test_prefetch_offload_rejects_eplb: patch flashinfer_max_size
    # so that the optional model_config.dtype assert in _set_compile_ranges is
    # skipped during this lightweight VllmConfig construction.
    with (
        patch.object(current_platform, "is_cuda_alike", return_value=True),
        patch.object(current_platform, "device_count", return_value=2),
        patch.object(PassConfig, "flashinfer_max_size", return_value=None),
    ):
        parallel_config = ParallelConfig(
            enable_expert_parallel=True,
            enable_eplb=True,
            tensor_parallel_size=2,
        )
        config = VllmConfig(
            device_config=DeviceConfig(device="cpu"),
            parallel_config=parallel_config,
            offload_config=offload_config,
        )

    assert config.offload_config.offload_backend == "uva"


@pytest.mark.parametrize("offload_backend", ["auto", "prefetch"])
def test_prefetch_offload_rejects_zero_prefetch_step_when_enabled(
    offload_backend: str,
):
    with pytest.raises(
        ValidationError,
        match="offload_prefetch_step.*must be >= 1",
    ):
        OffloadConfig(
            offload_backend=offload_backend,
            prefetch=PrefetchOffloadConfig(
                offload_group_size=1,
                offload_prefetch_step=0,
            ),
        )


def test_prefetch_offload_rejects_invalid_selector():
    with pytest.raises(ValidationError):
        PrefetchOffloadConfig(offload_selectors={"not_a_selector"})
