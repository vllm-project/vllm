# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import envs
from vllm.platforms import current_platform


@pytest.fixture(autouse=True)
def skip_unsupported_flashinfer_sampler():
    # CI runs this suite with FlashInfer both explicitly enabled and disabled.
    if (
        current_platform.is_cuda()
        and envs.is_set("VLLM_USE_FLASHINFER_SAMPLER")
        and envs.VLLM_USE_FLASHINFER_SAMPLER
        and current_platform.num_compute_units(torch.accelerator.current_device_index())
        <= 16
    ):
        pytest.skip("FlashInfer top-k masking requires more than 16 SMs")
