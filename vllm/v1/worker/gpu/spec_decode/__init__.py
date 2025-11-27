# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.config import VllmConfig


def init_speculator(
    vllm_config: VllmConfig,
    device: torch.device,
):
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.use_eagle():
        from vllm.v1.worker.gpu.spec_decode.eagle import EagleSpeculator

        return EagleSpeculator(vllm_config, device)
    raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
