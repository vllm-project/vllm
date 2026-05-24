# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.config import VllmConfig


def init_speculator(vllm_config: VllmConfig, device: torch.device):
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    # Gumiho is part of use_eagle() but needs a custom MLP post-step that the
    # V2 model runner's EagleSpeculator does not implement; surface a clear
    # error instead of silently falling through to EagleSpeculator.
    if speculative_config.method == "gumiho":
        raise NotImplementedError(
            "gumiho speculative decoding is not supported by the V2 model "
            "runner yet. Use the default V1 model runner instead."
        )
    if speculative_config.use_eagle():
        from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator

        return EagleSpeculator(vllm_config, device)
    raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
