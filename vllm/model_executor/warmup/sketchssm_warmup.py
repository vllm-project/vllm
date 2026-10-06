# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def sketchssm_warmup(runner: "GPUModelRunner") -> None:
    """Compile the SketchSSM Triton kernels before the first request."""
    if runner.vllm_config.cache_config.sketchssm is None:
        return
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2

    for module in runner.get_model().modules():
        if isinstance(module, MambaMixer2) and module.sketchssm is not None:
            module.sketchssm.warmup(module.kv_cache[1])
