# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark Omni model for CPU."""

import torch as _torch
import torch.nn.functional as _F

from vllm.config import get_current_vllm_config as _get_current_vllm_config
from vllm.model_executor.layers.quantization.inc.inc import INCConfig
from vllm.models.muse_spark.common import vision as _vision
from vllm.models.muse_spark.nvidia.multimodal import (
    MuseSparkForConditionalGeneration as _BaseMuseSparkForConditionalGeneration,
)

from .model import MuseSpark1xDecoderLayer, MuseSpark1xModel, make_cpu_weights_mapper


class _CPUSafeGridSampleF:
    """Proxy for common.vision's module-level `F` alias that upcasts
    grid_sample to fp32 on CPU inputs before downcasting the result back.

    torch's CPU grid_sampler_2d kernel produces NaN for bf16 inputs at the
    scale NaViT samples positional embeddings at (large channel count,
    hundreds of grid points) -- a CPU-only kernel gap confirmed by isolated
    repro against plain torch, independent of any MuseSpark code; CUDA's
    kernel handles bf16 correctly. Reassigning common.vision's `F` name to
    this proxy (rather than patching torch.nn.functional.grid_sample
    itself) confines the fix to NaViT's one call site instead of every
    grid_sample call in the process.
    """

    def __getattr__(self, name):
        return getattr(_F, name)

    def grid_sample(self, input, grid, *args, **kwargs):
        if input.is_cpu and input.dtype != _torch.float32:
            dtype = input.dtype
            return _F.grid_sample(input.float(), grid.float(), *args, **kwargs).to(
                dtype
            )
        return _F.grid_sample(input, grid, *args, **kwargs)


_vision.F = _CPUSafeGridSampleF()


_orig_navit_encoder_init = _vision.NaViTEncoder.__init__


def _cpu_navit_encoder_init(self, *args, vllm_config=None, **kwargs):
    """Force-enable `compile_mm_encoder` for NaViTEncoder's own construction.

    `NaViTEncoder` compiles cleanly with no graph breaks or recompiles
    across varying image shapes on CPU (verified at both reduced and full
    scale), unlike the general `compile_mm_encoder` opt-in flag's
    off-by-default assumption. Rather than editing the shared decorator
    call site in `common/vision.py`, temporarily flip the flag on the
    shared `VllmConfig` for just the duration of this constructor call --
    it's read once, synchronously, inside `__init__`, and restored before
    control returns to the caller, so sibling components (e.g. the audio
    encoder's separate `compile_mm_encoder` check) are unaffected.
    """
    cfg = vllm_config if vllm_config is not None else _get_current_vllm_config()
    compilation_config = cfg.compilation_config
    prev = compilation_config.compile_mm_encoder
    compilation_config.compile_mm_encoder = True
    try:
        _orig_navit_encoder_init(self, *args, vllm_config=vllm_config, **kwargs)
    finally:
        compilation_config.compile_mm_encoder = prev


_vision.NaViTEncoder.__init__ = _cpu_navit_encoder_init


class MuseSparkForConditionalGeneration(_BaseMuseSparkForConditionalGeneration):
    """MuseSpark Omni model using the CPU text decoder stack."""

    model_cls = MuseSpark1xModel
    decoder_layer_cls = MuseSpark1xDecoderLayer
    supported_quantization_configs = (INCConfig,)
    supported_quantization_description = "INC/AutoRound GPTQ-int4"
    hf_to_vllm_mapper = make_cpu_weights_mapper(
        _BaseMuseSparkForConditionalGeneration.hf_to_vllm_mapper
    )


__all__ = ["MuseSparkForConditionalGeneration"]
