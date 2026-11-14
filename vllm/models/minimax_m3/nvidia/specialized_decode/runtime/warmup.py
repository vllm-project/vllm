# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tuning and compiling the specialized decode's stages.

The model state calls `warm_up` once the executor is built, before memory
profiling and CUDA graph capture, so no kernel compiles or tunes inside a
capture. Each stage implementation warms up for the row counts it runs; the
stock stages need nothing (the eager stock forward before each capture
compiles their kernels). A stage whose warm-up is collective (FlashInfer's
tuner reduces its timings over the TP group) runs the same calls in the same
order on every rank, which this order guarantees.
"""

import torch

from vllm.config import VllmConfig

from .decode import SpecializedDecode


def step_rows(decode: SpecializedDecode, vllm_config: VllmConfig) -> list[int]:
    """Every row count an executor step can have: the FULL decode CUDA graph
    capture sizes up to the executor's `max_rows`."""
    sizes = vllm_config.compilation_config.cudagraph_capture_sizes or []
    return sorted({size for size in sizes if 0 < size <= decode.max_rows})


def warm_up(decode: SpecializedDecode, vllm_config: VllmConfig) -> None:
    """Warm every stage implementation up for the step rows it supports."""
    rows = step_rows(decode, vllm_config)
    with torch.inference_mode():
        for stage in decode.stages.all():
            supported = [r for r in rows if stage.supports(r)]
            if supported:
                stage.warm_up(supported)
