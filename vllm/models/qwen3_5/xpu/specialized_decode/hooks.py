# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build optional MoE executors before memory profiling and graph capture."""

import torch

from vllm.config import get_current_vllm_config

from . import is_requested


@torch.no_grad()
def initialize_specialized_decode(model: torch.nn.Module) -> None:
    config = getattr(model, "vllm_config", None)
    if config is None:
        config = get_current_vllm_config()
    if not is_requested(config):
        return

    from ..model import Qwen3_5SparseMoeBlock
    from .config import agree
    from .moe import FusedMoeDecode
    from .planning import DecodeRuntime

    runtime = getattr(model, "_xpu_moe_decode_runtime", None)
    if runtime is None:
        runtime = DecodeRuntime()
    failures = []
    for name, module in model.named_modules():
        if not isinstance(module, Qwen3_5SparseMoeBlock):
            continue
        if module._decode is not None:
            if not module._decode.bindings_match():
                failures.append(f"{name}: reload changed canonical weight bindings")
            continue
        executor = FusedMoeDecode(module, runtime)
        if executor.plan is None:
            failures.append(f"{name}: {executor.reason}")
        else:
            runtime.executors.append(executor)
            module._decode = executor
    if not runtime.executors:
        failures.append("requires at least one qualified MoE block")
    agree(failures)
    model._xpu_moe_decode_runtime = runtime
