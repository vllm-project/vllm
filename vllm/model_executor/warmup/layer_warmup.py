# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warmup hooks owned by model layers.

Common warmup code does not know about individual models. A layer whose
kernels need warming before serving subclasses `LayerWarmup` and overrides the
hooks for the phases it needs. `kernel_warmup` calls each overridden hook once
per class, with every instance of that class in the model.
"""

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal, Self

import torch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

WarmupHook = Literal["warmup_jit_kernels", "warmup_model_runs", "autotune_kernels"]


class LayerWarmup:
    @classmethod
    def warmup_jit_kernels(cls, layers: list[Self], worker: "Worker") -> None:
        """JIT kernels by launching them directly.

        Runs only with `kernel_config.enable_jit_warmup`. Process-local: must
        not communicate with other ranks.
        """

    @classmethod
    def warmup_model_runs(cls, layers: list[Self], worker: "Worker") -> None:
        """Warm paths that only full model steps reach.

        Runs on every rank in lockstep, and not in process-local warmup.
        """

    @classmethod
    def autotune_kernels(cls, layers: list[Self]) -> None:
        """Autotune kernels inside the FlashInfer autotune context."""


def run_layer_warmups(model: torch.nn.Module, hook: WarmupHook, *args: Any) -> None:
    """Call `hook` once per class overriding it, in module order."""
    layers_by_owner: dict[type[LayerWarmup], list[LayerWarmup]] = defaultdict(list)
    for module in model.modules():
        if not isinstance(module, LayerWarmup):
            continue
        owner = next(cls for cls in type(module).__mro__ if hook in vars(cls))
        if owner is not LayerWarmup:
            layers_by_owner[owner].append(module)
    for owner, layers in layers_by_owner.items():
        getattr(owner, hook)(layers, *args)
