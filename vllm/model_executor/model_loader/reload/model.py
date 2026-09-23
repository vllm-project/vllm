# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-level reload states for post-load derived model tensors."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import cast

import torch

from .trace import ReloadError, ReloadState


@dataclass
class ModelFinalizeReloadPolicy:
    """Replay a model's post-load finalization after child states finish.

    The callback must only update already allocated runtime objects. It must
    not replace parameters or rebuild quantization kernels during reload.
    """

    finalize: Callable[[], None]
    runtime_targets: tuple[tuple[str, Callable[[], torch.Tensor | None]], ...] = ()

    def bind(self, state: ReloadState) -> None:
        # PWAL creates derived tensors after the state builder has run.
        for role, resolve in self.runtime_targets:
            if resolve() is not None:
                state.bind_target(role, cast(Callable[[], torch.Tensor], resolve))

    def validate(self, state: ReloadState) -> None:
        for target in state.targets.values():
            target.validate()

    def destination(self, state: ReloadState, role, bound):
        del state, role, bound
        raise ReloadError("Model-level finalize states do not receive weights")

    def prepare_for_load(self, state: ReloadState) -> None:
        del state

    def finish(self, state: ReloadState) -> None:
        self.validate(state)
        self.finalize()
        self.validate(state)


def create_deepseek_model_reload_state(model, key: str) -> ReloadState:
    """Build the DeepSeek V4/V4.1 model-level finalize state.

    The current NVIDIA model hook has two parts: MegaMoE packing and the
    mHC broadcast tensor. MegaMoE is only enabled on SM100, so H200 runs the
    second part. The state still depends on all decoder layers that own
    ``hc_attn_fn``; this keeps the ordering correct for both paths.
    """

    dependencies: list[str] = []
    runtime_targets = []
    for subkey, module in model.named_modules():
        if not hasattr(module, "hc_attn_fn"):
            continue
        dependency = f"{key}.{subkey}" if key and subkey else (key or subkey)
        if dependency:
            dependencies.append(dependency)
        runtime_targets.append(
            (
                f"{subkey}.hc_attn_fn_broadcast",
                partial(getattr, module, "hc_attn_fn_broadcast", None),
            )
        )

    target = getattr(model, "language_model", model)
    target_model = getattr(target, "model", target)

    def finalize() -> None:
        """Run the child finalizers without the wrapper's one-shot guard."""
        if hasattr(target_model, "finalize_mega_moe_weights"):
            target_model.finalize_mega_moe_weights()
        if hasattr(target_model, "finalize_mhc_broadcast_weights"):
            target_model.finalize_mhc_broadcast_weights()
        if not hasattr(target_model, "finalize_mega_moe_weights") and hasattr(
            target, "_finalize_moe"
        ):
            target._finalize_moe()

    return ReloadState(
        key=key,
        module=model,
        roles=(),
        policy=ModelFinalizeReloadPolicy(
            finalize=finalize,
            runtime_targets=tuple(runtime_targets),
        ),
        dependencies=tuple(sorted(set(dependencies))),
    )
