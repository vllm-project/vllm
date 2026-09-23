# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reload policies for MXFP4 expert weights."""

from dataclasses import dataclass
from functools import partial

from .trace import ReloadError, ReloadState


@dataclass
class Mxfp4EmulationMoEReloadPolicy:
    """Reload canonical MXFP4 tensors retained by the emulation backend.

    The emulation backend does not repack expert weights during PWAL. The
    policy therefore updates the existing tensors in place and keeps the
    already-created kernel object valid.
    """

    def bind(self, state: ReloadState) -> None:
        method = state.module.quant_method
        if method.mxfp4_backend.name != "EMULATION":
            raise ReloadError("MXFP4 emulation policy bound to another backend")
        self.method = method
        self.kernel = method.moe_kernel
        if self.kernel is None:
            raise ReloadError("MXFP4 emulation kernel is not initialized")
        for role in state.roles:
            state.bind_target(role, partial(getattr, state.module, role))

    def validate(self, state: ReloadState) -> None:
        method = state.module.quant_method
        if method is not self.method or method.moe_kernel is not self.kernel:
            raise ReloadError("MXFP4 emulation kernel changed since binding")

    def destination(self, state: ReloadState, role, bound):
        del bound
        if not state.checkpoint:
            self.prepare_for_load(state)
        return state.checkpoint[role]

    def prepare_for_load(self, state: ReloadState) -> None:
        state.prepare_sources(reuse_roles=state.roles)

    def finish(self, state: ReloadState) -> None:
        self.validate(state)
        for role in state.roles:
            state.copy_(role, state.work(role))


@dataclass
class FlashInferMxfp4MoEReloadPolicy:
    """Replay FlashInfer MXFP4's deterministic weight conversion."""

    def bind(self, state: ReloadState) -> None:
        method = state.module.quant_method
        if method.mxfp4_backend.name != "FLASHINFER_CUTLASS_MXFP4_BF16":
            raise ReloadError("FlashInfer MXFP4 policy bound to another backend")
        self.method = method
        self.kernel = method.moe_kernel
        self.config = method.moe_quant_config
        if self.kernel is None or self.config is None:
            raise ReloadError("FlashInfer MXFP4 kernel is not initialized")
        for role in state.roles:
            state.bind_target(role, partial(getattr, state.module, role))

    def validate(self, state: ReloadState) -> None:
        method = state.module.quant_method
        if (
            method is not self.method
            or method.moe_kernel is not self.kernel
            or method.moe_quant_config is not self.config
        ):
            raise ReloadError("FlashInfer MXFP4 runtime objects changed")

    def destination(self, state: ReloadState, role, bound):
        del bound
        if not state.checkpoint:
            self.prepare_for_load(state)
        return state.checkpoint[role]

    def prepare_for_load(self, state: ReloadState) -> None:
        # Packed runtime tensors are not canonical loading destinations.
        state.prepare_sources(reuse_roles=())

    def finish(self, state: ReloadState) -> None:
        from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
            convert_weight_to_mxfp4_moe_kernel_format,
        )

        self.validate(state)
        layer = state.module
        values = {
            role: state.work(role)
            for role in (
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w2_weight_scale",
            )
        }
        converted = convert_weight_to_mxfp4_moe_kernel_format(
            mxfp4_backend=self.method.mxfp4_backend,
            layer=layer,
            **values,
            _cache_permute_indices=self.method._cache_permute_indices,
            activation=self.method.moe.activation,
        )
        for role, value in zip(values, converted[:4]):
            target = state.targets[role].tensor
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ReloadError(f"{state.key}/{role}: MXFP4 runtime layout changed")
            state.copy_(role, value)
        for role, value in zip(("w13_bias", "w2_bias"), converted[4:]):
            if role not in state.roles or value is None:
                continue
            target = state.targets[role].tensor
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ReloadError(f"{state.key}/{role}: MXFP4 bias layout changed")
            state.copy_(role, value)
