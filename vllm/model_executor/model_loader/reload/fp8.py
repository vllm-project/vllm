# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 checkpoint-to-runtime policies; never install parameters or kernels."""

import inspect
from copy import copy
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace

import torch

from .trace import ReloadError, ReloadState


@dataclass
class DeepGEMMReloadPolicy:
    pairs: tuple[tuple[str, str], ...]
    block_shape: tuple[int, ...]
    use_e8m0: bool
    is_bmm: bool = False
    bmm_batch_size: int = 0

    def bind(self, state: ReloadState) -> None:
        method = state.module.quant_method
        self.kernel = getattr(method, "fp8_linear", getattr(method, "moe_kernel", None))
        for weight, scale in self.pairs:
            if state.targets[weight].tensor.dtype != torch.float8_e4m3fn:
                raise ReloadError("DeepGEMM reload requires E4M3FN runtime weights")

    def validate(self, state: ReloadState) -> None:
        self._validate_kernel(state)

    def _validate_kernel(self, state: ReloadState) -> None:
        method = state.module.quant_method
        kernel = getattr(method, "fp8_linear", getattr(method, "moe_kernel", None))
        if kernel is not self.kernel:
            raise ReloadError("DeepGEMM kernel changed since runtime binding")

    def destination(
        self, state: ReloadState, role: str, bound: inspect.BoundArguments
    ) -> torch.Tensor:
        self._validate_kernel(state)
        return state.source(
            role, alias_runtime=all(role != scale for _, scale in self.pairs)
        )

    def finish(self, state: ReloadState) -> None:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            deepgemm_post_process_fp8_weight_block,
        )

        self._validate_kernel(state)
        for weight, scale in self.pairs:
            converted_weight, converted_scale = deepgemm_post_process_fp8_weight_block(
                state.work(weight),
                state.work(scale),
                self.block_shape,
                use_e8m0=self.use_e8m0,
                is_bmm=self.is_bmm,
                bmm_batch_size=self.bmm_batch_size,
            )
            state.copy_(weight, converted_weight)
            state.copy_(scale, converted_scale)
        converted_roles = {role for pair in self.pairs for role in pair}
        for role in state.roles:
            if role not in converted_roles:
                state.copy_(role, state.work(role))


@dataclass
class CutlassMoEReloadPolicy:
    block_quant: bool
    is_act_and_mul: bool
    shard_size: int
    num_experts: int

    @property
    def scale_names(self) -> tuple[str, str]:
        suffix = "weight_scale_inv" if self.block_quant else "weight_scale"
        return f"w13_{suffix}", f"w2_{suffix}"

    def bind(self, state: ReloadState) -> None:
        method = state.module.quant_method
        self.kernel = method.moe_kernel
        self.config = method.moe_quant_config
        if self.kernel is None or self.config is None:
            raise ReloadError("Bind CUTLASS reload only after kernel initialization")
        for name in ("w13_weight", "w2_weight"):
            if state.targets[name].tensor.dtype != torch.float8_e4m3fn:
                raise ReloadError("CUTLASS reload requires E4M3FN runtime weights")
        if not self.block_quant:
            for name in ("g1_alphas", "g2_alphas", "a1_gscale", "a2_gscale"):
                if not isinstance(getattr(self.config, name), torch.Tensor):
                    raise ReloadError(
                        "CUTLASS per-tensor requires static activation scales"
                    )
                state.bind_target(name, partial(getattr, self.config, name))

    def validate(self, state: ReloadState) -> None:
        self._validate_kernel(state)

    def _validate_kernel(self, state: ReloadState) -> None:
        method = state.module.quant_method
        if (
            method.moe_kernel is not self.kernel
            or method.moe_quant_config is not self.config
        ):
            raise ReloadError("CUTLASS kernel/config changed since runtime binding")

    def destination(
        self, state: ReloadState, role: str, bound: inspect.BoundArguments
    ) -> torch.Tensor:
        self._validate_kernel(state)
        # Block weights/scales are written directly in W31 order by the original
        # loader. With preservation enabled the source stays in checkpoint order.
        alias = self.block_quant or role in ("w13_weight", "w2_weight")
        destination = state.source(role, alias_runtime=alias)
        runtime = state.targets[role].tensor
        if (
            self.block_quant
            and self.is_act_and_mul
            and role.startswith("w13_")
            and destination.untyped_storage().data_ptr()
            == runtime.untyped_storage().data_ptr()
        ):
            shard = bound.arguments.get("shard_id")
            if shard not in ("w1", "w3"):
                raise ReloadError("CUTLASS direct W31 writes require w1/w3 arrivals")
            bound.arguments["shard_id"] = "w3" if shard == "w1" else "w1"
        return destination

    def finish(self, state: ReloadState) -> None:
        from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
            clamp_fp8_moe_block_scale,
            prepare_fp8_moe_layer_for_fi,
            swap_w13_to_w31,
        )
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            process_fp8_input_tensor_strategy_moe,
            process_fp8_weight_tensor_strategy_moe,
        )

        self._validate_kernel(state)
        s13, s2 = self.scale_names
        if self.block_quant:
            for name in ("w13_weight", "w2_weight", s13, s2):
                value = state.work(name)
                runtime = state.targets[name].tensor
                aliases_runtime = (
                    value.untyped_storage().data_ptr()
                    == runtime.untyped_storage().data_ptr()
                )
                if (
                    self.is_act_and_mul
                    and name.startswith("w13_")
                    and not aliases_runtime
                ):
                    value = swap_w13_to_w31(value)
                if name in (s13, s2):
                    clamp_fp8_moe_block_scale(value)
                state.copy_(name, value)
            return

        w13, w2 = state.work("w13_weight"), state.work("w2_weight")
        w13_scale, w2_scale = state.work(s13), state.work(s2)
        a1, a2 = process_fp8_input_tensor_strategy_moe(
            state.work("w13_input_scale"),
            state.work("w2_input_scale"),
            state.module.moe_config.moe_parallel_config.enable_eplb,
        )
        w13, w13_scale = process_fp8_weight_tensor_strategy_moe(
            w13, w13_scale, self.shard_size, self.num_experts, self.is_act_and_mul
        )
        # The cold-load helper updates alignment metadata. Give it a shell, not
        # the live module/config already referenced by the kernel.
        shell = SimpleNamespace(
            moe_config=copy(state.module.moe_config),
            activation=state.module.activation,
        )
        w13, w2, w13_scale, w2_scale = prepare_fp8_moe_layer_for_fi(
            shell, w13, w2, w13_scale, a1, w2_scale, a2
        )
        for name, value in (
            ("w13_weight", w13),
            ("w2_weight", w2),
            (s13, w13_scale),
            (s2, w2_scale),
            ("w13_input_scale", a1),
            ("w2_input_scale", a2),
            ("g1_alphas", w13_scale * a1),
            ("g2_alphas", w2_scale * a2),
            ("a1_gscale", 1.0 / a1),
            ("a2_gscale", 1.0 / a2),
        ):
            state.copy_(name, value)
