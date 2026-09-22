# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trace reload support for common MLAAttention backend layouts."""

import inspect
from dataclasses import dataclass
from functools import partial

import torch

from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_and_maybe_dequant_weights,
)

from .trace import ReloadError, ReloadState


class MLAProcessingPolicy:
    """Shared cold-load and reload processing contract for MLA layouts."""

    def process_weights(self, layer: MLAAttention, act_dtype: torch.dtype) -> None:
        raise NotImplementedError

    def warmup(self, layer: MLAAttention) -> None:
        pass

    def runtime_roles(self) -> tuple[str, ...]:
        raise NotImplementedError

    def bind_runtime(self, layer: MLAAttention, state: ReloadState) -> None:
        for role in self.runtime_roles():
            state.bind_target(role, partial(getattr, layer, role))

    def validate_runtime(self, layer: MLAAttention, state: ReloadState) -> None:
        for role in self.runtime_roles():
            target = state.targets[role].tensor
            if getattr(layer, role) is not target:
                raise ReloadError(f"MLA {role} was replaced")

    def process_reload(
        self, layer: MLAAttention, weight: torch.Tensor, act_dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError


@dataclass
class CommonMLAProcessingPolicy(MLAProcessingPolicy):
    """Materialize the regular ``W_UK_T`` and ``W_UV`` MLA tensors."""

    dcp_q_replicate: bool = False

    def runtime_roles(self) -> tuple[str, ...]:
        roles: tuple[str, ...] = ("W_UK_T", "W_UV")
        if self.dcp_q_replicate:
            roles += ("W_UK_T_dcp_qrep",)
        return roles

    def _split(
        self, layer: MLAAttention, weight: torch.Tensor, act_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (
            layer.kv_lora_rank,
            layer.num_heads * (layer.qk_nope_head_dim + layer.v_head_dim),
        )
        weight = weight.to(act_dtype).T.contiguous()
        if tuple(weight.shape) != expected:
            raise ReloadError(
                f"MLA kv_b_proj weight has shape {tuple(weight.shape)}, "
                f"expected {expected}"
            )
        weight = weight.view(
            layer.kv_lora_rank,
            layer.num_heads,
            layer.qk_nope_head_dim + layer.v_head_dim,
        )
        w_uk, w_uv = weight.split([layer.qk_nope_head_dim, layer.v_head_dim], dim=-1)
        return w_uk.permute(1, 2, 0).contiguous(), w_uv.transpose(0, 1).contiguous()

    def process_reload(
        self, layer: MLAAttention, weight: torch.Tensor, act_dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        w_uk_t, w_uv = self._split(layer, weight, act_dtype)
        values = {"W_UK_T": w_uk_t, "W_UV": w_uv}
        if self.dcp_q_replicate:
            from vllm.distributed import get_dcp_group

            # Gather the new projection on every DCP rank before publishing it.
            values["W_UK_T_dcp_qrep"] = get_dcp_group().all_gather(
                w_uk_t.contiguous(), dim=0
            )
        return values

    def process_weights(self, layer: MLAAttention, act_dtype: torch.dtype) -> None:
        weight = get_and_maybe_dequant_weights(layer.kv_b_proj, out_dtype=act_dtype)
        values = self.process_reload(layer, weight, act_dtype)
        from vllm.model_executor.utils import replace_parameter

        replace_parameter(layer, "W_UK_T", values["W_UK_T"], prefer_copy=True)
        replace_parameter(layer, "W_UV", values["W_UV"], prefer_copy=True)
        if self.dcp_q_replicate:
            layer.W_UK_T_dcp_qrep = values["W_UK_T_dcp_qrep"]


@dataclass
class AiterMLAProcessingPolicy(CommonMLAProcessingPolicy):
    """Materialize AITER FP8/FP4 batched GEMM weights and scales."""

    def runtime_roles(self) -> tuple[str, ...]:
        return ("W_K", "W_K_scale", "W_V", "W_V_scale")

    def warmup(self, layer: MLAAttention) -> None:
        if not layer.is_aiter_triton_fp8_bmm_enabled:
            return
        from tqdm import tqdm

        from vllm.distributed.parallel_state import is_global_first_rank
        from vllm.model_executor.layers.attention.mla_attention import (
            rocm_aiter_ops,
        )

        batch_sizes = range(1, 1025)
        if is_global_first_rank():
            batch_sizes = tqdm(
                batch_sizes,
                desc="[Aiter Triton] Pre-compiling fp8 BMM kernel",
                total=1024,
            )
        for batch_size in batch_sizes:
            q = torch.empty(
                (layer.W_K.shape[0], batch_size, layer.W_K.shape[2]),
                dtype=torch.bfloat16,
                device=layer.W_K.device,
            )
            rocm_aiter_ops.triton_fp8_bmm(
                q, layer.W_K, layer.W_K_scale, group_size=128, transpose_bm=True
            )
            v = torch.empty(
                (layer.W_V.shape[0], batch_size, layer.W_V.shape[2]),
                dtype=torch.bfloat16,
                device=layer.W_V.device,
            )
            rocm_aiter_ops.triton_fp8_bmm(
                v, layer.W_V, layer.W_V_scale, group_size=128, transpose_bm=True
            )

    def process_weights(self, layer: MLAAttention, act_dtype: torch.dtype) -> None:
        weight = get_and_maybe_dequant_weights(layer.kv_b_proj, out_dtype=act_dtype)
        for name, value in self.process_reload(layer, weight, act_dtype).items():
            setattr(layer, name, value)

    def process_reload(
        self, layer: MLAAttention, weight: torch.Tensor, act_dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        w_uk_t, w_uv = self._split(layer, weight, act_dtype)
        w_k = w_uk_t.transpose(1, 2)
        w_v = w_uv.transpose(1, 2)
        if layer.is_aiter_triton_fp4_bmm_enabled:
            from vllm.model_executor.layers.quantization.quark.utils import (
                quark_quantize_weight_to_mxfp4,
            )

            w_k, w_k_scale = quark_quantize_weight_to_mxfp4(w_k.transpose(0, 1))
            w_k = w_k.transpose(0, 1)
            w_k_scale = w_k_scale.transpose(0, 1)
            w_v, w_v_scale = quark_quantize_weight_to_mxfp4(w_v)
        else:
            from vllm.model_executor.layers.attention.mla_attention import (
                dynamic_per_batched_tensor_quant,
            )
            from vllm.platforms import current_platform

            w_k, w_k_scale = dynamic_per_batched_tensor_quant(
                w_k, dtype=current_platform.fp8_dtype()
            )
            w_v, w_v_scale = dynamic_per_batched_tensor_quant(
                w_v, dtype=current_platform.fp8_dtype()
            )
        return {
            "W_K": w_k,
            "W_K_scale": w_k_scale,
            "W_V": w_v,
            "W_V_scale": w_v_scale,
        }


@dataclass
class AMXMLAProcessingPolicy(MLAProcessingPolicy):
    """Replay the AMX MLA PWAL layout for reload.

    This implementation intentionally mirrors
    ``vllm.v1.attention.backends.mla.amx_mla.AMXMLAImpl.
    process_weights_after_loading``. Keep the two paths structurally aligned:
    the backend method is the cold-load source of truth, while this policy
    applies the same conversion to a new reload payload and copies the result
    into the existing packed storage.
    """

    def runtime_roles(self) -> tuple[str, ...]:
        return ("_w_uk_packed", "_w_uv_packed")

    def bind_runtime(self, layer: MLAAttention, state: ReloadState) -> None:
        state.bind_target("_w_uk_packed", partial(getattr, layer.impl, "_w_uk_packed"))
        state.bind_target("_w_uv_packed", partial(getattr, layer.impl, "_w_uv_packed"))

    def validate_runtime(self, layer: MLAAttention, state: ReloadState) -> None:
        impl = layer.impl
        for role in self.runtime_roles():
            if getattr(impl, role) is not state.targets[role].tensor:
                raise ReloadError(f"AMX MLA {role} was replaced")

    def process_weights(self, layer: MLAAttention, act_dtype: torch.dtype) -> None:
        layer.impl.process_weights_after_loading(act_dtype)
        layer.kv_b_proj.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)

    def process_reload(
        self, layer: MLAAttention, weight: torch.Tensor, act_dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        # Keep this conversion aligned with AMXMLAImpl.process_weights_after_loading:
        # transpose -> reshape -> split -> (N, L, P)/(N, V, L) -> pack.
        weight = weight.to(act_dtype).T.contiguous()
        weight = weight.view(
            layer.kv_lora_rank,
            layer.num_heads,
            layer.qk_nope_head_dim + layer.v_head_dim,
        )
        w_uk, w_uv = weight.split([layer.qk_nope_head_dim, layer.v_head_dim], dim=-1)
        w_uk = torch.ops._C.convert_weight_packed(w_uk.permute(1, 0, 2).contiguous())
        w_uv = torch.ops._C.convert_weight_packed(w_uv.permute(1, 2, 0).contiguous())
        return {"_w_uk_packed": w_uk, "_w_uv_packed": w_uv}


def get_mla_processing_policy(layer: MLAAttention) -> MLAProcessingPolicy:
    if layer.is_amx_bmm_enabled:
        return AMXMLAProcessingPolicy()
    if layer.is_aiter_triton_fp8_bmm_enabled or layer.is_aiter_triton_fp4_bmm_enabled:
        return AiterMLAProcessingPolicy()
    return CommonMLAProcessingPolicy(dcp_q_replicate=layer.dcp_q_replicate)


@dataclass
class MLAReloadPolicy:
    """Reload common MLA runtime targets from ``kv_b_proj``.

    These backends differ in attention execution, but share the same
    post-load targets. Backends that replace them with packed or quantized
    BMM state must use a separate policy.
    """

    attention: MLAAttention
    source: torch.nn.Module
    processing: MLAProcessingPolicy

    def destination(
        self, state: ReloadState, role: str, bound: inspect.BoundArguments
    ) -> torch.Tensor:
        if not state.checkpoint:
            self.prepare_for_load(state)
        return state.checkpoint[role]

    def bind(self, state: ReloadState) -> None:
        layer = self.attention
        if not isinstance(layer, MLAAttention):
            raise TypeError(f"{state.key}: expected an MLAAttention module")
        if (
            layer.dcp_q_replicate or layer.W_UK_T_dcp_qrep is not None
        ) and "W_UK_T_dcp_qrep" not in self.processing.runtime_roles():
            raise NotImplementedError(
                f"{state.key}: MLA DCP query replication reload is unsupported"
            )
        self.processing.bind_runtime(layer, state)

    def validate(self, state: ReloadState) -> None:
        self.processing.validate_runtime(self.attention, state)

    def prepare_for_load(self, state: ReloadState) -> None:
        pass

    def finish(self, state: ReloadState) -> None:
        self.validate(state)
        weight = get_and_maybe_dequant_weights(
            self.source, out_dtype=self.attention._mla_act_dtype
        )
        values = self.processing.process_reload(
            self.attention, weight, self.attention._mla_act_dtype
        )
        for role, value in values.items():
            state.copy_(role, value)
