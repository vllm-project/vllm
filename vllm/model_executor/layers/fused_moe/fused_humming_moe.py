# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MoE utilities for Humming."""

import json
import math
from typing import TYPE_CHECKING, Any

import torch
from humming import dtypes
from humming.config import GemmType as HummingGemmType
from humming.layer import HummingLayerMeta, HummingMethod

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import moe_fused_mul_sum
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_permute,
    moe_unpermute,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms import current_platform
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.humming import HummingMoEMethod


logger = init_logger(__name__)


def get_humming_moe_gemm_type() -> str:
    env_gemm_type: str = envs.VLLM_HUMMING_MOE_GEMM_TYPE or ""
    env_gemm_type = env_gemm_type.lower()
    if env_gemm_type in ["indexed", "grouped"]:
        gemm_type = env_gemm_type
    elif current_platform.has_device_capability(90):
        # for device that supports TMA, use grouped gemm
        gemm_type = "grouped"
    else:
        gemm_type = "indexed"

    logger.info_once(f"Using {gemm_type} gemm for humming moe")  # noqa
    return gemm_type


class HummingExpertsBase(mk.FusedMoEExpertsModular):
    def __init__(
        self,
        layer: torch.nn.Module,
        quant_method: "HummingMoEMethod",
        prepare_finalize: mk.FusedMoEPrepareAndFinalizeModular | None = None,
    ):
        self.layer = layer
        self.num_experts = self.layer.num_experts
        self.global_num_experts = self.layer.global_num_experts
        self.init_humming_moe()

        if prepare_finalize is not None:
            max_num_tokens: int | None = None
            num_dispatchers: int | None = None
            if self.is_batched:
                max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
                num_dispatchers = prepare_finalize.num_dispatchers()

            assert quant_method.moe_quant_config is not None
            super().__init__(
                moe_config=quant_method.moe,
                quant_config=quant_method.moe_quant_config,
                max_num_tokens=max_num_tokens,
                num_dispatchers=num_dispatchers,
            )
        else:
            assert not self.is_batched

    def init_humming_moe(self):
        self.compute_config = {
            "use_batch_invariant": envs.VLLM_BATCH_INVARIANT,
            "use_f16_accum": envs.VLLM_HUMMING_USE_F16_ACCUM,
            "gemm_type": self.humming_gemm_type.value,
        }
        self.w13_tuning_config = HummingMethod.get_default_tuning_configs(
            layer=self.layer,
            use_f16_accum=envs.VLLM_HUMMING_USE_F16_ACCUM,
            use_batch_invariant=envs.VLLM_BATCH_INVARIANT,
            gemm_type=self.humming_gemm_type,
            sublayer_name="w13",
        )
        self.w2_tuning_config = HummingMethod.get_default_tuning_configs(
            layer=self.layer,
            use_f16_accum=envs.VLLM_HUMMING_USE_F16_ACCUM,
            use_batch_invariant=envs.VLLM_BATCH_INVARIANT,
            gemm_type=self.humming_gemm_type,
            sublayer_name="w2",
        )
        self.compute_config_str = json.dumps(self.compute_config)
        self.w13_tuning_config_str = json.dumps(self.w13_tuning_config)
        self.w2_tuning_config_str = json.dumps(self.w2_tuning_config)

    def get_global_valid_shape_m(self, topk_ids: torch.Tensor):
        num_tokens = topk_ids.size(0)
        ctx = get_forward_context()
        if ctx.dp_metadata is not None:
            num_tokens = ctx.dp_metadata.num_tokens_across_dp_cpu.sum().item()

        return num_tokens * topk_ids.size(1)

    def estimate_local_valid_shape_m(self, topk_ids: torch.Tensor):
        # estimate shape_m for kernel tuning
        global_valid_shape_m = self.get_global_valid_shape_m(topk_ids)
        num_experts = self.num_experts
        global_num_experts = self.global_num_experts
        return math.ceil(global_valid_shape_m * num_experts / global_num_experts)

    @property
    def humming_gemm_type(self) -> HummingGemmType:
        raise NotImplementedError

    @property
    def is_batched(self) -> bool:
        return self.activation_format() == mk.FusedMoEActivationFormat.BatchedExperts

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        platform = current_platform
        return platform.is_cuda() and platform.has_device_capability((7, 5))

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # Humming uses apply_moe_activation() callback for activation,
        # so any activation supported there can be used here.
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUSTEP,
            MoEActivation.SILU_NO_MUL,
            MoEActivation.GELU_NO_MUL,
            MoEActivation.RELU2_NO_MUL,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        meta1: HummingLayerMeta = self.layer.humming_metas["w13"]
        meta2: HummingLayerMeta = self.layer.humming_metas["w2"]

        assert meta1.num_experts == meta2.num_experts

        num_experts = meta1.num_experts
        top_k = topk_ids.size(1)
        assert w1.size(0) == num_experts
        assert w2.size(0) == num_experts

        if not self.is_batched:
            num_tokens = a1.size(0)
            assert topk_ids.size(0) == num_tokens
        else:
            assert a1.dim() == 3
            assert a1.size(0) == num_experts
            num_tokens = a1.size(1)

        return meta1.num_experts, num_tokens, meta1.shape_n // 2, meta1.shape_k, top_k

    def get_buffer_metas(self, M: int, topk: int, activation: MoEActivation):
        num_experts = self.num_experts
        N = self.layer.intermediate_size
        K = self.layer.hidden_size
        assert isinstance(num_experts, int)
        assert isinstance(N, int)
        assert isinstance(K, int)

        # hidden_states
        # (-> quanted_gate_up_input) (if not BF16/FP16 activation)
        # -> gate_up_output
        # -> activation_output
        # (-> quanted_down_input) (if not BF16/FP16 activation)
        # -> down_output
        # (-> output) (if not is_batched)
        # Neighboring nodes are required to utilize distinct workspaces.
        # The output must be derived from workspace1.

        output_shape: tuple[int, ...]
        if self.is_batched:
            max_num_tokens = self.max_num_tokens
            num_dispatchers = self.num_dispatchers
            assert max_num_tokens is not None and num_dispatchers is not None
            input_shape_m = num_experts * max_num_tokens
            real_shape_m = num_experts * max_num_tokens * num_dispatchers
            output_shape = (num_experts, max_num_tokens * num_dispatchers, K)
        else:
            input_shape_m = M
            if self.humming_gemm_type != HummingGemmType.INDEXED:
                input_shape_m = M * topk
            real_shape_m = M * topk
            output_shape = (M, K)

        down_input_size = N if activation.is_gated else (N * 2)
        a_dtype = self.layer.humming_metas["w13"].a_dtype
        c_dtype = self.layer.humming_metas["w13"].c_dtype
        num_bits = a_dtype.num_bits
        torch_dtype_map = {
            dtypes.float16: torch.float16,
            dtypes.bfloat16: torch.bfloat16,
            dtypes.float8e4m3: torch.float8_e4m3fn,
            dtypes.int8: torch.int8,
            dtypes.int4: torch.uint8,
        }

        buffer_metas = {
            "quanted_gate_up_input": {
                "shape": (input_shape_m, K),
                "dtype": torch_dtype_map[a_dtype],
            },
            "gate_up_output": {
                "shape": (real_shape_m, N * 2),
                "dtype": torch_dtype_map[c_dtype],
            },
            "activation_output": {
                "shape": (real_shape_m, down_input_size),
                "dtype": torch_dtype_map[c_dtype],
            },
            "quanted_down_input": {
                "shape": (real_shape_m, down_input_size),
                "dtype": torch_dtype_map[a_dtype],
            },
            "down_output": {
                "shape": output_shape if self.is_batched else (real_shape_m, K),
                "dtype": torch_dtype_map[c_dtype],
            },
            "output": {
                "shape": output_shape,
                "dtype": torch_dtype_map[c_dtype],
            },
        }

        for key in buffer_metas:
            meta = buffer_metas[key]
            if "quanted" in key and a_dtype.num_bits == 4:
                meta["shape"] = meta["shape"][:-1] + (meta["shape"][-1] // 2,)

        if num_bits == 16:
            required_buffers = ["gate_up_output", "activation_output", "down_output"]
        else:
            required_buffers = [
                "quanted_gate_up_input",
                "gate_up_output",
                "activation_output",
                "quanted_down_input",
                "down_output",
            ]

        # batched moe use down_output as output
        if not self.is_batched:
            required_buffers.append("output")

        return buffer_metas, required_buffers

    def _workspace_shapes(self, M: int, topk: int, activation: MoEActivation):
        buffer_metas, required_buffers = self.get_buffer_metas(M, topk, activation)

        workspace1_nbytes = 0
        workspace2_nbytes = 0

        for index, name in enumerate(required_buffers[::-1]):
            buffer_meta = buffer_metas[name]
            nelement = math.prod(buffer_meta["shape"])
            nbytes = nelement * buffer_meta["dtype"].itemsize
            if index % 2 == 0:
                workspace1_nbytes = max(workspace1_nbytes, nbytes)
            else:
                workspace2_nbytes = max(workspace2_nbytes, nbytes)

        output_key = "down_output" if self.is_batched else "output"
        output_shape = buffer_metas[output_key]["shape"]

        return (workspace1_nbytes // 2,), (workspace2_nbytes // 2,), output_shape

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return self._workspace_shapes(M, topk, activation)

    def make_workspaces(self, M: int, topk: int, activation: MoEActivation):
        shapes = self._workspace_shapes(M, topk, activation)
        workspace1_shape, workspace2_shape, output_shape = shapes
        torch_dtype = self.layer.param_dtype
        workspace1, workspace2 = current_workspace_manager().get_simultaneous(
            (workspace1_shape, torch_dtype),
            (workspace2_shape, torch_dtype),
        )
        output = _resize_cache(workspace1, output_shape)
        return workspace1, workspace2, output

    def prepare_buffers(
        self,
        workspace1: torch.Tensor,
        workspace2: torch.Tensor,
        M: int,
        topk: int,
        activation: MoEActivation,
    ) -> dict[str, torch.Tensor]:
        buffer_metas, required_buffers = self.get_buffer_metas(M, topk, activation)
        buffers = {}
        for index, name in enumerate(required_buffers[::-1]):
            buffer_meta = buffer_metas[name]
            workspace = workspace1 if index % 2 == 0 else workspace2
            workspace = workspace.view(buffer_meta["dtype"])
            buffers[name] = _resize_cache(workspace, buffer_meta["shape"])

        return buffers

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert not apply_router_weight_on_input

        self.main_apply(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            workspace1=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=expert_tokens_meta,
        )

    def main_apply(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        workspace1: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
    ):
        raise NotImplementedError


class HummingIndexedExperts(HummingExpertsBase):
    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @property
    def humming_gemm_type(self) -> HummingGemmType:
        return HummingGemmType.INDEXED

    def prepare_humming_moe_kwargs(
        self,
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        valid_shape_m = self.estimate_local_valid_shape_m(topk_ids)

        for min_shape_m, max_shape_m, config in self.w13_tuning_config:
            if valid_shape_m > min_shape_m and valid_shape_m <= max_shape_m:
                moe_block_size = config["block_shape"][0]
                break
        else:
            raise ValueError(f"cannot found moe_block_size for shape {valid_shape_m}")

        sorted_ids, expert_ids, num_tokens_padded = moe_align_block_size(
            topk_ids=topk_ids,
            block_size=moe_block_size,
            num_experts=self.global_num_experts,
            expert_map=expert_map,
            ignore_invalid_experts=True,
        )

        moe_common_kwargs = {
            "sorted_ids": sorted_ids,
            "expert_ids": expert_ids,
            "num_tokens_padded": num_tokens_padded,
            "compute_config": self.compute_config_str,
            "valid_shape_m": valid_shape_m,
        }

        top_k = topk_ids.size(1)
        moe_kwargs1 = {"top_k": top_k, "tuning_config": self.w13_tuning_config_str}
        moe_kwargs2 = {"top_k": 1, "tuning_config": self.w2_tuning_config_str}
        moe_kwargs1.update(moe_common_kwargs)
        moe_kwargs2.update(moe_common_kwargs)

        return moe_kwargs1, moe_kwargs2

    def main_apply(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        workspace1: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
    ):
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        buffers = self.prepare_buffers(
            workspace1,
            workspace2,
            topk_ids.size(0),
            topk_ids.size(1),
            self.layer.activation,
        )

        moe_kwargs1, moe_kwargs2 = self.prepare_humming_moe_kwargs(
            topk_ids=topk_ids,
            expert_map=self.layer.expert_map,
            expert_tokens_meta=expert_tokens_meta,
        )

        inputs, input_scale = HummingMethod.may_quant_input(
            layer=self.layer,
            inputs=hidden_states,
            quanted_input=buffers.get("quanted_gate_up_input", None),
            sublayer_name="w13",
        )

        HummingMethod.forward_layer(
            layer=self.layer,
            inputs=inputs,
            input_scale=input_scale,
            outputs=buffers["gate_up_output"],
            sublayer_name="w13",
            **moe_kwargs1,
        )

        self.activation(
            activation=self.layer.activation,
            input=buffers["gate_up_output"],
            output=buffers["activation_output"],
        )

        inputs, input_scale = HummingMethod.may_quant_input(
            layer=self.layer,
            inputs=buffers["activation_output"],
            quanted_input=buffers.get("quanted_down_input", None),
            sublayer_name="w2",
        )

        HummingMethod.forward_layer(
            layer=self.layer,
            inputs=inputs,
            input_scale=input_scale,
            outputs=buffers["down_output"].view(-1, hidden_states.size(-1)),
            sublayer_name="w2",
            **moe_kwargs2,
        )

        moe_fused_mul_sum(
            inputs=buffers["down_output"].view(*topk_ids.shape, -1),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_map=self.layer.expert_map,
            outputs=buffers["output"],
        )


class HummingGroupedExperts(HummingExpertsBase):
    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @property
    def humming_gemm_type(self) -> HummingGemmType:
        return HummingGemmType.GROUPED_CONTIGUOUS

    def main_apply(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        workspace1: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
    ):
        valid_shape_m = self.estimate_local_valid_shape_m(topk_ids)

        buffers = self.prepare_buffers(
            workspace1,
            workspace2,
            topk_ids.size(0),
            topk_ids.size(1),
            self.layer.activation,
        )

        hidden_states, _, expert_first_token_offset, inv_perm, _ = moe_permute(
            hidden_states=hidden_states,
            a1q_scale=None,
            topk_ids=topk_ids,
            n_expert=self.global_num_experts,
            n_local_expert=self.num_experts,
            expert_map=self.layer.expert_map,
        )

        inputs, input_scale = HummingMethod.may_quant_input(
            layer=self.layer,
            inputs=hidden_states,
            quanted_input=buffers.get("quanted_gate_up_input", None),
            sublayer_name="w13",
        )

        HummingMethod.forward_layer(
            layer=self.layer,
            inputs=inputs,
            input_scale=input_scale,
            outputs=buffers["gate_up_output"],
            valid_shape_m=valid_shape_m,
            expert_layout=expert_first_token_offset,
            compute_config=self.compute_config_str,
            tuning_config=self.w13_tuning_config_str,
            sublayer_name="w13",
        )

        self.activation(
            activation=self.layer.activation,
            input=buffers["gate_up_output"],
            output=buffers["activation_output"],
        )

        inputs, input_scale = HummingMethod.may_quant_input(
            layer=self.layer,
            inputs=buffers["activation_output"],
            quanted_input=buffers.get("quanted_down_input", None),
            sublayer_name="w2",
        )

        HummingMethod.forward_layer(
            layer=self.layer,
            inputs=inputs,
            input_scale=input_scale,
            outputs=buffers["down_output"],
            valid_shape_m=valid_shape_m,
            expert_layout=expert_first_token_offset,
            compute_config=self.compute_config_str,
            tuning_config=self.w2_tuning_config_str,
            sublayer_name="w2",
        )

        moe_unpermute(
            out=buffers["output"],
            permuted_hidden_states=buffers["down_output"].view(*topk_ids.shape, -1),
            topk_weights=topk_weights,
            inv_permuted_idx=inv_perm,
            expert_first_token_offset=expert_first_token_offset,
        )


class BatchedHummingGroupedExperts(HummingExpertsBase):
    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceDelegate()

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    @property
    def humming_gemm_type(self) -> HummingGemmType:
        return HummingGemmType.GROUPED_MASKED

    def main_apply(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        workspace1: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
    ):
        assert expert_tokens_meta is not None
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        valid_shape_m = self.estimate_local_valid_shape_m(topk_ids)
        expert_num_tokens = expert_tokens_meta.expert_num_tokens

        buffers = self.prepare_buffers(
            workspace1,
            workspace2,
            topk_ids.size(0),
            topk_ids.size(1),
            self.layer.activation,
        )

        inputs, input_scale = HummingMethod.may_quant_input(
            layer=self.layer,
            inputs=hidden_states,
            quanted_input=buffers.get("quanted_gate_up_input", None),
            sublayer_name="w13",
        )

        HummingMethod.forward_layer(
            layer=self.layer,
            inputs=inputs,
            input_scale=input_scale,
            outputs=buffers["gate_up_output"],
            valid_shape_m=valid_shape_m,
            expert_layout=expert_num_tokens,
            compute_config=self.compute_config_str,
            tuning_config=self.w13_tuning_config_str,
            sublayer_name="w13",
        )

        self.activation(
            activation=self.layer.activation,
            input=buffers["gate_up_output"],
            output=buffers["activation_output"],
        )

        inputs, input_scale = HummingMethod.may_quant_input(
            layer=self.layer,
            inputs=buffers["activation_output"],
            quanted_input=buffers.get("quanted_down_input", None),
            sublayer_name="w2",
        )

        HummingMethod.forward_layer(
            layer=self.layer,
            inputs=inputs,
            input_scale=input_scale,
            outputs=buffers["down_output"].view(-1, hidden_states.size(-1)),
            valid_shape_m=valid_shape_m,
            expert_layout=expert_num_tokens,
            compute_config=self.compute_config_str,
            tuning_config=self.w2_tuning_config_str,
            sublayer_name="w2",
        )
