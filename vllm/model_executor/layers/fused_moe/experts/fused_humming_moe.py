# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MoE utilities for Humming."""

import json
import math
from typing import TYPE_CHECKING, Any, cast

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import moe_fused_mul_sum
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    MoEPermuteScratch,
    moe_permute,
    moe_permute_unpermute_supported,
    moe_unpermute,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kInt4Static,
    kInt8DynamicTokenSym,
    kInt8Static,
    kInt8StaticChannelSym,
    kMxfp4Dynamic,
    kMxfp4Static,
    kMxfp8Dynamic,
    kMxfp8Static,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_humming
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.utils.humming_utils import (
        HummingMoEQuantConfig,
    )
    from vllm.utils.humming import (
        GemmType as HummingGemmType,
    )
    from vllm.utils.humming import (
        LayerConfig as HummingLayerConfig,
    )


logger = init_logger(__name__)


def get_humming_moe_gemm_type() -> str:
    env_gemm_type: str | None = envs.VLLM_HUMMING_MOE_GEMM_TYPE
    gemm_type = "indexed"
    if env_gemm_type is not None:
        env_gemm_type = env_gemm_type.lower()
        if env_gemm_type == "indexed":
            gemm_type = env_gemm_type
        elif env_gemm_type in ["grouped_contiguous", "grouped"]:
            gemm_type = "grouped_contiguous"
        else:
            gemm_type = "indexed"

    logger.info_once(f"Using {gemm_type} gemm for humming moe")  # noqa
    return gemm_type


class HummingExpertsBase(mk.FusedMoEExpertsModular):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        humming_quant_config = cast("HummingMoEQuantConfig", quant_config)
        self.humming_configs: dict[str, HummingLayerConfig] = {
            "w13": humming_quant_config.w1_humming_config,
            "w2": humming_quant_config.w2_humming_config,
        }
        self.locks = torch.zeros(1024, dtype=torch.int32, device=moe_config.device)
        self.num_experts = moe_config.num_local_experts
        self.global_num_experts = moe_config.num_experts
        self.quant_config = quant_config
        self.init_humming_moe()

        if self.is_batched():
            assert max_num_tokens is not None and num_dispatchers is not None

        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        self._permute_scratch: MoEPermuteScratch | None = None

    def init_humming_moe(self):
        from vllm.utils.humming import get_heuristics_config

        self.compute_config = {
            "use_batch_invariant": envs.VLLM_BATCH_INVARIANT,
            "use_f16_accum": envs.VLLM_HUMMING_USE_F16_ACCUM,
            "gemm_type": self.humming_gemm_type().value,
        }
        self.w13_tuning_config = get_heuristics_config(
            layer_config=self.humming_configs["w13"],
            use_f16_accum=envs.VLLM_HUMMING_USE_F16_ACCUM,
            use_batch_invariant=envs.VLLM_BATCH_INVARIANT,
            gemm_type=self.humming_gemm_type(),
        )
        self.w2_tuning_config = get_heuristics_config(
            layer_config=self.humming_configs["w2"],
            use_f16_accum=envs.VLLM_HUMMING_USE_F16_ACCUM,
            use_batch_invariant=envs.VLLM_BATCH_INVARIANT,
            gemm_type=self.humming_gemm_type(),
        )
        self.compute_config_str = json.dumps(self.compute_config)
        self.w13_tuning_config_str = json.dumps(self.w13_tuning_config)
        self.w2_tuning_config_str = json.dumps(self.w2_tuning_config)

    def quantize_input(
        self,
        sublayer_name: str,
        inputs: torch.Tensor,
        quanted_input: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from vllm.utils.humming import may_quant_input

        return may_quant_input(
            self.humming_configs[sublayer_name],
            inputs=inputs,
            quanted_input=quanted_input,
        )

    def humming_forward(
        self,
        sublayer_name: str,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor | None,
        outputs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        from vllm.utils.humming import humming_forward

        is_w13 = sublayer_name == "w13"
        return humming_forward(
            self.humming_configs[sublayer_name],
            inputs=inputs,
            weight=weight,
            weight_scale=(
                self.quant_config.w1_scale if is_w13 else self.quant_config.w2_scale
            ),
            zero_point=(self.quant_config.w1_zp if is_w13 else self.quant_config.w2_zp),
            bias=(self.quant_config.w1_bias if is_w13 else self.quant_config.w2_bias),
            weight_scale_2=(
                self.quant_config.g1_alphas if is_w13 else self.quant_config.g2_alphas
            ),
            input_scale=input_scale,
            outputs=outputs,
            locks=self.locks,
            **kwargs,
        )

    def _get_permute_scratch(self) -> MoEPermuteScratch | None:
        if self._permute_scratch is None and moe_permute_unpermute_supported():
            self._permute_scratch = MoEPermuteScratch(
                max_num_tokens=self.moe_config.max_num_tokens,
                topk=self.moe_config.experts_per_token,
                num_experts=self.moe_config.num_experts,
                num_local_experts=self.moe_config.num_local_experts,
                device=torch.device(self.moe_config.device),
                hidden_size=self.moe_config.hidden_dim,
                hidden_dtype=self.moe_config.in_dtype,
            )
        return self._permute_scratch

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

    @staticmethod
    def humming_gemm_type() -> "HummingGemmType":
        raise NotImplementedError

    @classmethod
    def is_batched(cls) -> bool:
        return cls.activation_format() == mk.FusedMoEActivationFormat.BatchedExperts

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp4Static, None),
            (kMxfp4Static, kMxfp4Dynamic),
            (kMxfp4Static, kMxfp8Dynamic),
            (kMxfp4Static, kFp8DynamicTokenSym),
            (kNvfp4Static, None),
            (kNvfp4Static, kFp8DynamicTokenSym),
            (kMxfp8Static, None),
            (kMxfp8Static, kFp8DynamicTokenSym),
            (kFp8StaticChannelSym, None),
            (kFp8StaticChannelSym, kFp8DynamicTokenSym),
            (kFp8Static128BlockSym, None),
            (kFp8Static128BlockSym, kFp8DynamicTokenSym),
            (kInt4Static, None),
            (kInt4Static, kFp8DynamicTokenSym),
            (kInt8Static, None),
            (kInt8Static, kFp8DynamicTokenSym),
            # Checkpoint-driven (weight, activation) pairs the dense/MoE oracles
            # pass. Humming defers input quant (see expects_unquantized_inputs),
            # so the activation key does not constrain support.
            # fp8 (compressed-tensors / native / modelopt)
            (kFp8StaticChannelSym, kFp8StaticTensorSym),
            (kFp8StaticChannelSym, kFp8Dynamic128Sym),
            (kFp8StaticTensorSym, None),
            (kFp8StaticTensorSym, kFp8DynamicTokenSym),
            (kFp8StaticTensorSym, kFp8StaticTensorSym),
            (kFp8StaticTensorSym, kFp8Dynamic128Sym),
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
            # int8 (compressed-tensors w8a8 / experts_int8)
            (kInt8StaticChannelSym, None),
            (kInt8StaticChannelSym, kInt8DynamicTokenSym),
            # nvfp4 (compressed-tensors / modelopt / quark)
            (kNvfp4Static, kNvfp4Dynamic),
            # mxfp8 (compressed-tensors / modelopt / online)
            (kMxfp8Static, kMxfp8Dynamic),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @property
    def expects_unquantized_inputs(self) -> bool:
        """
        Humming kernels handle input quantization internally in apply().

        This property tells the prepare/finalize step to skip input
        quantization (by setting defer_input_quant=True) and pass
        unquantized inputs to the experts. This prevents double
        quantization: once in prepare and once in Humming's apply().

        Returns:
            True to indicate that this expert expects unquantized inputs
            and will handle quantization internally.
        """
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        platform = current_platform
        return (
            has_humming()
            and platform.is_cuda()
            and platform.has_device_capability((7, 5))
        )

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
            MoEActivation.GELU_TANH,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            MoEActivation.SWIGLUSTEP,
            MoEActivation.SILU_NO_MUL,
            MoEActivation.GELU_NO_MUL,
            MoEActivation.GELU_TANH_NO_MUL,
            MoEActivation.RELU2_NO_MUL,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    @staticmethod
    def _supports_batch_invariance() -> bool:
        return True

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        meta1 = self.humming_configs["w13"]
        meta2 = self.humming_configs["w2"]

        assert meta1.num_experts == meta2.num_experts

        num_experts = meta1.num_experts
        top_k = topk_ids.size(1)
        assert w1.size(0) == num_experts
        assert w2.size(0) == num_experts

        if not self.is_batched():
            num_tokens = a1.size(0)
            assert topk_ids.size(0) == num_tokens
        else:
            assert a1.dim() == 3
            assert a1.size(0) == num_experts
            num_tokens = a1.size(1)

        return meta1.num_experts, num_tokens, meta1.shape_n, meta1.shape_k, top_k

    def get_buffer_metas(self, M: int, topk: int, activation: MoEActivation):
        from vllm.utils.humming import GemmType as HummingGemmType
        from vllm.utils.humming import dtypes

        num_experts = self.num_experts
        gate_up_dim = self.humming_configs["w13"].shape_n
        intermediate_dim = self.humming_configs["w2"].shape_k
        K = self.humming_configs["w13"].shape_k
        assert isinstance(num_experts, int)
        assert isinstance(gate_up_dim, int)
        assert isinstance(intermediate_dim, int)
        assert isinstance(K, int)
        assert intermediate_dim == self.adjust_N_for_activation(gate_up_dim, activation)

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
        if self.is_batched():
            max_num_tokens = self.max_num_tokens
            num_dispatchers = self.num_dispatchers
            assert max_num_tokens is not None and num_dispatchers is not None
            input_shape_m = num_experts * max_num_tokens
            real_shape_m = num_experts * max_num_tokens * num_dispatchers
            output_shape = (num_experts, max_num_tokens * num_dispatchers, K)
        else:
            input_shape_m = M
            if self.humming_gemm_type() != HummingGemmType.INDEXED:
                input_shape_m = M * topk
            real_shape_m = M * topk
            output_shape = (M, K)

        a_dtype = self.humming_configs["w13"].a_dtype
        c_dtype = self.humming_configs["w13"].c_dtype
        num_bits = a_dtype.num_bits
        torch_dtype_map = {
            dtypes.float16: torch.float16,
            dtypes.bfloat16: torch.bfloat16,
            dtypes.float32: torch.float32,
            dtypes.float8e4m3: torch.float8_e4m3fn,
            dtypes.float8e5m2: torch.float8_e5m2,
            dtypes.int8: torch.int8,
            dtypes.int4: torch.uint8,
        }

        buffer_metas = {
            "quanted_gate_up_input": {
                "shape": (input_shape_m, K),
                "dtype": torch_dtype_map[a_dtype],
            },
            "gate_up_output": {
                "shape": (real_shape_m, gate_up_dim),
                "dtype": torch_dtype_map[c_dtype],
            },
            "activation_output": {
                "shape": (real_shape_m, intermediate_dim),
                "dtype": torch_dtype_map[c_dtype],
            },
            "quanted_down_input": {
                "shape": (real_shape_m, intermediate_dim),
                "dtype": torch_dtype_map[a_dtype],
            },
            "down_output": {
                "shape": output_shape if self.is_batched() else (real_shape_m, K),
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
                last_dim = meta["shape"][-1]
                if last_dim % 2 != 0:
                    raise ValueError(
                        f"Int4 packing requires last dimension to be even, "
                        f"got {last_dim} for buffer '{key}'"
                    )
                meta["shape"] = meta["shape"][:-1] + (last_dim // 2,)

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
        if not self.is_batched():
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

        output_key = "down_output" if self.is_batched() else "output"
        output_shape = buffer_metas[output_key]["shape"]
        elem_size = self.moe_config.in_dtype.itemsize

        return (
            (workspace1_nbytes // elem_size,),
            (workspace2_nbytes // elem_size,),
            output_shape,
        )

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
        torch_dtype = self.moe_config.in_dtype
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

    # Note: apply method is implemented by subclasses following the
    # standard FusedMoEExpertsModular.apply signature

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = mk.FusedMoEExpertsModular.is_supported_config(
            cls,
            moe_config,
            weight_key,
            activation_key,
            activation_format,
        )

        if supported:
            assert hasattr(cls, "humming_gemm_type")
            gemm_type = cls.humming_gemm_type().value.lower()
            preferred_gemm_type = get_humming_moe_gemm_type()
            supported = preferred_gemm_type.lower() == gemm_type
            if not supported:
                reason = (
                    f"preferred gemm type {preferred_gemm_type} != "
                    f"supported gemm type {gemm_type}"
                )

        return supported, reason

    def apply_activation(
        self,
        activation: MoEActivation,
        output: torch.Tensor,
        input: torch.Tensor,
    ) -> None:
        clamp_limit = self.quant_config.gemm1_clamp_limit

        self.activation(
            activation=activation,
            input=input,
            output=output,
            clamp_limit=clamp_limit,
            alpha=(
                self.quant_config.gemm1_alpha
                if self.quant_config.gemm1_alpha is not None
                else 1.0
            ),
            beta=(
                self.quant_config.gemm1_beta
                if self.quant_config.gemm1_beta is not None
                else 0.0
            ),
        )


class HummingIndexedExperts(HummingExpertsBase):
    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def humming_gemm_type() -> "HummingGemmType":
        from vllm.utils.humming import GemmType as HummingGemmType

        return HummingGemmType.INDEXED

    def prepare_humming_moe_kwargs(
        self,
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        valid_shape_m = self.estimate_local_valid_shape_m(topk_ids)

        moe_block_size = None
        for min_shape_m, max_shape_m, config in self.w13_tuning_config:
            if valid_shape_m > min_shape_m and valid_shape_m <= max_shape_m:
                moe_block_size = config["block_shape"][0]
                break

        if moe_block_size is None:
            logger.warning_once(
                "No tuning config found for shape %s, using default block_size=64",
                valid_shape_m,
            )
            moe_block_size = 64

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
    ) -> None:
        """
        Standard apply implementation for Humming indexed experts.

        Humming performs activation quantization internally and consumes the
        weights supplied by the modular kernel interface.
        The output is written into workspace13 via the buffer management.
        """
        assert not apply_router_weight_on_input

        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        buffers = self.prepare_buffers(
            workspace13,
            workspace2,
            topk_ids.size(0),
            topk_ids.size(1),
            activation,
        )

        moe_kwargs1, moe_kwargs2 = self.prepare_humming_moe_kwargs(
            topk_ids=topk_ids,
            expert_map=expert_map,
            expert_tokens_meta=expert_tokens_meta,
        )

        inputs, input_scale = self.quantize_input(
            "w13",
            inputs=hidden_states,
            quanted_input=buffers.get("quanted_gate_up_input", None),
        )

        self.humming_forward(
            "w13",
            inputs=inputs,
            weight=w1,
            input_scale=input_scale,
            outputs=buffers["gate_up_output"],
            **moe_kwargs1,
        )

        self.apply_activation(
            activation=activation,
            input=buffers["gate_up_output"],
            output=buffers["activation_output"],
        )

        inputs, input_scale = self.quantize_input(
            "w2",
            inputs=buffers["activation_output"],
            quanted_input=buffers.get("quanted_down_input", None),
        )

        self.humming_forward(
            "w2",
            inputs=inputs,
            weight=w2,
            input_scale=input_scale,
            outputs=buffers["down_output"].view(-1, hidden_states.size(-1)),
            **moe_kwargs2,
        )

        moe_fused_mul_sum(
            inputs=buffers["down_output"].view(*topk_ids.shape, -1),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_map=expert_map,
            outputs=buffers["output"],
        )

        # Note: output is already written to buffers["output"]
        # which aliases workspace13/output


class HummingGroupedExperts(HummingExpertsBase):
    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def humming_gemm_type() -> "HummingGemmType":
        from vllm.utils.humming import GemmType as HummingGemmType

        return HummingGemmType.GROUPED_CONTIGUOUS

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
    ) -> None:
        """
        Standard apply implementation for Humming grouped experts.

        Humming performs activation quantization internally and consumes the
        weights supplied by the modular kernel interface.
        The output is written into workspace13 via the buffer management.
        """
        assert not apply_router_weight_on_input

        valid_shape_m = self.estimate_local_valid_shape_m(topk_ids)

        buffers = self.prepare_buffers(
            workspace13,
            workspace2,
            topk_ids.size(0),
            topk_ids.size(1),
            activation,
        )

        hidden_states, _, expert_first_token_offset, inv_perm, _ = moe_permute(
            hidden_states=hidden_states,
            a1q_scale=None,
            topk_ids=topk_ids,
            n_expert=global_num_experts,
            n_local_expert=self.num_experts,
            expert_map=expert_map,
            scratch=self._get_permute_scratch(),
        )

        inputs, input_scale = self.quantize_input(
            "w13",
            inputs=hidden_states,
            quanted_input=buffers.get("quanted_gate_up_input", None),
        )

        self.humming_forward(
            "w13",
            inputs=inputs,
            weight=w1,
            input_scale=input_scale,
            outputs=buffers["gate_up_output"],
            valid_shape_m=valid_shape_m,
            expert_layout=expert_first_token_offset,
            compute_config=self.compute_config_str,
            tuning_config=self.w13_tuning_config_str,
        )

        self.apply_activation(
            activation=activation,
            input=buffers["gate_up_output"],
            output=buffers["activation_output"],
        )

        inputs, input_scale = self.quantize_input(
            "w2",
            inputs=buffers["activation_output"],
            quanted_input=buffers.get("quanted_down_input", None),
        )

        self.humming_forward(
            "w2",
            inputs=inputs,
            weight=w2,
            input_scale=input_scale,
            outputs=buffers["down_output"],
            valid_shape_m=valid_shape_m,
            expert_layout=expert_first_token_offset,
            compute_config=self.compute_config_str,
            tuning_config=self.w2_tuning_config_str,
        )

        moe_unpermute(
            out=buffers["output"],
            permuted_hidden_states=buffers["down_output"].view(*topk_ids.shape, -1),
            topk_weights=topk_weights,
            inv_permuted_idx=inv_perm,
            expert_first_token_offset=expert_first_token_offset,
        )

        # Note: output is already written to buffers["output"]
        # which aliases workspace13/output


class BatchedHummingGroupedExperts(HummingExpertsBase):
    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceDelegate()

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    @staticmethod
    def humming_gemm_type() -> "HummingGemmType":
        from vllm.utils.humming import GemmType as HummingGemmType

        return HummingGemmType.GROUPED_MASKED

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
    ) -> None:
        """
        Standard apply implementation for Humming batched grouped experts.

        Humming performs activation quantization internally and consumes the
        weights supplied by the modular kernel interface.
        The output is written into workspace13 via the buffer management.
        """
        assert not apply_router_weight_on_input
        assert expert_tokens_meta is not None

        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        valid_shape_m = self.estimate_local_valid_shape_m(topk_ids)
        expert_num_tokens = expert_tokens_meta.expert_num_tokens

        buffers = self.prepare_buffers(
            workspace13,
            workspace2,
            topk_ids.size(0),
            topk_ids.size(1),
            activation,
        )

        inputs, input_scale = self.quantize_input(
            "w13",
            inputs=hidden_states,
            quanted_input=buffers.get("quanted_gate_up_input", None),
        )

        self.humming_forward(
            "w13",
            inputs=inputs,
            weight=w1,
            input_scale=input_scale,
            outputs=buffers["gate_up_output"],
            valid_shape_m=valid_shape_m,
            expert_layout=expert_num_tokens,
            compute_config=self.compute_config_str,
            tuning_config=self.w13_tuning_config_str,
        )

        self.apply_activation(
            activation=activation,
            input=buffers["gate_up_output"],
            output=buffers["activation_output"],
        )

        inputs, input_scale = self.quantize_input(
            "w2",
            inputs=buffers["activation_output"],
            quanted_input=buffers.get("quanted_down_input", None),
        )

        self.humming_forward(
            "w2",
            inputs=inputs,
            weight=w2,
            input_scale=input_scale,
            outputs=buffers["down_output"].view(-1, hidden_states.size(-1)),
            valid_shape_m=valid_shape_m,
            expert_layout=expert_num_tokens,
            compute_config=self.compute_config_str,
            tuning_config=self.w2_tuning_config_str,
        )

        # Note: output is already written to buffers["down_output"]
        # which aliases workspace13/output
