# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4-BF16 MoE experts through vLLM's FlyDSL kernels."""

import functools
import json
from pathlib import Path

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._aiter_ops import is_aiter_found_and_supported
from vllm.kernels.flydsl.nvfp4_moe_2stages import (
    nvfp4_moe_stage1,
    nvfp4_moe_stage2,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    moe_kernel_quantize_input,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx942, on_gfx950
from vllm.utils.platform_utils import get_device_name_as_file_name

logger = init_logger(__name__)


class FlydslNvfp4Experts(mk.FusedMoEExpertsModular):
    """NVFP4-BF16 MoE experts using vLLM's FlyDSL implementation."""

    @staticmethod
    def shuffle_nvfp4_weight_for_flydsl(weight: torch.Tensor) -> torch.Tensor:
        """Preshuffle packed NVFP4 MoE weights for FlyDSL's kpack-bytes-8 layout."""
        experts, n_out, packed_k = weight.shape
        if n_out % 16 or packed_k % 32:
            raise ValueError(
                "FlyDSL NVFP4 MoE requires N to be divisible by 16 and "
                "packed K to be divisible by 32, "
                f"got shape={tuple(weight.shape)}"
            )
        flattened = weight.contiguous().view(experts * n_out, packed_k)
        shuffled = flattened.view(experts * n_out // 16, 16, packed_k // 32, 4, 8)
        return shuffled.permute(0, 2, 3, 1, 4).contiguous().view_as(weight)

    @staticmethod
    def _get_default_bf16_nvfp4_fused_moe_params(
        token: int,
        topk: int,
        expert: int,
        model_dim: int,
        inter_dim: int,
        block_m: int | None = None,
    ) -> dict[str, int]:
        """Port AITER's BF16-by-NVFP4 FlyDSL fallback tile selection."""
        if block_m is None:
            cu_num = torch.cuda.get_device_properties("cuda").multi_processor_count
            tile_n = 128
            work = []
            for candidate in (32, 64, 128):
                groups_n = (inter_dim + tile_n - 1) // tile_n
                max_tokens = token * topk + expert * candidate - topk
                groups = groups_n * (max_tokens + candidate - 1) // candidate
                work.append(
                    (
                        (groups + cu_num - 1) // cu_num,
                        cu_num - groups % cu_num,
                        candidate,
                    )
                )
            block_m = sorted(work)[0][-1]

        def select_k(dim: int) -> int:
            return next(
                k for k in (256, 128, 64) if dim % k == 0 and 4 * block_m * k <= 65536
            )

        return {
            "tile_m": block_m,
            "tile_n": next(n for n in (128, 64) if inter_dim % n == 0),
            "tile_k": select_k(model_dim),
            "k_batch": 1,
            "tile_n2": next(n for n in (128, 64) if model_dim % n == 0),
            "tile_k2": select_k(inter_dim),
        }

    @staticmethod
    @functools.lru_cache(maxsize=64)
    def _load_tuned_configs(experts: int, inter_dim: int) -> dict[int, dict[str, int]]:
        name = (
            f"E={experts},N={inter_dim},"
            f"device_name={get_device_name_as_file_name()},"
            "dtype=nvfp4_bf16,backend=flydsl.json"
        )
        path = Path(__file__).parent.parent / "configs" / name
        if not path.exists():
            return {}
        with path.open() as handle:
            raw = json.load(handle)
        return {int(key): value for key, value in raw.items()}

    @classmethod
    def _select_params(
        cls, token: int, topk: int, expert: int, model_dim: int, inter_dim: int
    ) -> dict[str, int]:
        configs = cls._load_tuned_configs(expert, inter_dim)
        padded = 1 << max(0, token - 1).bit_length()
        config = configs.get(padded)
        required = {"tile_m", "tile_n", "tile_k", "k_batch", "tile_n2", "tile_k2"}
        if config is not None and required.issubset(config):
            return config
        command = (
            "python benchmarks/kernels/benchmark_flydsl_moe_nvfp4.py "
            f"--experts {expert} --hidden-size {model_dim} "
            f"--inter-dim {inter_dim} --topk {topk}"
        )
        logger.warning_once(
            "No tuned FlyDSL NVFP4 MoE config is available for token bucket %d "
            "(E=%d, N=%d, device=%s). Run:\n%s\n"
            "The generated config will be picked up automatically on the next run.",
            padded,
            expert,
            inter_dim,
            get_device_name_as_file_name(),
            command,
        )
        return cls._get_default_bf16_nvfp4_fused_moe_params(
            token, topk, expert, model_dim, inter_dim
        )

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        self.w1_scale_val = quant_config.w1_scale
        self.w2_scale_val = quant_config.w2_scale
        self.w1_global_scale = quant_config.g1_alphas
        self.w2_global_scale = quant_config.g2_alphas

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @property
    def expects_unquantized_inputs(self) -> bool:
        # FlyDSL NVFP4-BF16 consumes BF16 activations and NVFP4 weights.
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_rocm()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kNvfp4Static, kNvfp4Dynamic)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        if moe_config.is_lora_enabled:
            return False, "kernel does not support LoRA"
        if moe_config.has_bias:
            return False, "kernel does not support bias"

        is_supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls,
            moe_config,
            weight_key,
            activation_key,
            activation_format,
        )

        if not is_supported:
            return is_supported, reason

        if moe_config.in_dtype != torch.bfloat16:
            return False, "kernel only supports bfloat16 activations"

        if not current_platform.is_rocm() or not (on_gfx950() or on_gfx942()):
            return False, "kernel available only on AMD gfx950 devices for now"

        if not is_aiter_found_and_supported():
            return (
                False,
                "kernel requires aiter library (not found in user environment)",
            )

        return is_supported, reason

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

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
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        return (M, topk, activation_out_dim), (0,), (M, K)

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
        from aiter.fused_moe import moe_sorting

        assert activation == MoEActivation.SILU
        assert hidden_states.dtype == torch.bfloat16
        assert w1.dtype == torch.uint8
        assert w2.dtype == torch.uint8
        assert self.w1_scale_val is not None
        assert self.w2_scale_val is not None
        assert self.w1_global_scale is not None
        assert self.w2_global_scale is not None

        E, num_tokens, N, K, topk = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )
        inter_dim = self.adjust_N_for_activation(N, activation)
        is_g1u1 = inter_dim != w1.shape[1]

        if expert_tokens_meta is not None:
            num_local_tokens = expert_tokens_meta.expert_num_tokens
        else:
            num_local_tokens = None

        if expert_map is not None:
            local_mask = (expert_map >= 0).to(torch.int32)
            expert_mask = torch.cat([local_mask, local_mask.new_zeros(1)])
        else:
            expert_mask = None

        if not is_g1u1:
            raise NotImplementedError("FlyDSL NVFP4 experts require gated MoE weights")

        params = self._select_params(num_tokens, topk, E, K, inter_dim)
        block_m = params["tile_m"]
        global_num_experts_for_sort = (
            expert_mask.numel() if expert_mask is not None else E
        )
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
            topk_ids.to(torch.int32),
            topk_weights.to(torch.float32),
            global_num_experts_for_sort,
            K,
            output.dtype,
            block_m,
            expert_mask,
            num_local_tokens,
        )

        hidden_states_qdq, _ = moe_kernel_quantize_input(
            A=hidden_states,
            A_scale=self.quant_config.a1_gscale,
            quant_dtype="nvfp4",
            per_act_token_quant=False,
            quantization_emulation=True,
        )

        intermediate = _resize_cache(workspace13, (num_tokens, topk, inter_dim))
        nvfp4_moe_stage1(
            hidden_states_qdq,
            w1,
            self.w1_scale_val,
            self.w1_global_scale,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            topk=topk,
            inter_dim=inter_dim,
            tile_m=params["tile_m"],
            tile_n=params["tile_n"],
            tile_k=params["tile_k"],
            k_batch=params["k_batch"],
            output=intermediate,
        )

        intermediate_qdq, _ = moe_kernel_quantize_input(
            A=intermediate.view(-1, inter_dim),
            A_scale=self.quant_config.a2_gscale,
            quant_dtype="nvfp4",
            per_act_token_quant=False,
            quantization_emulation=True,
        )
        intermediate_qdq = intermediate_qdq.view(num_tokens, topk, inter_dim)

        output.zero_()
        nvfp4_moe_stage2(
            intermediate_qdq,
            w2,
            self.w2_scale_val,
            self.w2_global_scale,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            topk=topk,
            model_dim=K,
            tile_m=params["tile_m"],
            tile_n=params["tile_n2"],
            tile_k=params["tile_k2"],
            output=output,
            sorted_weights=sorted_weights if not apply_router_weight_on_input else None,
        )
