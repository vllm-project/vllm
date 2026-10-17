# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU MoE implementation for MuseSpark1x checkpoints."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._custom_ops import CPUQuantMethod, cpu_fused_moe, fused_experts_cpu
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
    CPUExpertsInt4,
    CPUUnquantizedExperts,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQMoEMethod
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    resolve_quant_method,
)
from vllm.model_executor.layers.quantization.inc.inc import INCConfig
from vllm.models.muse_spark.common.moe import MuseSpark1xRouter


def _validate_tp_post_norm_deployment(moe_config: FusedMoEConfig) -> None:
    parallel_config = moe_config.moe_parallel_config
    if parallel_config.use_ep:
        raise NotImplementedError(
            "MuseSpark1x on CPU currently supports tensor-parallel experts, "
            "not expert parallelism."
        )
    if parallel_config.dp_size != 1 or parallel_config.pcp_size != 1:
        raise NotImplementedError("MuseSpark1x on CPU currently requires DP1 and PCP1.")
    if parallel_config.is_sequence_parallel:
        raise NotImplementedError(
            "MuseSpark1x on CPU does not support sequence-parallel MoE."
        )
    if parallel_config.enable_eplb:
        raise NotImplementedError("MuseSpark1x on CPU does not support EPLB.")
    if moe_config.is_lora_enabled:
        raise NotImplementedError("MuseSpark1x on CPU does not support MoE LoRA.")
    if moe_config.skip_final_all_reduce:
        raise NotImplementedError(
            "MuseSpark1x on CPU performs the TP reduction before its "
            "post-expert norm and cannot return an unreduced MoE output."
        )


def validate_gptq_int4_deployment(moe_config: FusedMoEConfig) -> None:
    """Reject deployments whose GPTQ-int4 finalization is not implemented."""
    _validate_tp_post_norm_deployment(moe_config)


@dataclass
class MuseSparkGPTQMoEQuantConfig(FusedMoEQuantConfig):
    post_expert_norm_fn: Callable[..., torch.Tensor] | None = None


@dataclass
class MuseSparkUnquantizedMoEQuantConfig(FusedMoEQuantConfig):
    post_expert_norm_fn: Callable[..., torch.Tensor] | None = None


class MuseSparkCPUTopKWeightAndReduce(mk.TopKWeightAndReduce):
    """Apply MuseSpark's per-route norm before router weighting and reduction."""

    def __init__(
        self,
        post_expert_norm_fn: Callable[..., torch.Tensor],
        *,
        pre_norm_all_reduce: bool,
    ) -> None:
        self.post_expert_norm_fn = post_expert_norm_fn
        self.pre_norm_all_reduce = pre_norm_all_reduce

    def apply(
        self,
        output: torch.Tensor | None,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "MuseSpark1x requires router weights after the post-expert norm."
            )

        num_tokens, top_k = topk_ids.shape
        hidden_dim = fused_expert_output.shape[-1]
        routed = fused_expert_output.view(num_tokens, top_k, hidden_dim)

        if self.pre_norm_all_reduce:
            routed = tensor_model_parallel_all_reduce(routed.float())

        normalized = self.post_expert_norm_fn(routed, out_dtype=torch.float32)
        normalized.mul_(topk_weights.float().unsqueeze(-1))
        reduced = normalized.sum(dim=1)

        if output is None:
            output = torch.empty(
                (num_tokens, hidden_dim),
                dtype=fused_expert_output.dtype,
                device=fused_expert_output.device,
            )
        output.copy_(reduced.to(output.dtype))
        return output


class CPUMuseSparkExpertsInt4(CPUExpertsInt4):
    """CPU INT4 W4A16 experts that hand back raw per-route rows.

    Uses ``fused_experts_cpu(..., defer_finalize=True)`` to skip the kernel's
    topk-weighted reduce, since MuseSpark1x needs to apply a post-expert
    RMSNorm before that reduce happens.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ) -> None:
        super().__init__(moe_config, quant_config, max_num_tokens, num_dispatchers)
        if not isinstance(quant_config, MuseSparkGPTQMoEQuantConfig):
            raise TypeError(
                "CPUMuseSparkExpertsInt4 requires a MuseSparkGPTQMoEQuantConfig."
            )
        if quant_config.post_expert_norm_fn is None:
            raise ValueError("MuseSpark1x post-expert norm is not configured.")

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
        workspace13, workspace2, _ = super().workspace_shapes(
            M,
            N,
            K,
            topk,
            global_num_experts,
            local_num_experts,
            expert_tokens_meta,
            activation,
        )
        return workspace13, workspace2, (M, topk, K)

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        quant_config = self.quant_config
        assert isinstance(quant_config, MuseSparkGPTQMoEQuantConfig)
        assert quant_config.post_expert_norm_fn is not None
        return MuseSparkCPUTopKWeightAndReduce(
            quant_config.post_expert_norm_fn,
            pre_norm_all_reduce=self.moe_config.moe_parallel_config.tp_size > 1,
        )

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
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "CPUMuseSparkExpertsInt4 does not support "
                "apply_router_weight_on_input=True."
            )

        fused_experts_cpu(
            output,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            CPUQuantMethod.INT4_W4A8,
            self.w1_scale,
            self.w2_scale,
            self.w1_zp,
            self.w2_zp,
            None,  # block_size
            None,  # w1_bias
            None,  # w2_bias
            None,  # alpha
            None,  # limit
            is_vnni=True,
            defer_finalize=True,
        )


_MUSE_SPARK_UNQUANTIZED_EXPERTS_CACHE: dict[
    type[CPUUnquantizedExperts], type[CPUUnquantizedExperts]
] = {}


def _muse_spark_unquantized_experts_cls(
    base_cls: type[CPUUnquantizedExperts],
) -> type[CPUUnquantizedExperts]:
    """Build (and cache) a MuseSpark-aware subclass of *base_cls* that hands
    back raw per-route rows via ``cpu_fused_moe(..., defer_finalize=True)``,
    instead of *base_cls*'s usual topk-weighted reduce.
    """
    cached = _MUSE_SPARK_UNQUANTIZED_EXPERTS_CACHE.get(base_cls)
    if cached is not None:
        return cached

    class _MuseSparkUnquantizedExperts(base_cls):
        def __init__(
            self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig
        ) -> None:
            super().__init__(moe_config, quant_config)
            if not isinstance(quant_config, MuseSparkUnquantizedMoEQuantConfig):
                raise TypeError(
                    "MuseSpark unquantized CPU experts require a "
                    "MuseSparkUnquantizedMoEQuantConfig."
                )
            if quant_config.post_expert_norm_fn is None:
                raise ValueError("MuseSpark1x post-expert norm is not configured.")

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
            workspace13, workspace2, _ = super().workspace_shapes(
                M,
                N,
                K,
                topk,
                global_num_experts,
                local_num_experts,
                expert_tokens_meta,
                activation,
            )
            return workspace13, workspace2, (M, topk, K)

        def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
            quant_config = self.quant_config
            assert isinstance(quant_config, MuseSparkUnquantizedMoEQuantConfig)
            assert quant_config.post_expert_norm_fn is not None
            return MuseSparkCPUTopKWeightAndReduce(
                quant_config.post_expert_norm_fn,
                pre_norm_all_reduce=self.moe_config.moe_parallel_config.tp_size > 1,
            )

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
            if apply_router_weight_on_input:
                raise NotImplementedError(
                    "MuseSpark unquantized CPU experts do not support "
                    "apply_router_weight_on_input=True."
                )
            if self._use_zentorch:
                raise NotImplementedError(
                    "MuseSpark1x unquantized MoE does not support the "
                    "zentorch backend."
                )
            cpu_fused_moe(
                output,
                hidden_states,
                w1,
                w2,
                self.w1_bias,
                self.w2_bias,
                topk_weights,
                topk_ids,
                activation.value,
                self.isa,
                defer_finalize=True,
            )

    _MuseSparkUnquantizedExperts.__name__ = f"MuseSpark{base_cls.__name__}"
    _MuseSparkUnquantizedExperts.__qualname__ = _MuseSparkUnquantizedExperts.__name__
    _MUSE_SPARK_UNQUANTIZED_EXPERTS_CACHE[base_cls] = _MuseSparkUnquantizedExperts
    return _MuseSparkUnquantizedExperts


class MuseSparkCPUAutoGPTQMoEMethod(AutoGPTQMoEMethod):
    """AutoGPTQ MoE method pinned to the MuseSpark-aware CPU experts."""

    def __init__(self, quant_config, moe: FusedMoEConfig) -> None:
        validate_gptq_int4_deployment(moe)
        super().__init__(quant_config, moe)
        from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
            WNA16MoEBackend,
        )

        if self.wna16_moe_backend != WNA16MoEBackend.CPU:
            raise NotImplementedError(
                "MuseSpark1x GPTQ-int4 MoE is only implemented for the CPU "
                f"WNA16 backend; got {self.wna16_moe_backend.value}."
            )
        self.experts_cls = CPUMuseSparkExpertsInt4

    def get_fused_moe_quant_config(self, layer: RoutedExperts) -> FusedMoEQuantConfig:
        base_config = super().get_fused_moe_quant_config(layer)
        post_expert_norm_fn = getattr(layer, "post_expert_norm_fn", None)
        if post_expert_norm_fn is None:
            raise RuntimeError("MuseSpark1x post-expert norm is not configured.")
        return MuseSparkGPTQMoEQuantConfig(
            **vars(base_config),
            post_expert_norm_fn=post_expert_norm_fn,
        )


class MuseSparkCPUUnquantizedMoEMethod(UnquantizedFusedMoEMethod):
    """Unquantized MoE method pinned to the MuseSpark-aware CPU experts."""

    def __init__(self, moe: FusedMoEConfig) -> None:
        _validate_tp_post_norm_deployment(moe)
        super().__init__(moe)
        self.experts_cls = _muse_spark_unquantized_experts_cls(self.experts_cls)

    def get_fused_moe_quant_config(self, layer: RoutedExperts) -> FusedMoEQuantConfig:
        base_config = super().get_fused_moe_quant_config(layer)
        post_expert_norm_fn = getattr(layer, "post_expert_norm_fn", None)
        if post_expert_norm_fn is None:
            raise RuntimeError("MuseSpark1x post-expert norm is not configured.")
        return MuseSparkUnquantizedMoEQuantConfig(
            **vars(base_config),
            post_expert_norm_fn=post_expert_norm_fn,
        )


class CPUMuseSparkRoutedExperts(RoutedExperts):
    """CPU routed experts using an INC/AutoRound GPTQ-int4 MoE method.

    This checkpoint family stores each expert's gate/up/down projections
    under their own name (``gate_proj``/``up_proj``/``down_proj``) rather
    than the consolidated/stacked ``w13_weight``/``w2_weight`` tensors the
    NVIDIA NVFP4 experts load, so this inherits the plain ``RoutedExperts``
    loader instead of ``MuseSparkConsolidatedRoutedExperts``.
    """

    def _get_quant_method(
        self,
        prefix: str,
        quant_config: QuantizationConfig | None,
        moe_config: FusedMoEConfig,
    ):
        if not isinstance(quant_config, INCConfig):
            got = quant_config.get_name() if quant_config else None
            raise NotImplementedError(
                "MuseSpark1x on CPU requires an INC/AutoRound GPTQ-int4 MoE "
                f"checkpoint; got {got!r}."
            )
        resolved = resolve_quant_method(quant_config, self, prefix)
        if isinstance(resolved, AutoGPTQMoEMethod):
            return MuseSparkCPUAutoGPTQMoEMethod(resolved.quant_config, moe_config)
        if isinstance(resolved, UnquantizedFusedMoEMethod):
            return MuseSparkCPUUnquantizedMoEMethod(moe_config)
        raise NotImplementedError(
            "MuseSpark1x on CPU requires GPTQ-style int4 or fully-unquantized "
            f"MoE weights; got {type(resolved).__name__}."
        )


class MuseMoERunnerCPU(MoERunner):
    """MoE runner whose custom reducer owns the TP reduction."""

    def __init__(
        self,
        *args,
        post_expert_norm: torch.nn.Module,
        fuse_all_reduce: bool = False,
        **kwargs,
    ) -> None:
        del fuse_all_reduce  # GPU-only MegaMoE all-reduce fusion; no-op on CPU.
        super().__init__(*args, **kwargs)
        if not isinstance(self.routed_experts, CPUMuseSparkRoutedExperts):
            raise TypeError(
                "MuseMoERunnerCPU requires "
                "routed_experts_cls=CPUMuseSparkRoutedExperts."
            )
        self.routed_experts.post_expert_norm_fn = post_expert_norm.forward

    @property
    def _fused_output_is_reduced(self) -> bool:
        return True


__all__ = [
    "CPUMuseSparkExpertsInt4",
    "CPUMuseSparkRoutedExperts",
    "MuseMoERunnerCPU",
    "MuseSpark1xRouter",
    "MuseSparkCPUAutoGPTQMoEMethod",
    "MuseSparkCPUTopKWeightAndReduce",
    "MuseSparkCPUUnquantizedMoEMethod",
    "validate_gptq_int4_deployment",
]
