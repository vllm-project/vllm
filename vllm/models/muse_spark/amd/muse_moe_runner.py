# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm MoE implementation for MuseSpark1x checkpoints."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields

import torch
import torch.distributed as dist

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
    AiterMxfp4Experts,
    rocm_aiter_fused_experts,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    QuarkOCP_MX_MoEMethod,
    QuarkW8A8Fp8MoEMethod,
)
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
    _ACTIVATION_QUANT_KEY_MAP,
    _WEIGHT_QUANT_KEY_MAP,
    OCP_MX_BLOCK_SIZE,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.model_executor.models.utils import extract_layer_index
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.muse_spark.amd.ops.fused_norm_expand import take_expanded
from vllm.models.muse_spark.common.layers import MuseSpark1xRMSNorm
from vllm.models.muse_spark.common.moe import (
    MuseSpark1xRouter,
    MuseSparkConsolidatedRoutedExperts,
    needs_pre_norm_all_reduce,
)
from vllm.models.muse_spark.common.ops.fused_finalize import (
    fused_post_expert_finalize,
)

_FINALIZE_CHUNK_ELEMS = 32 * 1024 * 1024
# Below this, one custom all-reduce beats the two-RCCL-kernel RS + AG pair.
_TOKEN_SHARDED_MIN_TOKENS = 256


def _validate_tp_post_norm_deployment(
    moe_config: FusedMoEConfig, precision: str
) -> None:
    parallel_config = moe_config.moe_parallel_config
    if parallel_config.dp_size != 1 or parallel_config.pcp_size != 1:
        raise NotImplementedError(
            f"MuseSpark1x {precision} on ROCm currently requires DP1 and PCP1."
        )
    if parallel_config.is_sequence_parallel:
        raise NotImplementedError(
            f"MuseSpark1x {precision} on ROCm does not support sequence-parallel MoE."
        )
    if parallel_config.enable_eplb:
        raise NotImplementedError(
            f"MuseSpark1x {precision} on ROCm does not support EPLB."
        )
    if moe_config.is_lora_enabled:
        raise NotImplementedError(
            f"MuseSpark1x {precision} on ROCm does not support MoE LoRA."
        )
    if moe_config.skip_final_all_reduce:
        raise NotImplementedError(
            f"MuseSpark1x {precision} performs the TP reduction before its post-expert "
            "norm and cannot return an unreduced MoE output."
        )


def validate_bf16_deployment(moe_config: FusedMoEConfig) -> None:
    """Reject deployments whose BF16 finalization is not implemented."""
    _validate_tp_post_norm_deployment(moe_config, "BF16")
    if moe_config.moe_backend not in ("auto", "triton"):
        raise NotImplementedError(
            "MuseSpark1x BF16 on ROCm requires the Triton MoE backend; got "
            f"{moe_config.moe_backend!r}."
        )


def validate_mxfp4_deployment(moe_config: FusedMoEConfig) -> None:
    """Reject deployments whose MXFP4 finalization is not implemented."""
    _validate_tp_post_norm_deployment(moe_config, "MXFP4")


def validate_fp8_deployment(moe_config: FusedMoEConfig) -> None:
    """Reject deployments whose FP8 finalization is not implemented."""
    _validate_tp_post_norm_deployment(moe_config, "FP8")
    if moe_config.moe_backend != "triton":
        raise NotImplementedError(
            "MuseSpark1x FP8 on ROCm requires the Triton MoE backend; got "
            f"{moe_config.moe_backend!r}."
        )


@dataclass
class MuseSparkUnquantizedMoEQuantConfig(FusedMoEQuantConfig):
    post_expert_norm_fn: Callable[..., torch.Tensor] | None = None


@dataclass
class MuseSparkQuarkFp8MoEQuantConfig(FusedMoEQuantConfig):
    post_expert_norm_fn: Callable[..., torch.Tensor] | None = None


@dataclass
class MuseSparkQuarkOCPMXMoEQuantConfig(FusedMoEQuantConfig):
    post_expert_norm_fn: Callable[..., torch.Tensor] | None = None


class MuseSparkTopKWeightAndReduce(mk.TopKWeightAndReduce):
    """Apply MuseSpark's per-route norm before router weighting and reduction."""

    def __init__(
        self,
        post_expert_norm_fn: Callable[..., torch.Tensor],
        *,
        pre_norm_all_reduce: bool,
        take_route_output: Callable[[], torch.Tensor | None] | None = None,
        take_route_expert_map: Callable[[], torch.Tensor | None] | None = None,
    ) -> None:
        self.post_expert_norm_fn = post_expert_norm_fn
        self.pre_norm_all_reduce = pre_norm_all_reduce
        self.take_route_output = take_route_output
        self.take_route_expert_map = take_route_expert_map
        self.reduce_dtype = (
            torch.bfloat16
            if envs.VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE == "bf16"
            else torch.float32
        )
        self.rs_min_tokens = _TOKEN_SHARDED_MIN_TOKENS
        norm = getattr(post_expert_norm_fn, "__self__", None)
        self.fused_norm = (
            norm
            if isinstance(norm, MuseSpark1xRMSNorm)
            and norm.variance_size_override is None
            else None
        )

    @staticmethod
    def _chunk_size(top_k: int, hidden_dim: int) -> int:
        elems_per_token = max(top_k * hidden_dim, 1)
        return max(1, _FINALIZE_CHUNK_ELEMS // elems_per_token)

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

        if self.take_route_output is not None:
            route_output = self.take_route_output()
            if route_output is not None:
                fused_expert_output = route_output
        expert_map = (
            self.take_route_expert_map()
            if self.take_route_expert_map is not None
            else None
        )
        num_tokens, top_k = topk_ids.shape
        hidden_dim = fused_expert_output.shape[-1]
        routed = fused_expert_output.view(num_tokens, top_k, hidden_dim)

        if output is None:
            output = torch.empty(
                (num_tokens, hidden_dim),
                dtype=routed.dtype,
                device=routed.device,
            )
        if output.shape != (num_tokens, hidden_dim):
            raise ValueError(
                "MuseSpark1x MoE output has shape "
                f"{tuple(output.shape)}, expected {(num_tokens, hidden_dim)}."
            )
        if expert_map is not None and (
            self.pre_norm_all_reduce or self.fused_norm is None
        ):
            # Only the fused finalize masks unwritten non-local routes itself.
            routed.masked_fill_((expert_map[topk_ids.long()] < 0).unsqueeze(-1), 0)
            expert_map = None

        token_sharded = (
            self.pre_norm_all_reduce
            and self.fused_norm is not None
            and (tp_size := get_tensor_model_parallel_world_size()) > 1
        )
        chunk_size = max_padded_tokens = self._chunk_size(top_k, hidden_dim)
        if token_sharded and routed.dtype == self.reduce_dtype:
            # The chunks only bound the reduce-dtype copy of ``routed``; with no
            # copy, one reduce-scatter and all-gather beat one pair per chunk.
            chunk_size = max(num_tokens, 1)
        for start in range(0, num_tokens, chunk_size):
            end = min(start + chunk_size, num_tokens)
            tail = start
            if token_sharded and end - start >= self.rs_min_tokens:
                tail = end
                if end - start > max_padded_tokens:
                    # Padding copies the whole chunk; past one default chunk
                    # the few tokens beyond a TP multiple take the all-reduce.
                    tail -= (end - start) % tp_size
                self._token_sharded_finalize(
                    routed[start:tail].to(self.reduce_dtype),
                    topk_weights[start:tail],
                    output[start:tail],
                    tp_size,
                )
                if tail == end:
                    continue
            rows = routed[tail:end]
            if self.pre_norm_all_reduce:
                rows = tensor_model_parallel_all_reduce(rows.to(self.reduce_dtype))

            if self.fused_norm is not None:
                fused_post_expert_finalize(
                    rows,
                    self.fused_norm.weight if self.fused_norm.has_weight else None,
                    topk_weights[tail:end],
                    self.fused_norm.variance_epsilon,
                    self.fused_norm.fused_gain_center,
                    self.fused_norm.applies_weight_before_norm,
                    output[tail:end],
                    topk_ids=None if expert_map is None else topk_ids[tail:end],
                    expert_map=expert_map,
                )
                continue
            normalized = self.post_expert_norm_fn(rows, out_dtype=torch.float32)
            normalized.mul_(topk_weights[tail:end].float().unsqueeze(-1))
            output[tail:end].copy_(normalized.sum(dim=1).to(output.dtype))

        return output

    def _token_sharded_finalize(
        self,
        rows: torch.Tensor,
        topk_weights: torch.Tensor,
        out: torch.Tensor,
        tp_size: int,
    ) -> None:
        num_tokens, top_k, hidden_dim = rows.shape
        pad = (-num_tokens) % tp_size
        if pad:
            rows = torch.cat([rows, rows.new_zeros((pad, top_k, hidden_dim))])
            topk_weights = torch.cat(
                [topk_weights, topk_weights.new_zeros((pad, top_k))]
            )
        local = tensor_model_parallel_reduce_scatter(rows, dim=0)
        per_rank = local.shape[0]
        rank = get_tensor_model_parallel_rank()
        tw_local = topk_weights[rank * per_rank : (rank + 1) * per_rank]
        gather_in_place = not pad and out.is_contiguous()
        if gather_in_place:
            out_local = out[rank * per_rank : (rank + 1) * per_rank]
        else:
            out_local = torch.empty(
                (per_rank, hidden_dim), dtype=out.dtype, device=out.device
            )
        norm = self.fused_norm
        assert norm is not None
        fused_post_expert_finalize(
            local,
            norm.weight if norm.has_weight else None,
            tw_local,
            norm.variance_epsilon,
            norm.fused_gain_center,
            norm.applies_weight_before_norm,
            out_local,
        )
        if gather_in_place:
            dist.all_gather_into_tensor(
                out, out_local, group=get_tp_group().device_group
            )
            return
        gathered = tensor_model_parallel_all_gather(out_local, dim=0)
        out.copy_(gathered[:num_tokens])


class _RouteOutputHandoff:
    """Hand the experts' per-route buffer to the finalize instead of copying it."""

    _route_output: torch.Tensor | None = None
    _route_expert_map: torch.Tensor | None = None

    def _park_route_output(
        self, route_output: torch.Tensor, expert_map: torch.Tensor | None = None
    ) -> None:
        self._route_output = route_output
        self._route_expert_map = expert_map

    def _take_route_output(self) -> torch.Tensor | None:
        route_output, self._route_output = self._route_output, None
        return route_output

    def _take_route_expert_map(self) -> torch.Tensor | None:
        expert_map, self._route_expert_map = self._route_expert_map, None
        return expert_map


class MuseSparkTritonExperts(_RouteOutputHandoff, TritonExperts):
    """Triton experts that expose unweighted GEMM2 output per route."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ) -> None:
        super().__init__(moe_config, quant_config)
        if not isinstance(
            quant_config,
            (MuseSparkUnquantizedMoEQuantConfig, MuseSparkQuarkFp8MoEQuantConfig),
        ):
            raise TypeError(
                "MuseSparkTritonExperts requires a MuseSpark Triton quant config."
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

    def _apply_router_weight_in_gemm2(self, apply_router_weight_on_input: bool) -> bool:
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "MuseSpark1x requires router weights after the post-expert norm."
            )
        return False

    def _finalize_expert_output(
        self,
        expert_output: torch.Tensor,
        output: torch.Tensor,
        topk_ids: torch.Tensor | None = None,
        expert_map: torch.Tensor | None = None,
    ) -> None:
        # Under EP the GEMM skips non-local routes, leaving their rows unwritten.
        self._park_route_output(expert_output, expert_map)

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        quant_config = self.quant_config
        assert isinstance(
            quant_config,
            (MuseSparkUnquantizedMoEQuantConfig, MuseSparkQuarkFp8MoEQuantConfig),
        )
        assert quant_config.post_expert_norm_fn is not None
        return MuseSparkTopKWeightAndReduce(
            quant_config.post_expert_norm_fn,
            pre_norm_all_reduce=self.moe_config.moe_parallel_config.tp_size > 1,
            take_route_output=self._take_route_output,
            take_route_expert_map=self._take_route_expert_map,
        )


class MuseSparkUnquantizedFusedMoE(UnquantizedFusedMoEMethod):
    """BF16 MoE method pinned to the MuseSpark-aware Triton experts."""

    def __init__(self, moe: FusedMoEConfig) -> None:
        validate_bf16_deployment(moe)
        # Triton-only: skip the generic constructor's backend oracle.
        FusedMoEMethodBase.__init__(self, moe)
        self.unquantized_backend = UnquantizedMoeBackend.TRITON
        self.experts_cls = MuseSparkTritonExperts

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        base_config = super().get_fused_moe_quant_config(layer)
        post_expert_norm_fn = getattr(layer, "post_expert_norm_fn", None)
        if post_expert_norm_fn is None:
            raise RuntimeError("MuseSpark1x post-expert norm is not configured.")
        return MuseSparkUnquantizedMoEQuantConfig(
            **{
                f.name: getattr(base_config, f.name)
                for f in fields(FusedMoEQuantConfig)
                if f.init
            },
            post_expert_norm_fn=post_expert_norm_fn,
        )


class MuseSparkAiterExperts(_RouteOutputHandoff, AiterMxfp4Experts):
    """AITER experts that return one unweighted output per selected route."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ) -> None:
        super().__init__(moe_config, quant_config)
        if not isinstance(quant_config, MuseSparkQuarkOCPMXMoEQuantConfig):
            raise TypeError(
                "MuseSparkAiterExperts requires MuseSparkQuarkOCPMXMoEQuantConfig."
            )
        if quant_config.post_expert_norm_fn is None:
            raise ValueError("MuseSpark1x post-expert norm is not configured.")
        self._unit_weights_bufs: dict[int, torch.Tensor] = {}

    def _unit_weights(
        self, num_routes: int, width: int, device: torch.device
    ) -> torch.Tensor:
        """A cached ``[num_routes, width]`` fp32 ones buffer (grown on demand)."""
        buf = self._unit_weights_bufs.get(width)
        if buf is None or buf.shape[0] < num_routes or buf.device != device:
            buf = torch.ones(
                (max(num_routes, 1), width), dtype=torch.float32, device=device
            )
            self._unit_weights_bufs[width] = buf
        return buf[:num_routes]

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
                "MuseSpark1x requires router weights after the post-expert norm."
            )

        num_tokens, top_k = topk_ids.shape
        hidden_dim = hidden_states.shape[-1]
        expected_output_shape = (num_tokens, top_k, hidden_dim)
        if output.shape != expected_output_shape:
            raise ValueError(
                "MuseSpark1x AITER output has shape "
                f"{tuple(output.shape)}, expected {expected_output_shape}."
            )

        expanded_hidden_states = take_expanded(hidden_states, top_k)
        if expanded_hidden_states is None:
            expanded_hidden_states = (
                hidden_states[:, None, :]
                .expand(num_tokens, top_k, hidden_dim)
                .reshape(num_tokens * top_k, hidden_dim)
                .contiguous()
            )
        # Only the shape of topk_ids is read after the experts.
        num_routes = num_tokens * top_k
        flat_topk_ids = topk_ids.view(num_routes, 1)
        if expert_map is not None:
            # AITER's EP convention: one extra route per row on the always-masked
            # last expert_mask slot; its tuned-config lookup subtracts it.
            masked_slot = flat_topk_ids.new_full(
                (num_routes, 1), expert_map.shape[0] - 1
            )
            flat_topk_ids = torch.cat((flat_topk_ids, masked_slot), dim=1)
        unit_topk_weights = self._unit_weights(
            num_routes, flat_topk_ids.shape[1], topk_ids.device
        )

        if expert_tokens_meta is not None:
            num_local_tokens = expert_tokens_meta.expert_num_tokens
        else:
            num_local_tokens = None

        route_output = rocm_aiter_fused_experts(
            hidden_states=expanded_hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=unit_topk_weights,
            topk_ids=flat_topk_ids,
            activation=activation,
            expert_mask=expert_map,
            quant_config=self.quant_config,
            moe_config=self.moe_config,
            a1q_scale=a1q_scale,
            num_local_tokens=num_local_tokens,
            output_dtype=output.dtype,
            moe_sorting_dispatch_policy=rocm_aiter_ops.get_moe_dispatch_policy(),
            apply_router_weight_on_input=False,
        )
        self._park_route_output(route_output.view_as(output))

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        quant_config = self.quant_config
        assert isinstance(quant_config, MuseSparkQuarkOCPMXMoEQuantConfig)
        assert quant_config.post_expert_norm_fn is not None
        return MuseSparkTopKWeightAndReduce(
            quant_config.post_expert_norm_fn,
            pre_norm_all_reduce=self.moe_config.moe_parallel_config.tp_size > 1,
            take_route_output=self._take_route_output,
        )


class MuseSparkQuarkW8A8Fp8FusedMoE(QuarkW8A8Fp8MoEMethod):
    """Quark W8A8 FP8 method using route-preserving Triton experts."""

    def __init__(
        self,
        moe: FusedMoEConfig,
        weight_quant_key: QuantKey,
        activation_quant_key: QuantKey | None,
    ) -> None:
        validate_fp8_deployment(moe)
        super().__init__(moe, weight_quant_key, activation_quant_key)
        if self.fp8_backend is not Fp8MoeBackend.TRITON:
            raise NotImplementedError(
                "MuseSpark1x FP8 on ROCm requires the Triton FP8 MoE backend; "
                f"got {self.fp8_backend.value}."
            )
        self.experts_cls = MuseSparkTritonExperts

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> MuseSparkQuarkFp8MoEQuantConfig:
        base_config = super().get_fused_moe_quant_config(layer)
        post_expert_norm_fn = getattr(layer, "post_expert_norm_fn", None)
        if post_expert_norm_fn is None:
            raise RuntimeError("MuseSpark1x post-expert norm is not configured.")
        return MuseSparkQuarkFp8MoEQuantConfig(
            **{
                f.name: getattr(base_config, f.name)
                for f in fields(FusedMoEQuantConfig)
                if f.init
            },
            post_expert_norm_fn=post_expert_norm_fn,
        )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        loaded_slots = getattr(layer, "_quark_expert_loaded_slots", None)
        if loaded_slots is not None:
            _validate_quark_expert_weights(
                loaded_slots, layer.moe_config.num_logical_experts
            )
            del layer._quark_expert_loaded_slots
        super().process_weights_after_loading(layer)


class MuseSparkQuarkOCPMXFusedMoE(QuarkOCP_MX_MoEMethod):
    """Quark W4A4 method using route-preserving stock AITER experts."""

    def __init__(
        self,
        moe: FusedMoEConfig,
        weight_quant_key: QuantKey,
        activation_quant_key: QuantKey | None,
    ) -> None:
        validate_mxfp4_deployment(moe)
        super().__init__(moe, weight_quant_key, activation_quant_key)
        if self.ocp_mx_scheme != "w_mxfp4_a_mxfp4":
            raise NotImplementedError(
                "MuseSpark1x MXFP4 on ROCm requires Quark OCP W4A4 with "
                f"dynamic MXFP4 activations; got {self.ocp_mx_scheme!r}."
            )
        if self.mxfp4_backend is not Mxfp4MoeBackend.AITER_MXFP4_MXFP4:
            raise NotImplementedError(
                "MuseSpark1x MXFP4 on ROCm requires the "
                "AITER_MXFP4_MXFP4 backend; got "
                f"{self.mxfp4_backend.value}."
            )
        self.experts_cls = MuseSparkAiterExperts

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> MuseSparkQuarkOCPMXMoEQuantConfig:
        base_config = super().get_fused_moe_quant_config(layer)
        if base_config is None:
            raise RuntimeError("MuseSpark1x MXFP4 quant config is not initialized.")
        post_expert_norm_fn = getattr(layer, "post_expert_norm_fn", None)
        if post_expert_norm_fn is None:
            raise RuntimeError("MuseSpark1x post-expert norm is not configured.")
        return MuseSparkQuarkOCPMXMoEQuantConfig(
            **{
                f.name: getattr(base_config, f.name)
                for f in fields(FusedMoEQuantConfig)
                if f.init
            },
            post_expert_norm_fn=post_expert_norm_fn,
        )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        loaded_slots = getattr(layer, "_quark_expert_loaded_slots", None)
        if loaded_slots is not None:
            _validate_quark_expert_weights(
                loaded_slots, layer.moe_config.num_logical_experts
            )
            del layer._quark_expert_loaded_slots
        super().process_weights_after_loading(layer)


def _quantize_experts_to_mxfp4(
    weight: torch.Tensor, experts_per_chunk: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Round ``[experts, rows, cols]`` weights to packed OCP MXFP4."""
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    num_experts, rows, cols = weight.shape
    packed = torch.empty(
        (num_experts, rows, cols // 2), dtype=torch.uint8, device=weight.device
    )
    scales = torch.empty(
        (num_experts, rows, cols // OCP_MX_BLOCK_SIZE),
        dtype=torch.uint8,
        device=weight.device,
    )
    for start in range(0, num_experts, experts_per_chunk):
        chunk = weight[start : start + experts_per_chunk]
        chunk_packed, chunk_scales = dynamic_mxfp4_quant(chunk.reshape(-1, cols))
        packed[start : start + chunk.shape[0]] = chunk_packed.view(
            chunk.shape[0], rows, -1
        )
        scales[start : start + chunk.shape[0]] = chunk_scales.view(torch.uint8).view(
            chunk.shape[0], rows, -1
        )[..., : cols // OCP_MX_BLOCK_SIZE]
    return packed, scales


class MuseSparkOnlineMXFP4FusedMoE(MuseSparkQuarkOCPMXFusedMoE):
    """Unquantized routed experts rounded to OCP MXFP4 (W4A4) after loading."""

    def __init__(self, moe: FusedMoEConfig) -> None:
        super().__init__(
            moe, _WEIGHT_QUANT_KEY_MAP["mxfp4"], _ACTIVATION_QUANT_KEY_MAP["mxfp4"]
        )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        for name, shape in (
            (
                "w13_weight",
                (
                    num_experts,
                    self.moe.w13_num_shards * intermediate_size_per_partition,
                    hidden_size,
                ),
            ),
            ("w2_weight", (num_experts, hidden_size, intermediate_size_per_partition)),
        ):
            param = torch.nn.Parameter(
                torch.empty(shape, dtype=params_dtype), requires_grad=False
            )
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra_weight_attrs)
        layer.w13_bias, layer.w2_bias = None, None
        layer.w13_input_scale, layer.w2_input_scale = None, None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        for name in ("w13_weight", "w2_weight"):
            packed, scales = _quantize_experts_to_mxfp4(getattr(layer, name).data)
            delattr(layer, name)
            layer.register_parameter(
                name, torch.nn.Parameter(packed, requires_grad=False)
            )
            layer.register_parameter(
                f"{name}_scale", torch.nn.Parameter(scales, requires_grad=False)
            )
        torch.accelerator.empty_cache()
        super().process_weights_after_loading(layer)


def _online_mxfp4_applies(prefix: str) -> bool:
    """Whether this layer's unquantized experts are rounded to MXFP4 on load.

    The first and last decoder layers stay BF16, as in the Quark MXFP4 export.
    """
    if envs.VLLM_ROCM_MUSE_MOE_ONLINE_QUANT != "mxfp4":
        return False
    num_layers = get_current_vllm_config().model_config.hf_text_config.num_hidden_layers
    return extract_layer_index(prefix) not in (0, num_layers - 1)


_QUARK_EXPERT_PROJECTIONS = {
    "gate_proj": ("w13_weight", "w1"),
    "up_proj": ("w13_weight", "w3"),
    "down_proj": ("w2_weight", "w2"),
}


def _quark_expert_weight_slot(name: str) -> tuple[str, str, int] | None:
    parts = name.split(".")
    if len(parts) != 3 or not parts[0].isdigit():
        return None
    projection = _QUARK_EXPERT_PROJECTIONS.get(parts[1])
    if projection is None or parts[2] not in ("weight", "weight_scale"):
        return None
    param_name, shard_id = projection
    if parts[2] == "weight_scale":
        param_name += "_scale"
    return param_name, shard_id, int(parts[0])


def _expected_quark_expert_weight_slots(
    num_experts: int,
) -> set[tuple[str, str, int]]:
    return {
        (param_name, shard_id, expert_id)
        for expert_id in range(num_experts)
        for param_name, shard_id in (
            ("w13_weight", "w1"),
            ("w13_weight", "w3"),
            ("w2_weight", "w2"),
            ("w13_weight_scale", "w1"),
            ("w13_weight_scale", "w3"),
            ("w2_weight_scale", "w2"),
        )
    }


def _track_quark_expert_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    loaded: set[tuple[str, str, int]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Track expert tensors across the loader's independently grouped calls."""
    for name, tensor in weights:
        slot = _quark_expert_weight_slot(name)
        if slot is not None:
            if slot in loaded:
                raise ValueError(f"Duplicate MuseSpark1x expert tensor {name!r}.")
            loaded.add(slot)
        yield name, tensor


def _validate_quark_expert_weights(
    loaded: set[tuple[str, str, int]],
    num_experts: int,
) -> None:
    expected = _expected_quark_expert_weight_slots(num_experts)
    missing = expected - loaded
    if missing:
        loaded_counts: dict[str, int] = {}
        for param_name, shard_id, _ in loaded:
            key = f"{param_name}:{shard_id}"
            loaded_counts[key] = loaded_counts.get(key, 0) + 1
        preview = ", ".join(
            f"{param_name}:{shard_id}:expert{expert_id}"
            for param_name, shard_id, expert_id in sorted(missing)[:8]
        )
        if len(missing) > 8:
            preview += f", ... ({len(missing)} total)"
        raise ValueError(
            "Missing MuseSpark1x Quark expert tensors: "
            f"{preview}; loaded slot counts: {loaded_counts}"
        )

    unexpected = loaded - expected
    if unexpected:
        preview = ", ".join(
            f"{param_name}:{shard_id}:expert{expert_id}"
            for param_name, shard_id, expert_id in sorted(unexpected)[:8]
        )
        if len(unexpected) > 8:
            preview += f", ... ({len(unexpected)} total)"
        raise ValueError(f"Unexpected MuseSpark1x Quark expert tensors: {preview}")


class MuseSparkRoutedExperts(MuseSparkConsolidatedRoutedExperts):
    """MuseSpark routed experts using ROCm BF16, FP8, or MXFP4 methods."""

    def _get_quant_method(
        self,
        prefix: str,
        quant_config: QuantizationConfig | None,
        moe_config: FusedMoEConfig,
    ) -> FusedMoEMethodBase:
        if quant_config is None:
            if _online_mxfp4_applies(prefix):
                return MuseSparkOnlineMXFP4FusedMoE(moe_config)
            return MuseSparkUnquantizedFusedMoE(moe_config)
        if not isinstance(quant_config, QuarkConfig):
            raise NotImplementedError(
                "MuseSpark1x on ROCm supports BF16, Quark FP8, or Quark OCP-MXFP4 "
                f"checkpoints; got {quant_config.get_name()!r}."
            )

        quant_method = quant_config.get_quant_method(self, prefix)
        if isinstance(quant_method, UnquantizedFusedMoEMethod):
            return MuseSparkUnquantizedFusedMoE(moe_config)
        if isinstance(quant_method, QuarkW8A8Fp8MoEMethod):
            return MuseSparkQuarkW8A8Fp8FusedMoE(
                moe_config,
                quant_method.weight_quant_key,
                quant_method.activation_quant_key,
            )
        if not isinstance(quant_method, QuarkOCP_MX_MoEMethod):
            raise NotImplementedError(
                "MuseSpark1x on ROCm requires Quark W8A8 FP8 or OCP-MXFP4 for routed "
                f"experts; got {type(quant_method).__name__}."
            )
        return MuseSparkQuarkOCPMXFusedMoE(
            moe_config,
            quant_method.weight_quant_key,
            quant_method.activation_quant_key,
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        if isinstance(
            self.quant_method, MuseSparkOnlineMXFP4FusedMoE
        ) or not isinstance(
            self.quant_method,
            (MuseSparkQuarkW8A8Fp8FusedMoE, MuseSparkQuarkOCPMXFusedMoE),
        ):
            # Quark's unquantized layers (0 and 61) store per-expert tensors.
            consolidated: list[tuple[str, torch.Tensor]] = []
            per_expert: list[tuple[str, torch.Tensor]] = []
            for name, tensor in weights:
                if _quark_expert_weight_slot(name) is None:
                    consolidated.append((name, tensor))
                else:
                    per_expert.append((name, tensor))
            if consolidated:
                yield from super().load_weights(consolidated)
            if per_expert:
                yield from RoutedExperts.load_weights(self, per_expert)
            return

        loaded_slots = getattr(self, "_quark_expert_loaded_slots", None)
        if loaded_slots is None:
            loaded_slots = set()
            self._quark_expert_loaded_slots = loaded_slots
        tracked_weights = _track_quark_expert_weights(weights, loaded_slots)
        yield from RoutedExperts.load_weights(self, tracked_weights)


class MuseMoERunner(MoERunner):
    """MoE runner whose custom reducer owns the TP reduction."""

    def __init__(
        self,
        *args,
        post_expert_norm: torch.nn.Module,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.routed_experts, MuseSparkRoutedExperts):
            raise TypeError(
                "MuseMoERunner requires routed_experts_cls=MuseSparkRoutedExperts."
            )
        self.routed_experts.post_expert_norm_fn = post_expert_norm.forward

    @property
    def _fused_output_is_reduced(self) -> bool:
        return needs_pre_norm_all_reduce(self.moe_config)


__all__ = [
    "MuseMoERunner",
    "MuseSpark1xRouter",
    "MuseSparkAiterExperts",
    "MuseSparkQuarkFp8MoEQuantConfig",
    "MuseSparkQuarkOCPMXFusedMoE",
    "MuseSparkQuarkW8A8Fp8FusedMoE",
    "MuseSparkRoutedExperts",
    "MuseSparkTopKWeightAndReduce",
    "MuseSparkTritonExperts",
    "validate_bf16_deployment",
    "validate_fp8_deployment",
    "validate_mxfp4_deployment",
]
