# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 multimodal model implementation for vLLM."""

import math
from collections.abc import Iterable
from typing import Any, cast

import torch
from torch import nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul, SituAndMul
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.fused_moe.router.base_router import (
    eplb_map_to_physical_and_record,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    fused_grouped_topk,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention as KimiLinearGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsEncoderCudaGraph,
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
)
from vllm.model_executor.models.kimi_k25 import KimiK25MediaPixelInputs
from vllm.model_executor.models.kimi_k25_vit import (
    KimiK25MultiModalProjector,
    MoonViT3dPretrainedModel,
    vision_tower_forward,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    init_vllm_registered_model,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.models.vision import is_vit_use_data_parallel
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_reduce_scatter,
    sp_shard,
)
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MegaMoEExperts
from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import prepare_megamoe_inputs
from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention
from vllm.models.kimi_k3.nvidia.latent_moe_runner import (
    LatentMoERunner,
)
from vllm.models.kimi_k3.nvidia.low_latency_gemm import (
    enable_kimi_k3_low_latency_gemm,
)
from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention
from vllm.models.kimi_k3.nvidia.ops import attn_res
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import NestedTensors
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.kimi_k3 import KimiK3Config
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
from vllm.utils.torch_utils import aux_stream
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

from ..common.mm_preprocess import (
    KimiK3DummyInputsBuilder,
    KimiK3MultiModalProcessor,
    KimiK3ProcessingInfo,
)

logger = init_logger(__name__)

# Token-count cutoff for overlapping the MoE router gate with the routed-expert
# down projection on a separate CUDA stream (latent MoE). At or below this many
# tokens the launch-bound decode path benefits from multi-stream overlap; above
# it the GEMMs saturate the device and the cross-stream sync is pure overhead,
# so it falls back to sequential.
_ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD = 256


def shard_sequence_parallel_mlp(
    hidden_size: int,
    intermediate_size: int,
    use_sequence_parallel: bool,
) -> bool:
    """Whether to TP-shard a sequence-parallel MLP instead of replicating it.

    Opt-in via ``VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT``; see :class:`KimiMLP` for
    the trade-off and :mod:`vllm.envs` for when it is worth enabling.
    """
    enabled = envs.VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT
    if not (use_sequence_parallel and enabled):
        return False
    tp_size = get_tensor_model_parallel_world_size()
    return (
        tp_size > 1 and intermediate_size % tp_size == 0 and hidden_size % tp_size == 0
    )


class KimiMLP(nn.Module):
    """Dense / shared-expert MLP, optionally TP-sharded under sequence parallel.

    Under sequence parallelism each rank owns a distinct slice of the tokens, so
    by default both projections are replicated (``disable_tp``) and the block
    needs no collective. That makes every rank stream the entire weight to serve
    its own token shard.

    With ``VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT`` the weights are TP-sharded
    instead. A rank then holds only a slice of the intermediate dim, so it
    cannot finish its own tokens alone: ``forward`` all-gathers the full token
    set, computes this rank's partial, and reduce-scatters. The reduce-scatter
    sums across TP and restores the sequence sharding in one collective, so the
    block still ends with one collective per direction.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        use_sequence_parallel: bool = False,
        prefix: str = "",
        activation_situ_beta: float | None = None,
        activation_situ_linear_beta: float | None = None,
    ) -> None:
        super().__init__()

        self.shard_sequence_parallel = shard_sequence_parallel_mlp(
            hidden_size,
            intermediate_size,
            use_sequence_parallel,
        )
        replicate = use_sequence_parallel and not self.shard_sequence_parallel

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=replicate,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            # Sharded sequence parallel reduces via the reduce-scatter in
            # forward(), which also restores the sequence sharding.
            reduce_results=False if self.shard_sequence_parallel else reduce_results,
            disable_tp=replicate,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act == "silu":
            self.act_fn = SiluAndMul()
        elif hidden_act == "situ":
            self.act_fn = SituAndMul(
                beta=activation_situ_beta or 1.0,
                linear_beta=activation_situ_linear_beta,
            )
        else:
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu and situ are supported."
            )

    def forward(self, x):
        if self.shard_sequence_parallel:
            # Each rank holds a weight shard but only its own tokens, so it
            # cannot finish those tokens alone: gather the full token set,
            # compute this rank's partial for all of them, then reduce-scatter,
            # which sums across TP and restores the sequence sharding.
            x = sp_all_gather(x)
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        if self.shard_sequence_parallel:
            x = sp_reduce_scatter(x)
        return x


class KimiRoutedOutputTransform(nn.Module):
    def __init__(
        self,
        norm: RMSNorm | None,
        up_proj: ReplicatedLinear,
    ) -> None:
        super().__init__()
        self.norm = norm
        self.up_proj = up_proj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        hidden_states, _ = self.up_proj(hidden_states)
        return hidden_states


class KimiK3MegaMoEExperts(DeepseekV4MegaMoEExperts):
    """Kimi K3 adapter for the DeepGEMM MegaMoE kernel."""

    _kimi_symm_buffer_cache: dict[tuple[object, ...], object] = {}
    _synchronized_ep_groups: set[tuple[int, int]] = set()

    def __init__(
        self,
        *args,
        activation: str,
        activation_beta: float | None,
        activation_linear_beta: float | None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.activation = activation
        self.activation_beta = activation_beta
        self.activation_linear_beta = activation_linear_beta

    def synchronize_first_launch(self) -> None:
        ep_group = get_ep_group()
        device = torch.accelerator.current_device_index()
        key = (id(ep_group.cpu_group), device)
        if key in self._synchronized_ep_groups:
            return
        torch.accelerator.synchronize()
        torch.distributed.barrier(group=ep_group.cpu_group)
        self._synchronized_ep_groups.add(key)

    def finalize_weights(self) -> None:
        if self._transformed_l1_weights is not None:
            return

        self._check_runtime_supported()
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()
        w13_scale = deep_gemm.transform_sf_into_required_layout(
            self._ue8m0_uint8_to_float(self.w13_weight_scale.data).contiguous(),
            2 * self.intermediate_size,
            self.hidden_size,
            (1, 32),
            self.num_local_experts,
        )
        w2_scale = deep_gemm.transform_sf_into_required_layout(
            self._ue8m0_uint8_to_float(self.w2_weight_scale.data).contiguous(),
            self.hidden_size,
            self.intermediate_size,
            (1, 32),
            self.num_local_experts,
        )
        self._transformed_l1_weights, self._transformed_l2_weights = (
            deep_gemm.transform_weights_for_mega_moe(
                (self.w13_weight.data.view(torch.int8).contiguous(), w13_scale),
                (self.w2_weight.data.view(torch.int8).contiguous(), w2_scale),
                activation=self.activation,
            )
        )
        self.w13_weight = None
        self.w13_weight_scale = None
        self.w2_weight = None
        self.w2_weight_scale = None

    def get_symm_buffer(self):
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()
        group = get_ep_group().device_group
        device = torch.accelerator.current_device_index()
        key = (
            id(group),
            device,
            self.num_experts,
            self.max_num_tokens,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.activation,
        )
        symm_buffer = self._kimi_symm_buffer_cache.get(key)
        if symm_buffer is None:
            symm_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
                group,
                self.num_experts,
                self.max_num_tokens,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
                activation=self.activation,
            )
            self._kimi_symm_buffer_cache[key] = symm_buffer
        return symm_buffer

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        activation_clamp: float | None,
        fast_math: bool = True,
    ) -> torch.Tensor:
        self.synchronize_first_launch()
        if hidden_states.shape[0] > self.max_num_tokens:
            raise ValueError(
                f"Kimi K3 MegaMoE got {hidden_states.shape[0]} tokens, "
                f"but its symmetric buffer supports {self.max_num_tokens}."
            )
        y = torch.empty_like(hidden_states, dtype=torch.bfloat16)
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()
        symm_buffer = self.get_symm_buffer()
        num_tokens = hidden_states.shape[0]
        is_padding = None
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            is_padding = get_forward_context().is_padding
            if is_padding is not None:
                is_padding = is_padding[:num_tokens]

        eplb_state = self.eplb_state
        if eplb_state.logical_to_physical_map is not None:
            assert eplb_state.expert_load_view is not None
            assert eplb_state.logical_replica_count is not None
            assert eplb_state.should_record_tensor is not None
            if is_padding is not None:
                topk_ids = torch.where(is_padding.unsqueeze(1), -1, topk_ids)
            topk_ids = eplb_map_to_physical_and_record(
                topk_ids=topk_ids,
                expert_load_view=eplb_state.expert_load_view,
                logical_to_physical_map=eplb_state.logical_to_physical_map,
                logical_replica_count=eplb_state.logical_replica_count,
                record_enabled=eplb_state.should_record_tensor,
                num_unpadded_tokens=eplb_state.num_unpadded_tokens_tensors[
                    dbo_current_ubatch_id()
                ]
                if eplb_state.num_unpadded_tokens_tensors is not None
                else None,
            )

        prepare_megamoe_inputs(
            hidden_states,
            topk_weights,
            topk_ids,
            symm_buffer.x[:num_tokens],
            symm_buffer.x_sf[:num_tokens],
            symm_buffer.topk_idx[:num_tokens],
            symm_buffer.topk_weights[:num_tokens],
            is_padding=is_padding,
        )
        self.finalize_weights()
        assert self._transformed_l1_weights is not None
        assert self._transformed_l2_weights is not None
        deep_gemm.fp8_fp4_mega_moe(
            y,
            self._transformed_l1_weights,
            self._transformed_l2_weights,
            symm_buffer,
            activation_clamp=activation_clamp,
            activation=self.activation,
            activation_beta=self.activation_beta,
            activation_linear_beta=self.activation_linear_beta,
            fast_math=fast_math,
        )
        return y


def make_kimi_k3_mega_moe_expert_params_mapping(
    num_experts: int,
) -> list[tuple[str, str, int, str]]:
    mapping = []
    for expert_id in range(num_experts):
        for shard_id in ("w1", "w2", "w3"):
            param_prefix = "w13" if shard_id in ("w1", "w3") else "w2"
            for suffix in ("weight_packed", "weight_scale"):
                param_suffix = "weight" if suffix == "weight_packed" else suffix
                mapping.append(
                    (
                        f"experts.{param_prefix}_{param_suffix}",
                        f"experts.{expert_id}.{shard_id}.{suffix}",
                        expert_id,
                        shard_id,
                    )
                )
    return mapping


class KimiMoE(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        layer_idx: int = 0,
        use_sequence_parallel: bool = False,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        moe_intermediate_size = config.moe_intermediate_size
        num_experts = config.num_experts
        num_experts_per_token = config.num_experts_per_token
        assert moe_intermediate_size is not None
        assert num_experts is not None
        assert num_experts_per_token is not None
        moe_renormalize = config.moe_renormalize
        routed_expert_hidden_size = config.routed_expert_hidden_size
        self.use_latent_moe = routed_expert_hidden_size is not None
        self.moe_hidden_size = (
            routed_expert_hidden_size
            if routed_expert_hidden_size is not None
            else hidden_size
        )
        self.latent_moe_use_norm = config.latent_moe_use_norm
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.moe_renormalize = moe_renormalize
        self.use_grouped_topk = config.use_grouped_topk
        self.num_expert_group = config.num_expert_group
        self.topk_group = config.topk_group
        self.moe_router_activation_func = config.moe_router_activation_func
        self.num_shared_experts = config.num_shared_experts
        self.layer_idx = layer_idx
        self.use_mega_moe = (
            vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
        )
        if self.use_mega_moe and not vllm_config.parallel_config.enable_expert_parallel:
            raise NotImplementedError(
                "Kimi K3 MegaMoE requires expert parallel. Enable it with "
                "--enable-expert-parallel."
            )
        if self.use_mega_moe and config.hidden_act != "situ":
            raise ValueError("Kimi K3 MegaMoE requires SITU activation.")
        if self.use_mega_moe and not self.use_latent_moe:
            raise ValueError("Kimi K3 MegaMoE requires latent MoE projections.")
        if self.use_mega_moe and not self.use_grouped_topk:
            raise ValueError("Kimi K3 MegaMoE requires grouped top-k routing.")
        if self.use_mega_moe and (self.num_expert_group != 1 or self.topk_group != 1):
            raise NotImplementedError(
                "Kimi K3 MegaMoE currently requires one expert group."
            )
        self.padded_moe_intermediate_size = moe_intermediate_size
        min_moe_intermediate_per_partition = getattr(
            config, "min_moe_intermediate_per_partition", 256
        )
        if self.tp_size > 1:
            moe_intermediate_per_partition = moe_intermediate_size // self.tp_size
            if moe_intermediate_per_partition < min_moe_intermediate_per_partition:
                self.padded_moe_intermediate_size = (
                    min_moe_intermediate_per_partition * self.tp_size
                )
        activation_situ_beta = (
            config.activation_situ_beta if config.hidden_act == "situ" else None
        )
        activation_situ_linear_beta = (
            config.activation_situ_linear_beta if config.hidden_act == "situ" else None
        )

        # Route with fp32 logits for numerically stable expert selection.
        self.gate = GateLinear(
            input_size=hidden_size,
            output_size=num_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32)
        )

        if self.num_shared_experts is not None:
            shared_intermediate_size = moe_intermediate_size * self.num_shared_experts
            self.shared_experts = KimiMLP(
                hidden_size=config.hidden_size,  # 7618
                intermediate_size=shared_intermediate_size,  # 3072*2
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                use_sequence_parallel=use_sequence_parallel,
                prefix=f"{prefix}.shared_experts",
                activation_situ_beta=activation_situ_beta,
                activation_situ_linear_beta=activation_situ_linear_beta,
            )
        else:
            self.shared_experts = None

        self.routed_expert_down_proj: ReplicatedLinear | None
        self.routed_expert_norm: RMSNorm | None
        self.routed_expert_up_proj: ReplicatedLinear | None
        self.routed_output_transform: KimiRoutedOutputTransform | None
        if self.use_latent_moe:
            self.routed_expert_down_proj = ReplicatedLinear(
                hidden_size,
                self.moe_hidden_size,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.routed_expert_down_proj",
            )
            self.routed_expert_norm = (
                RMSNorm(self.moe_hidden_size, eps=config.rms_norm_eps)
                if self.latent_moe_use_norm
                else None
            )
            # Replicated up-proj: the full weight lives on every rank and
            # produces the full hidden dim locally. This lets LatentMoERunner
            # fuse the latent and shared reductions into a single all-reduce
            # (concat the two partials, reduce once), then run the up-proj and
            # shared add locally with no further collective.
            self.routed_expert_up_proj = ReplicatedLinear(
                self.moe_hidden_size,
                hidden_size,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.routed_expert_up_proj",
            )

            self.routed_output_transform = KimiRoutedOutputTransform(
                self.routed_expert_norm, self.routed_expert_up_proj
            )
            # Auxiliary CUDA stream to overlap the router gate with the routed
            # down projection on decode-sized batches (gated by
            # _ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD).
            self._down_proj_stream: torch.cuda.Stream | None = aux_stream()
            self._down_proj_events = (torch.cuda.Event(), torch.cuda.Event())
        else:
            self.routed_expert_down_proj = None
            self.routed_expert_norm = None
            self.routed_expert_up_proj = None
            self.routed_output_transform = None

        if self.use_mega_moe:
            ep_group = get_ep_group()
            ep_size = ep_group.world_size
            ep_rank = ep_group.rank_in_group
            if num_experts % ep_size != 0:
                raise ValueError(
                    f"Kimi K3 num_experts={num_experts} must be divisible by "
                    f"EP size {ep_size}."
                )
            num_local_experts = num_experts // ep_size
            self.experts = KimiK3MegaMoEExperts(
                vllm_config,
                num_experts=num_experts,
                num_local_experts=num_local_experts,
                experts_start_idx=ep_rank * num_local_experts,
                top_k=num_experts_per_token,
                hidden_size=self.moe_hidden_size,
                intermediate_size=self.padded_moe_intermediate_size,
                prefix=f"{prefix}.experts",
                activation="situ",
                activation_beta=activation_situ_beta,
                activation_linear_beta=activation_situ_linear_beta,
            )
        else:
            self.experts = FusedMoEFactory(
                shared_experts=self.shared_experts,
                num_experts=num_experts,
                top_k=num_experts_per_token,
                hidden_size=self.moe_hidden_size,
                intermediate_size=self.padded_moe_intermediate_size,
                activation=config.hidden_act,
                activation_situ_beta=activation_situ_beta,
                activation_situ_linear_beta=activation_situ_linear_beta,
                renormalize=moe_renormalize,
                quant_config=quant_config,
                use_grouped_topk=config.use_grouped_topk,
                num_expert_group=config.num_expert_group,
                topk_group=config.topk_group,
                prefix=f"{prefix}.experts",
                scoring_func=config.moe_router_activation_func,
                e_score_correction_bias=self.gate.e_score_correction_bias,
                routed_scaling_factor=self.routed_scaling_factor,
                # Down projection runs outside MoERunner so it can overlap the
                # router gate on the aux stream (see forward()); the original
                # hidden states are passed to forward() as shared_experts_input
                # so shared experts still see the untransformed input.
                routed_input_transform=None,
                routed_output_transform=self.routed_output_transform,
                is_sequence_parallel=use_sequence_parallel,
                runner_cls=LatentMoERunner if self.use_latent_moe else None,
            )
        if self.padded_moe_intermediate_size != moe_intermediate_size:
            w13_weight = getattr(self.experts, "w13_weight", None)
            if w13_weight is None:
                w13_weight = getattr(self.experts, "w13_weight_packed", None)
            w2_weight = getattr(self.experts, "w2_weight", None)
            if w2_weight is None:
                w2_weight = getattr(self.experts, "w2_weight_packed", None)
            if w13_weight is not None:
                w13_weight.data.zero_()
            if w2_weight is not None:
                w2_weight.data.zero_()
            self.experts.moe_config.intermediate_size_per_partition_unpadded = (
                moe_intermediate_size // self.tp_size
            )

    def _maybe_overlap_router_and_down_proj(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Compute the routed-expert down projection alongside the router,
        overlapping them on separate CUDA streams when latent MoE is enabled.

        The router gate and the down projection both read ``hidden_states``, so
        the gate runs on the default stream and the down projection on the aux
        stream, joined via ``maybe_execute_in_parallel``. For MegaMoE the
        grouped top-k selection consumes only the gate logits, so it also runs
        on the default stream and overlaps the down projection.

        Returns:
            ``(routed_hidden_states, router_output, topk_ids)``.
            ``routed_hidden_states`` is the down-projected latent (or the
            original ``hidden_states`` when latent MoE is disabled). For MegaMoE
            ``router_output`` holds the grouped top-k weights and ``topk_ids``
            the selected experts; otherwise ``router_output`` holds the raw gate
            logits and ``topk_ids`` is ``None``.
        """

        def _router(
            hidden_states: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            router_logits, _ = self.gate(hidden_states)
            if not self.use_mega_moe:
                return router_logits, None
            return fused_grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.experts.top_k,
                renormalize=self.moe_renormalize,
                e_score_correction_bias=self.gate.e_score_correction_bias.data,
                num_expert_group=self.num_expert_group,
                topk_group=self.topk_group,
                scoring_func=self.moe_router_activation_func,
                routed_scaling_factor=self.routed_scaling_factor,
            )

        down_proj = self.routed_expert_down_proj
        if down_proj is None:
            router_output, topk_ids = _router(hidden_states)
            return hidden_states, router_output, topk_ids
        num_tokens = hidden_states.shape[0]
        (router_output, topk_ids), (routed_hidden_states, _) = (
            maybe_execute_in_parallel(
                lambda: _router(hidden_states),
                lambda: down_proj(hidden_states),
                self._down_proj_events[0],
                self._down_proj_events[1],
                self._down_proj_stream
                if num_tokens <= _ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD
                else None,
            )
        )
        return routed_hidden_states, router_output, topk_ids

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        # Overlap the gate with the routed down projection; the returned hidden
        # states are already down-projected. Keep the original ``hidden_states``
        # for the shared experts.
        routed_hidden_states, router_output, topk_ids = (
            self._maybe_overlap_router_and_down_proj(hidden_states)
        )
        if self.use_mega_moe:
            assert self.routed_output_transform is not None
            assert topk_ids is not None
            final_hidden_states = self.experts(
                routed_hidden_states,
                router_output,
                topk_ids,
                activation_clamp=None,
            )
            final_hidden_states = self.routed_output_transform(final_hidden_states)
            if self.shared_experts is not None:
                final_hidden_states = final_hidden_states + self.shared_experts(
                    hidden_states
                )
        else:
            # Routed experts consume the down-projected latent; shared experts
            # (inside MoERunner) get the original hidden states via
            # shared_experts_input.
            final_hidden_states = self.experts(
                hidden_states=routed_hidden_states,
                router_logits=router_output,
                shared_experts_input=hidden_states,
            )
        return final_hidden_states.view(num_tokens, hidden_size)


class KimiDecoderLayer(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
        aux_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = int(prefix.rsplit(".", 1)[1])

        self.is_moe = config.is_moe
        layer_idx = self.layer_idx
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        self.is_moe_layer = (
            self.is_moe
            and config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        )

        use_mega_moe = vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
        self.use_sequence_parallel = (
            parallel_config.pipeline_parallel_size == 1
            and parallel_config.enable_expert_parallel
            and parallel_config.tensor_parallel_size > 1
            and (use_mega_moe or parallel_config.data_parallel_size > 1)
        )
        if config.is_kda_layer(layer_idx):
            kda_config = config.linear_attn_config
            assert kda_config is not None
            # This class also serves standalone Kimi-Linear through the model
            # registry. Only Kimi-K3's full-rank gate uses the private KDA path.
            if kda_config.get("use_full_rank_gate", False):
                self.self_attn = KimiK3DeltaAttention(
                    config,
                    vllm_config,
                    prefix=f"{prefix}.self_attn",
                )
                self._self_attn_writes_output = False
            else:
                self.self_attn = KimiLinearGatedDeltaNetAttention(
                    config,
                    vllm_config,
                    prefix=f"{prefix}.self_attn",
                )
                self._self_attn_writes_output = True
        else:
            qk_nope_head_dim = config.qk_nope_head_dim
            qk_rope_head_dim = config.qk_rope_head_dim
            v_head_dim = config.v_head_dim
            kv_lora_rank = config.kv_lora_rank
            mla_use_nope = config.mla_use_nope
            assert qk_nope_head_dim is not None
            assert qk_rope_head_dim is not None
            assert v_head_dim is not None
            assert kv_lora_rank is not None
            assert mla_use_nope, "Kimi-K3 MLA (MultiHeadLatentAttention) is NoPE-only"
            # q_lora_rank may be None (Kimi-Linear): the MLA layer then uses an
            # uncompressed q_proj instead of the fused q-LoRA front-end.
            self.self_attn = MultiHeadLatentAttention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                use_output_gate=bool(config.mla_use_output_gate),
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                aux_stream=aux_stream,
            )
            self._self_attn_writes_output = False

        if self.use_sequence_parallel:
            self.self_attn.o_proj.reduce_results = False

        if self.is_moe_layer:
            self.block_sparse_moe = KimiMoE(
                config=config,
                vllm_config=vllm_config,
                quant_config=quant_config,
                prefix=f"{prefix}.block_sparse_moe",
                layer_idx=layer_idx,
                use_sequence_parallel=self.use_sequence_parallel,
            )
            self.mlp = self.block_sparse_moe
        else:
            self.mlp = KimiMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                use_sequence_parallel=self.use_sequence_parallel,
                activation_situ_beta=config.activation_situ_beta,
                activation_situ_linear_beta=config.activation_situ_linear_beta,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        attn_res_block_size = config.attn_res_block_size
        self.use_attn_res = attn_res_block_size is not None
        if self.use_attn_res:
            assert attn_res_block_size is not None
            self.attn_res_block_size = attn_res_block_size
            self.is_block_write_layer = layer_idx % self.attn_res_block_size == 0
            self.block_write_idx = layer_idx // self.attn_res_block_size
            self.prev_valid_blocks = cdiv(layer_idx, self.attn_res_block_size)
            self.self_attention_res_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.mlp_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attention_res_proj = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.self_attention_res_proj",
            )
            self.mlp_res_proj = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.mlp_res_proj",
            )

    def _run_self_attn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self._self_attn_writes_output:
            output = torch.empty_like(hidden_states)
            self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
                output=output,
            )
            return output
        return self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
        )

    def _pre_attn_norm(
        self,
        hidden_states: torch.Tensor | None,
        residual: torch.Tensor | None,
        prefix_sum: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if not self.use_attn_res:
            assert hidden_states is not None
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            return hidden_states, prefix_sum, residual

        assert prefix_sum is not None
        assert residual is not None
        hidden_states = attn_res(
            prefix_sum,
            hidden_states,
            residual,
            self.self_attention_res_norm.weight,
            self.self_attention_res_proj.weight.squeeze(0),
            self.input_layernorm.weight,
            num_blocks=self.prev_valid_blocks,
            block_write_idx=(self.block_write_idx if self.is_block_write_layer else -1),
            eps=self.self_attention_res_norm.variance_epsilon,
            output_norm_eps=self.input_layernorm.variance_epsilon,
        )
        return hidden_states, prefix_sum, residual

    def _post_attn_norm(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        prefix_sum: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if not self.use_attn_res:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
            return hidden_states, prefix_sum, residual

        assert prefix_sum is not None
        if self.is_block_write_layer:
            prefix_sum = hidden_states
            prefix_delta = None
        else:
            prefix_delta = hidden_states
        mlp_valid_blocks = self.prev_valid_blocks + self.is_block_write_layer
        hidden_states = attn_res(
            prefix_sum,
            prefix_delta,
            residual,
            self.mlp_res_norm.weight,
            self.mlp_res_proj.weight.squeeze(0),
            self.post_attention_layernorm.weight,
            num_blocks=mlp_valid_blocks,
            block_write_idx=-1,
            eps=self.mlp_res_norm.variance_epsilon,
            output_norm_eps=self.post_attention_layernorm.variance_epsilon,
        )
        return hidden_states, prefix_sum, residual

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None,
        residual: torch.Tensor | None,
        prefix_sum: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        hidden_states, prefix_sum, residual = self._pre_attn_norm(
            hidden_states, residual, prefix_sum
        )
        assert hidden_states is not None

        if self.use_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)
            # Remove SP padding before attention.
            hidden_states = hidden_states[: positions.shape[0]]

        # Attention.
        hidden_states = self._run_self_attn(positions, hidden_states)

        if self.use_sequence_parallel:
            # Add SP padding if needed, and then perform reduce scatter.
            hidden_states = sp_reduce_scatter(hidden_states)

        hidden_states, prefix_sum, residual = self._post_attn_norm(
            hidden_states, residual, prefix_sum
        )

        # MoE/MLP.
        hidden_states = self.mlp(hidden_states)
        return hidden_states, prefix_sum, residual


class KimiLinearModel(nn.Module, EagleModelMixin, SupportsQuant):
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvgfab": ["q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj"],
        "conv1d": ["q_conv1d", "k_conv1d", "v_conv1d"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.attn_res_block_size: int | None = config.attn_res_block_size
        self.use_attn_res = self.attn_res_block_size is not None
        parallel_config = vllm_config.parallel_config
        use_mega_moe = vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
        self.use_sequence_parallel = (
            parallel_config.pipeline_parallel_size == 1
            and parallel_config.enable_expert_parallel
            and parallel_config.tensor_parallel_size > 1
            and (use_mega_moe or parallel_config.data_parallel_size > 1)
        )

        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Aux stream for overlapping the MLA g_proj output-gate GEMM with the
        # attention front-end (DeepseekV4 convention: created at the model
        # level and threaded into each attention layer).
        aux_stream = torch.cuda.Stream()

        def get_layer(prefix: str):
            return KimiDecoderLayer(
                config,
                vllm_config,
                prefix,
                aux_stream=aux_stream,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        self.num_attn_res_blocks = (
            cdiv(self.end_layer, self.attn_res_block_size)
            if self.attn_res_block_size is not None
            else 0
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if self.use_attn_res:
                self.output_attn_res_norm = RMSNorm(
                    config.hidden_size, eps=config.rms_norm_eps
                )
                self.output_attn_res_proj = ReplicatedLinear(
                    config.hidden_size,
                    1,
                    bias=False,
                    quant_config=None,
                    prefix=f"{prefix}.output_attn_res_proj",
                )
        else:
            self.norm = PPMissingLayer()
            if self.use_attn_res:
                self.output_attn_res_norm = PPMissingLayer()
                self.output_attn_res_proj = PPMissingLayer()

        world_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % world_size == 0, (
            "num_attention_heads must be divisible by world_size"
        )

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        residual_shape: tuple[int, ...] = (batch_size, self.config.hidden_size)
        if self.use_attn_res:
            assert self.attn_res_block_size is not None
            residual_shape = (
                batch_size,
                cdiv(self.start_layer, self.attn_res_block_size),
                self.config.hidden_size,
            )
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
                "residual": torch.zeros(residual_shape, dtype=dtype, device=device),
            }
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        assert hidden_states is not None

        aux_hidden_states: list[torch.Tensor] = []
        if self.start_layer in self.aux_hidden_state_layers:
            if self.use_attn_res or residual is None:
                aux_hidden_states.append(hidden_states)
            else:
                aux_hidden_states.append(hidden_states + residual)

        full_num_tokens = positions.shape[0]
        if self.use_sequence_parallel:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding, hidden_states
                )
            hidden_states = sp_shard(hidden_states)
            assert residual is None, "Currently, SP is not supported with PP"

        prefix_sum = None
        if self.use_attn_res:
            block_residual = hidden_states.new_empty(
                hidden_states.size(0),
                self.num_attn_res_blocks,
                hidden_states.size(1),
            )
            if residual is not None:
                block_residual[:, : residual.size(1), :].copy_(residual)
            prefix_sum = hidden_states
            hidden_states = None
            residual = block_residual

        for layer_idx, layer in enumerate(
            self.layers[self.start_layer : self.end_layer],
            start=self.start_layer,
        ):
            hidden_states, prefix_sum, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                prefix_sum=prefix_sum,
                residual=residual,
            )
            if (layer_idx + 1) in self.aux_hidden_state_layers:
                if self.use_attn_res:
                    assert prefix_sum is not None
                    aux_hidden_state = prefix_sum + hidden_states
                else:
                    assert residual is not None
                    aux_hidden_state = hidden_states + residual

                if self.use_sequence_parallel:
                    # Gather SP-sharded aux hidden states.
                    # TODO: Optimize this.
                    aux_hidden_state = sp_all_gather(aux_hidden_state)
                    aux_hidden_state = aux_hidden_state[:full_num_tokens]
                aux_hidden_states.append(aux_hidden_state)

        assert hidden_states is not None
        assert residual is not None
        if not get_pp_group().is_last_rank:
            assert not self.use_sequence_parallel, (
                "Currently, SP is not supported with PP"
            )
            if prefix_sum is not None:
                hidden_states = hidden_states + prefix_sum
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if self.use_attn_res:
            assert prefix_sum is not None
            hidden_states = attn_res(
                prefix_sum,
                hidden_states,
                residual,
                self.output_attn_res_norm.weight,
                self.output_attn_res_proj.weight.squeeze(0),
                None,
                num_blocks=self.num_attn_res_blocks,
                block_write_idx=-1,
                eps=self.output_attn_res_norm.variance_epsilon,
                output_norm_eps=0.0,
            )
        else:
            hidden_states = hidden_states + residual

        if self.use_sequence_parallel:
            # Gather SP-sharded hidden states.
            hidden_states = sp_all_gather(hidden_states)
            hidden_states = hidden_states[:full_num_tokens]

        # NOTE: the final norm is applied in compute_logits instead of here, so
        # the MTP draft model receives the pre-norm hidden states.
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(
        self,
        weights: Iterable[
            tuple[str, torch.Tensor] | tuple[str, torch.Tensor, dict[str, Any]]
        ],
    ) -> set[str]:
        kda_config = self.config.linear_attn_config
        use_full_rank_gate = bool(
            kda_config and kda_config.get("use_full_rank_gate", False)
        )
        beta_shard_id = 5 if use_full_rank_gate else 3
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".in_proj_qkvgfab", ".q_proj", 0),
            (".in_proj_qkvgfab", ".k_proj", 1),
            (".in_proj_qkvgfab", ".v_proj", 2),
            (".in_proj_qkvgfab", ".b_proj", beta_shard_id),
            (".in_proj_qkvgfab", ".f_a_proj", 4),
            (".conv1d", ".q_conv1d", 0),
            (".conv1d", ".k_conv1d", 1),
            (".conv1d", ".v_conv1d", 2),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        if use_full_rank_gate:
            stacked_params_mapping.append((".in_proj_qkvgfab", ".g_proj", 3))
        if getattr(self.config, "q_lora_rank", None) is not None:
            stacked_params_mapping += [
                (".fused_qkv_a_proj", ".q_a_proj", 0),
                (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
            ]
        use_mega_moe = any(
            module.use_mega_moe
            for module in self.modules()
            if isinstance(module, KimiMoE)
        )
        if self.config.is_moe and use_mega_moe:
            expert_params_mapping = make_kimi_k3_mega_moe_expert_params_mapping(
                self.config.num_experts
            )
        elif self.config.is_moe:
            # Params for weights, fp8 weight scales, fp8 activation scales
            # (param_name, weight_name, expert_id, shard_id)
            expert_params_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=self.config.num_experts,
            )
        else:
            expert_params_mapping = []
        params_dict = dict(self.named_parameters())

        # Under the MXFP4 quant interface the routed experts register unpacked
        # params (``w13_weight``), while the compressed-tensors checkpoint names
        # them ``.weight_packed``. Rebind so the expert mapping resolves; scales
        # already share the ``.weight_scale`` suffix.
        experts_unpacked = not use_mega_moe and not any(
            n.endswith("w13_weight_packed") for n in params_dict
        )
        loaded_params: set[str] = set()
        for args in weights:
            name, loaded_weight = args[0], args[1]
            kwargs: dict[str, Any] = args[2] if len(args) > 2 else {}
            if "rotary_emb.inv_freq" in name:
                continue
            if experts_unpacked and name.endswith(".weight_packed"):
                name = name.replace(".weight_packed", ".weight")

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                # Packed projections are only present on compatible layers.
                if name_mapped not in params_dict:
                    continue
                name = name_mapped
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for (
                    expert_param_name,
                    expert_weight_name,
                    expert_id,
                    expert_shard_id,
                ) in expert_params_mapping:
                    if expert_weight_name not in name:
                        continue
                    name = name.replace(expert_weight_name, expert_param_name)
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        expert_id=expert_id,
                        shard_id=expert_shard_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias")
                        and name not in params_dict
                        and not self.config.is_linear_attn
                    ):  # noqa: E501
                        continue
                    # Remapping the name of FP8 kv-scale.
                    remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                    if remapped_name is None:
                        continue
                    name = remapped_name
                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, **kwargs)
            loaded_params.add(name)
        return loaded_params

    def finalize_mega_moe_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, KimiMoE) and module.use_mega_moe:
                module.experts.finalize_weights()


class KimiLinearForCausalLM(
    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.config = self.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.quant_config = quant_config
        self.model = KimiLinearModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        enable_kimi_k3_low_latency_gemm(self, self.model_config.dtype)
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size, scale=logit_scale
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        return self.model.make_empty_intermediate_tensors(batch_size, dtype, device)

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        return self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.kda_state_shape(
            tp_size,
            hf_config.linear_attn_config["num_heads"],
            hf_config.linear_attn_config["head_dim"],
            conv_kernel_size=hf_config.linear_attn_config["short_conv_kernel_size"],
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        # The model's final norm is applied here (not at the end of forward) so
        # that the pre-norm hidden states can be fed to the MTP draft model.
        hidden_states = self.model.norm(hidden_states, None)
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        loaded = loader.load_weights(weights)
        self.model.finalize_mega_moe_weights()
        # The fused MultiHeadLatentAttention's process_weights_after_loading
        # (W_UK_T / W_UV absorption) is driven by the loader's generic post-load
        # hook for any AttentionLayerBase, so no manual trigger is needed here.
        return loaded


def get_spec_layer_idx_from_weight_name(
    config: KimiLinearConfig, weight_name: str
) -> int | None:
    if hasattr(config, "num_nextn_predict_layers") and (
        config.num_nextn_predict_layers > 0
    ):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            # Match regardless of the surrounding prefix. The name may arrive as
            # ``model.layers.{i}.``, a bare ``layers.{i}.`` (after AutoWeightsLoader
            # has stripped the ``model.`` prefix in the main model), or with the
            # multimodal ``language_model.model.layers.{i}.`` prefix.
            if f"layers.{layer_idx + i}." in weight_name:
                return layer_idx + i
    return None


@MULTIMODAL_REGISTRY.register_processor(
    KimiK3MultiModalProcessor,
    info=KimiK3ProcessingInfo,
    dummy_inputs=KimiK3DummyInputsBuilder,
)
class KimiK3ForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsEncoderCudaGraph,
    SupportsPP,
    SupportsQuant,
    SupportsEagle3,
    HasInnerState,
    IsHybrid,
):
    """Kimi-K3 model with Kimi-K2.5 vision and KimiLinear text."""

    supports_encoder_tp_data = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.layers.": "language_model.model.layers.",
            "mm_projector.proj.0": "mm_projector.linear_1",
            "mm_projector.proj.2": "mm_projector.linear_2",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return "<|kimi_image_placeholder|>"
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        config: KimiK3Config = model_config.hf_config
        self.config = config
        self.model_config = model_config
        quant_config = vllm_config.quant_config

        multimodal_config = model_config.multimodal_config
        assert multimodal_config is not None
        self.use_data_parallel = is_vit_use_data_parallel(
            config.vision_config.num_attention_heads
        )
        self.hidden_size = config.text_config.hidden_size
        self.device = current_platform.current_device()

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_tower = MoonViT3dPretrainedModel(
                config.vision_config,
                quant_config=self._maybe_ignore_quant_config(quant_config),
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            if self._maybe_ignore_quant_config(quant_config) is not None:
                self.vision_tower = self.vision_tower.to(device=self.device)
            else:
                self.vision_tower = self.vision_tower.to(
                    device=self.device, dtype=model_config.dtype
                )

            vision_attn = self.vision_tower.encoder.blocks[0].attn
            if vision_attn.is_flash_attn_backend and vision_attn._fa_version == 4:
                from vllm.models.kimi_k3.nvidia.ops.vision_fa4_warmup import (
                    KimiK3VisionFA4WarmupConfig,
                    register_kimi_k3_vision_fa4_warmup,
                )

                merge_height, merge_width = config.vision_config.merge_kernel_size
                mm_config = model_config.get_multimodal_config()
                assert mm_config is not None
                register_kimi_k3_vision_fa4_warmup(
                    KimiK3VisionFA4WarmupConfig(
                        num_heads=vision_attn.num_heads,
                        head_dim=vision_attn.head_size,
                        dtype=vision_attn.dtype,
                        max_batch_size=(
                            vllm_config.scheduler_config.max_num_seqs
                            * mm_config.get_limit_per_prompt("image")
                        ),
                        max_seqlen=(
                            vllm_config.scheduler_config.max_num_encoder_input_tokens
                            * merge_height
                            * merge_width
                        ),
                    )
                )

            self.mm_projector = KimiK25MultiModalProjector(
                config=config.vision_config,
                use_data_parallel=self.use_data_parallel,
                quant_config=self._maybe_ignore_quant_config(quant_config),
                prefix=maybe_prefix(prefix, "mm_projector"),
            )
            self.mm_projector = self.mm_projector.to(
                device=self.device, dtype=model_config.dtype
            )

        self.quant_config = quant_config
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["KimiLinearForCausalLM"],
            )
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.language_model.make_empty_intermediate_tensors
        )
        self.media_placeholder: int = self.config.media_placeholder_token_id

    # -- SupportsEncoderCudaGraph protocol methods --

    def get_encoder_cudagraph_config(self):
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphConfig

        return EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=[
                "pixel_values",
                "pos_embeds",
                "rope_freqs_cis",
                "cu_seqlens",
                "max_seqlen",
                "sequence_lengths",
                "merge_gather_idx",
            ],
            out_hidden_size=self.hidden_size,
        )

    def get_encoder_cudagraph_budget_range(
        self, vllm_config: VllmConfig
    ) -> tuple[int, int]:
        min_budget = 64
        max_budget = min(
            vllm_config.scheduler_config.max_num_batched_tokens,
            self.model_config.max_model_len,
        )
        return min_budget, max_budget

    @staticmethod
    def _get_grid_thws(mm_kwargs: dict[str, Any]) -> list[list[int]]:
        grid_thws = mm_kwargs["grid_thws"]
        if not isinstance(grid_thws, list):
            grid_thws = grid_thws.tolist()
        return grid_thws

    @staticmethod
    def _get_pixel_values(mm_kwargs: dict[str, Any]) -> torch.Tensor:
        pixel_values = mm_kwargs["pixel_values"]
        if isinstance(pixel_values, list):
            pixel_values = torch.cat(pixel_values)
        if pixel_values.ndim in (3, 5):
            pixel_values = pixel_values.reshape(
                pixel_values.shape[0] * pixel_values.shape[1],
                *pixel_values.shape[2:],
            )
        return pixel_values

    def get_encoder_cudagraph_item_specs(self, mm_kwargs: dict[str, Any]):
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderItemSpec

        kh, kw = self.config.vision_config.merge_kernel_size
        return [
            EncoderItemSpec(
                input_size=t * h * w,
                output_tokens=(h // kh) * (w // kw),
            )
            for t, h, w in self._get_grid_thws(mm_kwargs)
        ]

    def select_encoder_cudagraph_items(
        self, mm_kwargs: dict[str, Any], indices: list[int]
    ) -> dict[str, Any]:
        grid_thws = self._get_grid_thws(mm_kwargs)
        pixel_values = self._get_pixel_values(mm_kwargs)
        source_grid = mm_kwargs["grid_thws"]

        if not indices:
            empty_grid = (
                source_grid[:0] if isinstance(source_grid, torch.Tensor) else []
            )
            return {"pixel_values": pixel_values[:0], "grid_thws": empty_grid}

        patch_counts = [t * h * w for t, h, w in grid_thws]
        offsets = [0]
        for count in patch_counts:
            offsets.append(offsets[-1] + count)
        selected_pixel_values = torch.cat(
            [pixel_values[offsets[i] : offsets[i + 1]] for i in indices]
        )
        grid_device = (
            source_grid.device if isinstance(source_grid, torch.Tensor) else None
        )
        selected_grid = torch.tensor(
            [grid_thws[i] for i in indices],
            dtype=torch.long,
            device=grid_device,
        )
        return {"pixel_values": selected_pixel_values, "grid_thws": selected_grid}

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
    ):
        from vllm.v1.worker.encoder_cudagraph_defs import (
            EncoderCudaGraphCaptureInputs,
        )

        kh, kw = self.config.vision_config.merge_kernel_size
        per_item_output = (token_budget + max_batch_size - 1) // max_batch_size
        rope = self.vision_tower.encoder.rope_2d
        max_output_width = rope.max_width // kw
        max_output_height = rope.max_height // kh
        output_width = min(math.ceil(math.sqrt(per_item_output)), max_output_width)
        output_height = (per_item_output + output_width - 1) // output_width
        if output_height > max_output_height:
            output_height = max_output_height
            output_width = (per_item_output + output_height - 1) // output_height
        if output_width > max_output_width:
            raise ValueError(
                f"Encoder CUDA graph budget {token_budget} exceeds K3 RoPE "
                f"capacity for max_batch_size={max_batch_size}"
            )
        grid_thws = [
            [1, output_height * kh, output_width * kw] for _ in range(max_batch_size)
        ]

        patch_size: int | tuple[int, int] = self.config.vision_config.patch_size
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        total_patches = sum(t * h * w for t, h, w in grid_thws)
        pixel_values = torch.randn(
            total_patches,
            3,
            patch_size[0],
            patch_size[1],
            device=device,
            dtype=dtype,
        )
        metadata = self.vision_tower.prepare_encoder_cudagraph_metadata(
            grid_thws,
            max_batch_size=max_batch_size,
            max_seqlen_override=max(
                token_budget * kh * kw,
                max(t * h * w for t, h, w in grid_thws),
            ),
            device=device,
        )
        return EncoderCudaGraphCaptureInputs(
            values=metadata | {"pixel_values": pixel_values}
        )

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ):
        from vllm.v1.worker.encoder_cudagraph_defs import (
            EncoderCudaGraphReplayBuffers,
        )

        pixel_values = self._get_pixel_values(mm_kwargs)
        metadata = self.vision_tower.prepare_encoder_cudagraph_metadata(
            self._get_grid_thws(mm_kwargs),
            max_batch_size=max_batch_size,
            device=pixel_values.device,
        )
        return EncoderCudaGraphReplayBuffers(
            values=metadata | {"pixel_values": pixel_values}
        )

    def _project_encoder_features(self, image_features: torch.Tensor) -> torch.Tensor:
        projector_dtype = next(self.mm_projector.parameters()).dtype
        if image_features.dtype != projector_dtype:
            image_features = image_features.to(projector_dtype)
        output = self.mm_projector(image_features)
        return output.reshape(-1, output.shape[-1])

    def encoder_cudagraph_forward(
        self,
        values: dict[str, torch.Tensor],
        path: str = "default",
    ) -> torch.Tensor:
        pixel_values = values.pop("pixel_values")
        image_features = self.vision_tower(pixel_values, None, encoder_metadata=values)
        return self._project_encoder_features(image_features)

    def encoder_eager_forward(
        self,
        mm_kwargs: dict[str, Any],
        path: str = "default",
    ) -> torch.Tensor:
        image_features = self.vision_tower(
            self._get_pixel_values(mm_kwargs).to(
                next(self.vision_tower.parameters()).dtype
            ),
            self._get_grid_thws(mm_kwargs),
        )
        return self._project_encoder_features(torch.cat(image_features))

    def _maybe_ignore_quant_config(
        self, quant_config: QuantizationConfig | None
    ) -> QuantizationConfig | None:
        if isinstance(quant_config, compressed_tensors.CompressedTensorsConfig):
            return None
        return quant_config

    def _parse_and_validate_media_input(
        self, **kwargs: object
    ) -> KimiK25MediaPixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        grid_thws = kwargs.pop("grid_thws", None)
        if pixel_values is None:
            return None

        if isinstance(pixel_values, list):
            pixel_values = torch.cat(cast(list[torch.Tensor], pixel_values), dim=0)
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError(
                "pixel_values must be a tensor or a list of tensors, "
                f"got {type(pixel_values)}"
            )

        if len(pixel_values.shape) == 5 or len(pixel_values.shape) == 3:
            pixel_values = pixel_values.reshape(
                pixel_values.shape[0] * pixel_values.shape[1], *pixel_values.shape[2:]
            )

        target_dtype = next(self.vision_tower.parameters()).dtype
        pixel_values = pixel_values.to(target_dtype)
        assert isinstance(grid_thws, torch.Tensor), (
            f"expect grid_thws to be a tensor, got {type(grid_thws)}"
        )
        grid_thws = grid_thws.reshape(-1, grid_thws.shape[-1])
        assert grid_thws.ndim == 2 and grid_thws.size(1) == 3, (
            f"unexpected shape for grid_thws: {grid_thws.shape}"
        )

        return KimiK25MediaPixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            grid_thws=grid_thws,
        )

    def _process_media_input(
        self, media_input: KimiK25MediaPixelInputs
    ) -> list[torch.Tensor]:
        media_features = vision_tower_forward(
            self.vision_tower,
            media_input["pixel_values"],
            media_input["grid_thws"],
            mm_projector=self.mm_projector,
            use_data_parallel=self.use_data_parallel,
        )
        return media_features

    def embed_multimodal(self, **kwargs: object) -> NestedTensors | None:
        media_input = self._parse_and_validate_media_input(**kwargs)
        if media_input is None:
            return None
        return self._process_media_input(media_input)

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.language_model.compute_logits(hidden_states)

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.language_model.mamba_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs
        )

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.language_model.mamba_cache.get_seqlen_agnostic_capture_inputs(
            batch_size
        )

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        text_config = vllm_config.model_config.hf_config.text_config
        temp_vllm_config = vllm_config.with_hf_config(text_config)
        return KimiLinearForCausalLM.get_mamba_state_dtype_from_config(temp_vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        text_config = vllm_config.model_config.hf_config.text_config
        temp_vllm_config = vllm_config.with_hf_config(text_config)
        return KimiLinearForCausalLM.get_mamba_state_shape_from_config(temp_vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return KimiLinearForCausalLM.get_mamba_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
