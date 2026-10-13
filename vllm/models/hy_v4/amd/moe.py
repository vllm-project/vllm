# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dense and routed MoE blocks for HY V4 on ROCm."""

import weakref

import torch
from torch import nn
from transformers import PreTrainedConfig

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import get_current_vllm_config
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    GateLinear,
    RoutedExperts,
)
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMxFp8Config,
    ModelOptMxFp8FusedMoE,
)
from vllm.platforms.rocm import on_gfx950

logger = init_logger(__name__)


class HYV4FeedForward(nn.Module):
    """Dense SwiGLU block used by dense and shared HY V4 experts."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


def _scale_as_e8m0_bytes(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.float8_e8m0fnu:
        return scale.contiguous().view(torch.uint8)
    if scale.dtype == torch.uint8:
        return scale.contiguous()
    if scale.dtype == torch.float32:
        bits = scale.contiguous().view(torch.int32)
        exponent = (bits >> 23).to(torch.uint8)
        if not torch.equal(bits, exponent.to(torch.int32) << 23):
            raise ValueError("HY V4 shared-expert scales are not exact E8M0.")
        return exponent
    raise ValueError(f"Unsupported HY V4 shared-expert scale dtype: {scale.dtype}.")


def _expand_fp8_block_scale(scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Return one E8M0 scale per output row and 32 consecutive K values."""
    scale = _scale_as_e8m0_bytes(scale)
    expected_128 = (rows // 128, cols // 128)
    expected_32 = (rows, cols // 32)
    if tuple(scale.shape) == expected_32:
        return scale
    if tuple(scale.shape) != expected_128:
        raise ValueError(
            f"Expected HY V4 FP8 block scale shape {expected_128}, "
            f"got {tuple(scale.shape)}."
        )
    return scale.repeat_interleave(128, dim=0).repeat_interleave(4, dim=1)


@torch.no_grad()
def _prepare_hyv4_fp8_shared_expert(
    shared_expert: HYV4FeedForward,
    routed_w1: torch.Tensor,
    routed_w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert the ModelOpt MXFP8 shared MLP to AITER's FHMoE layout."""
    shared_w1 = shared_expert.gate_up_proj.weight
    shared_w2 = shared_expert.down_proj.weight
    shared_s1 = getattr(shared_expert.gate_up_proj, "weight_scale", None)
    shared_s2 = getattr(shared_expert.down_proj, "weight_scale", None)
    if shared_s1 is None or shared_s2 is None:
        raise ValueError("HY V4 shared expert must have E8M0 weight scales.")
    if shared_w1.dtype != torch.float8_e4m3fn or shared_w2.dtype != torch.float8_e4m3fn:
        raise ValueError("HY V4 shared expert weights must be FP8 E4M3.")

    hidden_size = routed_w1.shape[-1]
    intermediate_size = routed_w2.shape[-1]
    native_intermediate = shared_w1.shape[0] // 2
    if tuple(shared_w1.shape[1:]) != (hidden_size,):
        raise ValueError(
            f"HY V4 shared W1 has shape {tuple(shared_w1.shape)}, expected "
            f"(2*I, {hidden_size})."
        )
    if tuple(shared_w2.shape) != (hidden_size, native_intermediate):
        raise ValueError(
            f"HY V4 shared W2 has shape {tuple(shared_w2.shape)}, expected "
            f"({hidden_size}, {native_intermediate})."
        )
    if native_intermediate > intermediate_size:
        raise ValueError(
            "HY V4 shared expert is wider than the routed expert partition: "
            f"{native_intermediate} > {intermediate_size}."
        )
    if hidden_size % 32 or intermediate_size % 32:
        raise ValueError("HY V4 FHMoE dimensions must be divisible by 32.")

    # FHMoE uses the routed expert's padded intermediate width. Padding rows
    # decode as zero weights and use E8M0 exponent 127 (scale 1).
    w1 = torch.zeros(
        (1, 2 * intermediate_size, hidden_size),
        dtype=shared_w1.dtype,
        device=shared_w1.device,
    )
    w1[:, :native_intermediate].copy_(shared_w1[:native_intermediate])
    w1[:, intermediate_size : intermediate_size + native_intermediate].copy_(
        shared_w1[native_intermediate:]
    )
    w2 = torch.zeros(
        (1, hidden_size, intermediate_size),
        dtype=shared_w2.dtype,
        device=shared_w2.device,
    )
    w2[:, :, :native_intermediate].copy_(shared_w2)

    native_s1 = _expand_fp8_block_scale(shared_s1, 2 * native_intermediate, hidden_size)
    native_s2 = _expand_fp8_block_scale(shared_s2, hidden_size, native_intermediate)
    s1 = torch.full(
        (1, 2 * intermediate_size, hidden_size // 32),
        0x7F,
        dtype=torch.uint8,
        device=shared_w1.device,
    )
    s1[:, :native_intermediate].copy_(native_s1[:native_intermediate])
    s1[:, intermediate_size : intermediate_size + native_intermediate].copy_(
        native_s1[native_intermediate:]
    )
    s2 = torch.full(
        (1, hidden_size, intermediate_size // 32),
        0x7F,
        dtype=torch.uint8,
        device=shared_w2.device,
    )
    s2[:, :, : native_intermediate // 32].copy_(native_s2)

    return (
        rocm_aiter_ops.moe_shuffle_weight(
            w1,
            experts_cnt=1,
            is_guinterleave=True,
            gate_up=True,
        ).contiguous(),
        rocm_aiter_ops.moe_shuffle_weight(
            w2,
            experts_cnt=1,
            is_guinterleave=True,
            gate_up=False,
        ).contiguous(),
        rocm_aiter_ops.moe_shuffle_scale(
            s1.view(-1, s1.shape[-1]).view(torch.float8_e8m0fnu),
            experts_cnt=1,
            is_guinterleave=True,
            gate_up=True,
        ).contiguous(),
        rocm_aiter_ops.moe_shuffle_scale(
            s2.view(-1, s2.shape[-1]).view(torch.float8_e8m0fnu),
            experts_cnt=1,
            is_guinterleave=True,
            gate_up=False,
        ).contiguous(),
    )


class HYV4HeterogeneousMxfp8MoEMethod(ModelOptMxFp8FusedMoE):
    """Finalize routed MXFP8 weights before attaching the shared operand."""

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        super().process_weights_after_loading(layer)
        assert isinstance(layer, HYV4HeterogeneousSharedRoutedExperts)
        # On initial load the shared projections are already final. During
        # layerwise reload, sibling modules can finish in checkpoint order, so
        # defer this cross-layer conversion to process_weights_after_reload().
        if layer._shared_kernel_storage is None:
            layer.prepare_heterogeneous_shared_expert()


class HYV4HeterogeneousSharedRoutedExperts(RoutedExperts):
    """MXFP8 routed experts with one separately stored MXFP8 shared expert."""

    def __init__(
        self,
        *args,
        shared_expert: HYV4FeedForward,
        shared_expert_id: int,
        **kwargs,
    ):
        self._shared_expert_ref = weakref.ref(shared_expert)
        self.shared_expert_id = shared_expert_id
        super().__init__(*args, **kwargs)
        if self.expert_map_manager.num_fused_shared_experts != 1:
            raise ValueError("HY V4 heterogeneous fusion requires one shared expert.")
        if self.shared_expert_id != self.global_num_experts:
            raise ValueError("HY V4 shared expert ID must be the appended slot.")
        hidden_size = self.moe_config.hidden_dim
        intermediate_size = self.moe_config.intermediate_size_per_partition
        device = self.w13_weight.device
        self.register_buffer(
            "shared_w1",
            torch.empty(
                (1, 2 * intermediate_size, hidden_size),
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "shared_w2",
            torch.empty(
                (1, hidden_size, intermediate_size),
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "shared_w1_scale",
            torch.empty(
                (2 * intermediate_size, hidden_size // 32),
                dtype=torch.float8_e8m0fnu,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "shared_w2_scale",
            torch.empty(
                (hidden_size, intermediate_size // 32),
                dtype=torch.float8_e8m0fnu,
                device=device,
            ),
            persistent=False,
        )
        # Set after the initial post-load conversion. Unlike registered
        # buffers, this tuple is not replaced with temporary tensors during
        # layerwise reload, so it retains the storage captured by CUDA graphs.
        self._shared_kernel_storage: tuple[torch.Tensor, ...] | None = None

    def _get_quant_method(self, prefix, quant_config, moe_config):
        quant_method = super()._get_quant_method(prefix, quant_config, moe_config)
        if not isinstance(quant_method, ModelOptMxFp8FusedMoE):
            raise ValueError(
                "HY V4 heterogeneous fusion requires ModelOpt MXFP8 routed "
                f"experts, got {type(quant_method).__name__}."
            )
        return HYV4HeterogeneousMxfp8MoEMethod(
            quant_method.quant_config,
            moe_config,
        )

    @torch.no_grad()
    def prepare_heterogeneous_shared_expert(self) -> None:
        shared_expert = self._shared_expert_ref()
        if shared_expert is None:
            raise RuntimeError("HY V4 shared expert module was released.")
        kernel = getattr(self.quant_method, "moe_kernel", None)
        fused_experts = getattr(kernel, "fused_experts", None)
        from vllm.model_executor.layers.fused_moe.experts.aiter_mxfp8_moe import (
            AiterMxfp8Experts,
        )

        if not isinstance(fused_experts, AiterMxfp8Experts):
            raise RuntimeError(
                "HY V4 heterogeneous fusion requires the AITER MXFP8 backend; "
                f"got {type(fused_experts).__name__}."
            )
        prepared = _prepare_hyv4_fp8_shared_expert(
            shared_expert,
            self.w13_weight,
            self.w2_weight,
        )
        storage = getattr(self, "_shared_kernel_storage", None)
        if storage is None:
            storage = (
                self.shared_w1,
                self.shared_w2,
                self.shared_w1_scale,
                self.shared_w2_scale,
            )
            self._shared_kernel_storage = storage
        for target, source in zip(
            storage,
            prepared,
        ):
            if target.shape != source.shape or target.dtype != source.dtype:
                raise RuntimeError(
                    "HY V4 prepared shared-expert storage changed shape or dtype: "
                    f"expected {tuple(target.shape)}/{target.dtype}, got "
                    f"{tuple(source.shape)}/{source.dtype}."
                )
            target.copy_(source)
        shared_w1, shared_w2, shared_w1_scale, shared_w2_scale = storage
        shared_w1.is_shuffled = True
        shared_w2.is_shuffled = True
        fused_experts.set_shared_expert(
            shared_w1,
            shared_w2,
            shared_w1_scale,
            shared_w2_scale,
            self.shared_expert_id,
        )
        logger.info_once(
            "HY V4 enabled AITER heterogeneous MXFP8 MoE: routed and appended "
            "shared experts use separate FP8 payload/E8M0 scale operands."
        )

    def process_weights_after_reload(self) -> None:
        """Rebuild shared operands after all sibling weights are restored."""
        self.prepare_heterogeneous_shared_expert()

    def forward_modular(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if shared_experts is not None or shared_experts_input is not None:
            raise ValueError("HY V4 heterogeneous fusion owns the shared path.")
        expected_shape = (x.shape[0], self.moe_config.experts_per_token + 1)
        if (
            tuple(topk_ids.shape) != expected_shape
            or tuple(topk_weights.shape) != expected_shape
        ):
            raise ValueError(
                "HY V4 heterogeneous routes must contain routed top-k plus "
                "one shared slot; got IDs/weights shapes "
                f"{tuple(topk_ids.shape)}/{tuple(topk_weights.shape)}."
            )
        return super().forward_modular(
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=None,
            shared_experts_input=None,
        )


def _hyv4_fusion_weights_may_be_offloaded(
    offload_config,
    prefix: str,
) -> bool:
    """Whether the active offloader can move an FHMoE source parameter."""
    if offload_config is None:
        return False

    uva = getattr(offload_config, "uva", None)
    prefetch = getattr(offload_config, "prefetch", None)
    cpu_offload_gb = getattr(uva, "cpu_offload_gb", 0)
    offload_group_size = getattr(prefetch, "offload_group_size", 0)
    backend = getattr(offload_config, "offload_backend", "auto")
    if backend == "auto":
        if offload_group_size > 0:
            backend = "prefetch"
        elif cpu_offload_gb > 0:
            backend = "uva"
        else:
            return False

    selected_params: set[str]
    if backend == "prefetch":
        if offload_group_size <= 0:
            return False
        selected_params = getattr(prefetch, "offload_params", set())
    elif backend == "uva":
        if cpu_offload_gb <= 0:
            return False
        selected_params = getattr(uva, "cpu_offload_params", set())
    else:
        return True

    # An empty selector means every parameter is eligible. Otherwise mirror
    # the offloaders' exact parameter-segment matching.
    if not selected_params:
        return True
    relevant_params = (
        f"{prefix}.experts.w13_weight",
        f"{prefix}.experts.w2_weight",
        f"{prefix}.experts.w13_weight_scale",
        f"{prefix}.experts.w2_weight_scale",
        f"{prefix}.shared_experts.gate_up_proj.weight",
        f"{prefix}.shared_experts.gate_up_proj.weight_scale",
        f"{prefix}.shared_experts.down_proj.weight",
        f"{prefix}.shared_experts.down_proj.weight_scale",
    )
    return any(
        f".{selected}." in f".{parameter}."
        for selected in selected_params
        for parameter in relevant_params
    )


def _use_hyv4_heterogeneous_fusion(
    config: PreTrainedConfig,
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> bool:
    vllm_config = get_current_vllm_config()
    parallel_config = vllm_config.parallel_config
    offload_config = getattr(vllm_config, "offload_config", None)
    relevant_prefixes = (
        f"{prefix}.experts",
        f"{prefix}.shared_experts.gate_up_proj",
        f"{prefix}.shared_experts.down_proj",
    )
    is_layer_excluded = getattr(quant_config, "is_layer_excluded", None)
    relevant_layer_excluded = (
        any(is_layer_excluded(layer_prefix) for layer_prefix in relevant_prefixes)
        if callable(is_layer_excluded)
        else bool(getattr(quant_config, "exclude_modules", None))
    )
    return bool(
        getattr(config, "num_shared_experts", 0) == 1
        and envs.VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS
        and isinstance(quant_config, ModelOptMxFp8Config)
        and quant_config.is_checkpoint_mxfp8_serialized
        and not relevant_layer_excluded
        and config.hidden_size == 6144
        and config.expert_hidden_dim == 2048
        and config.num_experts == 256
        and config.num_experts_per_tok == 8
        and get_tensor_model_parallel_world_size() == 8
        and vllm_config.model_config.dtype == torch.bfloat16
        and on_gfx950()
        and vllm_config.kernel_config.moe_backend == "aiter"
        and rocm_aiter_ops.fused_moe_supports_hy4_shared_expert(
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        and not _hyv4_fusion_weights_may_be_offloaded(offload_config, prefix)
        and vllm_config.load_config.load_format != "ipc_cache"
        and getattr(vllm_config, "weight_transfer_config", None) is None
        and not parallel_config.enable_expert_parallel
        and not parallel_config.enable_eplb
        and parallel_config.data_parallel_size == 1
        and parallel_config.prefill_context_parallel_size == 1
    )


def _assert_hyv4_eplb_consistency(
    *,
    layer_enabled: bool,
    configured_enabled: bool,
    num_redundant_experts: int,
) -> None:
    assert layer_enabled == configured_enabled, (
        "HY V4 MoE layer EPLB setting must match ParallelConfig.enable_eplb."
    )
    assert layer_enabled or num_redundant_experts == 0, (
        "HY V4 redundant experts require EPLB to be enabled."
    )


class HYV4MoEFused(nn.Module):
    """HY V4 sigmoid-routed MoE with optional FHMoE shared fusion."""

    def __init__(
        self,
        config: PreTrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )
        top_k = config.num_experts_per_tok
        intermediate_size = config.expert_hidden_dim
        router_scaling_factor = getattr(config, "router_scaling_factor", 1.0)
        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        _assert_hyv4_eplb_consistency(
            layer_enabled=self.enable_eplb,
            configured_enabled=vllm_config.parallel_config.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
        )
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.gate = GateLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            out_dtype=torch.float32,
            params_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.shared_experts: HYV4FeedForward | None = None
        if config.num_shared_experts > 0:
            self.shared_experts = HYV4FeedForward(
                hidden_size=config.hidden_size,
                intermediate_size=config.expert_hidden_dim * config.num_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts",
                reduce_results=False,
            )

        self.fuse_heterogeneous_shared_expert = bool(
            self.shared_experts is not None
            and _use_hyv4_heterogeneous_fusion(config, quant_config, prefix)
        )
        assert not self.fuse_heterogeneous_shared_expert or (
            not self.enable_eplb and self.n_redundant_experts == 0
        ), "HY V4 heterogeneous shared-expert fusion is incompatible with EPLB."
        if (
            self.shared_experts is not None
            and envs.VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS
            and not self.fuse_heterogeneous_shared_expert
        ):
            logger.warning_once(
                "HY V4 heterogeneous shared-expert fusion was requested but "
                "the model, quantization, parallel configuration, platform, or "
                "AITER FHMoE capability or safe token range is incompatible; "
                "using the separate shared-expert path."
            )
        self.expert_bias = nn.Parameter(
            torch.empty(config.num_experts, dtype=torch.float32)
        )
        raw_swiglu_limit = getattr(config, "swiglu_limit", 0)
        use_swiglu_clamp = bool(raw_swiglu_limit) and float(raw_swiglu_limit) > 0
        moe_swiglu_limit = float(raw_swiglu_limit) if use_swiglu_clamp else None

        fuse_shared = self.fuse_heterogeneous_shared_expert
        self.experts = FusedMoEFactory(
            num_experts=self.n_routed_experts,
            top_k=top_k,
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            renormalize=config.route_norm,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            scoring_func="sigmoid",
            use_grouped_topk=True,
            num_expert_group=1,
            topk_group=1,
            routed_scaling_factor=router_scaling_factor,
            e_score_correction_bias=self.expert_bias,
            shared_experts=None if fuse_shared else self.shared_experts,
            n_shared_experts=config.num_shared_experts if fuse_shared else None,
            fuse_shared_experts=fuse_shared,
            swiglu_limit=moe_swiglu_limit,
            routed_experts_cls=(
                HYV4HeterogeneousSharedRoutedExperts if fuse_shared else None
            ),
            routed_experts_args=(
                {
                    "shared_expert": self.shared_experts,
                    "shared_expert_id": self.n_routed_experts,
                }
                if fuse_shared
                else None
            ),
        )
        self.prefix = prefix
        if use_swiglu_clamp:
            logger.debug_once(
                "HYV4MoEFused: swiglu_limit=%.1f enabled for routed experts",
                moe_swiglu_limit,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        router_logits, _ = self.gate(hidden_states)
        output = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        return output.view(orig_shape)
