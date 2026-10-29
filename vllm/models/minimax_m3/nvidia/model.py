# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only MiniMax M3 (text backbone) model.

The MiniMax-M3-preview config selects a single set of branches:
    * qk_norm_type == "per_head"
    * hidden_act == "swigluoai"
    * use_gemma_norm == True  -> Gemma-style RMSNorm everywhere
    * attention_output_gate == False
    * scoring_func == "sigmoid" with a routing-bias correction term
    * sparse_attention_config present -> a subset of layers run the extra
      "index" attention branch.
"""

from collections.abc import Iterable

import torch
from torch import nn
from transformers import PreTrainedConfig

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import set_default_quant_scales
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
    fused_allreduce_gemma_rms_norm,
)
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    MEGA_MOE_NVFP4,
    DeepGemmMegaMoEExperts,
    MegaMoEActivation,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
    TrtLlmNvFp4ExpertsModular,
    TrtLlmNvFp4ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.moe_output import (
    MoEOutput,
    UnfinalizedMoEOutput,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import _unpack
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
    get_input_quant_key,
)
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    MinimaxM3QKVParallelLinearWithIndexer,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp8Dynamic,
    kNvfp4DynamicToken,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
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
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    init_vllm_registered_model,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    spec_decode_needs_target_embed,
)
from vllm.model_executor.models.vision import run_dp_sharded_mrope_vision_model
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_shard,
)
from vllm.models.minimax_m3.common.encoder_cudagraph import (
    MiniMaxM3EncoderCudaGraphMixin,
)
from vllm.models.minimax_m3.common.indexer import MiniMaxM3Indexer
from vllm.models.minimax_m3.common.mm_preprocess import (
    MiniMaxM3VLDummyInputsBuilder,
    MiniMaxM3VLMultiModalProcessor,
    MiniMaxM3VLProcessingInfo,
)
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseBackend,
    MiniMaxM3SparseImpl,
    select_main_backend_and_impl_cls,
)
from vllm.models.minimax_m3.common.vision_tower import MiniMaxVLVisionModel
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.flashinfer import is_flashinfer_autotuning
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

logger = init_logger(__name__)

# Largest batch whose MoE top-k finalize runs inside the next layer's fused
# all-reduce + input norm + MXFP8 quantization. Larger batches finalize in the
# MoE and use the regular fused all-reduce + norm.
FUSED_MOE_FINALIZE_MAX_TOKENS = 32

_QKV_FUSED_EPILOGUE_MIN_TOKENS = 512


def _supports_qkv_fused_epilogue(attn: "MiniMaxM3SparseAttention") -> bool:
    """Whether minimax_m3_qkv_index_fused_epilogue can replace this layer's
    qkv_proj + norm/RoPE/cache-insert chain."""
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutedslMxfp8LinearKernel,
        FlashInferCutlassMxfp8LinearKernel,
    )

    if not current_platform.is_device_capability_family(100):
        return False
    # Both keep the weight K-major and its scales F8_128x4-swizzled.
    kernel = getattr(attn.qkv_proj.quant_method, "kernel", None)
    if not isinstance(
        kernel, (FlashInferCutedslMxfp8LinearKernel, FlashInferCutlassMxfp8LinearKernel)
    ):
        return False
    nkv = attn.num_kv_heads
    return (
        attn.use_nvfp4_kv
        and attn.indexer_kv_dtype == "nvfp4"
        and nkv in (1, 2, 4)
        and attn.num_heads == 16 * nkv
        and attn.num_idx_heads == nkv
        and attn.head_dim == attn.idx_head_dim == 128
        and attn.rotary_emb.rotary_dim == 64
        and attn.hidden_size % 128 == 0
    )


def _sparse_attention_layer_ids(config: PreTrainedConfig) -> set[int]:
    """Layer ids whose attention runs the extra sparse "index" branch."""
    cfg = getattr(config, "sparse_attention_config", None)
    if not cfg:
        return set()
    freq = cfg.get("sparse_attention_freq")
    if freq is None:
        return set()
    return {i for i, f in enumerate(freq) if f != 0}


def _is_moe_layer(config: PreTrainedConfig, layer_id: int) -> bool:
    """Whether this layer's MLP is a sparse MoE block (vs a dense MLP)."""
    moe_layer_freq = getattr(config, "moe_layer_freq", None)
    if moe_layer_freq is None:
        return True
    return moe_layer_freq[layer_id] != 0


def _use_mega_moe(vllm_config: VllmConfig) -> bool:
    """Whether MoE layers run the DeepGEMM NVFP4 MegaMoE kernel.

    Opt in with ``--kernel-config '{"moe_backend": "deep_gemm_mega_moe"}'``. Only
    checkpoints with NVFP4 routed experts (MiniMax-M3.1) are supported.
    """
    if vllm_config.kernel_config.moe_backend != "deep_gemm_mega_moe":
        return False
    moe_quant_algo = getattr(vllm_config.quant_config, "moe_quant_algo", None)
    if moe_quant_algo != "NVFP4":
        raise NotImplementedError(
            "MiniMax M3 MegaMoE requires a checkpoint with NVFP4 routed experts "
            f"(moe_quant_algo: NVFP4), got moe_quant_algo={moe_quant_algo!r}. "
            "Pick a different moe backend for this checkpoint."
        )
    if not vllm_config.parallel_config.enable_expert_parallel:
        raise NotImplementedError(
            "MiniMax M3 MegaMoE requires expert parallel. Enable it with "
            "--enable-expert-parallel."
        )
    return True


# MiniMax-M3.1 QAT quantizes the FC2 input with a fixed global scale of 16;
# checkpoints carry a placeholder ``w2.input_global_scale`` of 1.0.
MEGA_MOE_FC2_INPUT_SCALE = 1.0 / 16.0


def _mega_moe_expert_params_mapping(
    num_experts: int,
) -> list[tuple[str, str, int, str]]:
    """Map ``experts.{e}.w{1,2,3}.*`` to the MegaMoE ``w13_*``/``w2_*`` params."""
    return [
        (
            "experts.w2_" if shard_id == "w2" else "experts.w13_",
            f"experts.{expert_id}.{shard_id}.",
            expert_id,
            shard_id,
        )
        for expert_id in range(num_experts)
        for shard_id in ("w1", "w2", "w3")
    ]


def _mega_moe_expert_weight(
    name: str, weight: torch.Tensor
) -> tuple[str, torch.Tensor] | None:
    """Convert a compressed-tensors NVFP4 routed-expert tensor to the ModelOpt
    convention ``DeepGemmMegaMoEExperts`` loads, or None to drop it.

    The global scales become dequantization scales. The FC1 input is quantized
    per token, so its input scale is dropped, and the FC2 input scale is the
    fixed QAT scale whatever the checkpoint says.
    """
    prefix, _, suffix = name.rpartition(".")
    if suffix == "weight_packed":
        return f"{prefix}.weight", weight
    if suffix == "weight_global_scale":
        return f"{prefix}.weight_scale_2", 1.0 / weight.float()
    if suffix == "input_global_scale":
        if not prefix.endswith(".w2"):
            return None
        scale = torch.full_like(weight, MEGA_MOE_FC2_INPUT_SCALE, dtype=torch.float32)
        return f"{prefix}.input_scale", scale
    return name, weight


class MiniMAXGemmaRMSNorm(nn.Module):
    """Gemma-style RMS normalization backed by FlashInfer kernels.

    When ``residual`` is given, the fused add + norm runs in place and the
    updated ``(x, residual)`` pair is returned.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from flashinfer.norm import gemma_fused_add_rmsnorm, gemma_rmsnorm

        if residual is None:
            return gemma_rmsnorm(x, self.weight, self.variance_epsilon)

        # gemma_fused_add_rmsnorm mutates x and residual in place.
        gemma_fused_add_rmsnorm(x, residual, self.weight, self.variance_epsilon)
        return x, residual


class MiniMaxM3MLP(nn.Module):
    """Dense SwiGLU-OAI MLP (used by the leading dense layers)."""

    def __init__(
        self,
        config: PreTrainedConfig,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        disable_tp: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=disable_tp,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=disable_tp,
            prefix=f"{prefix}.down_proj",
        )
        if config.hidden_act != "swigluoai":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only swigluoai is supported."
            )
        # gate * sigmoid(alpha * gate) * (up + beta), with both halves clamped.
        self.act_fn = SiluAndMulWithClamp(
            swiglu_limit=config.swiglu_limit,
            alpha=config.swiglu_alpha,
            beta=config.swiglu_beta,
        )
        from vllm.models.minimax_m3.nvidia.ops import dense_gemm1_mxfp8

        # At or below the shared-experts stream threshold the persistent kernel
        # would contend with the routed experts on the aux stream.
        self.fused_gemm1_min_tokens: int | None = None
        if dense_gemm1_mxfp8.can_fuse_gemm1(
            self.gate_up_proj, self.down_proj, self.act_fn
        ):
            self.fused_gemm1_min_tokens = max(
                dense_gemm1_mxfp8.MIN_TOKENS,
                envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens = x.shape[0]
        # Run unfused under FlashInfer autotuning so gate_up_proj is tuned too
        if (
            self.fused_gemm1_min_tokens is not None
            and num_tokens > self.fused_gemm1_min_tokens
            and not is_flashinfer_autotuning()
        ):
            from vllm.models.minimax_m3.nvidia.ops.dense_gemm1_mxfp8 import (
                fused_gemm1,
            )

            x = fused_gemm1(x, self.gate_up_proj, self.act_fn)
        else:
            gate_up, _ = self.gate_up_proj(x)
            x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MiniMaxM3MoE(nn.Module):
    """Sigmoid-routed MoE block with a routing-bias correction and a shared
    expert."""

    def __init__(
        self,
        config: PreTrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
        use_mega_moe: bool = False,
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size > config.num_local_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_local_experts}."
            )

        self.use_mega_moe = use_mega_moe
        self.reduce_results = reduce_results
        self.scoring_func = config.scoring_func
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.n_shared_experts = getattr(config, "n_shared_experts", None)

        # Sigmoid routing uses a per-expert score-correction bias for selection.
        self.use_routing_bias = getattr(config, "use_routing_bias", False)
        if self.use_routing_bias:
            self.e_score_correction_bias = nn.Parameter(
                torch.empty(config.num_local_experts, dtype=torch.float32)
            )
            self.e_score_correction_bias.weight_loader = (
                MiniMaxM3MoE.ebias_weight_loader
            )
        else:
            self.e_score_correction_bias = None

        # Router weights are stored in fp32; GateLinear upcasts the bf16
        # activations and computes the gate in fp32 (fp32 router logits).
        self.gate = GateLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            params_dtype=torch.float32,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

        self.shared_experts: MiniMaxM3MLP | None = None
        if self.use_mega_moe:
            self._init_mega_moe_experts(config, quant_config, prefix)
            return

        if self.n_shared_experts:
            self.shared_experts = MiniMaxM3MLP(
                config=config,
                intermediate_size=config.intermediate_size * self.n_shared_experts,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        self.experts = FusedMoEFactory(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            scoring_func=config.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            renormalize=True,
            # w13 (gate_up_proj) is loaded packed via MergedColumnParallelLinear
            # ([all gates; all ups]), so use the uninterleaved SwiGLU-OAI variant
            # rather than the interleaved gpt-oss layout.
            activation="swigluoai_uninterleave",
            swiglu_limit=config.swiglu_limit,
            swiglu_alpha=config.swiglu_alpha,
            swiglu_beta=config.swiglu_beta,
            routed_scaling_factor=self.routed_scaling_factor,
            router_logits_dtype=self.gate.out_dtype,
            # The NVFP4 routed experts are QAT'd with a per-token FC1 input
            # scale; their ``input_global_scale`` is a 1.0 placeholder that
            # the quantization config cannot tell from a calibrated scale.
            qat_activation_key=kNvfp4DynamicToken,
            shared_experts=self.shared_experts,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.experts",
        )

    def _init_mega_moe_experts(
        self,
        config: PreTrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        if config.hidden_act != "swigluoai":
            raise ValueError(
                f"MiniMax M3 MegaMoE requires swigluoai, got {config.hidden_act}."
            )
        # Hidden states are replicated across TP ranks, so each rank routes
        # only its own token shard through the EP-wide kernel.
        self.use_sequence_parallel = self.tp_size > 1

        def replicated_shared_experts() -> MiniMaxM3MLP:
            # Replicated, so the kernel can fuse it.
            return MiniMaxM3MLP(
                config=config,
                intermediate_size=config.intermediate_size * self.n_shared_experts,
                quant_config=quant_config,
                reduce_results=False,
                disable_tp=True,
                prefix=f"{prefix}.shared_experts",
            )

        shared_experts = None
        if self.n_shared_experts:
            # Only describes the weights: a fused layer allocates its own.
            with torch.device("meta"):
                shared_experts = replicated_shared_experts()
        ep_group = get_ep_group()
        num_experts = config.num_local_experts
        if num_experts % ep_group.world_size != 0:
            raise ValueError(
                f"MiniMax M3 MegaMoE needs num_experts={num_experts} divisible "
                f"by the EP size {ep_group.world_size}."
            )
        num_local_experts = num_experts // ep_group.world_size
        self.experts = DeepGemmMegaMoEExperts(
            get_current_vllm_config(),
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            experts_start_idx=ep_group.rank_in_group * num_local_experts,
            top_k=self.top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation=MegaMoEActivation(
                "swiglu",
                clamp=config.swiglu_limit,
                alpha=config.swiglu_alpha,
                beta=config.swiglu_beta,
            ),
            weight_format=MEGA_MOE_NVFP4,
            shared_experts=shared_experts,
            prefix=f"{prefix}.experts",
        )
        if shared_experts is not None and not self.experts.has_fused_shared_experts:
            self.shared_experts = replicated_shared_experts()

    @staticmethod
    def ebias_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight.to(torch.float32))

    @property
    def ffn_all_reduce_deferred(self) -> bool:
        if self.use_mega_moe:
            return not self.reduce_results
        return self.experts.moe_config.skip_final_all_reduce

    def defer_finalize(self, max_num_tokens: int) -> bool:
        """Leave the routed top-k reduction of calls up to ``max_num_tokens``
        to the consumer; return whether the experts accepted.

        Only the TRTLLM-Gen NVFP4 experts can stop after GEMM2.
        """
        if self.use_mega_moe:
            return False
        experts_cls = getattr(
            self.experts.routed_experts.quant_method, "experts_cls", None
        )
        if experts_cls not in (TrtLlmNvFp4ExpertsMonolithic, TrtLlmNvFp4ExpertsModular):
            return False
        moe_config = self.experts.moe_config
        moe_config.defer_moe_finalize(max_num_tokens)
        return moe_config.use_deferred_moe_finalize

    def defers_finalize(self, num_tokens: int) -> bool:
        return (
            not self.use_mega_moe
            and self.experts.moe_config.should_defer_moe_finalize(num_tokens)
        )

    def forward_unfinalized(self, hidden_states: torch.Tensor) -> MoEOutput:
        """``forward`` with the routed top-k reduction and all-reduce left open."""
        router_logits, _ = self.gate(hidden_states)
        # The runner's custom op returns tensors only, so run its body directly.
        shared_output, routed = _unpack(
            self.experts._forward_impl(hidden_states, router_logits, hidden_states)
        )
        assert shared_output is not None
        assert isinstance(routed, UnfinalizedMoEOutput)
        return MoEOutput(routed=routed, shared_output=shared_output)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if self.use_mega_moe:
            return self._forward_mega_moe(hidden_states)

        # router_logits: (num_tokens, n_experts); GateLinear casts to fp32.
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        return final_hidden_states.view(num_tokens, hidden_dim)

    def _forward_mega_moe(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.use_sequence_parallel:
            return self._forward_mega_moe_local(hidden_states)

        num_tokens = hidden_states.shape[0]
        # Routing and the experts read the padding mask from the forward
        # context; hand them this rank's shard of it.
        forward_context = None
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            forward_context = get_forward_context()
            is_padding = forward_context.is_padding
            forward_context.is_padding = sp_padding_mask(
                is_padding[:num_tokens] if is_padding is not None else None,
                hidden_states,
            )
        try:
            local_output = self._forward_mega_moe_local(sp_shard(hidden_states))
        finally:
            if forward_context is not None:
                forward_context.is_padding = is_padding
        if self.reduce_results:
            return sp_all_gather(local_output)[:num_tokens]

        # Leave the other ranks' rows zero, so the deferred TP all-reduce (fused
        # into the next RMSNorm) gathers them.
        shard_size, hidden_dim = local_output.shape
        tp_rank = get_tensor_model_parallel_rank()
        output = local_output.new_zeros(shard_size * self.tp_size, hidden_dim)
        output[tp_rank * shard_size : (tp_rank + 1) * shard_size] = local_output
        return output[:num_tokens]

    def _forward_mega_moe_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits, _ = self.gate(hidden_states)
        topk_weights, topk_ids = fused_topk_bias(
            hidden_states=hidden_states,
            gating_output=router_logits,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias.data
            if self.e_score_correction_bias is not None
            else None,
            topk=self.top_k,
            renormalize=True,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        return self.experts(
            hidden_states, topk_weights, topk_ids, shared_experts=self.shared_experts
        )


class MiniMaxM3Attention(nn.Module):
    """Dense attention with per-head QK norm and partial RoPE."""

    def __init__(
        self,
        config: PreTrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        cache_config: CacheConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        # reduce_results=False: the attention all-reduce is fused with the
        # following post_attention_layernorm (GemmaRMSNorm) in the decoder layer
        # via fused_allreduce_gemma_rms_norm.
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Per-head QK norm (qk_norm_type == "per_head", use_gemma_norm == True).
        self.q_norm = MiniMAXGemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MiniMAXGemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Partial RoPE: rotary_dim == head_dim * partial_rotary_factor.
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters={
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": config.partial_rotary_factor,
            },
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        # Fused per-head Gemma QK-norm + partial NeoX RoPE on q/k, in place.

        ops.fused_minimax_m3_qknorm_rope_kv_insert(
            qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            self.rotary_emb.cos_sin_cache,
            positions,
            self.num_heads,
            self.num_kv_heads,
            self.rotary_emb.rotary_dim,
            self.q_norm.variance_epsilon,
            kv_cache_dtype="auto",
        )
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class MiniMaxM3SparseAttention(nn.Module, AttentionLayerBase):
    """Block-sparse attention layer with the lightning-indexer branch.

    This is a merged attention layer: it owns the projections (qkv + index
    q/k), per-head QK norms and RoPE, *and* the attention-backend wiring that a
    generic ``Attention`` layer would normally provide — it binds the
    ``MiniMaxM3SparseBackend`` + main impl, registers the main paged K/V cache,
    and owns the lightning indexer (``MiniMaxM3Indexer``), which holds the
    index-key side cache.

    The index branch (index_{q,k}_proj + index_{q,k}_norm) feeds the sparse
    top-k block selection. M3 always disables the index value/output
    projections (``sparse_disable_index_value`` set for every sparse layer), so
    ``index_{v,o}_proj`` are never created.
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        cache_config: CacheConfig | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # Sparse "index" branch dims. index_q has the same head count as the KV
        # heads (sparse_num_index_heads == num_key_value_heads), so it shards
        # identically -- including replication when tp_size > num_key_value_heads.
        sparse_cfg = config.sparse_attention_config
        self.total_idx_heads = sparse_cfg["sparse_num_index_heads"]
        self.num_idx_heads = self.num_kv_heads
        self.idx_head_dim = sparse_cfg["sparse_index_dim"]
        self.index_q_size = self.num_idx_heads * self.idx_head_dim

        # Single fused projection: q, k, v, index_q, index_k in one GEMM.
        self.qkv_proj = MinimaxM3QKVParallelLinearWithIndexer(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            self.total_idx_heads,
            self.idx_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        # reduce_results=False: the attention all-reduce is fused with the
        # following post_attention_layernorm (GemmaRMSNorm) in the decoder layer
        # via fused_allreduce_gemma_rms_norm.
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Per-head QK norm (qk_norm_type == "per_head", use_gemma_norm == True).
        self.q_norm = MiniMAXGemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MiniMAXGemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Partial RoPE: rotary_dim == head_dim * partial_rotary_factor.
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters={
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": config.partial_rotary_factor,
            },
        )

        self.index_q_norm = MiniMAXGemmaRMSNorm(
            self.idx_head_dim, eps=config.rms_norm_eps
        )
        self.index_k_norm = MiniMAXGemmaRMSNorm(
            self.idx_head_dim, eps=config.rms_norm_eps
        )
        self.index_rotary_emb = self.rotary_emb

        # Attention-backend wiring.
        vllm_config = get_current_vllm_config()
        self.layer_name = f"{prefix}.attn"
        self.kv_cache_dtype = (
            cache_config.cache_dtype if cache_config is not None else "auto"
        )
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, vllm_config.model_config
        )
        # MiniMax-M3 sparse attention owns its KV-cache insert/read path instead
        # of wrapping the generic Attention module. Keep the same runtime scale
        # attributes so FP8 KV reads can honor vLLM's per-layer descale contract.
        set_default_quant_scales(self, register_buffer=True)
        # The fused insert quantizes k/v into NVFP4 pages with these scales.
        self.use_nvfp4_kv = get_kv_quant_mode(self.kv_cache_dtype).is_nvfp4
        # Indexer side-cache dtype, mirroring --kv-cache-dtype for the main
        # cache (--attention-config '{"indexer_kv_dtype": ...}').
        self.indexer_kv_dtype = vllm_config.attention_config.resolve_indexer_kv_dtype(
            "bf16"
        )

        # Shared top-k buffer: the indexer writes the selected blocks into it and
        # the attend impl reads them back (so nothing crosses the eager break as a
        # Python value, which would freeze at capture).
        self.topk_indices_buffer = topk_indices_buffer
        # Indexer (top-k selection) and main attention are separate impls, each
        # picking Triton vs MSA off its cache dtype. impl is AttentionImplBase
        # (broader than the AttentionImpl that AttentionLayerBase annotates).
        self.attn_backend, impl_cls = select_main_backend_and_impl_cls(
            topk_blocks=sparse_cfg["sparse_topk_blocks"],
            kv_cache_dtype=self.kv_cache_dtype,
            num_kv_heads=self.num_kv_heads,
        )
        self.impl: MiniMaxM3SparseImpl = impl_cls(  # type: ignore[assignment]
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype,
            topk_blocks=sparse_cfg["sparse_topk_blocks"],
            sparse_block_size=sparse_cfg["sparse_block_size"],
            msa_decode_backend=(
                vllm_config.attention_config.minimax_m3_msa_decode_backend
            ),
        )
        # Self-contained nn.Module: owns its side cache, selects its impl in init.
        self.indexer = MiniMaxM3Indexer(
            num_kv_heads=self.num_kv_heads,
            scale=self.scaling,
            topk_blocks=sparse_cfg["sparse_topk_blocks"],
            sparse_block_size=sparse_cfg["sparse_block_size"],
            num_index_heads=self.num_idx_heads,
            index_head_dim=self.idx_head_dim,
            prefix=self.layer_name,
            init_blocks=sparse_cfg.get("sparse_init_block", 0),
            local_blocks=sparse_cfg.get("sparse_local_block", 0),
            score_type=sparse_cfg.get("sparse_score_type", "max"),
            cache_config=cache_config,
            indexer_kv_dtype=self.indexer_kv_dtype,
            topk_indices_buffer=topk_indices_buffer,
        )

        # Register the main K/V cache so the KV-cache manager allocates it.
        compilation_config = vllm_config.compilation_config
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self
        self.kv_cache = torch.tensor([])  # replaced by bind_kv_cache

        self.use_qkv_fused_epilogue = _supports_qkv_fused_epilogue(self)
        if self.use_qkv_fused_epilogue:
            from vllm.models.minimax_m3.nvidia.ops.qkv_index_fused_epilogue import (
                register_qkv_index_fused_epilogue_warmup,
            )

            logger.info_once("MiniMax-M3: QKV+indexer GEMM with fused epilogue enabled")
            block_size = vllm_config.cache_config.block_size
            # The query is E4M3-only when CUTLASS decode serves the layer.
            q_is_fp8 = getattr(self.impl, "use_cutlass_decode", False)
            if vllm_config.kernel_config.enable_jit_warmup and block_size is not None:
                register_qkv_index_fused_epilogue_warmup(
                    has_q=not q_is_fp8,
                    has_qfp8=q_is_fp8,
                    nkv=self.num_kv_heads,
                    block_size=block_size,
                    weight_bytes=self.qkv_proj.output_size_per_partition
                    * self.hidden_size,
                )

    def get_attn_backend(self) -> type[MiniMaxM3SparseBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        # Main GQA K/V cache. Block size may change after load, refresh it.
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    def _qkv_proj_fused_epilogue(
        self,
        hidden_states: torch.Tensor | QuantizedActivation,
        positions: torch.Tensor,
        q: torch.Tensor,
        index_q: torch.Tensor,
        slot_mapping: torch.Tensor,
        index_slot_mapping: torch.Tensor,
    ) -> None:
        """qkv_proj + QK-norm/RoPE + KV/index-cache insert as one MXFP8 kernel,
        writing q (bf16 or E4M3, as allocated) and index_q."""
        from vllm.models.minimax_m3.nvidia.ops.qkv_index_fused_epilogue import (
            minimax_m3_qkv_index_fused_epilogue,
        )

        weight = self.qkv_proj.weight
        if weight.shape[1] != self.hidden_size:
            weight = weight.t()  # the CuTe-DSL kernel keeps a [K, N] view
        qa = as_quantized_activation(hidden_states, kMxfp8Dynamic)
        if qa is not None:
            xq, xsf = qa.data, qa.scale
        else:
            assert isinstance(hidden_states, torch.Tensor)
            xq, xsf = mxfp8_e4m3_quantize(hidden_states, is_sf_swizzled_layout=True)
        q_is_fp8 = q.dtype == torch.float8_e4m3fn
        minimax_m3_qkv_index_fused_epilogue(
            xq,
            xsf,
            weight,
            self.qkv_proj.weight_scale,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.q_norm.weight,
            self.k_norm.weight,
            self.index_q_norm.weight,
            self.index_k_norm.weight,
            self.q_norm.variance_epsilon,
            slot_mapping=slot_mapping,
            index_slot_mapping=index_slot_mapping,
            kv_cache=self.kv_cache,
            index_cache=self.indexer.index_cache.kv_cache,
            q_out=None if q_is_fp8 else q,
            index_q_out=index_q,
            q_fp8_out=q if q_is_fp8 else None,
            q_fp8_scale=self._q_scale_float,
            k_scale=self._k_scale_float,
            v_scale=self._v_scale_float,
        )

    def _allocate_query(
        self, num_tokens: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        # CUTLASS decode reads an E4M3 query, so the insert emits only that.
        fp8 = getattr(self.impl, "use_cutlass_decode", False)
        return torch.empty(
            (num_tokens, self.q_size),
            dtype=torch.float8_e4m3fn if fp8 else dtype,
            device=device,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | QuantizedActivation,
    ) -> torch.Tensor:
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        rotary_dim = self.rotary_emb.rotary_dim
        eps = self.q_norm.variance_epsilon

        fwd_slot_mapping = get_forward_context().slot_mapping
        if (
            not isinstance(fwd_slot_mapping, dict)
            or self.layer_name not in fwd_slot_mapping
        ):
            # Memory-profiling run: caches not yet bound, slot_mapping is empty.
            qkv, _ = self.qkv_proj(hidden_states)
            self.indexer.reserve_workspace()
            return qkv.new_zeros((qkv.shape[0], self.hidden_size))

        main_slot_mapping = fwd_slot_mapping[self.layer_name]
        index_slot_mapping = fwd_slot_mapping[self.indexer.index_cache.prefix]
        # FlashInfer autotunes the GEMMs its dummy runs call: take the unfused
        # path there so qkv_proj is tuned for the M < 512 fallback.
        if isinstance(hidden_states, QuantizedActivation):
            num_tokens = hidden_states.orig_shape[0]
            dtype, device = hidden_states.orig_dtype, hidden_states.data.device
        else:
            num_tokens = hidden_states.shape[0]
            dtype, device = hidden_states.dtype, hidden_states.device
        fused = (
            self.use_qkv_fused_epilogue
            and num_tokens >= _QKV_FUSED_EPILOGUE_MIN_TOKENS
            and not is_flashinfer_autotuning()
        )
        if not fused:
            # Single fused projection emitting [q | k | v | index_q | index_k].
            qkv, _ = self.qkv_proj(hidden_states)
        q = self._allocate_query(num_tokens, dtype, device)
        q_is_fp8 = q.dtype == torch.float8_e4m3fn
        # index_q is e4m3 for the fp8 and nvfp4 index caches, else bf16; the
        # fused kernel emits fp8 directly when this buffer is e4m3.
        index_q = torch.empty(
            (num_tokens, self.index_q_size),
            dtype=self.indexer.index_cache.query_dtype,
            device=device,
        )
        if fused:
            self._qkv_proj_fused_epilogue(
                hidden_states,
                positions,
                q,
                index_q,
                main_slot_mapping,
                index_slot_mapping,
            )
        else:
            # Horizontally-fused per-head Gemma QK-norm + partial NeoX RoPE on
            # the main (q/k) and index (index_q/index_k) branches, all read
            # straight out of qkv. k/v and index_k are scatter-inserted into the
            # caches; q and index_q are de-interleaved into the dedicated
            # contiguous buffers above. With an E4M3 q nothing reads qkv
            # afterwards, so k and index_k are not rewritten in place.
            ops.fused_minimax_m3_qknorm_rope_kv_insert(
                qkv,
                self.q_norm.weight,
                self.k_norm.weight,
                cos_sin_cache,
                positions,
                self.num_heads,
                self.num_kv_heads,
                rotary_dim,
                eps,
                self.index_q_norm.weight,
                self.index_k_norm.weight,
                self.num_idx_heads,
                main_slot_mapping,
                index_slot_mapping,
                self.kv_cache,
                self.indexer.index_cache.kv_cache,
                self.kv_cache.size(2),  # paged-cache block size
                None if q_is_fp8 else q,
                index_q,
                self.kv_cache_dtype,
                q_fp8_out=q if q_is_fp8 else None,
                q_fp8_scale=self._q_scale_float,
                kv_k_scale=self._k_scale if self.use_nvfp4_kv else None,
                kv_v_scale=self._v_scale if self.use_nvfp4_kv else None,
                write_qkv=not q_is_fp8,
            )

        output = torch.empty((num_tokens, self.q_size), dtype=dtype, device=device)
        attn_output = self._run_attention(q, index_q, output)
        output, _ = self.o_proj(attn_output)
        return output

    @eager_break_during_capture
    def _run_attention(
        self,
        query: torch.Tensor,
        index_query: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        # Single eager break around both: their split-K kernels read per-request
        # metadata and can't be captured into a cudagraph. The indexer writes its
        # top-k into the shared ``topk_indices_buffer``; the attend reads it back.
        self.indexer(index_query)
        return self.impl.forward(self, query, self.kv_cache, output)


class MiniMaxM3DecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str,
        force_sparse_attn: bool = False,
        force_moe: bool = False,
        is_mtp_block: bool = False,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if is_mtp_block:
            assert vllm_config.speculative_config is not None
            config = vllm_config.speculative_config.draft_model_config.hf_config
        else:
            config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.hidden_size = config.hidden_size
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_id = int(prefix.split(sep=".")[-1])
        self.layer_id = layer_id

        # When set, complete the preceding layer's deferred FFN all-reduce
        # fused into this layer's input_layernorm.
        # Configured by MiniMaxM3Model.__init__
        self.fuse_input_allreduce = False

        is_sparse_attention_layer = (
            force_sparse_attn or layer_id in _sparse_attention_layer_ids(config)
        )

        if is_sparse_attention_layer:
            self.self_attn = MiniMaxM3SparseAttention(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                cache_config=cache_config,
                topk_indices_buffer=topk_indices_buffer,
            )
        else:
            self.self_attn = MiniMaxM3Attention(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                cache_config=cache_config,
            )

        # Leave the FFN output un-reduced so its all-reduce fuses into the
        # next RMSNorm. MTP blocks add the residual directly and PP sends
        # hidden states across stages, so both must reduce.
        reduce_results = (
            is_mtp_block or vllm_config.parallel_config.pipeline_parallel_size > 1
        )
        self.is_moe_layer = force_moe or _is_moe_layer(config, layer_id)
        self.mlp: MiniMaxM3MoE | MiniMaxM3MLP
        if self.is_moe_layer:
            use_mega_moe = _use_mega_moe(vllm_config)
            if use_mega_moe and is_mtp_block:
                raise NotImplementedError(
                    "MiniMax M3 MegaMoE does not support MTP draft layers."
                )
            self.mlp = MiniMaxM3MoE(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                reduce_results=reduce_results,
                prefix=f"{prefix}.mlp",
                use_mega_moe=use_mega_moe,
            )
        else:
            self.mlp = MiniMaxM3MLP(
                config=config,
                intermediate_size=config.dense_intermediate_size,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                reduce_results=reduce_results,
            )

        # config.use_gemma_norm is True for M3 -> Gemma-style RMSNorm.
        self.input_layernorm = MiniMAXGemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = MiniMAXGemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | MoEOutput,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor | MoEOutput, torch.Tensor]:
        hidden_states, residual = self.forward_input_norm(hidden_states, residual)
        return self.forward_after_input_norm(positions, hidden_states, residual)

    def forward_input_norm(
        self, hidden_states: torch.Tensor | MoEOutput, residual: torch.Tensor | None
    ) -> tuple[torch.Tensor | QuantizedActivation, torch.Tensor]:
        if isinstance(hidden_states, MoEOutput):
            # The preceding MoE left a small batch unfinalized: finalize,
            # all-reduce, norm and MXFP8-quantize the qkv_proj input at once.
            assert residual is not None
            from vllm.models.minimax_m3.nvidia.ops.finalize_allreduce_norm import (
                finalize_allreduce_gemma_rms_norm_mxfp8,
            )

            return finalize_allreduce_gemma_rms_norm_mxfp8(
                hidden_states,
                residual,
                self.input_layernorm.weight,
                self.input_layernorm.variance_epsilon,
            )
        if self.fuse_input_allreduce and residual is not None:
            hidden_states, residual = fused_allreduce_gemma_rms_norm(
                hidden_states, residual, self.input_layernorm
            )
        else:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
        return hidden_states, residual

    def forward_after_input_norm(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | QuantizedActivation,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor | MoEOutput, torch.Tensor]:
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = fused_allreduce_gemma_rms_norm(
            hidden_states, residual, self.post_attention_layernorm
        )
        if isinstance(self.mlp, MiniMaxM3MoE) and self.mlp.defers_finalize(
            hidden_states.shape[0]
        ):
            # The next layer's input norm finalizes it with the all-reduce.
            return self.mlp.forward_unfinalized(hidden_states), residual
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    @property
    def ffn_all_reduce_deferred(self) -> bool:
        """This layer's FFN output is left un-reduced; the caller fuses the
        all-reduce into the next RMSNorm."""
        if isinstance(self.mlp, MiniMaxM3MoE):
            return self.mlp.ffn_all_reduce_deferred
        return not self.mlp.down_proj.reduce_results


class MiniMaxM3Model(nn.Module, EagleModelMixin):
    fall_back_to_pt_during_load = False
    supports_aux_hidden_states_over_pp = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.use_mega_moe = _use_mega_moe(vllm_config)

        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or spec_decode_needs_target_embed(
            vllm_config, include_mtp=vllm_config.use_v2_model_runner
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Reserved top-k indices buffer shared by all sparse-attention indexer
        # layers (mirrors DeepseekV4); kept at a stable address so the indexer's
        # top-k output survives cudagraph capture/replay. Token-major
        # [total_q, num_index_heads, topk] so the indexer writes its native
        # [token, head, topk] top-k; the attend transposes to [H, tokens, topk].
        sparse_cfg = getattr(config, "sparse_attention_config", None)
        if sparse_cfg is not None:
            tp_size = get_tensor_model_parallel_world_size()
            num_index_heads = max(1, sparse_cfg["sparse_num_index_heads"] // tp_size)
            # Pad tokens to a multiple of 4 so the buffer head stride stays
            # int4-aligned for build_k2q_csr's vectorised int4 loads.
            max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            padded_num_tokens = (max_num_batched_tokens + 3) // 4 * 4
            self.topk_indices_buffer = torch.empty(
                padded_num_tokens,
                num_index_heads,
                sparse_cfg["sparse_topk_blocks"],
                dtype=torch.int32,
            )
        else:
            self.topk_indices_buffer = None

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MiniMaxM3DecoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = MiniMAXGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        # Configure cross-layer all-reduce/RMSNorm fusion: a layer whose FFN output
        # is left un-reduced has that all-reduce fused into the next layer's
        # input_layernorm (or the final norm).
        prev_defers = False
        for idx, layer in enumerate(self.layers[self.start_layer : self.end_layer]):
            layer.fuse_input_allreduce = idx > 0 and prev_defers
            prev_defers = layer.ffn_all_reduce_deferred
        self.fuse_final_norm_allreduce = prev_defers
        if current_platform.is_cuda():
            self._init_fused_moe_finalize(vllm_config)

    def _init_fused_moe_finalize(self, vllm_config: VllmConfig) -> None:
        """Let small batches leave a MoE's top-k finalize to the next layer's
        input norm, which already fuses the TP all-reduce, and have it quantize
        the normed rows to the MXFP8 its qkv_proj reads.

        Each MoE whose successor can take the output gets the hint here, and
        ``defers_finalize`` decides per call. The final norm feeds the LM head
        in BF16, so the last layer keeps its own finalize. Builds a collective
        kernel: every TP rank decides alike.
        """
        from vllm.models.minimax_m3.nvidia.ops.finalize_allreduce_norm import (
            init_fused_moe_finalize,
            supports_fused_moe_finalize,
        )

        if not supports_fused_moe_finalize(vllm_config):
            return
        layers = self.layers[self.start_layer : self.end_layer]
        deferring = [
            layer.mlp
            for layer, successor in zip(layers, layers[1:])
            if layer.is_moe_layer
            and successor.fuse_input_allreduce
            and get_input_quant_key(successor.self_attn.qkv_proj) == kMxfp8Dynamic
            and layer.mlp.defer_finalize(FUSED_MOE_FINALIZE_MAX_TOKENS)
        ]
        if not deferring:
            return
        # The modular experts return the router's FP32 routing weights; the
        # monolithic kernel routes itself in BF16.
        experts_cls = getattr(
            deferring[0].experts.routed_experts.quant_method, "experts_cls", None
        )
        init_fused_moe_finalize(
            hidden_size=self.config.hidden_size,
            max_num_tokens=FUSED_MOE_FINALIZE_MAX_TOKENS,
            top_k=self.config.num_experts_per_tok,
            fp32_weights=experts_cls is TrtLlmNvFp4ExpertsModular,
        )
        logger.info_once(
            "MiniMax M3: MoE top-k finalize fused into the next all-reduce + "
            "input norm + MXFP8 quantization for up to %d tokens (%d layers).",
            FUSED_MOE_FINALIZE_MAX_TOKENS,
            len(deferring),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
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

        remote_aux = self.collect_remote_aux_hidden_states(intermediate_tensors)

        # Aux hidden state k is the residual stream entering layer k. Under TP
        # it is only complete after layer k's input norm (or the final norm)
        # finishes the previous FFN's deferred all-reduce, so it is read there.
        # Cloned, as the norms may update `residual` in place.
        aux_hidden_states: list[torch.Tensor] = []
        for idx, layer in enumerate(
            self.layers[self.start_layer : self.end_layer], start=self.start_layer
        ):
            hidden_states, residual = layer.forward_input_norm(hidden_states, residual)
            # A later PP stage's start boundary is captured by the stage before.
            if idx in self.aux_hidden_state_layers and (
                idx > self.start_layer or get_pp_group().is_first_rank
            ):
                aux_hidden_states.append(residual.clone())
            hidden_states, residual = layer.forward_after_input_norm(
                positions, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            # Under PP every FFN reduces its own output: no all-reduce is pending.
            self._maybe_add_hidden_state(
                aux_hidden_states, self.end_layer, hidden_states, residual
            )
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                    **self.pack_local_aux_hidden_states(aux_hidden_states),
                }
            )

        if self.fuse_final_norm_allreduce:
            hidden_states, residual = fused_allreduce_gemma_rms_norm(
                hidden_states, residual, self.norm
            )
        else:
            hidden_states, residual = self.norm(hidden_states, residual)
        if self.end_layer in self.aux_hidden_state_layers:
            aux_hidden_states.append(residual.clone())

        aux_hidden_states = remote_aux + aux_hidden_states
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        if self.use_mega_moe:
            return _mega_moe_expert_params_mapping(self.config.num_local_experts)
        # Checkpoint experts use w1=gate, w2=down, w3=up.
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # q/k/v_proj -> fused qkv_proj; gate_proj/up_proj -> fused gate_up_proj
        # (dense MLP and shared expert). On sparse layers the indexer
        # index_q/index_k_proj fold into the same fused qkv_proj
        # (MinimaxM3QKVParallelLinearWithIndexer); these entries simply never match on
        # dense layers, whose checkpoints have no index_*_proj weights. Leading
        # dots keep `q_proj`/`k_proj` from matching `index_q_proj`/`index_k_proj`
        # (preceded by `_`, not `.`).
        stacked_params_mapping: list[tuple[str, str, int | str]] = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".qkv_proj", ".index_q_proj", "index_q"),
            (".qkv_proj", ".index_k_proj", "index_k"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = self.get_expert_mapping()

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            # The MTP module is not modeled yet.
            if "mtp." in name:
                continue

            # The checkpoint stores block scales as ``weight_scale_inv``; the
            # ModelOpt MXFP8 layers expose them as ``weight_scale``.
            if "weight_scale_inv" in name:
                name = name.replace("weight_scale_inv", "weight_scale")

            if self.use_mega_moe and ".mlp.experts." in name:
                converted = _mega_moe_expert_weight(name, loaded_weight)
                if converted is None:
                    continue
                name, loaded_weight = converted
            if self.use_mega_moe and ".shared_experts." in name:
                if is_pp_missing_parameter(name, self):
                    continue
                moe = self.layers[extract_layer_index(name)].mlp
                if moe.experts.has_fused_shared_experts:
                    name = name.replace(".gate_proj.", ".w1.")
                    name = name.replace(".up_proj.", ".w3.")
                    loaded_params.add(
                        moe.experts.load_shared_expert_weight(name, loaded_weight)
                    )
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Routed experts (w1/w2/w3) are handled below; don't let the
                # stacked mapping rewrite them.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    expert_shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=expert_shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    remapped = maybe_remap_kv_scale_name(name, params_dict)
                    if remapped is None:
                        continue
                    name = remapped
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Modules not modeled yet (e.g. attention) are skipped until
                    # they are ported.
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class MiniMaxM3SparseForCausalLM(nn.Module, SupportsPP, SupportsEagle3):
    """MiniMax M3 (sparse/dense backbone) for causal language modeling."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            ".mlp.gate.e_score_correction_bias": ".mlp.e_score_correction_bias",
            ".self_attn.indexer.": ".self_attn.index_",
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = MiniMaxM3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


@MULTIMODAL_REGISTRY.register_processor(
    MiniMaxM3VLMultiModalProcessor,
    info=MiniMaxM3VLProcessingInfo,
    dummy_inputs=MiniMaxM3VLDummyInputsBuilder,
)
class MiniMaxM3SparseForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    MiniMaxM3EncoderCudaGraphMixin,
    SupportsPP,
    SupportsEagle3,
):
    """Top-level (VL) entry point for MiniMax M3.

    The vision tower is not modeled yet; this wrapper routes the text
    backbone by constructing ``MiniMaxM3SparseForCausalLM`` from the nested
    ``text_config`` and delegating generation to it.
    """

    # The vision tower runs replicated per rank under ``--mm-encoder-tp-mode
    # data``; ``run_dp_sharded_mrope_vision_model`` shards the work across
    # ranks (see ``_process_image_input`` / ``_process_video_input``).
    supports_encoder_tp_data = True

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
            "model.vision_tower.embeddings.proj.": (
                "vision_tower.vision_model.embeddings.patch_embedding."
            ),
            "model.vision_tower.layers.": "vision_tower.vision_model.encoder.layers.",
            "model.vision_tower.": "vision_tower.vision_model.",
            "model.multi_modal_projector.merge_linear_": (
                "vision_tower.patch_merge_mlp.linear_"
            ),
            "model.multi_modal_projector.": "vision_tower.multi_modal_projector.",
        },
        orig_to_new_substr={
            ".mlp.fc1.": ".fc1.",
            ".mlp.fc2.": ".fc2.",
        },
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return MiniMaxM3VLProcessingInfo.IMAGE_TOKEN
        if modality == "video":
            return MiniMaxM3VLProcessingInfo.VIDEO_TOKEN
        raise ValueError(f"Unsupported modality: {modality!r}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        assert self.multimodal_config is not None
        self.use_data_parallel = self.multimodal_config.mm_encoder_tp_mode == "data"

        text_hidden_size = getattr(config.text_config, "hidden_size", None)
        assert text_hidden_size is not None, "text_config.hidden_size is required"
        projector_hidden_size = getattr(config, "projector_hidden_size", None)

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            vision_config = config.vision_config
            self.vision_tower = MiniMaxVLVisionModel(
                config=PreTrainedConfig.from_dict(vision_config),
                text_hidden_size=text_hidden_size,
                projector_hidden_size=projector_hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["MiniMaxM3SparseForCausalLM"],
        )
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.language_model.make_empty_intermediate_tensors
        )

    # Expose language model / lm_head for EAGLE3 spec decode.
    @property
    def model(self) -> nn.Module:
        return self.language_model.model

    @property
    def lm_head(self) -> nn.Module:
        return self.language_model.lm_head

    def _parse_and_validate_image_input(self, **kwargs: object) -> dict | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        if pixel_values is None:
            return None
        return {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}

    def _parse_and_validate_video_input(self, **kwargs: object) -> dict | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        if pixel_values_videos is None:
            return None
        return {
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }

    def _process_image_input(self, image_input: dict) -> tuple[torch.Tensor, ...]:
        pixel_values: torch.Tensor = image_input["pixel_values"].type(
            self.vision_tower.dtype
        )
        grid_thw: torch.Tensor = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if self.use_data_parallel:
            # Already returns a per-item tuple of embeddings.
            return run_dp_sharded_mrope_vision_model(
                self.vision_tower,
                pixel_values,
                grid_thw.tolist(),
                rope_type="rope_3d",
            )

        image_embeds = self.vision_tower(
            pixel_values=pixel_values,
            grid_thw=grid_thw.tolist(),
        )

        # Split the concatenated output into one tensor per image item.
        merge_size = self.vision_tower.spatial_merge_size
        sizes = (grid_thw.prod(-1) // (merge_size * merge_size)).tolist()
        return image_embeds.split(sizes)

    def _process_video_input(self, video_input: dict) -> tuple[torch.Tensor, ...]:
        pixel_values: torch.Tensor = video_input["pixel_values_videos"].type(
            self.vision_tower.dtype
        )
        grid_thw: torch.Tensor = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        if self.use_data_parallel:
            # Already returns a per-item tuple of embeddings.
            return run_dp_sharded_mrope_vision_model(
                self.vision_tower,
                pixel_values,
                grid_thw.tolist(),
                rope_type="rope_3d",
            )

        video_embeds = self.vision_tower(
            pixel_values=pixel_values,
            grid_thw=grid_thw.tolist(),
        )

        # Split the concatenated output into one tensor per video item.
        merge_size = self.vision_tower.spatial_merge_size
        sizes = (grid_thw.prod(-1) // (merge_size * merge_size)).tolist()
        return video_embeds.split(sizes)

    def _parse_and_validate_multimodal_inputs(
        self, **kwargs: object
    ) -> dict[str, dict]:
        mm_input_by_modality: dict[str, dict] = {}
        for input_key in kwargs:
            if input_key == "pixel_values" and "image" not in mm_input_by_modality:
                image_input = self._parse_and_validate_image_input(**kwargs)
                if image_input is not None:
                    mm_input_by_modality["image"] = image_input
            if (
                input_key == "pixel_values_videos"
                and "video" not in mm_input_by_modality
            ):
                video_input = self._parse_and_validate_video_input(**kwargs)
                if video_input is not None:
                    mm_input_by_modality["video"] = video_input
        return mm_input_by_modality

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        multimodal_embeddings: list[torch.Tensor] = []
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                image_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings.extend(image_embeddings)
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                multimodal_embeddings.extend(video_embeddings)

        return tuple(multimodal_embeddings)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.compute_logits_local(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
