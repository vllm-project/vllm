# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only MiniMax M3 (text backbone) model — AMD ROCm implementation.

Self-contained per-platform impl (mirrors ``deepseek_v4/amd``). It is identical
to ``../nvidia/model.py`` except for RMS normalization: FlashInfer's Gemma
RMSNorm kernels are CUDA-only, so ``MiniMAXGemmaRMSNorm`` here uses a native
(FlashInfer-free) implementation.

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
from transformers import PretrainedConfig

from vllm import _custom_ops as ops
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import (
    CacheConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import set_default_quant_scales
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
    fused_allreduce_gemma_rms_norm,
)
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    MinimaxM3QKVParallelLinearWithIndexer,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
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
    init_vllm_registered_model,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.models.vision import run_dp_sharded_mrope_vision_model
from vllm.models.minimax_m3.amd.ops import (
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
    swiglu_oai_split,
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
    minimax_m3_use_aiter_sparse_pa,
    select_main_impl_cls,
)
from vllm.models.minimax_m3.common.vision_tower import MiniMaxVLVisionModel
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
    is_quantized_kv_cache,
)


def _fuse_shared_experts_enabled(config: PretrainedConfig) -> bool:
    """Whether to fuse the shared expert with routed experts.

    ROCm only. Opt-in via ``VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS`` (the
    router-append fusion runs on both aiter and non-aiter MoE);
    it is disabled under expert parallelism (the shared slot is appended to
    the routed top-k, which the EP expert-mapping path does not handle).
    """
    from vllm.platforms import current_platform

    return bool(
        current_platform.is_rocm()
        and getattr(config, "n_shared_experts", None)
        and envs.VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS
        and not get_current_vllm_config().parallel_config.enable_expert_parallel
    )


def _sparse_attention_layer_ids(config: PretrainedConfig) -> set[int]:
    """Layer ids whose attention runs the extra sparse "index" branch."""
    cfg = getattr(config, "sparse_attention_config", None)
    if not cfg:
        return set()
    freq = cfg.get("sparse_attention_freq")
    if freq is None:
        return set()
    return {i for i, f in enumerate(freq) if f != 0}


def _sparse_attention_layer_ordinals(config: PretrainedConfig) -> dict[int, int]:
    """Map each sparse-attention layer id to its ordinal among sparse layers."""
    return {
        lid: ordinal
        for ordinal, lid in enumerate(sorted(_sparse_attention_layer_ids(config)))
    }


def _should_skip_index_topk(config: PretrainedConfig, layer_id: int) -> bool:
    """ATOM ``index_topk_freq`` (cross-layer index sharing).

    Only 1 of every ``index_topk_freq`` sparse-attention layers recomputes the
    lightning-indexer top-k block selection; the rest reuse the selection the
    preceding compute layer wrote into the shared ``topk_indices_buffer`` this
    same forward pass. This cuts the indexer score + top-k cost ~``freq``x with
    negligible accuracy impact (adjacent sparse layers pick nearly the same
    blocks; ATOM validated GSM8K with freq=4). Gated by ``use_index_cache``;
    enable via ``--hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}'``.
    """
    if not getattr(config, "use_index_cache", False):
        return False
    freq = int(getattr(config, "index_topk_freq", 1) or 1)
    if freq <= 1:
        return False
    ordinal = _sparse_attention_layer_ordinals(config).get(layer_id)
    if ordinal is None:
        return False
    offset = int(getattr(config, "index_skip_topk_offset", 0) or 0)
    return max(ordinal - offset, 0) % freq != 0


def _is_moe_layer(config: PretrainedConfig, layer_id: int) -> bool:
    """Whether this layer's MLP is a sparse MoE block (vs a dense MLP)."""
    moe_layer_freq = getattr(config, "moe_layer_freq", None)
    if moe_layer_freq is None:
        return True
    return moe_layer_freq[layer_id] != 0


def _build_rotary_emb(config: PretrainedConfig, head_dim: int):
    """Build the (partial NeoX) RoPE, honoring an optional ``rope_scaling`` config.

    Without scaling the cos/sin cache is sized to ``max_position_embeddings``
    (524288 native); a request whose positions exceed that reads the cache out of
    bounds and the worker hard-crashes (no Python traceback). When ``rope_scaling``
    is set (e.g. YaRN ``factor: 2`` to reach 1M), thread it into ``get_rope`` so the
    proper scaled embedding is built and its cache covers
    ``original_max_position_embeddings * factor`` positions. Default behavior
    (no scaling) is unchanged. Shared by the dense and sparse attention layers, and
    the index branch reuses the returned module.

    Note: for the VL checkpoint, set ``rope_scaling`` on the *text* config
    (``--hf-overrides '{"text_config":{"rope_scaling":{...}}}'``) -- that is the
    config the decoder reads here; a top-level override does not reach it.
    """
    rope_parameters = {
        "rope_theta": config.rope_theta,
        "partial_rotary_factor": config.partial_rotary_factor,
    }
    max_position = config.max_position_embeddings
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling:
        rope_parameters.update(rope_scaling)
        # HF uses "rope_type" (older configs: "type"); get_rope reads "rope_type".
        if "rope_type" not in rope_parameters and "type" in rope_scaling:
            rope_parameters["rope_type"] = rope_scaling["type"]
        rope_parameters.setdefault(
            "original_max_position_embeddings", config.max_position_embeddings
        )
        factor = float(rope_scaling.get("factor", 1.0))
        # Cover the extended range (informational for get_rope's default branch;
        # the YaRN embedding sizes its own cache from original * factor).
        max_position = int(rope_parameters["original_max_position_embeddings"] * factor)
    return get_rope(
        head_dim,
        max_position=max_position,
        rope_parameters=rope_parameters,
    )


class MiniMAXGemmaRMSNorm(nn.Module):
    """Gemma-style RMS normalization (native ROCm implementation).

    Normalizes in fp32 and scales by ``(1 + weight)`` — numerically equivalent
    to the FlashInfer ``gemma_rmsnorm`` / ``gemma_fused_add_rmsnorm`` kernels
    used in the NVIDIA path, which are unavailable on ROCm. When ``residual`` is
    given, the fused add + norm returns the updated ``(normed, residual)`` pair.

    The fp32 normalize + scale + (optional) residual-add run in a single fused
    Triton pass (``amd.ops.gemma_rmsnorm`` / ``gemma_fused_add_rmsnorm``) instead
    of a chain of elementwise PyTorch kernels.
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
        if residual is None:
            return gemma_rmsnorm(x, self.weight, self.variance_epsilon)
        return gemma_fused_add_rmsnorm(x, residual, self.weight, self.variance_epsilon)


class MiniMaxM3MLP(nn.Module):
    """Dense SwiGLU-OAI MLP (used by the leading dense layers)."""

    def __init__(
        self,
        config: PretrainedConfig,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if config.hidden_act != "swigluoai":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only swigluoai is supported."
            )
        # gate * sigmoid(alpha * gate) * (up + beta), with both halves clamped.
        # Kept as our fp32 Triton kernel (not the #22 SWIGLUOAI_UNINTERLEAVE op
        # ``silu_and_mul_with_clamp``): that op IS built on ROCm but rounds
        # intermediates to bf16 (rel ~3e-3 vs our fp32 ~1e-6), which costs gsm8k
        # accuracy since this activation feeds the MXFP8 quant + MoE.
        self.swiglu_alpha = config.swiglu_alpha
        self.swiglu_beta = config.swiglu_beta
        self.swiglu_limit = config.swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = swiglu_oai_split(
            gate_up,
            alpha=self.swiglu_alpha,
            beta=self.swiglu_beta,
            limit=self.swiglu_limit,
        )
        x, _ = self.down_proj(x)
        return x


def _aiter_moe_fused_shared_experts_enabled(config: PretrainedConfig) -> bool:
    """Whether the fused shared expert routes through aiter's grouped top-k MoE.

    A strict sub-case of :func:`_fuse_shared_experts_enabled`: shared-expert
    fusion must already be opted in (``VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS``)
    and allowed (not under expert parallelism). When additionally on gfx950 with
    an active aiter MoE backend, the shared expert is appended inside aiter's
    biased grouped top-k kernel (``num_fused_shared_experts``) instead of the
    vLLM router's torch concat. Otherwise FSE still runs via the vLLM top-k bias
    router.
    """
    if not _fuse_shared_experts_enabled(config):
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950() and rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()


class MiniMaxM3MoE(nn.Module):
    """Sigmoid-routed MoE block with a routing-bias correction and a shared
    expert."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size > config.num_local_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_local_experts}."
            )

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

        # Shared-expert fusion (opt-in via VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS,
        # off under expert parallelism) folds the shared expert into the routed
        # MoE call as the last expert slot, so we don't build a separate module.
        # On gfx950 with aiter MoE the append is fused inside aiter's grouped
        # top-k kernel; otherwise it goes through the vLLM top-k bias router.
        self.fuse_shared_experts = _fuse_shared_experts_enabled(config)
        self.use_aiter_moe_fse = _aiter_moe_fused_shared_experts_enabled(config)

        self.shared_experts: MiniMaxM3MLP | None = None
        if self.n_shared_experts and not self.fuse_shared_experts:
            self.shared_experts = MiniMaxM3MLP(
                config=config,
                intermediate_size=config.intermediate_size * self.n_shared_experts,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        # The aiter MoE fused path goes through aiter's biased grouped top-k
        # (GroupedTopKRouter, as in DeepSeek-V4): M3 is not group-routed, so a
        # trivial single group (num_expert_group=topk_group=1) reduces to plain
        # top-k while applying the sigmoid + bias correction and appending the
        # always-on shared expert; aiter applies the routed scaling internally.
        # Every other path (vLLM top-k bias router, or no fusion) applies the
        # routed scaling to the MoE output here.
        self.experts = FusedMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            intermediate_pad=0,
            scoring_func=config.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            renormalize=True,
            use_grouped_topk=self.use_aiter_moe_fse,
            num_expert_group=1 if self.use_aiter_moe_fse else None,
            topk_group=1 if self.use_aiter_moe_fse else None,
            activation="swigluoai_uninterleave",
            swiglu_limit=config.swiglu_limit,
            swiglu_alpha=config.swiglu_alpha,
            swiglu_beta=config.swiglu_beta,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scale_to_output=not self.use_aiter_moe_fse,
            router_logits_dtype=self.gate.out_dtype,
            shared_experts=self.shared_experts,
            n_shared_experts=(
                self.n_shared_experts if self.fuse_shared_experts else None
            ),
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

    @staticmethod
    def ebias_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight.to(torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (num_tokens, n_experts); GateLinear casts to fp32.
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        return final_hidden_states.view(num_tokens, hidden_dim)


class MiniMaxM3Attention(nn.Module):
    """Dense attention with per-head QK norm and partial RoPE."""

    def __init__(
        self,
        config: PretrainedConfig,
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

        # Partial RoPE: rotary_dim == head_dim * partial_rotary_factor. Honors
        # config.rope_scaling (e.g. YaRN) so long-context positions are covered.
        self.rotary_emb = _build_rotary_emb(config, self.head_dim)

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
        # Fused per-head Gemma QK-norm + partial NeoX RoPE on q/k, in place (dense
        # mode: no index branch, no KV-cache insert). Matches nvidia/model.py and
        # replaces the unfused split -> q_norm/k_norm -> rotary_emb chain; verified
        # bit-equivalent on ROCm (q/k rel ~2e-3 bf16 noise, v untouched).
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
        config: PretrainedConfig,
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

        # Cross-layer index sharing (ATOM index_topk_freq): when True this sparse
        # layer reuses the previous compute layer's top-k block selection from the
        # shared topk_indices_buffer instead of recomputing it. Static per layer
        # -> cudagraph-capture-safe.
        self.skip_index_topk = _should_skip_index_topk(config, layer_id)

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

        # Partial RoPE: rotary_dim == head_dim * partial_rotary_factor. Honors
        # config.rope_scaling (e.g. YaRN) so long-context positions are covered.
        self.rotary_emb = _build_rotary_emb(config, self.head_dim)

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
        self.calculate_kv_scales = False
        set_default_quant_scales(self, register_buffer=True)

        # Shared top-k buffer: the indexer writes the selected blocks into it and
        # the attend impl reads them back (no Python value crosses the break).
        self.topk_indices_buffer = topk_indices_buffer
        self.attn_backend = MiniMaxM3SparseBackend
        # Indexer and main attention are separate impls. On ROCm the SM100 gate
        # is always False, so both pick Triton and the index cache stays bf16.
        # impl is AttentionImplBase (broader than AttentionLayerBase's annotation).
        self.impl: MiniMaxM3SparseImpl = select_main_impl_cls(  # type: ignore[assignment]
            topk_blocks=sparse_cfg["sparse_topk_blocks"],
            kv_cache_dtype=self.kv_cache_dtype,
            num_kv_heads=self.num_kv_heads,
        )(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype,
            topk_blocks=sparse_cfg["sparse_topk_blocks"],
            sparse_block_size=sparse_cfg["sparse_block_size"],
        )
        self.use_aiter_sparse_pa = minimax_m3_use_aiter_sparse_pa(self.num_kv_heads)
        self.kv_cache_k = torch.tensor([])
        self.kv_cache_v = torch.tensor([])
        self._aiter_sparse_pa_cache_data_ptr = 0
        # Self-contained nn.Module: owns its side cache, selects its impl in init
        # (Triton on ROCm, where the SM100 gate is always False).
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
            topk_indices_buffer=topk_indices_buffer,
        )

        # Register the main K/V cache so the KV-cache manager allocates it.
        compilation_config = vllm_config.compilation_config
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self
        self.kv_cache = torch.tensor([])  # replaced by bind_kv_cache

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

    def _ensure_aiter_sparse_pa_kv_cache(self) -> None:
        if self.kv_cache.numel() == 0:
            return
        if self._aiter_sparse_pa_cache_data_ptr == self.kv_cache.data_ptr():
            return

        kv_cache = self.kv_cache
        if is_quantized_kv_cache(self.kv_cache_dtype):
            kv_cache = kv_cache.view(self.impl.kv_cache_fp8_dtype)
        key_cache, value_cache = kv_cache.unbind(1)
        if not key_cache.is_contiguous() or not value_cache.is_contiguous():
            raise RuntimeError(
                "MiniMax-M3 AITER sparse PA requires K/V-separated KV cache "
                "storage. Set VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1 before "
                "initializing the engine."
            )

        x = 16 // key_cache.element_size()
        if self.head_dim % x != 0:
            raise RuntimeError(
                "MiniMax-M3 AITER sparse PA requires head_dim divisible by "
                f"16 / dtype_size, got head_dim={self.head_dim}, x={x}"
            )
        num_blocks = key_cache.shape[0]
        num_phys16 = num_blocks * 8
        self.kv_cache_k = key_cache.view(
            num_phys16,
            self.num_kv_heads,
            self.head_dim // x,
            16,
            x,
        )
        self.kv_cache_v = value_cache.view(
            num_phys16,
            self.num_kv_heads,
            16 // x,
            self.head_dim,
            x,
        )
        self._aiter_sparse_pa_cache_data_ptr = self.kv_cache.data_ptr()

    def get_aiter_sparse_pa_kv_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_aiter_sparse_pa_kv_cache()
        return self.kv_cache_k, self.kv_cache_v

    def _insert_aiter_sparse_pa_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        index_k: torch.Tensor | None,
        slot_mapping: torch.Tensor,
        index_slot_mapping: torch.Tensor | None,
    ) -> None:
        if self.kv_cache.numel() == 0:
            return
        from aiter import reshape_and_cache

        from vllm.models.minimax_m3.amd.ops.sparse_pa import (
            minimax_m3_insert_index_cache,
        )

        key_cache, value_cache = self.get_aiter_sparse_pa_kv_cache()
        kv_cache_dtype = (
            self.kv_cache_dtype
            if is_quantized_kv_cache(self.kv_cache_dtype)
            else "auto"
        )
        reshape_and_cache(
            k.contiguous(),
            v.contiguous(),
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype=kv_cache_dtype,
            k_scale=getattr(self, "_k_scale", None),
            v_scale=getattr(self, "_v_scale", None),
            asm_layout=True,
        )
        if index_k is None or index_slot_mapping is None:
            return
        index_cache = self.indexer.index_cache.kv_cache
        if index_cache.numel() == 0:
            return
        minimax_m3_insert_index_cache(index_k, index_cache, index_slot_mapping)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Single fused projection emitting [q | k | v | index_q | index_k].
        qkv, _ = self.qkv_proj(hidden_states)

        # Horizontally-fused per-head Gemma QK-norm + partial NeoX RoPE on the
        # main (q/k) and index (index_q/index_k) branches, all read straight out
        # of the single fused ``qkv`` tensor. Once the paged caches are bound the
        # kernel also inserts k/v and the index key into them (each with its own
        # slot_mapping); the memory-profiling run (caches unbound, no slot_mapping)
        # short-circuits to zeros below. The main and index slot mappings are read
        # from the forward context's slot_mapping dict, matching the
        # breakable-cudagraph path -- see nvidia/model.py.
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        rotary_dim = self.rotary_emb.rotary_dim
        eps = self.q_norm.variance_epsilon
        num_tokens = qkv.shape[0]

        fwd_slot_mapping = get_forward_context().slot_mapping
        if (
            not isinstance(fwd_slot_mapping, dict)
            or self.layer_name not in fwd_slot_mapping
        ):
            # Memory-profiling run: caches not yet bound, slot_mapping is empty.
            return qkv.new_zeros((num_tokens, self.hidden_size))

        main_slot_mapping = fwd_slot_mapping[self.layer_name]
        q = qkv.new_empty((num_tokens, self.q_size))
        if self.skip_index_topk:
            index_q = None
            if self.use_aiter_sparse_pa:
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
                    num_index_heads=self.num_idx_heads,
                    q_out=q,
                    skip_index_branch=True,
                )
                k_start = self.q_size
                v_start = k_start + self.kv_size
                k = qkv[:, k_start:v_start].view(
                    num_tokens, self.num_kv_heads, self.head_dim
                )
                v = qkv[:, v_start : v_start + self.kv_size].view(
                    num_tokens, self.num_kv_heads, self.head_dim
                )
                self._insert_aiter_sparse_pa_kv(
                    k,
                    v,
                    None,
                    main_slot_mapping,
                    None,
                )
            else:
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
                    num_index_heads=self.num_idx_heads,
                    slot_mapping=main_slot_mapping,
                    kv_cache=self.kv_cache,
                    block_size=self.kv_cache.size(2),  # paged-cache block size
                    q_out=q,
                    kv_cache_dtype=self.kv_cache_dtype,
                    skip_index_branch=True,
                )
        else:
            index_slot_mapping = fwd_slot_mapping[self.indexer.index_cache.prefix]
            index_q = qkv.new_empty((num_tokens, self.index_q_size))
            if self.use_aiter_sparse_pa:
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
                    q_out=q,
                    index_q_out=index_q,
                    kv_cache_dtype=self.kv_cache_dtype,
                )
                k_start = self.q_size
                v_start = k_start + self.kv_size
                index_k_start = v_start + self.kv_size + self.index_q_size
                k = qkv[:, k_start:v_start].view(
                    num_tokens, self.num_kv_heads, self.head_dim
                )
                v = qkv[:, v_start : v_start + self.kv_size].view(
                    num_tokens, self.num_kv_heads, self.head_dim
                )
                index_k = qkv[
                    :, index_k_start : index_k_start + self.idx_head_dim
                ].view(num_tokens, self.idx_head_dim)
                self._insert_aiter_sparse_pa_kv(
                    k,
                    v,
                    index_k,
                    main_slot_mapping,
                    index_slot_mapping,
                )
            else:
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
                    q,
                    index_q,
                    self.kv_cache_dtype,
                )

        output = torch.empty_like(q)
        attn_output = self._run_attention(q, index_q, output)
        output, _ = self.o_proj(attn_output)
        return output

    @eager_break_during_capture
    def _run_attention(
        self,
        query: torch.Tensor,
        index_query: torch.Tensor | None,
        output: torch.Tensor,
    ) -> torch.Tensor:
        # Single eager break around both: their split-K kernels read per-request
        # metadata and can't be captured into a cudagraph. The indexer writes its
        # top-k into the shared ``topk_indices_buffer``; the attend reads it back.
        # When skip_index_topk is set (ATOM index_topk_freq), reuse the selection
        # the preceding compute layer wrote into the shared buffer this forward.
        if not self.skip_index_topk:
            assert index_query is not None
            self.indexer(index_query)
        return self.impl.forward(self, query, self.kv_cache, output)


class MiniMaxM3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        force_sparse_attn: bool = False,
        force_moe: bool = False,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_id = int(prefix.split(sep=".")[-1])
        self.layer_id = layer_id

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

        # Dense layers store the FFN under `mlp`; MoE layers under
        # `block_sparse_moe` -- matching the checkpoint's naming.
        self.is_moe_layer = force_moe or _is_moe_layer(config, layer_id)
        if self.is_moe_layer:
            self.block_sparse_moe = MiniMaxM3MoE(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.block_sparse_moe",
            )
        else:
            self.mlp = MiniMaxM3MLP(
                config=config,
                intermediate_size=config.dense_intermediate_size,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
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
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = fused_allreduce_gemma_rms_norm(
            hidden_states, residual, self.post_attention_layernorm
        )
        ffn = self.block_sparse_moe if self.is_moe_layer else self.mlp
        hidden_states = ffn(hidden_states)
        return hidden_states, residual


class MiniMaxM3Model(nn.Module, EagleModelMixin):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Reserved top-k indices buffer shared by all sparse-attention indexer
        # layers (mirrors DeepseekV4); the indexer writes its per-head decode/
        # prefill block selection into it, the attend reads it back.
        sparse_cfg = getattr(config, "sparse_attention_config", None)
        if sparse_cfg is not None:
            tp_size = get_tensor_model_parallel_world_size()
            num_index_heads = max(1, sparse_cfg["sparse_num_index_heads"] // tp_size)
            self.topk_indices_buffer = torch.empty(
                num_index_heads,
                vllm_config.scheduler_config.max_num_batched_tokens,
                sparse_cfg["sparse_topk_blocks"],
                dtype=torch.int32,
            )
        else:
            self.topk_indices_buffer = None

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MiniMaxM3DecoderLayer(
                config,
                prefix,
                cache_config=cache_config,
                quant_config=quant_config,
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

        # EAGLE3 is not yet compatible with pipeline parallel
        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for idx, layer in enumerate(self.layers[self.start_layer : self.end_layer]):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(
                aux_hidden_states, idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Checkpoint experts use w1=gate, w2=down, w3=up. When fusing the shared
        # expert, include the appended slot (id == num_local_experts).
        n_shared = getattr(self.config, "n_shared_experts", 0) or 0
        num_experts = self.config.num_local_experts + (
            n_shared if _fuse_shared_experts_enabled(self.config) else 0
        )
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=num_experts,
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

        _fuse_shared = _fuse_shared_experts_enabled(self.config)

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

            # Shared-expert fusion: redirect the checkpoint shared expert into
            # routed-expert slot ``num_local_experts`` (gate->w1, up->w3,
            # down->w2) so it loads via the routed expert loader. Runs before the
            # stacked/dense mappings so shared_experts.gate_proj/up_proj are not
            # captured by the dense gate_up_proj mapping.
            if _fuse_shared and ".shared_experts." in name:
                sid = self.config.num_local_experts
                name = name.replace(".shared_experts.gate_proj.", f".experts.{sid}.w1.")
                name = name.replace(".shared_experts.up_proj.", f".experts.{sid}.w3.")
                name = name.replace(".shared_experts.down_proj.", f".experts.{sid}.w2.")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Routed experts (w1/w2/w3) are handled below; don't let the
                # stacked mapping rewrite them.
                if ("block_sparse_moe.experts." in name) and name not in params_dict:
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

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


# TODO(refactor): this VL wrapper is platform-agnostic and byte-identical to the
# NVIDIA copy — it only orchestrates the shared vision tower + the per-platform
# language model (resolved via ``init_vllm_registered_model``). Hoist it into
# ``common/`` to drop the amd/nvidia duplication once the split stabilizes.
@MULTIMODAL_REGISTRY.register_processor(
    MiniMaxM3VLMultiModalProcessor,
    info=MiniMaxM3VLProcessingInfo,
    dummy_inputs=MiniMaxM3VLDummyInputsBuilder,
)
class MiniMaxM3SparseForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsEagle3
):
    """Top-level (VL) entry point for MiniMax M3.

    Owns the shared MiniMax-M3 vision tower on ROCm and delegates text
    generation to the AMD language-model path.
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
            "multi_modal_projector.": "vision_tower.multi_modal_projector.",
            "patch_merge_mlp.": "vision_tower.patch_merge_mlp.",
        },
        orig_to_new_substr={
            ".mlp.fc1": ".fc1",
            ".mlp.fc2": ".fc2",
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
                config=PretrainedConfig.from_dict(vision_config),
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

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
