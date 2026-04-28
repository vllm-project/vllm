# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Laguna model compatible with HuggingFace weights."""

import typing
from collections.abc import Callable, Iterable
from itertools import islice

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
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
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


class LagunaMLP(nn.Module):
    """Dense MLP for Laguna (used in mlp_only_layers)."""

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
        # gate_proj and up_proj are kept as separate ColumnParallelLinear
        # rather than merged via MergedColumnParallelLinear. The merged form
        # requires per-partition NVFP4 global scales (weight_global_scale,
        # input_global_scale) to be packed into a length-2 PerTensorScaleParameter
        # and then collapsed via .max() in process_weights_after_loading; this
        # doesn't round-trip cleanly through Marlin's NVFP4 stacked-layer code
        # path. Splitting yields one global scale per Linear, exactly matching
        # the standard compressed-tensors per-Linear schema on disk.
        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_proj",
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, _ = self.gate_proj(x)
        up, _ = self.up_proj(x)
        x, _ = self.down_proj(F.silu(gate) * up)
        return x


class LagunaMoE(nn.Module):
    """Sparse MoE block for Laguna with optional shared expert and sigmoid routing.

    Key differences from other MoE implementations:
    - Uses SIGMOID routing activation (not softmax)
    - Shared expert runs in parallel with routed experts (when enabled)
    - Matches HF reference: modular_laguna.py LagunaSparseMoeBlock
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_group = get_ep_group().device_group
        self.ep_rank = self.ep_group.rank()
        self.ep_size = self.ep_group.size()

        self.n_routed_experts = config.num_experts
        self.n_shared_experts = 1 if config.shared_expert_intermediate_size > 0 else 0
        self.routed_scaling_factor = float(
            getattr(config, "moe_routed_scaling_factor", 1.0)
        )

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # Load balancing settings.
        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = enable_eplb
        eplb_config.num_redundant_experts = (
            eplb_config.num_redundant_experts
            if eplb_config.num_redundant_experts is not None
            else 0
        )
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        # Router gate
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        # Shared expert (optional) - passed to FusedMoE for overlap optimization
        self.shared_expert: LagunaMLP | None
        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = LagunaMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,  # Reduce after shared+routed combine
                prefix=f"{prefix}.shared_expert",
            )
        else:
            self.shared_expert = None

        # Auxiliary-loss-free load-balancing bias (arXiv:2408.15664). The
        # checkpoint stores one [num_experts] tensor per MoE layer at
        # `mlp.experts.e_score_correction_bias`; registering it as a Parameter
        # on the FusedMoE lets the weight loader pick it up and the router
        # add it during top-k selection. The fused top-k bias router requires
        # float32 regardless of model dtype.
        e_score_correction_bias = torch.nn.Parameter(
            torch.zeros(config.num_experts, dtype=torch.float32),
            requires_grad=False,
        )

        # FusedMoE with SIGMOID routing. Passing `shared_experts=` lets the
        # layer overlap the shared-expert compute with the all2all dispatch.
        # `apply_routed_scale_to_output=True` makes FusedMoE handle the
        # routed_scaling_factor, shared+routed combine, and TP all-reduce
        # internally, so forward() just returns the final hidden states.
        self.experts = FusedMoE(
            shared_experts=self.shared_expert,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func="sigmoid",
            use_grouped_topk=False,
            apply_router_weight_on_input=bool(config.moe_apply_router_weight_on_input),
            e_score_correction_bias=e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scale_to_output=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits, _ = self.gate(hidden_states)
        router_logits = router_logits.float()
        softcap = getattr(self.config, "moe_router_logit_softcapping", 0.0) or 0.0
        if softcap > 0.0:
            router_logits = torch.tanh(router_logits / softcap) * softcap

        final_hidden_states = self.experts(hidden_states, router_logits)
        return final_hidden_states.view(orig_shape)


class LagunaAttention(nn.Module):
    """Laguna attention with optional softplus output gating.

    Supports per-layer sliding window attention when ``config.layer_types``
    is present.  Layers whose type is ``"sliding_attention"`` use
    ``config.sliding_window``; all other layers (typically labelled
    ``"full_attention"``) use full attention.  When ``layer_types`` is
    absent every layer defaults to full attention for backwards
    compatibility.
    """

    def __init__(
        self,
        config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 131072,
        head_dim: int | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attention_sink: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        # Gating flag
        self.gating = config.gating

        # Per-layer sliding window (follows Gemma2/Cohere2 convention)
        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None:
            layer_idx = extract_layer_index(prefix)
            is_sliding = layer_types[layer_idx] == "sliding_attention"
            self.sliding_window = config.sliding_window if is_sliding else None
        else:
            self.sliding_window = None

        # QKV projection (no bias for Laguna)
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        # Output projection
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Gating projection (Laguna-specific, optional)
        # config.gating may be:
        #   - True / "per-element": one gate per (head, head_dim) channel
        #   - "per-head":           one gate per head, broadcast across head_dim
        if self.gating:
            # v5 LagunaConfig uses ``gating=True`` for per-head; older configs
            # used ``"per-head"``. Accept both. ``"per-element"`` (or legacy
            # ``True``) means per-element gating with output size num_heads ×
            # head_dim.
            gate_per_head = self.gating is True or self.gating == "per-head"
            g_out = (
                self.total_num_heads
                if gate_per_head
                else self.total_num_heads * self.head_dim
            )
            self.g_proj = ColumnParallelLinear(
                hidden_size,
                g_out,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )
            self.gate_per_head = gate_per_head
        else:
            self.g_proj = None
            self.gate_per_head = False

        # Attention sinks (learnable per-head bias for SWA layers)
        sinks = None
        if attention_sink:
            self.sink = torch.nn.Parameter(
                torch.empty(self.total_num_heads // tp_size, requires_grad=False)
            )
            sinks = self.sink

        # Resolve rope params per-layer-type. ``config.rope_parameters`` is
        # either a flat dict (legacy) or a nested ``{layer_type: rope_dict}``
        # (v5 Laguna-XS schema). The v5 form is unhashable as-is and would
        # crash `get_rope`'s cache lookup, so always pull out the layer's
        # sub-dict before forwarding.
        layer_type = (
            layer_types[extract_layer_index(prefix)]
            if layer_types is not None
            else "full_attention"
        )
        is_sliding = layer_type == "sliding_attention"

        top_rope = getattr(config, "rope_parameters", None) or {}
        if any(isinstance(v, dict) for v in top_rope.values()):
            # Nested per-layer-type form.
            base_rope = top_rope.get(layer_type) or top_rope.get("full_attention") or {}
        else:
            base_rope = top_rope

        # Older flat-rope ckpts can carry a separate `swa_rope_parameters`
        # for SWA layers. Prefer it when present; otherwise the nested
        # rope dict above already supplies the correct sub-config.
        swa_rope = getattr(config, "swa_rope_parameters", None)
        if (
            is_sliding
            and swa_rope is None
            and not any(isinstance(v, dict) for v in top_rope.values())
        ):
            logger.warning_once(
                "Laguna config has sliding_attention layers but neither "
                "`swa_rope_parameters` nor a nested per-layer-type "
                "`rope_parameters` — SWA layers will reuse the global rope. "
                "If the checkpoint was trained with distinct SWA rope "
                "(theta / partial_rotary_factor), regenerate its HF config "
                "to include either form."
            )
        rope_params = swa_rope if (is_sliding and swa_rope is not None) else base_rope
        # `partial_rotary_factor` may live on the top-level config (main attention)
        # or on the per-layer rope dict itself (e.g. SWA can differ). Inject the
        # top-level value into `rope_params` if the dict doesn't already set it.
        top_partial = getattr(config, "partial_rotary_factor", None)
        if top_partial is not None and "partial_rotary_factor" not in rope_params:
            rope_params = {**rope_params, "partial_rotary_factor": top_partial}

        # Rotary embeddings (YaRN)
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=max_position_embeddings,
            is_neox_style=True,
            rope_parameters=rope_params,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=self.sliding_window,
            prefix=f"{prefix}.attn",
            sinks=sinks,
        )

        # QK normalization (like Qwen3)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)

        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)

        # Apply gating if enabled (compute softplus in float32 for precision)
        if self.gating and self.g_proj is not None:
            gate, _ = self.g_proj(hidden_states)
            gate = F.softplus(gate.float()).type_as(attn_output)
            if self.gate_per_head:
                # gate: [..., num_heads]; broadcast across head_dim
                attn_shape = attn_output.shape
                attn_output = (
                    attn_output.view(*attn_shape[:-1], self.num_heads, self.head_dim)
                    * gate.unsqueeze(-1)
                ).view(attn_shape)
            else:
                attn_output = attn_output * gate

        output, _ = self.o_proj(attn_output)
        return output


class LagunaDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        layer_idx = extract_layer_index(prefix)

        # Determine if this layer uses sliding window attention
        layer_types = getattr(config, "layer_types", None)
        is_sliding = (
            layer_types is not None and layer_types[layer_idx] == "sliding_attention"
        )

        # Enable attention sinks on SWA layers when configured
        attention_sink = is_sliding and getattr(
            config, "swa_attention_sink_enabled", False
        )

        # Optional per-layer override of head count (Laguna-XS).
        per_layer_heads = getattr(config, "num_attention_heads_per_layer", None)
        layer_num_heads = (
            per_layer_heads[layer_idx]
            if per_layer_heads is not None
            else config.num_attention_heads
        )

        self.self_attn = LagunaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=layer_num_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            attention_sink=attention_sink,
        )

        # Check if this layer uses MoE or dense MLP (matches Qwen2/Qwen3 convention)
        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        self.is_moe_layer = (
            (layer_idx not in mlp_only_layers)
            and (config.num_experts > 0)
            and ((layer_idx + 1) % config.decoder_sparse_step == 0)
        )

        if self.is_moe_layer:
            self.mlp = LagunaMoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                enable_eplb=enable_eplb,
            )
        else:
            self.mlp = LagunaMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
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

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


@support_torch_compile
class LagunaModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        enable_eplb = vllm_config.parallel_config.enable_eplb
        eplb_config = vllm_config.parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts
        self.config = config
        self.quant_config = quant_config

        # Disable the model-level sliding-window fallback in Attention.__init__.
        # Laguna drives SWA per-layer via `layer_types`, passing
        # `per_layer_sliding_window=self.sliding_window` (None for global
        # layers). Without this, global layers whose `per_layer_sliding_window`
        # is None would pick up `cache_config.sliding_window`
        # (populated from `config.sliding_window`) as a fallback, silently
        # applying a 512-token window to full-attention layers.
        if cache_config is not None:
            cache_config.sliding_window = None

        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: LagunaDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
                enable_eplb=enable_eplb,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """Get expert parameter mapping for weight loading.

        Returns mapping tuples of (param_name, weight_name, expert_id, shard_id)
        that handle both weights and quantization scales.
        """
        return FusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # gate_proj and up_proj are loaded as separate Linears (see
            # LagunaMLP) so no merge entry is needed here.
        ]

        # Suffixes to skip for GPTQ/modelopt models if param doesn't exist
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()

        tp_rank = get_tensor_model_parallel_rank()

        for name, loaded_weight in weights:
            # Handle attention sinks (distributed across ranks). Derive the
            # per-rank slice from the parameter's own shape so per-layer
            # variations in head count are handled correctly.
            if "sink" in name:
                param = params_dict.get(name)
                if param is not None:
                    layer_heads_per_rank = param.shape[0]
                    layer_head_start = tp_rank * layer_heads_per_rank
                    narrow_weight = loaded_weight.narrow(
                        0, layer_head_start, layer_heads_per_rank
                    )
                    param.data.copy_(narrow_weight)
                    loaded_params.add(name)
                continue

            # Handle KV cache quantization scales
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                assert loaded_weight.numel() == 1, (
                    f"KV scale numel {loaded_weight.numel()} != 1"
                )
                loaded_weight = loaded_weight.squeeze()
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            # Handle stacked params (QKV, gate_up for
            # non-expert layers and shared_expert)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Skip expert weights - handled below via expert_params_mapping
                if "mlp.experts" in name and "shared_expert" not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                # Remap FP8 kv_scale names for backwards compatibility
                if name.endswith("scale"):
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                # Try expert params mapping (handles weights + quantization scales)
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    # Mark as expert weight so we skip regular loading below
                    is_expert_weight = True

                    # Create mapped name without modifying original
                    name_mapped = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    if (
                        name_mapped.endswith(ignore_suffixes)
                        and name_mapped not in params_dict
                    ):
                        continue
                    if name_mapped not in params_dict:
                        continue

                    param = params_dict[name_mapped]
                    # Use return_success to handle expert parallelism correctly
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        loaded_params.add(name_mapped)
                        break
                else:
                    # Expert weight not mapped to this rank - skip
                    if is_expert_weight:
                        continue

                    # Remap kv_scale names before the ignore_suffixes filter:
                    # the suffix list includes .k_scale/.v_scale, so filtering
                    # first drops the checkpoint key before remap can rewrite
                    # it to the .attn.* name that exists in params_dict.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    if name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)

        return loaded_params


class LagunaForCausalLM(nn.Module, SupportsPP, SupportsLoRA):
    fall_back_to_pt_during_load = False

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.model = LagunaModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if self.config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
