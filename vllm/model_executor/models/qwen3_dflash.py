# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from flashinfer import rmsnorm
from flashinfer.rope import apply_rope_with_cos_sin_cache
from torch import nn
from transformers import Qwen3Config

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.multimodal.inputs import NestedTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backend import AttentionType

from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen3 import Qwen3ForCausalLM
from .utils import (
    AutoWeightsLoader,
    get_draft_quant_config,
    maybe_prefix,
    process_eagle_weight,
)

logger = init_logger(__name__)


class DFlashQwen3Attention(nn.Module):
    """Attention for DFlash speculative decoding.

    Context KVs are pre-inserted into the KV cache before the forward pass.
    This layer handles only query tokens via standard attention.
    Adapted from Qwen3Attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        self.layer_name = prefix
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
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            dtype=torch.float32,  # required for flashinfer rope
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = F.linear(hidden_states, self.qkv_proj.weight, self.qkv_proj.bias)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Per-head RMSNorm
        q_shape, k_shape = q.shape, k.shape
        q = self.q_norm(
            q.view(*q_shape[:-1], q_shape[-1] // self.head_dim, self.head_dim)
        ).view(q_shape)
        k = self.k_norm(
            k.view(*k_shape[:-1], k_shape[-1] // self.head_dim, self.head_dim)
        ).view(k_shape)

        q, k = self.rotary_emb(positions, q, k)

        # Standard attention: writes query K/V to cache, reads full
        # sequence from cache (including pre-inserted context KVs).
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class DFlashQwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config: Qwen3Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)
        attn_type = AttentionType.DECODER

        self.self_attn = DFlashQwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
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
        if residual is not None:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# NOTE: torch.compile is not stable here
@support_torch_compile
class DFlashQwen3Model(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size
        self.quant_config = get_draft_quant_config(vllm_config)

        drafter_config = getattr(self.config, "eagle_config", {})
        drafter_config.update(getattr(self.config, "dflash_config", {}))

        if drafter_config is not None and "use_aux_hidden_state" in drafter_config:
            self.use_aux_hidden_state = drafter_config["use_aux_hidden_state"]
        else:
            self.use_aux_hidden_state = True

        current_vllm_config = get_current_vllm_config()

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.layers = nn.ModuleList(
            [
                DFlashQwen3DecoderLayer(
                    current_vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{layer_idx + start_layer_id}"),
                    config=self.config,
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        if self.use_aux_hidden_state:
            num_features_to_use = self.config.num_hidden_layers
            if "target_layer_ids" in drafter_config:
                num_features_to_use = len(drafter_config["target_layer_ids"])
            elif "layer_ids" in drafter_config:
                num_features_to_use = len(drafter_config["layer_ids"])
            if hasattr(self.config, "target_hidden_size"):
                fc_input_size = self.config.target_hidden_size * num_features_to_use
            else:
                fc_input_size = self.config.hidden_size * num_features_to_use
            self.fc = ReplicatedLinear(
                input_size=fc_input_size,
                output_size=self.config.hidden_size,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "fc"),
                return_bias=False,
            )
        self.hidden_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _build_fused_kv_buffers(self) -> None:
        """Build fused weight buffers for precompute_and_store_context_kv.

        Must be called after weights are loaded.  Stacks the KV-projection
        weights, K-norm weights, and RoPE parameters from every attention
        layer so that precompute_and_store_context_kv can run one fused
        GEMM for all layers at once.
        """
        layers_attn = [layer.self_attn for layer in self.layers]
        attn0 = layers_attn[0]
        has_bias = attn0.qkv_proj.bias is not None

        # Hidden norm weight:
        self._hidden_norm_weight = self.hidden_norm.weight.data

        # KV projection weights: [num_layers * 2 * kv_size, hidden_size]
        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
        self._fused_kv_weight = torch.cat(kv_weights, dim=0)
        if has_bias:
            kv_biases = [a.qkv_proj.bias[a.q_size :] for a in layers_attn]
            self._fused_kv_bias: torch.Tensor | None = torch.cat(kv_biases, dim=0)
        else:
            self._fused_kv_bias = None

        # K-norm weights: list of [head_dim] tensors, one per layer.
        # Used with flashinfer rmsnorm (one fused kernel per layer).
        self._k_norm_weights = [a.k_norm.weight.data for a in layers_attn]

        # RoPE params (shared across all layers)
        self._rope_head_size = attn0.rotary_emb.head_size
        self._rope_cos_sin_cache = attn0.rotary_emb.cos_sin_cache
        self._rope_is_neox = attn0.rotary_emb.is_neox_style

        # Layer metadata
        self._num_attn_layers = len(layers_attn)
        self._kv_size = attn0.kv_size
        self._head_dim = attn0.head_dim
        self._num_kv_heads = attn0.num_kv_heads
        self._rms_norm_eps = attn0.q_norm.variance_epsilon

        # References to inner Attention layers for direct cache writes
        self._attn_layers = [layer.self_attn.attn for layer in self.layers]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        """Precompute projected, normed, RoPE'd K/V for context states
        and optionally write them directly into each layer's KV cache.

        When context_slot_mapping is None (e.g. during dummy_run), only
        the computation runs — useful for memory profiling without
        requiring valid slot mappings or a bound KV cache.
        """
        num_ctx = context_states.shape[0]
        L = self._num_attn_layers
        kv = self._kv_size
        hd = self._head_dim
        nkv = self._num_kv_heads

        # --- Fused KV projection (one GEMM for all layers) ---
        normed_context_states = rmsnorm(
            input=context_states,
            weight=self._hidden_norm_weight,
            eps=self._rms_norm_eps,
        )
        all_kv_flat = F.linear(
            normed_context_states, self._fused_kv_weight, self._fused_kv_bias
        )
        # Single contiguous copy that separates K/V and transposes to
        # layer-major layout.  Result: [2, L, num_ctx, nkv, hd] contiguous.
        # Indexing dim-0 gives contiguous [L, num_ctx, nkv, hd] for K and V.
        all_kv = (
            all_kv_flat.view(num_ctx, L, 2, nkv, hd).permute(2, 1, 0, 3, 4).contiguous()
        )
        all_k = all_kv[0]  # [L, num_ctx, nkv, hd], contiguous
        all_v = all_kv[1]  # [L, num_ctx, nkv, hd], contiguous

        # --- Per-layer RMSNorm K (3D: [num_ctx, nkv, hd] per layer) ---
        all_k_normed = torch.empty_like(all_k)
        for i in range(L):
            rmsnorm(
                input=all_k[i],
                weight=self._k_norm_weights[i],
                eps=self._rms_norm_eps,
                out=all_k_normed[i],
            )

        # --- Fused RoPE across all layers ---
        # View as [L * num_ctx, kv] so RoPE sees one big batch (no copy).
        # We pass all_k_normed as both query and key; the non-inplace version
        # allocates separate outputs so both read from the unmodified input.
        all_k_flat = all_k_normed.view(L * num_ctx, kv)
        positions_repeated = context_positions.repeat(L)
        _, all_k_flat = apply_rope_with_cos_sin_cache(
            positions=positions_repeated,
            query=all_k_flat,
            key=all_k_flat,
            head_size=self._rope_head_size,
            cos_sin_cache=self._rope_cos_sin_cache,
            is_neox=self._rope_is_neox,
        )

        if context_slot_mapping is None:
            return

        # --- Per-layer cache insert ---
        all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
        for i in range(L):
            attn = self._attn_layers[i]
            kv_cache = attn.kv_cache[0]  # virtual_engine=0
            attn.impl.do_kv_cache_update(
                attn,
                all_k_final[i],
                all_v[i],
                kv_cache,
                context_slot_mapping,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)

        hidden_states = input_embeds

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "midlayer." in name:
                name = name.replace("midlayer.", "layers.0.")
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.config.target_layer_count = target_layer_num
        self.model = DFlashQwen3Model(
            vllm_config=vllm_config,
            prefix="model",
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        self.draft_id_to_target_id = None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            return logits

        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full(
            (logits.shape[0], self.config.vocab_size),
            float("-inf"),
        )
        logits_new[:, targets] = logits
        return logits_new

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        """Precompute projected + RoPE'd K/V and write to cache."""
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not self.model.use_aux_hidden_state:
            return hidden_states
        needs_squeeze = hidden_states.dim() == 1
        if needs_squeeze:
            hidden_states = hidden_states.unsqueeze(0)
        result = self.model.fc(hidden_states)
        if needs_squeeze:
            result = result.squeeze(0)
        return result

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        for name, loaded_weight in weights:
            assert "mask_hidden" not in name, (
                "DFlash should use mask_token_id to embed the padding hidden state"
            )
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        skip_substrs = []
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not self.model.use_aux_hidden_state:
            skip_substrs.append("fc.")
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None,
            skip_substrs=skip_substrs,
        )
        loader.load_weights(model_weights.items())
        self.model._build_fused_kv_buffers()
