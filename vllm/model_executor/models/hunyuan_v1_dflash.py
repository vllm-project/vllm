# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from vllm import _custom_ops as ops
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
from vllm.multimodal.inputs import NestedTensors
from vllm.v1.attention.backend import AttentionType

from .hunyuan_v1 import HunYuanMLP, HunYuanSparseMoeBlock, _is_moe
from .utils import (
    AutoWeightsLoader,
    get_draft_quant_config,
    maybe_prefix,
    process_eagle_weight,
)

logger = init_logger(__name__)


class DFlashHunYuanAttention(nn.Module):
    """DFlash Attention for HunyuanVL draft model.

    Key differences from the standard DFlash attention:
    1. RoPE is applied before QK-Norm (consistent with HunYuanAttention).
    2. QK-Norm uses view(-1, heads, head_dim) reshape.
    3. Q/K norm layers are named query_layernorm / key_layernorm.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 8192,
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

        # head_dim: prefer head_dim/attention_head_dim if present, else compute
        if hasattr(config, "head_dim") and config.head_dim:
            self.head_dim = config.head_dim
        elif hasattr(config, "attention_head_dim"):
            self.head_dim = config.attention_head_dim
        else:
            self.head_dim = hidden_size // self.total_num_heads

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.use_qk_norm = getattr(config, "use_qk_norm", False)

        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False
        )

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        # Draft model always uses standard RoPE (not XDRoPE from target model).
        # HunYuanVLTextConfig has no rope_parameters field; build from rope_theta.
        rope_params = getattr(config, "rope_parameters", None)
        if not rope_params:
            rope_params = {
                "rope_type": "default",
                "rope_theta": getattr(config, "rope_theta", 10000.0),
            }
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_params,
            is_neox_style=True,
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
        if self.use_qk_norm:
            # HunyuanVL naming: query_layernorm / key_layernorm
            self.query_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """DFlash attention forward: context KV is already in cache from
        precompute_and_store_context_kv; this processes query tokens only.

        Note: RoPE is applied before QK-Norm.
        """
        qkv = F.linear(hidden_states, self.qkv_proj.weight, self.qkv_proj.bias)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # HunyuanVL: RoPE first, then QK-Norm
        q, k = self.rotary_emb(positions, q, k)

        if self.use_qk_norm:
            q = self.query_layernorm(
                q.view(-1, self.num_heads, self.head_dim).contiguous()
            ).view(-1, self.q_size)
            k = self.key_layernorm(
                k.view(-1, self.num_kv_heads, self.head_dim).contiguous()
            ).view(-1, self.kv_size)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class DFlashHunYuanDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        layer_id: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        self.self_attn = DFlashHunYuanAttention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            attn_type=AttentionType.DECODER,
        )

        # Supports Dense / MoE FFN variants
        if _is_moe(config):
            self.mlp = HunYuanSparseMoeBlock(
                config=config,
                quant_config=quant_config,
                layer_id=layer_id,
                prefix=f"{prefix}.mlp",
            )
        else:
            intermediate_size = config.intermediate_size
            if isinstance(intermediate_size, list):
                intermediate_size = intermediate_size[layer_id]
            self.mlp = HunYuanMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                bias=getattr(config, "mlp_bias", False),
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

        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class DFlashHunYuanModel(nn.Module):
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

        # Read dflash_config fields
        drafter_config = getattr(self.config, "eagle_config", {})
        drafter_config.update(getattr(self.config, "dflash_config", {}))

        self.use_aux_hidden_state = drafter_config.get("use_aux_hidden_state", True)

        current_vllm_config = get_current_vllm_config()

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.layers = nn.ModuleList(
            [
                DFlashHunYuanDecoderLayer(
                    current_vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{layer_idx + start_layer_id}"),
                    config=self.config,
                    layer_id=layer_idx,
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )

        # Projection from concatenated aux hidden states to draft hidden size
        if self.use_aux_hidden_state:
            num_features = self.config.num_hidden_layers
            if "target_layer_ids" in drafter_config:
                num_features = len(drafter_config["target_layer_ids"])
            elif "layer_ids" in drafter_config:
                num_features = len(drafter_config["layer_ids"])

            target_hidden = getattr(
                self.config, "target_hidden_size", self.config.hidden_size
            )
            fc_input_size = target_hidden * num_features
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
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _build_fused_kv_buffers(self) -> None:
        """Build fused weight buffers for precompute_and_store_context_kv.

        Must be called after weights are loaded. Stacks KV-projection weights,
        K-norm weights (after RoPE), and RoPE parameters from every attention
        layer so that precompute_and_store_context_kv can run one fused GEMM
        for all layers at once.

        Key difference: QK-Norm is applied after RoPE, so
        the stage order is: GEMM -> RoPE -> K-Norm (per-layer, optional).
        """
        layers_attn = [layer.self_attn for layer in self.layers]
        attn0 = layers_attn[0]
        has_bias = attn0.qkv_proj.bias is not None

        self._hidden_norm_weight = self.hidden_norm.weight.data

        # KV projection weights: [L * 2 * kv_size, hidden_size]
        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
        self._fused_kv_weight = torch.cat(kv_weights, dim=0)
        if has_bias:
            kv_biases = [a.qkv_proj.bias[a.q_size :] for a in layers_attn]
            self._fused_kv_bias: torch.Tensor | None = torch.cat(kv_biases, dim=0)
        else:
            self._fused_kv_bias = None

        # HunyuanVL: K-norm named key_layernorm; applied after RoPE
        self._k_norm_weights = None
        if attn0.use_qk_norm:
            self._k_norm_weights = [a.key_layernorm.weight.data for a in layers_attn]

        # RoPE parameters (validate consistency across all layers)
        self._rope_head_size = attn0.rotary_emb.head_size
        self._rope_cos_sin_cache = attn0.rotary_emb.cos_sin_cache
        self._rope_is_neox = attn0.rotary_emb.is_neox_style
        for a in layers_attn[1:]:
            assert (
                a.rotary_emb.head_size == self._rope_head_size
                and a.rotary_emb.is_neox_style == self._rope_is_neox
            ), "All layers must have the same RoPE parameters for DFlash precomputation"

        # Layer metadata (validate consistency across all layers)
        self._num_attn_layers = len(layers_attn)
        self._kv_size = attn0.kv_size
        self._head_dim = attn0.head_dim
        self._num_kv_heads = attn0.num_kv_heads
        self._rms_norm_eps = (
            attn0.query_layernorm.variance_epsilon
            if attn0.use_qk_norm
            else self.config.rms_norm_eps
        )
        for a in layers_attn[1:]:
            assert (
                a.kv_size == self._kv_size
                and a.head_dim == self._head_dim
                and a.num_kv_heads == self._num_kv_heads
            ), "All layers must have the same attn config for DFlash precomputation"
        self._attn_layers = [layer.self_attn.attn for layer in self.layers]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        """Precompute K/V for context states and write to each layer's KV cache.

        Input context states are projected to K/V, have RoPE applied, then
        optional per-layer K-Norm. Uses fused kernels to minimize ops.

        Stage order:
          GEMM -> RoPE(fused) -> K-Norm(per-layer, optional) -> cache

        When context_slot_mapping is None (e.g. during dummy_run), only
        computation runs without writing to cache.
        """
        if not hasattr(self, "_num_attn_layers"):
            logger.warning_once(
                "DFlash buffer initialization was skipped. If dummy weights "
                "are not in use, this may indicate an error in weight loading."
            )
            self._build_fused_kv_buffers()

        num_ctx = context_states.shape[0]
        L = self._num_attn_layers
        kv = self._kv_size
        hd = self._head_dim
        nkv = self._num_kv_heads

        # Stage 1: hidden_norm
        normed = torch.empty_like(context_states)
        ops.rms_norm(
            normed, context_states, self._hidden_norm_weight, self._rms_norm_eps
        )

        # Stage 2: Fused KV projection (one GEMM for all layers)
        # _fused_kv_weight: [L * 2 * kv_size, hidden_size]
        all_kv_flat = F.linear(normed, self._fused_kv_weight, self._fused_kv_bias)
        # Single contiguous copy that separates K/V and transposes to
        # layer-major layout. Result: [2, L, num_ctx, nkv, hd] contiguous.
        all_kv = (
            all_kv_flat.view(num_ctx, L, 2, nkv, hd).permute(2, 1, 0, 3, 4).contiguous()
        )
        all_k = all_kv[0]  # [L, num_ctx, nkv, hd]
        all_v = all_kv[1]  # [L, num_ctx, nkv, hd]

        # Stage 3: Fused RoPE across all layers (HunyuanVL: RoPE before K-Norm)
        # View as [L * num_ctx, kv] so RoPE sees one big batch (no copy).
        all_k_flat = all_k.view(L * num_ctx, kv)
        positions_repeated = context_positions.repeat(L)
        cos_sin_cache = self._rope_cos_sin_cache
        if cos_sin_cache.dtype != all_k_flat.dtype:
            cos_sin_cache = cos_sin_cache.to(dtype=all_k_flat.dtype)
        ops.rotary_embedding(
            positions_repeated,
            all_k_flat,
            None,
            self._rope_head_size,
            cos_sin_cache,
            self._rope_is_neox,
        )

        # Stage 4 (optional): Per-layer K-Norm applied after RoPE
        if self._k_norm_weights is not None:
            all_k_after_rope = all_k_flat.view(L, num_ctx, nkv, hd)
            all_k_normed = torch.empty_like(all_k_after_rope)
            for i in range(L):
                ops.rms_norm(
                    all_k_normed[i],
                    all_k_after_rope[i],
                    self._k_norm_weights[i],
                    self._rms_norm_eps,
                )
            all_k_final = all_k_normed
        else:
            all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)

        if context_slot_mapping is None:
            return  # dummy_run: compute only, do not write to cache

        # Stage 5: Write to per-layer KV cache
        for i in range(L):
            attn = self._attn_layers[i]
            kv_cache = attn.kv_cache
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
        """Process query tokens only (context KV already in cache)."""
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


class DFlashHunYuanV1ForCausalLM(nn.Module):
    """DFlash draft model base class shared by Dense and MoE variants.

    Naming convention: eagle.py's dflash method auto-prepends "DFlash" to the
    target model architecture name, yielding:
      DFlashHunYuanDenseV1ForCausalLM / DFlashHunYuanMoEV1ForCausalLM.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.config.target_layer_count = target_layer_num
        self.model = DFlashHunYuanModel(
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
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None
        # Signal that embed_tokens / lm_head are not shared with target model
        self.has_own_embed_tokens = False
        self.has_own_lm_head = False

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

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            return logits
        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full(
            (logits.shape[0], self.config.vocab_size), float("-inf")
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

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project multi-layer aux hidden states to draft model dimension via fc."""
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
        loader = AutoWeightsLoader(self, skip_prefixes=None, skip_substrs=skip_substrs)
        loader.load_weights(model_weights.items())
        self.model._build_fused_kv_buffers()


# Two named subclasses corresponding to Dense and MoE target models.
# eagle.py dflash method concatenates: "DFlash" + "HunYuanDenseV1ForCausalLM"
class DFlashHunYuanDenseV1ForCausalLM(DFlashHunYuanV1ForCausalLM):
    pass


class DFlashHunYuanMoEV1ForCausalLM(DFlashHunYuanV1ForCausalLM):
    pass
