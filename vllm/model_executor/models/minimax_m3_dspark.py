# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3.1 DSpark draft: dense non-causal GQA + Markov/confidence heads.

Official MiniMax-M3.1-Flash ships this drafter under ``{model}/dspark`` with
architecture ``DSparkMiniMaxDraftModel``. The backbone is MiniMax M3's dense
building blocks (Gemma RMSNorm, partial RoPE, SwiGLU-OAI MLP) rather than MSA
or routed MoE. Context K/V is precomputed DFlash-style; the query block is
non-causal over a 4096-token sliding window.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
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
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.models.minimax_m3.common.dspark_mapper import (
    DSPARK_ORIG_TO_NEW_REGEX,
    DSPARK_ORIG_TO_NEW_STACKED,
)
from vllm.v1.attention.backend import AttentionType

from .qwen3_dflash import DFlashQwen3Model
from .qwen3_dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
    Qwen3DSparkForCausalLM,
)
from .utils import WeightsMapper, maybe_prefix, process_eagle_weight

logger = init_logger(__name__)


def _cfg_get(config: Any, *names: str, default: Any = None) -> Any:
    """Read a DSpark field from the draft config or its nested text_config."""
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
        text = getattr(config, "text_config", None)
        if text is not None:
            value = getattr(text, name, None)
            if value is not None:
                return value
    return default


def _minimax_dspark_rotary_emb(config: Any, head_dim: int):
    rope_parameters = {
        "rope_theta": getattr(config, "rope_theta", 10000000),
        "partial_rotary_factor": getattr(config, "partial_rotary_factor", 0.5),
    }
    max_position = getattr(config, "max_position_embeddings", 1048576)
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling:
        rope_parameters.update(rope_scaling)
        if "rope_type" not in rope_parameters and "type" in rope_scaling:
            rope_parameters["rope_type"] = rope_scaling["type"]
        rope_parameters.setdefault(
            "original_max_position_embeddings",
            getattr(config, "max_position_embeddings", max_position),
        )
        factor = float(rope_scaling.get("factor", 1.0))
        max_position = int(rope_parameters["original_max_position_embeddings"] * factor)
    return get_rope(
        head_dim,
        max_position=max_position,
        rope_parameters=rope_parameters,
    )


class MiniMaxM3DSparkMLP(nn.Module):
    """Dense SwiGLU-OAI MLP used by every MiniMax DSpark draft layer."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        swiglu_limit: float,
        swiglu_alpha: float,
        swiglu_beta: float,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
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
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMulWithClamp(
            swiglu_limit=swiglu_limit,
            alpha=swiglu_alpha,
            beta=swiglu_beta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MiniMaxM3DSparkAttention(nn.Module):
    """Dense GQA with MiniMax Gemma QK-norm and partial RoPE."""

    def __init__(
        self,
        config: Any,
        *,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
        sliding_window: int | None,
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
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
        # MiniMax-M3.1 DSpark is non-causal; the DFlash mask supplies visibility.
        self.causal = False
        rms_eps = getattr(config, "rms_norm_eps", 1e-6)

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
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
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=rms_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=rms_eps)
        self.rotary_emb = _minimax_dspark_rotary_emb(config, self.head_dim)
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.DECODER,
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_shape, k_shape = q.shape, k.shape
        q = self.q_norm(
            q.view(*q_shape[:-1], q_shape[-1] // self.head_dim, self.head_dim)
        ).view(q_shape)
        k = self.k_norm(
            k.view(*k_shape[:-1], k_shape[-1] // self.head_dim, self.head_dim)
        ).view(k_shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class MiniMaxM3DSparkDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
        sliding_window: int | None,
    ) -> None:
        super().__init__()
        rms_eps = getattr(config, "rms_norm_eps", 1e-6)
        ffn_size = int(
            _cfg_get(
                config,
                "dspark_ffn_hidden_size",
                "dense_intermediate_size",
                default=config.intermediate_size,
            )
        )
        self.self_attn = MiniMaxM3DSparkAttention(
            config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            sliding_window=sliding_window,
        )
        self.mlp = MiniMaxM3DSparkMLP(
            config.hidden_size,
            ffn_size,
            swiglu_limit=float(getattr(config, "swiglu_limit", 7.0)),
            swiglu_alpha=float(getattr(config, "swiglu_alpha", 1.702)),
            swiglu_beta=float(getattr(config, "swiglu_beta", 1.0)),
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=rms_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=rms_eps)

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
class MiniMaxM3DSparkModel(DFlashQwen3Model):
    """MiniMax DSpark backbone plus Markov / confidence heads."""

    decoder_layer_cls = MiniMaxM3DSparkDecoderLayer
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_regex=DSPARK_ORIG_TO_NEW_REGEX,
        orig_to_new_stacked=DSPARK_ORIG_TO_NEW_STACKED,
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        text = getattr(config, "text_config", None) or config
        self.config = text if getattr(text, "num_hidden_layers", None) else config
        self.quant_config = vllm_config.quant_config
        self.use_aux_hidden_state = True
        target_ids = _cfg_get(
            config, "dspark_target_layer_ids", "target_layer_ids", default=()
        )
        self.target_layer_ids = tuple(int(i) for i in target_ids)
        current = get_current_vllm_config()
        cache_config = current.cache_config
        quant_config = current.quant_config
        rms_eps = getattr(self.config, "rms_norm_eps", 1e-6)
        sliding_window = int(getattr(self.config, "sliding_window", 4096) or 4096)

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.fc = ReplicatedLinear(
            self.config.hidden_size * max(len(self.target_layer_ids), 1),
            self.config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "fc"),
        )
        self.hidden_norm = GemmaRMSNorm(self.config.hidden_size, eps=rms_eps)
        self.layers = nn.ModuleList(
            MiniMaxM3DSparkDecoderLayer(
                self.config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, f"layers.{i}"),
                sliding_window=sliding_window,
            )
            for i in range(self.config.num_hidden_layers)
        )
        self.norm = GemmaRMSNorm(self.config.hidden_size, eps=rms_eps)
        markov_rank = int(
            _cfg_get(config, "markov_rank", "dspark_markov_rank", default=256)
        )
        draft_vocab = (
            getattr(self.config, "draft_vocab_size", None) or self.config.vocab_size
        )
        self.markov_head = DSparkMarkovHead(
            self.config.vocab_size,
            draft_vocab,
            markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
            quant_config=quant_config,
        )
        self.confidence_head: DSparkConfidenceHead | None = None
        if _cfg_get(config, "enable_confidence_head", default=False):
            with_markov = bool(
                _cfg_get(config, "confidence_head_with_markov", default=False)
            )
            input_dim = self.config.hidden_size + (markov_rank if with_markov else 0)
            self.confidence_head = DSparkConfidenceHead(
                input_dim,
                prefix=maybe_prefix(prefix, "confidence_head"),
                bias=True,
                with_markov=with_markov,
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _build_fused_kv_buffers(self) -> None:
        super()._build_fused_kv_buffers()
        # Gemma RMSNorm is ``x * rsqrt(var+eps) * (1+w)``. DFlash's fused
        # ``ops.rms_norm`` multiplies by ``w``, so pass ``1+w``.
        self._hidden_norm_weight = (1.0 + self.hidden_norm.weight.data).contiguous()
        layers_attn = [layer.self_attn for layer in self.layers]
        self._k_norm_weights = (
            1.0 + torch.stack([a.k_norm.weight.data for a in layers_attn], dim=0)
        ).contiguous()

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        """Gemma-norm KV precompute with MiniMax partial RoPE."""
        if not hasattr(self, "_num_attn_layers"):
            logger.warning_once(
                "MiniMax DSpark buffer initialization was skipped. If dummy "
                "weights are not in use, this may indicate an error in weight "
                "loading."
            )
            self._build_fused_kv_buffers()

        num_ctx = context_states.shape[0]
        layers = self._num_attn_layers
        head_dim = self._head_dim
        num_kv_heads = self._num_kv_heads
        all_k, all_v = self._project_context_kv(
            context_states, num_ctx, layers, num_kv_heads, head_dim
        )
        all_k_normed = self._normalize_context_k(all_k)
        if context_slot_mapping is None:
            return
        per_layer = isinstance(context_slot_mapping, (list, tuple))
        for i, layer in enumerate(self.layers):
            slot_mapping = (
                context_slot_mapping[i] if per_layer else context_slot_mapping
            )
            if slot_mapping is None:
                continue
            k = all_k_normed[i].reshape(num_ctx, num_kv_heads * head_dim)
            dummy_q = k
            _, k = layer.self_attn.rotary_emb(context_positions, dummy_q, k)
            attn = self._attn_layers[i]
            attn.impl.do_kv_cache_update(
                attn,
                k.view(num_ctx, num_kv_heads, head_dim),
                all_v[i],
                attn.kv_cache,
                slot_mapping,
            )


class MiniMaxM3DSparkForCausalLM(Qwen3DSparkForCausalLM):
    """Registered as ``DSparkMiniMaxDraftModel`` for MiniMax-M3.1-Flash."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        self.draft_model_config = speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        text = getattr(self.config, "text_config", None)
        if text is not None and getattr(text, "hidden_size", None):
            self.config = text
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        self.model = MiniMaxM3DSparkModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.draft_vocab_size)
        self.target_vocab_size = vllm_config.model_config.get_vocab_size()
        self.draft_id_to_target_id = None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = MiniMaxM3DSparkModel.hf_to_vllm_mapper
        params = dict(self.named_parameters())
        params.update(dict(self.named_buffers()))
        loaded: set[str] = set()
        includes_confidence_head = False
        for name, weight in weights:
            mapped = mapper._map_name_with_shard(name)
            if mapped is None:
                continue
            mapped_name, shard_id = mapped
            if not mapped_name.startswith("lm_head"):
                mapped_name = "model." + mapped_name
            if "confidence_head" in mapped_name:
                includes_confidence_head = True
            process_eagle_weight(self, mapped_name)
            if mapped_name not in params:
                continue
            param = params[mapped_name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            if shard_id is not None:
                loader(param, weight, shard_id)
            else:
                loader(param, weight)
            loaded.add(mapped_name)
        if not includes_confidence_head:
            self.model.confidence_head = None
        self.model._build_fused_kv_buffers()
        return loaded


# Official MiniMax-M3.1-Flash architecture name.
DSparkMiniMaxDraftModel = MiniMaxM3DSparkForCausalLM
