# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXAONE MoE DSpark draft model backed by in-checkpoint ``mtp.*`` weights."""

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm._custom_ops as ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.exaone_moe import (
    ExaoneMoeDecoderLayer,
    ExaoneMoeForCausalLM,
)
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
from vllm.model_executor.models.qwen3_dspark import (
    DSparkMarkovHead,
    Qwen3DSparkForCausalLM,
)

from .utils import AutoWeightsLoader, WeightsMapper, maybe_prefix


@support_torch_compile
class ExaoneMoeDSparkModel(DFlashQwen3Model):
    """EXAONE MoE DSpark draft backbone (EXAONE dense layers + DSpark Markov head)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.use_aux_hidden_state = True

        self.target_layer_ids = tuple(config.dspark_target_layer_ids)
        num_dspark_layers = config.num_nextn_predict_layers

        # Filled with the target embedding by load_dspark_model.
        self.embed_tokens: nn.Module | None = None
        self.main_proj = ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "mtp.main_proj"),
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        current_vllm_config = get_current_vllm_config()
        self.layers = nn.ModuleList(
            ExaoneMoeDecoderLayer(
                config,
                cache_config=current_vllm_config.cache_config,
                quant_config=self.quant_config,
                is_mtp=True,
                prefix=maybe_prefix(prefix, f"mtp.layers.{layer_idx}"),
            )
            for layer_idx in range(num_dspark_layers)
        )
        for layer in self.layers:
            layer.self_attn.causal = False
            layer.self_attn.apply_rope_all_layers = True

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            config.vocab_size,
            config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, "mtp.markov_head"),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.embed_tokens is not None
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.main_norm(self.main_proj(hidden_states))

    def _build_context_kv_buffers(
        self, layers_attn: list[nn.Module], has_bias: bool
    ) -> None:
        kv_weights = [attn.qkv_proj.weight[attn.q_size :] for attn in layers_attn]
        self._fused_kv_weight = torch.cat(kv_weights, dim=0)
        self._fused_kv_bias: torch.Tensor | None = None
        if has_bias:
            self._fused_kv_bias = torch.cat(
                [attn.qkv_proj.bias[attn.q_size :] for attn in layers_attn], dim=0
            )
        self._k_norm_weights = torch.stack(
            [attn.k_norm.weight.data for attn in layers_attn], dim=0
        ).contiguous()

    def _project_context_kv(
        self,
        context_states: torch.Tensor,
        num_ctx: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_kv_flat = F.linear(
            context_states, self._fused_kv_weight, self._fused_kv_bias
        )
        all_kv = (
            all_kv_flat.view(num_ctx, num_layers, 2, num_kv_heads, head_dim)
            .permute(2, 1, 0, 3, 4)
            .contiguous()
        )
        return all_kv[0], all_kv[1]

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        if not hasattr(self, "_num_attn_layers"):
            self._build_fused_kv_buffers()

        num_ctx = context_states.shape[0]
        num_layers = self._num_attn_layers
        all_k, all_v = self._project_context_kv(
            context_states,
            num_ctx,
            num_layers,
            self._num_kv_heads,
            self._head_dim,
        )
        all_k = self._normalize_context_k(all_k)
        per_layer_slots = isinstance(context_slot_mapping, (list, tuple))

        for layer_idx, layer in enumerate(self.layers):
            dspark_attn = layer.self_attn
            k = all_k[layer_idx]
            if dspark_attn.sliding_window_size or dspark_attn.apply_rope_all_layers:
                k_flat = k.view(num_ctx, self._kv_size)
                cos_sin_cache = dspark_attn.rotary_emb.cos_sin_cache
                if cos_sin_cache.dtype != k_flat.dtype:
                    cos_sin_cache = cos_sin_cache.to(dtype=k_flat.dtype)
                ops.rotary_embedding(
                    context_positions,
                    k_flat,
                    None,
                    dspark_attn.rotary_emb.head_size,
                    cos_sin_cache,
                    dspark_attn.rotary_emb.is_neox_style,
                )

            if context_slot_mapping is None:
                continue
            slot_mapping = (
                context_slot_mapping[layer_idx]
                if per_layer_slots
                else context_slot_mapping
            )
            if slot_mapping is None:
                continue
            attn = dspark_attn.attn
            attn.impl.do_kv_cache_update(
                attn,
                k,
                all_v[layer_idx],
                attn.kv_cache,
                slot_mapping,
            )


class ExaoneMoeDSparkForCausalLM(Qwen3DSparkForCausalLM):
    """EXAONE MoE DSpark speculator over draft weights bundled in the target."""

    hf_to_vllm_mapper = ExaoneMoeForCausalLM.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_prefix={
            "mtp.confidence_head.": None,
            "mtp.fc.": "model.main_proj.",
            "mtp.hidden_norm.": "model.main_norm.",
            "mtp.layers.": "model.layers.",
            "mtp.markov_head.": "model.markov_head.",
            "mtp.norm.": "model.norm.",
        }
    )
    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        self.model = ExaoneMoeDSparkModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # Assigned from the target after the in-checkpoint draft weights load.
        self.lm_head: nn.Module | None = None
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
            scale=getattr(self.config, "logit_scale", 1.0),
        )

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = (
            (name, weight) for name, weight in weights if name.startswith("mtp.")
        )
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        self.model._build_fused_kv_buffers()
        return loaded
