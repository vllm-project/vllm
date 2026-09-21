# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 MLA draft for DFlash2 speculative decoding."""

from collections.abc import Iterable

import torch
from torch import nn

import vllm._custom_ops as ops
from vllm.compilation.backends import set_model_tag
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    yarn_get_mscale,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.qwen3_dflash import (
    _get_dflash_fc_input_size,
    _resolve_layer_attention,
)
from vllm.model_executor.models.qwen3_dflash2 import (
    CandidateSelector,
    DFlash2DecoderLayer,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    get_draft_quant_config,
    maybe_prefix,
)
from vllm.models.common.ops import fused_q_kv_rmsnorm
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)


class DFlash2K3Attention(nn.Module):
    """Non-causal MLA over target context and the complete draft query block."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        sliding_window: int | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.causal = False
        self.layer_name = prefix
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        tp_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % tp_size == 0
        self.num_local_heads = config.num_attention_heads // tp_size
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size != 1
            or parallel_config.decode_context_parallel_size != 1
        ):
            raise NotImplementedError(
                "MLA DFlash2 does not support context parallelism."
            )

        quant_config = get_draft_quant_config(vllm_config)
        self.fused_qkv_a_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            bias=False,
            quant_config=quant_config,
            disable_tp=True,
            prefix=f"{prefix}.fused_qkv_a_proj",
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            config.num_attention_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            config.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.v_head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=True,
            prefix=f"{prefix}.o_proj",
        )

        rope_parameters = dict(config.rope_parameters)
        if rope_parameters["rope_type"] != "default":
            rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=False,
            dtype=torch.float32,
        )
        scale = self.qk_head_dim**-0.5
        if rope_parameters["rope_type"] == "deepseek_yarn":
            mscale = yarn_get_mscale(
                rope_parameters["factor"], rope_parameters.get("mscale_all_dim", 0)
            )
            scale *= mscale * mscale
        self.attn = MLAAttention(
            num_heads=self.num_local_heads,
            scale=scale,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            kv_b_proj=self.kv_b_proj,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=prefix,
            attn_backend=(
                AttentionBackendEnum.TRITON_MLA.get_class()
                if sliding_window is not None
                else None
            ),
            non_causal_multi_token_decode=True,
            sliding_window=sliding_window,
        )
        logger.info_once(
            "Kimi-K3 DFlash2 %s attention uses %s.",
            "sliding-window" if sliding_window is not None else "full",
            self.attn.attn_backend.get_name(),
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        qkv = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_c, k_pe = qkv.split(
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        q_c, kv_c = fused_q_kv_rmsnorm(
            q_c,
            kv_c,
            self.q_a_layernorm.weight,
            self.kv_a_layernorm.weight,
            self.q_a_layernorm.variance_epsilon,
        )
        q = self.q_b_proj(q_c)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        q_pe = q[..., self.qk_nope_head_dim :]
        k_pe = k_pe.unsqueeze(1)
        ops.rotary_embedding(
            positions,
            q_pe,
            k_pe,
            self.rotary_emb.head_size,
            self.rotary_emb.cos_sin_cache,
            self.rotary_emb.is_neox_style,
        )
        attn_out = self.attn(
            q,
            kv_c,
            k_pe,
            output_shape=torch.Size(
                (hidden_states.shape[0], self.num_local_heads * self.v_head_dim)
            ),
        )
        return self.o_proj(attn_out)[0]


class DFlash2K3MLP(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, config, prefix: str) -> None:
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError("MLA DFlash2 requires the silu activation.")
        quant_config = get_draft_quant_config(vllm_config)
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.act_fn = SiluAndMul()
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=True,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(hidden_states)[0]
        return self.down_proj(self.act_fn(gate_up))[0]


class DFlash2K3DecoderLayer(DFlash2DecoderLayer):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        layer_idx: int,
        start_layer_id: int,
        prefix: str,
    ) -> None:
        super().__init__()
        sliding_window, causal = _resolve_layer_attention(config, layer_idx)
        if causal:
            raise ValueError("MLA DFlash2 requires non-causal draft attention.")
        prefix = maybe_prefix(prefix, f"layers.{start_layer_id + layer_idx}")
        self.self_attn = DFlash2K3Attention(
            vllm_config=vllm_config,
            config=config,
            prefix=f"{prefix}.self_attn",
            sliding_window=sliding_window,
        )
        self.mlp = DFlash2K3MLP(
            vllm_config=vllm_config, config=config, prefix=f"{prefix}.mlp"
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self._init_convs(vllm_config, config, prefix)


class DFlash2K3Model(nn.Module):
    """DFlash2 backbone with per-layer latent context caches and a selector."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
            ".q_a_proj": (".fused_qkv_a_proj", 0),
            ".kv_a_proj_with_mqa": (".fused_qkv_a_proj", 1),
        },
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = config = draft_model_config.hf_config
        if not draft_model_config.use_mla:
            raise ValueError("MLA DFlash2 requires VLLM_MLA_DISABLE=0.")
        unsupported = [
            name
            for name in ("mla_use_nope", "mla_use_output_gate", "mla_use_qk_norm")
            if getattr(config, name, False)
        ]
        if config.q_lora_rank is None:
            unsupported.append("q_lora_rank=None")
        if unsupported:
            raise ValueError("MLA DFlash2 does not support " + ", ".join(unsupported))
        if config.draft_vocab_size != vllm_config.model_config.get_vocab_size():
            raise ValueError("MLA DFlash2 requires the target's full vocabulary.")

        self.quant_config = get_draft_quant_config(vllm_config)
        self.use_aux_hidden_state = True
        self.vocab_size = config.vocab_size
        draft_config = config.dflash_config
        self.input_embedding_scale = float(
            draft_config.get("input_embedding_scale", 1.0)
        )
        self.mask_token_id = draft_config.get("mask_token_id")
        self.has_separate_mask_embedding = False
        self.mask_embedding = nn.Parameter(
            torch.zeros(config.hidden_size, dtype=draft_model_config.dtype),
            requires_grad=False,
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.fc = ReplicatedLinear(
            _get_dflash_fc_input_size(vllm_config),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "fc"),
        )
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            DFlash2K3DecoderLayer(
                vllm_config=vllm_config,
                config=config,
                layer_idx=i,
                start_layer_id=start_layer_id,
                prefix=prefix,
            )
            for i in range(config.num_hidden_layers)
        )
        kv_width = config.kv_lora_rank + config.qk_rope_head_dim
        self.context_kv_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [kv_width] * config.num_hidden_layers,
            bias=False,
            return_bias=False,
            quant_config=self.quant_config,
            prefix=maybe_prefix(
                prefix, f"layers.{start_layer_id}.self_attn.fused_qkv_a_proj"
            ),
            disable_tp=True,
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._max_num_context_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        with set_model_tag("dflash2_candidate_selector"):
            self.candidate_selector = CandidateSelector(
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
                rank=int(draft_config["selector_rank"]),
                top_k=int(draft_config["selector_top_k"]),
                params_dtype=draft_model_config.dtype,
                prefix=maybe_prefix(prefix, "candidate_selector"),
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.embed_tokens is not None
        embeds = self.embed_tokens(input_ids)
        if self.has_separate_mask_embedding and self.mask_token_id is not None:
            embeds = torch.where(
                (input_ids == self.mask_token_id).unsqueeze(-1),
                self.mask_embedding.to(embeds.dtype),
                embeds,
            )
        return embeds * self.input_embedding_scale

    def _build_fused_kv_buffers(self) -> None:
        self._context_kv_norm_weights = torch.stack(
            [layer.self_attn.kv_a_layernorm.weight.detach() for layer in self.layers]
        ).contiguous()

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        if not hasattr(self, "_context_kv_norm_weights"):
            self._build_fused_kv_buffers()
        if isinstance(context_slot_mapping, (list, tuple)) and len(
            context_slot_mapping
        ) != len(self.layers):
            raise ValueError("Expected one context slot mapping per draft layer.")

        num_ctx = context_states.shape[0]
        num_layers = len(self.layers)
        kv_rank = self.config.kv_lora_rank
        rope_dim = self.config.qk_rope_head_dim
        all_kv = self.context_kv_proj(self.hidden_norm(context_states))
        all_kv = all_kv.view(num_ctx, num_layers, kv_rank + rope_dim)
        all_kv_c = all_kv[..., :kv_rank].permute(1, 0, 2).contiguous()
        all_kv_c_normed = torch.empty_like(all_kv_c)
        ops.rms_norm(
            all_kv_c_normed,
            all_kv_c,
            self._context_kv_norm_weights,
            self.config.rms_norm_eps,
        )

        all_k_pe = all_kv[..., kv_rank:].permute(1, 0, 2).contiguous()
        all_k_pe_flat = all_k_pe.view(num_layers * num_ctx, 1, rope_dim)
        (repeated_positions,) = current_workspace_manager().get_simultaneous(
            ((num_layers * self._max_num_context_tokens,), torch.int64),
        )
        repeated_positions = repeated_positions[: num_layers * num_ctx]
        repeated_positions.view(num_layers, num_ctx).copy_(context_positions)
        rotary_emb = self.layers[0].self_attn.rotary_emb
        ops.rotary_embedding(
            repeated_positions,
            all_k_pe_flat,
            None,
            rotary_emb.head_size,
            rotary_emb.cos_sin_cache,
            rotary_emb.is_neox_style,
        )

        if context_slot_mapping is None:
            return
        cache_layers = [layer.self_attn.attn for layer in self.layers]
        if (
            not any(is_quantized_kv_cache(attn.kv_cache_dtype) for attn in cache_layers)
            and self._has_uniform_block_layout(cache_layers)
            and (
                isinstance(context_slot_mapping, torch.Tensor)
                or all(s is not None for s in context_slot_mapping)
            )
        ):
            if isinstance(context_slot_mapping, torch.Tensor):
                slots = context_slot_mapping.unsqueeze(0).expand(num_layers, -1)
            else:
                per_layer_slots = [s for s in context_slot_mapping if s is not None]
                if len({s.data_ptr() for s in per_layer_slots}) == 1:
                    slots = per_layer_slots[0].unsqueeze(0).expand(num_layers, -1)
                else:
                    slots = torch.stack(per_layer_slots)
            ref_cache = cache_layers[0].kv_cache
            ops.concat_and_cache_mla_grouped(
                all_kv_c_normed,
                all_k_pe,
                self._get_context_kv_cache_ptrs(cache_layers),
                slots,
                ref_cache.size(1),
                ref_cache.stride(0),
                ref_cache.stride(1),
            )
            return

        for layer_idx, attn in enumerate(cache_layers):
            slots = (
                context_slot_mapping[layer_idx]
                if isinstance(context_slot_mapping, (list, tuple))
                else context_slot_mapping
            )
            if slots is not None:
                attn.impl.do_kv_cache_update(
                    all_kv_c_normed[layer_idx],
                    all_k_pe[layer_idx].unsqueeze(1),
                    attn.kv_cache,
                    slots,
                    attn.kv_cache_dtype,
                    attn._k_scale,
                )

    def _has_uniform_block_layout(self, cache_layers: list[MLAAttention]) -> bool:
        if not hasattr(self, "_layers_share_kv_block_layout"):
            ref_cache = cache_layers[0].kv_cache
            self._layers_share_kv_block_layout = all(
                attn.kv_cache.size(1) == ref_cache.size(1)
                and attn.kv_cache.stride(0) == ref_cache.stride(0)
                and attn.kv_cache.stride(1) == ref_cache.stride(1)
                and attn.kv_cache.dtype == ref_cache.dtype
                for attn in cache_layers
            )
        return self._layers_share_kv_block_layout

    def _get_context_kv_cache_ptrs(
        self, cache_layers: list[MLAAttention]
    ) -> torch.Tensor:
        if not hasattr(self, "_context_cache_ptrs"):
            self._context_cache_ptrs = torch.tensor(
                [attn.kv_cache.data_ptr() for attn in cache_layers],
                dtype=torch.int64,
                device=cache_layers[0].kv_cache.device,
            )
        return self._context_cache_ptrs

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        )
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            self._duplicate_context_kv_weights(weights), mapper=self.hf_to_vllm_mapper
        )

    def _duplicate_context_kv_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Load per-layer KV projections into the fused context projection too."""
        for name, weight in weights:
            yield name, weight
            layer_prefix, marker, param_name = name.partition(
                ".self_attn.kv_a_proj_with_mqa."
            )
            if not marker:
                continue
            layer_idx = layer_prefix.rsplit(".", 1)[-1]
            if not layer_idx.isdecimal() or int(layer_idx) >= len(self.layers):
                continue
            fused_weight = weight.detach()
            fused_weight.shard_id = int(layer_idx)
            yield f"context_kv_proj.{param_name}", fused_weight
