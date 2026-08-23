# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Orthrus model compatible with HuggingFace weights."""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import set_default_rope_theta

from .interfaces import SupportsEagle, SupportsEagle3, SupportsLoRA, SupportsPP
from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen2 import Qwen2Model
from .qwen3 import Qwen3Attention
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    maybe_prefix,
)


class OrthrusAttention(Qwen3Attention):
    """Qwen3 attention with Orthrus diffusion-path projection weights.

    The standard ``forward`` method intentionally uses the autoregressive Qwen3
    path. Diffusion-mode generation needs additional scheduler and attention
    metadata support before it can be exposed safely in vLLM.
    """

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
        **kwargs,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope_parameters=rope_parameters,
            max_position=max_position,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            qkv_bias=qkv_bias,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            **kwargs,
        )
        self.qkv_proj_diff = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj_diff",
        )
        self.o_proj_diff = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj_diff",
        )
        self.q_norm_diff = type(self.q_norm)(self.head_dim, eps=rms_norm_eps)
        self.k_norm_diff = type(self.k_norm)(self.head_dim, eps=rms_norm_eps)

    def forward_diffusion(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        cached_key: torch.Tensor,
        cached_value: torch.Tensor,
    ) -> torch.Tensor:
        """Diffusion-mode forward for a block of mask-token queries.

        Attends over this layer's AR-path cached key/value states (already
        computed by a prior autoregressive forward) plus this block's own
        diffusion keys/values, with no causal restriction. This matches the
        reference implementation's inference-time behavior: the
        block-diagonal masking in ``generate_dual_pass_mask`` is only
        exercised during training on multiple packed diffusion blocks -- a
        single served diffusion block simply attends over the fixed AR
        prefix plus itself (see ``OrthrusModel.forward`` in the reference
        ``modeling_orthrus.py``, where ``causal_mask``/``flex_block_mask``
        are only built when both ``self.training`` and ``is_diffusion_pass``
        are true).

        Args:
            positions: ``[block_len]`` absolute sequence positions.
            hidden_states: ``[block_len, hidden_size]``.
            cached_key: ``[ar_seq_len, num_kv_heads, head_dim]`` AR-path
                cached keys for this layer, already RoPE-applied.
            cached_value: ``[ar_seq_len, num_kv_heads, head_dim]`` AR-path
                cached values for this layer.

        Returns:
            ``[block_len, hidden_size]`` attention output for the block.
        """
        block_len = hidden_states.shape[0]
        qkv, _ = self.qkv_proj_diff(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm_diff(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm_diff(k_by_head)
        k = k_by_head.view(k.shape)

        q, k = self.rotary_emb(positions, q, k)

        q = q.view(block_len, self.num_heads, self.head_dim)
        k = k.view(block_len, self.num_kv_heads, self.head_dim)
        v = v.view(block_len, self.num_kv_heads, self.head_dim)

        full_key = torch.cat([cached_key, k], dim=0)
        full_value = torch.cat([cached_value, v], dim=0)

        num_groups = self.num_heads // self.num_kv_heads
        if num_groups > 1:
            full_key = full_key.repeat_interleave(num_groups, dim=1)
            full_value = full_value.repeat_interleave(num_groups, dim=1)

        # [1, heads, seq, head_dim] for scaled_dot_product_attention.
        q_t = q.transpose(0, 1).unsqueeze(0)
        k_t = full_key.transpose(0, 1).unsqueeze(0)
        v_t = full_value.transpose(0, 1).unsqueeze(0)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q_t, k_t, v_t, attn_mask=None, is_causal=False, scale=self.scaling
        ).squeeze(0)
        attn_output = attn_output.transpose(0, 1).reshape(block_len, -1)

        output, _ = self.o_proj_diff(attn_output)
        return output


class OrthrusDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)
        self.self_attn = OrthrusAttention(
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
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = type(self.self_attn.q_norm)(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = type(self.self_attn.q_norm)(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def forward_diffusion(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        cached_key: torch.Tensor,
        cached_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn.forward_diffusion(
            positions=positions,
            hidden_states=hidden_states,
            cached_key=cached_key,
            cached_value=cached_value,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class OrthrusModel(Qwen2Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=OrthrusDecoderLayer,
        )

    def forward_diffusion(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        per_layer_cached_kv: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Runs a diffusion-mode forward for a block of mask tokens.

        This proposes ``len(input_ids)`` future tokens in a single forward
        pass by attending each layer's diffusion projections over that
        layer's already-computed AR cache, instead of the usual sequential
        one-token-per-step autoregressive path. Candidates produced from
        this pass must still be verified against a normal AR forward before
        being accepted (see the reference implementation's ``generate``
        method for the accept/reject loop that makes this lossless).

        Args:
            input_ids: ``[block_len]`` token ids for the diffusion block --
                the first position holds the last accepted AR token and the
                rest are ``mask_token_id``, matching the reference
                implementation's block construction.
            positions: ``[block_len]`` absolute sequence positions.
            per_layer_cached_kv: one ``(key, value)`` pair per decoder layer,
                each shaped ``[ar_seq_len, num_kv_heads, head_dim]``.

        Returns:
            ``[block_len, hidden_size]`` final hidden states for the block.
        """
        hidden_states = self.embed_input_ids(input_ids)
        residual = None
        for layer, (cached_key, cached_value) in zip(self.layers, per_layer_cached_kv):
            hidden_states, residual = layer.forward_diffusion(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                cached_key=cached_key,
                cached_value=cached_value,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("qkv_proj_diff", "q_proj_diff", "q"),
            ("qkv_proj_diff", "k_proj_diff", "k"),
            ("qkv_proj_diff", "v_proj_diff", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name.endswith("scale"):
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class OrthrusForCausalLM(
    nn.Module, SupportsLoRA, SupportsPP, SupportsEagle, SupportsEagle3
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "qkv_proj_diff": [
            "q_proj_diff",
            "k_proj_diff",
            "v_proj_diff",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vllm_config = vllm_config
        self.quant_config = quant_config
        self.model = OrthrusModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
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
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def propose_diffusion_block(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        per_layer_cached_kv: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Proposes a block of future tokens via Orthrus' diffusion path.

        Returns ``[block_len, vocab_size]`` logits. Candidates sampled from
        these logits are not yet verified against the AR path -- callers
        must run an AR forward over the proposed block and accept/reject
        per-position (matching the reference implementation's exact
        intra-model consistency check) before treating them as generated
        output.
        """
        hidden_states = self.model.forward_diffusion(
            input_ids, positions, per_layer_cached_kv
        )
        return self.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)


OrthrusLM = OrthrusForCausalLM
