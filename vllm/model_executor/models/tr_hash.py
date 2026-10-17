# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TR-HASH inference with checkpoint-defined token-to-expert routing."""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors

from .llama import LlamaMLP
from .qwen3 import Qwen3Attention
from .utils import AutoWeightsLoader, WeightsMapper, maybe_prefix


class TRHashMLP(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.shared_scale = config.shared_output_scale
        self.routed_scale = config.routed_output_scale
        primary = config.top_k_primary_weight
        if self.top_k == 1:
            weights = [1.0]
        else:
            primary = 1 / self.top_k if primary is None else primary
            weights = [primary] + [(1 - primary) / (self.top_k - 1)] * (self.top_k - 1)
        self.register_buffer(
            "route_weights",
            torch.tensor(weights, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "route_table",
            torch.zeros(self.top_k, config.vocab_size, dtype=torch.int64),
        )
        self.experts = FusedMoEFactory(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.expert_width,
            custom_routing_function=self.route,
            prefix=f"{prefix}.experts",
        )
        self.shared_experts = (
            LlamaMLP(
                config.hidden_size,
                config.shared_intermediate_size,
                "silu",
                prefix=f"{prefix}.shared_experts",
            )
            if config.shared_expert
            else None
        )

    def route(self, hidden_states, gating_output, topk, renormalize):
        ids = gating_output[:, :topk].to(torch.int32)
        weights = self.route_weights.expand(ids.shape[0], -1).contiguous()
        return weights, ids

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor):
        routes = self.route_table[:, input_ids].T.contiguous()
        routed = self.experts(hidden_states, routes) * self.routed_scale
        if self.shared_experts is not None:
            routed = routed + self.shared_experts(hidden_states) * self.shared_scale
        return routed


class TRHashDecoderLayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters={"rope_type": "default", "rope_theta": config.rope_theta},
            max_position=config.max_position_embeddings,
            rms_norm_eps=1e-6,
            cache_config=vllm_config.cache_config,
            prefix=f"{prefix}.self_attn",
        )
        if not config.use_qk_norm:
            self.self_attn.q_norm = nn.Identity()
            self.self_attn.k_norm = nn.Identity()
        self.mlp = TRHashMLP(vllm_config=vllm_config, prefix=f"{prefix}.mlp")

    def forward(self, positions, hidden_states, residual, input_ids):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states, input_ids), residual


@support_torch_compile
class TRHashModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )
        self.layers = nn.ModuleList(
            TRHashDecoderLayer(vllm_config=vllm_config, prefix=f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        hidden_states = self.embed_input_ids(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, residual, input_ids
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class TRHashForCausalLM(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".shared_experts.gate_proj": (".shared_experts.gate_up_proj", 0),
            ".shared_experts.up_proj": (".shared_experts.gate_up_proj", 1),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("TR-HASH does not support pipeline parallelism.")
        if vllm_config.parallel_config.enable_expert_parallel:
            raise ValueError("TR-HASH does not support expert parallelism.")
        if vllm_config.quant_config is not None:
            raise ValueError("TR-HASH currently requires an unquantized checkpoint.")
        if vllm_config.model_config.enable_prompt_embeds:
            raise ValueError(
                "TR-HASH routing requires token IDs, not prompt embeddings."
            )
        if config.routing_strategy != "token_id_multi_hash":
            raise ValueError(f"Unsupported TR-HASH routing: {config.routing_strategy}")
        if not config.tie_word_embeddings:
            raise ValueError("TR-HASH requires tied word embeddings.")
        self.model = TRHashModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor):
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor):
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def converted():
            for name, weight in weights:
                if name.endswith(
                    ("rotary_emb.inv_freq", "fused_route_codes", "fused_expert_pairs")
                ):
                    continue
                name = "model." + name.replace(".mlp.engine.", ".mlp.")
                for source, target in (
                    ("expert_gate", "gate_proj"),
                    ("expert_up", "up_proj"),
                    ("expert_down", "down_proj"),
                ):
                    if name.endswith("." + source):
                        base = name.removesuffix("." + source)
                        for i, expert in enumerate(weight):
                            yield f"{base}.experts.{i}.{target}.weight", expert.T
                        break
                else:
                    name = name.replace(".shared_gate.", ".shared_experts.gate_proj.")
                    name = name.replace(".shared_up.", ".shared_experts.up_proj.")
                    name = name.replace(".shared_down.", ".shared_experts.down_proj.")
                    yield name, weight

        return AutoWeightsLoader(self).load_weights(
            converted(), mapper=self.hf_to_vllm_mapper
        )
