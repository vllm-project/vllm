# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only ZAYA1 model."""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.zaya.configuration_zaya import ZayaConfig

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.utils import maybe_prefix, sequence_parallel_chunk
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.short_conv_attn import ShortConvAttentionMetadata
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


def _cca_state_shapes(config: ZayaConfig) -> tuple[tuple[int, int], tuple[int]]:
    """`(raw q/k history, previous delayed value)` of one CCA layer."""
    heads = config.num_attention_heads + config.num_key_value_heads
    history = config.cca_time0 + config.cca_time1 - 2
    delayed_dim = config.num_key_value_heads * config.head_dim // 2
    return (heads * config.head_dim, history), (delayed_dim,)


def _cca_state_dtypes(vllm_config: VllmConfig) -> tuple[torch.dtype, torch.dtype]:
    (dtype,) = MambaStateDtypeCalculator.short_conv_state_dtype(
        vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
    )
    return dtype, dtype


class ZayaCCA(MambaBase, nn.Module):
    """Compressed Convolutional Attention q/k/v projection.

    q/k mix over a causal conv of the raw projections, and half of v comes from the
    previous token, so each request keeps that history in Mamba state.
    """

    def __init__(
        self,
        config: ZayaConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ):
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.kv_groups = self.num_q_heads // self.num_kv_heads
        self.history_len = config.cca_time0 + config.cca_time1 - 2
        self.q_dim = self.num_q_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        def linear(name: str, out_dim: int) -> ReplicatedLinear:
            return ReplicatedLinear(
                config.hidden_size,
                out_dim,
                bias=config.attention_bias,
                quant_config=quant_config,
                return_bias=False,
                prefix=f"{prefix}.{name}",
            )

        self.q_proj = linear("q_proj", self.q_dim)
        self.k_proj = linear("k_proj", kv_dim)
        self.v_proj_current = linear("v_proj_current", kv_dim // 2)
        self.v_proj_delayed = linear("v_proj_delayed", kv_dim // 2)
        channels = self.q_dim + kv_dim
        self.conv_qk_depthwise = nn.Conv1d(
            channels, channels, config.cca_time0, groups=channels
        )
        self.conv_qk_grouped = nn.Conv1d(
            channels,
            channels,
            config.cca_time1,
            groups=self.num_q_heads + self.num_kv_heads,
        )

        self.vllm_config = get_current_vllm_config()
        self.vllm_config.compilation_config.static_forward_context[prefix] = self
        self.kv_cache = (torch.tensor([]), torch.tensor([]))

    def forward(self, hidden_states: torch.Tensor, output: torch.Tensor) -> None:
        torch.ops.vllm.zaya_cca(hidden_states, output, self.prefix)

    def _conv(self, qk: torch.Tensor) -> torch.Tensor:
        return self.conv_qk_grouped(self.conv_qk_depthwise(qk))

    def project(self, hidden_states: torch.Tensor, output: torch.Tensor) -> None:
        num_tokens = hidden_states.shape[0]
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        qk = torch.cat([queries, keys], dim=-1)
        delayed = self.v_proj_delayed(hidden_states)
        delayed_init = self.v_proj_delayed(
            hidden_states.new_zeros(hidden_states.shape[-1])
        )

        attn_metadata = get_forward_context().attn_metadata
        metadata = None
        if attn_metadata is not None:
            assert isinstance(attn_metadata, dict)
            metadata = attn_metadata.get(self.prefix)
        if metadata is None:  # Profile run
            conv_out = self._conv(F.pad(qk.T[None], (self.history_len, 0)))[0].T
            delayed_prev = delayed
        else:
            assert isinstance(metadata, ShortConvAttentionMetadata)
            conv_out, delayed_prev = self._mix_with_state(
                metadata, qk, delayed, delayed_init
            )

        heads = (num_tokens, -1, self.head_dim)
        query_residual = 0.5 * (
            queries.view(heads)
            + keys.view(heads).repeat_interleave(self.kv_groups, dim=1)
        )
        key_residual = query_residual.view(
            num_tokens, -1, self.kv_groups, self.head_dim
        ).mean(dim=2)
        query = conv_out[:, : self.q_dim].view(heads) + query_residual
        key = conv_out[:, self.q_dim :].view(heads) + key_residual
        value = torch.cat([self.v_proj_current(hidden_states), delayed_prev], dim=-1)
        output[:] = torch.cat([query, key, value.view(heads)], dim=1).flatten(1)

    def _mix_with_state(
        self,
        metadata: ShortConvAttentionMetadata,
        qk: torch.Tensor,
        delayed: torch.Tensor,
        delayed_init: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Causal conv of `qk` and the previous token's `delayed` value, reading
        and updating each request's state without syncing with the host.

        Decodes are one-token requests that always continue from their state.
        """
        device = qk.device
        num_decodes = metadata.num_decode_tokens
        num_tokens = num_decodes + metadata.num_prefill_tokens
        slots, starts, lens, has_state = [], [], [], []
        if num_decodes:
            assert metadata.state_indices_tensor_d is not None
            slots.append(metadata.state_indices_tensor_d[:, 0])
            starts.append(torch.arange(num_decodes, device=device))
            lens.append(torch.ones(num_decodes, dtype=torch.long, device=device))
            has_state.append(torch.ones(num_decodes, dtype=torch.bool, device=device))
        if metadata.num_prefills:
            query_start_loc = metadata.query_start_loc_p
            assert query_start_loc is not None
            assert metadata.state_indices_tensor_p is not None
            assert metadata.has_initial_states_p is not None
            slots.append(metadata.state_indices_tensor_p)
            starts.append(num_decodes + query_start_loc[:-1])
            lens.append(query_start_loc.diff())
            has_state.append(metadata.has_initial_states_p)
        # Padding rows of a captured graph use the unused null block
        slot = torch.cat(slots).clamp_min(NULL_BLOCK_ID)
        start, length = torch.cat(starts), torch.cat(lens)
        has_history = torch.cat(has_state)

        conv_state, delayed_state = self.kv_cache
        n = self.history_len
        history = conv_state[slot].to(qk.dtype) * has_history[:, None, None]
        first_delayed = torch.where(
            has_history[:, None], delayed_state[slot].to(delayed.dtype), delayed_init
        )

        # Tokens at least `n` into their request only see that request
        x = qk[:num_tokens]
        conv_out = qk.new_empty(qk.shape[0] + 1, qk.shape[1])
        conv_out[:num_tokens] = self._conv(F.pad(x.T[None], (n, 0)))[0].T
        # The first `n` tokens of each request also see its saved history; the
        # spare last row of `conv_out` absorbs those of requests shorter than `n`
        offset = torch.arange(n, device=device)
        first_tokens = start[:, None] + offset
        heads = self._conv(
            torch.cat([history, x[first_tokens.clamp(max=num_tokens - 1)].mT], -1)
        )
        valid = offset < length[:, None]
        conv_out[torch.where(valid, first_tokens, qk.shape[0])] = heads.mT

        delayed_prev = delayed.new_empty(delayed.shape)
        delayed_prev[1:num_tokens] = delayed[: num_tokens - 1]
        delayed_prev[start] = first_delayed

        # The new history is the last `n` inputs of the saved history and tokens
        last = length[:, None] - n + offset
        new_history = torch.where(
            (last >= 0)[..., None],
            x[(start[:, None] + last).clamp(min=0)],
            history.mT.gather(
                1, (last + n).clamp(0, n - 1)[..., None].expand(-1, -1, x.shape[1])
            ),
        )
        conv_state[slot] = new_history.mT.to(conv_state.dtype)
        delayed_state[slot] = delayed[start + length - 1].to(delayed_state.dtype)
        return conv_out[:-1], delayed_prev

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return _cca_state_shapes(self.config)

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return _cca_state_dtypes(self.vllm_config)

    @property
    def is_kv_cache_tp_replicated(self) -> bool:
        return True

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        # CCA needs only the per-request slots, which is all ShortConv's metadata has
        return MambaAttentionBackendEnum.SHORT_CONV


@eager_break_during_capture
def _zaya_cca(
    hidden_states: torch.Tensor, output: torch.Tensor, layer_name: str
) -> None:
    get_forward_context().no_compile_layers[layer_name].project(hidden_states, output)


direct_register_custom_op(
    op_name="zaya_cca", op_func=_zaya_cca, mutates_args=["output"]
)


class ZayaQKNorm(nn.Module):
    def __init__(self, config: ZayaConfig):
        super().__init__()
        self.scale = config.head_dim**0.5
        self.temp = nn.Parameter(torch.zeros(config.num_key_value_heads))

    def forward(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = F.normalize(query.float(), dim=-1) * self.scale
        key = F.normalize(key.float(), dim=-1) * self.scale * self.temp[None, :, None]
        return query, key


class ZayaAttention(nn.Module):
    def __init__(
        self,
        config: ZayaConfig,
        layer_idx: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ):
        super().__init__()
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.qkv_proj = ZayaCCA(config, quant_config, f"{prefix}.qkv_proj")
        self.qk_norm = ZayaQKNorm(config)
        self.o_proj = ReplicatedLinear(
            self.num_q_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.o_proj",
        )
        layer_type = config.layer_types[layer_idx]
        self.attn = Attention(
            self.num_q_heads,
            self.head_dim,
            self.head_dim**-0.5,
            self.num_kv_heads,
            per_layer_sliding_window=(
                config.sliding_window if layer_type == "hybrid_sliding" else None
            ),
            cache_config=get_current_vllm_config().cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            is_neox_style=True,
            rope_parameters=config.rope_parameters[layer_type],
        )

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        total_heads = self.num_q_heads + 2 * self.num_kv_heads
        output = hidden_states.new_empty(num_tokens, total_heads * self.head_dim)
        self.qkv_proj(hidden_states, output)
        q, k, v = output.view(num_tokens, total_heads, self.head_dim).split(
            [self.num_q_heads, self.num_kv_heads, self.num_kv_heads], dim=1
        )
        q, k = self.qk_norm(q, k)
        q, k = self.rotary_emb(positions, q.flatten(1), k.flatten(1))
        attn_output = self.attn(
            q.to(hidden_states.dtype), k.to(hidden_states.dtype), v.flatten(1)
        )
        return self.o_proj(attn_output)


class ZayaRouterMLP(nn.Module):
    def __init__(self, config: ZayaConfig, num_classes: int, prefix: str):
        super().__init__()
        size = config.router_hidden_size
        self.norm = RMSNorm(size, eps=config.rms_norm_eps)
        self.fc1 = ReplicatedLinear(
            size, size, return_bias=False, prefix=f"{prefix}.fc1"
        )
        self.fc2 = ReplicatedLinear(
            size, size, return_bias=False, prefix=f"{prefix}.fc2"
        )
        self.out_proj = ReplicatedLinear(
            size,
            num_classes,
            bias=False,
            return_bias=False,
            prefix=f"{prefix}.out_proj",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.gelu(self.fc1(self.norm(hidden_states)))
        return self.out_proj(F.gelu(self.fc2(hidden_states)))


class ZayaRouter(nn.Module):
    """Picks an expert per token; an extra "skip" class bypasses the experts."""

    def __init__(self, config: ZayaConfig, layer_idx: int, prefix: str):
        super().__init__()
        self.skip_id = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.down_proj = ReplicatedLinear(
            config.hidden_size,
            config.router_hidden_size,
            return_bias=False,
            prefix=f"{prefix}.down_proj",
        )
        self.use_eda = layer_idx != 0
        if self.use_eda:
            self.router_states_scale = nn.Parameter(
                torch.ones(config.router_hidden_size)
            )
        self.router_mlp = ZayaRouterMLP(
            config, config.num_experts + 1, f"{prefix}.router_mlp"
        )
        self.balancing_biases = nn.Buffer(
            torch.zeros(config.num_experts + 1, dtype=torch.float32)
        )

    def forward(
        self, hidden_states: torch.Tensor, router_states: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        router_hidden = self.down_proj(hidden_states)
        if self.use_eda and router_states is not None:
            router_hidden = router_hidden + router_states * self.router_states_scale
        probs = self.router_mlp(router_hidden).softmax(dim=-1).float()
        _, indices = (probs + self.balancing_biases).topk(self.top_k, dim=-1)
        weights = probs.gather(1, indices)
        skip = indices == self.skip_id
        routing = torch.cat(
            [weights.masked_fill(skip, 0), indices.masked_fill(skip, 0).float()], dim=-1
        )
        return routing, router_hidden


def _precomputed_routing(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return gating_output[:, :topk].contiguous(), gating_output[:, topk:].to(torch.int32)


class ZayaSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: ZayaConfig,
        layer_idx: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ):
        super().__init__()
        self.gate = ZayaRouter(config, layer_idx, f"{prefix}.gate")
        parallel_config = get_current_vllm_config().parallel_config
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe
        self.experts = FusedMoEFactory(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=False,
            custom_routing_function=_precomputed_routing,
            router_logits_dtype=torch.float32,
            quant_config=quant_config,
            is_sequence_parallel=self.is_sequence_parallel,
            prefix=f"{prefix}.experts",
        )

    def forward(
        self, hidden_states: torch.Tensor, router_states: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_sequence_parallel:
            routing, router_states = self.gate(hidden_states, router_states)
            return self.experts(hidden_states, routing), router_states

        # Under SP, router_states stays chunked from one MoE block to the next
        num_tokens = hidden_states.shape[0]
        hidden_states = sequence_parallel_chunk(hidden_states)
        routing, router_states = self.gate(hidden_states, router_states)
        hidden_states = self.experts(hidden_states, routing)
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        return hidden_states[:num_tokens], router_states


class ZayaResidualScaling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_states_scale = nn.Parameter(torch.ones(hidden_size))
        self.hidden_states_bias = nn.Parameter(torch.zeros(hidden_size))
        self.residual_scale = nn.Parameter(torch.ones(hidden_size))
        self.residual_bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = (hidden_states + self.hidden_states_bias) * (
            self.hidden_states_scale
        )
        return hidden_states + (residual + self.residual_bias) * self.residual_scale


class ZayaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: ZayaConfig,
        layer_idx: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ):
        super().__init__()
        size = config.hidden_size
        self.input_layernorm = RMSNorm(size, eps=config.rms_norm_eps)
        self.self_attn = ZayaAttention(
            config, layer_idx, quant_config, f"{prefix}.self_attn"
        )
        self.post_attention_residual_scale = ZayaResidualScaling(size)
        self.post_attention_layernorm = RMSNorm(size, eps=config.rms_norm_eps)
        self.mlp = ZayaSparseMoeBlock(config, layer_idx, quant_config, f"{prefix}.mlp")
        self.post_mlp_residual_scale = ZayaResidualScaling(size)

    def forward(
        self,
        residual: torch.Tensor,
        positions: torch.Tensor,
        router_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The residual stream stays in fp32; each sublayer runs in the model dtype
        dtype = self.input_layernorm.weight.dtype
        hidden_states = self.self_attn(
            self.input_layernorm(residual.to(dtype)), positions
        )
        residual = self.post_attention_residual_scale(hidden_states, residual)
        hidden_states, router_states = self.mlp(
            self.post_attention_layernorm(residual.to(dtype)), router_states
        )
        return self.post_mlp_residual_scale(hidden_states, residual), router_states


@support_torch_compile
class ZayaModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.input_hidden_states_scale = nn.Parameter(torch.ones(config.hidden_size))
        self.input_hidden_states_bias = nn.Parameter(torch.zeros(config.hidden_size))
        self.layers = nn.ModuleList(
            ZayaDecoderLayer(
                config, i, vllm_config.quant_config, f"{prefix}.layers.{i}"
            )
            for i in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = (inputs_embeds + self.input_hidden_states_bias) * (
            self.input_hidden_states_scale
        )
        hidden_states = hidden_states.to(torch.float32)
        router_states = None
        for layer in self.layers:
            hidden_states, router_states = layer(
                hidden_states, positions, router_states
            )
        return self.norm(hidden_states.to(self.norm.weight.dtype))


class ZayaForCausalLM(nn.Module, HasInnerState, IsHybrid):
    # load_weights takes the stacked experts.{gate_up,down}_proj names itself;
    # the load plan must not split them into per-expert names.
    loads_stacked_experts = True

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        return _cca_state_shapes(vllm_config.model_config.hf_config)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return _cca_state_dtypes(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        if vllm_config.num_speculative_tokens:
            raise NotImplementedError("ZAYA1 does not support speculative decoding.")
        self.config = config
        self.model = ZayaModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters(remove_duplicate=False))
        params.update(self.named_buffers())
        loaded: set[str] = set()

        def load_expert(name: str, weight: torch.Tensor, shard_id: str) -> None:
            param = params[name]
            for expert_id, expert_weight in enumerate(weight):
                param.weight_loader(
                    param, expert_weight, name, shard_id=shard_id, expert_id=expert_id
                )
            loaded.add(name)

        for name, weight in weights:
            # Experts are stacked: `experts.{gate_up,down}_proj[.<quant param>]`
            prefix, is_expert, suffix = name.partition(".mlp.experts.")
            proj, _, param_name = suffix.partition(".")
            experts = f"{prefix}.mlp.experts.routed_experts"
            param_name = param_name or "weight"
            if is_expert and proj == "gate_up_proj":
                gate, up = weight.chunk(2, dim=1)
                load_expert(f"{experts}.w13_{param_name}", gate, "w1")
                load_expert(f"{experts}.w13_{param_name}", up, "w3")
            elif is_expert and proj == "down_proj":
                load_expert(f"{experts}.w2_{param_name}", weight, "w2")
            elif name != "lm_head.weight" or not self.config.tie_word_embeddings:
                param = params[name]
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, weight)
                loaded.add(name)
        return loaded
