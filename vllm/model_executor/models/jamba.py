# coding=utf-8
"""Inference-only Jamba model."""
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import JambaConfig

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.layer import Attention
from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn, causal_conv1d_update)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_scan_fn, selective_state_update)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    composed_weight_loader, default_weight_loader, sharded_weight_loader)
from vllm.model_executor.models.mamba_cache import MambaCacheManager
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors
from vllm.worker.model_runner import (_BATCH_SIZES_TO_CAPTURE,
                                      _get_graph_batch_size)

from .interfaces import HasInnerState, SupportsLoRA

KVCache = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class MambaCacheParams:
    is_prompt: bool = False
    conv_state: torch.Tensor = torch.Tensor()
    ssm_state: torch.Tensor = torch.Tensor()


# Adapted from transformers.models.mamba.modeling_mamba.MambaMixer
class JambaMambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute
    the `contextualized_states`. A, D are input independent
    (see Mamba paper [1] Section 3.5.2 "Interpretation of A"
    for why A isn't selective) ∆, B, C are input-dependent
    (this is a key difference between Mamba and the linear time
    invariant S4, and is why Mamba is called
    **selective** state spaces)
    """

    def __init__(self, config: JambaConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = config.mamba_expand * config.hidden_size
        self.time_step_rank = config.mamba_dt_rank
        self.use_conv_bias = config.mamba_conv_bias
        self.use_bias = config.mamba_proj_bias
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.intermediate_size,
            bias=self.use_conv_bias,
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        self.in_proj = MergedColumnParallelLinear(self.hidden_size,
                                                  [self.intermediate_size] * 2,
                                                  bias=self.use_bias)
        # selective projection used to make dt, B and C input dependent
        self.x_proj = RowParallelLinear(
            self.intermediate_size,
            self.time_step_rank + self.ssm_state_size * 2,
            bias=False,
        )
        # time step projection (discretization) -
        # In the forward we need to apply dt_proj without the bias,
        # as the bias is added in the selective scan kernel.
        self.dt_proj = ColumnParallelLinear(self.time_step_rank,
                                            self.intermediate_size,
                                            bias=True,
                                            skip_bias_add=True)

        tp_size = get_tensor_model_parallel_world_size()
        self.A = nn.Parameter(
            torch.empty(
                self.intermediate_size // tp_size,
                self.ssm_state_size,
                dtype=torch.float32,
            ))
        self.D = nn.Parameter(torch.ones(self.intermediate_size // tp_size))

        set_weight_attrs(self.D, {"weight_loader": sharded_weight_loader(0)})
        a_weight_loader = composed_weight_loader(
            sharded_weight_loader(0), lambda x: -torch.exp(x.float()))
        set_weight_attrs(self.A, {"weight_loader": a_weight_loader})

        self.out_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=self.use_bias,
            input_is_parallel=True,
        )
        self.activation = config.hidden_act

        self.dt_layernorm = RMSNorm(self.time_step_rank,
                                    eps=config.rms_norm_eps)
        self.b_layernorm = RMSNorm(self.ssm_state_size,
                                   eps=config.rms_norm_eps)
        self.c_layernorm = RMSNorm(self.ssm_state_size,
                                   eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor,
                attn_metadata: AttentionMetadata, conv_state: torch.Tensor,
                ssm_state: torch.Tensor):

        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states)[0].transpose(-2, -1)
        hidden_states, gate = projected_states.chunk(2, dim=-2)

        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0),
                                               self.conv1d.weight.size(2))

        if attn_metadata.query_start_loc is not None \
            and attn_metadata.context_lens_tensor is not None:
            # |---------- N-1 iteration --------|
            # |---------------- N iteration ---------------------|
            # |- tokenA -|......................|-- newTokens ---|
            # |---------- context_len ----------|
            # |-------------------- seq_len ---------------------|
            #                                   |-- query_len ---|
            hidden_states = causal_conv1d_fn(
                hidden_states,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=attn_metadata.context_lens_tensor > 0,
                query_start_loc=attn_metadata.query_start_loc)
        else:
            hidden_states = causal_conv1d_update(
                hidden_states.transpose(0, 1),
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
            )
            hidden_states = hidden_states.transpose(0, 1)

        # 3. State Space Model sequence transformation
        # 3.a. input varying initialization of time_step, B and C
        ssm_parameters = self.x_proj(hidden_states.transpose(-2, -1))[0]

        time_step, B, C = torch.split(
            ssm_parameters,
            [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
            dim=-1,
        )
        time_step = self.dt_layernorm(time_step.contiguous())
        B = self.b_layernorm(B.contiguous())
        C = self.c_layernorm(C.contiguous())

        discrete_time_step = self.dt_proj(time_step)[0].transpose(-2, -1)
        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        time_proj_bias = (self.dt_proj.bias.float() if hasattr(
            self.dt_proj, "bias") else None)

        if attn_metadata.query_start_loc is not None \
            and attn_metadata.context_lens_tensor is not None:
            scan_outputs = selective_scan_fn(
                hidden_states,
                ssm_state,
                discrete_time_step,
                self.A,
                B.transpose(-2, -1),
                C.transpose(-2, -1),
                self.D.float(),
                gate,
                time_proj_bias,
                delta_softplus=True,
                has_initial_state=attn_metadata.context_lens_tensor > 0,
                query_start_loc=attn_metadata.query_start_loc)
        else:
            scan_outputs = selective_state_update(
                ssm_state,
                hidden_states.transpose(0, 1),
                discrete_time_step.transpose(0, 1),
                self.A,
                B,
                C,
                self.D,
                gate.transpose(0, 1),
                time_proj_bias,
                dt_softplus=True,
            )
            scan_outputs = scan_outputs.transpose(0, 1)

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_outputs.transpose(-2,
                                                                     -1))[0]
        return contextualized_states


class JambaMoE(nn.Module):

    def __init__(self,
                 config: JambaConfig,
                 num_experts: Optional[int] = None,
                 top_k: Optional[int] = None,
                 params_dtype: Optional[torch.dtype] = None,
                 tp_size: Optional[int] = None,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.num_total_experts = num_experts or config.num_experts
        self.top_k = top_k or config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        if self.num_total_experts > 1:
            self.router = ReplicatedLinear(self.hidden_size,
                                           self.num_total_experts,
                                           bias=False,
                                           quant_config=None,
                                           params_dtype=params_dtype)

        self.experts = FusedMoE(self.num_total_experts,
                                self.top_k,
                                self.hidden_size,
                                self.intermediate_size,
                                tp_size=tp_size,
                                params_dtype=params_dtype,
                                reduce_results=True,
                                renormalize=False,
                                use_grouped_topk=False,
                                quant_config=quant_config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # router_logits: (batch * sequence_length, n_experts)
        if self.num_total_experts > 1:
            router_logits, _ = self.router(hidden_states)
        else:
            router_logits = torch.ones((hidden_states.shape[0], 1),
                                       device=hidden_states.device,
                                       dtype=hidden_states.dtype)
        hidden_states = self.experts(hidden_states, router_logits)
        return hidden_states.view(orig_shape)


class JambaMLP(JambaMoE):

    def __init__(self,
                 config: JambaConfig,
                 params_dtype: Optional[torch.dtype] = None,
                 tp_size: Optional[int] = None,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__(config,
                         num_experts=1,
                         top_k=1,
                         params_dtype=params_dtype,
                         tp_size=tp_size,
                         quant_config=quant_config)


class JambaMambaDecoderLayer(nn.Module):

    def __init__(self,
                 config: JambaConfig,
                 layer_idx: int,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.mamba = JambaMambaMixer(config, layer_idx)

        num_experts = config.layers_num_experts[layer_idx]
        ffn_layer_class = JambaMoE if num_experts > 1 else JambaMLP
        self.feed_forward = ffn_layer_class(config, quant_config=quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size,
                                        eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.mamba(hidden_states, attn_metadata, conv_state,
                                   ssm_state)
        # Fully Connected
        hidden_states, residual = self.pre_ff_layernorm(
            hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual


class JambaAttentionDecoderLayer(nn.Module):

    def __init__(
        self,
        config: JambaConfig,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim,
                                        config.hidden_size,
                                        bias=False,
                                        quant_config=quant_config)

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
        )

        num_experts = config.layers_num_experts[layer_idx]
        ffn_layer_class = JambaMoE if num_experts > 1 else JambaMLP
        self.feed_forward = ffn_layer_class(config, quant_config=quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size,
                                        eps=config.rms_norm_eps)

    def self_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.self_attention(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        # Fully Connected
        hidden_states, residual = self.pre_ff_layernorm(
            hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "attention": JambaAttentionDecoderLayer,
    "mamba": JambaMambaDecoderLayer
}


class JambaModel(nn.Module):

    def __init__(
        self,
        config: JambaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = ((lora_config.lora_extra_vocab_size *
                       (lora_config.max_loras or 1)) if lora_config else 0)
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        decoder_layers = []
        for i in range(config.num_hidden_layers):
            layer_class = ALL_DECODER_LAYER_TYPES[config.layers_block_type[i]]
            decoder_layers.append(
                layer_class(config,
                            layer_idx=i,
                            cache_config=cache_config,
                            quant_config=quant_config))
        self.layers = nn.ModuleList(decoder_layers)
        self.final_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        for i in range(len(self.layers)):
            layer = self.layers[i]
            kv_cache = None
            current_ssm_state = None
            current_conv_state = None
            if isinstance(layer, JambaAttentionDecoderLayer):
                kv_cache = kv_caches[(i - self.config.attn_layer_offset) //
                                     self.config.attn_layer_period]
            if isinstance(layer, JambaMambaDecoderLayer):
                current_state_layer = i - (1 +
                                           (i - self.config.attn_layer_offset)
                                           // self.config.attn_layer_period)
                current_ssm_state = ssm_state[current_state_layer]
                current_conv_state = conv_state[current_state_layer]

            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                residual=residual,
                conv_state=current_conv_state,
                ssm_state=current_ssm_state,
            )
        hidden_states, _ = self.final_layernorm(hidden_states, residual)
        return hidden_states


class JambaForCausalLM(nn.Module, HasInnerState, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "embed_tokens",
        "lm_head",
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(
        self,
        config: JambaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        scheduler_config: Optional[SchedulerConfig] = None,
    ) -> None:
        assert not cache_config.enable_prefix_caching, \
            "Jamba currently does not support prefix caching"

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = JambaModel(config,
                                cache_config=cache_config,
                                quant_config=quant_config,
                                lora_config=lora_config)
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
        )
        # Used to track and store by the Mamba cache between steps.
        self.mamba_cache: Optional[MambaCacheManager] = None

        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
        self.sampler = Sampler()

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List[KVCache],
                attn_metadata: AttentionMetadata,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                **kwargs):
        if self.mamba_cache is None:
            max_batch_size = (_get_graph_batch_size(
                self.scheduler_config.max_num_seqs) if self.scheduler_config
                              else max(_BATCH_SIZES_TO_CAPTURE) + 2)

            layers_type = self.config.layers_block_type
            num_mamba_layers = sum(
                [layer_type == "mamba" for layer_type in layers_type])

            self.mamba_cache = MambaCacheManager(
                self.lm_head.weight.dtype, num_mamba_layers, max_batch_size,
                *self._get_mamba_cache_shape())

        mamba_cache_tensors = self.mamba_cache.current_run_tensors(
            input_ids, attn_metadata, **kwargs)

        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, mamba_cache_tensors[0],
                                   mamba_cache_tensors[1])
        return hidden_states

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.mamba_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def _get_mamba_cache_shape(
            self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        world_size = get_tensor_model_parallel_world_size()
        hidden_size = self.config.hidden_size
        conv_state_shape = (
            self.config.mamba_expand * hidden_size // world_size,
            self.config.mamba_d_conv - 1,
        )
        temporal_state_shape = (
            self.config.mamba_expand * hidden_size // world_size,
            self.config.mamba_d_state,
        )
        return conv_state_shape, temporal_state_shape

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts)

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if "A_log" in name:
                name = name.replace("A_log", "A")

            if ".self_attn." in name:
                name = name.replace(".self_attn", "")

            if "feed_forward" in name and not _is_moe_layer(name):
                ## map MLP layers to expert with ID=0
                name = name.replace("feed_forward", "feed_forward.experts.0")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if 'experts' in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
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
                        shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue

                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)


def _is_moe_layer(name: str):
    return any(
        [experts_name in name for experts_name in [
            "experts",
            "router",
        ]])
