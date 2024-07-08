# coding=utf-8
"""Inference-only Jurassic model."""
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from torch import nn
from torch.nn.parameter import Parameter
from transformers import JambaConfig

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.layer import Attention
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors, SamplerOutput
from vllm.worker.model_runner import _BATCH_SIZES_TO_CAPTURE

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

        def weight_loader(param: Parameter, loaded_weight: torch.Tensor):
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
            param.data.copy_(
                loaded_weight.data.split(loaded_weight.shape[0] // tp_size,
                                         dim=0)[tp_rank])

        def A_weight_loader(param: Parameter, loaded_weight: torch.Tensor):
            weight_loader(param, -torch.exp(loaded_weight.float()))

        tp_size = get_tensor_model_parallel_world_size()
        self.A = nn.Parameter(
            torch.empty(
                self.intermediate_size // tp_size,
                self.ssm_state_size,
                dtype=torch.float32,
            ))
        self.D = nn.Parameter(torch.ones(self.intermediate_size // tp_size))

        set_weight_attrs(self.D, {"weight_loader": weight_loader})
        set_weight_attrs(self.A, {"weight_loader": A_weight_loader})

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

    def mamba_forward(self,
                      hidden_states: torch.Tensor,
                      cache_params: MambaCacheParams = None):
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states)[0].transpose(1, 2)
        hidden_states, gate = projected_states.chunk(2, dim=1)

        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0),
                                               self.conv1d.weight.size(2))
        if cache_params is not None and not cache_params.is_prompt:
            hidden_states = causal_conv1d_update(
                hidden_states.squeeze(-1),
                cache_params.conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
            )
            hidden_states = hidden_states.unsqueeze(-1)
        else:
            if cache_params is not None:
                conv_states = nn.functional.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0))
                cache_params.conv_state.copy_(conv_states)

            hidden_states = causal_conv1d_fn(
                hidden_states,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
            )

        # 3. State Space Model sequence transformation
        # 3.a. input varying initialization of time_step, B and C
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))[0]

        time_step, B, C = torch.split(
            ssm_parameters,
            [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
            dim=-1,
        )
        time_step = self.dt_layernorm(time_step.contiguous())
        B = self.b_layernorm(B.contiguous())
        C = self.c_layernorm(C.contiguous())

        discrete_time_step = self.dt_proj(time_step)[0].transpose(1, 2)
        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        time_proj_bias = (self.dt_proj.bias.float() if hasattr(
            self.dt_proj, "bias") else None)
        if cache_params is not None and not cache_params.is_prompt:
            scan_outputs = selective_state_update(
                cache_params.ssm_state,
                hidden_states[..., 0],
                discrete_time_step[..., 0],
                self.A,
                B[:, 0],
                C[:, 0],
                self.D,
                gate[..., 0],
                time_proj_bias,
                dt_softplus=True,
            ).unsqueeze(-1)
        else:
            scan_outputs, ssm_state = selective_scan_fn(
                hidden_states,
                discrete_time_step,
                self.A,
                B.transpose(1, 2),
                C.transpose(1, 2),
                self.D.float(),
                gate,
                time_proj_bias,
                delta_softplus=True,
                return_last_state=True,
            )
            if ssm_state is not None and cache_params is not None:
                cache_params.ssm_state.copy_(ssm_state)

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))[0]
        return contextualized_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ):
        if attn_metadata.prefill_metadata is not None:
            offset = 0
            for i, prompt_len in enumerate(
                    attn_metadata.prefill_metadata.seq_lens):
                cache = MambaCacheParams(True,
                                         conv_state=conv_state[i].unsqueeze(0),
                                         ssm_state=ssm_state[i].unsqueeze(0))
                hidden_states[offset:offset + prompt_len].copy_(
                    self.mamba_forward(hidden_states[offset:offset +
                                                     prompt_len].unsqueeze(0),
                                       cache_params=cache)[0])
                offset += prompt_len
        else:
            cache = MambaCacheParams(False,
                                     conv_state=conv_state,
                                     ssm_state=ssm_state)
            hidden_states = self.mamba_forward(hidden_states.unsqueeze(1),
                                               cache_params=cache)
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class JambaMLP(nn.Module):

    def __init__(
        self,
        config: JambaConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        hidden_act = config.hidden_act
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class JambaMoE(nn.Module):
    """A tensor-parallel MoE implementation for Mixtral that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(
        self,
        config: JambaConfig,
        params_dtype: Optional[torch.dtype] = None,
        tp_size: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.tp_size = tp_size or get_tensor_model_parallel_world_size()
        self.num_total_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size // self.tp_size

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        self.router = ReplicatedLinear(self.hidden_size,
                                       self.num_total_experts,
                                       bias=False,
                                       params_dtype=self.params_dtype)

        self.ws = nn.Parameter(
            torch.empty(
                self.num_total_experts,
                2 * self.intermediate_size,
                self.hidden_size,
                device="cuda",
                dtype=self.params_dtype,
            ))
        self.w2s = nn.Parameter(
            torch.empty(
                self.num_total_experts,
                self.hidden_size,
                self.intermediate_size,
                device="cuda",
                dtype=self.params_dtype,
            ))

        set_weight_attrs(
            self.ws,
            {
                "weight_loader": self.weight_loader,
            },
        )
        set_weight_attrs(
            self.w2s,
            {
                "weight_loader": self.weight_loader,
            },
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        expert_id: int,
    ):
        tp_rank = get_tensor_model_parallel_rank()
        param_data = param.data
        shard_size = self.intermediate_size
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        if weight_name.endswith("gate_proj.weight"):
            param_data[expert_id, 0:shard_size, :] = loaded_weight[shard, :]
        if weight_name.endswith("up_proj.weight"):
            param_data[expert_id,
                       shard_size:2 * shard_size, :] = loaded_weight[shard, :]
        if weight_name.endswith("down_proj.weight"):
            param_data[expert_id, :, :] = loaded_weight[:, shard]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits, _ = self.router(hidden_states)

        final_hidden_states = fused_moe(
            hidden_states,
            self.ws,
            self.w2s,
            router_logits,
            self.top_k,
            renormalize=
            False,  # Mixtral normalize the expert probs to 1. We don't!
            inplace=True,
        )

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_size)


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


class JambaForCausalLM(nn.Module):
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
    ) -> None:
        super().__init__()
        self.config = config
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
        # Current step used indices
        self.current_indices: List[int] = []
        # Used to track and store by the Mamba cache between steps.
        self.mamba_cache: Tuple[torch.Tensor, torch.Tensor] = tuple()
        # Used as an input_buffer for the CUDA graph runs.
        self.mamba_gc_cache_buffer: Tuple[torch.Tensor, torch.Tensor] = tuple()
        # Maps between the request id and a dict that maps between the seq_id
        # and its index inside the self.mamba_cache
        self.mamba_cache_indices_mapping: Dict[str, Dict[int, int]] = {}
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
        if not self.mamba_cache:
            self._prepare_mamba_cache()

        if "seqlen_agnostic_capture_inputs" not in kwargs:
            # We get here only on Prefill/Eager mode runs
            assert all(
                key in kwargs
                for key in ["request_ids_to_seq_ids", "finished_requests_ids"])

            request_ids_to_seq_ids = kwargs["request_ids_to_seq_ids"]
            batch_size = input_ids.shape[0]
            if attn_metadata.prefill_metadata:
                batch_size = len(request_ids_to_seq_ids)
            (
                current_seqlen_agnostic_cache,
                indices,
            ) = self._prepare_current_run_mamba_cache(request_ids_to_seq_ids,
                                                      batch_size)
            finished_requests_ids = kwargs["finished_requests_ids"]
            self._release_mamba_cache(finished_requests_ids)
        else:
            # CUDA graph capturing runs
            current_seqlen_agnostic_cache, indices = (
                kwargs["seqlen_agnostic_capture_inputs"],
                [],
            )
        self.current_indices = indices

        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata,
                                   current_seqlen_agnostic_cache[0],
                                   current_seqlen_agnostic_cache[1])

        if "seqlen_agnostic_capture_inputs" not in kwargs:
            self._copy_mamba_cache_by_indices(self.current_indices,
                                              current_seqlen_agnostic_cache)

        return hidden_states

    def _copy_mamba_cache_by_indices(
            self, indices: List[int],
            current_seqlen_agnostic_cache: Tuple[torch.Tensor, torch.Tensor]):
        for i, offset in enumerate(indices):
            self._copy_mamba_cache(offset, i, current_seqlen_agnostic_cache)

    def _copy_mamba_cache(self, index_to: int, index_from: int,
                          from_buffer: Tuple[torch.Tensor, torch.Tensor]):
        assert len(self.mamba_cache) > 0
        for (cache_t, from_buffer_t) in zip(self.mamba_cache, from_buffer):
            cache_t[:, index_to].copy_(from_buffer_t[:, index_from],
                                       non_blocking=True)

    def _assign_seq_id_to_mamba_cache(self, cur_rid: str,
                                      seqs_id: List[int]) -> List[int]:
        indices_for_current_run = []
        for seq_id in seqs_id:
            if cur_rid not in self.mamba_cache_indices_mapping:
                self.mamba_cache_indices_mapping[cur_rid] = {}
                first_free_index = self._first_free_index_in_mamba_cache()
                self.mamba_cache_indices_mapping[cur_rid][
                    seq_id] = first_free_index
                index_for_current_run = first_free_index
            ## case of decoding n>1, copy prefill cache to decoding indices
            elif seq_id not in (seq_ids2indices :=
                                self.mamba_cache_indices_mapping[cur_rid]):
                first_free_index = self._first_free_index_in_mamba_cache()
                index_exist = list(seq_ids2indices.values())[0]
                self._copy_mamba_cache(index_from=index_exist,
                                       index_to=first_free_index,
                                       from_buffer=self.mamba_cache)
                self.mamba_cache_indices_mapping[cur_rid][
                    seq_id] = first_free_index
                index_for_current_run = first_free_index
            else:
                index_for_current_run = self.mamba_cache_indices_mapping[
                    cur_rid][seq_id]

            indices_for_current_run.append(index_for_current_run)
        return indices_for_current_run

    def _prepare_current_run_mamba_cache(
        self, request_ids_to_seq_ids: Dict[str, list[int]], batch_size: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], List[int]]:
        indices_for_current_run = []
        for request_id, seqs_id in request_ids_to_seq_ids.items():
            indices_for_current_run += self._assign_seq_id_to_mamba_cache(
                request_id, seqs_id)
        ## Pad the batch in case of running batch that was not captured via CG
        padded_indices = indices_for_current_run.copy()
        pad_index = self._first_free_index_in_mamba_cache()

        for _ in range(batch_size - len(indices_for_current_run)):
            padded_indices.append(pad_index)

        conv_state = self.mamba_cache[0][:, padded_indices]
        temporal_state = self.mamba_cache[1][:, padded_indices]

        return (conv_state, temporal_state), indices_for_current_run

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        """
        Copy the relevant Mamba cache into the CUDA graph input buffer 
        that was provided during the capture runs 
        (JambaForCausalLM.mamba_gc_cache_buffer). 
        """
        assert all(
            key in kwargs
            for key in ["request_ids_to_seq_ids", "finished_requests_ids"])
        request_ids_to_seq_ids = kwargs["request_ids_to_seq_ids"]
        cg_batch_size = input_buffers['input_ids'].shape[0]
        (
            current_mamba_cache,
            indices,
        ) = self._prepare_current_run_mamba_cache(request_ids_to_seq_ids,
                                                  cg_batch_size)
        self.current_indices = indices
        finished_requests_ids = kwargs["finished_requests_ids"]
        self._release_mamba_cache(finished_requests_ids)

        for input_buffer, current_cache_buffer in zip(
                input_buffers["seqlen_agnostic_capture_inputs"],
                current_mamba_cache):
            input_buffer.copy_(current_cache_buffer, non_blocking=True)

    def copy_outputs_after_cuda_graphs(self, input_buffers, **kwargs):
        """
        Copy the relevant Mamba cache from the CUDA graph input_buffers
        back to the JambaForCausalLM.mamba_cache after CUDA 
        graph replay run is done.
        """
        self._copy_mamba_cache_by_indices(
            self.current_indices,
            input_buffers["seqlen_agnostic_capture_inputs"])

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        """
        Provide the CUDA graph capture runs with a buffer in adjusted size.
        The buffer is used to maintain the Mamba Cache during the CUDA graph 
        replay runs.
        """
        return tuple(buffer[:, :batch_size]
                     for buffer in self.mamba_gc_cache_buffer)

    def _release_mamba_cache(self, finished_seq_groups_req_ids: List[str]):
        for req_id in finished_seq_groups_req_ids:
            if req_id in self.mamba_cache_indices_mapping:
                self.mamba_cache_indices_mapping.pop(req_id)

    def _first_free_index_in_mamba_cache(self) -> int:
        if self.mamba_cache:
            max_possible_batch_size = self.mamba_cache[0].shape[1]
            occupied = [
                id for seq_ids in self.mamba_cache_indices_mapping.values()
                for id in seq_ids.values()
            ]
            first_free_index = [
                i not in occupied for i in range(max_possible_batch_size)
            ].index(True)
            return first_free_index
        return 0

    def _get_mamba_cache_shape(
            self
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        world_size = get_tensor_model_parallel_world_size()
        hidden_size = self.config.hidden_size
        conv_state_shape = (
            self.config.mamba_expand * hidden_size // world_size,
            self.config.mamba_d_conv,
        )
        temporal_state_shape = (
            self.config.mamba_expand * self.config.hidden_size // world_size,
            self.config.mamba_d_state,
        )
        return conv_state_shape, temporal_state_shape

    def _prepare_mamba_cache(self):
        dtype = self.lm_head.weight.dtype
        layers_type = self.config.layers_block_type
        mamba_layers = sum(
            [layer_type == "mamba" for layer_type in layers_type])
        max_batch_size = _BATCH_SIZES_TO_CAPTURE[-1] + 10
        conv_state_shape, temporal_state_shape = self._get_mamba_cache_shape()
        assert conv_state_shape is not None and temporal_state_shape is not None
        for buffername in ["mamba_cache", "mamba_gc_cache_buffer"]:
            buffer = (torch.empty(size=(mamba_layers, max_batch_size) +
                                  conv_state_shape,
                                  dtype=dtype,
                                  device="cuda"),
                      torch.empty(size=(mamba_layers, max_batch_size) +
                                  temporal_state_shape,
                                  dtype=dtype,
                                  device="cuda"))
            setattr(self, buffername, buffer)

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
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
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = [
            # (param_name, weight_name, expert_id)
            (
                "ws" if weight_name in ["gate_proj", "up_proj"] else "w2s",
                f"experts.{expert_id}.{weight_name}.weight",
                expert_id,
            ) for expert_id in range(self.config.num_experts)
            for weight_name in ["down_proj", "up_proj", "gate_proj"]
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if "A_log" in name:
                name = name.replace("A_log", "A")

            if ".self_attn." in name:
                name = name.replace(".self_attn", "")

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
                for param_name, weight_name, expert_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  weight_name,
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
