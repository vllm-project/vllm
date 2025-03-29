from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.fla.rwkv6 import fused_recurrent_rwkv6
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, row_parallel_weight_loader)
from vllm.model_executor.models.interfaces import (HasInnerState,
                                                   IsAttentionFree)
from vllm.model_executor.models.mamba_cache import (MambaCacheManager,
                                                    MambaCacheParams)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader, PPMissingLayer, make_empty_intermediate_tensors_factory,
    make_layers, maybe_prefix)
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           RowvLLMParameter)
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput


def column_parallel_weight_loader(param: torch.Tensor,
                                  loaded_weight: torch.Tensor) -> None:
    """Load weights that are row-parallelized."""
    tp_rank = get_tensor_model_parallel_rank()
    shard_dim = -1

    if shard_dim is not None:
        shard_size = param.data.shape[shard_dim]
        start_idx = tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(shard_dim, start_idx, shard_size)

    return default_weight_loader(param, loaded_weight)


class RWKV6MLP(torch.nn.Module):

    def __init__(self, dims, mult=3.5, layer_id=0, config=None):
        super().__init__()
        self.layer_id = layer_id
        self.time_maa_k = torch.nn.Parameter(torch.zeros(1, 1, dims))
        self.time_maa_r = torch.nn.Parameter(torch.zeros(1, 1, dims))

        self.key = ColumnParallelLinear(dims, int(dims * mult), bias=False)
        self.value = RowParallelLinear(int(dims * mult), dims, bias=False)
        self.receptance = ColumnParallelLinear(dims,
                                               dims,
                                               bias=False,
                                               gather_output=True)

    def forward(self, hidden_states, positions, kv_cache: MambaCacheParams,
                input_ids):
        bsz, hidden_dim = hidden_states.size()

        input_shift_state = kv_cache.conv_state

        input_shift_state = input_shift_state.to(
            torch.bfloat16)[kv_cache.state_indices_tensor, 1]

        qlen = bsz // kv_cache.state_indices_tensor.size(0)
        bsz = kv_cache.state_indices_tensor.size(0)
        x = hidden_states.view(bsz, qlen, -1)

        token_positions = positions.view(bsz, qlen)
        resetBatchItem = (token_positions[:, 0] == 0) * -100.0

        input_shift_state = input_shift_state * resetBatchItem.exp().to(
            input_shift_state.dtype).view([-1] + [1] *
                                          (input_shift_state.dim() - 1))
        output_shift_state = x[:, -1:].detach().clone()

        xx = torch.cat([input_shift_state, x[:, :-1]], dim=1)
        delta_hidden_to_shifted = xx - x
        xk = x + delta_hidden_to_shifted * self.time_maa_k
        xr = x + delta_hidden_to_shifted * self.time_maa_r
        rr, _ = self.receptance(xr)

        kv_cache.conv_state[kv_cache.state_indices_tensor,
                            1] = output_shift_state

        k, _ = self.key(xk)
        k = torch.relu(k)**2
        kv, _ = self.value(k)
        return (rr.sigmoid() * kv).view(bsz * qlen, -1)


class GroupNormParralelVertical(torch.nn.GroupNorm):

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        torch.nn.Module.__init__(self)
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = RowvLLMParameter(
            data=torch.zeros(num_channels),
            input_dim=0,
            weight_loader=column_parallel_weight_loader)
        self.bias = RowvLLMParameter(
            data=torch.zeros(num_channels),
            input_dim=0,
            weight_loader=column_parallel_weight_loader)


class RWKV6Attention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        hidden_size = config.hidden_size
        attention_hidden_size = config.attention_hidden_size
        self.attention_hidden_size = attention_hidden_size
        head_size = config.head_size
        num_heads = attention_hidden_size // head_size

        self.time_maa_x = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_maa_w = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_maa_k = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_maa_v = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_maa_r = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_maa_g = nn.Parameter(torch.empty(1, 1, hidden_size))

        TIME_MIX_EXTRA_DIM = 32
        if hidden_size == 4096:
            TIME_MIX_EXTRA_DIM = 64
        self.time_maa_w1 = nn.Parameter(
            torch.empty(hidden_size, TIME_MIX_EXTRA_DIM * 5))
        self.time_maa_w2 = nn.Parameter(
            torch.empty(5, TIME_MIX_EXTRA_DIM, hidden_size))

        self.time_decay = ChannelQuantScaleParameter(
            data=torch.empty(
                1, 1, attention_hidden_size //
                get_tensor_model_parallel_world_size()),
            output_dim=2,
            weight_loader=column_parallel_weight_loader)

        TIME_DECAY_EXTRA_DIM = 64
        if hidden_size == 4096:
            TIME_DECAY_EXTRA_DIM = 128
        self.time_decay_w1 = nn.Parameter(
            torch.empty(hidden_size, TIME_DECAY_EXTRA_DIM))

        self.time_decay_w2 = ChannelQuantScaleParameter(
            data=torch.zeros(
                TIME_DECAY_EXTRA_DIM, attention_hidden_size //
                get_tensor_model_parallel_world_size()).uniform_(-0.01, 0.01),
            output_dim=1,
            weight_loader=column_parallel_weight_loader)

        self.time_faaaa = RowvLLMParameter(
            data=torch.zeros(
                num_heads // get_tensor_model_parallel_world_size(),
                config.head_size),
            input_dim=0,
            weight_loader=row_parallel_weight_loader)

        self.receptance = ColumnParallelLinear(hidden_size,
                                               attention_hidden_size,
                                               bias=False)
        self.key = ColumnParallelLinear(hidden_size,
                                        attention_hidden_size,
                                        bias=False)
        self.value = ColumnParallelLinear(hidden_size,
                                          attention_hidden_size,
                                          bias=False)
        self.gate = ColumnParallelLinear(hidden_size,
                                         attention_hidden_size,
                                         bias=False)
        self.output = RowParallelLinear(attention_hidden_size,
                                        hidden_size,
                                        bias=False)
        self.ln_x = GroupNormParralelVertical(
            num_heads // get_tensor_model_parallel_world_size(),
            hidden_size // get_tensor_model_parallel_world_size(),
            eps=(1e-5) * (config.head_size_divisor**2))
        self.head_dim = config.head_size

    def extract_key_value(self, hidden, shifted):
        # Mix hidden with the previous timestep
        # to produce key, value, receptance

        x = hidden

        B, T, C = hidden.shape

        xx = shifted - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 5,
                                                      -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        time_decay = x + xx * (self.time_maa_w + mw)
        key = x + xx * (self.time_maa_k + mk)
        value = x + xx * (self.time_maa_v + mv)
        receptance = x + xx * (self.time_maa_r + mr)
        gate = x + xx * (self.time_maa_g + mg)

        receptance, _ = self.receptance(receptance)
        key, _ = self.key(key)
        value, _ = self.value(value)
        gg, _ = self.gate(gate)
        gate = F.silu(gg)

        time_decay = torch.tanh(
            time_decay @ self.time_decay_w1) @ self.time_decay_w2
        time_decay = self.time_decay + time_decay

        time_decay = time_decay.float().exp().neg()

        return receptance, key, value, gate, time_decay

    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)

        x = self.ln_x(x).view(B, T, C)
        x, _ = self.output(x * g)
        return x

    def forward(
        self,
        hidden_states,
        kv_cache: MambaCacheParams,
        positions: torch.Tensor = None,
    ):

        bsz, hidden_dim = hidden_states.size()

        input_kv_state = kv_cache.ssm_state
        input_shift_state = kv_cache.conv_state

        input_kv_state = input_kv_state[kv_cache.state_indices_tensor]
        input_shift_state = input_shift_state.to(
            torch.bfloat16)[kv_cache.state_indices_tensor, 0]

        qlen = bsz // kv_cache.state_indices_tensor.size(0)
        bsz = kv_cache.state_indices_tensor.size(0)
        x = hidden_states.view(bsz, qlen, -1)
        token_positions = positions.view(bsz, qlen)
        resetBatchItem = (token_positions[:, 0] == 0) * -100.0

        input_shift_state = input_shift_state * resetBatchItem.exp().to(
            input_shift_state.dtype).view([-1] + [1] *
                                          (input_shift_state.dim() - 1))
        output_shift_state = x[:, -1:].detach().clone()

        B = bsz
        T = qlen
        Z = self.head_dim

        r, k, v, g, w = self.extract_key_value(x, input_shift_state)

        u = self.time_faaaa.view(-1, Z)

        r = r.view(B, T, -1, Z).transpose(1, 2).contiguous()
        k = k.view(B, T, -1, Z).transpose(1, 2).contiguous()
        v = v.view(B, T, -1, Z).transpose(1, 2).contiguous()
        w = w.view(B, T, -1, Z).transpose(1, 2).contiguous()

        w[:, :, 0] = w[:, :, 0] + (
            (token_positions[:, 0] == 0) * -100.0).view(bsz, 1, 1).float()

        s = input_kv_state
        xo, sout = fused_recurrent_rwkv6(r, k, v, w, u, 1, s, True)
        xo = xo.transpose(1, 2).contiguous().view(B, T, -1)

        kv_cache.ssm_state[kv_cache.state_indices_tensor] = sout.to(
            torch.bfloat16)
        kv_cache.conv_state[kv_cache.state_indices_tensor,
                            0] = output_shift_state

        return self.jit_func_2(xo, g).view(B * T, -1)


class RWKV6DecoderLayer(nn.Module):

    def __init__(self,
                 config: VllmConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",
                 mlpclass=RWKV6MLP) -> None:
        nn.Module.__init__(self)
        layer_id = int(prefix.split(".")[-1])
        self.hidden_size = config.hidden_size
        self.attention = RWKV6Attention(config)
        self.feed_forward = mlpclass(config.hidden_size,
                                     layer_id=layer_id,
                                     config=config)
        self.ln1 = torch.nn.LayerNorm(config.hidden_size)
        self.ln2 = torch.nn.LayerNorm(config.hidden_size)
        self.pre_ln = torch.nn.LayerNorm(
            config.hidden_size) if ".0" in prefix else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        rwkv6_cache_params: MambaCacheParams,
        positions: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Self Attention
        hidden_states = self.pre_ln(
            hidden_states) if self.pre_ln else hidden_states
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        hidden_states = self.attention(hidden_states, rwkv6_cache_params,
                                       positions)
        hidden_states = hidden_states + residual
        # MLP
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.feed_forward(
            hidden_states, positions, rwkv6_cache_params, input_ids) + residual
        return hidden_states


class rwkv6Model(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 mlpclass=RWKV6MLP):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        # TODO (@robertgshaw2): see if this can be moved out
        if (cache_config.sliding_window is not None
                and hasattr(config, "max_window_layers")):
            raise ValueError("Sliding window for some but all layers is not "
                             "supported. This model uses sliding window "
                             "but `max_window_layers` = {} is less than "
                             "`num_hidden_layers` = {}. Please open an issue "
                             "to discuss this feature.".format(
                                 config.max_window_layers,
                                 config.num_hidden_layers,
                             ))

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embeddings = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embeddings",
            )
        else:
            self.embeddings = PPMissingLayer()

        self.start_layer, self.end_layer, self.blocks = make_layers(
            config.num_hidden_layers,
            lambda prefix: RWKV6DecoderLayer(config=config,
                                             cache_config=cache_config,
                                             quant_config=quant_config,
                                             prefix=f"{prefix}",
                                             mlpclass=mlpclass),
            prefix=f"{prefix}.blocks",
        )

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

        if get_pp_group().is_last_rank:
            self.ln_out = torch.nn.LayerNorm(config.hidden_size)
        else:
            self.ln_out = PPMissingLayer()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttentionMetadata,
        rwkv6_cache_params: MambaCacheParams,
    ) -> torch.Tensor:

        hidden_states = self.embeddings(input_ids)

        for i in range(len(self.blocks)):
            layer = self.blocks[i]
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                rwkv6_cache_params=rwkv6_cache_params.at_layer_idx(i),
                input_ids=input_ids)
        hidden_states = self.ln_out(hidden_states)

        return hidden_states


class Rwkv6ForCausalLM(nn.Module, HasInnerState, IsAttentionFree):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 mlpclass=RWKV6MLP,
                 modelprefix="rwkv"):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.scheduler_config = vllm_config.scheduler_config
        self.model_config = vllm_config.model_config

        self.config = config
        self.lora_config = lora_config
        self.modelprefix = modelprefix

        self.quant_config = quant_config
        setattr(
            self, self.modelprefix,
            rwkv6Model(vllm_config=vllm_config,
                       prefix=maybe_prefix(prefix, modelprefix),
                       mlpclass=mlpclass))

        if config.tie_word_embeddings:
            self.head = getattr(self, self.modelprefix).embeddings
        else:
            self.head = ParallelLMHead(config.vocab_size,
                                       config.hidden_size,
                                       quant_config=quant_config,
                                       prefix=maybe_prefix(prefix, "head"))

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = get_sampler()

        self.make_empty_intermediate_tensors = (getattr(
            self, self.modelprefix).make_empty_intermediate_tensors)

        if self.scheduler_config is not None and \
            not self.model_config.enforce_eager:
            if self.scheduler_config.max_num_seqs > \
                vllm_config.compilation_config.max_capture_size:
                self.max_batch_size = \
                    vllm_config.compilation_config.max_capture_size
            else:
                self.max_batch_size = vllm_config.pad_for_cudagraph(
                    self.scheduler_config.max_num_seqs)
        else:
            self.max_batch_size = 8192 + 2

        self.rwkv6_cache: Optional[MambaCacheManager] = None

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List,
                attn_metadata: AttentionMetadata,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                **kwargs):
        if self.rwkv6_cache is None:

            self.rwkv6_cache = MambaCacheManager(
                torch.bfloat16, self.config.num_hidden_layers,
                self.max_batch_size, *self._get_rwkv6_cache_shape())

        (
            rwkv6_cache_tensors,
            state_indices_tensor,
        ) = self.rwkv6_cache.current_run_tensors(input_ids, attn_metadata,
                                                 **kwargs)

        rwkv6_cache_params = MambaCacheParams(rwkv6_cache_tensors[0],
                                              rwkv6_cache_tensors[1],
                                              state_indices_tensor)

        hidden_states = getattr(self, self.modelprefix)(input_ids, positions,
                                                        attn_metadata,
                                                        rwkv6_cache_params)

        return hidden_states

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.rwkv6_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.rwkv6_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def _get_rwkv6_cache_shape(
            self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        world_size = get_tensor_model_parallel_world_size()
        head_size = self.config.head_size
        num_heads = self.config.hidden_size // head_size
        conv_state_shape = (2, 1, self.config.hidden_size)
        temporal_state_shape = (num_heads // world_size, head_size, head_size)
        return conv_state_shape, temporal_state_shape

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["head."]
                           if self.config.tie_word_embeddings else None),
        )
        loader.load_weights(weights)
