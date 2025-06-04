from typing import List, Optional, Tuple, Union, Iterable, Dict
import math
import copy

import torch
import torch.nn as nn

from einops import rearrange
from transformers.activations import ACT2FN
from typing import Iterable, List, Optional, Set, Tuple, Union

from vllm.config import CacheConfig, VllmConfig
from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               RowParallelLinear,
                                               ColumnParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.mamba_cache import (MambaCacheManager,
                                                    MambaCacheParams)
from vllm.model_executor.models.interfaces import (HasInnerState,
                                                   IsHybrid, SupportsV0Only)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn, causal_conv1d_update)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_scan_fn, selective_state_update)
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.vllm_flash_attn import (flash_attn_varlen_func,
                                  flash_attn_with_kvcache)

from vllm.logger import init_logger
from .utils import (maybe_prefix, make_layers)
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)

logger = init_logger(__name__)


class SwiGLUActivation(nn.Module):

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # print(f"x1 shape: {x1.shape}, x2 shape: {x2.shape}")
        return x1 * nn.functional.silu(x2)
    
class SambaMLP(nn.Module):
    """Gated Linear Unit.

    Reference:
        Language Modeling with Gated Convolutional Networks.
        https://arxiv.org/pdf/1612.08083v3.pdf.

    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        y = self.fc1(hidden_states)
        gate, y = y.chunk(2, dim=-1)
        y = y * self.activation_fn(gate)
        return self.fc2(y)


class SambaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, 
                 config, 
                 layer_idx: Optional[int] = None, 
                 yoco_cross: bool = False, 
                 cache_config: Optional[CacheConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.yoco_cross = yoco_cross
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        op_size = self.num_heads * self.head_dim + 2 * (self.num_key_value_heads * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)
        if yoco_cross:
            self.Wqkv =  nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        else:
            self.Wqkv = nn.Linear(self.hidden_size, op_size, bias=True)

        assert self.config.attention_dropout == 0.0, 'Attention dropout is not supported for now'

        # disable sliding window for the second half of the model
        sliding_window = config.interleaved_sliding_window[layer_idx]
        if layer_idx >= config.num_hidden_layers // 2 or layer_idx % 2 == 0:
            assert sliding_window == None, "sliding_window is not none"

        assert self.num_heads % 2 == 0, 'num_heads should be even'
        assert self.num_key_value_heads % 2 == 0, 'num_heads should be even'

        self.attn = Attention(
            self.num_heads//2,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_key_value_heads//2,
            cache_config=cache_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.DECODER_DECODER if self.yoco_cross else AttentionType.DECODER
        )
        self.subln = nn.RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

        self.lambda_init = self.lambda_init_fn(layer_idx)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self._k_scale = torch.tensor(1.0, dtype=torch.float32)
        self._v_scale = torch.tensor(1.0, dtype=torch.float32)

    def lambda_init_fn(self, depth):
        return 0.8 - 0.6 * math.exp(-0.3 * depth)


    def split_heads(self, x):
        # split by num_heads, the stripe pattern is friendly to tensor parallel.
        x = rearrange(x, "... (H two) D -> ... H two D", two=2)
        x1 = x[..., 0, :]
        x2 = x[..., 1, :]
        return x1.contiguous(), x2.contiguous()
    
    def split_kv_cache(self, x):
        # split by num_heads, the stripe pattern is friendly to tensor parallel.
        if x.numel() == 0:
            return torch.empty(0), torch.empty(0)
        
        x1, x2 = x[0], x[1]
        return x1, x2

    def forward_decode(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ):
        if not attn_metadata.decode_metadata:            
            block_tables_arg = attn_metadata.cross_layer_shared_block_tables                
        else:
            block_tables_arg = attn_metadata.block_tables

        output = flash_attn_with_kvcache(
            q=query.unsqueeze(1),
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_tables_arg,
            cache_seqlens=attn_metadata.seq_lens_tensor,
            softmax_scale=self.attn.impl.scale,
            causal=True,
            window_size=self.attn.impl.sliding_window,
            alibi_slopes=self.attn.impl.alibi_slopes,
            softcap=self.attn.impl.logits_soft_cap,
        ).squeeze(1)
        return output

    def populate_kv_cache(self,
                          key, 
                          value, 
                          kv_cache, 
                          attn_metadata):
        if (kv_cache.numel() > 0):
            if (key is not None) and (value is not None):
                updated_slot_mapping = attn_metadata.slot_mapping
                # previous_key_cache_sum = key_cache.sum()
                # previous_value_cache_sum = value_cache.sum()

                torch.ops._C_cache_ops.reshape_and_cache_flash(
                    key,
                    value,
                    kv_cache[0],
                    kv_cache[1],
                    updated_slot_mapping.flatten(),
                    self.attn.impl.kv_cache_dtype,
                    self._k_scale,
                    self._v_scale,
                )
                # assert key_cache.sum() - previous_key_cache_sum == key.sum(), "key_cache sum mismatch"
                # assert value_cache.sum() - previous_value_cache_sum == value.sum(), "value_cache sum mismatch"
                # if key_cache.sum() - previous_key_cache_sum != key.sum():
                #     print("key_cache sum mismatch")
                # if value_cache.sum() - previous_value_cache_sum != value.sum():
                #     print("value_cache sum mismatch")

    def forward_customized(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: AttentionMetadata
    ) -> torch.Tensor:

        head_size = self.head_dim
        num_heads = self.num_heads // 2
        num_kv_heads = self.num_key_value_heads // 2

        query = query.view(-1, num_heads, head_size)
        if key is not None:
            assert value is not None
            key = key.view(-1, num_kv_heads, head_size)
            value = value.view(-1, num_kv_heads, head_size)
        else:
            assert value is None

        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        assert key.shape[0] == num_prefill_tokens + num_decode_tokens, "key shape mismatch"
        assert value.shape[0] == num_prefill_tokens + num_decode_tokens, "value shape mismatch"
        
        output = torch.empty_like(query)
        # Query for decode. KV is not needed because it is already cached.
        decode_query = query[num_prefill_tokens:]
        # QKV for prefill.
        query = query[:num_prefill_tokens]
        if key is not None and value is not None:
            key = key[:num_prefill_tokens]
            value = value[:num_prefill_tokens]

        assert query.shape[0] == num_prefill_tokens, "query shape mismatch"
        assert decode_query.shape[0] == num_decode_tokens, "decode query shape mismatch"

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            if k_cache.numel() == 0 or prefill_meta.block_tables.numel() == 0:
                # normal attention
                prefill_output = flash_attn_varlen_func(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=prefill_meta.seq_start_loc,
                    cu_seqlens_k=prefill_meta.seq_start_loc,
                    max_seqlen_q=prefill_meta.max_prefill_seq_len,
                    max_seqlen_k=prefill_meta.max_prefill_seq_len,
                    softmax_scale=self.attn.impl.scale,
                    causal=True,
                    window_size=self.attn.impl.sliding_window,
                    alibi_slopes=self.attn.impl.alibi_slopes,
                    softcap=self.attn.impl.logits_soft_cap,
                )
                assert prefill_output.shape == output[:num_prefill_tokens].shape
                output[:num_prefill_tokens] = prefill_output
            else:
                raise Exception("prefix caching not supported")

        if decode_meta := attn_metadata.decode_metadata:
            block_tables_arg = decode_meta.block_tables
            try:
                output[num_prefill_tokens:] = flash_attn_with_kvcache(
                    q=decode_query.unsqueeze(1),
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=block_tables_arg,
                    cache_seqlens=decode_meta.seq_lens_tensor,
                    softmax_scale=self.attn.impl.scale,
                    causal=True,
                    window_size=self.attn.impl.sliding_window,
                    alibi_slopes=self.attn.impl.alibi_slopes,
                    softcap=self.attn.impl.logits_soft_cap,
                ).squeeze(1)
            except Exception as e:
                logger.error(
                    f"Error in PagedAttention.forward_decode: {str(e)}")
                raise e

        # Reshape the output tensor.
        return output.view(-1, num_heads, head_size)

    def forward(
            self,
            hidden_states: torch.Tensor,
            positions: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
        ):

        if not self.yoco_cross: # need to generate kv-cache
            qkv = self.Wqkv(hidden_states)
            q, k, v = qkv.split([self.hidden_size, self.num_key_value_heads * self.head_dim, self.num_key_value_heads * self.head_dim], dim=-1)
            # q, k = self.rotary_emb(positions, q, k)
            # reshape
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_key_value_heads, self.head_dim)
            v = v.view(-1, self.num_key_value_heads, self.head_dim)

            q1, q2 = self.split_heads(q)
            k1, k2 = self.split_heads(k)
            v1, v2 = self.split_heads(v)

            # kv_cache shape is (2, 2, num_blocks, block_size * num_kv_heads // 2 * head_size)
            # Split by half along the first dimension.
            kv_cache1, kv_cache2 = self.split_kv_cache(kv_cache)
            assert kv_cache1.is_contiguous(), "kv_cache1 is not contiguous"
            assert kv_cache2.is_contiguous(), "kv_cache2 is not contiguous"
            
            if kv_cache1.numel() != 0:
                self.populate_kv_cache(k1, v1, kv_cache1, attn_metadata)
                self.populate_kv_cache(k2, v2, kv_cache2, attn_metadata)
                
                key_cache1, value_cache1 = self.split_kv_cache(kv_cache1)
                key_cache2, value_cache2 = self.split_kv_cache(kv_cache2)
            else:
                key_cache1, value_cache1 = torch.empty(0), torch.empty(0)
                key_cache2, value_cache2 = torch.empty(0), torch.empty(0)
            attn11 = self.forward_customized(q1, k1, v1, key_cache1, value_cache1, attn_metadata)
            attn12 = self.forward_customized(q1, k1, v2, key_cache1, value_cache2, attn_metadata)
            attn11 = attn11.view(q1.shape)
            attn12 = attn12.view(q1.shape)
            attn1 = torch.cat([attn11, attn12], dim=-1)

            attn21 = self.forward_customized(q2, k2, v1, key_cache2, value_cache1, attn_metadata)
            attn22 = self.forward_customized(q2, k2, v2, key_cache2, value_cache2, attn_metadata)
            attn21 = attn21.view(q2.shape)
            attn22 = attn22.view(q2.shape)
            attn2 = torch.cat([attn21, attn22], dim=-1)

            lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
            lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
            lambda_full = lambda_1 - lambda_2 + self.lambda_init
            attn = attn1 - lambda_full * attn2
            # attn shape (-1, self.num_heads // 2, 2 * self.head_dim)
            attn = self.subln(attn)
            attn = attn * (1 - self.lambda_init)
            # reshape back to 2 * num_head
            attn_output = rearrange(attn, "... H (two D) -> ... (H two) D", two=2)

        else: # re-use the kv cache, full attention
            q = self.Wqkv(hidden_states)
            q = q.view(-1, self.num_heads, self.head_dim)
            q1, q2 = self.split_heads(q)
            # kv_cache shape is (2, num_blocks, block_size * num_kv_heads * head_size)
            kv_cache1, kv_cache2 = self.split_kv_cache(kv_cache)
            key_cache1, value_cache1 = kv_cache1[0], kv_cache1[1]
            key_cache2, value_cache2 = kv_cache2[0], kv_cache2[1]
            
            attn11 = self.forward_decode(q1, key_cache1, value_cache1, attn_metadata)
            attn12 = self.forward_decode(q1, key_cache1, value_cache2, attn_metadata)
            attn11 = attn11.view(q1.shape)
            attn12 = attn12.view(q1.shape)
            attn1 = torch.cat([attn11, attn12], dim=-1)

            attn21 = self.forward_decode(q2, key_cache2, value_cache1, attn_metadata)
            attn22 = self.forward_decode(q2, key_cache2, value_cache2, attn_metadata)
            attn21 = attn21.view(q2.shape)
            attn22 = attn22.view(q2.shape)
            attn2 = torch.cat([attn21, attn22], dim=-1)

            lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
            lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
            lambda_full = lambda_1 - lambda_2 + self.lambda_init
            attn = attn1 - lambda_full * attn2
            attn = self.subln(attn)
            attn = attn * (1 - self.lambda_init)
            # reshape back to 2 * num_head
            attn_output = rearrange(attn, "... H (two D) -> ... (H two) D", two=2)
        attn_output = attn_output.view(-1, self.num_heads * self.head_dim)
        return self.out_proj(attn_output)


class Phi3Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random", # difference
        dt_scale=1.0,  # difference
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        yoco_cross=False,
        yoco_kv=False,
    ):
        factory_kwargs = {"params_dtype": dtype} # difference
        super().__init__()
        self.yoco_cross = yoco_cross
        self.yoco_kv = yoco_kv
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.swiGluActivation = SwiGLUActivation()
        if self.yoco_cross:
            self.in_proj = MergedColumnParallelLinear(self.d_model, [self.d_inner], bias=bias, **factory_kwargs)
            self.out_proj = RowParallelLinear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
            return
        # self.conv1d = nn.Conv1d(
        #     in_channels=self.d_inner,
        #     out_channels=self.d_inner,
        #     bias=conv_bias,
        #     kernel_size=d_conv,
        #     groups=self.d_inner,
        #     padding=d_conv - 1,
        #     **factory_kwargs,
        # )

        self.conv1d = ColumnParallelLinear(
            input_size=d_conv,
            output_size=self.d_inner,
            bias=conv_bias,
            params_dtype=dtype,
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj = MergedColumnParallelLinear(self.d_model,
                                                  [self.d_inner] * 2,
                                                  bias=bias,
                                                  params_dtype=dtype,
                                                 )

        # self.x_proj = nn.Linear(
        #     self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        # )
        # selective projection used to make dt, B and C input dependent
        self.x_proj = RowParallelLinear(
            self.d_inner,
            self.dt_rank + self.d_state * 2,
            bias=False,
            params_dtype=dtype,
        )

        # self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        # time step projection (discretization) -
        # In the forward we need to apply dt_proj without the bias,
        # as the bias is added in the selective scan kernel.
        self.dt_proj = ColumnParallelLinear(self.dt_rank,
                                            self.d_inner,
                                            bias=True,
                                            skip_bias_add=True,
                                            params_dtype=dtype,
                                        )

        # # S4D real initialization
        # A = repeat(
        #     torch.arange(1, self.d_state + 1, dtype=torch.float32),
        #     "n -> d n",
        #     d=self.d_inner,
        # ).contiguous()
        # A_log = torch.log(A)  # Keep A_log in fp32
        # self.A_log = nn.Parameter(A_log)

        # # D "skip" parameter
        # self.D = nn.Parameter(torch.ones(self.d_inner))  # Keep in fp32
        self.A = nn.Parameter(
            torch.empty(
                self.d_inner,
                self.d_state,
                dtype=torch.float32,
            ))
        self.D = nn.Parameter(torch.ones(self.d_inner, dtype=torch.float32))

        # self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj = RowParallelLinear(
            self.d_inner,
            self.d_model,
            bias=bias,
            input_is_parallel=True,
            params_dtype=dtype,
        )
        print(f"-------- layer_idx {layer_idx}")
        self.activation = "silu"

    def forward(
            self,
            hidden_states: torch.Tensor,
            attn_metadata: AttentionMetadata,
            mamba_cache_params: MambaCacheParams,
            yoco_key_values = None
        ) -> torch.Tensor:
        
        if self.yoco_cross:
            out = self.in_proj(hidden_states)[0]
            out = self.swiGluActivation(yoco_key_values, out)
            out = self.out_proj(out)
            return out[0], yoco_key_values 

        # 1. Gated MLP's linear projection
        # projected_states = self.in_proj(hidden_states)[0].transpose(-2, -1)
        projected_states = self.in_proj(hidden_states.to(self.in_proj.weight.dtype))[0].transpose(-2, -1)
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
                conv_states=mamba_cache_params.conv_state,
                has_initial_state=attn_metadata.context_lens_tensor > 0,
                cache_indices=mamba_cache_params.state_indices_tensor,
                query_start_loc=attn_metadata.query_start_loc)
        else:
            hidden_states = causal_conv1d_update(
                hidden_states.transpose(0, 1),
                mamba_cache_params.conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=mamba_cache_params.state_indices_tensor)
            hidden_states = hidden_states.transpose(0, 1)

        # 3. State Space Model sequence transformation
        # 3.a. input varying initialization of time_step, B and C
        ssm_parameters = self.x_proj(hidden_states.transpose(-2, -1))[0]

        time_step, B, C = torch.split(
            ssm_parameters,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )

        # Note that Jamba normalizes B, C, and time_step here but Mamba doesn't.

        discrete_time_step = self.dt_proj(time_step)[0].transpose(-2, -1)
        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        time_proj_bias = (self.dt_proj.bias.float() if hasattr(
            self.dt_proj, "bias") else None)

        if attn_metadata.query_start_loc is not None \
            and attn_metadata.context_lens_tensor is not None:
            scan_outputs = selective_scan_fn(
                hidden_states,
                mamba_cache_params.ssm_state,
                discrete_time_step,
                self.A,
                B.transpose(-2, -1),
                C.transpose(-2, -1),
                self.D.float(),
                # z,
                None if self.yoco_kv else gate,
                time_proj_bias,
                delta_softplus=True,
                cache_indices=mamba_cache_params.state_indices_tensor,
                has_initial_state=attn_metadata.context_lens_tensor > 0,
                query_start_loc=attn_metadata.query_start_loc)
        else:
            scan_outputs = selective_state_update(
                mamba_cache_params.ssm_state,
                hidden_states.transpose(0, 1),
                discrete_time_step.transpose(0, 1),
                self.A,
                B,
                C,
                self.D,
                # z
                # gate.transpose(0, 1),
                None if self.yoco_kv else gate.transpose(0, 1),
                time_proj_bias,
                dt_softplus=True,
                state_batch_indices=mamba_cache_params.state_indices_tensor)
            scan_outputs = scan_outputs.transpose(0, 1)

        # 4. Final linear projection
        if self.yoco_kv:
            # gate = gate.transpose(-1,-2).contiguous()
            yoco_key_values = scan_outputs.transpose(-2, -1)
            scan_outputs = self.swiGluActivation(scan_outputs, gate)

        contextualized_states = self.out_proj(scan_outputs.transpose(-2,
                                                                     -1))[0]

        return contextualized_states, yoco_key_values


class SambaDecoderLayer(nn.Module):
    
    def __init__(self, 
                 config, 
                 layer_idx, 
                 cache_config,
                 prefix: str = "",) -> None:
        super().__init__()
    
        self.config = config
        self.layer_idx = layer_idx

        self.mlp = SambaMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.yoco_mb = False
        self.yoco_kv = False
        self.yoco_cross = False
        assert config.num_hidden_layers % 4 == 0, 'n_layer should be divisible by 4 for samba + yoco'
        if layer_idx >= config.num_hidden_layers//2:
            self.yoco_mb = True
            self.yoco_kv = (layer_idx >= (config.num_hidden_layers//2 +1))
            self.yoco_cross = (layer_idx >= (config.num_hidden_layers//2 +2))
        self.use_mamba = config.mb_per_layer > 0 and layer_idx % config.mb_per_layer == 0
        if self.use_mamba:
            factory_kwargs = {"dtype": None}
            self.attn = Phi3Mamba(config.hidden_size, layer_idx=layer_idx, 
                                  yoco_cross=self.yoco_cross, yoco_kv=self.yoco_mb, **factory_kwargs)
        else:
            self.attn = SambaAttention(config, layer_idx=layer_idx, yoco_cross=self.yoco_cross, cache_config=cache_config, prefix=f"{prefix}.self_attn")

        self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(config.resid_pdrop)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        mamba_cache_params: MambaCacheParams,
        ssm_output: Optional[torch.LongTensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if self.use_mamba:
            assert kv_cache is None and mamba_cache_params is not None
        else:
            assert kv_cache is not None and mamba_cache_params is None

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states.to(dtype=self.input_layernorm.weight.dtype))

        if self.use_mamba:
            attn_outputs, ssm_output = self.attn(
                hidden_states,
                attn_metadata,
                mamba_cache_params,
                yoco_key_values = ssm_output
            )
            residual = residual.to(torch.float32)
        else:
            attn_outputs = self.attn(
                hidden_states,
                positions,
                kv_cache,
                attn_metadata,
            )
        try:
            hidden_states = residual + self.resid_attn_dropout(attn_outputs)
        except Exception as e:
            print('>>> exception: ', e)    
            print('>>>', hidden_states.shape)
            print('>>>', self.layer_idx)
            print('>>>', residual.shape)
            print('>>>', self.resid_attn_dropout)
            print('>>>', attn_outputs)
            raise

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states.to(dtype=self.post_attention_layernorm.weight.dtype))
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)

        return hidden_states, ssm_output

def get_kv_cache(layer_name):
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    kv_cache = self.kv_cache[forward_context.virtual_engine]
    return kv_cache

class SambaModel(nn.Module):

    def __init__(
        self,
        config,
        cache_config = None,
        quant_config = None,
        lora_config = None,
        prefix: str = ""
    ) -> None:
        super().__init__()

        self.config = config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )
        self.embed_dropout = nn.Dropout(config.embd_pdrop)
        # Pipeline parallel is not supported since the second half of the layers share the kv cache.
        if get_pp_group().world_size != 1:
            raise ValueError("Pipeline Parallel not supported")
        
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: SambaDecoderLayer(config,
                                             int(prefix.split('.')[-1]),
                                             cache_config,
                                             prefix=prefix),
            prefix=f"{prefix}.layers")
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        attn_metadata: AttentionMetadata,
        mamba_cache_params: MambaCacheParams,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
    
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        kv_cache_idx = 0
        mamba_state_idx = 0
        ssm_output = None
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            if i == self.config.num_hidden_layers // 2 + 2:
                # profile run
                cache_layer = self.layers[kv_cache_idx]
                kv_cache = get_kv_cache(cache_layer.attn.attn.layer_name)
                if kv_cache.numel() == 0:
                    break

                # Starting from this layer, we do not need to cuculate the kv cache since we reuse
                # the kv cache from last layer. If in prefill phase, we can prune truncate 
                # hidden state to save computation cost.
                if attn_metadata.prefill_metadata:
                    selected_token_indices = torch.cumsum(attn_metadata.seq_lens_tensor, dim=0) - 1
                    hidden_states = hidden_states.index_select(0, selected_token_indices)
                    ssm_output = ssm_output.index_select(0, selected_token_indices)


            # start_env = torch.cuda.Event(enable_timing=True)
            # end_env = torch.cuda.Event(enable_timing=True)
            # start_env.record()
            if layer.use_mamba:
                if i < self.config.num_hidden_layers // 2:
                    mamba_cache = mamba_cache_params.at_layer_idx(mamba_state_idx)
                    mamba_state_idx += 1
                elif not layer.yoco_cross:
                    mamba_cache = mamba_cache_params.at_layer_idx(mamba_state_idx)
                    mamba_state_idx += 1
                else:
                    mamba_cache = mamba_cache_params.at_layer_idx(mamba_state_idx-1)
        
                hidden_states, ssm_output = layer(
                    hidden_states,
                    positions,
                    None, # kv_cache
                    attn_metadata,
                    mamba_cache,
                    ssm_output = ssm_output
                )
            else:
                if i < self.config.num_hidden_layers // 2:
                    # sliding window attention
                    cache_layer = self.layers[i]
                    kv_cache = get_kv_cache(cache_layer.attn.attn.layer_name)
                    kv_cache_idx = i
                elif not layer.yoco_cross:
                    # full attention that generates kv cache
                    cache_layer = self.layers[i]
                    kv_cache = get_kv_cache(cache_layer.attn.attn.layer_name)
                    kv_cache_idx = i
                else:
                    # full attention that reuses kv cache
                    cache_layer = self.layers[kv_cache_idx]
                    kv_cache = get_kv_cache(cache_layer.attn.attn.layer_name)

                hidden_states, ssm_output = layer(
                    hidden_states,
                    positions,
                    kv_cache,
                    attn_metadata,
                    None, # mamba_cache_params
                    ssm_output = ssm_output
                )
            # end_env.record()
            # torch.cuda.synchronize()
            # print('>>> layer', i, 'time', start_env.elapsed_time(end_env))

        hidden_states = self.final_layernorm(hidden_states.to(dtype=self.final_layernorm.weight.dtype))
        return hidden_states


class SambaForCausalLM(nn.Module, HasInnerState, IsHybrid, SupportsV0Only):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        lora_config = vllm_config.lora_config
        quant_config = vllm_config.quant_config
        scheduler_config = vllm_config.scheduler_config
        self.compilation_config = vllm_config.compilation_config
        self.vllm_config = vllm_config
        # Prefix caching is not supported since there are mamba layers in this 
        # mode.
        assert not cache_config.enable_prefix_caching, \
            "Samba currently does not support prefix caching"

        super().__init__()
        self.config = config
        self.model_config = vllm_config.model_config
        self.scheduler_config = scheduler_config
        self.model = SambaModel(
            config, 
            cache_config=cache_config,
            prefix=maybe_prefix(prefix, "model")
        )
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=(
                DEFAULT_VOCAB_PADDING_SIZE
                # We need bigger padding if using lora for kernel
                # compatibility
                if not lora_config else
                lora_config.lora_vocab_padding_size),
            quant_config=quant_config,
        )
        self.embedding_bias = None
        # Used to track and store by the Mamba cache between steps.
        self.mamba_cache: Optional[MambaCacheManager] = None
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size,
                                                logits_as_input=False)
        # self.sampler = Sampler()
        self.sampler = get_sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if self.mamba_cache is None:
            num_mamba_layers = self.config.num_hidden_layers // 2 // self.config.mb_per_layer + 1
            self.mamba_cache = MambaCacheManager(
                self.vllm_config,
                self.lm_head.weight.dtype, num_mamba_layers, *self._get_mamba_cache_shape()
            )
        mamba_cache_params = self.mamba_cache.current_run_tensors(**kwargs)

        attn_metadata = get_forward_context().attn_metadata
        hidden_states = self.model(input_ids, positions,
                                   attn_metadata, mamba_cache_params, 
                                   intermediate_tensors, inputs_embeds)
        return hidden_states

    def _get_mamba_cache_shape(self) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        world_size = get_tensor_model_parallel_world_size()
        hidden_size = self.config.hidden_size
        mamba_expand = self.config.mamba_expand # 2
        mamba_d_conv = self.config.mamba_d_conv # 4
        mamba_d_state = self.config.mamba_d_state # 16
        conv_state_shape = (
            mamba_expand * hidden_size // world_size,
            mamba_d_conv - 1,
        )
        temporal_state_shape = (
            mamba_expand * hidden_size // world_size,
            mamba_d_state,
        )
        return conv_state_shape, temporal_state_shape

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.mamba_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        # If the shape is the same, it means that we have already prune hidden states manually.
        prune_hidden_states = hidden_states.size(0) != sampling_metadata.selected_token_indices.size(0)
        processed_logits = self.logits_processor(
                self.lm_head, 
                hidden_states, 
                sampling_metadata, 
                self.embedding_bias,
                prune_hidden_states=prune_hidden_states
            )
        return processed_logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ):
        weights = {name: weight for name, weight in weights}
        print(f"--------- num of keys: {len(weights.keys())}")
        adjusted_weights = {}
        for name, weight in weights.items():
            if "A_log" in name:
                name = name.replace("A_log", "A")
                weight = -torch.exp(weight.float())
            if "inner_cross_attn." in name:
                name = name.replace("inner_cross_attn.", "")
            adjusted_weights[name] = weight
        adjusted_weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
        for name, loaded_weight in adjusted_weights.items():
            print(name, loaded_weight.shape)

        params_dict = dict(self.named_parameters())
        
        print(f"{adjusted_weights.keys() - params_dict.keys()} not in model")
        print(f"{params_dict.keys() - adjusted_weights.keys()} not in weights")

        loaded_params: Set[str] = set()

        for name, param in self.named_parameters():
            weight = adjusted_weights.get(name, None)
            if weight is not None and weight.shape != param.shape:
                print(f"Shape mismatch: {name} {weight.shape} {param.shape}")
            loaded_params.add(name)
        missing_keys, unexpected_keys = self.load_state_dict(adjusted_weights, strict=False)
        print(f"--------------- missing keys {missing_keys}")
        print("--------------- unexpected keys ---------------")
        for key in unexpected_keys:
            print(key)
            if not key.endswith("bias"):
                print("------- not bias -------")
        # assert missing_keys == ['embedding_bias', 'lm_head.weight',], f"Missing keys: {missing_keys}"
        # assert unexpected_keys == ['lm_head.bias',], f"Unexpected keys: {unexpected_keys}"
        # self.lm_head.weight.data.copy_(adjusted_weights['model.embed_tokens.weight'])
        # self.embedding_bias.data.copy_(adjusted_weights['lm_head.bias'])
        # self.embedding_bias = None
        return loaded_params