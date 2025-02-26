# SPDX-License-Identifier: Apache-2.0
"""Inference-only FalconMamba2 model."""
# Added by the IBM Team, 2024
from typing import Iterable, List, Optional, Set, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.layer import Attention
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_mixer2 import (MambaMixer2,
                                                           extra_groups_for_head_shards)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (DEFAULT_VOCAB_PADDING_SIZE,
                                                                 ParallelLMHead,
                                                                 VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.mamba_cache import (MambaCacheManager,
                                                    MambaCacheParams)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import HasInnerState, IsHybrid, SupportsLoRA, SupportsPP
from .utils import (is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

KVCache = Tuple[torch.Tensor, torch.Tensor]


class FalconMamba2Config(PretrainedConfig):
    """
    This configuration class merges attributes from both `Mamba2Config` 
    and `MHAConfig`. It is used to instantiate a FalconMamba2 model architecture 
    that combines the capabilities of both models.

    Args:
        num_heads (`int`, *optional*, defaults to 128): Number of heads for the
        Mamba2 part of the model.
        head_dim (`int`, *optional*, defaults to 64): Dimension of each head
        in Mamba2.
        vocab_size (`int`, *optional*, defaults to 32768): Vocabulary size 
        of the model.
        hidden_size (`int`, *optional*, defaults to 4096): Dimensionality 
        of the embeddings and hidden states in Mamba2.
        state_size (`int`, *optional*, defaults to 128): Shape of the state space
        latents.
        num_hidden_layers (`int`, *optional*, defaults to 64): Number of hidden 
        layers in Mamba2.
        expand (`int`, *optional*, defaults to 2): Expanding factor used
        in Mamba2.
        conv_kernel (`int`, *optional*, defaults to 4): Convolution kernel size 
        in Mamba2.
        n_groups (`int`, *optional*, defaults to 8): Number of groups 
        for evolution matrices.
        use_bias (`bool`, *optional*, defaults to `False`): Whether to use bias
        in Mamba2 projections.
        use_conv_bias (`bool`, *optional*, defaults to `True`): Whether to use
        bias in the convolution layer in Mamba2.
        hidden_act (`str`, *optional*, defaults to `"silu"`): Non-linear 
        activation function in the decoder.
        initializer_range (`float`, *optional*, defaults to 0.1): Initialization
        range for weights.
        time_step_rank (`Union[int,str]`, *optional*, defaults to `"auto"`):
        Rank of the time-step projection matrix.
        time_step_min (`float`, *optional*, defaults to 0.001): Minimum 
        time step for `dt_proj.bias`.
        time_step_max (`float`, *optional*, defaults to 0.1): Maximum time step 
        for `dt_proj.bias`.
        time_step_floor (`float`, *optional*, defaults to 1e-4): Floor value 
        for the `dt_proj.bias`.
        time_step_limit (`tuple`, *optional*, defaults to `(0.0, inf)`): 
        Time-step limit range.
        rescale_prenorm_residual (`bool`, *optional*, defaults to `False`): 
        Whether to rescale pre-norm residuals.
        use_cache (`bool`, *optional*, defaults to `True`): Whether to use cache
        for the model.
        rms_norm (`bool`, *optional*, defaults to `True`): Whether to use 
        RMS norm.
        chunk_size (`int`, defaults to 256): Chunk size for processing.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`): 
        Whether to tie word embeddings.

        # Parameters from MHAConfig
        hidden_size_mha (`int`): Embedding dimension for MHA part of the model.
        num_heads_mha (`int`): Number of attention heads in MHA part.
        num_key_value_heads (`int`, *optional*): Number of heads for the 
        key-value projections in MHA.
        head_dim_mha (`int`, *optional*): Dimension for heads in MHA.
        rotary_emb_dim (`int`, *optional*, defaults to 0): Dimension for rotary embeddings in MHA.
        rotary_emb_base (`int`, *optional*, defaults to 10000): Base for rotary embeddings.
        softmax_scale (`float`, *optional*): Scaling factor for softmax in MHA.
        causal (`bool`, *optional*, defaults to `False`): Whether attention mechanism is causal in MHA.
        sliding_window (`tuple`, *optional*): Window size for attention in MHA.
        qkv_proj_bias (`bool`, *optional*, defaults to `True`): Whether to use bias in the query, key, value projections.
        out_proj_bias (`bool`, *optional*, defaults to `True`): Whether to use bias in the output projection of MHA.
    """

    model_type = "falconmamba2"

    def __init__(
        self,
        vocab_size=128000,
        tie_word_embeddings=False,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        num_logits_to_keep=1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        max_position_embeddings=262144,
        attention_dropout=0.0,
        attn_layer_indices=None,
        mlp_expansion_factor=8,
        mamba_d_ssm=1024,
        mamba_n_heads=128,
        mamba_d_head="auto",
        mamba_n_groups=1,
        mamba_d_state=256,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_chunk_size=256,
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        mamba_use_mlp=True,
        mamba_norm_before_gate=True,
        mamba_rms_norm=False,
        projectors_bias=False,
        rope_theta=100000.0,
        rope_scaling=None,
        lm_head_multiplier=1.0,
        embedding_multiplier=1.0,
        mlp_multipliers=None,
        key_multiplier=None,
        attention_out_multiplier=None,
        attention_in_multiplier=None,
        ssm_multipliers=None,
        ssm_in_multiplier=None,
        ssm_out_multiplier=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout
        self.attention_bias = False
        self.mlp_bias = False

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps

        self.use_cache = use_cache
        self.num_logits_to_keep = num_logits_to_keep

        self.attn_layer_indices = attn_layer_indices
        self.rope_theta = rope_theta
        self.rope_scaling = None
        self.rope_scaling = rope_scaling
        self.mlp_expansion_factor = mlp_expansion_factor
        self.projectors_bias = projectors_bias
        mamba_intermediate = (
            mamba_expand * hidden_size if mamba_d_ssm is None else mamba_d_ssm
        )

        if mamba_intermediate % mamba_n_heads != 0:
            raise ValueError("mamba_n_heads must divide mamba_expand * hidden_size")

        # for the mamba_v2, must satisfy the following
        if mamba_d_head == "auto":
            mamba_d_head = mamba_intermediate // mamba_n_heads

        if mamba_d_head * mamba_n_heads != mamba_intermediate:
            raise ValueError(
                "The dimensions for the Mamba head state do not match the model intermediate_size"
            )

        self.mamba_d_ssm = mamba_d_ssm
        self.mamba_n_heads = mamba_n_heads
        self.mamba_d_head = mamba_d_head
        self.mamba_n_groups = mamba_n_groups
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_chunk_size = mamba_chunk_size
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias
        self.mamba_use_mlp = mamba_use_mlp
        self.mamba_norm_before_gate = mamba_norm_before_gate
        self.mamba_rms_norm = mamba_rms_norm

        self.lm_head_multiplier = lm_head_multiplier
        self.embedding_multiplier = embedding_multiplier

        if mlp_multipliers is not None:
            self.mlp_multipliers = mlp_multipliers
        else:
            self.mlp_multipliers = [1.0, 1.0]

        if attention_out_multiplier is not None:
            self.attention_out_multiplier = attention_out_multiplier
        else:
            self.attention_out_multiplier = 1.0

        if attention_in_multiplier is not None:
            self.attention_in_multiplier = attention_in_multiplier
        else:
            self.attention_in_multiplier = 1.0

        if key_multiplier is not None:
            self.key_multiplier = key_multiplier
        else:
            self.key_multiplier = 1.0

        if ssm_multipliers is not None:
            self.ssm_multipliers = ssm_multipliers
        else:
            #
            self.ssm_multipliers = [1.0, 1.0, 1.0, 1.0, 1.0]

        if ssm_in_multiplier is not None:
            self.ssm_in_multiplier = ssm_in_multiplier
        else:
            self.ssm_in_multiplier = 1.0

        if ssm_out_multiplier is not None:
            self.ssm_out_multiplier = ssm_out_multiplier
        else:
            self.ssm_out_multiplier = 1.0

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class FalconMamba2MLP(nn.Module):
    def __init__(
        self,
        config: FalconMamba2Config,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
        )
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=bias,
            quant_config=quant_config,
        )
        self.gate_multiplier, self.down_multiplier = config.mlp_multipliers
        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x * self.gate_multiplier)
        x, _ = self.down_proj(x) * self.down_multiplier
        return x


class FalconMamba2SSMDecoderLayer(nn.Module):
    def __init__(
        self,
        config: FalconMamba2Config,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mamba = MambaMixer2(
            hidden_size=config.hidden_size,
            ssm_state_size=config.mamba_d_state,
            conv_kernel_size=config.mamba_d_conv,
            intermediate_size=config.mamba_expand * config.hidden_size,
            use_conv_bias=config.mamba_conv_bias,
            use_bias=config.mamba_proj_bias,
            n_groups=config.mamba_n_groups,
            num_heads=config.mamba_n_heads,
            head_dim=config.mamba_d_head,
            rms_norm_eps=config.rms_norm_eps,
            activation=config.hidden_act,
            chunk_size=config.mamba_chunk_size,
            quant_config=quant_config,
        )
        self.zxbcdt_multipliers = config.ssm_multipliers
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _init_mup_vector(self):
        vector_shape = (
            2 * self.d_ssm + 2 * self.groups_time_state_size + self.config.num_heads
        )
        mup_vector = torch.ones(1, 1, vector_shape)

        mup_vector[:, :, : self.d_ssm] *= self.zxbcdt_multipliers[0]

        mup_vector[:, :, self.d_ssm : 2 * self.d_ssm] *= self.zxbcdt_multipliers[1]
        mup_vector[
            :, :, 2 * self.d_ssm : 2 * self.d_ssm + self.groups_time_state_size
        ] *= self.zxbcdt_multipliers[2]
        mup_vector[
            :,
            :,
            2 * self.d_ssm
            + self.groups_time_state_size : 2 * self.d_ssm
            + 2 * self.groups_time_state_size,
        ] *= self.zxbcdt_multipliers[3]

        mup_vector[
            :, :, 2 * self.d_ssm + 2 * self.groups_time_state_size :
        ] *= self.zxbcdt_multipliers[4]

        self.register_buffer("mup_vector", mup_vector, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        mamba_cache_params: MambaCacheParams,
        sequence_idx: Optional[torch.Tensor] = None,
        ssm_in_multiplier: float = 1.0,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.mamba(
            hidden_states,
            attn_metadata,
            mamba_cache_params,
            sequence_idx,
            ssm_in_multiplier,
        )
        return hidden_states, residual


class FalconMamba2AttentionDecoderLayer(nn.Module):
    def __init__(
        self,
        config: FalconMamba2Config,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
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
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        if hasattr(config, "partial_rotary_factor"):
            rotary_dim = self.head_dim * config.partial_rotary_factor
        elif hasattr(config, "attn_rotary_emb"):
            rotary_dim = config.attn_rotary_emb  # for backward compatibility
        else:
            rotary_dim = self.head_dim  # default

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=rotary_dim,
            max_position=max_position_embeddings,
            rope_scaling=rope_scaling,
            base=rope_theta,
            is_neox_style=True,
            dtype=torch.get_default_dtype(),  # see impl of get_rope
        )

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )
        self.key_multiplier = config.key_multiplier
        self.attn_out_multiplier = config.attention_out_multiplier
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        k = k * self.key_multiplier
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output) * self.attn_out_multiplier
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
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attention(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        return hidden_states, residual


class FalconMamba2ParallelHybrid(nn.Module):
    """
    A hybrid decoder layer for FalconMamba2 where the input is processed
    in parallel through both the self-attention branch and the SSM (Mamba)
    branch. Their outputs are then summed to produce the final hidden state.

    This layer uses:
      - FalconMamba2AttentionDecoderLayer for the multi-head self-attention branch.
      - FalconMamba2SSMDecoderLayer for the state-space (Mamba) branch.
    """

    def __init__(
        self,
        config: FalconMamba2Config,  # FalconMamba2Config instance
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Instantiate the attention branch
        self.attn_layer = FalconMamba2AttentionDecoderLayer(
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        # Instantiate the SSM branch
        self.ssm_layer = FalconMamba2SSMDecoderLayer(
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.ssm_out_multiplier = config.ssm_out_multiplier
        self.ssm_in_multiplier = config.ssm_in_multiplier
        self.attention_in_multiplier = config.attention_in_multiplier

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        mamba_cache_params: MambaCacheParams,
        sequence_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Process input through the attention branch.
        # FalconMamba2AttentionDecoderLayer expects positions, hidden_states,
        # kv_cache, attn_metadata, and residual.
        attn_hidden, residuals = self.attn_layer(
            positions=positions,
            hidden_states=hidden_states * self.attention_in_multiplier,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            residual=residual,
            **kwargs,
        )

        # Process input through the SSM branch.
        # FalconMamba2SSMDecoderLayer expects hidden_states, attn_metadata, 
        # residual, mamba_cache_params, and sequence_idx.
        ssm_hidden, residuals = self.ssm_layer(
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            residual=residual,
            mamba_cache_params=mamba_cache_params,
            sequence_idx=sequence_idx,
            ssm_in_multiplier=self.ssm_in_multiplier,
            **kwargs,
        )

        # Sum the outputs from both branches.
        # We assume both branches produce outputs of the same
        # dimensionality (config.hidden_size).
        hybrid_hidden = attn_hidden + ssm_hidden * self.ssm_out_multiplier
        # For the residual, resi.
        # Here we simply return the residual from the attention branch.
        hybrid_res = residuals

        return hybrid_hidden, hybrid_res


# ALL_DECODER_LAYER_TYPES = {
#     "attention": FalconMamba2AttentionDecoderLayer,
#     "mamba": FalconMamba2SSMDecoderLayer,
#     "parallel_hybrid_falcon": FalconMamba2ParallelHybrid,
# }


class FalconMamba2Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = (
            FalconMamba2Config()
        )  # no HF integration, initialize falconMamba2Config locally
        # config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        lora_vocab = (
            (lora_config.lora_extra_vocab_size * (lora_config.max_loras or 1))
            if lora_config
            else 0
        )
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size, config.hidden_size, org_num_embeddings=config.vocab_size
        )
        self.embedding_multiplier = config.embedding_multiplier

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            layer_class = FalconMamba2ParallelHybrid
            return layer_class(
                config,
                layer_idx,
                cache_config,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        mamba_cache_params: MambaCacheParams,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # pass a sequence index tensor, that is required for
        # proper continuous batching computation including
        # chunked prefill
        seq_idx = None
        if attn_metadata.num_prefills > 0:
            seq_idx = torch.zeros_like(input_ids, dtype=torch.int32)
            for i, (srt, end) in enumerate(
                zip(attn_metadata.query_start_loc, attn_metadata.query_start_loc[1:])
            ):
                seq_idx[srt:end] = i
            seq_idx.unsqueeze_(0)

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds * self.embedding_multiplier
            else:
                hidden_states = (
                    self.get_input_embeddings(input_ids) * self.embedding_multiplier
                )
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            kv_cache = kv_caches[i]
            layer_mamba_cache_params = mamba_cache_params.at_layer_idx(i)

            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                residual=residual,
                mamba_cache_params=layer_mamba_cache_params,
                sequence_idx=seq_idx,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.final_layernorm(hidden_states, residual)
        return hidden_states


class FalconMamba2ForCausalLM(
    nn.Module, HasInnerState, SupportsLoRA, SupportsPP, IsHybrid
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["up_proj", "down_proj"],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "embed_tokens",
        "lm_head",
        "in_proj",
        "out_proj",
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # config = vllm_config.model_config.hf_config
        config = FalconMamba2Config()
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        lora_config = vllm_config.lora_config
        scheduler_config = vllm_config.scheduler_config
        assert (
            not cache_config.enable_prefix_caching
        ), "Bamba currently does not support prefix caching"

        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = FalconMamba2Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
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
        self.lm_head_multiplier = config.lm_head_multiplier
        # Used to track and store by the Mamba cache between steps.
        self.mamba_cache: Optional[MambaCacheManager] = None

        self.logits_processor = LogitsProcessor(
            self.unpadded_vocab_size, config.vocab_size
        )
        self.sampler = get_sampler()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if self.mamba_cache is None:

            # num_mamba_layers = self.model_config.get_num_layers_by_block_type(
            #     self.vllm_config.parallel_config, LayerBlockType.mamba)

            self.mamba_cache = MambaCacheManager(
                self.vllm_config,
                self.lm_head.weight.dtype,
                self.config.num_hidden_layers,
                *self._get_mamba_cache_shape(),
            )
        mamba_cache_params = self.mamba_cache.current_run_tensors(**kwargs)
        hidden_states = self.model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            mamba_cache_params,
            intermediate_tensors,
            inputs_embeds,
        )

        return hidden_states

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.mamba_cache.copy_inputs_before_cuda_graphs(input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def _get_mamba_cache_shape(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        world_size = get_tensor_model_parallel_world_size()
        hidden_size = self.config.hidden_size

        conv_state_shape, temporal_state_shape = None, None

        intermediate_size = self.config.mamba_expand * hidden_size

        # if n_groups is not divisible by world_size, need to extend the shards
        # to ensure all groups needed by a head is sharded along with it
        n_groups = self.config.mamba_n_groups + extra_groups_for_head_shards(
            self.config.mamba_n_groups, world_size
        )

        # - heads and n_groups are TP-ed
        conv_dim = intermediate_size + 2 * n_groups * self.config.mamba_d_state
        conv_state_shape = (divide(conv_dim, world_size), self.config.mamba_d_conv - 1)

        # These are not TP-ed as they depend on A, dt_bias, D
        # - they are typically small
        #   e.g., (h_heads, d_head, d_state) = (128, 64, 128)
        temporal_state_shape = (
            divide(self.config.mamba_n_heads, world_size),
            self.config.mamba_d_head,
            self.config.mamba_d_state,
        )
        return conv_state_shape, temporal_state_shape

    def compute_logits(
        self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> Optional[torch.Tensor]:
        logits = (
            self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
            * self.lm_head_multiplier
        )
        return logits

    def sample(
        self, logits: Optional[torch.Tensor], sampling_metadata: SamplingMetadata
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
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

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
