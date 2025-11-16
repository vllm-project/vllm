# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch Zamba2 model implementation for vLLM.

This module implements the Zamba2 architecture from
https://arxiv.org/abs/2411.15242, which combines Mamba and Transformer
architectures in a hybrid model optimized for efficient sequence modeling. The
model alternates between state space model layers and attention-based layers.
"""

from collections.abc import Iterable
from itertools import cycle
from typing import Any

import torch
from torch import nn
from transformers import Zamba2Config

from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .interfaces import HasInnerState, IsHybrid, SupportsMambaPrefixCaching
from .utils import AutoWeightsLoader, WeightsMapper, maybe_prefix


class Zamba2LoRA(nn.Module):
    """LoRA layer for the Zamba2 model.

    Implements a LoRA layer that is used in shared attention and gated MLP
    blocks.
    """

    def __init__(
        self,
        input_dim: int,
        rank: int,
        output_dim: int | list[int],
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        """Initialize the attention layer.

        Args:
            input_dim: input dimension
            rank: LoRA rank
            output_dim: output dimension
            quant_config: Configuration for model quantization
        """
        super().__init__()

        self.A = ColumnParallelLinear(
            input_dim,
            rank,
            bias=False,
            quant_config=quant_config,
            gather_output=True,
            prefix=f"{prefix}.A",
        )

        if isinstance(output_dim, list):
            B_class = MergedColumnParallelLinear
        else:
            B_class = ColumnParallelLinear
        self.B = B_class(rank, output_dim, bias=False, quant_config=quant_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        lora_output, _ = self.A(hidden_states)
        lora_output, _ = self.B(lora_output)
        return lora_output


class Zamba2Attention(nn.Module):
    """Multi-head attention mechanism for the Zamba2 model.

    Implements attention with parallel computation, QKV projections, optional
    adapters and rotary position embeddings. The attention is computed across
    distributed blocks for efficient processing.
    """

    def __init__(
        self,
        config: Zamba2Config,
        bare_block_idx: int,
        num_hybrid_layers: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        """Initialize the attention layer.

        Args:
            config: The Zamba2 model configuration
            bare_block_idx: Index of the bare attention block
            num_hybrid_layers: Total number of hybrid layers
            cache_config: Configuration for key-value caching
            quant_config: Configuration for model quantization
            prefix: Optional prefix for parameter names
        """
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.config = config
        self.num_hybrid_layers = num_hybrid_layers
        self.rope_theta = config.rope_theta

        self.attention_hidden_size = config.attention_hidden_size
        self.total_num_attention_heads = config.num_attention_heads
        assert self.total_num_attention_heads % tp_size == 0
        self.num_attention_heads = config.num_attention_heads // tp_size
        self.attention_head_dim = config.attention_head_dim
        self.qkv_size = self.attention_hidden_size // tp_size
        self.scale = (self.attention_head_dim / 2) ** -0.5

        if (
            self.attention_head_dim * self.total_num_attention_heads
        ) != self.attention_hidden_size:
            raise ValueError(
                f"attention_hidden_size must be divisible by"
                f" num_attention_heads"
                f" (got `attention_hidden_size`: {self.attention_hidden_size}"
                f" and `num_heads`: {self.num_attention_heads})."
            )

        self.qkv_proj = QKVParallelLinear(
            self.attention_hidden_size,
            self.attention_head_dim,
            self.total_num_attention_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.attention_hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Even though in Zamba2 weights are shared between attention layers, KV
        # cache is unique for every attention layer. Hence, we need to define
        # separate Attention objects, because in recent vLLM KV cache tensors
        # are tied to specific Attention objects.

        # Initialize attention blocks with proper indexing
        self.dpa_list = nn.ModuleList([])
        j = (
            bare_block_idx
            * (self.num_hybrid_layers + config.num_mem_blocks - 1)
            // config.num_mem_blocks
        )
        for block_idx in range(self.num_hybrid_layers):
            if block_idx % config.num_mem_blocks == bare_block_idx:
                dpa = Attention(
                    self.num_attention_heads,
                    self.attention_head_dim,
                    self.scale,
                    cache_config=cache_config,
                    prefix=f"{prefix}.attn.{j}",
                )
                j += 1
            else:
                dpa = nn.Identity()
            self.dpa_list.append(dpa)

        # Initialize adapter layers if enabled
        if config.use_shared_attention_adapter:
            self.linear_q_adapter_list = nn.ModuleList([])
            self.linear_k_adapter_list = nn.ModuleList([])
            self.linear_v_adapter_list = nn.ModuleList([])

            for block_idx in range(self.num_hybrid_layers):
                if block_idx % config.num_mem_blocks == bare_block_idx:
                    linear_q_adapter = Zamba2LoRA(
                        self.attention_hidden_size,
                        config.adapter_rank,
                        self.attention_hidden_size,
                        quant_config=quant_config,
                        prefix=f"{prefix}.linear_q_adapter",
                    )
                    linear_k_adapter = Zamba2LoRA(
                        self.attention_hidden_size,
                        config.adapter_rank,
                        self.attention_hidden_size,
                        quant_config=quant_config,
                        prefix=f"{prefix}.linear_k_adapter",
                    )
                    linear_v_adapter = Zamba2LoRA(
                        self.attention_hidden_size,
                        config.adapter_rank,
                        self.attention_hidden_size,
                        quant_config=quant_config,
                        prefix=f"{prefix}.linear_v_adapter",
                    )
                else:
                    linear_q_adapter = nn.Identity()
                    linear_k_adapter = nn.Identity()
                    linear_v_adapter = nn.Identity()

                self.linear_q_adapter_list.append(linear_q_adapter)
                self.linear_k_adapter_list.append(linear_k_adapter)
                self.linear_v_adapter_list.append(linear_v_adapter)

        if config.use_mem_rope:
            self.rotary_emb = get_rope(
                head_size=self.attention_head_dim,
                rotary_dim=self.attention_head_dim,
                max_position=config.max_position_embeddings,
                base=self.rope_theta,
                rope_scaling=None,
                is_neox_style=True,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_idx: int,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the attention layer.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            position_ids: Position IDs for positional embeddings
            block_idx: Current shared transformer block index

        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        qkv, _ = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv.split([self.qkv_size] * 3, dim=-1)

        if self.config.use_shared_attention_adapter:
            # Apply adapter transformations to Q, K, V if enabled
            q_adapter = self.linear_q_adapter_list[block_idx]
            assert not isinstance(q_adapter, nn.Identity)
            q_lora_output = q_adapter(hidden_states)
            query_states = query_states + q_lora_output

            k_adapter = self.linear_k_adapter_list[block_idx]
            assert not isinstance(k_adapter, nn.Identity)
            k_lora_output = k_adapter(hidden_states)
            key_states = key_states + k_lora_output

            v_adapter = self.linear_v_adapter_list[block_idx]
            assert not isinstance(v_adapter, nn.Identity)
            v_lora_output = v_adapter(hidden_states)
            value_states = value_states + v_lora_output

        if self.config.use_mem_rope:
            query_states, key_states = self.rotary_emb(
                position_ids, query_states, key_states
            )

        y = self.dpa_list[block_idx](query_states, key_states, value_states)
        y, _ = self.o_proj(y)
        return y


class Zamba2MLP(nn.Module):
    """Feed-forward MLP layer for the Zamba2 model.

    Implements a gated feed-forward network that projects inputs to a larger
    intermediate size, applies GELU activation with gating, then projects back
    to the original size. Includes optional adapter layers for model adaptation.
    """

    def __init__(
        self,
        config: Zamba2Config,
        bare_block_idx: int,
        num_hybrid_layers: dict[int, int],
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        """Initialize the MLP layer.

        Args:
            config: The Zamba2 model configuration
            bare_block_idx: Index of the bare block in the model
            num_hybrid_layers: Total number of hybrid layers
            quant_config: Configuration for model quantization
        """
        super().__init__()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_hybrid_layers = num_hybrid_layers
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Main projection layers with gating
        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size,
            2 * [self.intermediate_size],  # 2x for gate and input projections
            bias=self.config.add_bias_linear,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )

        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=self.config.add_bias_linear,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

        # Only allow GELU activations
        if config.hidden_act != "gelu":
            raise ValueError(
                f"Only GELU activation is supported "
                f"(got `hidden_act`: {config.hidden_act})"
            )
        self.act_fn = GeluAndMul()

        # Initialize adapter layers
        self.gate_up_proj_adapter_list = nn.ModuleList([])
        for block_idx in range(self.num_hybrid_layers):
            if block_idx % config.num_mem_blocks == bare_block_idx:
                gate_up_proj_adapter = Zamba2LoRA(
                    config.hidden_size,
                    config.adapter_rank,
                    2 * [self.intermediate_size],
                    quant_config,
                )
            else:
                gate_up_proj_adapter = nn.Identity()
            self.gate_up_proj_adapter_list.append(gate_up_proj_adapter)

    def forward(self, hidden_states: torch.Tensor, block_idx: int) -> torch.Tensor:
        """Forward pass through the MLP layer.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            block_idx: Current shared transformer block index

        Returns:
            Output tensor [batch_size, seq_len, hidden_size] after applying
            gated feed-forward transformation
        """
        # Project input to intermediate size with gating
        gate_up_states, _ = self.gate_up_proj(hidden_states)

        # Apply adapter transformation if present
        adapter = self.gate_up_proj_adapter_list[block_idx]
        assert not isinstance(adapter, nn.Identity)
        lora_output = adapter(hidden_states)
        gate_up_states = gate_up_states + lora_output

        # Apply GELU activation with gating
        hidden_states = self.act_fn(gate_up_states)

        # Project back to hidden size
        output, _ = self.down_proj(hidden_states)
        return output


class Zamba2AttentionDecoderLayer(nn.Module):
    """Single decoder layer combining attention and feed-forward networks.

    This layer implements a standard transformer block with:
    - Input layer normalization
    - Multi-head self-attention
    - Pre-feed-forward layer normalization
    - Feed-forward network (MLP)
    """

    def __init__(
        self,
        config: Zamba2Config,
        bare_block_idx: int,
        num_hybrid_layers: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        """Initialize the decoder layer.

        Args:
            config: The Zamba2 model configuration
            bare_block_idx: Index of the bare block
            num_hybrid_layers: Total number of hybrid layers
            cache_config: Configuration for key-value caching
            quant_config: Configuration for model quantization
            prefix: Optional prefix for parameter names
        """
        super().__init__()

        # Initialize attention sublayer
        self.self_attn = Zamba2Attention(
            config,
            bare_block_idx=bare_block_idx,
            num_hybrid_layers=num_hybrid_layers,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )

        # Initialize feed-forward sublayer
        self.feed_forward = Zamba2MLP(
            config,
            bare_block_idx=bare_block_idx,
            num_hybrid_layers=num_hybrid_layers,
            quant_config=quant_config,
            prefix=f"{prefix}.feed_forward",
        )

        # Initialize layer normalizations
        # Input normalization operates on concatenated states
        self.input_layernorm = RMSNorm(2 * config.hidden_size, eps=config.rms_norm_eps)
        # Pre-FF normalization operates on attention output
        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
        block_idx: int,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the decoder layer.

        Args:
            hidden_states: Input tensor from previous layer
            original_hidden_states: Original input tensor for residual
                connection
            block_idx: Current shared transformer block index
            positions: IDs for positional embeddings

        Returns:
            Transformed hidden states after attention and feed-forward
        """

        # The argument original_hidden_states is concatenated with hidden_states
        # (which is the output of the previous (mamba) layer).
        # The concatenated tensor is then used as input of the pre-attention
        # RMSNorm (see fig. 2 in https://arxiv.org/pdf/2405.16712).
        hidden_states = torch.concatenate(
            [hidden_states, original_hidden_states], dim=-1
        )

        # Layer norm before attention
        hidden_states = self.input_layernorm(hidden_states)

        # Self attention
        hidden_states = self.self_attn(
            hidden_states,
            position_ids=positions,
            block_idx=block_idx,
        )

        # Layer norm before feed-forward
        hidden_states = self.pre_ff_layernorm(hidden_states)

        # Feed-forward network
        hidden_states = self.feed_forward(hidden_states, block_idx=block_idx)

        return hidden_states


class Zamba2MambaDecoderLayer(nn.Module):
    """Single Mamba decoder layer with normalization.

    This implements a  Mamba block. It includes input normalization
    and can process sequences using either chunked or full
    computation depending on configuration.
    """

    def __init__(
        self,
        config: Zamba2Config,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        """Initialize the Mamba decoder layer.

        Args:
            config: The Zamba2 model configuration
            quant_config: Configuration for model quantization
        """
        super().__init__()

        # Initialize Mamba mixer with expanded intermediate size
        intermediate_size = config.mamba_expand * config.hidden_size
        self.mamba = MambaMixer2(
            hidden_size=config.hidden_size,
            ssm_state_size=config.mamba_d_state,
            conv_kernel_size=config.mamba_d_conv,
            intermediate_size=intermediate_size,
            use_conv_bias=config.use_conv_bias,
            use_bias=config.add_bias_linear,
            n_groups=config.mamba_ngroups,
            num_heads=config.n_mamba_heads,
            head_dim=intermediate_size // config.n_mamba_heads,
            rms_norm_eps=config.rms_norm_eps,
            activation="silu",
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mixer",
        )

        # Input normalization
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        transformer_hidden_states: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        original_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the Mamba decoder layer.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            transformer_hidden_states: Optional output from transformer path
                Added to input if provided (used in hybrid architecture)
            positions: Optional position IDs (unused in Mamba)
            original_hidden_states: Optional original inputs (unused in Mamba)

        Returns:
            Transformed hidden states with residual connection applied
        """
        # Store input for residual connection
        residual = hidden_states

        # `transformer_hidden_states` is the output from shared
        # transformer + linear layer (see fig. 2 in
        # https://arxiv.org/pdf/2405.16712).
        # `transformer_hidden_states` is then added to the input to the mamba
        # layer below (as described in eq. (6) of
        # https://arxiv.org/pdf/2405.16712).
        if transformer_hidden_states is not None:
            hidden_states = hidden_states + transformer_hidden_states

        # Apply input normalization
        hidden_states = self.input_layernorm(hidden_states)

        # Process through Mamba mixer
        output = torch.empty_like(hidden_states)
        self.mamba(
            hidden_states,
            output,
        )

        # residual connection after mamba
        hidden_states = residual + output

        return hidden_states


class Zamba2HybridLayer(nn.Module):
    """Hybrid layer combining Transformer and Mamba architectures.

    This layer implements the hybrid architecture described in the Zamba paper,
    where a shared transformer pathway processes input in parallel with a Mamba
    pathway. The transformer output is projected and added to the Mamba input
    for enhanced representation learning.
    """

    def __init__(
        self,
        shared_transformer: Zamba2AttentionDecoderLayer,
        config: Zamba2Config,
        block_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        """Initialize the hybrid layer.

        Args:
            shared_transformer: Transformer decoder layer for attention pathway
        """
        super().__init__()
        self.block_idx = block_idx
        self.shared_transformer = shared_transformer
        self.linear = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.linear",
        )
        self.mamba_decoder = Zamba2MambaDecoderLayer(
            config,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the hybrid layer.

        Processes input through parallel transformer and Mamba paths:
        1. Transformer path processes input with attention
        2. Transformer output is projected to match hidden size
        3. Projected output is added to Mamba path input
        4. Final output combines both paths' representations

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            original_hidden_states: Original input for transformer residual
                connection
            positions: Position IDs for positional embeddings

        Returns:
            Output tensor combining transformer and Mamba representations
        """
        # Process through transformer pathway
        transformer_hidden_states = self.shared_transformer(
            hidden_states,
            original_hidden_states=original_hidden_states,
            block_idx=self.block_idx,
            positions=positions,
        )

        # Project transformer output
        transformer_hidden_states, _ = self.linear(transformer_hidden_states)

        # Process through Mamba pathway with transformer injection
        layer_outputs = self.mamba_decoder(
            hidden_states,
            transformer_hidden_states=transformer_hidden_states,
        )

        return layer_outputs


@support_torch_compile
class Zamba2Model(nn.Module):
    """Core Zamba2 model combining transformer and Mamba architectures.

    The model processes input through a sequence of hybrid and Mamba-only
    layers, using token embeddings and final layer normalization.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        """Initialize the Zamba2 model.

        Args:
            vllm_config: Configuration object containing model, cache,
                quantization and LoRA settings
            prefix: Optional prefix for parameter names in state dict
        """
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        is_lora_enabled = bool(lora_config)
        assert not is_lora_enabled

        self.config = config

        self.vocab_size = config.vocab_size

        # Initialize token embeddings
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        # Map hybrid layer indices to block indices
        layer2block_map = {
            layer_idx: block_idx
            for block_idx, layer_idx in enumerate(config.hybrid_layer_ids)
        }

        # Create cyclic iterator of transformer blocks
        blocks = cycle(
            [
                Zamba2AttentionDecoderLayer(
                    config,
                    bare_block_idx=idx,
                    num_hybrid_layers=len(layer2block_map),
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}",
                )
                for idx in range(config.num_mem_blocks)
            ]
        )

        # Initialize layers according to block type configuration
        layers = []
        for layer_idx, layer_type in enumerate(config.layers_block_type):
            # tdoublep: avoid layers getting same index
            # somewhat hacky but correct (I think)
            prefix = str(len(layer2block_map) + layer_idx)
            if layer_type == "hybrid":
                block = next(blocks)
                block_idx = layer2block_map[layer_idx]
                layers.append(
                    Zamba2HybridLayer(
                        block,
                        config,
                        block_idx,
                        model_config=model_config,
                        cache_config=cache_config,
                        quant_config=quant_config,
                        prefix=prefix,
                    )
                )
            else:
                layers.append(
                    Zamba2MambaDecoderLayer(
                        config,
                        model_config=model_config,
                        cache_config=cache_config,
                        quant_config=quant_config,
                        prefix=prefix,
                    )
                )
        self.layers = nn.ModuleList(layers)

        # Final layer normalization
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings.

        Args:
            input_ids: Tensor of input token IDs

        Returns:
            Embedded representation of the input tokens
        """
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs
            positions: Position IDs for embeddings
            inputs_embeds: Optional pre-computed input embeddings

        Returns:
            Either final hidden states or intermediate tensors for pipeline
            parallelism
        """
        # Handle pipeline parallelism for first rank
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = inputs_embeds

        # Process through layers
        original_hidden_states = torch.clone(hidden_states)
        for layer_idx, layer in enumerate(self.layers):
            layer_outputs = layer(
                hidden_states,
                original_hidden_states=original_hidden_states,
                positions=positions,
            )
            hidden_states = layer_outputs

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for chkpt_weight_name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in chkpt_weight_name:
                    continue
                chkpt_weight_name = chkpt_weight_name.replace(weight_name, param_name)
                param = params_dict[chkpt_weight_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if chkpt_weight_name not in params_dict:
                    continue
                param = params_dict[chkpt_weight_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(chkpt_weight_name)
        return loaded_params


class Zamba2ForCausalLM(nn.Module, HasInnerState, IsHybrid, SupportsMambaPrefixCaching):
    """Zamba2 model with causal language modeling head.

    This class wraps the core Zamba2 model and adds:
    - A language modeling head for next token prediction
    - Mamba state caching functionality
    - Support for model parallelism and quantization
    - Sampling capabilities for text generation
    """

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            "A_log": "A",
            "0.weight": "A.weight",
            "1.weight": "B.weight",
        }
    )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.mamba2_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        """Calculate shapes for Mamba's convolutional and state caches.

        Args:
            vllm_config: vLLM config

        Returns:
            Tuple containing:
            - conv_state_shape: Shape for convolutional state cache
            - temporal_state_shape: Shape for state space model cache
        """

        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        intermediate_size = hf_config.mamba_expand * hf_config.hidden_size

        return MambaStateShapeCalculator.mamba2_state_shape(
            intermediate_size=intermediate_size,
            tp_world_size=parallel_config.tensor_parallel_size,
            n_groups=hf_config.mamba_ngroups,
            num_heads=hf_config.n_mamba_heads,
            head_dim=hf_config.mamba_headdim,
            state_size=hf_config.mamba_d_state,
            conv_kernel=hf_config.mamba_d_conv,
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        """Initialize the Zamba2 model for causal language modeling.

        Args:
            vllm_config: Configuration containing model, cache, quantization,
                        LoRA and scheduler settings
            prefix: Optional prefix for parameter names

        Raises:
            AssertionError: If prefix caching is enabled
                (not supported by Mamba)
        """
        config = vllm_config.model_config.hf_config

        scheduler_config = vllm_config.scheduler_config

        super().__init__()
        self.config = config
        self.vllm_config = vllm_config
        self.scheduler_config = scheduler_config
        self.model_config = vllm_config.model_config

        # Initialize core model
        self.model = Zamba2Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        # Initialize language modeling head
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        # Tie weights with input embeddings if using same dimensions
        self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        # Initialize logits processing and sampling
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings.
        Args:
            input_ids: Tensor of input token IDs
        Returns:
            Embedded representation of the input tokens
        """
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs
            positions: Position IDs for embeddings
            inputs_embeds: Optional pre-computed input embeddings
            **kwargs: Additional arguments passed to cache manager

        Returns:
            Output hidden states
        """
        # Forward pass through model
        hidden_states = self.model(
            input_ids,
            positions,
            inputs_embeds,
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute logits for next token prediction.

        Args:
            hidden_states: Hidden states from model forward pass

        Returns:
            Logits for next token prediction
        """
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
