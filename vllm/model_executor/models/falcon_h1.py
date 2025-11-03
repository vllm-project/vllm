# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only FalconH1 model."""

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn
from transformers import FalconH1Config

from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
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
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsLoRA,
    SupportsMambaPrefixCaching,
    SupportsPP,
)
from .utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


class FalconH1MLP(nn.Module):
    def __init__(
        self,
        config: FalconH1Config,
        quant_config: QuantizationConfig | None = None,
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
        self.tp_size = get_tensor_model_parallel_world_size()
        self.intermediate_size = config.intermediate_size
        self.gate_multiplier, self.down_multiplier = config.mlp_multipliers
        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x[:, : self.intermediate_size // self.tp_size] *= self.gate_multiplier
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        x = x * self.down_multiplier
        return x


class FalconH1SSMDecoderLayer(nn.Module):
    def __init__(
        self,
        config: FalconH1Config,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()

        self.d_ssm = (
            int(config.mamba_expand * config.hidden_size)
            if config.mamba_d_ssm is None
            else config.mamba_d_ssm
        )

        self.mamba = MambaMixer2(
            hidden_size=config.hidden_size,
            ssm_state_size=config.mamba_d_state,
            conv_kernel_size=config.mamba_d_conv,
            intermediate_size=self.d_ssm,
            use_conv_bias=config.mamba_conv_bias,
            use_bias=config.mamba_proj_bias,
            n_groups=config.mamba_n_groups,
            num_heads=config.mamba_n_heads,
            head_dim=config.mamba_d_head,
            rms_norm_eps=config.rms_norm_eps,
            activation=config.hidden_act,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            use_rms_norm=config.mamba_rms_norm,
            prefix=f"{prefix}.mixer",
        )
        # n_groups is overridden later by `MambaMixer2`
        self.groups_time_state_size = self.mamba.n_groups * config.mamba_d_state
        self.zxbcdt_multipliers = config.ssm_multipliers
        self._init_mup_vector()

    def _init_mup_vector(self):
        """
        Non learnable per-block scaling vector composed of element-wise
        multipliersapplied to each separate contiguous block of the output
        of the linear projection (in_proj) before further processing
        (gating, convolution, SSM):

            - Z block:  [0 : d_ssm]                      → zxbcdt_multipliers[0]
            - X block:  [d_ssm : 2 * d_ssm]              → zxbcdt_multipliers[1]
            - B block:  [2 * d_ssm : 2 * d_ssm + G * S]  → zxbcdt_multipliers[2]
            - C block:  [2 * d_ssm + G * S : 2 * d_ssm + 2 * G * S]
                        → zxbcdt_multipliers[3]
            - dt block: [2 * d_ssm + 2 * G * S : end]    → zxbcdt_multipliers[4]

        where:
            - d_ssm:     Dimension of state-space model latent
            - G:         Number of groups (n_groups)
            - S:         SSM state size per group
            - All indices are divided by tp_size to support tensor parallelism
        """
        vector_shape = (
            2 * self.d_ssm + 2 * self.groups_time_state_size + self.config.mamba_n_heads
        ) // self.tp_size
        mup_vector = torch.ones(1, vector_shape)
        # Z vector 0 -> d_ssm
        mup_vector[:, : self.d_ssm // self.tp_size] *= self.zxbcdt_multipliers[0]
        # X vector d_ssm -> 2 * d_ssm
        mup_vector[
            :, (self.d_ssm // self.tp_size) : (2 * self.d_ssm // self.tp_size)
        ] *= self.zxbcdt_multipliers[1]
        # B vector 2 * d_ssm -> 2 * d_ssm + (n_group * d_state)
        mup_vector[
            :,
            (2 * self.d_ssm) // self.tp_size : (
                2 * self.d_ssm + self.groups_time_state_size
            )
            // self.tp_size,
        ] *= self.zxbcdt_multipliers[2]
        # C vector 2 * d_ssm + (n_group * d_state)
        # -> 2 * d_ssm + 2 * (n_group * d_state)
        mup_vector[
            :,
            (2 * self.d_ssm + self.groups_time_state_size) // self.tp_size : (
                2 * self.d_ssm + 2 * self.groups_time_state_size
            )
            // self.tp_size,
        ] *= self.zxbcdt_multipliers[3]
        # dt vector 2 * d_ssm + 2 * (n_group * d_state)
        # -> 2 * d_ssm + 2 * (n_group * d_state) + n_heads
        mup_vector[
            :,
            (2 * self.d_ssm + 2 * self.groups_time_state_size) // self.tp_size :,
        ] *= self.zxbcdt_multipliers[4]

        self.register_buffer("mup_vector", mup_vector, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ):
        output = torch.empty_like(hidden_states)
        self.mamba(
            hidden_states,
            output,
            mup_vector=self.mup_vector,
        )
        return output, residual


class FalconH1AttentionDecoderLayer(nn.Module):
    def __init__(
        self,
        config: FalconH1Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        rope_theta = getattr(config, "rope_theta", 1e11)
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
        self.head_dim = (
            config.hidden_size // self.total_num_heads
            if getattr(config, "head_dim", None) is None
            else config.head_dim
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
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
            dtype=None,  # see impl of get_rope
        )

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
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

    def self_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        k = k * self.key_multiplier

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ):
        hidden_states = self.self_attention(
            positions=positions,
            hidden_states=hidden_states,
        )
        return hidden_states, residual


class FalconH1ParallelHybrid(nn.Module):
    """
    A hybrid decoder layer for FalconH1 where the input is processed
    in parallel through both the self-attention branch and the SSM (Mamba)
    branch. Their outputs are then summed to produce the final hidden state.

    This layer uses:
      - FalconH1AttentionDecoderLayer for the multi-head self-attention branch.
      - FalconH1SSMDecoderLayer for the state-space (Mamba) branch.
    """

    def __init__(
        self,
        config: FalconH1Config,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # Instantiate the attention branch
        self.self_attn = FalconH1AttentionDecoderLayer(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )

        # In V1 all attention/ssm layers must have
        # different index in prefix
        ssm_layer_idx = config.num_hidden_layers + layer_idx
        ssm_prefix = prefix.split(".")[0] + f".{ssm_layer_idx}"

        # Instantiate the SSM branch
        self.mamba = FalconH1SSMDecoderLayer(
            config=config,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=ssm_prefix,
        )
        self.ssm_out_multiplier = config.ssm_out_multiplier
        self.ssm_in_multiplier = config.ssm_in_multiplier

        self.attention_in_multiplier = config.attention_in_multiplier
        self.attn_out_multiplier = config.attention_out_multiplier

        self.feed_forward = FalconH1MLP(config)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Process input through the attention branch.
        # FalconH1AttentionDecoderLayer expects positions, hidden_states,
        # kv_cache, attn_metadata, and residual.
        attn_hidden, _ = self.self_attn(
            positions=positions,
            hidden_states=hidden_states * self.attention_in_multiplier,
            residual=residual,
            **kwargs,
        )

        # Process input through the SSM branch.
        # FalconH1SSMDecoderLayer expects hidden_states, attn_metadata,
        # residual, and sequence_idx.
        ssm_hidden, _ = self.mamba(
            hidden_states=hidden_states * self.ssm_in_multiplier,
            residual=residual,
            **kwargs,
        )
        # Sum the outputs from both branches.
        # We assume both branches produce outputs of the same
        # dimensionality (config.hidden_size).
        hidden_states = (attn_hidden * self.attn_out_multiplier) + (
            ssm_hidden * self.ssm_out_multiplier
        )
        hidden_states = hidden_states + residual

        # feed-forward
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


@support_torch_compile
class FalconH1Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: FalconH1Config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
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
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
            )
            self.embedding_multiplier = config.embedding_multiplier
        else:
            self.embed_tokens = PPMissingLayer()
            self.embedding_multiplier = 1.0

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            layer_class = FalconH1ParallelHybrid
            return layer_class(
                config,
                layer_idx,
                model_config,
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
        if get_pp_group().is_last_rank:
            self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.final_layernorm = PPMissingLayer()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds * self.embedding_multiplier
            else:
                hidden_states = (
                    self.get_input_embeddings(input_ids) * self.embedding_multiplier
                )
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                }
            )
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


class FalconH1ForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    IsHybrid,
    SupportsMambaPrefixCaching,
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

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

        intermediate_size = (
            int(hf_config.mamba_expand * hf_config.hidden_size)
            if hf_config.mamba_d_ssm is None
            else hf_config.mamba_d_ssm
        )

        return MambaStateShapeCalculator.mamba2_state_shape(
            intermediate_size=intermediate_size,
            tp_world_size=parallel_config.tensor_parallel_size,
            n_groups=hf_config.mamba_n_groups,
            num_heads=hf_config.mamba_n_heads,
            head_dim=hf_config.mamba_d_head,
            state_size=hf_config.mamba_d_state,
            conv_kernel=hf_config.mamba_d_conv,
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        lora_config = vllm_config.lora_config
        scheduler_config = vllm_config.scheduler_config

        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = FalconH1Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.tie_word_embeddings = config.tie_word_embeddings
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=(
                    DEFAULT_VOCAB_PADDING_SIZE
                    # We need bigger padding if using lora for kernel
                    # compatibility
                    if not lora_config
                    else lora_config.lora_vocab_padding_size
                ),
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            self.lm_head_multiplier = config.lm_head_multiplier
            if self.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
            # Used to track and store by the Mamba cache between steps.

            self.logits_processor = LogitsProcessor(
                self.unpadded_vocab_size,
                config.vocab_size,
                scale=config.lm_head_multiplier,
            )
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)

        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if "A_log" in name:
                name = name.replace("A_log", "A")

            if "mamba" in name:
                name = name.replace("mamba", "mamba.mamba")

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
                if self.tie_word_embeddings and "lm_head" in name:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        if self.tie_word_embeddings:
            loaded_params.add("lm_head.weight")
        return loaded_params
