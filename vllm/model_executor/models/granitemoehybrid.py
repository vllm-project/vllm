# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only GraniteMoeHybrid model."""
# Added by the IBM Team, 2025
from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn
from transformers import GraniteMoeHybridConfig

from vllm import envs
from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba2_metadata import (
    Mamba2Metadata, prepare_mamba2_metadata)
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateShapeCalculator)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.mamba_cache import (MambaCacheManager,
                                                    MambaCacheParams)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.utils import LayerBlockType

from .granitemoe import GraniteMoeMoE
from .granitemoeshared import GraniteMoeSharedMLP
from .interfaces import (HasInnerState, IsHybrid, SupportsLoRA, SupportsPP,
                         SupportsQuant)
from .utils import (AutoWeightsLoader, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)


class GraniteMoeHybridMambaDecoderLayer(nn.Module):

    def __init__(self,
                 config: GraniteMoeHybridConfig,
                 layer_idx: int,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.residual_multiplier = config.residual_multiplier

        self.mamba = MambaMixer2(hidden_size= config.hidden_size,
                                ssm_state_size = config.mamba_d_state,
                                conv_kernel_size = config.mamba_d_conv,
                                intermediate_size = config.mamba_expand *\
                                                    config.hidden_size,
                                use_conv_bias = config.mamba_conv_bias,
                                use_bias = config.mamba_proj_bias,
                                n_groups=config.mamba_n_groups,
                                num_heads=config.mamba_n_heads,
                                head_dim=config.mamba_d_head,
                                rms_norm_eps=config.rms_norm_eps,
                                activation=config.hidden_act,
                                quant_config=quant_config,
                                prefix=f"{prefix}.mixer")

        self.block_sparse_moe = None
        if getattr(config, "num_local_experts", 0) > 0:
            self.block_sparse_moe = GraniteMoeMoE(
                num_experts=config.num_local_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                quant_config=quant_config,
                prefix=f"{prefix}.block_sparse_moe")

        self.shared_mlp = None if \
            getattr(config, 'shared_intermediate_size', 0) == 0 \
            else GraniteMoeSharedMLP(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_mlp"
            )

        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        mamba_cache_params: MambaCacheParams,
        mamba2_metadata: Mamba2Metadata,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        output = torch.empty_like(hidden_states)
        self.mamba(hidden_states, output, mamba_cache_params, mamba2_metadata)
        hidden_states = residual + output * self.residual_multiplier

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.shared_mlp is None:
            if self.block_sparse_moe is not None:
                hidden_states = self.block_sparse_moe(hidden_states)
            # else: skip
        else:
            # create a copy since block_sparse_moe modifies in-place
            if self.block_sparse_moe is not None:
                moe_hidden_states = hidden_states.clone()
                moe_hidden_states = self.block_sparse_moe(moe_hidden_states)
                hidden_states = moe_hidden_states + self.shared_mlp(
                    hidden_states)
                del moe_hidden_states
            else:
                hidden_states = self.shared_mlp(hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier

        return hidden_states, residual


class GraniteMoeHybridAttentionDecoderLayer(nn.Module):

    def __init__(
        self,
        config: GraniteMoeHybridConfig,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.residual_multiplier = config.residual_multiplier

        self.self_attn = GraniteMoeHybridAttention(
            config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn")

        self.block_sparse_moe = None
        if getattr(config, "num_local_experts", 0) > 0:
            self.block_sparse_moe = GraniteMoeMoE(
                num_experts=config.num_local_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                quant_config=quant_config,
                prefix=f"{prefix}.block_sparse_moe")

        self.shared_mlp = None if \
            getattr(config, 'shared_intermediate_size', 0) == 0 \
            else GraniteMoeSharedMLP(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_mlp"
            )

        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        mamba_cache_params: MambaCacheParams,
        mamba2_metadata: Mamba2Metadata,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states = residual + hidden_states * self.residual_multiplier

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.shared_mlp is None:
            if self.block_sparse_moe is not None:
                hidden_states = self.block_sparse_moe(hidden_states)
            # else: skip
        else:
            # create a copy since block_sparse_moe modifies in-place
            if self.block_sparse_moe is not None:
                moe_hidden_states = hidden_states.clone()
                moe_hidden_states = self.block_sparse_moe(moe_hidden_states)
                hidden_states = moe_hidden_states + self.shared_mlp(
                    hidden_states)
                del moe_hidden_states
            else:
                hidden_states = self.shared_mlp(hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier

        return hidden_states, residual


class GraniteMoeHybridAttention(nn.Module):

    def __init__(
        self,
        config: GraniteMoeHybridConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.causal = True
        self.hidden_size = config.hidden_size
        self.attention_bias = config.attention_bias
        self.attention_multiplier = config.attention_multiplier
        self.total_num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.total_num_heads
        self.total_num_kv_heads = config.num_key_value_heads

        # TensorParallel logic
        tp_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_key_value_heads = max(1, self.total_num_kv_heads // tp_size)

        self.qkv_proj = QKVParallelLinear(self.hidden_size,
                                          self.head_dim,
                                          self.total_num_heads,
                                          self.total_num_kv_heads,
                                          bias=self.attention_bias,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.qkv_proj")

        self.o_proj = RowParallelLinear(self.hidden_size,
                                        self.hidden_size,
                                        bias=self.attention_bias,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.o_proj")

        if config.position_embedding_type == "rope":
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=config.max_position_embeddings,
                base=int(config.rope_theta),
                rope_scaling=config.rope_scaling \
                    if hasattr(config, "rope_scaling") \
                    and config.rope_scaling is not None else None,
                is_neox_style=True,
            )
        else:
            self.rotary_emb = None

        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.attention_multiplier,
                              num_kv_heads=self.num_key_value_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        qkv, _ = self.qkv_proj(hidden_states)
        query, key, value = qkv.split([
            self.num_heads * self.head_dim, self.num_key_value_heads *
            self.head_dim, self.num_key_value_heads * self.head_dim
        ],
                                      dim=-1)

        if self.rotary_emb is not None:
            query, key = self.rotary_emb(positions, query, key)

        hidden_states = self.attn(query, key, value)
        del query, key, value

        hidden_states = self.o_proj(hidden_states)[0]
        return hidden_states


ALL_DECODER_LAYER_TYPES = {
    "attention": GraniteMoeHybridAttentionDecoderLayer,
    "mamba": GraniteMoeHybridMambaDecoderLayer,
}


@support_torch_compile
class GraniteMoeHybridModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        lora_vocab = ((lora_config.lora_extra_vocab_size *
                       (lora_config.max_loras or 1)) if lora_config else 0)
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )
        self.embedding_multiplier = config.embedding_multiplier

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            layer_class = ALL_DECODER_LAYER_TYPES[
                config.layer_types[layer_idx]]
            return layer_class(
                config,
                layer_idx,
                cache_config,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers")
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mamba_cache_params: MambaCacheParams,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        attn_metadata = get_forward_context().attn_metadata

        if not envs.VLLM_USE_V1:
            mamba2_metadata = prepare_mamba2_metadata(
                chunk_size=self.config.mamba_chunk_size,
                attn_metadata=attn_metadata,
            )
        else:
            # v1 get mamba2_metadata from forward_context
            mamba2_metadata = None

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
                hidden_states = hidden_states * self.embedding_multiplier
            residual = None
        else:
            if intermediate_tensors is None:
                raise RuntimeError('Intermediate tensors may not be None!')
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        num_attn = 0
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if isinstance(layer, GraniteMoeHybridAttentionDecoderLayer):
                num_attn += 1

            layer_mamba_cache_params = None
            if isinstance(
                    layer,
                    GraniteMoeHybridMambaDecoderLayer) and mamba_cache_params:
                layer_mamba_cache_params = mamba_cache_params.at_layer_idx(
                    i - num_attn)

            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                mamba_cache_params=layer_mamba_cache_params,
                mamba2_metadata=mamba2_metadata)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        def _load(n, p):
            param = params_dict[n]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, p)
            loaded_params.add(n)

        def _load_shard(n, p, shard_id):
            # Skip layers on other devices.
            if not is_pp_missing_parameter(n, self):
                param = params_dict[n]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, p, shard_id)
                loaded_params.add(n)

        def _load_expert(n, p, name, shard_id, expert_id):
            param = params_dict[n]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param,
                          p,
                          name,
                          shard_id=shard_id,
                          expert_id=expert_id)
            loaded_params.add(n)

        for n, p in weights:
            if "A_log" in n:
                n = n.replace("A_log", "A")

            # Logic analogous to: https://github.com/vllm-project/vllm/blob/f49e5aff11c986ed4d45202b1716c5d74786efa9/vllm/model_executor/models/granitemoeshared.py#L215
            # Mapping different experts' layout:
            #  from HF (input_linear, output_linear, router)
            #  to vLLM (experts_w13({e}.w1, {e}.w2), experts_w3({e}.w3), gate)
            if n.endswith('.block_sparse_moe.input_linear.weight'):
                for e in range(p.size(0)):
                    w1_name = n.replace(
                        '.block_sparse_moe.input_linear.weight',
                        f".block_sparse_moe.experts.{e}.w1.weight")
                    w3_name = n.replace(
                        '.block_sparse_moe.input_linear.weight',
                        f".block_sparse_moe.experts.{e}.w3.weight")
                    w1_param, w3_param = p[e].chunk(2, dim=0)
                    _load_expert(n.replace('.input_linear.', '.experts.w13_'),
                                 w1_param,
                                 w1_name,
                                 shard_id='w1',
                                 expert_id=e)
                    _load_expert(n.replace('.input_linear.', '.experts.w13_'),
                                 w3_param,
                                 w3_name,
                                 shard_id='w3',
                                 expert_id=e)
            elif n.endswith('.block_sparse_moe.output_linear.weight'):
                for e in range(p.size(0)):
                    w2_name = n.replace(
                        '.block_sparse_moe.output_linear.weight',
                        f".block_sparse_moe.experts.{e}.w2.weight")
                    w2_param = p[e]
                    _load_expert(n.replace('.output_linear.', '.experts.w2_'),
                                 w2_param,
                                 w2_name,
                                 shard_id='w2',
                                 expert_id=e)
            elif n.endswith('.block_sparse_moe.router.layer.weight'):
                gate_name = n.replace('.block_sparse_moe.router.layer.weight',
                                      ".block_sparse_moe.gate.weight")
                _load(gate_name, p)
            else:
                loaded = False
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name in n:
                        _load_shard(n.replace(weight_name, param_name),
                                    p,
                                    shard_id=shard_id)
                        loaded = True
                if not loaded:
                    _load(n, p)

        return loaded_params


class GraniteMoeHybridForCausalLM(nn.Module, HasInnerState, SupportsLoRA,
                                  SupportsPP, IsHybrid, SupportsQuant):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    }
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: "VllmConfig",
        use_v1: bool = True,
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        """Calculate shapes for Mamba's convolutional and state caches.

        Args:
            vllm_config: vLLM config
            use_v1: Get shapes for V1 (or V0)

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
            n_groups=hf_config.mamba_n_groups,
            num_heads=hf_config.mamba_n_heads,
            head_dim=hf_config.mamba_d_head,
            state_size=hf_config.mamba_d_state,
            conv_kernel=hf_config.mamba_d_conv,
            use_v1=use_v1,
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        lora_config = vllm_config.lora_config
        scheduler_config = vllm_config.scheduler_config
        if cache_config.enable_prefix_caching:
            raise RuntimeError(
                "GraniteMoeHybrid currently does not support prefix caching")

        self.quant_config = vllm_config.quant_config
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = GraniteMoeHybridModel(vllm_config=vllm_config,
                                           prefix=maybe_prefix(
                                               prefix, "model"))
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
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"))
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size,
                                                scale=1 /
                                                self.config.logits_scaling)

        # Used to track and store by the Mamba cache between steps.
        self.mamba_cache: Optional[MambaCacheManager] = None

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs):

        mamba_cache_params = None
        if not envs.VLLM_USE_V1:
            if self.mamba_cache is None:
                num_mamba_layers = (
                    self.model_config.get_num_layers_by_block_type(
                        self.vllm_config.parallel_config,
                        LayerBlockType.mamba))
                mamba_state_shape = \
                    self.get_mamba_state_shape_from_config(
                        self.vllm_config, use_v1=False)
                self.mamba_cache = MambaCacheManager(self.vllm_config,
                                                     self.model_config.dtype,
                                                     num_mamba_layers,
                                                     *mamba_state_shape)

            mamba_cache_params = self.mamba_cache.current_run_tensors(**kwargs)

        hidden_states = self.model(input_ids, positions, mamba_cache_params,
                                   intermediate_tensors, inputs_embeds)

        return hidden_states

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
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
