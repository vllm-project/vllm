"""Inference-only SmallThinker model compatible with HuggingFace weights."""
from collections.abc import Iterable
from typing import Optional, Union, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size, tensor_model_parallel_all_reduce
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ReplicatedLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (PPMissingLayer, 
                    extract_layer_index, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix, WeightsMapper)
from vllm.model_executor.layers.fused_moe import FusedMoE



class SmallThinkerMoeBlock(nn.Module):
    """
    MoE Block with primary experts
    """
    def __init__(self, config, quant_config: Optional[QuantizationConfig] = None, prefix: str = ""):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.moe_ffn_hidden_size = config.moe_ffn_hidden_size
        self.num_primary_experts = config.moe_num_primary_experts
        self.num_active_primary_experts = config.moe_num_active_primary_experts
        self.moe_primary_router_apply_softmax = config.moe_primary_router_apply_softmax
        self.tp_size = get_tensor_model_parallel_world_size()

        # Primary router
        self.primary_router = ReplicatedLinear(self.hidden_dim, self.num_primary_experts, bias=False, 
                                               return_bias=False, quant_config=quant_config, prefix=f"{prefix}.primary_router")
        
        def custom_topk(hidden_states, gating_output, topk, renormalize):
            router_logits, selected_experts = torch.topk(gating_output, topk, dim=-1)
            if self.moe_primary_router_apply_softmax:
                routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            else:
                routing_weights = F.sigmoid(router_logits)
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            return routing_weights, selected_experts
        
        # Scoring function is either 'softmax' or 'sigmoid', but we pass scoring_func='softmax' to FusedMoE as it only accepts softmax.
        # This parameter does not affect the routing function as we are using a custom routing function.
        self.experts = FusedMoE(num_experts=self.num_primary_experts,
                                top_k=self.num_active_primary_experts,
                                custom_routing_function=custom_topk,
                                hidden_size=self.hidden_dim,
                                intermediate_size=self.moe_ffn_hidden_size,
                                reduce_results=False,
                                renormalize=True,
                                quant_config=quant_config,
                                scoring_func='softmax',
                                activation='relu',
                                prefix=f"{prefix}.experts")

    def forward(self, router_input: torch.Tensor, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length, hidden_dim = hidden_states.shape
        
        orig_shape = hidden_states.shape
        # Flatten for processing
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_input = router_input.view(-1, hidden_dim)
        
        router_logits = self.primary_router(router_input)

        final_hidden_states = self.experts(hidden_states=hidden_states,
                                            router_logits=router_logits)
        final_hidden_states = final_hidden_states
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        
        return final_hidden_states.view(orig_shape)


class SmallThinkerAttention(nn.Module):
    """Multi-head attention with optional sliding window."""
    
    def __init__(
        self,
        config,
        layer_idx: int = 0,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.rope_theta = config.rope_theta
        self.layer_idx = layer_idx
        self.rope_scaling = getattr(config, "rope_scaling", None)
        
        # Sliding window configuration
        self.sliding_window = None
        if hasattr(config, 'sliding_window_layout') and config.sliding_window_layout and config.sliding_window_layout[layer_idx]:
            self.sliding_window = config.sliding_window_size
        
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # QKV projection
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        
        # Output projection
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # RoPE
        if config.rope_layout[self.layer_idx]:
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=config.max_position_embeddings,
                base=self.rope_theta,
                rope_scaling=self.rope_scaling
            )
        else:
            self.rotary_emb = lambda positions, q, k: (q, k)

        # Attention mechanism with sliding window support
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            per_layer_sliding_window=self.sliding_window,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class SmallThinkerDecoderLayer(nn.Module):
    """Decoder layer combining attention and MLP/MoE."""
    
    def __init__(
        self,
        config,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        # Attention
        self.self_attn = SmallThinkerAttention(
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        
        self.mlp = SmallThinkerMoeBlock(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        
        # Layer norms
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states = hidden_states + residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        router_input = residual

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        
        # MLP
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(router_input, hidden_states)
        
        return hidden_states, residual

@support_torch_compile
class SmallThinkerModel(nn.Module):
    """Main model class."""
    
    def __init__(
        self,
        *, 
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        # Embeddings
        if get_pp_group().is_first_rank or (config.tie_word_embeddings and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Decoder layers
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: SmallThinkerDecoderLayer(
                config=config,
                layer_idx=extract_layer_index(prefix),
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        # Output norm
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states = hidden_states + residual
        hidden_states = self.norm(hidden_states)
        return hidden_states

class SmallThinkerWeightsMapper(WeightsMapper):
    _REPLACE_RULES: Tuple[Tuple[str, str], ...] = (
        (".block_sparse_moe", ".mlp"),
        (".up",   ".up_proj"),
        (".down", ".down_proj"),
        (".gate", ".gate_proj"),
    )

    def __init__(self, quant_config=None):
        super().__init__()
        self.quant_config = quant_config
        self._buf_qkv: dict[str, dict[str, torch.Tensor]] = {}   # q/k/v
        self._buf_gateup:  dict[str, dict[int,  torch.Tensor]] = {}   # 0/1

    @staticmethod
    def _rename(name: str) -> str:
        for old, new in SmallThinkerWeightsMapper._REPLACE_RULES:
            name = name.replace(old, new)
        return name

    def apply(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        """Generator that yields `(new_name, tensor)`"""

        for orig_name, w in weights:
            name = self._rename(orig_name)

            if self.quant_config is not None:
                scale_name = self.quant_config.get_cache_scale(name)
                if scale_name is not None:
                    yield scale_name, w.squeeze()
                    continue

            tokens = name.split(".")

            if len(tokens) >= 2 and tokens[-2] in {"q_proj", "k_proj", "v_proj"}:
                key  = tokens[-2][0] # "q" / "k" / "v"
                tokens[-2] = "qkv_proj"
                packed_name = ".".join(tokens)
                self._buf_qkv.setdefault(packed_name, {})[key] = w
                continue

            if len(tokens) >= 2 and tokens[-2] in {"gate_proj", "up_proj"}:
                shard_id = 0 if tokens[-2] == "gate_proj" else 1
                tokens[-2] = "gate_up_proj"
                packed_name = ".".join(tokens)
                self._buf_gateup.setdefault(packed_name, {})[shard_id] = w
                continue

            yield name, w

        for packed_name, parts in self._buf_qkv.items():
            if len(parts) == 3:
                yield packed_name, torch.cat(
                    [parts["q"], parts["k"], parts["v"]], dim=0
                )

        for packed_name, parts in self._buf_gateup.items():
            if len(parts) == 2:
                yield packed_name, torch.cat(
                    [parts[0], parts[1]], dim=0
                )

class SmallThinkerForCausalLM(nn.Module):
    """Causal language model with vLLM optimization."""
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config
        
        self.model = SmallThinkerModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

        # Language model head
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head")
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (".bias", "_bias", ".k_scale", "_k_scale",
                           ".v_scale", "_v_scale", ".weight_scale",
                           "_weight_scale", ".input_scale", "_input_scale")

        # Rule 1: QKV Fusion
        # Original weights:
        #   model.layers.{i}.self_attn.q_proj.weight
        #   model.layers.{i}.self_attn.k_proj.weight
        #   model.layers.{i}.self_attn.v_proj.weight
        # Fused weights:
        #   model.layers.{i}.self_attn.qkv_proj.weight (shards "q", "k", "v")
        # Format:
        #   (param_name, shard_name, shard_id)
        stacked_attn_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        # Params for weights
        # Rule 2: Expert fusion
        # Original weights:
        #   model.layers.{i}.block_sparse_moe.experts.{j}.gate.weight   (w1)
        #   model.layers.{i}.block_sparse_moe.experts.{j}.up.weight     (w3)
        #   model.layers.{i}.block_sparse_moe.experts.{j}.down.weight   (w2)
        # Fused weight:
        #   model.layers.{i}.mlp.experts.w13_weight (a lot of shards)
        #   model.layers.{i}.mlp.experts.w2_weight (a lot of shards)
        # Mapping data structure:
        #   (param_name, weight_name, expert_id, shard_id)
        #   ('experts.w13_', 'experts.0.gate.', 0, 'w1')
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate",
            ckpt_down_proj_name="down",
            ckpt_up_proj_name="up",
            num_experts=self.config.moe_num_primary_experts)
        
        # Rule 3: Router rename
        # Original weights:
        #   *.block_sparse_moe.*
        # Renamed weights:
        #   *.mlp.*
        router_rename_mapping = [("mlp.primary_router", "block_sparse_moe.primary_router")]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # Skip tied LMHead
            if self.config.tie_word_embeddings and name.endswith(
                "lm_head.weight"):
                continue
            for (param_name, weight_name, shard_id) in stacked_attn_params_mapping:
                # Attention process (Rule 1)
                # Check for {qkv}_proj in weight name to skip non-stacked modules
                if weight_name not in name:
                    continue
                # Original name:
                #   model.layers.0.self_attn.q_proj.weight
                # Mapped name:
                #   model.layers.0.self_attn.qkv_proj.weight
                name = name.replace(weight_name, param_name)

                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue

                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # FFN Process (Rule 2)
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # SmallThinker: expert name replace
                    name = name.replace("block_sparse_moe", "mlp")
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(
                            ignore_suffixes) and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    # Router process (Rule 3)
                    for mapping in router_rename_mapping:
                        param_name, weight_name = mapping
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        # Skip layers on other devices.
                        if is_pp_missing_parameter(name, self):
                            continue
                        # Skip loading extra parameters for GPTQ/modelopt models.
                        if name.endswith(
                                ignore_suffixes) and name not in params_dict:
                            continue
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        weight_loader(param, loaded_weight)
                        break
                    else:
                        # Skip loading extra parameters for GPTQ/modelopt models.
                        if name.endswith(
                                ignore_suffixes) and name not in params_dict:
                            continue
                        # Skip layers on other devices.
                        if is_pp_missing_parameter(name, self):
                            continue
                        # Remapping the name of FP8 kv-scale.
                        if name.endswith("kv_scale"):
                            remapped_kv_scale_name = name.replace(
                                ".kv_scale", ".attn.kv_scale")
                            if remapped_kv_scale_name not in params_dict:
                                continue
                            else:
                                name = remapped_kv_scale_name
                        param = params_dict[name]
                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                        weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def make_empty_intermediate_tensors(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> IntermediateTensors:
        return self.model.make_empty_intermediate_tensors(batch_size, dtype, device)