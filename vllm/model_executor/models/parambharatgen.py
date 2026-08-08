# SPDX-License-Identifier: Apache-2.0
"""ParamBharatGen vLLM model (LLaMA-compatible)."""

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn
from transformers import PretrainedConfig as LlamaConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


# ---------------- MLP ----------------
class ParamBharatGenMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, prefix=""):
        super().__init__()
        print(f"    MLP init: {prefix}", flush=True)
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()
        print(f"    MLP init complete", flush=True)

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


# ---------------- Attention ----------------
class ParamBharatGenAttention(nn.Module):
    def __init__(
        self,
        config,
        hidden_size,
        num_heads,
        num_kv_heads,
        cache_config=None,
        prefix="",
    ):
        super().__init__()
        print(f"    Attention init: {prefix}", flush=True)

        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads

        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = getattr(config, "head_dim", hidden_size // num_heads)
        print(f"    head_dim={self.head_dim}, num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}", flush=True)

        print(f"    Creating qkv_proj...", flush=True)
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            prefix=f"{prefix}.qkv_proj",
        )

        print(f"    Creating o_proj...", flush=True)
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            prefix=f"{prefix}.o_proj",
        )

        print(f"    Creating rotary_emb...", flush=True)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        rope_parameters = getattr(config, "rope_parameters", None)
        
        # Print the full config to see what we're dealing with
        print(f"    rope: max_pos={max_position_embeddings}, rope_parameters={rope_parameters}", flush=True)
        print(f"    config.rope_theta={getattr(config, 'rope_theta', 'N/A')}", flush=True)
        print(f"    config.rope_scaling={getattr(config, 'rope_scaling', 'N/A')}", flush=True)
        
        import time
        start = time.time()
        print(f"    Calling get_rope...", flush=True)
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=getattr(config, "rope_parameters", None),
            is_neox_style=True,
        )
        print(f"    rotary_emb created in {time.time() - start:.2f}s", flush=True)

        print(f"    Creating Attention layer...", flush=True)
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )
        print(f"    Attention init complete", flush=True)

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = torch.split(
            qkv,
            [
                self.num_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ],
            dim=-1,
        )

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


# ---------------- Decoder Layer ----------------
class ParamBharatGenDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix=""):
        super().__init__()
        print(f"=== Creating decoder layer: {prefix} ===", flush=True)

        config = vllm_config.model_config.hf_config

        self.hidden_size = config.hidden_size

        num_kv_heads = getattr(
            config,
            "num_key_value_heads",
            getattr(config, "num_kv_heads", config.num_attention_heads),
        )

        print(f"  Creating attention...", flush=True)
        self.self_attn = ParamBharatGenAttention(
            config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            cache_config=vllm_config.cache_config,
            prefix=f"{prefix}.self_attn",
        )
        print(f"  Attention created", flush=True)

        intermediate_size = getattr(
            config,
            "intermediate_size",
            int(config.hidden_size * getattr(config, "custom_mlp_ratio", 4)),
        )

        print(f"  Creating MLP (intermediate_size={intermediate_size})...", flush=True)
        self.mlp = ParamBharatGenMLP(
            self.hidden_size,
            intermediate_size,
            prefix=f"{prefix}.mlp",
        )
        print(f"  MLP created", flush=True)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        print(f"=== Decoder layer {prefix} complete ===", flush=True)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


# ---------------- Model ----------------
@support_torch_compile
class ParamBharatGenModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix=""):
        super().__init__()
        print("=== ParamBharatGenModel.__init__ started ===", flush=True)

        config = vllm_config.model_config.hf_config
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", None)

        # Normalize config
        config.num_kv_heads = getattr(
            config,
            "num_key_value_heads",
            config.num_attention_heads,
        )

        print("=== Creating embed_tokens ===", flush=True)
        if get_pp_group().is_first_rank or (
            getattr(config, "tie_word_embeddings", False) 
            and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        print("=== Creating layers ===", flush=True)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: ParamBharatGenDecoderLayer(
                vllm_config=vllm_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )
        print(f"=== Layers created: {self.start_layer} to {self.end_layer} ===", flush=True)

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# ---------------- LM Head ----------------
class ParamBharatGenForCausalLM(nn.Module, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix=""):
        super().__init__()
        print("=== ParamBharatGenForCausalLM.__init__ started ===", flush=True)

        config = vllm_config.model_config.hf_config
        self.config = config
        print(f"Config: hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}", flush=True)

        self.model = ParamBharatGenModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        print("=== ParamBharatGenModel created ===", flush=True)

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if getattr(config, "tie_word_embeddings", False):
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        import sys
        print("=== Starting weight loading ===", flush=True)
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        
        weight_count = 0
        for name, loaded_weight in weights:
            weight_count += 1
            if weight_count % 50 == 0:
                print(f"Loaded {weight_count} weights, current: {name}", flush=True)
            
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
                
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        
        print(f"=== Finished loading {weight_count} weights ===", flush=True)
        return loaded_params