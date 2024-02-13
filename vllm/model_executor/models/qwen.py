# coding=utf-8
# Adapted from
# https://huggingface.co/Qwen/Qwen-7B/blob/main/modeling_qwen.py
# Copyright (c) Alibaba Cloud.
# LICENSE: https://huggingface.co/Qwen/Qwen-7B/blob/main/LICENSE
"""Inference-only QWen model compatible with HuggingFace weights."""
from typing import Optional

from transformers import PretrainedConfig
from vllm.config import LoRAConfig

from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)


class QWenLMHeadModel(LlamaForCausalLM):

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        linear_method: Optional[LinearMethodBase] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        norm = RMSNorm(config.hidden_size, config.layer_norm_epsilon)
        config.use_qkv_bias = True
        config.intermediate_size = config.intermediate_size // 2
        super().__init__(config=config,
                         linear_method=linear_method,
                         norm=norm,
                         lora_config=lora_config)

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w2", 0),
            ("gate_up_proj", "w1", 1),
        ]
        param_weight_map = [
            ("model", "transformer"),
            (".self_attn.", ".attn."),
            (".layers.", ".h."),
            ("qkv_proj", "c_attn"),
            (".self_attn.o_proj", ".self_attn.c_proj"),
            ("norm", "ln_f"),
            ("mlp.down_proj", "mlp.c_proj"),
            ("input_layernorm", "ln_1"),
            ("post_attention_layernorm", "ln_2"),
            ("embed_tokens", "wte"),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            for (param_name, weight_name) in param_weight_map:
                name = name.replace(weight_name, param_name)

            if "rotary_emb.inv_freq" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
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
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
