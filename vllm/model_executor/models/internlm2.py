# -*- coding: utf-8 -*-
from typing import Optional

import torch
from transformers import PretrainedConfig
from vllm.config import LoRAConfig

from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)


class InternLM2ForCausalLM(LlamaForCausalLM):

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        linear_method: Optional[LinearMethodBase] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__(config=config,
                         linear_method=linear_method,
                         lora_config=lora_config)

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
        ]
        param_weight_map = [
            ("qkv_proj", "wqkv"),
            ("o_proj", "wo"),
            ("down_proj", "w2"),
            ("input_layernorm", "attention_norm"),
            ("post_attention_layernorm", "ffn_norm"),
            ("embed_tokens", "tok_embeddings"),
            (".self_attn.", ".attention."),
            ("mlp", "feed_forward"),
            ("lm_head", "output"),
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
                if "qkv_proj" in name:
                    config = self.config
                    kv_groups = config.num_attention_heads // config.num_key_value_heads
                    head_dim = config.hidden_size // config.num_attention_heads
                    loaded_weight = loaded_weight.view(-1, 2 + kv_groups,
                                                       head_dim,
                                                       loaded_weight.shape[-1])
                    wq, wk, wv = torch.split(loaded_weight, [kv_groups, 1, 1],
                                             dim=1)
                    wq = wq.reshape(-1, wq.shape[-1])
                    wk = wk.reshape(-1, wk.shape[-1])
                    wv = wv.reshape(-1, wv.shape[-1])
                    weight_loader = param.weight_loader
                    weight_loader(param, wq, 'q')
                    weight_loader(param, wk, 'k')
                    weight_loader(param, wv, 'v')
                else:
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
