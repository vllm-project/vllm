from typing import Iterable, List, Optional, Tuple

import torch
from transformers import LlamaConfig

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader)
from vllm.utils import print_warning_once
from vllm.model_executor.models.llama import LlamaForCausalLM


class AtlaLlamaForCausalLMWithAuxHead(LlamaForCausalLM):
    """ adapted from models/llama.py """
    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super(AtlaLlamaForCausalLMWithAuxHead, self).__init__(config=config,
                                               cache_config=cache_config,
                                               quant_config=quant_config,
                                               lora_config=lora_config)
        # Define the layers and initialization specific to LlamaForCausalLM
        self.regression_head = torch.nn.Linear(config.hidden_size, 1, dtype=config.torch_dtype)

    def forward_with_aux_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
		# hidden_states: [sequence_length, hidden_dimension]
		# When generating a sequence, the first hidden_states will contain
		# the hidden states for all the prompt tokens and the first completion
		# token. Subsequent calls will only get the hidden_states of the
		# currently generated token i.e. shape [1, hidden_dimensions]
		# We want to use the hidden state of the first completion token.
        prediction = self.regression_head(hidden_states[-1])
        return prediction

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # TODO: we should only add our parameter weights to the model.
        # everything else should be handled by the super class / the mixin
        # However, currently we re-implement it since we clean up the 
        # '.base_layer' in the weights name
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            # TODO: hacks to deal with the way how the model
            # has currently be saved
            name = name.replace('.base_layer', '')
            if "lora" in name:
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
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
                # Remapping the name of FP8 kv-scale.
                if name.endswith("kv_scale"):
                    remapped_kv_scale_name = name.replace(
                        ".kv_scale", ".attn.kv_scale")
                    if remapped_kv_scale_name not in params_dict:
                        print_warning_once(
                            f"Found kv scale in the checkpoint (e.g. {name}), "
                            "but not found the expected name in the model "
                            f"(e.g. {remapped_kv_scale_name}). kv-scale is "
                            "not loaded.")
                        continue
                    else:
                        name = remapped_kv_scale_name
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)