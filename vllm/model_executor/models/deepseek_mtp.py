from typing import Iterable, List, Optional, Set, Tuple

import torch
from torch import nn

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .utils import is_pp_missing_parameter
from .deepseek_v2 import DeepseekV2DecoderLayer

class DeepseekV3MTPSpeculator(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "", mtp_layer_index: int = 0):
        super().__init__()
        config = vllm_config.model_config.hf_config
        config.first_k_dense_replace = 0
        self.config = config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.quant_config = vllm_config.quant_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.shared_head = nn.ModuleDict({
            "head": ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=self.quant_config),
            "norm": RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        })

        layer_index = 61

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.transformer = DeepseekV2DecoderLayer(config, f"{prefix}.layers.{layer_index}", quant_config=self.quant_config, cache_config=self.cache_config, model_config=self.model_config)

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = get_sampler()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        previous_hidden_states: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:        
        if inputs_embeds is not None:
            embedding = inputs_embeds
        else:
            embedding = self.embed_tokens(input_ids)

        h_normed = self.hnorm(previous_hidden_states)
        e_normed = self.enorm(embedding)

        cat_in = torch.cat([e_normed, h_normed], dim=-1) # swapped from the paper
        proj_out = self.eh_proj(cat_in)

        (mtp_hidden, mtp_residual) = self.transformer(
            positions,
            proj_out,
            kv_cache=kv_caches[0],
            attn_metadata=attn_metadata,
            residual=None
        )

        return mtp_hidden + mtp_residual
        # hidden_states = mtp_hidden
        # hidden_states, _ = self.shared_head["norm"](hidden_states, mtp_residual)
        # return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.shared_head["head"], self.shared_head["norm"](hidden_states), sampling_metadata)
        return logits

    def sample(self, logits: torch.Tensor, sampling_metadata: SamplingMetadata) -> SamplerOutput:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens
    
    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts)

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            assert self.config.num_nextn_predict_layers == 1
            layer_idx = 61
            if name.startswith(f"model.layers.{layer_idx}"):
                name = name.replace(f"model.layers.{layer_idx}.", "")
                if name.startswith("input_layernorm") or name.startswith("post_attention_layernorm") or name.startswith("mlp") or name.startswith("self_attn"):
                    name = "transformer." + name
            else:
                continue
            
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if (("mlp.experts." in name) and name not in params_dict):
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                if name not in params_dict:
                    breakpoint()
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name, self):
                        continue
                    
                    if name not in params_dict:
                        breakpoint()
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    if name not in params_dict:
                        breakpoint()
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

