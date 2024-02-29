"""Inference-only LLaMA model compatible with HuggingFace weights."""
import os
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput

KVCache = Tuple[torch.Tensor, torch.Tensor]


class LlamaForCausalLM(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        linear_method=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = None
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        with torch.inference_mode():
            block_size = self.model.context_buckets[-1]
            if input_metadata.is_prompt:
                seq_ids = input_metadata.slot_mapping[:, 0] // block_size
            else:
                seq_ids = input_metadata.block_tables
            logits = self.model(input_ids,
                                cache_ids=positions,
                                start_ids=seq_ids.flatten())
        return logits

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.model.chkpt_model.lm_head,
                                   hidden_states, sampling_metadata)
        return next_tokens

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None,
                     **kwargs):
        from transformers_neuronx.llama.model import LlamaForSampling

        split_model_dir = f"{model_name_or_path}-split"
        if os.path.isdir(os.path.join(model_name_or_path,
                                      "pytorch_model.bin")):
            split_model_dir = model_name_or_path
        elif not os.path.exists(f"{model_name_or_path}-split"):
            from transformers.models.llama import LlamaForCausalLM
            from transformers_neuronx.module import save_pretrained_split

            hf_model = LlamaForCausalLM.from_pretrained(model_name_or_path,
                                                        low_cpu_mem_usage=True)
            save_pretrained_split(hf_model, f"{model_name_or_path}-split")

        self.model = LlamaForSampling.from_pretrained(split_model_dir,
                                                      **kwargs)
        self.model.to_neuron()
