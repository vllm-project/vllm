"""Utilities for selecting and loading neuron models."""
import importlib
import os
from typing import Optional, Type

import torch
import torch.nn as nn
import transformers
from transformers import PretrainedConfig

from vllm.config import ModelConfig, ParallelConfig, SchedulerConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput

TORCH_DTYPE_TO_NEURON_AMP = {
    "auto": "f32",
    "half": "f16",
    "float16": "f16",
    "bfloat16": "bf16",
    "float": "f32",
    "float32": "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
}

# Models supported by Neuron.
_NEURON_SUPPORTED_MODELS = {
    "LlamaForCausalLM": ("transformers_neuronx.llama.model",
                         "LlamaForSampling", "LlamaForCausalLM"),
    "MistralForCausalLM": ("transformers_neuronx.mistral.model",
                           "MistralForSampling", "MistralForCausalLM")
}


class NeuronCasualLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = None
        self.logits_processor = LogitsProcessor(config.vocab_size,
                                                logits_as_input=True)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_block_ids: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.model(input_ids,
                            cache_ids=positions,
                            start_ids=input_block_ids)
        return logits

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(None, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, model_name_or_path: str, **kwargs):
        arch = _get_model_architecture(self.config)
        neuronx_module_path, neuronx_model_cls, hf_model_cls = (
            _NEURON_SUPPORTED_MODELS[arch])
        neuronx_module = importlib.import_module(neuronx_module_path)
        neuronx_model_cls = getattr(neuronx_module, neuronx_model_cls)

        split_model_dir = f"{model_name_or_path}-split"
        if os.path.isdir(os.path.join(model_name_or_path,
                                      "pytorch_model.bin")):
            split_model_dir = model_name_or_path
        elif not os.path.exists(f"{model_name_or_path}-split"):
            hf_model_cls = getattr(transformers, hf_model_cls)
            from transformers_neuronx.module import save_pretrained_split

            hf_model = hf_model_cls.from_pretrained(model_name_or_path,
                                                    low_cpu_mem_usage=True)
            save_pretrained_split(hf_model, f"{model_name_or_path}-split")

        self.model = neuronx_model_cls.from_pretrained(split_model_dir,
                                                       **kwargs)
        self.model.to_neuron()


def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _NEURON_SUPPORTED_MODELS:
            return arch
    raise ValueError(
        f"Model architectures {architectures} are not supported on Neuron "
        f"for now. Supported architectures: "
        f"{list(_NEURON_SUPPORTED_MODELS.keys())}")


def get_neuron_model(model_config: ModelConfig,
                     parallel_config: ParallelConfig,
                     scheduler_config: SchedulerConfig) -> nn.Module:
    from transformers_neuronx.config import (ContinuousBatchingConfig,
                                             NeuronConfig)

    # Create a model instance.
    model = NeuronCasualLM(model_config.hf_config)

    continuous_batching_config = ContinuousBatchingConfig(
        batch_size_for_shared_caches=scheduler_config.max_num_seqs)
    neuron_config = NeuronConfig(
        continuous_batching=continuous_batching_config)

    # Load the weights from the cached or downloaded files.
    model.load_weights(
        model_config.model,
        tp_degree=parallel_config.tensor_parallel_size,
        amp=TORCH_DTYPE_TO_NEURON_AMP[model_config.dtype],
        neuron_config=neuron_config,
        context_length_estimate=[scheduler_config.max_model_len],
        n_positions=[scheduler_config.max_model_len],
        batch_size=scheduler_config.max_num_seqs)

    return model.eval()
