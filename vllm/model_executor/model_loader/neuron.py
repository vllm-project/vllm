# SPDX-License-Identifier: Apache-2.0
"""Utilities for selecting and loading neuron models."""
import copy
import importlib
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import ModelConfig, ParallelConfig, SchedulerConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import (CompletionSequenceGroupOutput, Logprob,
                           SequenceOutput)

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
_NEURON_SUPPORTED_MODELS: Dict[str, Tuple[str, str, str]] = {
    "LlamaForCausalLM": ("transformers_neuronx.llama.model",
                         "LlamaForSampling", "LlamaForCausalLM"),
    "MistralForCausalLM": ("transformers_neuronx.mistral.model",
                           "MistralForSampling", "MistralForCausalLM")
}


class NeuronCausalLM(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 on_device_sampling_disabled: bool = False) -> None:
        super().__init__()
        self.config = config
        self.logits_processor = LogitsProcessor(config.vocab_size,
                                                logits_as_input=True)

        self.on_device_sampling_disabled = on_device_sampling_disabled
        if self.on_device_sampling_disabled:
            # Use default sampler
            self.sampler = Sampler()

        # Lazy initialized
        self.model: nn.Module

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

        if self.on_device_sampling_disabled:
            next_tokens = self.sampler(logits, sampling_metadata)
            return next_tokens

        # On-device sampling outputs the token ids directly.
        sampled_token_ids = logits.flatten()
        next_tokens = []
        sample_idx = 0
        for seq_group in sampling_metadata.seq_groups:
            samples = []
            for seq_id in seq_group.seq_ids:
                token_id = sampled_token_ids[sample_idx].item()
                samples.append(
                    SequenceOutput(parent_seq_id=seq_id,
                                   output_token=token_id,
                                   logprobs={token_id: Logprob(token_id)}))
                sample_idx += 1
            next_tokens.append(
                CompletionSequenceGroupOutput(samples=samples,
                                              prompt_logprobs=None))

        return SamplerOutput(outputs=next_tokens)

    def load_weights(self, model_name_or_path: str, **kwargs):
        arch = _get_model_architecture(self.config)
        neuronx_module_path, neuronx_model_cls_name, hf_model_cls_name = (
            _NEURON_SUPPORTED_MODELS[arch])
        neuronx_module = importlib.import_module(neuronx_module_path)
        neuronx_model_cls = getattr(neuronx_module, neuronx_model_cls_name)

        self.model = neuronx_model_cls.from_pretrained(model_name_or_path,
                                                       **kwargs)
        self.model.to_neuron()


def _get_model_architecture(config: PretrainedConfig) -> str:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _NEURON_SUPPORTED_MODELS:
            return arch
    raise ValueError(
        f"Model architectures {architectures} are not supported on Neuron "
        f"for now. Supported architectures: "
        f"{list(_NEURON_SUPPORTED_MODELS.keys())}")


def _get_buckets(env: str, default_value: List[int]) -> List[int]:
    env_value = os.getenv(env)
    if env_value is None:
        return default_value
    buckets_remove_empty = filter(
        lambda x: x is not None and len(x.strip()) > 0, env_value.split(","))
    buckets_int = map(int, buckets_remove_empty)
    buckets_list = list(buckets_int)
    return buckets_list


def _get_default_neuron_config(model_config: ModelConfig,
                               parallel_config: ParallelConfig,
                               scheduler_config: SchedulerConfig):
    from transformers_neuronx.config import ContinuousBatchingConfig
    from transformers_neuronx.constants import LAYOUT_BSH

    continuous_batching_config = ContinuousBatchingConfig(
        batch_size_for_shared_caches=scheduler_config.max_num_seqs)
    quant_config = dict(
        dequant_dtype=TORCH_DTYPE_TO_NEURON_AMP[model_config.dtype],
        quantize_method="vector_dynamic")
    neuron_quantization_config_builder = lambda quant: get_quantization_config(
        quant).from_config(quant_config).get_quant_method(None, "")
    # TODO: Add Paged attention config to the default neuron arguments.
    default_neuron_args = dict(
        collectives_layout=LAYOUT_BSH,
        attention_layout=LAYOUT_BSH,
        fuse_qkv=True,
        quant=neuron_quantization_config_builder(model_config.quantization)
        if model_config.quantization else None,
        continuous_batching=continuous_batching_config,
        weight_tiling=bool(model_config.quantization),
        on_device_generation=_get_neuron_on_device_generation_config(
            model_config))
    return default_neuron_args


def _get_neuron_on_device_generation_config(model_config: ModelConfig):
    if not _is_neuron_on_device_sampling_disabled(model_config):
        return copy.deepcopy(model_config.neuron_sampling_params)
    return None


def _is_neuron_on_device_sampling_disabled(model_config: ModelConfig) -> bool:
    return not getattr(model_config, "neuron_sampling_params", None)


def _get_neuron_config_after_override(default_neuron_config,
                                      overridden_neuron_config):
    from transformers_neuronx.config import (ContinuousBatchingConfig,
                                             GenerationConfig,
                                             KVCacheQuantizationConfig,
                                             NeuronConfig, QuantizationConfig,
                                             SparseAttnConfig)

    sparse_attn = overridden_neuron_config.pop("sparse_attn", {})
    if sparse_attn:
        overridden_neuron_config["sparse_attn"] = SparseAttnConfig(
            **sparse_attn)

    kv_cache_quant = overridden_neuron_config.pop("kv_cache_quant", {})
    if kv_cache_quant:
        overridden_neuron_config["kv_cache_quant"] = KVCacheQuantizationConfig(
            **kv_cache_quant)

    continuous_batching = overridden_neuron_config.pop("continuous_batching",
                                                       {})
    if continuous_batching:
        overridden_neuron_config[
            "continuous_batching"] = ContinuousBatchingConfig(
                **continuous_batching)

    quant = overridden_neuron_config.pop("quant", {})
    if quant:
        overridden_neuron_config["quant"] = QuantizationConfig(**quant)

    on_device_generation = overridden_neuron_config.pop(
        "on_device_generation", {})
    if on_device_generation:
        overridden_neuron_config["on_device_generation"] = GenerationConfig(
            **on_device_generation)
    default_neuron_config.update(overridden_neuron_config)
    return NeuronConfig(**default_neuron_config)


def get_neuron_model(model_config: ModelConfig,
                     parallel_config: ParallelConfig,
                     scheduler_config: SchedulerConfig) -> nn.Module:

    # Create a model instance.
    model = NeuronCausalLM(
        model_config.hf_config,
        _is_neuron_on_device_sampling_disabled(model_config))

    default_neuron_config_args = _get_default_neuron_config(
        model_config, parallel_config, scheduler_config)

    neuron_config = _get_neuron_config_after_override(
        default_neuron_config_args, model_config.override_neuron_config)

    context_length_estimates = _get_buckets("NEURON_CONTEXT_LENGTH_BUCKETS",
                                            [scheduler_config.max_model_len])
    n_positions = _get_buckets("NEURON_TOKEN_GEN_BUCKETS",
                               [scheduler_config.max_model_len])

    # Load the weights from the cached or downloaded files.
    model.load_weights(model_config.model,
                       tp_degree=parallel_config.tensor_parallel_size,
                       amp=TORCH_DTYPE_TO_NEURON_AMP[model_config.dtype],
                       neuron_config=neuron_config,
                       context_length_estimate=context_length_estimates,
                       n_positions=n_positions,
                       batch_size=scheduler_config.max_num_seqs)

    return model.eval()
