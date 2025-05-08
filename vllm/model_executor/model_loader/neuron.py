# SPDX-License-Identifier: Apache-2.0
"""Utilities for selecting and loading Neuron models in transformers-neuronx
framework."""
import ast
import copy
import importlib
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import (ModelConfig, ParallelConfig, SchedulerConfig,
                         SpeculativeConfig)
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


class NeuronSpeculationCausalLM(nn.Module):
    """A Neuron-optimized causal language model with speculative decoding."""

    SPECULATION_TERMINATION_ID = -1

    def __init__(self, speculation_model) -> None:
        super().__init__()
        self.model = speculation_model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_block_ids: torch.Tensor,
    ) -> torch.Tensor:
        tokens, counts = self.model.speculative_iteration(
            input_ids, positions, input_block_ids)

        # Mark the end of accepted speculative tokens for each sequence with the
        # speculation termination id.
        batch_size, steps = tokens.shape
        mask = torch.arange(steps).expand(batch_size, -1) >= counts
        tokens[mask] = self.SPECULATION_TERMINATION_ID

        return tokens

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[List[SamplerOutput]]:
        batch_size, num_steps = logits.shape
        seq_ids = [
            seq_id for sg in sampling_metadata.seq_groups
            for seq_id in sg.seq_ids
        ]
        # Organize input tensors by step instead of by sequence.
        accepted_token_ids_by_step = logits.transpose(0, 1)
        accepted_token_ids_by_step = accepted_token_ids_by_step.tolist()

        sampler_output_list = []
        for step_index in range(num_steps):
            if all(token_id == self.SPECULATION_TERMINATION_ID
                   for token_id in accepted_token_ids_by_step[step_index]):
                break
            step_output_token_ids = []
            for sequence_index in range(batch_size):
                token_id = accepted_token_ids_by_step[step_index][
                    sequence_index]
                step_output_token_ids.append(
                    CompletionSequenceGroupOutput(samples=[
                        SequenceOutput(parent_seq_id=seq_ids[sequence_index],
                                       output_token=token_id,
                                       logprobs={token_id: Logprob(token_id)})
                    ],
                                                  prompt_logprobs=None))
            sampler_output_list.append(
                SamplerOutput(outputs=step_output_token_ids))
        return sampler_output_list


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
    """Generate a neuron config based on vllm config args."""
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


def _get_default_neuron_config_for_speculation(
        model_config: ModelConfig, parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig):
    """Generate a neuron config for speculative decoding based on
    vllm config args."""
    from transformers_neuronx.config import ContinuousBatchingConfig
    from transformers_neuronx.constants import LAYOUT_BSH

    continuous_batching_config = ContinuousBatchingConfig(
        batch_size_for_shared_caches=scheduler_config.max_num_seqs)

    default_neuron_args = dict(collectives_layout=LAYOUT_BSH,
                               attention_layout=LAYOUT_BSH,
                               fuse_qkv=True,
                               on_device_embedding=True,
                               continuous_batching=continuous_batching_config,
                               on_device_generation=copy.deepcopy(
                                   model_config.neuron_sampling_params))
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
    """Initializes a neuron-optimized model for inference."""
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

    model.load_weights(model_config.model,
                       tp_degree=parallel_config.tensor_parallel_size,
                       amp=TORCH_DTYPE_TO_NEURON_AMP[model_config.dtype],
                       neuron_config=neuron_config,
                       context_length_estimate=context_length_estimates,
                       n_positions=n_positions,
                       batch_size=scheduler_config.max_num_seqs)

    return model.eval()


def get_neuron_speculation_model(model_config: ModelConfig,
                                 parallel_config: ParallelConfig,
                                 scheduler_config: SchedulerConfig,
                                 speculation_config: SpeculativeConfig):
    """Initializes a neuron-optimized speculation model for inference.

    This method is only applicable for speculation with a standalone draft model
    """
    from transformers_neuronx.fused_speculation import FusedSpeculativeDecoder

    # For Eagle SD, we need to pass in additional parameters in neuron config.
    is_eagle = getattr(speculation_config.draft_model_config.hf_config,
                       "is_eagle", False)

    # Create target model instance.
    target_model = NeuronCausalLM(model_config.hf_config)

    default_neuron_config_args = _get_default_neuron_config_for_speculation(
        model_config, parallel_config, scheduler_config)
    if is_eagle:
        default_neuron_config_args['is_eagle_target'] = True

    neuron_config = _get_neuron_config_after_override(
        default_neuron_config_args, model_config.override_neuron_config)

    context_length_estimates = _get_buckets("NEURON_CONTEXT_LENGTH_BUCKETS",
                                            [scheduler_config.max_model_len])
    n_positions = _get_buckets("NEURON_TOKEN_GEN_BUCKETS",
                               [scheduler_config.max_model_len])

    target_model.load_weights(
        model_config.model,
        tp_degree=parallel_config.tensor_parallel_size,
        amp=TORCH_DTYPE_TO_NEURON_AMP[model_config.dtype],
        neuron_config=neuron_config,
        context_length_estimate=context_length_estimates,
        n_positions=n_positions,
        batch_size=scheduler_config.max_num_seqs)

    target_model.eval()

    # Create draft model instance.
    draft_model = NeuronCausalLM(
        speculation_config.draft_model_config.hf_config)

    default_draft_neuron_config_args = (
        _get_default_neuron_config_for_speculation(
            speculation_config.draft_model_config, parallel_config,
            scheduler_config))
    if is_eagle:
        default_draft_neuron_config_args['is_eagle_draft'] = True
        default_draft_neuron_config_args['has_pre_attention_norm'] = False

    draft_neuron_config = _get_neuron_config_after_override(
        default_draft_neuron_config_args,
        speculation_config.draft_model_config.override_neuron_config)

    draft_model.load_weights(speculation_config.draft_model_config.model,
                             tp_degree=speculation_config.
                             draft_parallel_config.tensor_parallel_size,
                             amp=TORCH_DTYPE_TO_NEURON_AMP[
                                 speculation_config.draft_model_config.dtype],
                             neuron_config=draft_neuron_config,
                             context_length_estimate=context_length_estimates,
                             n_positions=n_positions,
                             batch_size=scheduler_config.max_num_seqs)

    draft_model.eval()

    num_speculative_tokens = speculation_config.num_speculative_tokens
    # Create speculation model instance.
    speculation_model = FusedSpeculativeDecoder(draft_model.model,
                                                target_model.model,
                                                num_speculative_tokens)
    speculation_model.to_neuron()

    return NeuronSpeculationCausalLM(speculation_model)


def get_neuron_eagle_speculation_model(model_config: ModelConfig,
                                       parallel_config: ParallelConfig,
                                       scheduler_config: SchedulerConfig,
                                       speculation_config: SpeculativeConfig):
    """Initializes a neuron-optimized EAGLE speculation model for inference."""
    from transformers_neuronx.eagle_speculation import EagleSpeculativeDecoder

    # Create target model instance.
    target_model = NeuronCausalLM(model_config.hf_config)

    default_neuron_config_args = _get_default_neuron_config_for_speculation(
        model_config, parallel_config, scheduler_config)
    default_neuron_config_args['is_eagle_target'] = True
    neuron_config = _get_neuron_config_after_override(
        default_neuron_config_args, model_config.override_neuron_config)

    context_length_estimates = _get_buckets("NEURON_CONTEXT_LENGTH_BUCKETS",
                                            [scheduler_config.max_model_len])
    n_positions = _get_buckets("NEURON_TOKEN_GEN_BUCKETS",
                               [scheduler_config.max_model_len])

    target_model.load_weights(
        model_config.model,
        tp_degree=parallel_config.tensor_parallel_size,
        amp=TORCH_DTYPE_TO_NEURON_AMP[model_config.dtype],
        neuron_config=neuron_config,
        context_length_estimate=context_length_estimates,
        n_positions=n_positions,
        batch_size=scheduler_config.max_num_seqs)

    target_model.eval()

    # Create draft model instance.
    draft_model = NeuronCausalLM(
        speculation_config.draft_model_config.hf_config)

    default_draft_neuron_config_args = (
        _get_default_neuron_config_for_speculation(
            speculation_config.draft_model_config, parallel_config,
            scheduler_config))
    default_draft_neuron_config_args['is_eagle_draft'] = True
    default_draft_neuron_config_args['has_pre_attention_norm'] = False
    draft_neuron_config = _get_neuron_config_after_override(
        default_draft_neuron_config_args,
        speculation_config.draft_model_config.override_neuron_config)

    draft_model.load_weights(speculation_config.draft_model_config.model,
                             tp_degree=speculation_config.
                             draft_parallel_config.tensor_parallel_size,
                             amp=TORCH_DTYPE_TO_NEURON_AMP[
                                 speculation_config.draft_model_config.dtype],
                             neuron_config=draft_neuron_config,
                             context_length_estimate=context_length_estimates,
                             n_positions=n_positions,
                             batch_size=scheduler_config.max_num_seqs)

    draft_model.eval()

    token_tree: Dict[int, List[int]] = ast.literal_eval(
        speculation_config.speculative_token_tree)

    speculation_model = EagleSpeculativeDecoder(draft_model.model,
                                                target_model.model,
                                                token_tree=token_tree)
    speculation_model.to_neuron()

    return NeuronSpeculationCausalLM(speculation_model)
