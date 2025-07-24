# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
A model loader implementation for NeuronX Distributed Inference (NxDI).

This class serves as the primary interface for loading and managing 
machine learning models optimized for AWS Neuron hardware. It provides 
functionality for:
    - Loading pre-trained models and their configurations
    - Managing model compilation
    - Handling distributed inference across multiple Neuron cores
    - Supporting various model architectures and configurations
    - Managing key-value caches for optimized inference
    - Implementing sampling strategies for model outputs

The loader supports various model architectures and can be extended to handle
different model types and configurations. It integrates with the broader 
vLLM framework while providing specific optimizations for AWS Neuron hardware.
"""

import collections
import hashlib
import os
import shutil
from contextlib import contextmanager
from math import ceil
from pathlib import Path
from typing import Any, Optional

import regex as re
import torch
import torch.nn as nn
from neuronx_distributed_inference.models.config import (
    ChunkedPrefillConfig, NeuronConfig, OnDeviceSamplingConfig)
from neuronx_distributed_inference.modules.lora_serving import (
    LoraServingConfig)
from neuronx_distributed_inference.utils.constants import MODEL_TYPES
from neuronx_distributed_inference.utils.hf_adapter import (
    load_pretrained_config)
from transformers import AutoModelForCausalLM, PretrainedConfig

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler
from vllm.v1.outputs import SamplerOutput

logger = init_logger(__name__)

TORCH_DTYPE_TO_NEURON_AMP = {
    "auto": "float32",
    "half": "float16",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float": "float32",
    "float32": "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


class NeuronModelBase(nn.Module):
    """
    Base class for all Neuron models.
    It is used to load the model, run the model, and sample the model.
    It is also used to get the KV caches.
    """

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.logits_processor = LogitsProcessor(
            config.get_text_config().vocab_size, logits_as_input=True)
        self.on_device_sampling_disabled = bool(
            int(os.getenv("NEURON_ON_DEVICE_SAMPLING_DISABLED", "0")))
        if self.on_device_sampling_disabled:
            self.sampler = Sampler()

        # Lazy initialized
        self.model: nn.Module
        self.kv_caches: Optional[list[Any]] = None
        self.neuron_config: NeuronConfig
        self.is_reorder_needed: bool
        self.architecture: str
        self.num_key_value_heads: int
        self.head_size: int
        self.dtype: torch.dtype

    def forward(self, input_ids, positions, input_block_ids, sampling_params,
                **kwargs):
        raise NotImplementedError

    def sample(self, logits: torch.Tensor) -> Optional[SamplerOutput]:
        raise NotImplementedError

    def load_weights(self, model_name_or_path: str, architecture: str,
                     **kwargs):
        raise NotImplementedError

    def get_kv_caches(self):
        if self.kv_caches is None:
            kv_caches = []
            tp_tensors_map = collections.defaultdict(list)
            state = self.model.context_encoding_model.model.nxd_model.state

            for tp_idx, per_tp_state in enumerate(state):
                for key, val in per_tp_state.items():
                    tp_tensors_map[tp_idx].append(val)

            for i in range(len(tp_tensors_map[0])):
                for tp, tensors in tp_tensors_map.items():
                    kv_caches.append(tensors[i])
            self.kv_caches = kv_caches

        return self.kv_caches

    @contextmanager
    def _reordered(self, input_block_ids: torch.Tensor, **tensor_inputs):
        """
        Context manager that yields reordered input_block_ids, inputs, and a 
        restore function. Automatically restores output to original order 
        if needed.
        
        [NOTE] This is MANADATORY for contiguous kv cache as it will impact 
        the output accuracy.
        
        TODO: This sequence id reordering is better to live in NxD-Inference.
        """
        logger.debug("is_reorder_needed: %s", self.is_reorder_needed)
        if self.is_reorder_needed:
            sorted_ids, sorted_indices = torch.sort(input_block_ids)
            reordered_inputs = {
                k: (
                    torch.index_select(v, 0, sorted_indices)
                    # having v.shape[0] > 0 to avoid reorder empty tensors
                    if isinstance(v, torch.Tensor) and v.shape[0] > 0 else v)
                for k, v in tensor_inputs.items()
            }

            def restore(output: torch.Tensor) -> torch.Tensor:
                if sorted_ids.shape[0] != 1:
                    return torch.index_select(output, 0,
                                              torch.argsort(sorted_indices))
                return output

            yield sorted_ids, reordered_inputs, restore
        else:
            yield input_block_ids, tensor_inputs, lambda x: x

    def _load_weights_common(self, model_name_or_path: str, neuronx_model_cls,
                             **kwargs):
        neuron_config = neuronx_model_cls.get_neuron_config_cls()(
            **kwargs['neuron_config'])
        config = neuronx_model_cls.get_config_cls()(
            neuron_config,
            load_config=load_pretrained_config(model_name_or_path))

        hashed_config = hashlib.md5(
            config.to_json_string().encode('utf-8')).hexdigest()
        compiled_model_path = self._get_compiled_model_path(
            model_name_or_path, hashed_config)

        try:
            self._load_compiled_model(compiled_model_path, neuronx_model_cls,
                                      kwargs)
            return True, compiled_model_path, config
        except (FileNotFoundError, ValueError) as e:
            logger.warning("Exception: %s", e)
            logger.warning("Failed to load from %s. Recompiling...",
                           compiled_model_path)
            return False, compiled_model_path, config

    def _get_compiled_model_path(self, model_name_or_path: str,
                                 hashed_config: str):
        if os.getenv("NEURON_COMPILED_ARTIFACTS"):
            return os.getenv("NEURON_COMPILED_ARTIFACTS")
        elif os.path.exists(model_name_or_path):
            path = Path(model_name_or_path
                        ) / "neuron-compiled-artifacts" / hashed_config
            path.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(path, ignore_errors=True)
            return path
        else:
            path = Path(
                "local-models"
            ) / model_name_or_path / "neuron-compiled-artifacts" / hashed_config
            path.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(path, ignore_errors=True)
            return path

    def _load_compiled_model(self, compiled_model_path: str, neuronx_model_cls,
                             kwargs):
        self.model = neuronx_model_cls(compiled_model_path)
        override_neuron_config = kwargs.get("override_neuron_config")
        if override_neuron_config:
            for k, v in override_neuron_config.items():
                setattr(self.model.config.neuron_config, k, v)
        self.model.load(compiled_model_path)
        logger.info("Successfully loaded precompiled model artifacts from %s",
                    compiled_model_path)

    def _save_pretrained_model(self, model_name: str):
        hf_model = AutoModelForCausalLM.from_pretrained(model_name)
        saved_path = os.path.join("local-models", model_name)
        hf_model.save_pretrained(saved_path)
        return saved_path

    def _compile_and_load_model(self, model_path: str, neuronx_model_cls,
                                config, compiled_path: str):
        self.model = neuronx_model_cls(model_path, config)
        self.model.compile(compiled_path)
        self.model.load(compiled_path)


class NeuronCausalLM(NeuronModelBase):

    def forward(self, input_ids, input_block_ids, **kwargs):
        with self._reordered(input_block_ids, input_ids=input_ids,
                             **kwargs) as (sorted_ids, inputs, restore):
            output = self.model(
                inputs['input_ids'],
                attention_mask=None,
                seq_ids=sorted_ids,
                block_table=inputs['block_tables'],
                **{
                    k: v
                    for k, v in inputs.items() if k not in
                    ['input_ids', 'block_tables', 'prefill_completion_state']
                })

            if self.model.config.neuron_config.on_device_sampling_config:
                output = output.hidden_states
            else:
                if self.neuron_config.is_chunked_prefill:
                    assert kwargs.get('prefill_completion_state') is not None
                    idx_for_sampling = kwargs[
                        'prefill_completion_state'].nonzero().flatten()
                    output = output.logits[0, idx_for_sampling, :]
                else:
                    output = output.logits[:, -1, :]

            return restore(output)

    def sample(self, logits: torch.Tensor) -> Optional[SamplerOutput]:
        if self.model.config.neuron_config.on_device_sampling_config:
            return SamplerOutput(
                # The sampled tokens are expanded to 2D tensor with shape
                # [num_requests, 1], where each row represents one generated
                # token per request.
                sampled_token_ids=logits.unsqueeze(-1),
                logprobs_tensors=None,
            )
        else:
            raise NotImplementedError("CPU sampler not implemented")

    def load_weights(self, model_name_or_path: str, architecture: str,
                     **kwargs):
        neuronx_model_cls = _get_neuron_model_cls(architecture)
        success, compiled_model_path, config = self._load_weights_common(
            model_name_or_path, neuronx_model_cls, **kwargs)

        if not success:
            if not os.path.exists(model_name_or_path):
                model_name_or_path = self._save_pretrained_model(
                    model_name_or_path)
            self._compile_and_load_model(model_name_or_path, neuronx_model_cls,
                                         config, compiled_model_path)


def _get_model_configs(config: PretrainedConfig) -> str:
    archs = getattr(config, "architectures", [])
    num_key_value_heads = getattr(config, "num_key_value_heads", None)
    head_dim = getattr(config, "head_dim", None)
    if not archs or not num_key_value_heads or not head_dim:
        raise ValueError("Missing required fields in the model config.")
    return archs[0], int(num_key_value_heads), int(head_dim)


def _camel_to_kebab(name: str) -> str:
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1-\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1-\2', s1).lower()


def _get_neuron_model_cls(architecture: str):
    try:
        if "For" in architecture:
            model, task = architecture.split("For", 1)
            model, task = model.lower(), _camel_to_kebab(task)

            if architecture == "MllamaForConditionalGeneration":
                task = "causal-lm"

            return MODEL_TYPES[model][task]
        else:
            raise KeyError
    except KeyError:
        raise ValueError(
            "Model %s is not supported on Neuron for now. "
            "Supported models: %s", architecture,
            list(MODEL_TYPES.keys())) from None


def get_neuron_model(model_config: ModelConfig, cache_config: CacheConfig,
                     parallel_config: ParallelConfig,
                     scheduler_config: SchedulerConfig,
                     lora_serving_config: LoraServingConfig) -> nn.Module:
    # TODO: support other models
    model = NeuronCausalLM(model_config.hf_config)

    default_neuron_config_args = _get_default_neuron_config(
        model_config, cache_config, parallel_config, scheduler_config,
        lora_serving_config)

    neuron_config = _get_neuron_config_after_override(
        default_neuron_config_args, model_config.override_neuron_config)

    neuron_config = _validate_neuron_config(cache_config, scheduler_config,
                                            neuron_config)

    override_neuron_config = model_config.override_neuron_config
    architecture, num_key_value_heads, head_dim = _get_model_configs(
        model_config.hf_config)
    model.load_weights(model_name_or_path=model_config.model,
                       architecture=architecture,
                       neuron_config=neuron_config,
                       override_neuron_config=override_neuron_config)
    model.neuron_config = model.model.config.neuron_config
    model.architecture = architecture
    model.num_key_value_heads = num_key_value_heads
    model.head_dim = head_dim
    return model.eval()


# Helper functions for getting default configs
def _get_default_neuron_config(model_config: ModelConfig,
                               cache_config: CacheConfig,
                               parallel_config: ParallelConfig,
                               scheduler_config: SchedulerConfig,
                               lora_serving_config: LoraServingConfig):
    on_device_sampling_config = OnDeviceSamplingConfig(dynamic=True,
                                                       deterministic=False)

    if scheduler_config.chunked_prefill_enabled:
        batch_size = 1
        max_context_length = scheduler_config.max_num_batched_tokens
    else:
        batch_size = scheduler_config.max_num_seqs
        max_context_length = scheduler_config.max_model_len

    default_num_blocks = ceil(
        scheduler_config.max_model_len //
        cache_config.block_size) * scheduler_config.max_num_seqs
    if cache_config.num_gpu_blocks_override is not None:
        default_num_blocks = cache_config.num_gpu_blocks_override

    return {
        "tp_degree": parallel_config.tensor_parallel_size,
        "ctx_batch_size": 1,
        "batch_size": batch_size,
        "max_context_length": max_context_length,
        "seq_len": scheduler_config.max_model_len,
        "enable_bucketing": True,
        "is_continuous_batching": (batch_size > 1),
        "quantized": False,
        "torch_dtype": TORCH_DTYPE_TO_NEURON_AMP[model_config.dtype],
        "padding_side": "right",
        "on_device_sampling_config": on_device_sampling_config,
        "lora_config": lora_serving_config,
        "pa_num_blocks": default_num_blocks,
        "pa_block_size": cache_config.block_size,
        "is_block_kv_layout": (
            scheduler_config.chunked_prefill_enabled or \
            cache_config.enable_prefix_caching
        ),
        "is_prefix_caching": cache_config.enable_prefix_caching,
    }


def _validate_neuron_config(cache_config: CacheConfig,
                            scheduler_config: SchedulerConfig,
                            neuron_config: dict):
    if cache_config.enable_prefix_caching:
        assert neuron_config.get("is_prefix_caching", False)
        assert neuron_config.get("is_block_kv_layout", False)

    if scheduler_config.chunked_prefill_enabled:
        assert neuron_config.get("chunked_prefill_config")
        assert neuron_config.get("is_block_kv_layout", False)

    logger.debug("Neuron Config: %s", neuron_config)
    return neuron_config


def _get_neuron_config_after_override(default_neuron_config,
                                      overridden_neuron_config):
    overridden_neuron_config = overridden_neuron_config or {}
    cfg = overridden_neuron_config.pop("chunked_prefill_config", None)
    if cfg:
        overridden_neuron_config[
            "chunked_prefill_config"] = ChunkedPrefillConfig(**cfg)
    default_neuron_config.update(overridden_neuron_config)
    return default_neuron_config
