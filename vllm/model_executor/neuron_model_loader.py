"""Utilities for selecting and loading models."""
from typing import Type

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import ModelConfig, DeviceConfig
from vllm.model_executor.models import ModelRegistry

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


def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        model_cls = ModelRegistry.load_model_cls(arch)
        if model_cls is not None:
            return model_cls
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {ModelRegistry.get_supported_archs()}")


def get_model(model_config: ModelConfig, device_config: DeviceConfig,
              **kwargs) -> nn.Module:
    from transformers_neuronx.config import NeuronConfig, ContinuousBatchingConfig

    parallel_config = kwargs.get("parallel_config")
    scheduler_config = kwargs.get("scheduler_config")

    model_class = _get_model_architecture(model_config.hf_config)
    linear_method = None

    # Create a model instance.
    model = model_class(model_config.hf_config, linear_method)

    continuous_batching_config = ContinuousBatchingConfig(
        batch_size_for_shared_caches=scheduler_config.max_num_seqs)
    neuron_config = NeuronConfig(
        continuous_batching=continuous_batching_config)

    # Load the weights from the cached or downloaded files.
    model.load_weights(
        model_config.model,
        model_config.download_dir,
        model_config.load_format,
        model_config.revision,
        tp_degree=parallel_config.neuron_tp_degree,
        amp=TORCH_DTYPE_TO_NEURON_AMP[model_config.dtype],
        neuron_config=neuron_config,
        context_length_estimate=[scheduler_config.max_model_len],
        n_positions=[scheduler_config.max_model_len],
        batch_size=scheduler_config.max_num_seqs)

    return model.eval()
