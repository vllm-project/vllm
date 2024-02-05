"""Utilities for selecting and loading models."""
import contextlib
from typing import Type

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import ModelConfig
from vllm.model_executor.models import ModelRegistry


@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        model_cls = ModelRegistry.load_model_cls(arch)
        if model_cls is not None:
            return model_cls
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {ModelRegistry.get_supported_archs()}")


def get_model(model_config: ModelConfig, parallel_config,
              scheduler_config) -> nn.Module:
    model_class = _get_model_architecture(model_config.hf_config)
    linear_method = None

    with _set_default_torch_dtype(model_config.dtype):
        # Create a model instance.
        # The weights will be initialized as empty tensors.
        with torch.device("cuda"):
            model = model_class(model_config.hf_config, linear_method)
        if model_config.load_format == "dummy":
            # NOTE(woosuk): For accurate performance evaluation, we assign
            # random values to the weights.
            for param in model.state_dict().values():
                if torch.is_floating_point(param):
                    param.data.uniform_(-1e-3, 1e-3)
        else:
            # Load the weights from the cached or downloaded files.
            from transformers_neuronx.config import NeuronConfig, ContinuousBatchingConfig

            continuous_batching_config = ContinuousBatchingConfig(
                batch_size_for_shared_caches=scheduler_config.max_num_seqs)
            neuron_config = NeuronConfig(
                continuous_batching=continuous_batching_config)
            model.load_weights(
                model_config.model,
                model_config.download_dir,
                model_config.load_format,
                model_config.revision,
                tp_degree=parallel_config.neuron_tp_degree,
                amp='f32',
                neuron_config=neuron_config,
                context_length_estimate=[scheduler_config.max_model_len],
                n_positions=[scheduler_config.max_model_len],
                batch_size=scheduler_config.max_num_seqs)
    return model.eval()
