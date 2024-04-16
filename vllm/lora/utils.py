import logging
from typing import Tuple, Optional
from vllm.config import LoRAConfig
from transformers import PretrainedConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear,
                                               QKVParallelLinear,
                                               MergedColumnParallelLinear)
from vllm.lora.layers import (
    BaseLayerWithLoRA, VocabParallelEmbeddingWithLoRA,
    ColumnParallelLinearWithLoRA, RowParallelLinearWithLoRA,
    QKVParallelLinearWithLora, MergedColumnParallelLinearWithLoRA,
    SamplerWithLoRA)
from vllm.lora.fully_sharded_layers import (
    ColumnParallelLinearWithShardedLoRA, RowParallelLinearWithShardedLoRA,
    QKVParallelLinearWithShardedLora,
    MergedColumnParallelLinearWithShardedLoRA)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (ParallelLMHead
                                                                 )

from torch import nn

logger = logging.getLogger(__name__)


def from_layer(
        layer: nn.Module,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: Optional[PretrainedConfig] = None) -> BaseLayerWithLoRA:
    if lora_config.fully_sharded_loras:
        supported_layer_types = {
            VocabParallelEmbedding: VocabParallelEmbeddingWithLoRA,
            ColumnParallelLinear: ColumnParallelLinearWithShardedLoRA,
            QKVParallelLinear: QKVParallelLinearWithShardedLora,
            MergedColumnParallelLinear:
            MergedColumnParallelLinearWithShardedLoRA,
            RowParallelLinear: RowParallelLinearWithShardedLoRA,
        }
    else:
        supported_layer_types = {
            VocabParallelEmbedding: VocabParallelEmbeddingWithLoRA,
            ColumnParallelLinear: ColumnParallelLinearWithLoRA,
            QKVParallelLinear: QKVParallelLinearWithLora,
            MergedColumnParallelLinear: MergedColumnParallelLinearWithLoRA,
            RowParallelLinear: RowParallelLinearWithLoRA,
        }
    for src_layer_type, lora_layer_type in supported_layer_types.items():
        if type(layer) is src_layer_type:  # pylint: disable=unidiomatic-typecheck
            ret = lora_layer_type(layer)
            ret.create_lora_weights(max_loras, lora_config, model_config)
            return ret
    return layer


def from_layer_sampler(
    layer: Sampler,
    lm_head: ParallelLMHead,
    max_loras: int,
    lora_config: LoRAConfig,
    model_config: Optional[PretrainedConfig] = None,
) -> SamplerWithLoRA:
    ret = SamplerWithLoRA(layer, lm_head.embedding_dim, lm_head.weight.dtype,
                          lm_head.weight.device)
    ret.create_lora_weights(max_loras, lora_config, model_config)
    return ret


def replace_submodule(model: nn.Module, module_name: str,
                      new_module: nn.Module) -> nn.Module:
    """Replace a submodule in a model with a new module."""
    parent = model.get_submodule(".".join(module_name.split(".")[:-1]))
    target_name = module_name.split(".")[-1]
    setattr(parent, target_name, new_module)
    return new_module


def parse_fine_tuned_lora_name(name: str) -> Tuple[str, bool]:
    """Parse the name of lora weights.

    args:
        name: the name of the fine-tuned LoRA, e.g.
            base_model.model.dense1.weight
    return:
        Tuple(module_name, is_lora_a):
            module_name: the name of the module, e.g. model.dense1,
            is_lora_a whether the tensor is lora_a or lora_b.
    """
    parts = name.split(".")
    assert parts[0] == "base_model"
    assert parts[1] == "model"
    if parts[-1] == "weight":
        assert parts[-2] == "lora_A" or parts[-2] == "lora_B"
        return ".".join(parts[2:-2]), parts[-2] == "lora_A"

    if parts[-1] == "lora_embedding_A" or parts[-1] == "lora_embedding_B":
        return ".".join(parts[2:-1]), parts[-1] == "lora_embedding_A"

    raise ValueError(f"{name} is unsupported format")
