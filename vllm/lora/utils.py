from typing import List, Optional, Set, Tuple, Type

from torch import nn
from transformers import PretrainedConfig

from vllm.config import LoRAConfig
from vllm.logger import init_logger
from vllm.lora.fully_sharded_layers import (
    ColumnParallelLinearWithShardedLoRA,
    MergedColumnParallelLinearWithShardedLoRA,
    MergedQKVParallelLinearWithShardedLora, RowParallelLinearWithShardedLoRA)
# being imported for _all_lora_classes below
# yapf conflicts with isort for this block
# yapf: disable
from vllm.lora.layers import (BaseLayerWithLoRA, ColumnParallelLinearWithLoRA,
                              LogitsProcessorWithLoRA,
                              MergedColumnParallelLinearWithLoRA,
                              MergedQKVParallelLinearWithLora,
                              QKVParallelLinearWithLora,
                              RowParallelLinearWithLoRA,
                              VocabParallelEmbeddingWithLoRA)
# yapf: enable
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

logger = init_logger(__name__)

_all_lora_classes: Set[Type[BaseLayerWithLoRA]] = {
    VocabParallelEmbeddingWithLoRA, ColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithLoRA, QKVParallelLinearWithLora,
    MergedQKVParallelLinearWithLora, RowParallelLinearWithLoRA,
    LogitsProcessorWithLoRA, ColumnParallelLinearWithShardedLoRA,
    MergedColumnParallelLinearWithShardedLoRA,
    MergedQKVParallelLinearWithShardedLora, RowParallelLinearWithShardedLoRA
}


def from_layer(layer: nn.Module,
               max_loras: int,
               lora_config: LoRAConfig,
               packed_modules_list: List,
               model_config: Optional[PretrainedConfig] = None) -> nn.Module:
    for lora_cls in _all_lora_classes:
        # specifying kwargs so they can be easily accessed in decorator
        if lora_cls.can_replace_layer(source_layer=layer,
                                      lora_config=lora_config,
                                      packed_modules_list=packed_modules_list,
                                      model_config=model_config):
            ret = lora_cls(layer)
            ret.create_lora_weights(max_loras, lora_config, model_config)
            return ret
    return layer


def from_layer_logits_processor(
    layer: LogitsProcessor,
    lm_head: ParallelLMHead,
    max_loras: int,
    lora_config: LoRAConfig,
    model_config: Optional[PretrainedConfig] = None,
) -> LogitsProcessorWithLoRA:
    ret = LogitsProcessorWithLoRA(layer, lm_head.embedding_dim,
                                  lm_head.weight.dtype, lm_head.weight.device)
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
