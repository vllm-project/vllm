import logging
from typing import Tuple

from torch import nn

logger = logging.getLogger(__name__)


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
