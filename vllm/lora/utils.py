# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import TYPE_CHECKING, Optional

import huggingface_hub
from huggingface_hub.utils import (
    EntryNotFoundError,
    HfHubHTTPError,
    HFValidationError,
    RepositoryNotFoundError,
)
from torch import nn
from transformers import PretrainedConfig

from vllm.config.lora import LoRAConfig
from vllm.logger import init_logger

# being imported for _all_lora_classes below
from vllm.lora.layers import (
    BaseLayerWithLoRA,
    ColumnParallelLinearWithLoRA,
    ColumnParallelLinearWithShardedLoRA,
    FusedMoE3DWithLoRA,
    FusedMoEWithLoRA,
    LogitsProcessorWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithShardedLoRA,
    MergedQKVParallelLinearWithLoRA,
    MergedQKVParallelLinearWithShardedLoRA,
    QKVParallelLinearWithLoRA,
    QKVParallelLinearWithShardedLoRA,
    ReplicatedLinearWithLoRA,
    RowParallelLinearWithLoRA,
    RowParallelLinearWithShardedLoRA,
    VocabParallelEmbeddingWithLoRA,
)
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.utils import get_moe_expert_mapping, get_packed_modules_mapping

if TYPE_CHECKING:
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

_GLOBAL_LORA_ID = 0


def get_lora_id():
    global _GLOBAL_LORA_ID
    _GLOBAL_LORA_ID += 1
    return _GLOBAL_LORA_ID


_all_lora_classes: set[type[BaseLayerWithLoRA]] = {
    VocabParallelEmbeddingWithLoRA,
    ColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    QKVParallelLinearWithLoRA,
    MergedQKVParallelLinearWithLoRA,
    RowParallelLinearWithLoRA,
    ReplicatedLinearWithLoRA,
    LogitsProcessorWithLoRA,
    ColumnParallelLinearWithShardedLoRA,
    QKVParallelLinearWithShardedLoRA,
    MergedColumnParallelLinearWithShardedLoRA,
    MergedQKVParallelLinearWithShardedLoRA,
    RowParallelLinearWithShardedLoRA,
    FusedMoEWithLoRA,
    FusedMoE3DWithLoRA,
}


def is_moe_model(model: nn.Module) -> bool:
    """Checks if the model contains FusedMoE layers and warns the user."""
    if any(isinstance(module, FusedMoE) for module in model.modules()):
        logger.info_once("MoE model detected. Using fused MoE LoRA implementation.")
        return True
    return False


def from_layer(
    layer: nn.Module,
    max_loras: int,
    lora_config: LoRAConfig,
    packed_modules_list: list,
    model_config: PretrainedConfig | None = None,
) -> nn.Module:
    for lora_cls in _all_lora_classes:
        # specifying kwargs so they can be easily accessed in decorator
        if lora_cls.can_replace_layer(
            source_layer=layer,
            lora_config=lora_config,
            packed_modules_list=packed_modules_list,
            model_config=model_config,
        ):
            instance_layer = lora_cls(layer)
            instance_layer.create_lora_weights(max_loras, lora_config, model_config)
            return instance_layer
    return layer


def from_layer_logits_processor(
    layer: "LogitsProcessor",
    lm_head: "ParallelLMHead",
    max_loras: int,
    lora_config: LoRAConfig,
    model_config: PretrainedConfig | None = None,
) -> LogitsProcessorWithLoRA:
    ret = LogitsProcessorWithLoRA(
        layer,
        lm_head.embedding_dim,
        lm_head.weight.dtype,
        lm_head.weight.device,
        lm_head.get_sharded_to_full_mapping(),
    )
    ret.create_lora_weights(max_loras, lora_config, model_config)
    return ret


def replace_submodule(
    model: nn.Module, module_name: str, new_module: nn.Module
) -> nn.Module:
    """Replace a submodule in a model with a new module."""
    parent = model.get_submodule(".".join(module_name.split(".")[:-1]))
    target_name = module_name.split(".")[-1]
    setattr(parent, target_name, new_module)
    return new_module


def parse_fine_tuned_lora_name(
    name: str, weights_mapper: Optional["WeightsMapper"] = None
) -> tuple[str, bool]:
    """Parse the name of lora weights.

    args:
        name: the name of the fine-tuned LoRA, e.g.
            base_model.model.dense1.weight
        weights_mapper: maps the name of weight, e.g.
            `model.` -> `language_model.model.`,
    return:
        tuple(module_name, is_lora_a):
            module_name: the name of the module, e.g. model.dense1,
            is_lora_a whether the tensor is lora_a or lora_b.
    """

    # LoRA weight qualified name usually starts with `base_model.model.`,
    # so we remove the prefix `base_model.model.` to make the following
    # mapping correctly.
    if name.startswith("base_model.model."):
        name = name.replace("base_model.model.", "")
        name = weights_mapper._map_name(name) if weights_mapper else name
        # recover the prefix `base_model.model.`
        name = "base_model.model." + name
    else:
        name = weights_mapper._map_name(name) if weights_mapper else name

    # In some situations, we may not start with `base_model.model.`.
    # If we don't (e.g., ibm-granite/granite-speech-3.3-8b),
    # we should keep the prefix intact.
    start_index = 2 if name.startswith("base_model.model.") else 0

    parts = name.split(".")
    if parts[-1] == "weight" and (parts[-2] == "lora_A" or parts[-2] == "lora_B"):
        new_name = ".".join(parts[start_index:-2])
        return new_name, parts[-2] == "lora_A"

    if parts[-1] == "lora_embedding_A" or parts[-1] == "lora_embedding_B":
        new_name = ".".join(parts[start_index:-1])
        return new_name, parts[-1] == "lora_embedding_A"

    raise ValueError(f"{name} is unsupported LoRA weight")


def is_base_embeddding_weights(name: str) -> bool:
    # hardcoded subfixes for input & output embedding weights
    embedding_suffixes = (
        ".embed_tokens.base_layer.weight",
        ".lm_head.base_layer.weight",
    )
    return name.endswith(embedding_suffixes)


def get_supported_lora_modules(model: nn.Module) -> list[str]:
    """
    In vLLM, all linear layers support LoRA.
    """

    supported_lora_modules: set[str] = set()
    for name, module in model.named_modules():
        # get the embedding modules if the module's embedding_modules
        # is not empty.
        embedding_modules = getattr(module, "embedding_modules", None)
        if embedding_modules is not None:
            for name in embedding_modules:
                supported_lora_modules.add(name)

        # get all the linear subfixes.
        if isinstance(module, (LinearBase,)):
            supported_lora_modules.add(name.split(".")[-1])

        if isinstance(module, (FusedMoE,)):
            supported_lora_modules.add(name.split(".")[-1])

    return list(supported_lora_modules)


def get_adapter_absolute_path(lora_path: str) -> str:
    """
    Resolves the given lora_path to an absolute local path.

    If the lora_path is identified as a Hugging Face model identifier,
    it will download the model and return the local snapshot path.
    Otherwise, it treats the lora_path as a local file path and
    converts it to an absolute path.

    Parameters:
    lora_path (str): The path to the lora model, which can be an absolute path,
                     a relative path, or a Hugging Face model identifier.

    Returns:
    str: The resolved absolute local path to the lora model.
    """

    # Check if the path is an absolute path. Return it no matter exists or not.
    if os.path.isabs(lora_path):
        return lora_path

    # If the path starts with ~, expand the user home directory.
    if lora_path.startswith("~"):
        return os.path.expanduser(lora_path)

    # Check if the expanded relative path exists locally.
    if os.path.exists(lora_path):
        return os.path.abspath(lora_path)

    # If the path does not exist locally, assume it's a Hugging Face repo.
    try:
        local_snapshot_path = huggingface_hub.snapshot_download(repo_id=lora_path)
    except (
        HfHubHTTPError,
        RepositoryNotFoundError,
        EntryNotFoundError,
        HFValidationError,
    ):
        # Handle errors that may occur during the download
        # Return original path instead of throwing error here
        logger.exception("Error downloading the HuggingFace model")
        return lora_path

    return local_snapshot_path


def process_packed_modules_mapping(model: nn.Module) -> dict[str, list[str]]:
    if is_moe_model(model):
        if moe_packed_mapping := get_moe_expert_mapping(model):
            # This method generates and returns a dictionary mapping packed module
            # names to lists of their corresponding submodule names. It includes
            # both static mappings and dynamic mappings for expert layers, where
            # the expert indices are expanded based on the configured number
            # of routed experts.
            packed_modules_mapping = get_packed_modules_mapping(model)
            if not model.is_3d_moe_weight:
                # 3D MoE LoRA does not need `packed_modules_mapping`
                packed_modules_mapping["experts"] = [
                    weight_name.rstrip(".")
                    for _, weight_name, _, _ in moe_packed_mapping
                ]

            return packed_modules_mapping
        else:
            raise AttributeError(
                "To support LoRA for MoE model, "
                "'get_expert_mapping' must be implemented"
            )
    else:
        return get_packed_modules_mapping(model)
