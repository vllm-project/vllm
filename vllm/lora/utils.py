import os
import re
from typing import TYPE_CHECKING, List, Optional, Set, Tuple, Type, Union

import huggingface_hub
import torch
from huggingface_hub.utils import (EntryNotFoundError, HfHubHTTPError,
                                   HFValidationError, RepositoryNotFoundError)
from torch import nn
from transformers import PretrainedConfig

from vllm.config import LoRAConfig
from vllm.logger import init_logger
from vllm.lora.fully_sharded_layers import (
    ColumnParallelLinearWithShardedLoRA,
    MergedColumnParallelLinearWithShardedLoRA,
    MergedQKVParallelLinearWithShardedLora, QKVParallelLinearWithShardedLora,
    RowParallelLinearWithShardedLoRA)
# being imported for _all_lora_classes below
# yapf conflicts with isort for this block
# yapf: disable
from vllm.lora.layers import (BaseLayerWithLoRA, ColumnParallelLinearWithLoRA,
                              LinearScalingRotaryEmbeddingWithLora,
                              LogitsProcessorWithLoRA,
                              MergedColumnParallelLinearWithLoRA,
                              MergedQKVParallelLinearWithLora,
                              QKVParallelLinearWithLora,
                              ReplicatedLinearWithLoRA,
                              RowParallelLinearWithLoRA,
                              VocabParallelEmbeddingWithLoRA)
# yapf: enable
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

if TYPE_CHECKING:
    # avoid circuit import
    from vllm.lora.layers import LoRAMapping
    from vllm.lora.models import LongContextLoRAContext

logger = init_logger(__name__)

_all_lora_classes: Set[Type[BaseLayerWithLoRA]] = {
    VocabParallelEmbeddingWithLoRA,
    ColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    QKVParallelLinearWithLora,
    MergedQKVParallelLinearWithLora,
    RowParallelLinearWithLoRA,
    ReplicatedLinearWithLoRA,
    LogitsProcessorWithLoRA,
    ColumnParallelLinearWithShardedLoRA,
    QKVParallelLinearWithShardedLora,
    MergedColumnParallelLinearWithShardedLoRA,
    MergedQKVParallelLinearWithShardedLora,
    RowParallelLinearWithShardedLoRA,
    LinearScalingRotaryEmbeddingWithLora,
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
                                  lm_head.weight.dtype, lm_head.weight.device,
                                  lm_head.get_sharded_to_full_mapping())
    ret.create_lora_weights(max_loras, lora_config, model_config)
    return ret


def replace_submodule(model: nn.Module, module_name: str,
                      new_module: nn.Module) -> nn.Module:
    """Replace a submodule in a model with a new module."""
    parent = model.get_submodule(".".join(module_name.split(".")[:-1]))
    target_name = module_name.split(".")[-1]
    setattr(parent, target_name, new_module)
    return new_module


def parse_fine_tuned_lora_name(name: str) -> Tuple[str, bool, bool]:
    """Parse the name of lora weights.

    args:
        name: the name of the fine-tuned LoRA, e.g.
            base_model.model.dense1.weight
    return:
        Tuple(module_name, is_lora_a):
            module_name: the name of the module, e.g. model.dense1,
            is_lora_a whether the tensor is lora_a or lora_b.
            is_bias whether the tensor is lora bias.
    """
    parts = name.split(".")
    if parts[-1] == "weight" and (parts[-2] == "lora_A"
                                  or parts[-2] == "lora_B"):
        return ".".join(parts[2:-2]), parts[-2] == "lora_A", False

    if parts[-1] == "lora_embedding_A" or parts[-1] == "lora_embedding_B":
        return ".".join(parts[2:-1]), parts[-1] == "lora_embedding_A", False

    if parts[-1] == "bias":
        return ".".join(parts[2:-2]), False, True

    raise ValueError(f"{name} is unsupported LoRA weight")


def is_regex_target_modules(load_modules: Union[str, List[str]],
                            expected_lora_modules: List[str]) -> bool:
    """
    PEFT supports passing `target_modules` in the form of regular expressions, 
    such as `model.*(q_proj|k_proj|v_proj)$`. This function is mainly used to 
    determine whether the suffix in the regular expression is present in the 
    `expected_lora_modules`.
    """

    def is_valid_regex(pattern):
        try:
            re.compile(pattern)
            return True
        except re.error:
            return False

    def is_subset(sub_list, full_list):
        return set(sub_list).issubset(set(full_list))

    # Similar to PEFT's processing logic, regex-related operations are only
    #  executed when the load_modules is a `str`.
    if not isinstance(load_modules, str):
        return False

    if is_valid_regex(load_modules):
        match = re.search(r"\((.*?)\)\$?$", load_modules)
        if match:
            suffix = match.group(1).split("|")
            return is_subset(suffix, expected_lora_modules)
    return False


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
    if lora_path.startswith('~'):
        return os.path.expanduser(lora_path)

    # Check if the expanded relative path exists locally.
    if os.path.exists(lora_path):
        return os.path.abspath(lora_path)

    # If the path does not exist locally, assume it's a Hugging Face repo.
    try:
        local_snapshot_path = huggingface_hub.snapshot_download(
            repo_id=lora_path)
    except (HfHubHTTPError, RepositoryNotFoundError, EntryNotFoundError,
            HFValidationError):
        # Handle errors that may occur during the download
        # Return original path instead instead of throwing error here
        logger.exception("Error downloading the HuggingFace model")
        return lora_path

    return local_snapshot_path


def compute_meta(
    token_lora_tensor: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, bool]:
    """
    Get the information required for the sgmv kernel. With the  features:
    1. If consecutive requests in the batch use the same LoRA, this function
    will combine them into a single request, improving sgmv kernel inference
    performance.
    2. At the beginning of each prefill stage inference, recalculations are
    needed based on the input, but only once.
    """

    lora_indices_tensor, seq_length_tensor = torch.unique_consecutive(
        token_lora_tensor, return_counts=True)
    cum_result = torch.cumsum(seq_length_tensor, dim=0)
    b_seq_start_tensor = torch.zeros_like(seq_length_tensor)
    b_seq_start_tensor[1:].copy_(cum_result[:-1])
    max_length = seq_length_tensor.max().item()
    token_nums = seq_length_tensor.sum().item()
    batch_size = lora_indices_tensor.size(0)
    no_lora = False
    # -1 means no lora should be applied. Use `no_lora` to determine whether
    # the current step requires LoRA. If LoRA is not needed, the prefill stage
    # does not need to launch the triton kernel, which can improve performance
    if batch_size == 1 and lora_indices_tensor == -1:
        no_lora = True
    return (b_seq_start_tensor, seq_length_tensor, lora_indices_tensor,
            batch_size, max_length, token_nums, no_lora)


# TODO see if this can be vectorized
def convert_mapping(
    mapping: "LoRAMapping",
    lora_index_to_id: List[Optional[int]],
    max_loras: int,
    vocab_size: int,
    extra_vocab_size: int,
    device: torch.device,
    long_lora_context: Optional["LongContextLoRAContext"] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           Optional[torch.Tensor], List[int]]:
    """Converts LoRAMapping to index tensors.

    Args:
        mapping: LoRAMapping mapping rows in a batch to LoRA ids.
        lora_index_to_id: List mapping LoRA ids to LoRA indices.
        max_loras: Maximum number of LoRAs.
        vocab_size: Model vocab size.
        extra_vocab_size: Extra vocab size each LoRA can have.
        long_lora_context: Passed if there are long context lora in a batch.

    Returns:
        A tuple of tensors:
            base_indices: Tensor of shape [batch_size] mapping batch rows to
                LoRA indices.
            sampler_indices: Tensor of shape [batch_size] mapping requests to
                LoRA indices for sampler. For generation, this will be the
                same as base_indicies. For prefill, this will map requests
                to LoRA indices.
            sampler_indices_padded: Tensor of shape [batch_size] mapping
                requests to LoRA indices for sampler with padding.
                Same as sampler_indicies, but -1 is replaced with
                max_loras.
            embeddings_indices: Tensor of shape [2, batch_size] mapping
                requests to embedding indices. First row is for embeddings
                added by the LoRAs, second row is for the LoRA.lora_a
                embeddings.
            long_lora_indices: Tensor of shape [batch_size] mapping
                requests to RoPE offsets and rot dims for long LoRAs.
                None if long context lora doesn't exist.
            indices_len: List of lengths of the above tensors. It contains
                (base_indices, sampler_indices, sampler_indices_padded,
                embeddings_indices, long_lora_indices).
    """
    index_mapping_indices: List[int] = list(mapping.index_mapping).copy()
    embedding_indices = index_mapping_indices.copy()
    lora_indices = index_mapping_indices.copy()
    long_lora_offsets: Optional[torch.Tensor] = None
    if long_lora_context:
        long_lora_offsets = torch.zeros(len(index_mapping_indices),
                                        device=device,
                                        dtype=torch.long)
    prompt_mapping: List[int] = [
        lora_index_to_id.index(x) if x > 0 else -1
        for x in mapping.prompt_mapping
    ]
    lora_idx = None
    for i in range(len(index_mapping_indices)):
        # TODO index can be slow. optimize
        lora_idx = (lora_index_to_id.index(index_mapping_indices[i])
                    if index_mapping_indices[i] > 0 else -1)
        embedding_indices[i] = lora_idx if index_mapping_indices[i] > 0 else 0
        lora_indices[i] = lora_idx
        if long_lora_context:
            assert long_lora_offsets is not None
            lora_offset: int = long_lora_context.offsets_by_lora_id.get(
                index_mapping_indices[i], 0)
            long_lora_offsets[i] = lora_offset

    indices_list: List[Union[List[int], torch.Tensor]] = [
        index_mapping_indices,
        lora_indices,
        embedding_indices,
    ]
    if long_lora_context:
        assert long_lora_offsets is not None
        indices_list.append(long_lora_offsets)
    indices = torch.tensor(indices_list, dtype=torch.long, device=device)
    prompt_mapping_tensor = torch.tensor(prompt_mapping,
                                         dtype=torch.long,
                                         device=device)
    embeddings_indices = torch.stack([
        indices[2] * extra_vocab_size,
        indices[2] * (vocab_size + extra_vocab_size),
    ])
    embeddings_indices[embeddings_indices == -1] = max_loras - 1
    base_indices = indices[1]
    sampler_indices = prompt_mapping_tensor
    sampler_indices_padded = sampler_indices.clone()
    sampler_indices_padded[sampler_indices_padded == -1] = max_loras - 1
    sampler_indices_padded = torch.arange(
        0, len(sampler_indices_padded), device=device, dtype=torch.long) + (
            sampler_indices_padded * len(sampler_indices_padded))
    long_lora_indices = None
    long_lora_indices_len: Optional[int] = None
    if long_lora_context:
        long_lora_indices = indices[3]
        long_lora_indices_len = long_lora_indices.shape[-1]
    # Contain length of indices tensors. Used to index into each tensor.
    indices_len = [
        base_indices.shape[-1],
        sampler_indices.shape[-1],
        sampler_indices_padded.shape[-1],
        embeddings_indices.shape[-1],
    ]
    if long_lora_indices_len is not None:
        indices_len.append(long_lora_indices_len)
    else:
        # If long_lora doesn't exist,append None
        indices_len.append(None)

    return (
        base_indices,
        sampler_indices,
        sampler_indices_padded,
        embeddings_indices,
        long_lora_indices,
        indices_len,
    )
