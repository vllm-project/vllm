# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import huggingface_hub
import regex as re
import torch
from huggingface_hub.utils import (
    EntryNotFoundError,
    HfHubHTTPError,
    HFValidationError,
    RepositoryNotFoundError,
)
from safetensors.torch import save_file
from torch import nn
from transformers import PretrainedConfig

from vllm.config.lora import LoRAConfig
from vllm.logger import init_logger

# being imported for _all_lora_classes below
from vllm.lora.layers import (
    BaseLayerWithLoRA,
    ColumnParallelLinearWithLoRA,
    ColumnParallelLinearWithShardedLoRA,
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
from vllm.lora.lora_weights import LoRALayerWeights, PackedLoRALayerWeights
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.utils import get_moe_expert_mapping, get_packed_modules_mapping

if TYPE_CHECKING:
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

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


def is_regex_target_modules(
    load_modules: str | list[str], expected_lora_modules: list[str]
) -> bool:
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

            packed_modules_mapping["experts"] = [
                weight_name.rstrip(".") for _, weight_name, _, _ in moe_packed_mapping
            ]

            return packed_modules_mapping
        else:
            raise AttributeError(
                "To support LoRA for MoE model, "
                "'get_expert_mapping' must be implemented"
            )
    else:
        return get_packed_modules_mapping(model)


# Below utils are used in the tests.


class DummyLoRAManager:
    def __init__(self, device: torch.device = "cuda:0"):
        super().__init__()
        self._loras: dict[str, LoRALayerWeights] = {}
        self._device = device

    def set_module_lora(self, module_name: str, lora: LoRALayerWeights):
        self._loras[module_name] = lora

    def get_module_lora(self, module_name: str) -> LoRALayerWeights:
        return self._loras[module_name]

    def init_random_lora(
        self,
        module_name: str,
        weight: torch.Tensor,
        rank: int = 8,
        generate_embeddings_tensor: int = 0,
    ):
        lora = LoRALayerWeights(
            module_name,
            rank=rank,
            lora_alpha=1,
            lora_a=torch.rand(
                [rank, weight.shape[1]], dtype=weight.dtype, device=self._device
            ),
            lora_b=torch.rand(
                [weight.shape[0], rank], dtype=weight.dtype, device=self._device
            ),
        )
        if generate_embeddings_tensor:
            lora.embeddings_tensor = torch.rand(
                5,
                generate_embeddings_tensor,
                dtype=weight.dtype,
                device=self._device,
            )
        self.set_module_lora(module_name, lora)

        return lora

    def init_lora(
        self,
        module_name: str,
        input_dim: int,
        output_dim: int,
        rank=8,
        noop=False,
        embeddings_tensor=None,
    ):
        lora = LoRALayerWeights(
            module_name,
            rank=rank,
            lora_alpha=1,
            lora_a=torch.rand([rank, input_dim], device="cuda"),
            lora_b=torch.rand([output_dim, input_dim], device="cuda"),
            embeddings_tensor=embeddings_tensor,
        )
        self.set_module_lora(module_name, lora)
        return lora

    def reset_lora(self):
        self._loras = {}

    def init_packed_lora(
        self,
        module_name: str,
        input_dim: int,
        output_dims: list[int],
        noop_lora_index: list[int] | None = None,
        rank: int = 8,
    ):
        base_loras: list[LoRALayerWeights] = []
        noop_lora_index_set = set(noop_lora_index or [])

        for i, out_dim in enumerate(output_dims):
            base_lora = self.init_lora(
                module_name + "_000_" + str(i),
                input_dim,
                out_dim,
                rank=rank,
                noop=i in noop_lora_index_set,
            )
            base_loras.append(base_lora)
        packed_lora = PackedLoRALayerWeights.pack(base_loras)
        self.set_module_lora(module_name, packed_lora)
        return packed_lora


def assert_close(a, b):
    rtol, atol = {
        torch.float16: (6e-2, 6e-2),
        torch.bfloat16: (6e-2, 6e-2),
        torch.float32: (1e-2, 1e-2),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


@dataclass
class PunicaTensors:
    inputs_tensor: torch.Tensor
    lora_weights: torch.Tensor | list[torch.Tensor]
    our_out_tensor: torch.Tensor
    ref_out_tensor: torch.Tensor
    b_seq_start_loc: torch.Tensor
    prompt_lora_mapping: torch.Tensor
    seq_len_tensor: torch.Tensor
    token_lora_mapping: torch.Tensor

    def meta(self) -> tuple[int, int]:
        """
        Infer max_seq_length and token_nums from the tensors
        and return them.
        """
        max_seq_length = self.seq_len_tensor.max()
        token_nums = self.seq_len_tensor.sum().item()
        if isinstance(max_seq_length, tuple):
            max_seq_length = max_seq_length[0].item()
        else:
            max_seq_length = max_seq_length.item()
        return max_seq_length, token_nums


def generate_data(
    batches,
    hidden_size,
    lora_nums,
    max_rank,
    seq_length,
    dtype,
    op_type,
    device,
) -> PunicaTensors:
    seq_len_tensor = torch.randint(seq_length, seq_length + 1, (batches,)).to(device)
    b_seq_start_loc = torch.cumsum(
        torch.tensor([0] + seq_len_tensor[:-1].tolist(), dtype=torch.long),
        dim=0,
    ).to(device)
    total_tokens = seq_len_tensor.sum()
    if op_type == "shrink":
        inputs_tensor = torch.rand((total_tokens, hidden_size), dtype=dtype).to(device)
        lora_weights = torch.rand(
            (lora_nums, max_rank, hidden_size),  # col-major
            dtype=dtype,
        ).to(device)
        # shrink op need atomic_add, so output is initinized by 0
        ref_out_tensor = torch.zeros(
            (total_tokens, max_rank), dtype=dtype, device=inputs_tensor.device
        )
        # NOTE  shrink kernel using torch.float32 as output type
        our_out_tensor = torch.zeros((total_tokens, max_rank), dtype=torch.float32).to(
            device
        )
    else:
        inputs_tensor = torch.rand(
            (total_tokens, max_rank),
            dtype=dtype,
        ).to(device)
        lora_weights = torch.rand(
            (lora_nums, hidden_size, max_rank),  # col-major
            dtype=dtype,
        ).to(device)
        # expand op needs to complete y+=a@lora_b, so output is
        # initinized randomly
        ref_out_tensor = torch.rand(
            (total_tokens, hidden_size),
            dtype=dtype,
        ).to(device)
        # Ensure the same input.
        our_out_tensor = ref_out_tensor.clone()
    lora_indices_tensor = torch.randint(
        0, lora_nums - 1 if lora_nums > 1 else 1, (batches,)
    ).to(device)
    indices = torch.zeros((total_tokens), dtype=torch.long).to(device)
    current_offset = 0
    for b_id in range(batches):
        lora_index = lora_indices_tensor[b_id]
        indices[current_offset : current_offset + seq_len_tensor[b_id]].copy_(
            lora_index
        )
        current_offset += seq_len_tensor[b_id].item()

    return PunicaTensors(
        inputs_tensor,
        lora_weights,
        our_out_tensor,
        ref_out_tensor,
        b_seq_start_loc,
        lora_indices_tensor,
        seq_len_tensor,
        indices,
    )


def generate_data_for_expand_nslices(
    batches,
    hidden_size,
    lora_nums,
    max_rank,
    seq_length,
    dtype,
    nslices,
    device,
) -> PunicaTensors:
    seq_len_tensor = torch.randint(seq_length, seq_length + 1, (batches,)).to(device)
    b_seq_start_loc = torch.cumsum(
        torch.tensor([0] + seq_len_tensor[:-1].tolist(), dtype=torch.long),
        dim=0,
    ).to(device)
    total_tokens = seq_len_tensor.sum()
    inputs_tensor = torch.rand(
        (total_tokens, max_rank),
        dtype=dtype,
    ).to(device)
    lora_weights_lst = []
    for _ in range(nslices):
        lora_weights_lst.append(
            torch.rand(
                (lora_nums, hidden_size, max_rank),  # col-major
                dtype=dtype,
            ).to(device)
        )
    # expand op needs to complete y+=a@lora_b, so output is
    # initinized randomly
    ref_out_tensor = torch.rand((total_tokens, hidden_size * nslices), dtype=dtype).to(
        device
    )
    # Ensure the same input.
    our_out_tensor = ref_out_tensor.clone()
    lora_indices_tensor = torch.randint(
        0, lora_nums - 1 if lora_nums > 1 else 1, (batches,)
    )
    indices = torch.zeros((total_tokens), dtype=torch.long).to(device)
    current_offset = 0
    for b_id in range(batches):
        lora_index = lora_indices_tensor[b_id]
        indices[current_offset : current_offset + seq_len_tensor[b_id]] = (
            lora_index.item()
        )
        current_offset += seq_len_tensor[b_id].item()

    lora_indices_tensor = lora_indices_tensor.to(device)
    return PunicaTensors(
        inputs_tensor,
        lora_weights_lst,
        our_out_tensor,
        ref_out_tensor,
        b_seq_start_loc,
        lora_indices_tensor,
        seq_len_tensor,
        indices,
    )


def generate_data_for_nslices(
    batches,
    hidden_size,
    lora_nums,
    max_rank,
    seq_length,
    nslices,
    dtype,
    op_type,
    device,
) -> PunicaTensors:
    seq_len_tensor = torch.randint(seq_length, seq_length + 1, (batches,)).to(device)
    b_seq_start_loc = torch.cumsum(
        torch.tensor([0] + seq_len_tensor[:-1].tolist(), dtype=torch.long),
        dim=0,
    ).to(device)
    total_tokens = seq_len_tensor.sum()

    lora_weights_lst = []
    if op_type == "shrink":
        inputs_tensor = torch.rand((total_tokens, hidden_size), dtype=dtype).to(device)

        for _ in range(nslices):
            if op_type == "shrink":
                lora_weights_lst.append(
                    torch.rand(
                        (lora_nums, max_rank, hidden_size),  # col-major
                        dtype=dtype,
                    ).to(device)
                )
        # NOTE  shrink kernel using torch.float32 as output type
        # shrink op need atomic_add, so output is initinized by 0
        our_out_tensor = torch.zeros(
            (nslices, total_tokens, max_rank),
            dtype=torch.float32,
        ).to(device)
    else:
        inputs_tensor = torch.rand(
            (nslices, total_tokens, max_rank),
            dtype=dtype,
        ).to(device)
        for _ in range(nslices):
            lora_weights_lst.append(
                torch.rand(
                    (lora_nums, hidden_size, max_rank),  # col-major
                    dtype=dtype,
                ).to(device)
            )
        # expand op needs to complete y+=a@lora_b, so output is
        # initinized randomly
        our_out_tensor = torch.rand(
            (total_tokens, hidden_size * nslices), dtype=dtype
        ).to(device)

    # Ensure the same input.
    ref_out_tensor = our_out_tensor.clone()
    lora_indices_tensor = torch.randint(
        0, lora_nums - 1 if lora_nums > 1 else 1, (batches,)
    )
    indices = torch.zeros((total_tokens), dtype=torch.long).to(device)
    current_offset = 0
    for b_id in range(batches):
        lora_index = lora_indices_tensor[b_id]
        indices[current_offset : current_offset + seq_len_tensor[b_id]] = (
            lora_index.item()
        )
        current_offset += seq_len_tensor[b_id].item()

    lora_indices_tensor = lora_indices_tensor.to(device)
    return PunicaTensors(
        inputs_tensor,
        lora_weights_lst,
        our_out_tensor,
        ref_out_tensor,
        b_seq_start_loc,
        lora_indices_tensor,
        seq_len_tensor,
        indices,
    )


def create_peft_lora(
    model: torch.nn.Module,
    save_dir: str,
    target_modules: list[str],
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    lora_dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    lora_weights = {}
    adapter_config = {
        "peft_type": "LORA",
        "auto_mapping": None,
        "base_model_name_or_path": "dummy_model",
        "revision": None,
        "task_type": "CAUSAL_LM",
        "inference_mode": False,
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "fan_in_fan_out": False,
        "bias": "none",
        "modules_to_save": None,
        "init_lora_weights": True,
        "layers_to_transform": None,
        "layers_pattern": None,
        "target_modules": target_modules,
        "exclude_modules": None,
        "use_rslora": False,
        "use_dora": False,
        "loftq_config": None,
    }

    for module_name in target_modules:
        module = model
        for attr in module_name.split("."):
            module = getattr(module, attr)

        if hasattr(module, "input_size") and hasattr(module, "output_size"):
            in_features = module.input_size
            out_features = module.output_size

        elif hasattr(module, "embedding_dim") and hasattr(module, "num_embeddings"):
            # ParallelLMHead
            in_features = module.embedding_dim
            out_features = module.num_embeddings
        else:
            raise ValueError(f"Unable to determine dimensions for module {module_name}")

        lora_A = torch.randn(rank, in_features, dtype=lora_dtype)

        torch.nn.init.kaiming_uniform_(lora_A, a=5**0.5)

        lora_B = torch.zeros(out_features, rank, dtype=lora_dtype)

        # PEFT style
        lora_weights[f"base_model.model.{module_name}.lora_A.weight"] = lora_A
        lora_weights[f"base_model.model.{module_name}.lora_B.weight"] = lora_B

    config_path = os.path.join(save_dir, "adapter_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=2, ensure_ascii=False)

    weights_path = os.path.join(save_dir, "adapter_model.safetensors")
    save_file(lora_weights, weights_path)

    return lora_weights
