# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: SIM117
import copy
import fnmatch
import glob
import itertools
import math
import os
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import HfApi
from torch import nn
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.config import LoadConfig, ModelConfig, VllmConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
# yapf: enable
from vllm.logger import init_logger
# yapf conflicts with isort for this block
# yapf: disable
from vllm.model_executor.layers.linear import (LinearBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (ParamMapping,
                                                    initialize_model,
                                                    set_default_torch_dtype)
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf, download_weights_from_hf,
    filter_duplicate_safetensors_files, filter_files_not_needed_for_inference,
    pt_weights_iterator, safetensors_weights_iterator)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

logger = init_logger(__name__)


class BitsAndBytesModelLoader(BaseModelLoader):
    """Model loader to load model weights with BitAndBytes quantization."""

    possible_config_file_names = ["adapter_config.json"]

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

        # Save the module names without sharding.
        self.unsharded_weights_modules: List[str] = []
        # Save the module names that are sharded by column.
        self.column_sharded_weights_modules: List[str] = []
        # Store all module names (from transformers) that support
        # BNB quantization.
        self.target_modules: List[str] = []
        # mapping weight names from transformers to vllm.
        self.weight_mapper: Callable = lambda name: name

    def _get_weight_files(
        self,
        model_name_or_path: str,
        allowed_patterns: List[str],
        revision: Optional[str] = None,
    ) -> Tuple[str, List[str], str]:
        """Retrieve weight files. Download the files if necessary.

        Return the weight files and the file pattern."""
        is_local = os.path.isdir(model_name_or_path)

        if is_local:
            for pattern in allowed_patterns:
                weight_files = glob.glob(
                    os.path.join(model_name_or_path, pattern))
                if weight_files:
                    return model_name_or_path, weight_files, pattern
        else:
            hf_api = HfApi()
            repo_files = hf_api.list_repo_files(repo_id=model_name_or_path)
            for pattern in allowed_patterns:
                matching_files = fnmatch.filter(repo_files, pattern)
                if matching_files:
                    hf_folder = download_weights_from_hf(
                        model_name_or_path,
                        self.load_config.download_dir,
                        [pattern],
                        revision,
                        ignore_patterns=self.load_config.ignore_patterns,
                    )
                    return hf_folder, glob.glob(
                        os.path.join(hf_folder, pattern)), pattern

        raise RuntimeError(
            f"No model weights found in: `{model_name_or_path}`")

    def _prepare_weights(self, model_name_or_path: str,
                         revision: Optional[str]) -> Tuple[List[str], bool]:
        """Prepare weight files for the model."""

        allowed_patterns = ["*.safetensors", "*.bin", "*.pt"]

        hf_folder, hf_weights_files, matched_pattern = self._get_weight_files(
            model_name_or_path, allowed_patterns, revision)

        use_safetensors = matched_pattern == "*.safetensors"
        is_local = os.path.isdir(model_name_or_path)
        index_file = SAFE_WEIGHTS_INDEX_NAME
        if use_safetensors:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.
            if not is_local:
                download_safetensors_index_file_from_hf(
                    model_name_or_path,
                    index_file,
                    self.load_config.download_dir,
                    revision,
                )
            hf_weights_files = filter_duplicate_safetensors_files(
                hf_weights_files, hf_folder, index_file)
        else:
            hf_weights_files = filter_files_not_needed_for_inference(
                hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`")

        return hf_weights_files, use_safetensors

    def _hf_weight_iter(self, hf_weights_files, use_safetensors: bool):
        if use_safetensors:
            iterator = safetensors_weights_iterator(
                hf_weights_files,
                self.load_config.use_tqdm_on_load,
            )
        else:
            iterator = pt_weights_iterator(
                hf_weights_files,
                self.load_config.use_tqdm_on_load,
                self.load_config.pt_load_map_location,
            )
        for org_name, param in iterator:
            # mapping weight names from transformers to vllm while preserving
            # original names.
            mapped_name = self.weight_mapper(org_name)
            yield org_name, mapped_name, param

    def _get_quantized_weights_iterator(
        self,
        model_name_or_path: str,
        revision: Optional[str],
        pre_quant: bool,
        load_8bit: bool,
    ) -> Tuple[Generator[Tuple[str, torch.Tensor], None, None], Dict[str,
                                                                     Any]]:
        """Get an iterator to the model weights with bitsandbytes quantization,
        as well as the quantization state dictionary."""

        # only load the bitsandbytes module when needed
        try:
            import bitsandbytes

            if bitsandbytes.__version__ < "0.45.3":
                raise ImportError("bitsandbytes version is wrong. Please "
                                  "install bitsandbytes>=0.45.3.")
        except ImportError as err:
            raise ImportError("Please install bitsandbytes>=0.45.3 via "
                              "`pip install bitsandbytes>=0.45.3` to use "
                              "bitsandbytes quantizer.") from err

        hf_weights_files, use_safetensors = self._prepare_weights(
            model_name_or_path, revision)

        quant_state_dict: Dict[str, Any] = {}

        if pre_quant:
            if load_8bit:
                return self._quantized_8bit_generator(
                    hf_weights_files, use_safetensors,
                    quant_state_dict), quant_state_dict
            else:
                return self._quantized_4bit_generator(
                    hf_weights_files, use_safetensors,
                    quant_state_dict), quant_state_dict

        return self._unquantized_generator(hf_weights_files, use_safetensors,
                                           quant_state_dict), quant_state_dict

    def _is_8bit_weight_name(self, weight_name: str):
        quantized_suffix = {".scb", ".weight_format"}
        return any(weight_name.lower().endswith(suffix)
                   for suffix in quantized_suffix)

    def _is_4bit_weight_name(self, weight_name: str):
        quantized_suffix = {
            "absmax",
            "quant_map",
            "nested_absmax",
            "nested_quant_map",
            "bitsandbytes",
        }
        suffix = weight_name.split(".")[-1]
        return any(q_suffix in suffix for q_suffix in quantized_suffix)

    def _quantized_8bit_generator(self, hf_weights_files, use_safetensors,
                                  quant_state_dict) -> Generator:
        for (
                org_weight_name,
                mapped_weight_name,
                weight_tensor,
        ) in self._hf_weight_iter(hf_weights_files, use_safetensors):
            if not mapped_weight_name.lower().endswith(".scb"):
                continue

            weight_key = mapped_weight_name.lower().replace(".scb", ".weight")
            quant_state_dict[weight_key] = weight_tensor

        for (
                org_weight_name,
                mapped_weight_name,
                weight_tensor,
        ) in self._hf_weight_iter(hf_weights_files, use_safetensors):
            if self._is_8bit_weight_name(mapped_weight_name):
                continue

            if mapped_weight_name in quant_state_dict:
                set_weight_attrs(weight_tensor, {"load_in_8bit": True})
                yield org_weight_name, weight_tensor
            else:
                yield org_weight_name, weight_tensor

    def _quantized_4bit_generator(self, hf_weights_files, use_safetensors,
                                  quant_state_dict) -> Generator:
        from bitsandbytes.functional import QuantState

        # First iterate over all quant state weights
        weight_iterator = self._hf_weight_iter(hf_weights_files,
                                               use_safetensors)
        temp_state_dict = {}
        for (
                org_weight_name,
                mapped_weight_name,
                weight_tensor,
        ) in weight_iterator:
            if not self._is_4bit_weight_name(mapped_weight_name):
                continue
            # bitsandbytes library requires
            # weight.quant_state.bitsandbytes__* in CPU
            if "quant_state.bitsandbytes" in mapped_weight_name:
                temp_state_dict[mapped_weight_name] = weight_tensor.cpu().data
            else:
                temp_state_dict[mapped_weight_name] = weight_tensor

        # Closure to parse quant_state for each prequant weight
        def _parse_quant_state(param_name: str,
                               temp_state_dict: Dict) -> QuantState:
            quant_state = {}
            for k in temp_state_dict:
                if param_name + "." in k:
                    quant_state[k] = temp_state_dict[k]

            return QuantState.from_dict(quant_state,
                                        device=current_platform.device_type)

        # Second iterate over all prequant and normal weights
        # pre quantized weights would have a quant_state
        for (
                org_weight_name,
                mapped_weight_name,
                weight_tensor,
        ) in self._hf_weight_iter(hf_weights_files, use_safetensors):
            if self._is_4bit_weight_name(mapped_weight_name):
                continue

            if (f"{mapped_weight_name}.quant_state.bitsandbytes__nf4"
                    in temp_state_dict) or (
                        f"{mapped_weight_name}.quant_state.bitsandbytes__fp4"
                        in temp_state_dict):
                quant_state = _parse_quant_state(mapped_weight_name,
                                                 temp_state_dict)
                quant_state_dict[mapped_weight_name] = quant_state
                yield org_weight_name, weight_tensor
            else:
                yield org_weight_name, weight_tensor

    def _unquantized_generator(self, hf_weights_files, use_safetensors,
                               quant_state_dict) -> Generator:
        from bitsandbytes.functional import quantize_4bit

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        for (
                org_weight_name,
                mapped_weight_name,
                weight_tensor,
        ) in self._hf_weight_iter(hf_weights_files, use_safetensors):
            if any(target_module in mapped_weight_name
                   for target_module in self.target_modules
                   ) and mapped_weight_name.endswith(".weight"):
                # Without sharding
                if any(
                        mapped_weight_name.startswith(module)
                        for module in self.unsharded_weights_modules):
                    weight_sub_tensor = weight_tensor
                # Shard by column
                elif any(
                        mapped_weight_name.startswith(module)
                        for module in self.column_sharded_weights_modules):
                    total_size = weight_tensor.size(-1)
                    start_index = total_size // tp_size * tp_rank
                    end_index = total_size // tp_size * (tp_rank + 1)
                    weight_sub_tensor = weight_tensor[...,
                                                      start_index:end_index]
                # Weights have fused on disk. In this case, we assume that the
                # weight and module use same name.
                elif any(
                        mapped_weight_name.startswith(module)
                        for module in self.maybe_fused_weights_modules):
                    # special case for fused weights
                    # get the size of each shard weight tensor
                    total_shard_sizes = next(
                        (sizes for module, sizes in
                         self.maybe_fused_weights_modules.items()
                         if mapped_weight_name.startswith(module)))
                    total_size = weight_tensor.size(0)
                    assert total_size == sum(total_shard_sizes)
                    # get the start/end index of each shard weight tensor
                    total_start_index = list(
                        itertools.accumulate([0] + total_shard_sizes))[:-1]
                    shard_weights_index = [(
                        idx + size // tp_size * tp_rank,
                        idx + size // tp_size * (tp_rank + 1),
                    ) for idx, size in zip(total_start_index,
                                           total_shard_sizes)]
                    # slice and reorder the weight tensor
                    weight_tensor = [
                        weight_tensor[start_index:end_index, ...]
                        for start_index, end_index in shard_weights_index
                    ]
                    weight_sub_tensor = torch.cat(weight_tensor, dim=0)
                # Shard by row
                else:
                    total_size = weight_tensor.size(0)
                    start_index = total_size // tp_size * tp_rank
                    end_index = total_size // tp_size * (tp_rank + 1)
                    weight_sub_tensor = weight_tensor[start_index:end_index,
                                                      ...]

                # bitsandbytes requires data in GPU
                if weight_sub_tensor.is_cuda:
                    loaded_weight = weight_sub_tensor
                else:
                    loaded_weight = weight_sub_tensor.cuda()

                # remove the following after the issue is fixed:
                # https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1342
                if loaded_weight.is_contiguous() is False:
                    loaded_weight = loaded_weight.contiguous()

                with set_default_torch_dtype(torch.float32):
                    processed_weight, quant_state = quantize_4bit(
                        loaded_weight,
                        compress_statistics=True,
                        quant_type="nf4",
                    )

                quant_state_dict[mapped_weight_name] = quant_state
            else:
                processed_weight = weight_tensor
            yield org_weight_name, processed_weight

    def _get_bnb_target_modules(self, model: nn.Module) -> None:

        for name, module in model.named_modules():
            if isinstance(module, (LinearBase, )):
                if modules_info := self.modules_mapping.get_sub_modules(name):
                    # Map vllm's names to transformers's names.
                    rep_name, sub_modules = modules_info
                    for sub_name in sub_modules:
                        self.target_modules.append(
                            name.replace(rep_name, sub_name))
                # Add original module name even if the module has stacked map,
                # in case model has a mixture of disk-merged and disk-splitted
                # weights with same last name.
                self.target_modules.append(name)

        assert (self.target_modules
                ), "vllm currently does not support BNB quantization for"
        f" {type(model).__name__}"

    def _load_weights(self, model_config: ModelConfig,
                      model: nn.Module) -> None:
        if not hasattr(model, "load_weights"):
            raise AttributeError(
                "The required method 'load_weights' is not defined in class"
                f" {type(model).__name__}.")

        if not hasattr(model, "packed_modules_mapping"):
            raise AttributeError(
                f"Model {type(model).__name__} does not support BitsAndBytes "
                "quantization yet. No 'packed_modules_mapping' found.")

        self.modules_mapping = ParamMapping(
            copy.deepcopy(model.packed_modules_mapping))

        # For some models like Molmo, we need to use hf_to_vllm_mapper
        # to ensure correct loading of weights.
        if hf_to_vllm_mapper := getattr(model, "hf_to_vllm_mapper", None):
            self.weight_mapper = lambda name: hf_to_vllm_mapper._map_name(name)

        # Modules whose weights might have fused on disk
        # we need their output_sizes to make shard in flight correctly with TP
        self.maybe_fused_weights_modules: Dict[str, List[int]] = {}
        self._get_bnb_target_modules(model)
        for name, module in model.named_modules():
            # Some modules like `ReplicatedLinear` should not have their weights
            # sharded. The reason for implementing it this way is to avoid new
            # static variable in the model implementation.
            if isinstance(module, (ReplicatedLinear, )):
                self.unsharded_weights_modules.append(name)
            # `QKVParallelLinear` and `MergedColumnParallelLinear` might have
            # fused weights on disk. We need to use the output sizes of these
            # modules to shard the weights correctly.
            elif isinstance(module,
                            (QKVParallelLinear, MergedColumnParallelLinear)):
                self.maybe_fused_weights_modules[name] = module.output_sizes
            # In TP, these weights are partitioned along the column
            # dimension (dim=-1)
            elif isinstance(module, (RowParallelLinear, )):
                self.column_sharded_weights_modules.append(name)

        self.model_type = type(model).__name__

        logger.info("Loading weights with BitsAndBytes quantization. "
                    "May take a while ...")

        quant_config = getattr(model_config.hf_config, "quantization_config",
                               None)

        pre_quant = False
        if quant_config is not None:
            quant_method = quant_config.get("quant_method")
            if quant_method == "bitsandbytes":
                pre_quant = True
            else:
                raise ValueError(
                    f"BitsAndBytes loader does not support {quant_method} "
                    "quantization")

        # The quant_states in pre_quantized models cannot work with a split
        # weight tensor. So TP does not work with pre_quantized bnb models.
        if pre_quant and get_tensor_model_parallel_world_size() > 1:
            raise ValueError(
                "Prequant BitsAndBytes models with tensor parallelism is not "
                "supported. Please try with pipeline parallelism.")

        load_8bit = False
        if pre_quant:
            load_8bit = quant_config.get("load_in_8bit", False)

        qweight_iterator, quant_state_dict = (
            self._get_quantized_weights_iterator(model_config.model,
                                                 model_config.revision,
                                                 pre_quant, load_8bit))

        weights_to_load = {name for name, _ in model.named_parameters()}
        loaded_weights = model.load_weights(qweight_iterator)
        # Some models may have weights loading tracker unimplemented.
        if loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                raise ValueError("Following weights were not initialized from "
                                 f"checkpoint: {weights_not_loaded}")

        torch.cuda.empty_cache()

        param_dict = dict(model.named_parameters())
        stacked_quant_state_dict: Dict[str, Dict[int, Any]] = {}
        # TODO: Change this lazy import to normal import
        # after the checks are updated to run on a new version
        from vllm.model_executor.models.utils import is_pp_missing_parameter

        for quant_param_name in quant_state_dict:
            if is_pp_missing_parameter(quant_param_name, model):
                continue

            non_stacked_param_name = quant_param_name

            shard_index = 0
            for shard_name, (
                    weight_name,
                    index,
            ) in self.modules_mapping.inverse_packed_mapping.items():
                # Some models, such as MiniCPM V2.5/2.6, contain both
                # module names 'kv_proj' and 'qkv_proj'. To prevent 'kv_proj'
                # from being incorrectly identified as being present in
                # 'vpm.encoder.layers.0.self_attn.qkv_proj.weight
                shard_pos = quant_param_name.find(shard_name)
                can_correct_rename = (shard_pos
                                      > 0) and (quant_param_name[shard_pos - 1]
                                                == ".")
                # If the quant_param_name is packed, it won't occur in the
                # param_dict before renaming.
                new_quant_param_name = quant_param_name.replace(
                    shard_name, weight_name)
                need_rename = (quant_param_name not in param_dict) \
                              and (new_quant_param_name in param_dict)
                if can_correct_rename and need_rename:
                    shard_index = index
                    quant_param_name = new_quant_param_name
                    break

            # Models like Clip/Siglip may skip some layers in initialization,
            # causing unused quant_param_name in state_dict.
            if quant_param_name not in param_dict:
                continue

            if quant_param_name not in stacked_quant_state_dict:
                stacked_quant_state_dict[quant_param_name] = {}

            stacked_quant_state_dict[quant_param_name][shard_index] = (
                quant_state_dict[non_stacked_param_name])

        # save quant_states and offsets as the attributes of the parameters
        for param_name, param in param_dict.items():
            if param_name in stacked_quant_state_dict:
                quant_states = stacked_quant_state_dict[param_name]
                set_weight_attrs(param, {"bnb_quant_state": quant_states})

                pack_ratio = getattr(param, "pack_factor", -1)
                if pack_ratio == -1:
                    raise ValueError(
                        f"pack_factor not set for parameter {param_name}.")

                num_elements = [0] * len(quant_states)
                for seq, quant_state in quant_states.items():
                    num_elements[seq] = (math.prod(quant_state.shape) //
                                         pack_ratio)

                offsets = np.concatenate(([0], np.cumsum(num_elements)))
                # Make torch infer_schema happy
                offsets = torch.tensor(offsets).cpu()
                set_weight_attrs(param, {"bnb_shard_offsets": offsets})

                if load_8bit:
                    set_weight_attrs(
                        param, {"matmul_state": [None] * len(quant_states)})

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config.model, model_config.revision)

    def load_model(self, vllm_config: VllmConfig) -> nn.Module:
        device_config = vllm_config.device_config
        model_config = vllm_config.model_config

        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):

                model = initialize_model(vllm_config=vllm_config)

                self._load_weights(model_config, model)

        return model.eval()
