# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Generator

import gguf
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model, process_weights_after_loading, set_default_torch_dtype)
from vllm.model_executor.model_loader.weight_utils import (
    get_gguf_extra_tensor_names, get_gguf_weight_type_map,
    gguf_quant_weights_iterator)


class GGUFModelLoader(BaseModelLoader):
    """
    Model loader that can load GGUF files. This is useful for loading models
    that are quantized with GGUF and saved in the GGUF format. This loader
    supports loading both full models and sharded models.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(f"Model loader extra config is not supported for "
                             f"load format {load_config.load_format}")

    def _prepare_weights(self, model_name_or_path: str):
        if os.path.isfile(model_name_or_path):
            return model_name_or_path
        # for raw HTTPS link
        if model_name_or_path.startswith(
            ("http://", "https://")) and model_name_or_path.endswith(".gguf"):
            return hf_hub_download(url=model_name_or_path)
        # repo id/filename.gguf
        if "/" in model_name_or_path and model_name_or_path.endswith(".gguf"):
            repo_id, filename = model_name_or_path.rsplit("/", 1)
            return hf_hub_download(repo_id=repo_id, filename=filename)
        else:
            raise ValueError(
                f"Unrecognised GGUF reference: {model_name_or_path} "
                "(expected local file, raw URL, or <repo_id>/<filename>.gguf)")

    def _get_gguf_weights_map(self, model_config: ModelConfig):
        """
        GGUF uses this naming convention for their tensors from HF checkpoint:
        `blk.N.BB.weight` and `blk.N.BB.bias`
        where N signifies the block number of a layer, and BB signifies the
        attention/mlp layer components.
        See "Standardized tensor names" in
        https://github.com/ggerganov/ggml/blob/master/docs/gguf.md for details.
        """
        config = model_config.hf_config
        model_type = config.model_type
        gguf_to_hf_name_map = {}
        # hack: ggufs have a different name than transformers
        if model_type == "cohere":
            model_type = "command-r"
        if model_type in ("deepseek_v3", "deepseek_v2"):
            model_type = "deepseek2"
            # GGUF layer map assumes that we will have a merged expert weights
            # so we need to map them manually
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.exp_probs_b.bias"] = \
                        f"model.layers.{idx}.mlp.gate.e_score_correction_bias"
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = \
                        f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = \
                        f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = \
                        f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
        if model_type in ("qwen2_moe", "qwen3_moe"):
            model_type = model_type.replace("_", "")
            # GGUF layer map assumes that we will have a merged expert weights
            # so we need to map them manually
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = \
                        f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = \
                        f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = \
                        f"model.layers.{idx}.mlp.experts.0.up_proj.weight"

        arch = None
        for key, value in gguf.MODEL_ARCH_NAMES.items():
            if value == model_type:
                arch = key
                break
        if arch is None:
            raise RuntimeError(f"Unknown gguf model_type: {model_type}")
        num_layers = config.num_hidden_layers
        name_map = gguf.get_tensor_name_map(arch, num_layers)
        with torch.device("meta"):
            dummy_model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=model_config.trust_remote_code)
        state_dict = dummy_model.state_dict()

        for hf_name in state_dict:
            name, suffix = hf_name.rsplit(".", 1)
            gguf_name = name_map.get_name(name)
            gguf_to_hf_name_map[f"{gguf_name}.{suffix}"] = hf_name
        return gguf_to_hf_name_map

    def _get_weights_iterator(
        self, model_name_or_path: str, gguf_to_hf_name_map: dict[str, str]
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        return gguf_quant_weights_iterator(model_name_or_path,
                                           gguf_to_hf_name_map)

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config.model)

    def load_weights(self, model: nn.Module,
                     model_config: ModelConfig) -> None:
        local_model_path = self._prepare_weights(model_config.model)
        gguf_weights_map = self._get_gguf_weights_map(model_config)
        model.load_weights(
            self._get_weights_iterator(local_model_path, gguf_weights_map))

    def load_model(self, vllm_config: VllmConfig,
                   model_config: ModelConfig) -> nn.Module:
        device_config = vllm_config.device_config
        local_model_path = self._prepare_weights(model_config.model)
        gguf_weights_map = self._get_gguf_weights_map(model_config)
        # we can only know if tie word embeddings after mapping weights
        if "lm_head.weight" in get_gguf_extra_tensor_names(
                local_model_path, gguf_weights_map):
            model_config.hf_config.update({"tie_word_embeddings": True})

        weight_type_map = get_gguf_weight_type_map(model_config.model,
                                                   gguf_weights_map)

        # filter out unquantized modules to skip
        unquant_names = [
            name.removesuffix(".weight")
            for name, weight_type in weight_type_map.items()
            if weight_type == "F32" and name.endswith(".weight")
        ]
        vllm_config.quant_config.unquantized_modules.extend(unquant_names)

        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config)
            self.load_weights(model, model_config)

            process_weights_after_loading(model, model_config, target_device)
        return model
