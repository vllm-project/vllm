# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import os
from collections.abc import Generator

import gguf
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.model_executor.model_loader.weight_utils import (
    get_gguf_extra_tensor_names,
    get_gguf_weight_type_map,
    gguf_quant_weights_iterator,
)
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)


class GGUFModelLoader(BaseModelLoader):
    """
    Model loader that can load GGUF files. This is useful for loading models
    that are quantized with GGUF and saved in the GGUF format. This loader
    supports loading both full models and sharded models.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )

    def _prepare_weights(self, model_name_or_path: str):
        if os.path.isfile(model_name_or_path):
            return model_name_or_path
        # for raw HTTPS link
        if model_name_or_path.startswith(
            ("http://", "https://")
        ) and model_name_or_path.endswith(".gguf"):
            return hf_hub_download(url=model_name_or_path)
        # repo id/filename.gguf
        if "/" in model_name_or_path and model_name_or_path.endswith(".gguf"):
            repo_id, filename = model_name_or_path.rsplit("/", 1)
            return hf_hub_download(repo_id=repo_id, filename=filename)
        else:
            raise ValueError(
                f"Unrecognised GGUF reference: {model_name_or_path} "
                "(expected local file, raw URL, or <repo_id>/<filename>.gguf)"
            )

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
        # Get text config to handle both nested (multimodal) and flat
        # (text-only) config structures. For multimodal models like
        # Gemma3Config, this returns config.text_config. For text-only
        # models, this returns config itself.
        text_config = config.get_text_config()
        model_type = config.model_type
        is_multimodal = (
            hasattr(config, "vision_config") and config.vision_config is not None
        )
        gguf_to_hf_name_map = {}
        # hack: ggufs have a different name than transformers
        if model_type == "cohere":
            model_type = "command-r"
        if model_type == "gemma3_text":
            # Gemma3 models use "gemma3_text" in HuggingFace but
            # "gemma3" in GGUF architecture naming
            model_type = "gemma3"
        if model_type in ("deepseek_v3", "deepseek_v2"):
            model_type = "deepseek2"
            # GGUF layer map assumes that we will have a merged expert weights
            # so we need to map them manually
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.exp_probs_b.bias"] = (
                    f"model.layers.{idx}.mlp.gate.e_score_correction_bias"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                )
        if model_type in ("qwen2_moe", "qwen3_moe"):
            model_type = model_type.replace("_", "")
            # GGUF layer map assumes that we will have a merged expert weights
            # so we need to map them manually
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                )

        arch = None
        for key, value in gguf.MODEL_ARCH_NAMES.items():
            if value == model_type:
                arch = key
                break
        if arch is None:
            raise RuntimeError(f"Unknown gguf model_type: {model_type}")
        text_num_layers = text_config.num_hidden_layers
        text_name_map = gguf.get_tensor_name_map(arch, text_num_layers)

        if is_multimodal:
            mm_proj_arch = gguf.MODEL_ARCH.CLIP_VISION
            vision_num_layers = config.vision_config.num_hidden_layers
            vision_name_map = gguf.get_tensor_name_map(mm_proj_arch, vision_num_layers)
        else:
            vision_name_map = None

        with torch.device("meta"):
            dummy_model = AutoModelForCausalLM.from_config(
                text_config, trust_remote_code=model_config.trust_remote_code
            )

        state_dict = dummy_model.state_dict()
        if hf_checkpoint_map := getattr(
            dummy_model, "_checkpoint_conversion_mapping", None
        ):

            def revert_hf_rename(name: str) -> str:
                for original_name, hf_name in hf_checkpoint_map.items():
                    if hf_name in name:
                        name = name.replace(hf_name, original_name).lstrip("^")
                return name

            state_dict = {
                revert_hf_rename(name): tensor for name, tensor in state_dict.items()
            }

        # For Gemma3 multimodal, the registered HF names in gguf-py is wrong,
        # Remap it manually. See:
        # https://github.com/ggml-org/llama.cpp/blob/f117be185ef1b76129e51d26676354af253bf664/gguf-py/gguf/tensor_mapping.py#L1379-L1381
        MM_PROJ_MAP = {
            "multi_modal_projector.mm_input_projection_weight": "multi_modal_projector.mm_input_projection",  # noqa: E501
        }

        def find_hf_name_in_tensor_map(hf_name: str) -> str | None:
            if hf_name.endswith((".weight", ".bias")):
                name, suffix = hf_name.rsplit(".", 1)
            else:
                name, suffix = hf_name, "weight"
            gguf_name = None
            # 1st: search mm_proj for multimodal
            if vision_name_map is not None:
                name_to_map = MM_PROJ_MAP.get(name, name)
                gguf_name = vision_name_map.get_name(name_to_map)
            # 2nd: search text backbone
            if gguf_name is None:
                gguf_name = text_name_map.get_name(name)
            if gguf_name is None:
                return None
            return gguf_name + "." + suffix

        for hf_name in state_dict:
            gguf_name_with_suffix = find_hf_name_in_tensor_map(hf_name)
            # Multimodal may have extra prefix which will cause lookup failure,
            # so try removing prefix and lookup again.
            # TODO(Isotr0py): Prefix with more than one "."?
            if gguf_name_with_suffix is None:
                gguf_name_with_suffix = find_hf_name_in_tensor_map(
                    hf_name.split(".", 1)[-1]
                )
            if (
                gguf_name_with_suffix is None
                and hf_name not in gguf_to_hf_name_map.values()
            ):
                raise RuntimeError(
                    f"Failed to to map gguf name for HF param: {hf_name}"
                )
            logger.debug(
                "Map GGUF name %s to HF param %s", gguf_name_with_suffix, hf_name
            )
            if gguf_name_with_suffix is not None:
                gguf_to_hf_name_map[gguf_name_with_suffix] = hf_name
        return gguf_to_hf_name_map

    def _create_mmproj_tensor_mapping(self, mmproj_path: str) -> dict[str, str]:
        """
        Create mapping from GGUF mmproj tensor names to vLLM parameter names
        using GGUF library's automatic mapping system with filtering.

        Uses automatic mapping from gguf library, filtered to only include
        tensors that actually exist in the GGUF file.


        Args:
            mmproj_path: Path to mmproj.gguf file

        Returns:
            Dictionary mapping GGUF names to vLLM parameter names
        """
        # Read GGUF file to get metadata and actual tensor list
        reader = gguf.GGUFReader(mmproj_path)

        # Get layer count from metadata
        num_layers_field = reader.get_field("clip.vision.block_count")
        if num_layers_field is None:
            raise ValueError(
                "Missing 'clip.vision.block_count' in mmproj.gguf metadata. "
                "Cannot determine number of vision transformer layers."
            )
        num_layers = int(num_layers_field.parts[-1])

        # Get set of tensors that actually exist in this GGUF file
        actual_gguf_tensors = {tensor.name for tensor in reader.tensors}

        # Get automatic mapping from GGUF library
        mmproj_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.MMPROJ, num_layers)

        # Create candidate mappings: GGUF name -> vLLM-compatible HF names
        gguf_to_vllm_candidates = {}
        for hf_name, (tensor_type, gguf_base_name) in mmproj_map.mapping.items():
            # Prefer vLLM-specific naming (vision_tower.*, multi_modal_projector.*)
            if (
                "vision_tower" in hf_name or "multi_modal_projector" in hf_name
            ) and gguf_base_name not in gguf_to_vllm_candidates:
                gguf_to_vllm_candidates[gguf_base_name] = hf_name

        # Build filtered mapping - only include tensors that exist in GGUF file
        mapping = {}
        for gguf_base, vllm_base in gguf_to_vllm_candidates.items():
            # Try both .weight and .bias suffixes
            for suffix in [".weight", ".bias"]:
                gguf_name = f"{gguf_base}{suffix}"

                # FILTER: Only include if this tensor actually exists in GGUF file
                if gguf_name in actual_gguf_tensors:
                    vllm_name = f"{vllm_base}{suffix}"

                    # Special case: Gemma3's mm_input_projection uses underscore
                    # instead of dot before weight/bias suffix
                    if vllm_name == "multi_modal_projector.mm_input_projection.weight":
                        vllm_name = "multi_modal_projector.mm_input_projection_weight"
                    elif vllm_name == "multi_modal_projector.mm_input_projection.bias":
                        vllm_name = "multi_modal_projector.mm_input_projection_bias"

                    mapping[gguf_name] = vllm_name

        # Gemma3-specific FFN correction: GGUF automatic mapping has swapped naming
        # GGUF auto: ffn_down -> fc2, ffn_up -> fc1
        # vLLM expects: ffn_down -> fc1, ffn_up -> fc2
        # This swap is necessary due to how Gemma3's MLP layers are structured in vLLM
        ffn_corrections = {}
        for gguf_name, vllm_name in mapping.items():
            if ".ffn_down." in gguf_name and ".mlp.fc2." in vllm_name:
                # Swap fc2 -> fc1 for ffn_down
                ffn_corrections[gguf_name] = vllm_name.replace(".mlp.fc2.", ".mlp.fc1.")
            elif ".ffn_up." in gguf_name and ".mlp.fc1." in vllm_name:
                # Swap fc1 -> fc2 for ffn_up
                ffn_corrections[gguf_name] = vllm_name.replace(".mlp.fc1.", ".mlp.fc2.")

        # Apply corrections
        mapping.update(ffn_corrections)

        # Validation: Ensure critical Gemma3 components are present
        expected_prefixes = [
            "mm.input_projection",
            "mm.soft_emb_norm",
            "v.patch_embd",
            "v.position_embd",
            "v.post_ln",
        ]
        for prefix in expected_prefixes:
            if not any(k.startswith(prefix) for k in mapping):
                logger.warning(
                    "Automatic GGUF mapping missing expected prefix: %s. "
                    "This may indicate an incompatible vision encoder architecture.",
                    prefix,
                )

        logger.info(
            "Created %d tensor mappings from GGUF automatic mapping "
            "(filtered from %d candidates)",
            len(mapping),
            len(gguf_to_vllm_candidates) * 2,
        )

        return mapping

    def _get_weights_iterator(
        self,
        model_config: ModelConfig,
        model_name_or_path: str,
        gguf_to_hf_name_map: dict[str, str],
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """
        Iterate over GGUF model weights, loading from both main model file and
        mmproj.gguf for multimodal Gemma3 models.

        For Gemma3 multimodal GGUF models:
        - Main file (gemma-3-*.gguf): Language model weights (model.*)
        - mmproj file (mmproj*.gguf): Vision tower + projector weights (v.*, mm.*)

        Yields:
            Tuples of (parameter_name, tensor) for all model weights
        """
        from pathlib import Path

        model_dir = Path(model_name_or_path).parent

        # Detect companion mmproj.gguf file for multimodal models
        mmproj_files = list(model_dir.glob("mmproj*.gguf"))
        has_mmproj = len(mmproj_files) > 0

        hf_config = model_config.hf_config
        is_multimodal = has_mmproj and hasattr(hf_config, "vision_config")

        if is_multimodal:
            # Multimodal: Append mmproj weights to backbone weights
            # This addresses the maintainer's FIXME about simply appending weights

            # Helper to remap backbone weights to language_model hierarchy
            def remap_backbone_weights(weights_iter):
                """Remap model.* → language_model.model.* for Gemma3 hierarchy"""
                for name, tensor in weights_iter:
                    if name.startswith("model."):
                        name = name.replace("model.", "language_model.model.", 1)
                    yield (name, tensor)

            # 1. Backbone (language model) weights from main GGUF file
            logger.info("Loading language model weights from main GGUF file...")
            backbone_iter = remap_backbone_weights(
                gguf_quant_weights_iterator(model_name_or_path, gguf_to_hf_name_map)
            )

            # 2. MM projector weights from mmproj.gguf
            mmproj_path = str(mmproj_files[0])
            mmproj_mapping = self._create_mmproj_tensor_mapping(mmproj_path)

            logger.info(
                "Loading vision tower and projector weights from %s...",
                mmproj_files[0].name,
            )
            mmproj_iter = gguf_quant_weights_iterator(mmproj_path, mmproj_mapping)

            # Append mmproj weights to backbone weights using iterator chaining
            yield from itertools.chain(backbone_iter, mmproj_iter)
        else:
            # Standard GGUF loading (text-only or non-Gemma3)
            yield from gguf_quant_weights_iterator(
                model_name_or_path, gguf_to_hf_name_map
            )

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config.model)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        local_model_path = self._prepare_weights(model_config.model)
        gguf_weights_map = self._get_gguf_weights_map(model_config)
        model.load_weights(
            self._get_weights_iterator(model_config, local_model_path, gguf_weights_map)
        )

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig
    ) -> nn.Module:
        device_config = vllm_config.device_config
        local_model_path = self._prepare_weights(model_config.model)
        gguf_weights_map = self._get_gguf_weights_map(model_config)
        # we can only know if tie word embeddings after mapping weights
        if "lm_head.weight" in get_gguf_extra_tensor_names(
            local_model_path, gguf_weights_map
        ):
            model_config.hf_config.update({"tie_word_embeddings": True})

        weight_type_map = get_gguf_weight_type_map(model_config.model, gguf_weights_map)

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
