# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Generator

import gguf
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

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
        num_layers = config.num_hidden_layers
        name_map = gguf.get_tensor_name_map(arch, num_layers)

        auto_cls = (
            AutoModelForImageTextToText if is_multimodal else AutoModelForCausalLM
        )
        with torch.device("meta"):
            dummy_model = auto_cls.from_config(
                config, trust_remote_code=model_config.trust_remote_code
            )

        state_dict = dummy_model.state_dict()

        for hf_name in state_dict:
            name, suffix = hf_name.rsplit(".", 1)
            gguf_name = name_map.get_name(name)
            gguf_to_hf_name_map[f"{gguf_name}.{suffix}"] = hf_name
        return gguf_to_hf_name_map

    def _create_mmproj_tensor_mapping(self, mmproj_path: str) -> dict[str, str]:
        """
        Create mapping from GGUF mmproj tensor names to HuggingFace parameter names
        for Gemma3 multimodal models.

        Args:
            mmproj_path: Path to mmproj.gguf file

        Returns:
            Dictionary mapping GGUF names to vLLM parameter names
        """
        # Read layer count from GGUF metadata instead of hardcoding
        reader = gguf.GGUFReader(mmproj_path)
        num_layers_field = reader.get_field("clip.vision.block_count")
        if num_layers_field is None:
            raise ValueError(
                "Missing 'clip.vision.block_count' in mmproj.gguf metadata. "
                "Cannot determine number of vision transformer layers."
            )
        num_layers = int(num_layers_field.parts[-1])

        mapping = {}

        # FIXME(Isotr0py, GGUF): Use automatic weights mapping:
        # see: https://github.com/ggml-org/llama.cpp/blob/392e09a60852d0e879d4bbedd5ace3e6852f719e/gguf-py/gguf/constants.py#L1060-L1129
        # Multimodal Projector Mappings
        mapping["mm.input_projection.weight"] = (
            "multi_modal_projector.mm_input_projection_weight"
        )
        mapping["mm.soft_emb_norm.weight"] = (
            "multi_modal_projector.mm_soft_emb_norm.weight"
        )

        # Vision Tower - Patch and position embeddings
        mapping["v.patch_embd.weight"] = (
            "vision_tower.vision_model.embeddings.patch_embedding.weight"
        )
        mapping["v.patch_embd.bias"] = (
            "vision_tower.vision_model.embeddings.patch_embedding.bias"
        )
        mapping["v.position_embd.weight"] = (
            "vision_tower.vision_model.embeddings.position_embedding.weight"
        )

        # Vision transformer layers (read from GGUF metadata)
        for layer_idx in range(num_layers):
            # Layer norms
            ln_base = f"vision_tower.vision_model.encoder.layers.{layer_idx}"
            mapping[f"v.blk.{layer_idx}.ln1.weight"] = f"{ln_base}.layer_norm1.weight"
            mapping[f"v.blk.{layer_idx}.ln1.bias"] = f"{ln_base}.layer_norm1.bias"
            mapping[f"v.blk.{layer_idx}.ln2.weight"] = f"{ln_base}.layer_norm2.weight"
            mapping[f"v.blk.{layer_idx}.ln2.bias"] = f"{ln_base}.layer_norm2.bias"

            # Attention with bias support
            attn_base = f"{ln_base}.self_attn"
            mapping[f"v.blk.{layer_idx}.attn_q.weight"] = f"{attn_base}.q_proj.weight"
            mapping[f"v.blk.{layer_idx}.attn_q.bias"] = f"{attn_base}.q_proj.bias"
            mapping[f"v.blk.{layer_idx}.attn_k.weight"] = f"{attn_base}.k_proj.weight"
            mapping[f"v.blk.{layer_idx}.attn_k.bias"] = f"{attn_base}.k_proj.bias"
            mapping[f"v.blk.{layer_idx}.attn_v.weight"] = f"{attn_base}.v_proj.weight"
            mapping[f"v.blk.{layer_idx}.attn_v.bias"] = f"{attn_base}.v_proj.bias"
            mapping[f"v.blk.{layer_idx}.attn_out.weight"] = (
                f"{attn_base}.out_proj.weight"
            )
            mapping[f"v.blk.{layer_idx}.attn_out.bias"] = f"{attn_base}.out_proj.bias"

            # FFN (swapped naming + bias)
            mlp_base = f"{ln_base}.mlp"
            mapping[f"v.blk.{layer_idx}.ffn_down.weight"] = f"{mlp_base}.fc1.weight"
            mapping[f"v.blk.{layer_idx}.ffn_up.weight"] = f"{mlp_base}.fc2.weight"
            mapping[f"v.blk.{layer_idx}.ffn_down.bias"] = f"{mlp_base}.fc1.bias"
            mapping[f"v.blk.{layer_idx}.ffn_up.bias"] = f"{mlp_base}.fc2.bias"

        # Post layer normalization
        mapping["v.post_ln.weight"] = "vision_tower.vision_model.post_layernorm.weight"
        mapping["v.post_ln.bias"] = "vision_tower.vision_model.post_layernorm.bias"

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
            # FIXME(Isotr0py): We only need to append MM proj's weights to
            # backbone's weights.

            # Multimodal: Load weights from TWO GGUF files

            # 1. Load language model weights from main GGUF file
            # Remap "model.*" → "language_model.model.*" to match
            # Gemma3ForConditionalGeneration hierarchy
            logger.info("Loading language model weights from main GGUF file...")
            for name, tensor in gguf_quant_weights_iterator(
                model_name_or_path, gguf_to_hf_name_map
            ):
                # Remap to language_model.* hierarchy
                if name.startswith("model."):
                    name = name.replace("model.", "language_model.model.", 1)
                yield (name, tensor)

            # 2. Load vision tower + projector weights from mmproj.gguf
            mmproj_path = str(mmproj_files[0])
            mmproj_mapping = self._create_mmproj_tensor_mapping(mmproj_path)

            logger.info(
                "Loading vision tower and projector weights from %s...",
                mmproj_files[0].name,
            )
            for name, tensor in gguf_quant_weights_iterator(
                mmproj_path, mmproj_mapping
            ):
                yield (name, tensor)
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
