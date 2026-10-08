# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import regex

from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from vllm.model_executor.models.utils import WeightsMapper


class Cosmos3ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    # Cosmos3 unified checkpoints store a generation tower alongside the Qwen3-VL
    # understanding tower, whose weights are dropped here.
    hf_to_vllm_mapper = (
        Qwen3VLForConditionalGeneration.hf_to_vllm_mapper
        | WeightsMapper(
            orig_to_new_regex={
                regex.compile(r"^audio_modality_embed(?:\..*)?$"): None,
                regex.compile(r"^action_modality_embed(?:\..*)?$"): None,
            },
            orig_to_new_substr={
                "_moe_gen": None,
                ".add_q_proj.": None,
                ".add_k_proj.": None,
                ".add_v_proj.": None,
                ".to_add_out.": None,
                ".norm_added_q.": None,
                ".norm_added_k.": None,
                # ModelOpt-native dialect (diffusers/transformers read these; vLLM reads
                # weight_scale/input_scale instead), drop so AutoWeightsLoader passes
                ".input_quantizer.": None,
                ".weight_quantizer.": None,
                ".output_quantizer.": None,
            },
            orig_to_new_prefix={
                "proj_in.": None,
                "proj_out.": None,
                "time_embedder.": None,
                "audio_proj_in.": None,
                "audio_proj_out.": None,
                "action_proj_in.": None,
                "action_proj_out.": None,
            },
        )
    )

    # Cosmos3 unified Diffusers checkpoints store reasoner weights across
    # transformer/ and vision_encoder/. Match both while excluding VAE and
    # sound_tokenizer weights; the mapper drops generation-only tensors.
    allow_patterns_overrides = ["[tv]*er/*.safetensors"]
