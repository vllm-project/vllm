# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Separated from test_common.py because HF loading for PerceptronAI/Isaac-0.1
requires perceptron package (Run 'pip install perceptron').
"""

import pytest

pytest.importorskip("perceptron", reason="Requires 'pip install perceptron'")

import types
from pathlib import PosixPath

import torch
from perceptron.tensorstream import TextType
from perceptron.tensorstream.ops import compute_mrope_pos_tensor, modality_mask
from transformers.modeling_outputs import BaseModelOutputWithPast

from ....conftest import IMAGE_ASSETS, HfRunner, ImageTestAssets, VllmRunner
from .vlm_utils import runners
from .vlm_utils.case_filtering import get_parametrized_options
from .vlm_utils.types import ExpandableVLMTestArgs, VLMTestInfo, VLMTestType


def compute_position_ids_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
    r"""Create 3D positional indices for token input.
    Args:
        input_ids (`torch.Tensor`):
            Tensor of shape `(batch_size, seq_len)` containing token ids.
    Returns:
        `torch.Tensor`: Positional indices with shape `(batch_size, seq_len, 3)`
        where each channel duplicates the 1D position so it can be consumed by
        the 3-axis MRoPE rotary embedding.
    """
    batch_size, seq_length = input_ids.shape
    position_ids = torch.arange(seq_length, device=input_ids.device)
    position_ids = position_ids.view(1, -1).expand(batch_size, -1)
    position_ids = position_ids.unsqueeze(2).expand(-1, -1, 3)  # Add 3D for MRoPE
    return position_ids


def isaac_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    """Patch HF runner for Isaac:
    1) move processor outputs to model device
    2) ensure IsaacModel.forward returns hidden_states
    for compatibility with hidden_states_to_seq_logprobs()
    """

    model_device = next(hf_model.model.parameters()).device

    # ----------------------------
    # 1) Patch processor: move BatchFeature input_ids and TensorStream to model device
    # ----------------------------
    original_processor = hf_model.processor

    def patched_processor(*args, **kwargs):
        result = original_processor(*args, **kwargs)
        for k, v in result.data.items():
            result[k] = v.to(model_device)
        return result

    hf_model.processor = patched_processor

    # ----------------------------
    # 2) Patch IsaacModel.forward: add hidden_states to the output
    # ----------------------------
    isaac_model = hf_model.model.model  # IsaacModel

    def patched_forward(
        self,
        input_ids=None,
        tensor_stream=None,
        attention_mask=None,
        position_ids=None,
        modality_tensor=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        """
        Forward pass with MRoPE position embeddings.
        Computes position embeddings once and passes them through all layers.
        """
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Get inputs
        if tensor_stream is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both tensor_stream and inputs_embeds")
        elif tensor_stream is not None:
            # Embed TensorStream directly
            inputs_embeds = self.embed_stream(tensor_stream)
            # Create modality tensor if not provided
            if modality_tensor is None:
                modality_tensor = modality_mask(tensor_stream)
        elif input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
            # Create text modality tensor if not provided
            if modality_tensor is None:
                batch_size, seq_length = input_ids.shape
                modality_tensor = torch.full(
                    (batch_size, seq_length),
                    TextType.text.value,
                    device=input_ids.device,
                    dtype=torch.long,
                )
        elif inputs_embeds is None:
            raise ValueError(
                "You have to specify either tensor_stream, input_ids or inputs_embeds"
            )

        # Create default position_ids if not provided
        if position_ids is None:
            if tensor_stream is not None:
                position_ids = compute_mrope_pos_tensor(tensor_stream)  # (B,L,3)
            else:
                position_ids = compute_position_ids_input_ids(input_ids)

        # Compute MRoPE position embeddings if we have custom rotary_emb
        cos, sin = self.rotary_emb(position_ids, modality_tensor)
        cos = cos.to(inputs_embeds.dtype)
        sin = sin.to(inputs_embeds.dtype)

        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = self._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, False
            )

        # Initialize and collect hidden states
        all_hidden_states = ()
        hidden_states = inputs_embeds
        all_hidden_states += (hidden_states,)

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=(cos, sin),
                **kwargs,
            )

            hidden_states = (
                layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
            )
            all_hidden_states += (hidden_states,)

        # Final layer norm
        hidden_states = self.norm(hidden_states)
        all_hidden_states += (hidden_states,)

        # Include hiden_states for compatibility with hidden_states_to_seq_logprobs()
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )

    isaac_model.forward = types.MethodType(patched_forward, isaac_model)

    return hf_model


ISAAC_TEST_SETTINGS = {
    "isaac": VLMTestInfo(
        models=["PerceptronAI/Isaac-0.1"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: (
            f"<|im_start|>User\n{img_prompt}<|im_end|>\n<|im_start|>assistant\n"
        ),
        img_idx_to_prompt=lambda idx: "<image>",
        single_image_prompts=IMAGE_ASSETS.prompts(
            {
                "stop_sign": "<vlm_image>Please describe the image shortly.",
                "cherry_blossom": "<vlm_image>Please infer the season with reason.",
            }
        ),
        multi_image_prompt=(
            "Picture 1: <vlm_image>\n"
            "Picture 2: <vlm_image>\n"
            "Describe these two images with one paragraph respectively."
        ),
        enforce_eager=False,
        max_model_len=4096,
        max_num_seqs=2,
        hf_model_kwargs={"device_map": "auto"},
        patch_hf_runner=isaac_patch_hf_runner,
        image_size_factors=[(0.25,), (0.25, 0.25, 0.25), (0.25, 0.2, 0.15)],
    )
}


### Test wrappers
# Wrappers around the test running func for:
# - single image
# - multi-image
@pytest.mark.parametrize(
    "model_type,test_case",
    get_parametrized_options(
        ISAAC_TEST_SETTINGS,
        test_type=VLMTestType.IMAGE,
        create_new_process_for_each_test=False,
    ),
)
def test_isaac_single_image(
    tmp_path: PosixPath,
    model_type: str,
    test_case: ExpandableVLMTestArgs,
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    image_assets: ImageTestAssets,
):
    model_test_info = ISAAC_TEST_SETTINGS[model_type]
    runners.run_single_image_test(
        tmp_path=tmp_path,
        model_test_info=model_test_info,
        test_case=test_case,
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        image_assets=image_assets,
    )


@pytest.mark.parametrize(
    "model_type,test_case",
    get_parametrized_options(
        ISAAC_TEST_SETTINGS,
        test_type=VLMTestType.MULTI_IMAGE,
        create_new_process_for_each_test=False,
    ),
)
def test_isaac_multi_image(
    tmp_path: PosixPath,
    model_type: str,
    test_case: ExpandableVLMTestArgs,
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    image_assets: ImageTestAssets,
):
    model_test_info = ISAAC_TEST_SETTINGS[model_type]
    runners.run_multi_image_test(
        tmp_path=tmp_path,
        model_test_info=model_test_info,
        test_case=test_case,
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        image_assets=image_assets,
    )
