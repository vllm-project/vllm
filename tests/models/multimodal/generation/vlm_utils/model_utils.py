# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Common utility functions relating to different models that are useful
for manipulating the input / output of HF & vLLM test runners, which are
typically specific to a small subset of models.
"""

import logging
import types
import warnings
from pathlib import PosixPath

import numpy as np
import numpy.typing as npt
import PIL.Image
import pytest
import regex as re
import torch
from PIL.Image import Image
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BatchFeature,
    GenerationConfig,
    GenerationMixin,
)
from transformers.video_utils import VideoMetadata

from vllm.logprobs import SampleLogprobs
from vllm.platforms import current_platform
from vllm.utils.collection_utils import is_list_of

from .....conftest import HfRunner, ImageAsset, ImageTestAssets
from .types import RunnerOutput

logger = logging.getLogger(__name__)


####### vLLM output processors functions
def blip2_vllm_to_hf_output(vllm_output: RunnerOutput, model: str) -> RunnerOutput:
    """Sanitize vllm output [blip2 models] to be comparable with hf output."""
    _, output_str, out_logprobs = vllm_output

    hf_output_str = output_str + "\n"

    tokenizer = AutoTokenizer.from_pretrained(model)
    hf_output_ids = tokenizer.encode(hf_output_str)
    assert hf_output_ids[0] == tokenizer.bos_token_id
    hf_output_ids = hf_output_ids[1:]

    return hf_output_ids, hf_output_str, out_logprobs


def fuyu_vllm_to_hf_output(vllm_output: RunnerOutput, model: str) -> RunnerOutput:
    """Sanitize vllm output [fuyu models] to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    hf_output_str = output_str.lstrip() + "|ENDOFTEXT|"

    return output_ids, hf_output_str, out_logprobs


def qwen_vllm_to_hf_output(
    vllm_output: RunnerOutput, model: str
) -> tuple[list[int], str, SampleLogprobs | None]:
    """Sanitize vllm output [qwen models] to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    hf_output_str = output_str + "<|endoftext|>"

    return output_ids, hf_output_str, out_logprobs


def qwen2_vllm_to_hf_output(
    vllm_output: RunnerOutput, model: str
) -> tuple[list[int], str, SampleLogprobs | None]:
    """Sanitize vllm output [qwen2 models] to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    hf_output_str = output_str + "<|im_end|>"

    return output_ids, hf_output_str, out_logprobs


def kimiv_vl_vllm_to_hf_output(
    vllm_output: RunnerOutput, model: str
) -> tuple[list[int], str, SampleLogprobs | None]:
    """Sanitize vllm output [kimi_vl models] to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    hf_output_str = output_str + "<|im_end|>[EOS]"

    return output_ids, hf_output_str, out_logprobs


def llava_image_vllm_to_hf_output(
    vllm_output: RunnerOutput, model: str
) -> RunnerOutput:
    config = AutoConfig.from_pretrained(model)
    mm_token_id = config.image_token_index
    return _llava_vllm_to_hf_output(vllm_output, model, mm_token_id)


def llava_video_vllm_to_hf_output(
    vllm_output: RunnerOutput, model: str
) -> tuple[list[int], str, SampleLogprobs | None]:
    config = AutoConfig.from_pretrained(model)
    mm_token_id = config.video_token_index
    return _llava_vllm_to_hf_output(vllm_output, model, mm_token_id)


def _llava_vllm_to_hf_output(
    vllm_output: RunnerOutput, model: str, mm_token_id: int
) -> RunnerOutput:
    """Sanitize vllm output [Llava models] to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    tokenizer = AutoTokenizer.from_pretrained(model)
    eos_token_id = tokenizer.eos_token_id

    hf_output_ids = [
        token_id
        for idx, token_id in enumerate(output_ids)
        if token_id != mm_token_id or output_ids[idx - 1] != mm_token_id
    ]

    # output_str[0] is not " " in some cases, e.g., Granite Vision,
    # but for most llava based models, this is the case
    hf_output_str = output_str[1:] if output_str[0] == " " else output_str

    if hf_output_ids[-1] == eos_token_id:
        hf_output_str = hf_output_str + tokenizer.decode(eos_token_id)

    return hf_output_ids, hf_output_str, out_logprobs


def llava_onevision_hf_model_kwargs(model: str) -> dict:
    """Workaround to fix the sliding window issue in llava_onevision."""
    config = AutoConfig.from_pretrained(model)
    config.text_config.sliding_window = None
    return config.to_dict()


def llava_onevision_vllm_to_hf_output(
    vllm_output: RunnerOutput, model: str
) -> RunnerOutput:
    """Sanitize vllm output [llava-onevision] to compare with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    config = AutoConfig.from_pretrained(model)
    video_token_id = config.video_token_index

    tokenizer = AutoTokenizer.from_pretrained(model)
    eos_token_id = tokenizer.eos_token_id

    hf_output_ids = [
        token_id
        for idx, token_id in enumerate(output_ids)
        if token_id != video_token_id or output_ids[idx - 1] != video_token_id
    ]

    hf_output_str = output_str
    if hf_output_ids[-1] == eos_token_id:
        hf_output_str = hf_output_str + tokenizer.decode(eos_token_id)

    return hf_output_ids, hf_output_str, out_logprobs


def mantis_vllm_to_hf_output(vllm_output: RunnerOutput, model: str) -> RunnerOutput:
    """Sanitize vllm output [mantis] to compare with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    hf_output_str = output_str + "<|eot_id|>"

    return output_ids, hf_output_str, out_logprobs


def phi3v_vllm_to_hf_output(vllm_output: RunnerOutput, model: str) -> RunnerOutput:
    """Sanitize vllm output [phi3v] to be comparable with hf output."""
    _, output_str, out_logprobs = vllm_output

    output_str_without_image = re.sub(r"(<\|image_\d+\|>)+", "", output_str)
    assert output_str_without_image[0] == " "
    output_str_without_image = output_str_without_image[1:]

    hf_output_str = output_str_without_image + "<|end|><|endoftext|>"

    tokenizer = AutoTokenizer.from_pretrained(model)
    hf_output_ids = tokenizer.encode(output_str_without_image)
    assert hf_output_ids[0] == 1
    hf_output_ids = hf_output_ids[1:]

    return hf_output_ids, hf_output_str, out_logprobs


def paligemma_vllm_to_hf_output(vllm_output: RunnerOutput, model: str) -> RunnerOutput:
    """Sanitize vllm output to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    config = AutoConfig.from_pretrained(model)
    image_token_id = config.image_token_index

    tokenizer = AutoTokenizer.from_pretrained(model)
    eos_token_id = tokenizer.eos_token_id

    hf_output_ids = [
        token_id
        for idx, token_id in enumerate(output_ids)
        if token_id != image_token_id or output_ids[idx - 1] != image_token_id
    ]

    hf_output_str = output_str

    if hf_output_ids[-1] == eos_token_id:
        hf_output_str = hf_output_str + tokenizer.decode(eos_token_id)

    return hf_output_ids, hf_output_str, out_logprobs


####### Post-processors for HF outputs
def deepseekvl2_trunc_hf_output(hf_output: RunnerOutput, model: str) -> RunnerOutput:
    output_ids, output_str, out_logprobs = hf_output
    if output_str.endswith("<｜end▁of▁sentence｜>"):
        output_str = output_str.split("<｜end▁of▁sentence｜>")[0]
    return output_ids, output_str, out_logprobs


def idefics3_trunc_hf_output(hf_output: RunnerOutput, model: str) -> RunnerOutput:
    output_ids, output_str, out_logprobs = hf_output
    if output_str.endswith("<end_of_utterance>"):
        output_str = output_str.split("<end_of_utterance>")[0]
    return output_ids, output_str, out_logprobs


def smolvlm_trunc_hf_output(hf_output: RunnerOutput, model: str) -> RunnerOutput:
    # Based on Idefics3
    return idefics3_trunc_hf_output(hf_output, model)


def minicpmv_trunc_hf_output(hf_output: RunnerOutput, model: str) -> RunnerOutput:
    output_ids, output_str, out_logprobs = hf_output
    if output_str.endswith("<|eot_id|>"):
        output_str = output_str.split("<|eot_id|>")[0]
    return output_ids, output_str, out_logprobs


def minimax_vl_01_hf_output(hf_output: RunnerOutput, model: str) -> RunnerOutput:
    output_ids, output_str, out_logprobs = hf_output
    if output_str.endswith("<end_of_sentence>"):
        output_str = output_str.split("<end_of_sentence>")[0]
    return output_ids, output_str, out_logprobs


def ultravox_trunc_hf_output(hf_output: RunnerOutput, model: str) -> RunnerOutput:
    output_ids, output_str, out_logprobs = hf_output

    tokenizer = AutoTokenizer.from_pretrained(model)
    eos_token_id = tokenizer.eos_token_id
    eos_token = tokenizer.decode(eos_token_id)
    if output_str.endswith(eos_token):
        output_str = output_str.split(eos_token)[0]
    return output_ids, output_str, out_logprobs


####### Functions for converting image assets to embeddings
def get_llava_embeddings(image_assets: ImageTestAssets):
    return [asset.image_embeds for asset in image_assets]


####### Prompt path encoders for models that need models on disk
def qwen_prompt_path_encoder(
    tmp_path: PosixPath, prompt: str, assets: list[ImageAsset] | ImageTestAssets
) -> str:
    """Given a temporary dir path, export one or more image assets into the
    tempdir & replace its contents with the local path to the string so that
    the HF version of Qwen-VL can resolve the path and load the image in its
    forward() call.

    Args:
        tmp_path: Tempdir for test under consideration.
        prompt: Prompt with image placeholders.
        assets: list of image assets whose len equals the num placeholders.
    """
    # Ensure that the number of placeholders matches the number of assets;
    # If this is not true, the test is probably written incorrectly.
    assert prompt.count("<img></img>") == len(assets)

    # Replace the placeholders with local paths to the exported assets
    for asset in assets:
        image_tmp_path = tmp_path / f"{asset.name}.jpg"
        asset.pil_image.save(image_tmp_path)
        prompt = prompt.replace(
            "<img></img>",
            f"<img>{image_tmp_path}</img>",
            1,
        )
    return prompt


####### Model-specific HuggingFace runner patchers
def deepseekvl2_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    """Patches and returns an instance of the HfRunner to use for GLM4."""
    hf_processor = hf_model.processor

    def processor(*args, text="", images=None, **kwargs):
        if isinstance(images, Image):
            images = [images]
        # inputs is a custom class instead of dict or BatchFeature
        inputs = hf_processor(
            *args,
            prompt=text,
            images=images,
            **kwargs,
        )
        inputs = {
            k: inputs[k]
            for k in inputs.keys()  # noqa
            if k not in ("seq_lens", "sft_format")
        }
        return BatchFeature(data=inputs, tensor_type="pt")

    hf_model.processor = processor
    hf_model.model.get_output_embeddings = (
        lambda: hf_model.model.language.model.embed_tokens
    )
    return hf_model


def gemma3_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    """Patches and returns an instance of the HfRunner to use for Gemma 3."""
    hf_processor = hf_model.processor

    def processor(*args, **kwargs):
        return hf_processor(*args, do_pan_and_scan=True, **kwargs)

    hf_model.processor = processor

    orig_generate = hf_model.model.generate

    def _generate(self, *args, **kwargs):
        # FIXME: https://github.com/huggingface/transformers/issues/38333
        kwargs["disable_compile"] = True

        return orig_generate(*args, **kwargs)

    hf_model.model.generate = types.MethodType(_generate, hf_model.model)

    return hf_model


def gemma3_vllm_to_hf_output(vllm_output: RunnerOutput, model: str) -> RunnerOutput:
    """Sanitize vllm output [gemma-3] to compare with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    config = AutoConfig.from_pretrained(model)
    image_token_id = config.image_token_id

    tokenizer = AutoTokenizer.from_pretrained(model)
    eos_token_id = tokenizer.eos_token_id

    hf_output_ids = [
        token_id
        for idx, token_id in enumerate(output_ids)
        if token_id != image_token_id
    ]

    hf_output_str = output_str
    if hf_output_ids[-1] == eos_token_id:
        hf_output_str = hf_output_str + tokenizer.decode(eos_token_id)

    return hf_output_ids, hf_output_str, out_logprobs


def glm4v_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    """Patches and returns an instance of the HfRunner to use for GLM4V."""
    if current_platform.is_rocm():
        import types

        config = hf_model.model.config
        if hasattr(config, "num_layers") and not hasattr(config, "num_hidden_layers"):
            config.num_hidden_layers = config.num_layers
        config.output_hidden_states = True

        def patched_prepare_cache(
            self, generation_config, model_kwargs, *args, **kwargs
        ):
            model_kwargs["past_key_values"] = None
            model_kwargs["use_cache"] = False
            return model_kwargs

        hf_model.model._prepare_cache_for_generation = types.MethodType(
            patched_prepare_cache, hf_model.model
        )
        original_generate = hf_model.model.generate

        def patched_generate(*args, **kwargs):
            kwargs["output_hidden_states"] = True
            kwargs["return_dict_in_generate"] = True
            return original_generate(*args, **kwargs)

        hf_model.model.generate = patched_generate
        original_forward = hf_model.model.forward

        def patched_forward(*args, **kwargs):
            kwargs["output_hidden_states"] = True
            return original_forward(*args, **kwargs)

        hf_model.model.forward = patched_forward

    hf_processor = hf_model.processor

    def processor(*args, text="", images=None, **kwargs):
        if images is None:
            return hf_processor(*args, **kwargs)

        images = [images] if isinstance(images, Image) else images

        contents = re.findall(
            r"<\|begin_of_image\|><\|endoftext\|><\|end_of_image\|>(.*?)<\|assistant\|>",
            text,
        )
        assert len(contents) == len(images)

        return hf_processor.apply_chat_template(
            [
                {"role": "user", "image": image, "content": content}
                for image, content in zip(images, contents)
            ],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            **kwargs,
        )

    hf_model.processor = processor
    hf_model.model.get_output_embeddings = (
        lambda: hf_model.model.transformer.output_layer
    )
    return hf_model


def glm4_1v_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    """Patches and returns an instance of the HfRunner to use for GLM4.1V."""
    hf_processor = hf_model.processor

    def processor(*args, videos=None, **kwargs):
        if videos is not None and is_list_of(videos, tuple):
            # If videos is a list of tuples, we assume each tuple contains
            # (video_array, metadata) as in the case of GLM4.1V.
            # Filter out 'do_sample_frames' as it's not a valid VideoMetadata arg
            video_metadata = [
                [
                    VideoMetadata(
                        **{k: v for k, v in video[1].items() if k != "do_sample_frames"}
                    )
                ]
                for video in videos
            ]
            videos = [[video[0]] for video in videos]
        else:
            video_metadata = None

        return hf_processor(
            *args, videos=videos, video_metadata=video_metadata, **kwargs
        )

    hf_model.processor = processor
    return hf_model


def h2ovl_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    """Patches and returns an instance of the HfRunner to use for H2OVL."""

    class H2OVLProcessor:
        """A simple processor for H2OVL models."""

        def __init__(self, hf_runner: HfRunner):
            self.num_image_token = hf_runner.model.num_image_token
            self.tokenizer = hf_runner.tokenizer

            self.config = AutoConfig.from_pretrained(
                hf_runner.model_name, trust_remote_code=True
            )
            self.vision_config = self.config.vision_config
            self.use_thumbnail = self.config.use_thumbnail
            self.use_msac = self.config.use_msac
            self.min_num = self.config.min_dynamic_patch
            self.max_num = self.config.max_dynamic_patch
            self.image_size = self.vision_config.image_size

        def __call__(self, text: str, images: Image | list[Image], **kwargs):
            from vllm.model_executor.models.h2ovl import (
                IMG_CONTEXT,
                IMG_END,
                IMG_START,
                image_to_pixel_values_h2ovl,
            )

            images = [images] if isinstance(images, Image) else images
            pixel_values = [
                image_to_pixel_values_h2ovl(
                    image,
                    input_size=self.image_size,
                    min_num=self.min_num,
                    max_num=self.max_num,
                    use_thumbnail=self.use_thumbnail,
                    use_msac=self.use_msac,
                )
                for image in images
            ]
            num_patches_list = [pixel_value.shape[0] for pixel_value in pixel_values]
            pixel_values = torch.cat(pixel_values, dim=0)
            for num_patches in num_patches_list:
                context_tokens = IMG_CONTEXT * self.num_image_token * num_patches
                image_tokens = IMG_START + context_tokens + IMG_END
                text = text.replace("<image>", image_tokens, 1)
            prompt = self.tokenizer(text, return_tensors="pt")
            prompt.update({"pixel_values": pixel_values})
            return prompt

    img_context_token_id = hf_model.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    hf_model.model.img_context_token_id = img_context_token_id
    hf_model.processor = H2OVLProcessor(hf_model)
    hf_model.model.get_output_embeddings = (
        lambda: hf_model.model.language_model.get_output_embeddings()
    )
    hf_model.model.generate = types.MethodType(_internvl_generate, hf_model.model)
    return hf_model


def isaac_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    """Patch HF runner for Isaac:
    1) Move processor outputs to model device
    2) Ensure IsaacModel.forward returns hidden_states
    for compatibility with hidden_states_to_seq_logprobs()
    """

    from perceptron.tensorstream import TextType
    from perceptron.tensorstream.ops import compute_mrope_pos_tensor, modality_mask
    from transformers.modeling_outputs import BaseModelOutputWithPast

    def compute_position_ids_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
        """
        Create 3D positional indices for token input.
        """
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, device=input_ids.device)
        position_ids = position_ids.view(1, -1).expand(batch_size, -1)
        position_ids = position_ids.unsqueeze(2).expand(-1, -1, 3)  # Add 3D for MRoPE
        return position_ids

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

    tokenizer = AutoTokenizer.from_pretrained(
        hf_model.model_name, trust_remote_code=True
    )

    original_generate = hf_model.model.generate

    def patched_generate(*args, **kwargs):
        kwargs["pad_token_id"] = tokenizer.eos_token_id
        kwargs["eos_token_id"] = tokenizer.eos_token_id
        return original_generate(*args, **kwargs)

    hf_model.model.generate = patched_generate

    # ----------------------------
    # 2) Patch IsaacModel.forward: add hidden_states to the output
    # ----------------------------
    isaac_model = hf_model.model.model

    # [ROCm] Disable Flash/MemEfficient SDP on ROCm to avoid HF Transformers
    # accuracy issues: https://github.com/vllm-project/vllm/issues/30167
    # TODO: Remove once ROCm SDP accuracy issues are resolved on HuggingFace
    # ----------------------------
    from ...conftest import patch_hf_vision_attn_for_rocm

    try:
        patch_hf_vision_attn_for_rocm(hf_model.model)
    except AttributeError as e:
        if "vision_config" in str(e):
            warnings.warn(
                f"Skipping ROCm vision attention patch for Isaac model: {e}. "
                "This is expected for models without vision_config in "
                "attention layers (e.g., Siglip2VariableLengthAttention).",
                stacklevel=2,
            )
        else:
            logger.error(
                "Unexpected AttributeError during ROCm vision attention patch: %s. "
                "Model type: %s. Inner model type: %s.",
                e,
                type(hf_model.model).__name__,
                type(getattr(hf_model.model, "model", None)).__name__,
            )
            raise

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
        hidden_states = inputs_embeds
        hidden_states_list: list[torch.Tensor] = []

        if output_hidden_states:
            hidden_states_list.append(hidden_states)

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

            if output_hidden_states:
                hidden_states_list.append(hidden_states)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            hidden_states_list.append(hidden_states)

        # Convert to tuple or None
        all_hidden_states = tuple(hidden_states_list) if output_hidden_states else None

        # Include hiden_states for compatibility with hidden_states_to_seq_logprobs()
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )

    isaac_model.forward = types.MethodType(patched_forward, isaac_model)

    return hf_model


def skyworkr1v_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    """Patches and returns an instance of the HfRunner to use for SkyworkR1V."""

    class SkyworkR1VProcessor:
        """A simple processor for SkyworkR1V."""

        def __init__(self, hf_runner: HfRunner):
            self.num_image_token = hf_runner.model.num_image_token
            self.tokenizer = hf_runner.tokenizer

            self.config = AutoConfig.from_pretrained(
                hf_runner.model_name, trust_remote_code=True
            )
            self.vision_config = self.config.vision_config
            self.use_thumbnail = self.config.use_thumbnail
            self.min_num = self.config.min_dynamic_patch
            self.max_num = self.config.max_dynamic_patch
            self.image_size = self.vision_config.image_size

        def __call__(self, text: str, images: Image | list[Image], **kwargs):
            from vllm.model_executor.models.skyworkr1v import (
                IMG_CONTEXT,
                IMG_END,
                IMG_START,
                image_to_pixel_values_skyworkr1v,
            )

            images = [images] if isinstance(images, Image) else images
            pixel_values = [
                image_to_pixel_values_skyworkr1v(
                    image,
                    input_size=self.image_size,
                    min_num=self.min_num,
                    max_num=self.max_num,
                    use_thumbnail=self.use_thumbnail,
                )
                for image in images
            ]
            num_patches_list = [pixel_value.shape[0] for pixel_value in pixel_values]
            pixel_values = torch.cat(pixel_values, dim=0)
            for num_patches in num_patches_list:
                context_tokens = IMG_CONTEXT * self.num_image_token * num_patches
                image_tokens = IMG_START + context_tokens + IMG_END
                text = text.replace("<image>", image_tokens, 1)
            prompt = self.tokenizer(text, return_tensors="pt")
            prompt.update({"pixel_values": pixel_values})
            return prompt

    img_context_token_id = hf_model.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    hf_model.model.img_context_token_id = img_context_token_id
    hf_model.processor = SkyworkR1VProcessor(hf_model)
    hf_model.model.get_output_embeddings = (
        lambda: hf_model.model.language_model.get_output_embeddings()
    )
    hf_model.model.generate = types.MethodType(_internvl_generate, hf_model.model)
    return hf_model


def internvl_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    """Patches and returns an instance of the HfRunner to use for InternVL."""

    class InternVLProcessor:
        """A simple processor for InternVL2 which misses a processor."""

        def __init__(self, hf_runner: HfRunner):
            self.num_image_token = hf_runner.model.num_image_token
            self.tokenizer = hf_runner.tokenizer

            self.config = AutoConfig.from_pretrained(
                hf_runner.model_name, trust_remote_code=True
            )
            self.vision_config = self.config.vision_config
            self.use_thumbnail = self.config.use_thumbnail
            self.min_num = self.config.min_dynamic_patch
            self.max_num = self.config.max_dynamic_patch
            self.image_size = self.vision_config.image_size

        def __call__(
            self,
            text: str,
            images: Image | list[Image] = None,
            videos: npt.NDArray | list[npt.NDArray] = None,
            **kwargs,
        ):
            from vllm.model_executor.models.internvl import (
                IMG_CONTEXT,
                IMG_END,
                IMG_START,
                image_to_pixel_values_internvl,
                video_to_pixel_values_internvl,
            )

            images = [images] if isinstance(images, Image) else images
            videos = [videos] if isinstance(videos, np.ndarray) else videos
            if images is not None:
                pixel_values_images = [
                    image_to_pixel_values_internvl(
                        image,
                        input_size=self.image_size,
                        min_num=self.min_num,
                        max_num=self.max_num,
                        use_thumbnail=self.use_thumbnail,
                    )
                    for image in images
                ]
                num_patches_images = [
                    pixel_value.shape[0] for pixel_value in pixel_values_images
                ]
            else:
                pixel_values_images, num_patches_images = [], []

            if videos is not None:
                pixel_values_videos = [
                    video_to_pixel_values_internvl(
                        video,
                        input_size=self.image_size,
                        min_num=1,
                        max_num=1,
                        use_thumbnail=False,
                    )
                    for video in videos
                ]
                num_patches_videos = [
                    pixel_value.shape[0] for pixel_value in pixel_values_videos
                ]
            else:
                pixel_values_videos, num_patches_videos = [], []

            pixel_values = []
            while ("<image>" in text) or ("<video>" in text):
                image_index = text.find("<image>")
                video_index = text.find("<video>")
                if image_index == -1 or (
                    video_index > -1 and video_index < image_index
                ):
                    num_patches = num_patches_videos.pop(0)
                    pixel_values.append(pixel_values_videos.pop(0))
                    context_tokens = (
                        IMG_START + IMG_CONTEXT * self.num_image_token + IMG_END
                    )
                    video_tokens = "".join(
                        [f"Frame{i + 1}: {context_tokens}" for i in range(num_patches)]
                    )
                    text = text.replace("<video>", video_tokens, 1)
                else:
                    num_patches = num_patches_images.pop(0)
                    pixel_values.append(pixel_values_images.pop(0))
                    context_tokens = IMG_CONTEXT * self.num_image_token * num_patches
                    image_tokens = IMG_START + context_tokens + IMG_END
                    text = text.replace("<image>", image_tokens, 1)
            pixel_values = torch.cat(pixel_values, dim=0)

            prompt = self.tokenizer(text, return_tensors="pt")
            prompt.update({"pixel_values": pixel_values})
            return prompt

    img_context_token_id = hf_model.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    hf_model.model.img_context_token_id = img_context_token_id
    hf_model.processor = InternVLProcessor(hf_model)
    hf_model.model.get_output_embeddings = (
        lambda: hf_model.model.language_model.get_output_embeddings()
    )
    hf_model.model.generate = types.MethodType(_internvl_generate, hf_model.model)
    return hf_model


def _internvl_generate(
    self,
    pixel_values: torch.FloatTensor,
    input_ids: torch.FloatTensor,
    attention_mask: torch.LongTensor | None = None,
    **generate_kwargs,
) -> torch.LongTensor:
    """Generate method for InternVL2 model without fixed use_cache."""
    assert self.img_context_token_id is not None
    target_dtype = next(self.parameters()).dtype
    vit_embeds = self.extract_feature(pixel_values.to(target_dtype))
    input_embeds = self.language_model.get_input_embeddings()(input_ids)
    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)

    input_ids = input_ids.reshape(B * N)
    selected = input_ids == self.img_context_token_id
    assert selected.sum() != 0
    input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

    input_embeds = input_embeds.reshape(B, N, C)

    forward_kwargs = dict(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
    )
    if getattr(self, "use_visual_token_mask", False):
        visual_token_mask = selected.reshape(B, N, 1).to(input_embeds.dtype)
        forward_kwargs["visual_token_mask"] = visual_token_mask

    # e.g. InternVL2-2B
    if not isinstance(self.language_model, GenerationMixin):
        pytest.skip("HF impl is not compatible with current transformers")

    outputs = self.language_model.generate(
        **forward_kwargs,
        **generate_kwargs,
    )

    return outputs


def mantis_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    from mantis.models.mllava import MLlavaProcessor

    hf_model.processor = MLlavaProcessor.from_pretrained(hf_model.model_name)

    orig_generate = hf_model.model.generate
    tokenizer = hf_model.processor.tokenizer

    def _generate(self, *args, **kwargs):
        return orig_generate(
            *args,
            **kwargs,
            eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ],
        )

    hf_model.model.generate = types.MethodType(_generate, hf_model.model)

    return hf_model


def minicpmv_25_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    orig_generate = hf_model.model.generate

    def _generate(
        self,
        *args,
        input_ids=None,
        pixel_values=None,
        image_sizes=None,
        image_bound=None,
        tgt_sizes=None,
        **kwargs,
    ):
        model_inputs = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
            "image_bound": image_bound,
            "tgt_sizes": tgt_sizes,
        }
        for k in list(model_inputs.keys()):
            if model_inputs[k] is None:
                model_inputs.pop(k)

        return orig_generate(model_inputs, *args, decode_text=False, **kwargs)

    hf_model.model.generate = types.MethodType(_generate, hf_model.model)

    return hf_model


def minicpmo_26_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    orig_generate = hf_model.model.generate

    def _generate(self, *args, image_sizes=None, **kwargs):
        return orig_generate(*args, decode_text=False, **kwargs)

    hf_model.model.generate = types.MethodType(_generate, hf_model.model)

    return hf_model


def minicpmv_26_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    orig_generate = hf_model.model.generate

    def _generate(self, *args, image_sizes=None, **kwargs):
        return orig_generate(*args, decode_text=False, **kwargs)

    hf_model.model.generate = types.MethodType(_generate, hf_model.model)

    return hf_model


def minimax_vl_01_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    orig_generate = hf_model.model.generate

    def _generate(self, *args, image_sizes=None, **kwargs):
        return orig_generate(*args, decode_text=False, **kwargs)

    hf_model.model.generate = types.MethodType(_generate, hf_model.model)

    return hf_model


def molmo_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    """Patches and returns an instance of the HfRunner to use for Molmo."""
    hf_processor = hf_model.processor

    def _processor(*args, **kwargs):
        return hf_processor.process(*args, **kwargs)

    hf_model.processor = _processor

    def _generate(self, max_new_tokens=None, do_sample=None, **kwargs):
        batch = {
            k: kwargs.pop(k).unsqueeze(0)
            for k in ("input_ids", "images", "image_input_idx", "image_masks")
            if k in kwargs
        }
        batch = BatchFeature(batch).to(dtype=self.dtype)

        return self.generate_from_batch(
            batch,
            generation_config=GenerationConfig(
                max_new_tokens=max_new_tokens,
                stop_strings="<|endoftext|>",
                do_sample=do_sample,
            ),
            **kwargs,
        )

    hf_model.model.generate = types.MethodType(_generate, hf_model.model)

    return hf_model


def ovis_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    """Patches and returns an instance of the HfRunner to use for Ovis2."""
    hf_model.model.get_output_embeddings = (
        lambda: hf_model.model.llm.get_output_embeddings()
    )

    def processor(*args, text="", images=None, **kwargs):
        text_tokenizer = hf_model.model.get_text_tokenizer()
        images = [images] if isinstance(images, Image) else images

        prompt_start_and_end = {
            "qwen2": ("<|im_start|>user\n", "<|im_end|>\n"),
            "llama": ("<|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|>"),
            "gemma2": ("<start_of_turn>user\n", "<end_of_turn>\n"),
        }
        for start, end in prompt_start_and_end.values():
            if start in text and end in text:
                text = text.split(start)[1].split(end)[0]
                break

        prompt, input_ids, pixel_values = hf_model.model.preprocess_inputs(
            text_or_conversations=text, images=images
        )
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)

        inputs = {
            "inputs": input_ids.unsqueeze(0),
            "pixel_values": pixel_values.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
        }
        return BatchFeature(data=inputs, tensor_type="pt")

    hf_model.processor = processor
    return hf_model


def ovis2_5_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    """Patches and returns an instance of the HfRunner to use for Ovis2."""
    hf_model.model.get_output_embeddings = (
        lambda: hf_model.model.llm.get_output_embeddings()
    )

    def processor(*args, text="", images=None, videos=None, **kwargs):
        if images is None:
            images = []
        else:
            images = [images] if isinstance(images, Image) else images
        if videos is None:
            videos = []
        else:
            videos = [videos] if isinstance(videos, np.ndarray) else videos
            videos = [[PIL.Image.fromarray(frame) for frame in vid] for vid in videos]

        prompt_start_and_end = {
            "qwen2": ("<|im_start|>user\n", "<|im_end|>\n"),
            "llama": ("<|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|>"),
            "gemma2": ("<start_of_turn>user\n", "<end_of_turn>\n"),
        }
        for start, end in prompt_start_and_end.values():
            if start in text and end in text:
                text = text.split(start)[1].split(end)[0]
                break

        images_message = [{"type": "image", "image": img} for img in images]
        videos_message = [{"type": "video", "video": vid} for vid in videos]

        messages = [
            {
                "role": "user",
                "content": [
                    *images_message,
                    *videos_message,
                    {"type": "text", "text": text},
                ],
            }
        ]

        input_ids, pixel_values, grid_thws = hf_model.model.preprocess_inputs(
            messages=messages, enable_thinking=True
        )
        inputs = {
            "inputs": input_ids,
            "pixel_values": pixel_values,
            "grid_thws": grid_thws,
        }
        return BatchFeature(data=inputs, tensor_type="pt")

    hf_model.processor = processor
    return hf_model


def qwen2_5_omni_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    """Patches and returns an instance of the HfRunner for Qwen2.5-Omni."""
    thinker = hf_model.model.thinker
    thinker.get_output_embeddings = lambda: thinker.lm_head
    hf_model.model = thinker
    return hf_model


def qwen3_vl_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    """Patches and returns an instance of the HfRunner to use for GLM4.1V."""
    hf_processor = hf_model.processor

    def processor(*args, videos=None, **kwargs):
        if videos is not None and is_list_of(videos, tuple):
            # batched multi videos
            do_sample_frames = {video[1]["do_sample_frames"] for video in videos}
            assert len(do_sample_frames) == 1
            if kwargs.get("do_sample_frames") is None:
                kwargs["do_sample_frames"] = do_sample_frames
            video_metadata = [
                [
                    VideoMetadata(
                        **{k: v for k, v in video[1].items() if k != "do_sample_frames"}
                    )
                ]
                for video in videos
            ]
            videos = [[video[0]] for video in videos]
        elif videos is not None and isinstance(videos, tuple):
            # single video
            do_sample_frames = videos[1]["do_sample_frames"]
            if kwargs.get("do_sample_frames") is None:
                kwargs["do_sample_frames"] = do_sample_frames
            video_metadata = [
                [
                    VideoMetadata(
                        **{
                            k: v
                            for k, v in videos[1].items()
                            if k != "do_sample_frames"
                        }
                    )
                ]
            ]
            videos = [[videos[0]]]
        else:
            video_metadata = None

        return hf_processor(
            *args, videos=videos, video_metadata=video_metadata, **kwargs
        )

    hf_model.processor = processor
    return hf_model


def tarsier_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    from vllm.model_executor.models.tarsier import get_vision_encoder_info

    vision_encoder_info = get_vision_encoder_info(hf_model.config)

    hf_processor = hf_model.processor
    if hf_processor.patch_size is None:
        hf_processor.patch_size = vision_encoder_info.get_patch_size()

    return hf_model


def voxtral_patch_hf_runner(hf_model: "HfRunner") -> "HfRunner":
    """Patch HfRunner for Voxtral's conversation-based processor.

    Two issues in HfRunner require patching:

    1. VoxtralProcessor requires ``apply_chat_template()`` with conversation
       dicts (accepting ``url``, ``path``, or ``base64`` audio) rather than
       the standard ``processor(text=, audio=, sampling_rate=)`` interface.
    2. HfRunner.get_inputs cannot handle multi-audio per prompt because it
       mis-unpacks ``[(arr1, sr1), (arr2, sr2)]`` via a ``len == 2`` check.

    We override ``get_inputs`` to build conversation dicts and call
    ``apply_chat_template`` directly, bypassing both issues. We also wrap
    ``model.generate`` to strip prompt tokens before decoding, since
    HfRunner.generate calls batch_decode on the full sequence (prompt +
    generated).
    """

    import base64
    import io

    import soundfile as sf

    processor = hf_model.processor

    def _audio_to_base64(audio_array, sample_rate: int) -> str:
        """Encode a numpy audio array as a base64 WAV string."""
        buf = io.BytesIO()
        sf.write(buf, audio_array, int(sample_rate), format="WAV")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def patched_get_inputs(prompts, images=None, videos=None, audios=None, **kwargs):
        all_inputs = []
        for i, prompt in enumerate(prompts):
            content: list[dict] = []

            if audios is not None and audios[i] is not None:
                items = audios[i]
                if not isinstance(items, list):
                    items = [items]
                for item in items:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        arr, sr = item
                    else:
                        arr, sr = item, 16_000
                    content.append(
                        {
                            "type": "audio",
                            "base64": _audio_to_base64(arr, sr),
                        }
                    )

            content.append({"type": "text", "text": prompt})

            inputs = processor.apply_chat_template(
                [{"role": "user", "content": content}]
            )
            if hasattr(inputs, "to"):
                inputs = inputs.to(dtype=hf_model.dtype)
            all_inputs.append(inputs)

        return all_inputs

    _orig_generate = hf_model.model.generate

    def patched_generate(*args, **kwargs):
        """Strip prompt tokens so only generated tokens are decoded."""
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        prompt_len = input_ids.shape[1] if input_ids is not None else 0

        output_ids = _orig_generate(*args, **kwargs)
        if prompt_len:
            output_ids = output_ids[:, prompt_len:]
        return output_ids

    hf_model.get_inputs = patched_get_inputs  # type: ignore[method-assign, assignment]
    hf_model.model.generate = patched_generate  # type: ignore[method-assign]
    return hf_model
