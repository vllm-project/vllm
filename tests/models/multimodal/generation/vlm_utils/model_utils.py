# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Common utility functions relating to different models that are useful
for manipulating the input / output of HF & vLLM test runners, which are
typically specific to a small subset of models.
"""
import types
from pathlib import PosixPath
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import pytest
import regex as re
import torch
from PIL.Image import Image
from transformers import (AutoConfig, AutoTokenizer, BatchFeature,
                          GenerationConfig, GenerationMixin)
from transformers.video_utils import VideoMetadata

from vllm.sequence import SampleLogprobs
from vllm.transformers_utils.tokenizer import patch_padding_side
from vllm.utils import is_list_of

from .....conftest import HfRunner, ImageAsset, ImageTestAssets
from .types import RunnerOutput


####### vLLM output processors functions
def blip2_vllm_to_hf_output(vllm_output: RunnerOutput,
                            model: str) -> RunnerOutput:
    """Sanitize vllm output [blip2 models] to be comparable with hf output."""
    _, output_str, out_logprobs = vllm_output

    hf_output_str = output_str + "\n"

    tokenizer = AutoTokenizer.from_pretrained(model)
    hf_output_ids = tokenizer.encode(hf_output_str)
    assert hf_output_ids[0] == tokenizer.bos_token_id
    hf_output_ids = hf_output_ids[1:]

    return hf_output_ids, hf_output_str, out_logprobs


def fuyu_vllm_to_hf_output(vllm_output: RunnerOutput,
                           model: str) -> RunnerOutput:
    """Sanitize vllm output [fuyu models] to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    hf_output_str = output_str.lstrip() + "|ENDOFTEXT|"

    return output_ids, hf_output_str, out_logprobs


def qwen_vllm_to_hf_output(
        vllm_output: RunnerOutput,
        model: str) -> tuple[list[int], str, Optional[SampleLogprobs]]:
    """Sanitize vllm output [qwen models] to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    hf_output_str = output_str + "<|endoftext|>"

    return output_ids, hf_output_str, out_logprobs


def qwen2_vllm_to_hf_output(
        vllm_output: RunnerOutput,
        model: str) -> tuple[list[int], str, Optional[SampleLogprobs]]:
    """Sanitize vllm output [qwen2 models] to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    hf_output_str = output_str + "<|im_end|>"

    return output_ids, hf_output_str, out_logprobs


def kimiv_vl_vllm_to_hf_output(
        vllm_output: RunnerOutput,
        model: str) -> tuple[list[int], str, Optional[SampleLogprobs]]:
    """Sanitize vllm output [kimi_vl models] to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    hf_output_str = output_str + "<|im_end|>[EOS]"

    return output_ids, hf_output_str, out_logprobs


def llava_image_vllm_to_hf_output(vllm_output: RunnerOutput,
                                  model: str) -> RunnerOutput:
    config = AutoConfig.from_pretrained(model)
    mm_token_id = config.image_token_index
    return _llava_vllm_to_hf_output(vllm_output, model, mm_token_id)


def llava_video_vllm_to_hf_output(
        vllm_output: RunnerOutput,
        model: str) -> tuple[list[int], str, Optional[SampleLogprobs]]:
    config = AutoConfig.from_pretrained(model)
    mm_token_id = config.video_token_index
    return _llava_vllm_to_hf_output(vllm_output, model, mm_token_id)


def _llava_vllm_to_hf_output(vllm_output: RunnerOutput, model: str,
                             mm_token_id: int) -> RunnerOutput:
    """Sanitize vllm output [Llava models] to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    tokenizer = AutoTokenizer.from_pretrained(model)
    eos_token_id = tokenizer.eos_token_id

    hf_output_ids = [
        token_id for idx, token_id in enumerate(output_ids)
        if token_id != mm_token_id or output_ids[idx - 1] != mm_token_id
    ]

    assert output_str[0] == " "
    hf_output_str = output_str[1:]
    if hf_output_ids[-1] == eos_token_id:
        hf_output_str = hf_output_str + tokenizer.decode(eos_token_id)

    return hf_output_ids, hf_output_str, out_logprobs


def llava_onevision_hf_model_kwargs(model: str) -> dict:
    """Workaround to fix the sliding window issue in llava_onevision."""
    config = AutoConfig.from_pretrained(model)
    config.text_config.sliding_window = None
    return config.to_dict()


def llava_onevision_vllm_to_hf_output(vllm_output: RunnerOutput,
                                      model: str) -> RunnerOutput:
    """Sanitize vllm output [llava-onevision] to compare with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    config = AutoConfig.from_pretrained(model)
    video_token_id = config.video_token_index

    tokenizer = AutoTokenizer.from_pretrained(model)
    eos_token_id = tokenizer.eos_token_id

    hf_output_ids = [
        token_id for idx, token_id in enumerate(output_ids)
        if token_id != video_token_id or output_ids[idx - 1] != video_token_id
    ]

    hf_output_str = output_str
    if hf_output_ids[-1] == eos_token_id:
        hf_output_str = hf_output_str + tokenizer.decode(eos_token_id)

    return hf_output_ids, hf_output_str, out_logprobs


def mantis_vllm_to_hf_output(vllm_output: RunnerOutput,
                             model: str) -> RunnerOutput:
    """Sanitize vllm output [mantis] to compare with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    hf_output_str = output_str + "<|eot_id|>"

    return output_ids, hf_output_str, out_logprobs


def phi3v_vllm_to_hf_output(vllm_output: RunnerOutput,
                            model: str) -> RunnerOutput:
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


def paligemma_vllm_to_hf_output(vllm_output: RunnerOutput,
                                model: str) -> RunnerOutput:
    """Sanitize vllm output to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    config = AutoConfig.from_pretrained(model)
    image_token_id = config.image_token_index

    tokenizer = AutoTokenizer.from_pretrained(model)
    eos_token_id = tokenizer.eos_token_id

    hf_output_ids = [
        token_id for idx, token_id in enumerate(output_ids)
        if token_id != image_token_id or output_ids[idx - 1] != image_token_id
    ]

    hf_output_str = output_str

    if hf_output_ids[-1] == eos_token_id:
        hf_output_str = hf_output_str + tokenizer.decode(eos_token_id)

    return hf_output_ids, hf_output_str, out_logprobs


####### Post-processors for HF outputs
def deepseekvl2_trunc_hf_output(hf_output: RunnerOutput,
                                model: str) -> RunnerOutput:
    output_ids, output_str, out_logprobs = hf_output
    if output_str.endswith("<｜end▁of▁sentence｜>"):
        output_str = output_str.split("<｜end▁of▁sentence｜>")[0]
    return output_ids, output_str, out_logprobs


def idefics3_trunc_hf_output(hf_output: RunnerOutput,
                             model: str) -> RunnerOutput:
    output_ids, output_str, out_logprobs = hf_output
    if output_str.endswith("<end_of_utterance>"):
        output_str = output_str.split("<end_of_utterance>")[0]
    return output_ids, output_str, out_logprobs


def smolvlm_trunc_hf_output(hf_output: RunnerOutput,
                            model: str) -> RunnerOutput:
    # Based on Idefics3
    return idefics3_trunc_hf_output(hf_output, model)


def minicpmv_trunc_hf_output(hf_output: RunnerOutput,
                             model: str) -> RunnerOutput:
    output_ids, output_str, out_logprobs = hf_output
    if output_str.endswith("<|eot_id|>"):
        output_str = output_str.split("<|eot_id|>")[0]
    return output_ids, output_str, out_logprobs


def minimax_vl_01_hf_output(hf_output: RunnerOutput,
                            model: str) -> RunnerOutput:
    output_ids, output_str, out_logprobs = hf_output
    if output_str.endswith("<end_of_sentence>"):
        output_str = output_str.split("<end_of_sentence>")[0]
    return output_ids, output_str, out_logprobs


def ultravox_trunc_hf_output(hf_output: RunnerOutput,
                             model: str) -> RunnerOutput:
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
        tmp_path: PosixPath, prompt: str,
        assets: Union[list[ImageAsset], ImageTestAssets]) -> str:
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
    hf_model.model.get_output_embeddings = lambda: \
        hf_model.model.language.model.embed_tokens
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


def glm4v_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    """Patches and returns an instance of the HfRunner to use for GLM4V."""
    hf_processor = hf_model.processor
    patch_padding_side(hf_processor)

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
            [{
                "role": "user",
                "image": image,
                "content": content
            } for image, content in zip(images, contents)],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            **kwargs,
        )

    hf_model.processor = processor
    hf_model.model.get_output_embeddings = lambda: \
        hf_model.model.transformer.output_layer
    return hf_model


def glm4_1v_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    """Patches and returns an instance of the HfRunner to use for GLM4.1V."""
    hf_processor = hf_model.processor

    def processor(*args, videos=None, **kwargs):
        if videos is not None and is_list_of(videos, tuple):
            # If videos is a list of tuples, we assume each tuple contains
            # (video_array, metadata) as in the case of GLM4.1V.
            video_metadata = [[VideoMetadata(**video[1])] for video in videos]
            videos = [[video[0]] for video in videos]
        else:
            video_metadata = None

        return hf_processor(*args,
                            videos=videos,
                            video_metadata=video_metadata,
                            **kwargs)

    hf_model.processor = processor
    return hf_model


def h2ovl_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    """Patches and returns an instance of the HfRunner to use for H2OVL."""

    class H2OVLProcessor:
        """A simple processor for H2OVL models."""

        def __init__(self, hf_runner: HfRunner):
            self.num_image_token = hf_runner.model.num_image_token
            self.tokenizer = hf_runner.tokenizer

            self.config = AutoConfig.from_pretrained(hf_runner.model_name,
                                                     trust_remote_code=True)
            self.vision_config = self.config.vision_config
            self.use_thumbnail = self.config.use_thumbnail
            self.use_msac = self.config.use_msac
            self.min_num = self.config.min_dynamic_patch
            self.max_num = self.config.max_dynamic_patch
            self.image_size = self.vision_config.image_size

        def __call__(self, text: str, images: Union[Image, list[Image]],
                     **kwargs):
            # yapf: disable
            from vllm.model_executor.models.h2ovl import (
                IMG_CONTEXT, IMG_END, IMG_START, image_to_pixel_values_h2ovl)

            # yapf: enable
            images = [images] if isinstance(images, Image) else images
            pixel_values = [
                image_to_pixel_values_h2ovl(
                    image,
                    input_size=self.image_size,
                    min_num=self.min_num,
                    max_num=self.max_num,
                    use_thumbnail=self.use_thumbnail,
                    use_msac=self.use_msac,
                ) for image in images
            ]
            num_patches_list = [
                pixel_value.shape[0] for pixel_value in pixel_values
            ]
            pixel_values = torch.cat(pixel_values, dim=0)
            for num_patches in num_patches_list:
                context_tokens = IMG_CONTEXT * self.num_image_token \
                    * num_patches
                image_tokens = IMG_START + context_tokens + IMG_END
                text = text.replace('<image>', image_tokens, 1)
            prompt = self.tokenizer(text, return_tensors="pt")
            prompt.update({"pixel_values": pixel_values})
            return prompt

    img_context_token_id = hf_model.tokenizer.convert_tokens_to_ids(
        "<IMG_CONTEXT>")
    hf_model.model.img_context_token_id = img_context_token_id
    hf_model.processor = H2OVLProcessor(hf_model)
    hf_model.model.get_output_embeddings = lambda: \
        hf_model.model.language_model.get_output_embeddings()
    hf_model.model.generate = types.MethodType(_internvl_generate,
                                               hf_model.model)
    return hf_model


def skyworkr1v_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    """Patches and returns an instance of the HfRunner to use for SkyworkR1V."""

    class SkyworkR1VProcessor:
        """A simple processor for SkyworkR1V."""

        def __init__(self, hf_runner: HfRunner):
            self.num_image_token = hf_runner.model.num_image_token
            self.tokenizer = hf_runner.tokenizer

            self.config = AutoConfig.from_pretrained(hf_runner.model_name,
                                                     trust_remote_code=True)
            self.vision_config = self.config.vision_config
            self.use_thumbnail = self.config.use_thumbnail
            self.min_num = self.config.min_dynamic_patch
            self.max_num = self.config.max_dynamic_patch
            self.image_size = self.vision_config.image_size

        def __call__(self, text: str, images: Union[Image, list[Image]],
                     **kwargs):
            from vllm.model_executor.models.skyworkr1v import (
                IMG_CONTEXT, IMG_END, IMG_START,
                image_to_pixel_values_skyworkr1v)
            images = [images] if isinstance(images, Image) else images
            pixel_values = [
                image_to_pixel_values_skyworkr1v(
                    image,
                    input_size=self.image_size,
                    min_num=self.min_num,
                    max_num=self.max_num,
                    use_thumbnail=self.use_thumbnail,
                ) for image in images
            ]
            num_patches_list = [
                pixel_value.shape[0] for pixel_value in pixel_values
            ]
            pixel_values = torch.cat(pixel_values, dim=0)
            for num_patches in num_patches_list:
                context_tokens = IMG_CONTEXT * self.num_image_token \
                    * num_patches
                image_tokens = IMG_START + context_tokens + IMG_END
                text = text.replace('<image>', image_tokens, 1)
            prompt = self.tokenizer(text, return_tensors="pt")
            prompt.update({"pixel_values": pixel_values})
            return prompt

    img_context_token_id = hf_model.tokenizer.convert_tokens_to_ids(
        "<IMG_CONTEXT>")
    hf_model.model.img_context_token_id = img_context_token_id
    hf_model.processor = SkyworkR1VProcessor(hf_model)
    hf_model.model.get_output_embeddings = lambda: \
        hf_model.model.language_model.get_output_embeddings()
    hf_model.model.generate = types.MethodType(_internvl_generate,
                                               hf_model.model)
    return hf_model


def internvl_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    """Patches and returns an instance of the HfRunner to use for InternVL."""

    class InternVLProcessor:
        """A simple processor for InternVL2 which misses a processor."""

        def __init__(self, hf_runner: HfRunner):
            self.num_image_token = hf_runner.model.num_image_token
            self.tokenizer = hf_runner.tokenizer

            self.config = AutoConfig.from_pretrained(hf_runner.model_name,
                                                     trust_remote_code=True)
            self.vision_config = self.config.vision_config
            self.use_thumbnail = self.config.use_thumbnail
            self.min_num = self.config.min_dynamic_patch
            self.max_num = self.config.max_dynamic_patch
            self.image_size = self.vision_config.image_size

        def __call__(
            self,
            text: str,
            images: Union[Image, list[Image]] = None,
            videos: Union[npt.NDArray, list[npt.NDArray]] = None,
            **kwargs,
        ):
            from vllm.model_executor.models.internvl import (
                IMG_CONTEXT, IMG_END, IMG_START,
                image_to_pixel_values_internvl, video_to_pixel_values_internvl)
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
                    ) for image in images
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
                    ) for video in videos
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
                if image_index == -1 or (video_index > -1
                                         and video_index < image_index):
                    num_patches = num_patches_videos.pop(0)
                    pixel_values.append(pixel_values_videos.pop(0))
                    context_tokens = IMG_START + \
                        IMG_CONTEXT * self.num_image_token + IMG_END
                    video_tokens = ''.join([
                        f'Frame{i+1}: {context_tokens}'
                        for i in range(num_patches)
                    ])
                    text = text.replace('<video>', video_tokens, 1)
                else:
                    num_patches = num_patches_images.pop(0)
                    pixel_values.append(pixel_values_images.pop(0))
                    context_tokens = IMG_CONTEXT * self.num_image_token \
                        * num_patches
                    image_tokens = IMG_START + context_tokens + IMG_END
                    text = text.replace('<image>', image_tokens, 1)
            pixel_values = torch.cat(pixel_values, dim=0)

            prompt = self.tokenizer(text, return_tensors="pt")
            prompt.update({"pixel_values": pixel_values})
            return prompt

    img_context_token_id = hf_model.tokenizer.convert_tokens_to_ids(
        "<IMG_CONTEXT>")
    hf_model.model.img_context_token_id = img_context_token_id
    hf_model.processor = InternVLProcessor(hf_model)
    hf_model.model.get_output_embeddings = lambda: \
        hf_model.model.language_model.get_output_embeddings()
    hf_model.model.generate = types.MethodType(_internvl_generate,
                                               hf_model.model)
    return hf_model


def _internvl_generate(
    self,
    pixel_values: torch.FloatTensor,
    input_ids: torch.FloatTensor,
    attention_mask: Optional[torch.LongTensor] = None,
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
    selected = (input_ids == self.img_context_token_id)
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
    hf_model.model.get_output_embeddings = lambda: \
        hf_model.model.llm.get_output_embeddings()

    def processor(*args, text="", images=None, **kwargs):
        text_tokenizer = hf_model.model.get_text_tokenizer()
        images = [images] if isinstance(images, Image) else images

        prompt_start_and_end = {
            "qwen2": ("<|im_start|>user\n", "<|im_end|>\n"),
            "llama":
            ("<|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|>"),
            "gemma2": ("<start_of_turn>user\n", "<end_of_turn>\n"),
        }
        for start, end in prompt_start_and_end.values():
            if start in text and end in text:
                text = text.split(start)[1].split(end)[0]
                break

        prompt, input_ids, pixel_values = hf_model.model.preprocess_inputs(
            text_or_conversations=text, images=images)
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)

        inputs = {
            "inputs": input_ids.unsqueeze(0),
            "pixel_values": pixel_values.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
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


def tarsier_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    from vllm.model_executor.models.tarsier import get_vision_encoder_info

    vision_encoder_info = get_vision_encoder_info(hf_model.config)

    hf_processor = hf_model.processor
    if hf_processor.patch_size is None:
        hf_processor.patch_size = vision_encoder_info.get_patch_size()

    return hf_model
