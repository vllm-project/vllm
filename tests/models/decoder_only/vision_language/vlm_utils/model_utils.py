"""Common utility functions relating to different models that are useful
for manipulating the input / output of HF & vLLM test runners, which are
typically specific to a small subset of models.
"""
import re
import types
from pathlib import PosixPath
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from PIL.Image import Image
from transformers import (AutoConfig, AutoTokenizer, BatchEncoding,
                          GenerationConfig)

from vllm.sequence import SampleLogprobs
from vllm.transformers_utils.tokenizer import patch_padding_side
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE

from .....conftest import (HfRunner, ImageAsset, PromptAudioInput,
                           PromptImageInput, PromptVideoInput, _ImageAssets)
from ....utils import TokensTextLogprobs
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
        model: str) -> Tuple[List[int], str, Optional[SampleLogprobs]]:
    """Sanitize vllm output [qwen models] to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    hf_output_str = output_str + "<|endoftext|>"

    return output_ids, hf_output_str, out_logprobs


def qwen2_vllm_to_hf_output(
        vllm_output: RunnerOutput,
        model: str) -> Tuple[List[int], str, Optional[SampleLogprobs]]:
    """Sanitize vllm output [qwen2 models] to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    hf_output_str = output_str + "<|im_end|>"

    return output_ids, hf_output_str, out_logprobs


def llava_image_vllm_to_hf_output(vllm_output: RunnerOutput,
                                  model: str) -> RunnerOutput:
    config = AutoConfig.from_pretrained(model)
    mm_token_id = config.image_token_index
    return _llava_vllm_to_hf_output(vllm_output, model, mm_token_id)


def llava_video_vllm_to_hf_output(
        vllm_output: RunnerOutput,
        model: str) -> Tuple[List[int], str, Optional[SampleLogprobs]]:
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


def minicpmv_trunc_hf_output(hf_output: RunnerOutput,
                             model: str) -> RunnerOutput:
    output_ids, output_str, out_logprobs = hf_output
    if output_str.endswith("<|eot_id|>"):
        output_str = output_str.split("<|eot_id|>")[0]
    return output_ids, output_str, out_logprobs


####### Functions for converting image assets to embeddings
def get_llava_embeddings(image_assets: _ImageAssets):
    return [asset.image_embeds for asset in image_assets]


####### postprocessors to run on HF BatchEncoding
def cast_dtype_post_processor(
        hf_inp_key: str) -> Callable[[BatchEncoding, str], BatchEncoding]:
    """Gets a handle to a post processor which converts a given key into a
    target data type."""

    def process(hf_inputs: BatchEncoding, dtype: str):
        torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[dtype]
        hf_inputs[hf_inp_key] = hf_inputs[hf_inp_key].to(torch_dtype)
        return hf_inputs

    return process


def ignore_inputs_post_processor(
        hf_inp_key: str) -> Callable[[BatchEncoding, str], BatchEncoding]:
    """Gets a handle to a post processor which ignores a given key."""

    def process(hf_inputs: BatchEncoding, dtype: str):
        del hf_inputs[hf_inp_key]
        return hf_inputs

    return process


def wrap_inputs_post_processor(hf_inputs: BatchEncoding, dtype: str):
    return {"model_inputs": hf_inputs}


def molmo_post_processor(hf_inputs: BatchEncoding, dtype: str):
    hf_inputs = cast_dtype_post_processor("images")(hf_inputs, dtype)
    return {k: v.unsqueeze(0) for k, v in hf_inputs.items()}


####### Prompt path encoders for models that need models on disk
def qwen_prompt_path_encoder(
        tmp_path: PosixPath, prompt: str, assets: Union[List[ImageAsset],
                                                        _ImageAssets]) -> str:
    """Given a temporary dir path, export one or more image assets into the
    tempdir & replace its contents with the local path to the string so that
    the HF version of Qwen-VL can resolve the path and load the image in its
    forward() call.

    Args:
        tmp_path: Tempdir for test under consideration.
        prompt: Prompt with image placeholders.
        assets: List of image assets whose len equals the num placeholders.
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
        inputs = BatchEncoding(data=inputs, tensor_type="pt")
        return inputs

    hf_model.processor = processor
    hf_model.model.get_output_embeddings = lambda: \
        hf_model.model.language.model.embed_tokens
    return hf_model


def glm_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    """Patches and returns an instance of the HfRunner to use for GLM4."""
    hf_processor = hf_model.processor
    patch_padding_side(hf_processor)

    def processor(*args, text="", images=None, **kwargs):
        if images is None:
            return hf_processor(*args, **kwargs)

        return hf_processor.apply_chat_template(
            [{
                "role": "user",
                "image": images,
                "content": text
            }],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            **kwargs,
        )

    hf_model.processor = processor
    hf_model.model.get_output_embeddings = lambda: \
        hf_model.model.transformer.output_layer
    return hf_model


def h2ovl_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    """Patches and returns an instance of the HfRunner to use for H2OVL."""

    class H2OVLProcessor:
        """A simple processor for H2OVL models."""

        def __init__(self, hf_runner: HfRunner):
            self.num_image_token = hf_runner.model.num_image_token
            self.tokenizer = hf_runner.tokenizer
            self.dtype = hf_runner.model.dtype

            self.config = AutoConfig.from_pretrained(hf_runner.model_name,
                                                     trust_remote_code=True)
            self.vision_config = self.config.vision_config
            self.use_thumbnail = self.config.use_thumbnail
            self.min_num = self.config.min_dynamic_patch
            self.max_num = self.config.max_dynamic_patch
            self.image_size = self.vision_config.image_size

        def __call__(self, text: str, images: Union[Image, List[Image]],
                     **kwargs):
            # yapf: disable
            from vllm.model_executor.models.h2ovl import (
                IMG_CONTEXT, IMG_END, IMG_START, image_to_pixel_values)

            # yapf: enable
            images = [images] if isinstance(images, Image) else images
            pixel_values = [
                image_to_pixel_values(image,
                                      self.image_size,
                                      self.min_num,
                                      self.max_num,
                                      self.use_thumbnail,
                                      use_MSAC=self.config.use_msac).to(
                                          self.dtype) for image in images
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


def internvl_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    """Patches and returns an instance of the HfRunner to use for InternVL."""

    class InternVLProcessor:
        """A simple processor for InternVL2 which misses a processor."""

        def __init__(self, hf_runner: HfRunner):
            self.num_image_token = hf_runner.model.num_image_token
            self.tokenizer = hf_runner.tokenizer
            self.dtype = hf_runner.model.dtype

            self.config = AutoConfig.from_pretrained(hf_runner.model_name,
                                                     trust_remote_code=True)
            self.vision_config = self.config.vision_config
            self.use_thumbnail = self.config.use_thumbnail
            self.min_num = self.config.min_dynamic_patch
            self.max_num = self.config.max_dynamic_patch
            self.image_size = self.vision_config.image_size

        def __call__(self, text: str, images: Union[Image, List[Image]],
                     **kwargs):
            from vllm.model_executor.models.internvl import (
                IMG_CONTEXT, IMG_END, IMG_START, image_to_pixel_values)
            images = [images] if isinstance(images, Image) else images
            pixel_values = [
                image_to_pixel_values(image, self.image_size, self.min_num,
                                      self.max_num,
                                      self.use_thumbnail).to(self.dtype)
                for image in images
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
    vit_embeds = self.extract_feature(pixel_values)
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


def _generate_greedy_logprobs_limit(
    self,
    prompts: List[str],
    max_tokens: int,
    num_logprobs: int,
    images: Optional[PromptImageInput] = None,
    audios: Optional[PromptAudioInput] = None,
    videos: Optional[PromptVideoInput] = None,
    **kwargs: Any,
) -> List[TokensTextLogprobs]:
    all_inputs = self.get_inputs(prompts,
                                 images=images,
                                 videos=videos,
                                 audios=audios)

    # Process in batches for inference.
    if len(all_inputs):
        input_ids_lst = []
        images_lst = []
        images_input_idx_lst = []
        imges_masks_lst = []
        for inputs in all_inputs:
            input_ids_lst.append(inputs["input_ids"])
            images_lst.append(inputs["images"])
            images_input_idx_lst.append(inputs["image_input_idx"])
            imges_masks_lst.append(inputs["image_masks"])
        batch_inputs = {}
        batch_inputs['input_ids'] = torch.cat(input_ids_lst, dim=0)
        batch_inputs['images'] = torch.cat(images_lst, dim=0)
        batch_inputs['image_input_idx'] = torch.cat(images_input_idx_lst,
                                                    dim=0)
        batch_inputs['image_masks'] = torch.cat(imges_masks_lst, dim=0)

        outputs = self.model.generate_from_batch(
            batch=self.wrap_device(batch_inputs,
                                   device=self.model.device.type),
            generation_config=GenerationConfig(
                max_new_tokens=max_tokens,
                stop_strings="<|endoftext|>",
                do_sample=False,
            ),
            tokenizer=self.tokenizer,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    all_logprobs: List[List[Dict[int, float]]] = []
    all_output_ids: List[List[int]] = []
    all_output_strs: List[str] = []

    for index in range(len(all_inputs)):
        (
            seq_logprobs_lst,
            output_len,
        ) = self._hidden_states_to_logprobs(outputs.hidden_states,
                                            num_logprobs)
        all_logprobs.append(seq_logprobs_lst)
        seq_ids = outputs.sequences[index]
        output_ids = seq_ids[-output_len:]
        all_output_ids.append(output_ids.tolist())
        all_output_strs.append(self.tokenizer.decode(output_ids))
    outputs = zip(all_output_ids, all_output_strs, all_logprobs)
    return [(output_ids, output_str, output_logprobs)
            for output_ids, output_str, output_logprobs in outputs]


####### Molmo-specific HuggingFace runner patchers
def mlomo_patch_hf_runner(hf_model: HfRunner) -> HfRunner:
    """Patches and returns an instance of the HfRunner to use for Molmo."""
    hf_processor = hf_model.processor

    def _processor(*args, **kwargs):
        return hf_processor.process(*args, **kwargs)

    hf_model.processor = _processor

    setattr(  # noqa: B010
        hf_model,
        "generate_greedy_logprobs_limit",
        types.MethodType(_generate_greedy_logprobs_limit, hf_model),
    )

    return hf_model
