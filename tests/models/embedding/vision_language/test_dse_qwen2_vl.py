from functools import partial
from typing import Callable, Dict, List, Type

import pytest
import torch
from PIL import Image
from transformers import BatchEncoding, Qwen2VLForConditionalGeneration

from ....conftest import IMAGE_ASSETS, HfRunner, PromptImageInput, VllmRunner
from ....utils import large_gpu_test
from ..utils import check_embeddings_close

HF_TEXT_PROMPTS = [
    # T -> X
    (
        "Query: Find me an everyday image that matches the given caption: The label of the object is stop sign",  # noqa: E501,
        Image.new("RGB", (56, 56))),
    # T -> X
    ("Query: Retrieve an image of this caption: cherry blossom",
     Image.new("RGB", (56, 56))),
]

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    "What is shown in this image?",
    "cherry_blossom":
    "What is shown in this image?"
})

MODELS = ["MrLight/dse-qwen2-2b-mrl-v1"]


def get_messages(image: Image.Image, text: str, embed_text: bool):
    # assert False, 'remember to use outer [] as required'
    if embed_text:
        messages = [{
            "role":
            "user",
            "content": [
                {
                    "type": "image",
                    "image": Image.new("RGB", (56, 56)),
                    "resized_height": 1,
                    "resized_width": 1
                },  # need a dummy image here for an easier process.
                {
                    "type": "text",
                    "text": text
                },
            ]
        }]
    else:
        messages = [{
            "role":
            "user",
            "content": [{
                "type": "image",
                "image": image
            }, {
                "type": "text",
                "text": text
            }]
        }]
    return messages


def apply_chat_template_and_add_eos(
    messages: List[Dict],
    apply_chat_template_fn: Callable,
):
    prompt = apply_chat_template_fn(
        messages, tokenize=False, add_generation_prompt=True) + "<|endoftext|>"
    return prompt


def postprocess_inputs(hf_model: HfRunner, inputs: BatchEncoding, **kwargs):
    return hf_model.model.prepare_inputs_for_generation(**inputs, **kwargs)


def _run_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    input_texts: List[str],
    input_images: PromptImageInput,
    embed_texts: List[bool],
    model: str,
    *,
    dtype: str,
) -> None:
    '''SET PYTHONPATH'''
    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).
    with vllm_runner(model,
                     task="embedding",
                     dtype=dtype,
                     enforce_eager=True,
                     max_model_len=8192) as vllm_model:
        tokenizer = vllm_model.model.get_tokenizer()
        texts = [
            # this is necessary because vllm_model.encode will not apply any
            # templating to the prompt, and therefore lacks an image_pad
            # token unless one is inserted beforehand (the (28,28) image
            # above is converted to an image pad token by the chat template).
            apply_chat_template_and_add_eos(
                get_messages(image, text, False),
                apply_chat_template_fn=tokenizer.apply_chat_template,
            ) for text, image in zip(input_texts, input_images)
            # vllm will replace the pad token with the actual image,
            # which may be a placeholder image, later.
        ]
        vllm_outputs = vllm_model.encode(texts, images=input_images)

    hf_outputs = []
    with hf_runner(model,
                   dtype=dtype,
                   auto_cls=Qwen2VLForConditionalGeneration) as hf_model:
        hf_model.postprocess_inputs = partial(
            postprocess_inputs,
            hf_model,
            cache_position=torch.arange(
                0,
                1,  # 1 for batch size
                requires_grad=False),
            use_cache=False)
        for text, image, embed_text in zip(input_texts, input_images,
                                           embed_texts):
            # dse requires non-standard input processing
            # because it needs an image_pad token
            messages = get_messages(image, text, embed_text)
            prompt = apply_chat_template_and_add_eos(
                messages, hf_model.processor.apply_chat_template)
            inputs = hf_model.get_inputs(
                prompts=[[prompt]],
                images=[[image]],
            )
            with torch.no_grad():
                outputs = hf_model.model(
                    **hf_model.wrap_device(inputs[0],
                                           device=hf_model.model.device.type),
                    return_dict=True,
                    output_hidden_states=True,
                )
                pooled_output = torch.nn.functional.normalize(
                    outputs.hidden_states[-1][0, -1], p=2, dim=-1)
            hf_outputs.append(pooled_output.tolist())

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_models_text(
    hf_runner,
    vllm_runner,
    image_assets,
    model: str,
    dtype: str,
) -> None:
    input_texts_images = [(text, image_placeholder)
                          for text, image_placeholder in HF_TEXT_PROMPTS]
    input_texts = [text for text, _ in input_texts_images]
    input_images = [image for _, image in input_texts_images]
    embed_texts = [True] * len(input_texts)

    _run_test(
        hf_runner,
        vllm_runner,
        input_texts,
        input_images,  # type: ignore
        embed_texts,
        model,
        dtype=dtype,
    )


@large_gpu_test(min_gb=48)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_models_image(
    hf_runner,
    vllm_runner,
    image_assets,
    model: str,
    dtype: str,
) -> None:
    input_texts_images = [
        (text, asset.pil_image)
        for text, asset in zip(HF_IMAGE_PROMPTS, image_assets)
    ]
    input_texts = [text for text, _ in input_texts_images]
    input_images = [image for _, image in input_texts_images]
    embed_texts = [False] * len(input_texts)

    _run_test(
        hf_runner,
        vllm_runner,
        input_texts,
        input_images,
        embed_texts,
        model,
        dtype=dtype,
    )
