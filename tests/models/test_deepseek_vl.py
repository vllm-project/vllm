from typing import List, Optional, Tuple, Type

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

from vllm.model_executor.models.deepseek_vl import (
    MultiModalityPreTrainedModel, VLMImageProcessor, model_name_to_cls)
from vllm.sequence import SampleLogprobs
from vllm.transformers_utils.config import DeepSeekMultiModalityConfig

from ..conftest import HfRunner, VllmRunner, _ImageAssets
from .utils import check_outputs_equal

models = ["deepseek-ai/deepseek-vl-7b-chat"]
IMAGE_TOKEN_ID = 100015
pytestmark = pytest.mark.vlm

# The image token is placed before "user" on purpose so that the test can pass
HF_IMAGE_PROMPTS = [
    "You are a helpful language and vision assistant. You are able" \
      "to understand the visual content that the user provides, and assist " \
      "the user with a variety of tasks using natural language.\n User:" \
      " <image_placeholder>What's the content of the image?\nAssistant:",
    "You are a helpful language and vision assistant. You are able to "\
    "understand the visual content that the user provides, and assist the "\
    "user with a variety of tasks using natural language.\n User: "\
    "<image_placeholder>What is the season?\nAssistant:",
]


class MultiModalityCausalLM(MultiModalityPreTrainedModel):

    def __init__(self, config: DeepSeekMultiModalityConfig):
        super().__init__(config)

        vision_config = config.vision_config
        vision_cls = model_name_to_cls(vision_config.cls)
        self.vision_model = vision_cls(**vision_config.params)

        aligner_config = config.aligner_config
        aligner_cls = model_name_to_cls(aligner_config.cls)
        self.aligner = aligner_cls(aligner_config.params)

        language_config = config.language_config
        self.language_model = LlamaForCausalLM(language_config)

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        images_seq_mask: torch.LongTensor,
        **kwargs,
    ):
        """

        Args:
            input_ids (torch.LongTensor): [b, T]
            pixel_values (torch.FloatTensor):   [b, n_images, 3, h, w]
            images_seq_mask (torch.BoolTensor): [b, T]

            assert torch.sum(images_seq_mask) == torch.sum(images_emb_mask)

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """

        bs, n = pixel_values.shape[0:2]
        p_b, p_n, p_c, p_h, p_w = pixel_values.shape
        images = pixel_values.reshape(p_b * p_n, p_c, p_h, p_w)
        images_embeds = self.aligner(self.vision_model(images))

        # [b x n, T2, D] -> [b, n x T2, D]
        _, t, d = images_embeds.shape
        images_embeds = images_embeds.reshape(bs, n * t, d)

        # [b, T, D]
        input_ids[input_ids < 0] = 0  # ignore the image embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(
            input_ids).reshape(1, -1, 4096)

        # replace with the image embeddings
        images_embeds = images_embeds.reshape(
            1, -1, self.config.aligner_config.params["n_embed"])
        inputs_embeds[images_seq_mask] = images_embeds

        return inputs_embeds


def vllm_to_hf_output(vllm_output: Tuple[List[int], str,
                                         Optional[SampleLogprobs]],
                      model: str):
    """Sanitize vllm output to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    tokenizer = AutoTokenizer.from_pretrained(model)
    eos_token_id = tokenizer.eos_token_id

    hf_output_ids = [
        token_id for idx, token_id in enumerate(output_ids)
        if token_id != IMAGE_TOKEN_ID or output_ids[idx - 1] != IMAGE_TOKEN_ID
    ]

    assert output_str[0] == " "
    hf_output_str = output_str[1:]
    if hf_output_ids[-1] == eos_token_id:
        hf_output_str = hf_output_str + tokenizer.decode(eos_token_id)

    return hf_output_ids, hf_output_str, out_logprobs


def get_input(tokenizer, prompt, image, dtype):

    image_id = 100015
    prompt = prompt[0]
    image = image[0]
    vl_image = VLMImageProcessor(1024)
    prompt = prompt.replace('<image_placeholder>', '<image_placeholder>' * 576)
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)
    image_token_mask = input_ids == image_id
    images_outputs = vl_image(image, return_tensors="pt")
    images_emb_mask = torch.ones(1, 1, 576) == 1
    prepare = {
        "sft_format":
        prompt,
        "input_ids":
        input_ids,
        "pixel_values":
        images_outputs.pixel_values.to(dtype).reshape(1, -1, 3, 1024, 1024),
        "num_image_tokens":
        576,
        "images_seq_mask":
        image_token_mask.reshape(1, -1),
        "images_emb_mask":
        images_emb_mask,
        "attention_mask":
        torch.ones(1, len(input_ids)),
    }
    return prepare


def run_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    image_assets: _ImageAssets,
    model: str,
    *,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
) -> None:
    """Inference result should be the same between hf and vllm.

    All the image fixtures for the test is under tests/images.
    For huggingface runner, we provide the PIL images as input.
    For vllm runner, we provide MultiModalData objects and corresponding
    vision language config as input.
    Note, the text input is also adjusted to abide by vllm contract.
    The text output is sanitized to be able to compare with hf.
    """
    images = [asset.pil_image for asset in image_assets]

    inputs_per_image = [([prompt], [image])
                        for image, prompt in zip(images, HF_IMAGE_PROMPTS)]

    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).

    # max_model_len should be greater than image_feature_size
    with vllm_runner(model,
                     dtype=dtype,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True) as vllm_model:
        vllm_outputs_per_image = [
            vllm_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                images=images)
            for prompts, images in inputs_per_image
        ]

    AutoModelForCausalLM.register(DeepSeekMultiModalityConfig,
                                  MultiModalityCausalLM)
    tokenizer = AutoTokenizer.from_pretrained(model)
    hf_model = AutoModelForCausalLM.from_pretrained(model,
                                                    trust_remote_code=True)
    dtype_dict = {
        'float16': torch.float16,
        'half': torch.bfloat16,
        'float32': torch.float32,
        'auto': hf_model.dtype
    }
    dtype = dtype_dict.get(dtype, hf_model.dtype)
    hf_model = hf_model.to(dtype)
    hf_model = hf_model
    prepare_input_list = []
    inputs_embeds_list = []
    for prompts, images in inputs_per_image:
        print(f'prompt: {prompts}')
        print(f'images: {images}')
        prepare_input = get_input(tokenizer, prompts, images, dtype)
        prepare_input_list.append(prepare_input)
        inputs_embeds_list.append(
            hf_model.prepare_inputs_embeds(**prepare_input))

    inputs_embeds = torch.concat(inputs_embeds_list)
    attention_mask = torch.concat(
        [x['attention_mask'] for x in prepare_input_list])
    outputs = hf_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
        use_cache=True,
    )
    hf_outputs: List = []

    for o in outputs:
        hf_outputs.append(
            (o, tokenizer.decode(o.cpu().tolist(), skip_special_tokens=True)))

    for hf_output, vllm_output in zip(hf_outputs, vllm_outputs_per_image):
        check_outputs_equal(
            outputs_0_lst=hf_output,
            outputs_1_lst=vllm_output[:2],
            name_0="hf",
            name_1="vllm",
        )
    print('END---->')


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize(
    "size_factors",
    [
        # No image
        [],
        # Single-scale
        [1.0],
        # Single-scale, batched
        [1.0, 1.0, 1.0],
        # Multi-scale
        [0.25, 0.5, 1.0],
    ],
)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(hf_runner, vllm_runner, image_assets, model, size_factors,
                dtype: str, max_tokens: int, num_logprobs: int) -> None:
    run_test(
        hf_runner,
        vllm_runner,
        image_assets,
        model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
    )
