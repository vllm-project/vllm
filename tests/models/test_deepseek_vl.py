from dataclasses import dataclass
from typing import List, Optional, Tuple, Type

import pytest
import torch
from transformers import (AutoModelForVision2Seq, AutoTokenizer,
                          LlamaForCausalLM, LlamaTokenizerFast)
from transformers.processing_utils import ProcessorMixin

from vllm.model_executor.models.deepseek_vl import (
    MultiModalityPreTrainedModel, VLMImageProcessor, model_name_to_cls)
from vllm.multimodal.utils import rescale_image_size
from vllm.sequence import SampleLogprobs
from vllm.transformers_utils.config import DeepSeekMultiModalityConfig

from ..conftest import HfRunner, VllmRunner, _ImageAssets
from .utils import check_logprobs_close

models = ["deepseek-ai/deepseek-vl-1.3b-chat"]
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


class DictOutput(object):

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


@dataclass
class VLChatProcessorOutput(DictOutput):
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    attention_mask: torch.Tensor
    images_seq_mask: torch.BoolTensor

    def __len__(self):
        return len(self.input_ids)

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.images_seq_mask = self.images_seq_mask.to(device)
        self.pixel_values = self.pixel_values.to(device=device)
        return self


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
        pixel_values = pixel_values.to(self.dtype)
        images = pixel_values.reshape(p_b * p_n, p_c, p_h, p_w)
        images_embeds = self.aligner(self.vision_model(images))

        # [b x n, T2, D] -> [b, n x T2, D]
        _, t, d = images_embeds.shape
        images_embeds = images_embeds.reshape(bs, n * t, d)

        # [b, T, D]
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        inputs_embeds[images_seq_mask] = images_embeds

        return inputs_embeds

    def generate(self, *args, **kwargs):

        pixel_values = kwargs.pop('pixel_values')
        images_seq_mask = kwargs.pop('images_seq_mask')
        input_ids = kwargs.pop('input_ids')
        inputs_embeds = self.prepare_inputs_embeds(input_ids, pixel_values,
                                                   images_seq_mask)
        output = self.language_model.generate(
            *args,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs)
        return output

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()


class VLChatProcessor(ProcessorMixin):
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")

    attributes = ["image_processor", "tokenizer"]

    system_prompt = (
        "You are a helpful language and vision assistant. "
        "You are able to understand the visual content that the user provides, "
        "and assist the user with a variety of tasks using natural language.")

    def __init__(
        self,
        image_processor: VLMImageProcessor,
        tokenizer: LlamaTokenizerFast,
        image_tag: str = "<image_placeholder>",
        num_image_tokens: int = 576,
        add_special_token: bool = False,
        sft_format: str = "deepseek",
        mask_prompt: bool = True,
        ignore_id: int = -100,
        **kwargs,
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        image_id = self.tokenizer.vocab.get(image_tag)
        if image_id is None:
            special_tokens = [image_tag]
            special_tokens_dict = {"additional_special_tokens": special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            print(f"Add image tag = {image_tag} to the tokenizer")

        self.image_tag = image_tag
        self.num_image_tokens = num_image_tokens
        self.add_special_token = add_special_token
        self.sft_format = sft_format
        self.mask_prompt = mask_prompt
        self.ignore_id = ignore_id
        self.image_id = image_id

        super().__init__(
            image_processor,
            tokenizer,
            image_tag,
            num_image_tokens,
            add_special_token,
            sft_format,
            mask_prompt,
            ignore_id,
            **kwargs,
        )

    def __call__(self, *arg, **kwargs):
        prompt = kwargs.pop('text')
        image = kwargs.pop('images')
        return VLChatProcessorOutput(**self.get_input(prompt, image))

    def get_input(self, prompt, image):
        prompt = prompt.replace(self.image_tag,
                                self.image_tag * self.num_image_tokens)
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)
        image_token_mask = input_ids == self.image_id
        images_outputs = self.image_processor(image, return_tensors="pt")
        image_size = self.image_processor.image_size
        prepare = {
            "input_ids":
            input_ids.reshape(1, -1),
            "pixel_values":
            images_outputs.pixel_values.reshape(1, -1, 3, image_size,
                                                image_size),
            "images_seq_mask":
            image_token_mask.reshape(1, -1),
            "attention_mask":
            torch.ones(1, len(input_ids)),
        }
        return prepare


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


def run_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    image_assets: _ImageAssets,
    model: str,
    *,
    size_factors: List[float],
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

    inputs_per_image = [(
        [prompt for _ in size_factors],
        [rescale_image_size(image, factor) for factor in size_factors],
    ) for image, prompt in zip(images, HF_IMAGE_PROMPTS)]
    print(inputs_per_image)
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

    AutoModelForVision2Seq.register(DeepSeekMultiModalityConfig,
                                    MultiModalityCausalLM)

    with hf_runner(model, dtype=dtype, is_vision_model=True) as hf_model:
        hf_model.processor = VLChatProcessor.from_pretrained(model)
        hf_model.model.tokenizer = AutoTokenizer.from_pretrained(model)

        hf_outputs_per_image = [
            hf_model.generate_greedy_logprobs_limit(prompts,
                                                    max_tokens,
                                                    num_logprobs=num_logprobs,
                                                    images=images)
            for prompts, images in inputs_per_image
        ]

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_image,
                                        vllm_outputs_per_image):
        # TODO: Check whether using original CLIPVisionModel can improve
        # consistency against HF
        # print(f'hf_outputs: {hf_outputs}')
        # print(f'vllm_outputs: {vllm_outputs}')
        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=vllm_outputs,
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
        size_factors=size_factors,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
    )
