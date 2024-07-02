from typing import List, Optional, Tuple, Type

import pytest

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from transformers import LlamaForCausalLM
from transformers import AutoTokenizer

from vllm.config import VisionLanguageConfig

from ..conftest import IMAGE_ASSETS, HfRunner, VllmRunner, _ImageAssets

from vllm.model_executor.models.deepseek_vl import (
    model_name_to_cls,
    MultiModalityPreTrainedModel,
    VLMImageProcessor,
)
from vllm.transformers_utils.config import DeepSeekMultiModalityConfig


pytestmark = pytest.mark.vlm

# The image token is placed before "user" on purpose so that the test can pass
HF_IMAGE_PROMPTS = [
    "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n User: <image_placeholder>What's the content of the image?\nAssistant:",
    "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n User: <image_placeholder>What is the season?\nAssistant:",
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
            input_ids
        ).reshape(1, -1, 4096)

        # replace with the image embeddings
        images_embeds = images_embeds.reshape(
            1, -1, self.config.aligner_config.params["n_embed"]
        )
        inputs_embeds[images_seq_mask] = images_embeds

        return inputs_embeds


def get_input(tokenizer, prompt, image):

    image_id = 100015
    vl_image = VLMImageProcessor(1024)
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)
    image_token_mask = input_ids == image_id
    images_outputs = vl_image(image, return_tensors="pt")
    images_emb_mask = torch.ones(1, 1, 576) == 1
    prepare = {
        "sft_format": prompt,
        "input_ids": input_ids.to("cuda"),
        "pixel_values": images_outputs.pixel_values.to(torch.bfloat16)
        .to("cuda")
        .reshape(1, -1, 3, 1024, 1024),
        "num_image_tokens": 576,
        "images_seq_mask": image_token_mask.to("cuda").reshape(1, -1),
        "images_emb_mask": images_emb_mask.to("cuda"),
        "attention_mask": torch.ones(1, len(input_ids)).to("cuda"),
    }
    return prepare


def iter_deepseek_vl_configs(model_name: str):
    image_hw_to_feature_size = {
        (1024, 1024): 576,
    }

    for (h, w), f in image_hw_to_feature_size.items():
        for input_type, input_shape in [
            (VisionLanguageConfig.ImageInputType.PIXEL_VALUES, (1, 3, h, w)),
            (VisionLanguageConfig.ImageInputType.IMAGE_FEATURES, (1, f, 1024)),
        ]:
            yield (
                model_name,
                VisionLanguageConfig(
                    image_input_type=input_type,
                    image_feature_size=f,
                    image_token_id=100015,
                    image_input_shape=input_shape,
                    image_processor=model_name,
                    image_processor_revision=None,
                ),
            )


model_and_vl_config = [
    *iter_deepseek_vl_configs("deepseek-ai/deepseek-vl-7b-chat"),
]


def vllm_to_hf_output(
    vllm_output: Tuple[List[int], str],
    vlm_config: VisionLanguageConfig,
    model_id: str,
):
    """Sanitize vllm output to be comparable with hf output.
    The function reduces `input_ids` from 1, 32000, 32000, ..., 32000,
    x1, x2, x3 ... to 1, 32000, x1, x2, x3 ...
    It also reduces `output_str` from "<image_placeholder><image_placeholder>bla" to "bla".
    """
    input_ids, output_str = vllm_output
    image_token_id = vlm_config.image_token_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    image_token_str = tokenizer.decode(image_token_id)

    hf_input_ids = [
        input_id
        for idx, input_id in enumerate(input_ids)
        if input_id != image_token_id or input_ids[idx - 1] != image_token_id
    ]
    hf_output_str = output_str.replace(
        image_token_str * vlm_config.image_feature_size, ""
    )

    return hf_input_ids, hf_output_str


# TODO: Add test for `tensor_parallel_size` [ref: PR #3883]
@pytest.mark.parametrize("model_and_config", model_and_vl_config)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
def run_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    image_assets: _ImageAssets,
    model_and_config: Tuple[str, VisionLanguageConfig],
    *,
    dtype: str,
    max_tokens: int,
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
    model_id, vlm_config = model_and_config
    hf_images = [asset.for_hf() for asset in image_assets]

    vllm_image_prompts = [
        p.replace(
            "<image_placeholder>",
            "<image_placeholder>" * vlm_config.image_feature_size,
        )
        for p in HF_IMAGE_PROMPTS
    ]

    with vllm_runner(model_id,
                     dtype=dtype,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True,
                     **vlm_config.as_cli_args_dict()) as vllm_model:
        vllm_images = [asset.for_vllm(vlm_config) for asset in image_assets]
        vllm_outputs = vllm_model.generate_greedy(
            vllm_image_prompts, max_tokens, images=vllm_images
        )
    AutoModelForCausalLM.register(
        DeepSeekMultiModalityConfig, MultiModalityCausalLM
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True
    )
    hf_model = hf_model.to("cuda").eval()
    prepare_input_one = get_input(
        tokenizer,
        HF_IMAGE_PROMPTS[0].replace(
            "<image_placeholder>", "<image_placeholder>" * 576
        ),
        hf_images,
    )
    prepare_input_two = get_input(
        tokenizer,
        HF_IMAGE_PROMPTS[1].replace(
            "<image_placeholder>", "<image_placeholder>" * 576
        ),
        hf_images,
    )
    prepare_input_one = hf_model.prepare_inputs_embeds(**prepare_input_one)
    prepare_input_two = hf_model.prepare_inputs_embeds(**prepare_input_two)
    prepare_input = torch.concat(prepare_input_one, prepare_input_two)
    attention_mask = torch.concat(
        prepare_input_one["attention_mask"],
        prepare_input_two["attention_mask"],
    )
    outputs = hf_model.generate(
        inputs_embeds=prepare_input,
        attention_mask=attention_mask,
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
        use_cache=True,
    )
    hf_outputs = []
    for o in outputs:
        hf_outputs.append(
            o, tokenizer.decode(o.cpu().tolist(), skip_special_tokens=True)
        )

    for i in range(len(HF_IMAGE_PROMPTS)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_to_hf_output(
            vllm_outputs[i], vlm_config, model_id
        )
        assert (
            hf_output_str == vllm_output_str
        ), f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}"
        assert (
            hf_output_ids == vllm_output_ids
        ), f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}"


@pytest.mark.parametrize("model_and_config", model_and_vl_config)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
def test_models(
    hf_runner,
    vllm_runner,
    image_assets,
    model_and_config,
    dtype: str,
    max_tokens: int,
) -> None:
    run_test(
        hf_runner,
        vllm_runner,
        image_assets,
        model_and_config,
        dtype=dtype,
        max_tokens=max_tokens,
        tensor_parallel_size=1,
    )
