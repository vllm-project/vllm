from typing import List, Tuple

import pytest
from transformers import AutoTokenizer

from vllm.config import VisionLanguageConfig
from vllm.utils import is_cpu

from ..conftest import IMAGE_FILES

pytestmark = pytest.mark.vlm

# The image token is placed before "user" on purpose so that the test can pass
HF_IMAGE_PROMPTS = [
    "<|user|>\n<|image_1|>\nWhat's the content of the image?<|end|>\n<|assistant|>\n",  # noqa: E501
    "<|user|>\n<|image_1|>\nWhat is the season?<|end|>\n<|assistant|>\n",
]

assert len(HF_IMAGE_PROMPTS) == len(IMAGE_FILES)


def iter_phi3v_configs(model_name: str):
    image_hw_to_feature_size = {
        (1008, 1344): 1921,
    }

    for (h, w), f in image_hw_to_feature_size.items():
        for input_type, input_shape in [
            (VisionLanguageConfig.ImageInputType.PIXEL_VALUES, (1, 3, h, w)),
        ]:
            yield (model_name,
                   VisionLanguageConfig(image_input_type=input_type,
                                        image_feature_size=f,
                                        image_token_id=32044,
                                        image_input_shape=input_shape,
                                        image_processor=model_name,
                                        image_processor_revision=None))


model_and_vl_config = [
    *iter_phi3v_configs("microsoft/Phi-3-vision-128k-instruct"),
]


def vllm_to_hf_output(vllm_output: Tuple[List[int], str],
                      vlm_config: VisionLanguageConfig, model_id: str):
    """Sanitize vllm output to be comparable with hf output.
    The function reduces `input_ids` from 1, 32000, 32000, ..., 32000,
    x1, x2, x3 ... to 1, 32000, x1, x2, x3 ...
    It also reduces `output_str` from "<image><image>bla" to "bla".
    """
    input_ids, output_str = vllm_output
    image_token_id = vlm_config.image_token_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    image_token_str = tokenizer.decode(image_token_id)

    hf_input_ids = [
        input_id if input_id != image_token_id else 0
        for idx, input_id in enumerate(input_ids)
    ]
    hf_output_str = output_str \
        .replace(image_token_str * vlm_config.image_feature_size, "") \
        .replace("<s>", " ").replace("<|user|>", "") \
        .replace("<|end|>\n<|assistant|>", " ")

    return hf_input_ids, hf_output_str


target_dtype = "half"
if is_cpu():
    target_dtype = "bfloat16"


# TODO: Add test for `tensor_parallel_size` [ref: PR #3883]
# Since we use _attn_implementation="eager" for hf_runner, here is
# numeric difference for longer context and test can't pass
@pytest.mark.parametrize("model_and_config", model_and_vl_config)
@pytest.mark.parametrize("dtype", [target_dtype])
@pytest.mark.parametrize("max_tokens", [8])
def test_models(hf_runner, vllm_runner, hf_images, vllm_images,
                model_and_config, dtype: str, max_tokens: int) -> None:
    """Inference result should be the same between hf and vllm.

    All the image fixtures for the test is under tests/images.
    For huggingface runner, we provide the PIL images as input.
    For vllm runner, we provide MultiModalData objects and corresponding
    vision language config as input.
    Note, the text input is also adjusted to abide by vllm contract.
    The text output is sanitized to be able to compare with hf.
    """
    model_id, vlm_config = model_and_config

    # use eager mode for hf runner, since phi3_v didn't work with flash_attn
    hf_model_kwargs = {"_attn_implementation": "eager"}
    with hf_runner(model_id, dtype=dtype,
                   model_kwargs=hf_model_kwargs) as hf_model:
        hf_outputs = hf_model.generate_greedy(HF_IMAGE_PROMPTS,
                                              max_tokens,
                                              images=hf_images)

    vllm_image_prompts = [
        p.replace("<|image_1|>",
                  "<|image|>" * vlm_config.image_feature_size + "<s>")
        for p in HF_IMAGE_PROMPTS
    ]

    with vllm_runner(model_id,
                     max_model_len=2048,
                     dtype=dtype,
                     enforce_eager=True,
                     **vlm_config.as_cli_args_dict()) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(vllm_image_prompts,
                                                  max_tokens,
                                                  images=vllm_images)

    for i in range(len(HF_IMAGE_PROMPTS)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_to_hf_output(
            vllm_outputs[i], vlm_config, model_id)
        assert hf_output_str == vllm_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
        assert hf_output_ids == vllm_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")
