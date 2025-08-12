from typing import List, Optional, Tuple, Type

import pytest
from transformers import AutoTokenizer

from vllm.config import VisionLanguageConfig
from vllm.utils import is_cpu

from ..conftest import IMAGE_ASSETS, HfRunner, VllmRunner, _ImageAssets
from .utils import check_outputs_equal

pytestmark = pytest.mark.vlm

# The image token is placed before "user" on purpose so that the test can pass
HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    "<|user|>\n<|image_1|>\nWhat's the content of the image?<|end|>\n<|assistant|>\n",  # noqa: E501
    "cherry_blossom":
    "<|user|>\n<|image_1|>\nWhat is the season?<|end|>\n<|assistant|>\n",  # noqa: E501
})


def iter_phi3v_configs(model_name: str):
    image_hw_to_feature_size = {
        (1008, 1344): 1921,
        (2016, 2688): 1933,
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
    output_ids, output_str = vllm_output
    image_token_id = vlm_config.image_token_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    image_token_str = tokenizer.decode(image_token_id)

    hf_output_ids = [
        token_id if token_id != image_token_id else 0
        for idx, token_id in enumerate(output_ids)
    ]
    hf_output_str = output_str \
        .replace(image_token_str * vlm_config.image_feature_size, "") \
        .replace("<s>", " ").replace("<|user|>", "") \
        .replace("<|end|>\n<|assistant|>", " ")

    return hf_output_ids, hf_output_str


target_dtype = "half"
if is_cpu():
    target_dtype = "bfloat16"


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
):
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

    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).

    with vllm_runner(model_id,
                     max_model_len=2048,
                     dtype=dtype,
                     tensor_parallel_size=tensor_parallel_size,
                     enforce_eager=True,
                     distributed_executor_backend=distributed_executor_backend,
                     **vlm_config.as_cli_args_dict()) as vllm_model:
        # NOTE: `asset.for_vllm` will call `torch.cuda.device_count()`
        # we must put it inside the vllm_runner context manager
        # i.e. after creating vLLM instance.

        vllm_images = [asset.for_vllm(vlm_config) for asset in image_assets]

        vllm_image_prompts = [
            p.replace("<|image_1|>",
                      "<|image|>" * vlm_config.image_feature_size + "<s>")
            for p in HF_IMAGE_PROMPTS
        ]

        vllm_outputs = vllm_model.generate_greedy(vllm_image_prompts,
                                                  max_tokens,
                                                  images=vllm_images)

    # use eager mode for hf runner, since phi3_v didn't work with flash_attn
    hf_model_kwargs = {"_attn_implementation": "eager"}
    with hf_runner(model_id, dtype=dtype,
                   model_kwargs=hf_model_kwargs) as hf_model:
        hf_outputs = hf_model.generate_greedy(
            HF_IMAGE_PROMPTS,
            max_tokens,
            images=hf_images,
            eos_token_id=hf_model.processor.tokenizer.eos_token_id)

    check_outputs_equal(
        hf_outputs,
        [
            vllm_to_hf_output(vllm_output, vlm_config, model_id)
            for vllm_output in vllm_outputs
        ],
        name_0="hf",
        name_1="vllm",
    )


# Since we use _attn_implementation="eager" for hf_runner, here is
# numeric difference for longer context and test can't pass
@pytest.mark.xfail(
    reason="Inconsistent image processor being used due to lack "
    "of support for dynamic image token replacement")
@pytest.mark.parametrize("model_and_config", model_and_vl_config)
@pytest.mark.parametrize("dtype", [target_dtype])
@pytest.mark.parametrize("max_tokens", [128])
def test_models(hf_runner, vllm_runner, image_assets, model_and_config,
                dtype: str, max_tokens: int) -> None:
    run_test(
        hf_runner,
        vllm_runner,
        image_assets,
        model_and_config,
        dtype=dtype,
        max_tokens=max_tokens,
        tensor_parallel_size=1,
    )
