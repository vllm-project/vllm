from typing import List, Optional, Tuple, Type

import pytest
from transformers import AutoTokenizer

from vllm.config import VisionLanguageConfig

from ..conftest import IMAGE_ASSETS, HfRunner, VllmRunner, _ImageAssets
from .utils import check_outputs_equal

pytestmark = pytest.mark.vlm

# The image token is placed before "user" on purpose so that the test can pass
HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    "<image>\nUSER: What's the content of the image?\nASSISTANT:",
    "cherry_blossom":
    "<image>\nUSER: What is the season?\nASSISTANT:",
})


def iter_llava_configs(model_name: str):
    image_hw_to_feature_size = {
        (336, 336): 576,
    }

    for (h, w), f in image_hw_to_feature_size.items():
        input_shape = (1, 3, h, w)
        yield (model_name,
               VisionLanguageConfig(image_feature_size=f,
                                    image_token_id=32000,
                                    image_input_shape=input_shape))


model_and_vl_config = [
    *iter_llava_configs("llava-hf/llava-1.5-7b-hf"),
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
        token_id for idx, token_id in enumerate(output_ids)
        if token_id != image_token_id or output_ids[idx - 1] != image_token_id
    ]
    hf_output_str = output_str \
        .replace(image_token_str * vlm_config.image_feature_size, "")

    return hf_output_ids, hf_output_str


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
    For vllm runner, we provide MultiModalDataDict objects 
    and corresponding vision language config as input.
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
                     dtype=dtype,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True,
                     **vlm_config.as_cli_args_dict()) as vllm_model:

        # NOTE: `asset.for_vllm` will call `torch.cuda.device_count()`
        # we must put it inside the vllm_runner context manager
        # i.e. after creating vLLM instance.
        vllm_images = [asset.for_vllm() for asset in image_assets]

        vllm_image_prompts = [
            p.replace("<image>", "<image>" * vlm_config.image_feature_size)
            for p in HF_IMAGE_PROMPTS
        ]

        vllm_outputs = vllm_model.generate_greedy(vllm_image_prompts,
                                                  max_tokens,
                                                  images=vllm_images)

    with hf_runner(model_id, dtype=dtype, is_vision_model=True) as hf_model:
        hf_outputs = hf_model.generate_greedy(HF_IMAGE_PROMPTS,
                                              max_tokens,
                                              images=hf_images)

    check_outputs_equal(
        hf_outputs,
        [
            vllm_to_hf_output(vllm_output, vlm_config, model_id)
            for vllm_output in vllm_outputs
        ],
        name_0="hf",
        name_1="vllm",
    )


@pytest.mark.parametrize("model_and_config", model_and_vl_config)
@pytest.mark.parametrize("dtype", ["half"])
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
