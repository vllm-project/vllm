from typing import Optional

import pytest
from transformers import AutoModelForVision2Seq

from ....utils import multi_gpu_test
from .utils import VLMTestInfo
from .vlm_test_types import VLMTestInfo, VlmTestType, ImageSizeWrapper
from ...utils import check_outputs_equal
from ....conftest import _ImageAssets
from . import utils as vlm_utils


# Common settings leveraged by all of the broadcast tests
COMMON_BROADCAST_SETTINGS = {
    "test_type": VlmTestType.IMAGE,
    "dtype": "half",
    "max_tokens": 5,
    "tensor_parallel_size": 2,
    "image_size_factors": ((.25, 0.5, 1.0),),
    "distributed_executor_backend": ("ray", "mp"),
}


# Model specific settings
BROADCAST_TEST_SETTINGS = {
    "broadcast-chameleon": VLMTestInfo(
        models="facebook/chameleon-7b",
        prompt_formatter=lambda img_prompt: f"USER: {img_prompt}\nASSISTANT:",
        max_model_len=4096,
        auto_cls=AutoModelForVision2Seq,
        postprocess_inputs=vlm_utils.get_key_type_post_processor(
            "pixel_values",
            "half"
        ),
        vllm_output_post_proc = lambda vllm_output, model: vllm_output[:2],
        hf_output_post_proc = lambda hf_output, model: hf_output[:2],
        comparator=check_outputs_equal,
        **COMMON_BROADCAST_SETTINGS,
    ),
    "broadcast-llava": VLMTestInfo(
        models="llava-hf/llava-1.5-7b-hf",
        prompt_formatter=lambda img_prompt: f"USER: {img_prompt}\nASSISTANT:",
        max_model_len=4096,
        auto_cls=AutoModelForVision2Seq,
        vllm_output_post_proc=vlm_utils.llava_image_vllm_to_hf_output,
        **COMMON_BROADCAST_SETTINGS,
    ),
    "broadcast-llava-next": VLMTestInfo(
        models="llava-hf/llava-v1.6-mistral-7b-hf",
        prompt_formatter=lambda img_prompt: f"[INST] {img_prompt} [/INST]",
        max_model_len=10240,
        auto_cls=AutoModelForVision2Seq,
        vllm_output_post_proc=vlm_utils.llava_image_vllm_to_hf_output,
        **COMMON_BROADCAST_SETTINGS,
    )
}


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "model_type,model,max_tokens,num_logprobs,dtype,distributed_executor_backend,size_wrapper",
    vlm_utils.get_parametrized_options(BROADCAST_TEST_SETTINGS,
                                       test_type=VlmTestType.IMAGE))
def test_multi_gpu_broadcast(model_type: str,
                             model: str, max_tokens: int,
                             num_logprobs: int, dtype: str,
                             distributed_executor_backend: Optional[str],
                             size_wrapper: ImageSizeWrapper,
                             hf_runner, vllm_runner,
                             image_assets: _ImageAssets):
    test_info = BROADCAST_TEST_SETTINGS[model_type]
    inputs = vlm_utils.build_single_image_inputs_from_test_info(
        test_info, image_assets, size_wrapper
    )

    vlm_utils.run_test(
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        inputs=inputs,
        model=model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        limit_mm_per_prompt={"image": 1},
        distributed_executor_backend=distributed_executor_backend,
        size_factors=size_wrapper,
        **test_info.get_non_parametrized_runner_kwargs()
    )
