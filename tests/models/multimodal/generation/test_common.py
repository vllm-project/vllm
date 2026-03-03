# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Common tests for testing .generate() functionality for single / multiple
image, embedding, and video support for different VLMs in vLLM.
"""

import math
from collections import defaultdict
from pathlib import PosixPath

import pytest
from packaging.version import Version
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForTextToWaveform,
)
from transformers import __version__ as TRANSFORMERS_VERSION

from vllm.platforms import current_platform
from vllm.utils.func_utils import identity

from ....conftest import (
    IMAGE_ASSETS,
    AudioTestAssets,
    HfRunner,
    ImageTestAssets,
    VideoTestAssets,
    VllmRunner,
)
from ....utils import create_new_process_for_each_test, large_gpu_mark, multi_gpu_marks
from ...utils import check_outputs_equal
from .vlm_utils import custom_inputs, model_utils, runners
from .vlm_utils.case_filtering import get_parametrized_options
from .vlm_utils.types import (
    CustomTestOptions,
    ExpandableVLMTestArgs,
    VLMTestInfo,
    VLMTestType,
)

COMMON_BROADCAST_SETTINGS = {
    "test_type": VLMTestType.IMAGE,
    "dtype": "half",
    "max_tokens": 5,
    "tensor_parallel_size": 2,
    "hf_model_kwargs": {"device_map": "auto"},
    "image_size_factors": [(0.25, 0.5, 1.0)],
    "distributed_executor_backend": (
        "ray",
        "mp",
    ),
}

### Test configuration for specific models
# NOTE: The convention of the test settings below is to lead each test key
# with the name of the model arch used in the test, using underscores in place
# of hyphens; this makes it more convenient to filter tests for a specific kind
# of model. For example....
#
# To run all test types for a specific key:
#     use the k flag to substring match with a leading square bracket; if the
#     model arch happens to be a substring of another one, you can add a
#     trailing hyphen. E.g.,
#                 - pytest $TEST_FILE -k "[llava-"
#     prevents matching on "[llava_next-" & will match just the enabled cases
#     for llava, i.e., single image, image embedding, and custom input tests.
#
# To run a test for a Test Info for just one of multiple models:
#     use the k flag to substring match the model name, e.g.,
#                 - pytest $TEST_FILE -k OpenGVLab/InternVL2-1B
#     prevents matching on nGVLab/InternVL2-2B.
#
# You can also combine substrings to match more granularly.
#     ex 1:
#        pytest $TEST_FILE -k "test_single_image and OpenGVLab/InternVL2-1B"
#     will run only test_single_image* for OpenGVLab/InternVL2-1B; this would
#     match both wrappers for single image tests, since it also matches
#     test_single_image_heavy (which forks if we have a distributed backend)
#     ex 2:
#        pytest $TEST_FILE -k  "[llava- or [intern_vl-"
#     will run all of the tests for only llava & internvl.
#
# NOTE you can add --collect-only to any of the above commands to see
# which cases would be selected and deselected by pytest. In general,
# this is a good idea for checking your command first, since tests are slow.

VLM_TEST_SETTINGS = {
    #### Core tests to always run in the CI
    "llava": VLMTestInfo(
        models=["llava-hf/llava-1.5-7b-hf"],
        test_type=(VLMTestType.EMBEDDING, VLMTestType.IMAGE, VLMTestType.CUSTOM_INPUTS),
        prompt_formatter=lambda img_prompt: f"USER: {img_prompt}\nASSISTANT:",
        convert_assets_to_embeddings=model_utils.get_llava_embeddings,
        max_model_len=4096,
        auto_cls=AutoModelForImageTextToText,
        vllm_output_post_proc=model_utils.llava_image_vllm_to_hf_output,
        custom_test_opts=[
            CustomTestOptions(
                inputs=custom_inputs.multi_image_multi_aspect_ratio_inputs(
                    formatter=lambda img_prompt: f"USER: {img_prompt}\nASSISTANT:"
                ),
                limit_mm_per_prompt={"image": 4},
            )
        ],
        vllm_runner_kwargs={"enable_mm_embeds": True},
        marks=[pytest.mark.core_model, pytest.mark.cpu_model],
    ),
    "paligemma": VLMTestInfo(
        models=["google/paligemma-3b-mix-224"],
        test_type=VLMTestType.IMAGE,
        prompt_formatter=identity,
        img_idx_to_prompt=lambda idx: "",
        # Paligemma uses its own sample prompts because the default one fails
        single_image_prompts=IMAGE_ASSETS.prompts(
            {
                "stop_sign": "caption es",
                "cherry_blossom": "What is in the picture?",
            }
        ),
        auto_cls=AutoModelForImageTextToText,
        vllm_output_post_proc=model_utils.paligemma_vllm_to_hf_output,
    ),
    "qwen2_5_vl": VLMTestInfo(
        models=["Qwen/Qwen2.5-VL-3B-Instruct"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE, VLMTestType.VIDEO),
        prompt_formatter=lambda img_prompt: f"<|im_start|>User\n{img_prompt}<|im_end|>\n<|im_start|>assistant\n",  # noqa: E501
        img_idx_to_prompt=lambda idx: "<|vision_start|><|image_pad|><|vision_end|>",
        video_idx_to_prompt=lambda idx: "<|vision_start|><|video_pad|><|vision_end|>",
        enforce_eager=False,
        max_model_len=4096,
        max_num_seqs=2,
        auto_cls=AutoModelForImageTextToText,
        vllm_output_post_proc=model_utils.qwen2_vllm_to_hf_output,
        image_size_factors=[(0.25,), (0.25, 0.25, 0.25), (0.25, 0.2, 0.15)],
        marks=[pytest.mark.core_model, pytest.mark.cpu_model],
    ),
    "qwen2_5_omni": VLMTestInfo(
        models=["Qwen/Qwen2.5-Omni-3B"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE, VLMTestType.VIDEO),
        prompt_formatter=lambda img_prompt: f"<|im_start|>User\n{img_prompt}<|im_end|>\n<|im_start|>assistant\n",  # noqa: E501
        img_idx_to_prompt=lambda idx: "<|vision_bos|><|IMAGE|><|vision_eos|>",
        video_idx_to_prompt=lambda idx: "<|vision_bos|><|VIDEO|><|vision_eos|>",
        max_model_len=4096,
        max_num_seqs=2,
        num_logprobs=6 if current_platform.is_cpu() else 5,
        auto_cls=AutoModelForTextToWaveform,
        vllm_output_post_proc=model_utils.qwen2_vllm_to_hf_output,
        patch_hf_runner=model_utils.qwen2_5_omni_patch_hf_runner,
        image_size_factors=[(0.25,), (0.25, 0.25, 0.25), (0.25, 0.2, 0.15)],
        marks=[pytest.mark.core_model, pytest.mark.cpu_model],
    ),
    "qwen3_vl": VLMTestInfo(
        models=["Qwen/Qwen3-VL-4B-Instruct"],
        test_type=(
            VLMTestType.IMAGE,
            VLMTestType.MULTI_IMAGE,
            VLMTestType.VIDEO,
        ),
        enforce_eager=False,
        needs_video_metadata=True,
        prompt_formatter=lambda img_prompt: f"<|im_start|>User\n{img_prompt}<|im_end|>\n<|im_start|>assistant\n",  # noqa: E501
        img_idx_to_prompt=lambda idx: "<|vision_start|><|image_pad|><|vision_end|>",  # noqa: E501
        video_idx_to_prompt=lambda idx: "<|vision_start|><|video_pad|><|vision_end|>",  # noqa: E501
        max_model_len=4096,
        max_num_seqs=2,
        num_logprobs=20,
        auto_cls=AutoModelForImageTextToText,
        vllm_output_post_proc=model_utils.qwen2_vllm_to_hf_output,
        patch_hf_runner=model_utils.qwen3_vl_patch_hf_runner,
        vllm_runner_kwargs={
            "attention_config": {
                "backend": "ROCM_AITER_FA",
            },
        }
        if current_platform.is_rocm()
        else None,
        image_size_factors=[(0.25,), (0.25, 0.25, 0.25), (0.25, 0.2, 0.15)],
        marks=[
            pytest.mark.core_model,
        ],
    ),
    "ultravox": VLMTestInfo(
        models=["fixie-ai/ultravox-v0_5-llama-3_2-1b"],
        test_type=VLMTestType.AUDIO,
        prompt_formatter=lambda audio_prompt: f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{audio_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",  # noqa: E501
        audio_idx_to_prompt=lambda idx: "<|audio|>",
        max_model_len=4096,
        max_num_seqs=2,
        auto_cls=AutoModel,
        hf_output_post_proc=model_utils.ultravox_trunc_hf_output,
        marks=[pytest.mark.core_model, pytest.mark.cpu_model],
    ),
    #### Transformers fallback to test
    ## To reduce test burden, we only test batching arbitrary image size
    # Dynamic image length and number of patches
    "llava-onevision-transformers": VLMTestInfo(
        models=["llava-hf/llava-onevision-qwen2-0.5b-ov-hf"],
        test_type=VLMTestType.IMAGE,
        prompt_formatter=lambda vid_prompt: f"<|im_start|>user\n{vid_prompt}<|im_end|>\n<|im_start|>assistant\n",  # noqa: E501
        max_model_len=16384,
        hf_model_kwargs=model_utils.llava_onevision_hf_model_kwargs(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
        ),
        auto_cls=AutoModelForImageTextToText,
        vllm_output_post_proc=model_utils.llava_onevision_vllm_to_hf_output,
        image_size_factors=[(0.25, 0.5, 1.0)],
        vllm_runner_kwargs={
            "model_impl": "transformers",
            "default_torch_num_threads": 1,
        },
        # FIXME: Investigate why the test hangs
        # when processing the 3rd prompt in vLLM
        marks=[pytest.mark.core_model, pytest.mark.skip(reason="Test hangs")],
    ),
    # Gemma3 has bidirectional mask on images
    "gemma3-transformers": VLMTestInfo(
        models=["google/gemma-3-4b-it"],
        test_type=VLMTestType.IMAGE,
        prompt_formatter=lambda vid_prompt: f"<'<bos><start_of_turn>user\n{vid_prompt}<start_of_image><end_of_turn>\n<start_of_turn>model\n",  # noqa: E501
        max_model_len=4096,
        auto_cls=AutoModelForImageTextToText,
        vllm_output_post_proc=model_utils.gemma3_vllm_to_hf_output,
        image_size_factors=[(0.25, 0.5, 1.0)],
        vllm_runner_kwargs={
            "model_impl": "transformers",
        },
        marks=[pytest.mark.core_model],
    ),
    "idefics3-transformers": VLMTestInfo(
        models=["HuggingFaceTB/SmolVLM-256M-Instruct"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"<|begin_of_text|>User:{img_prompt}<end_of_utterance>\nAssistant:",  # noqa: E501
        img_idx_to_prompt=lambda idx: "<image>",
        max_model_len=8192,
        max_num_seqs=2,
        auto_cls=AutoModelForImageTextToText,
        hf_output_post_proc=model_utils.idefics3_trunc_hf_output,
        image_size_factors=[(0.25, 0.5, 1.0)],
        vllm_runner_kwargs={
            "model_impl": "transformers",
        },
        marks=[pytest.mark.core_model],
    ),
    # Pixel values from processor are not 4D or 5D arrays
    "qwen2_5_vl-transformers": VLMTestInfo(
        models=["Qwen/Qwen2.5-VL-3B-Instruct"],
        test_type=VLMTestType.IMAGE,
        prompt_formatter=lambda img_prompt: f"<|im_start|>User\n{img_prompt}<|im_end|>\n<|im_start|>assistant\n",  # noqa: E501
        img_idx_to_prompt=lambda idx: "<|vision_start|><|image_pad|><|vision_end|>",
        max_model_len=4096,
        max_num_seqs=2,
        auto_cls=AutoModelForImageTextToText,
        vllm_output_post_proc=model_utils.qwen2_vllm_to_hf_output,
        image_size_factors=[(0.25, 0.2, 0.15)],
        vllm_runner_kwargs={
            "model_impl": "transformers",
            # TODO: [ROCm] Revert this once issue #30167 is resolved
            **(
                {
                    "mm_processor_kwargs": {
                        "min_pixels": 256 * 28 * 28,
                        "max_pixels": 1280 * 28 * 28,
                    },
                }
                if current_platform.is_rocm()
                else {}
            ),
        },
        marks=[large_gpu_mark(min_gb=80 if current_platform.is_rocm() else 32)],
    ),
    #### Extended model tests
    "aria": VLMTestInfo(
        models=["rhymes-ai/Aria"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"<|im_start|>user\n{img_prompt}<|im_end|>\n<|im_start|>assistant\n ",  # noqa: E501
        img_idx_to_prompt=lambda idx: "<fim_prefix><|img|><fim_suffix>\n",
        max_model_len=4096,
        max_num_seqs=2,
        auto_cls=AutoModelForImageTextToText,
        single_image_prompts=IMAGE_ASSETS.prompts(
            {
                "stop_sign": "<vlm_image>Please describe the image shortly.",
                "cherry_blossom": "<vlm_image>Please infer the season with reason.",
            }
        ),
        multi_image_prompt="<vlm_image><vlm_image>Describe the two images shortly.",
        stop_str=["<|im_end|>"],
        image_size_factors=[(0.10, 0.15)],
        max_tokens=64,
        marks=[large_gpu_mark(min_gb=64)],
    ),
    "aya_vision": VLMTestInfo(
        models=["CohereLabs/aya-vision-8b"],
        test_type=(VLMTestType.IMAGE),
        prompt_formatter=lambda img_prompt: f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{img_prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",  # noqa: E501
        single_image_prompts=IMAGE_ASSETS.prompts(
            {
                "stop_sign": "<image>What's the content in the center of the image?",
                "cherry_blossom": "<image>What is the season?",
            }
        ),
        multi_image_prompt="<image><image>Describe the two images in detail.",
        max_model_len=4096,
        max_num_seqs=2,
        auto_cls=AutoModelForImageTextToText,
        vllm_runner_kwargs={"mm_processor_kwargs": {"crop_to_patches": True}},
    ),
    "aya_vision-multi_image": VLMTestInfo(
        models=["CohereLabs/aya-vision-8b"],
        test_type=(VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{img_prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",  # noqa: E501
        single_image_prompts=IMAGE_ASSETS.prompts(
            {
                "stop_sign": "<image>What's the content in the center of the image?",
                "cherry_blossom": "<image>What is the season?",
            }
        ),
        multi_image_prompt="<image><image>Describe the two images in detail.",
        max_model_len=4096,
        max_num_seqs=2,
        auto_cls=AutoModelForImageTextToText,
        vllm_runner_kwargs={"mm_processor_kwargs": {"crop_to_patches": True}},
        marks=[large_gpu_mark(min_gb=32)],
    ),
    "blip2": VLMTestInfo(
        models=["Salesforce/blip2-opt-2.7b"],
        test_type=VLMTestType.IMAGE,
        prompt_formatter=lambda img_prompt: f"Question: {img_prompt} Answer:",
        img_idx_to_prompt=lambda idx: "",
        auto_cls=AutoModelForImageTextToText,
        vllm_output_post_proc=model_utils.blip2_vllm_to_hf_output,
        # FIXME: https://github.com/huggingface/transformers/pull/38510
        marks=[pytest.mark.skip("Model is broken")],
    ),
    "chameleon": VLMTestInfo(
        models=["facebook/chameleon-7b"],
        test_type=VLMTestType.IMAGE,
        prompt_formatter=lambda img_prompt: f"USER: {img_prompt}\nASSISTANT:",
        max_model_len=4096,
        max_num_seqs=2,
        auto_cls=AutoModelForImageTextToText,
        # For chameleon, we only compare the sequences
        vllm_output_post_proc=lambda vllm_output, model: vllm_output[:2],
        hf_output_post_proc=lambda hf_output, model: hf_output[:2],
        comparator=check_outputs_equal,
        max_tokens=8,
        dtype="bfloat16",
    ),
    "deepseek_vl_v2": VLMTestInfo(
        models=["Isotr0py/deepseek-vl2-tiny"],  # model repo using dynamic module
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"<|User|>: {img_prompt}\n\n<|Assistant|>: ",  # noqa: E501
        max_model_len=4096,
        max_num_seqs=2,
        single_image_prompts=IMAGE_ASSETS.prompts(
            {
                "stop_sign": "<image>\nWhat's the content in the center of the image?",
                "cherry_blossom": "<image>\nPlease infer the season with reason in details.",  # noqa: E501
            }
        ),
        multi_image_prompt="image_1:<image>\nimage_2:<image>\nWhich image can we see the car and the tower?",  # noqa: E501
        patch_hf_runner=model_utils.deepseekvl2_patch_hf_runner,
        hf_output_post_proc=model_utils.deepseekvl2_trunc_hf_output,
        stop_str=["<｜end▁of▁sentence｜>", "<｜begin▁of▁sentence｜>"],
        image_size_factors=[(1.0,), (1.0, 1.0, 1.0), (0.1, 0.5, 1.0)],
    ),
    "fuyu": VLMTestInfo(
        models=["adept/fuyu-8b"],
        test_type=VLMTestType.IMAGE,
        prompt_formatter=lambda img_prompt: f"{img_prompt}\n",
        img_idx_to_prompt=lambda idx: "",
        max_model_len=2048,
        max_num_seqs=2,
        auto_cls=AutoModelForImageTextToText,
        use_tokenizer_eos=True,
        vllm_output_post_proc=model_utils.fuyu_vllm_to_hf_output,
        num_logprobs=10,
        image_size_factors=[(), (0.25,), (0.25, 0.25, 0.25), (0.25, 0.2, 0.15)],
        marks=[large_gpu_mark(min_gb=32)],
    ),
    "gemma3": VLMTestInfo(
        models=["google/gemma-3-4b-it"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"<bos><start_of_turn>user\n{img_prompt}<end_of_turn>\n<start_of_turn>model\n",  # noqa: E501
        single_image_prompts=IMAGE_ASSETS.prompts(
            {
                "stop_sign": "<start_of_image>What's the content in the center of the image?",  # noqa: E501
                "cherry_blossom": "<start_of_image>What is the season?",
            }
        ),
        multi_image_prompt="<start_of_image><start_of_image>Describe the two images in detail.",  # noqa: E501
        max_model_len=4096,
        max_num_seqs=2,
        auto_cls=AutoModelForImageTextToText,
        vllm_runner_kwargs={"mm_processor_kwargs": {"do_pan_and_scan": True}},
        patch_hf_runner=model_utils.gemma3_patch_hf_runner,
    ),
    "granite_vision": VLMTestInfo(
        models=["ibm-granite/granite-vision-3.3-2b"],
        test_type=(VLMTestType.IMAGE),
        prompt_formatter=lambda img_prompt: f"<|user|>\n{img_prompt}\n<|assistant|>\n",
        max_model_len=8192,
        auto_cls=AutoModelForImageTextToText,
        vllm_output_post_proc=model_utils.llava_image_vllm_to_hf_output,
    ),
    "glm4v": VLMTestInfo(
        models=["zai-org/glm-4v-9b"],
        test_type=VLMTestType.IMAGE,
        prompt_formatter=lambda img_prompt: f"<|user|>\n{img_prompt}<|assistant|>",
        single_image_prompts=IMAGE_ASSETS.prompts(
            {
                "stop_sign": "<|begin_of_image|><|endoftext|><|end_of_image|>What's the content in the center of the image?",  # noqa: E501
                "cherry_blossom": "<|begin_of_image|><|endoftext|><|end_of_image|>What is the season?",  # noqa: E501
            }
        ),
        max_model_len=2048,
        max_num_seqs=2,
        get_stop_token_ids=lambda tok: [151329, 151336, 151338],
        patch_hf_runner=model_utils.glm4v_patch_hf_runner,
        # The image embeddings match with HF but the outputs of the language
        # decoder are only consistent up to 2 decimal places.
        # So, we need to reduce the number of tokens for the test to pass.
        max_tokens=8,
        num_logprobs=10,
        auto_cls=AutoModelForCausalLM,
        marks=[large_gpu_mark(min_gb=32)],
    ),
    "glm4_1v": VLMTestInfo(
        models=["zai-org/GLM-4.1V-9B-Thinking"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"[gMASK]<|user|>\n{img_prompt}<|assistant|>\n",  # noqa: E501
        img_idx_to_prompt=lambda idx: "<|begin_of_image|><|image|><|end_of_image|>",
        video_idx_to_prompt=lambda idx: "<|begin_of_video|><|video|><|end_of_video|>",
        max_model_len=2048,
        max_num_seqs=2,
        get_stop_token_ids=lambda tok: [151329, 151336, 151338],
        num_logprobs=10,
        image_size_factors=[(), (0.25,), (0.25, 0.25, 0.25), (0.25, 0.2, 0.15)],
        auto_cls=AutoModelForImageTextToText,
        marks=[large_gpu_mark(min_gb=32)],
    ),
    "glm4_1v-video": VLMTestInfo(
        models=["zai-org/GLM-4.1V-9B-Thinking"],
        # GLM4.1V require include video metadata for input
        test_type=VLMTestType.CUSTOM_INPUTS,
        prompt_formatter=lambda vid_prompt: f"[gMASK]<|user|>\n{vid_prompt}<|assistant|>\n",  # noqa: E501
        max_model_len=4096,
        max_num_seqs=2,
        auto_cls=AutoModelForImageTextToText,
        patch_hf_runner=model_utils.glm4_1v_patch_hf_runner,
        custom_test_opts=[
            CustomTestOptions(
                inputs=custom_inputs.video_with_metadata_glm4_1v(),
                limit_mm_per_prompt={"video": 1},
            )
        ],
        marks=[large_gpu_mark(min_gb=32)],
    ),
    "glm_ocr": VLMTestInfo(
        models=["zai-org/GLM-OCR"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"[gMASK]<|user|>\n{img_prompt}<|assistant|>\n",  # noqa: E501
        img_idx_to_prompt=lambda idx: "<|begin_of_image|><|image|><|end_of_image|>",
        video_idx_to_prompt=lambda idx: "<|begin_of_video|><|video|><|end_of_video|>",
        max_model_len=2048,
        max_num_seqs=2,
        get_stop_token_ids=lambda tok: [151329, 151336, 151338],
        num_logprobs=10,
        image_size_factors=[(), (0.25,), (0.25, 0.25, 0.25), (0.25, 0.2, 0.15)],
        auto_cls=AutoModelForImageTextToText,
        marks=[large_gpu_mark(min_gb=32)],
    ),
    "h2ovl": VLMTestInfo(
        models=[
            "h2oai/h2ovl-mississippi-800m",
            "h2oai/h2ovl-mississippi-2b",
        ],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"<|prompt|>{img_prompt}<|end|><|answer|>",
        single_image_prompts=IMAGE_ASSETS.prompts(
            {
                "stop_sign": "<image>\nWhat's the content in the center of the image?",
                "cherry_blossom": "<image>\nWhat is the season?",
            }
        ),
        multi_image_prompt="Image-1: <image>\nImage-2: <image>\nDescribe the two images in short.",  # noqa: E501
        max_model_len=8192,
        use_tokenizer_eos=True,
        num_logprobs=10,
        patch_hf_runner=model_utils.h2ovl_patch_hf_runner,
    ),
    "idefics3": VLMTestInfo(
        models=["HuggingFaceTB/SmolVLM-256M-Instruct"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"<|begin_of_text|>User:{img_prompt}<end_of_utterance>\nAssistant:",  # noqa: E501
        img_idx_to_prompt=lambda idx: "<image>",
        max_model_len=8192,
        max_num_seqs=2,
        auto_cls=AutoModelForImageTextToText,
        hf_output_post_proc=model_utils.idefics3_trunc_hf_output,
    ),
    "intern_vl": VLMTestInfo(
        models=[
            "OpenGVLab/InternVL2-1B",
            "OpenGVLab/InternVL2-2B",
            # FIXME: Config cannot be loaded in transformers 4.52
            # "OpenGVLab/Mono-InternVL-2B",
        ],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"<|im_start|>User\n{img_prompt}<|im_end|>\n<|im_start|>Assistant\n",  # noqa: E501
        single_image_prompts=IMAGE_ASSETS.prompts(
            {
                "stop_sign": "<image>\nWhat's the content in the center of the image?",
                "cherry_blossom": "<image>\nWhat is the season?",
            }
        ),
        multi_image_prompt="Image-1: <image>\nImage-2: <image>\nDescribe the two images in short.",  # noqa: E501
        max_model_len=4096,
        use_tokenizer_eos=True,
        patch_hf_runner=model_utils.internvl_patch_hf_runner,
    ),
    "intern_vl-video": VLMTestInfo(
        models=[
            "OpenGVLab/InternVL3-1B",
        ],
        test_type=VLMTestType.VIDEO,
        prompt_formatter=lambda img_prompt: f"<|im_start|>User\n{img_prompt}<|im_end|>\n<|im_start|>Assistant\n",  # noqa: E501
        video_idx_to_prompt=lambda idx: "<video>",
        max_model_len=8192,
        use_tokenizer_eos=True,
        patch_hf_runner=model_utils.internvl_patch_hf_runner,
        num_logprobs=10 if current_platform.is_rocm() else 5,
    ),
    "intern_vl-hf": VLMTestInfo(
        models=["OpenGVLab/InternVL3-1B-hf"],
        test_type=(
            VLMTestType.IMAGE,
            VLMTestType.MULTI_IMAGE,
            VLMTestType.VIDEO,
        ),
        prompt_formatter=lambda img_prompt: f"<|im_start|>User\n{img_prompt}<|im_end|>\n<|im_start|>Assistant\n",  # noqa: E501
        img_idx_to_prompt=lambda idx: "<IMG_CONTEXT>",
        video_idx_to_prompt=lambda idx: "<video>",
        max_model_len=8192,
        use_tokenizer_eos=True,
        auto_cls=AutoModelForImageTextToText,
    ),
    "isaac": VLMTestInfo(
        models=[
            "PerceptronAI/Isaac-0.1",
            "PerceptronAI/Isaac-0.2-2B-Preview",
        ],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: (
            f"<|im_start|>User\n{img_prompt}<|im_end|>\n<|im_start|>assistant\n"
        ),
        img_idx_to_prompt=lambda idx: "<image>",
        single_image_prompts=IMAGE_ASSETS.prompts(
            {
                "stop_sign": "<vlm_image>Please describe the image shortly.",
                "cherry_blossom": "<vlm_image>Please infer the season with reason.",
            }
        ),
        multi_image_prompt=(
            "Picture 1: <vlm_image>\n"
            "Picture 2: <vlm_image>\n"
            "Describe these two images with one paragraph respectively."
        ),
        enforce_eager=False,
        max_model_len=4096,
        max_num_seqs=2,
        hf_model_kwargs={"device_map": "auto"},
        patch_hf_runner=model_utils.isaac_patch_hf_runner,
        image_size_factors=[(0.25,), (0.25, 0.25, 0.25), (0.25, 0.2, 0.15)],
    ),
    "kimi_vl": VLMTestInfo(
        models=["moonshotai/Kimi-VL-A3B-Instruct"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"<|im_user|>user<|im_middle|>{img_prompt}<|im_end|><|im_assistant|>assistant<|im_middle|>",  # noqa: E501
        img_idx_to_prompt=lambda _: "<|media_start|>image<|media_content|><|media_pad|><|media_end|>",  # noqa: E501
        max_model_len=8192,
        max_num_seqs=2,
        dtype="bfloat16",
        tensor_parallel_size=1,
        vllm_output_post_proc=model_utils.kimiv_vl_vllm_to_hf_output,
        marks=[large_gpu_mark(min_gb=48)],
    ),
    "llama4": VLMTestInfo(
        models=["meta-llama/Llama-4-Scout-17B-16E-Instruct"],
        prompt_formatter=lambda img_prompt: f"<|begin_of_text|><|header_start|>user<|header_end|>\n\n{img_prompt}<|eot|><|header_start|>assistant<|header_end|>\n\n",  # noqa: E501
        img_idx_to_prompt=lambda _: "<|image|>",
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        distributed_executor_backend="mp",
        image_size_factors=[(0.25, 0.5, 1.0)],
        hf_model_kwargs={"device_map": "auto"},
        max_model_len=8192,
        max_num_seqs=4,
        dtype="bfloat16",
        auto_cls=AutoModelForImageTextToText,
        tensor_parallel_size=4,
        marks=multi_gpu_marks(num_gpus=4),
    ),
    "llava_next": VLMTestInfo(
        models=["llava-hf/llava-v1.6-mistral-7b-hf"],
        test_type=(VLMTestType.IMAGE, VLMTestType.CUSTOM_INPUTS),
        prompt_formatter=lambda img_prompt: f"[INST] {img_prompt} [/INST]",
        max_model_len=10240,
        auto_cls=AutoModelForImageTextToText,
        vllm_output_post_proc=model_utils.llava_image_vllm_to_hf_output,
        custom_test_opts=[
            CustomTestOptions(
                inputs=custom_inputs.multi_image_multi_aspect_ratio_inputs(
                    formatter=lambda img_prompt: f"[INST] {img_prompt} [/INST]"
                ),
                limit_mm_per_prompt={"image": 4},
            )
        ],
    ),
    "llava_onevision": VLMTestInfo(
        models=["llava-hf/llava-onevision-qwen2-0.5b-ov-hf"],
        test_type=VLMTestType.CUSTOM_INPUTS,
        prompt_formatter=lambda vid_prompt: f"<|im_start|>user\n{vid_prompt}<|im_end|>\n<|im_start|>assistant\n",  # noqa: E501
        num_video_frames=16,
        max_model_len=16384,
        hf_model_kwargs=model_utils.llava_onevision_hf_model_kwargs(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
        ),
        auto_cls=AutoModelForImageTextToText,
        vllm_output_post_proc=model_utils.llava_onevision_vllm_to_hf_output,
        custom_test_opts=[
            CustomTestOptions(
                inputs=custom_inputs.multi_video_multi_aspect_ratio_inputs(
                    formatter=lambda vid_prompt: f"<|im_start|>user\n{vid_prompt}<|im_end|>\n<|im_start|>assistant\n",  # noqa: E501
                ),
                limit_mm_per_prompt={"video": 4},
            )
        ],
    ),
    "llava_next_video": VLMTestInfo(
        models=["llava-hf/LLaVA-NeXT-Video-7B-hf"],
        test_type=VLMTestType.VIDEO,
        prompt_formatter=lambda vid_prompt: f"USER: {vid_prompt} ASSISTANT:",
        num_video_frames=16,
        max_model_len=4096,
        max_num_seqs=2,
        auto_cls=AutoModelForImageTextToText,
        vllm_output_post_proc=model_utils.llava_video_vllm_to_hf_output,
    ),
    "mantis": VLMTestInfo(
        models=["TIGER-Lab/Mantis-8B-siglip-llama3"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"<|start_header_id|>user<|end_header_id|>\n\n{img_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",  # noqa: E501
        max_model_len=4096,
        get_stop_token_ids=lambda tok: [128009],
        auto_cls=AutoModelForImageTextToText,
        vllm_output_post_proc=model_utils.mantis_vllm_to_hf_output,
        patch_hf_runner=model_utils.mantis_patch_hf_runner,
    ),
    "minicpmv_25": VLMTestInfo(
        models=["openbmb/MiniCPM-Llama3-V-2_5"],
        test_type=VLMTestType.IMAGE,
        prompt_formatter=lambda img_prompt: f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{img_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",  # noqa: E501
        img_idx_to_prompt=lambda idx: "(<image>./</image>)\n",
        max_model_len=4096,
        max_num_seqs=2,
        get_stop_token_ids=lambda tok: [tok.eos_id, tok.eot_id],
        hf_output_post_proc=model_utils.minicpmv_trunc_hf_output,
        patch_hf_runner=model_utils.minicpmv_25_patch_hf_runner,
        # FIXME: https://huggingface.co/openbmb/MiniCPM-V-2_6/discussions/55
        marks=[pytest.mark.skip("HF import fails")],
    ),
    "minicpmo_26": VLMTestInfo(
        models=["openbmb/MiniCPM-o-2_6"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{img_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",  # noqa: E501
        img_idx_to_prompt=lambda idx: "(<image>./</image>)\n",
        max_model_len=4096,
        max_num_seqs=2,
        get_stop_token_ids=lambda tok: tok.convert_tokens_to_ids(
            ["<|im_end|>", "<|endoftext|>"]
        ),
        hf_output_post_proc=model_utils.minicpmv_trunc_hf_output,
        patch_hf_runner=model_utils.minicpmo_26_patch_hf_runner,
        # FIXME: https://huggingface.co/openbmb/MiniCPM-o-2_6/discussions/49
        marks=[pytest.mark.skip("HF import fails")],
    ),
    "minicpmv_26": VLMTestInfo(
        models=["openbmb/MiniCPM-V-2_6"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{img_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",  # noqa: E501
        img_idx_to_prompt=lambda idx: "(<image>./</image>)\n",
        max_model_len=4096,
        max_num_seqs=2,
        get_stop_token_ids=lambda tok: tok.convert_tokens_to_ids(
            ["<|im_end|>", "<|endoftext|>"]
        ),
        hf_output_post_proc=model_utils.minicpmv_trunc_hf_output,
        patch_hf_runner=model_utils.minicpmv_26_patch_hf_runner,
    ),
    "minimax_vl_01": VLMTestInfo(
        models=["MiniMaxAI/MiniMax-VL-01"],
        prompt_formatter=lambda img_prompt: f"<beginning_of_sentence>user: {img_prompt} assistant:<end_of_sentence>",  # noqa: E501
        img_idx_to_prompt=lambda _: "<image>",
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        max_model_len=8192,
        max_num_seqs=4,
        dtype="bfloat16",
        hf_output_post_proc=model_utils.minimax_vl_01_hf_output,
        patch_hf_runner=model_utils.minimax_vl_01_patch_hf_runner,
        auto_cls=AutoModelForImageTextToText,
        marks=[
            large_gpu_mark(min_gb=80),
            # TODO: [ROCm] Fix pickle issue with ROCm spawn and tp>1
            pytest.mark.skipif(
                current_platform.is_rocm(),
                reason=(
                    "ROCm: Model too large for single GPU; "
                    "multi-GPU blocked by HF _LazyConfigMapping pickle issue with spawn"
                ),
            ),
        ],
    ),
    "molmo": VLMTestInfo(
        models=["allenai/Molmo-7B-D-0924"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=identity,
        max_model_len=4096,
        max_num_seqs=2,
        patch_hf_runner=model_utils.molmo_patch_hf_runner,
    ),
    "ovis1_6-gemma2": VLMTestInfo(
        models=["AIDC-AI/Ovis1.6-Gemma2-9B"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"<bos><start_of_turn>user\n{img_prompt}<end_of_turn>\n<start_of_turn>model\n",  # noqa: E501
        img_idx_to_prompt=lambda idx: "<image>\n",
        max_model_len=4096,
        max_num_seqs=2,
        dtype="half",
        # use sdpa mode for hf runner since ovis2 didn't work with flash_attn
        hf_model_kwargs={"llm_attn_implementation": "sdpa"},
        patch_hf_runner=model_utils.ovis_patch_hf_runner,
        marks=[large_gpu_mark(min_gb=32)],
    ),
    "ovis2": VLMTestInfo(
        models=["AIDC-AI/Ovis2-1B"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{img_prompt}<|im_end|>\n<|im_start|>assistant\n",  # noqa: E501
        img_idx_to_prompt=lambda idx: "<image>\n",
        max_model_len=4096,
        max_num_seqs=2,
        dtype="half",
        # use sdpa mode for hf runner since ovis2 didn't work with flash_attn
        hf_model_kwargs={"llm_attn_implementation": "sdpa"},
        patch_hf_runner=model_utils.ovis_patch_hf_runner,
    ),
    "ovis2_5": VLMTestInfo(
        models=["AIDC-AI/Ovis2.5-2B"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE, VLMTestType.VIDEO),
        prompt_formatter=lambda img_prompt: f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{img_prompt}<|im_end|>\n<|im_start|>assistant\n",  # noqa: E501
        img_idx_to_prompt=lambda idx: "<image>\n",
        video_idx_to_prompt=lambda idx: "<video>\n",
        max_model_len=4096,
        max_num_seqs=2,
        dtype="half",
        num_logprobs=10,
        patch_hf_runner=model_utils.ovis2_5_patch_hf_runner,
        hf_model_kwargs={"revision": "refs/pr/5"},
    ),
    "paddleocr_vl": VLMTestInfo(
        models=["PaddlePaddle/PaddleOCR-VL"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"USER: {img_prompt}\nASSISTANT:",
        img_idx_to_prompt=lambda idx: (
            "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>"
        ),
        multi_image_prompt=(
            "Image-1: <|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>\n"
            "Image-2: <|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>\n"
            "Describe these two images separately."
        ),
        max_model_len=8192,
        max_num_seqs=2,
        auto_cls=AutoModelForCausalLM,
        image_size_factors=[(0.25,)],
        marks=[
            pytest.mark.skipif(
                Version(TRANSFORMERS_VERSION) == Version("4.57.3"),
                reason="This model is broken in Transformers v4.57.3",
            )
        ],
    ),
    "phi3v": VLMTestInfo(
        models=["microsoft/Phi-3.5-vision-instruct"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"<|user|>\n{img_prompt}<|end|>\n<|assistant|>\n",  # noqa: E501
        img_idx_to_prompt=lambda idx: f"<|image_{idx}|>\n",
        max_model_len=4096,
        max_num_seqs=2,
        runner="generate",
        # use sdpa mode for hf runner since phi3v didn't work with flash_attn
        hf_model_kwargs={"_attn_implementation": "sdpa"},
        use_tokenizer_eos=True,
        vllm_output_post_proc=model_utils.phi3v_vllm_to_hf_output,
        num_logprobs=10,
    ),
    "pixtral_hf": VLMTestInfo(
        models=["nm-testing/pixtral-12b-FP8-dynamic"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"<s>[INST]{img_prompt}[/INST]",
        img_idx_to_prompt=lambda idx: "[IMG]",
        max_model_len=8192,
        max_num_seqs=2,
        auto_cls=AutoModelForImageTextToText,
        marks=[
            large_gpu_mark(min_gb=48),
            pytest.mark.skipif(
                current_platform.is_rocm(),
                reason="Model produces a vector of <UNK> output in HF on ROCm",
            ),
        ],
    ),
    "qwen_vl": VLMTestInfo(
        models=["Qwen/Qwen-VL"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=identity,
        img_idx_to_prompt=lambda idx: f"Picture {idx}: <img></img>\n",
        max_model_len=1024,
        max_num_seqs=2,
        vllm_output_post_proc=model_utils.qwen_vllm_to_hf_output,
        prompt_path_encoder=model_utils.qwen_prompt_path_encoder,
    ),
    "qwen2_vl": VLMTestInfo(
        models=["Qwen/Qwen2-VL-2B-Instruct"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE, VLMTestType.VIDEO),
        prompt_formatter=lambda img_prompt: f"<|im_start|>User\n{img_prompt}<|im_end|>\n<|im_start|>assistant\n",  # noqa: E501
        img_idx_to_prompt=lambda idx: "<|vision_start|><|image_pad|><|vision_end|>",
        video_idx_to_prompt=lambda idx: "<|vision_start|><|video_pad|><|vision_end|>",
        multi_image_prompt="Picture 1: <vlm_image>\nPicture 2: <vlm_image>\nDescribe these two images with one paragraph respectively.",  # noqa: E501
        max_model_len=4096,
        max_num_seqs=2,
        auto_cls=AutoModelForImageTextToText,
        vllm_output_post_proc=model_utils.qwen2_vllm_to_hf_output,
        image_size_factors=[(0.25,), (0.25, 0.25, 0.25), (0.25, 0.2, 0.15)],
        marks=[pytest.mark.cpu_model],
    ),
    "skywork_r1v": VLMTestInfo(
        models=["Skywork/Skywork-R1V-38B"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"<｜begin▁of▁sentence｜><｜User｜>\n{img_prompt}<｜Assistant｜><think>\n",  # noqa: E501
        single_image_prompts=IMAGE_ASSETS.prompts(
            {
                "stop_sign": "<image>\nWhat's the content in the center of the image?",
                "cherry_blossom": "<image>\nWhat is the season?",
            }
        ),
        multi_image_prompt="<image>\n<image>\nDescribe the two images in short.",
        max_model_len=4096,
        use_tokenizer_eos=True,
        patch_hf_runner=model_utils.skyworkr1v_patch_hf_runner,
        marks=[large_gpu_mark(min_gb=80)],
    ),
    "smolvlm": VLMTestInfo(
        models=["HuggingFaceTB/SmolVLM2-2.2B-Instruct"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"<|im_start|>User:{img_prompt}<end_of_utterance>\nAssistant:",  # noqa: E501
        img_idx_to_prompt=lambda idx: "<image>",
        max_model_len=8192,
        max_num_seqs=2,
        auto_cls=AutoModelForImageTextToText,
        hf_output_post_proc=model_utils.smolvlm_trunc_hf_output,
        num_logprobs=10,
    ),
    "tarsier": VLMTestInfo(
        models=["omni-research/Tarsier-7b"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"USER: {img_prompt} ASSISTANT:",
        max_model_len=4096,
        max_num_seqs=2,
        auto_cls=AutoModelForImageTextToText,
        patch_hf_runner=model_utils.tarsier_patch_hf_runner,
    ),
    "tarsier2": VLMTestInfo(
        models=["omni-research/Tarsier2-Recap-7b"],
        test_type=(
            VLMTestType.IMAGE,
            VLMTestType.MULTI_IMAGE,
            VLMTestType.VIDEO,
        ),
        prompt_formatter=lambda img_prompt: f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{img_prompt}<|im_end|>\n<|im_start|>assistant\n",  # noqa: E501
        img_idx_to_prompt=lambda idx: "<|vision_start|><|image_pad|><|vision_end|>",
        video_idx_to_prompt=lambda idx: "<|vision_start|><|video_pad|><|vision_end|>",
        max_model_len=4096,
        max_num_seqs=2,
        auto_cls=AutoModelForImageTextToText,
        image_size_factors=[(0.25,), (0.25, 0.25, 0.25), (0.25, 0.2, 0.15)],
        marks=[pytest.mark.skip("Model initialization hangs")],
    ),
    ### Tensor parallel / multi-gpu broadcast tests
    "chameleon-broadcast": VLMTestInfo(
        models=["facebook/chameleon-7b"],
        prompt_formatter=lambda img_prompt: f"USER: {img_prompt}\nASSISTANT:",
        max_model_len=4096,
        auto_cls=AutoModelForImageTextToText,
        vllm_output_post_proc=lambda vllm_output, model: vllm_output[:2],
        hf_output_post_proc=lambda hf_output, model: hf_output[:2],
        comparator=check_outputs_equal,
        marks=multi_gpu_marks(num_gpus=2),
        **COMMON_BROADCAST_SETTINGS,  # type: ignore
    ),
    "llava-broadcast": VLMTestInfo(
        models=["llava-hf/llava-1.5-7b-hf"],
        prompt_formatter=lambda img_prompt: f"USER: {img_prompt}\nASSISTANT:",
        max_model_len=4096,
        auto_cls=AutoModelForImageTextToText,
        vllm_output_post_proc=model_utils.llava_image_vllm_to_hf_output,
        marks=multi_gpu_marks(num_gpus=2),
        **COMMON_BROADCAST_SETTINGS,  # type: ignore
    ),
    "llava_next-broadcast": VLMTestInfo(
        models=["llava-hf/llava-v1.6-mistral-7b-hf"],
        prompt_formatter=lambda img_prompt: f"[INST] {img_prompt} [/INST]",
        max_model_len=10240,
        auto_cls=AutoModelForImageTextToText,
        vllm_output_post_proc=model_utils.llava_image_vllm_to_hf_output,
        marks=multi_gpu_marks(num_gpus=2),
        **COMMON_BROADCAST_SETTINGS,  # type: ignore
    ),
    ### Custom input edge-cases for specific models
    "intern_vl-diff-patches": VLMTestInfo(
        models=["OpenGVLab/InternVL2-2B"],
        prompt_formatter=lambda img_prompt: f"<|im_start|>User\n{img_prompt}<|im_end|>\n<|im_start|>Assistant\n",  # noqa: E501
        test_type=VLMTestType.CUSTOM_INPUTS,
        max_model_len=4096,
        use_tokenizer_eos=True,
        patch_hf_runner=model_utils.internvl_patch_hf_runner,
        custom_test_opts=[
            CustomTestOptions(
                inputs=inp,
                limit_mm_per_prompt={"image": 2},
            )
            for inp in custom_inputs.different_patch_input_cases_internvl()
        ],
    ),
    "llava_onevision-multiple-images": VLMTestInfo(
        models=["llava-hf/llava-onevision-qwen2-0.5b-ov-hf"],
        test_type=VLMTestType.CUSTOM_INPUTS,
        max_model_len=16384,
        max_num_seqs=2,
        auto_cls=AutoModelForImageTextToText,
        hf_model_kwargs=model_utils.llava_onevision_hf_model_kwargs(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
        ),
        vllm_output_post_proc=model_utils.llava_onevision_vllm_to_hf_output,
        custom_test_opts=[
            CustomTestOptions(
                inputs=custom_inputs.multi_image_multi_aspect_ratio_inputs(
                    formatter=lambda vid_prompt: f"<|im_start|>user\n{vid_prompt}<|im_end|>\n<|im_start|>assistant\n",  # noqa: E501
                ),
                limit_mm_per_prompt={"image": 4},
            )
        ],
        marks=[
            pytest.mark.skipif(
                Version(TRANSFORMERS_VERSION) == Version("4.57.1"),
                reason="This model is broken in Transformers v4.57.1",
            )
        ],
    ),
    # regression test for https://github.com/vllm-project/vllm/issues/15122
    "qwen2_5_vl-windows-attention": VLMTestInfo(
        models=["Qwen/Qwen2.5-VL-3B-Instruct"],
        test_type=VLMTestType.CUSTOM_INPUTS,
        max_model_len=4096,
        max_num_seqs=2,
        auto_cls=AutoModelForImageTextToText,
        vllm_output_post_proc=model_utils.qwen2_vllm_to_hf_output,
        custom_test_opts=[
            CustomTestOptions(
                inputs=custom_inputs.windows_attention_image_qwen2_5_vl(),
                limit_mm_per_prompt={"image": 1},
            )
        ],
    ),
}


def _mark_splits(
    test_settings: dict[str, VLMTestInfo],
    *,
    num_groups: int,
) -> dict[str, VLMTestInfo]:
    name_by_test_info_id = {id(v): k for k, v in test_settings.items()}
    test_infos_by_model = defaultdict[str, list[VLMTestInfo]](list)

    for info in test_settings.values():
        for model in info.models:
            test_infos_by_model[model].append(info)

    models = sorted(test_infos_by_model.keys())
    split_size = math.ceil(len(models) / num_groups)

    new_test_settings = dict[str, VLMTestInfo]()

    for i in range(num_groups):
        models_in_group = models[i * split_size : (i + 1) * split_size]

        for model in models_in_group:
            for info in test_infos_by_model[model]:
                new_marks = (info.marks or []) + [pytest.mark.split(group=i)]
                new_info = info._replace(marks=new_marks)
                new_test_settings[name_by_test_info_id[id(info)]] = new_info

    missing_keys = test_settings.keys() - new_test_settings.keys()
    assert not missing_keys, f"Missing keys: {missing_keys}"

    return new_test_settings


VLM_TEST_SETTINGS = _mark_splits(VLM_TEST_SETTINGS, num_groups=2)


### Test wrappers
# Wrappers around the core test running func for:
# - single image
# - multi-image
# - image embeddings
# - video
# - audio
# - custom inputs
@pytest.mark.parametrize(
    "model_type,test_case",
    get_parametrized_options(
        VLM_TEST_SETTINGS,
        test_type=VLMTestType.IMAGE,
        create_new_process_for_each_test=False,
    ),
)
def test_single_image_models(
    tmp_path: PosixPath,
    model_type: str,
    test_case: ExpandableVLMTestArgs,
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    image_assets: ImageTestAssets,
):
    model_test_info = VLM_TEST_SETTINGS[model_type]
    runners.run_single_image_test(
        tmp_path=tmp_path,
        model_test_info=model_test_info,
        test_case=test_case,
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        image_assets=image_assets,
    )


@pytest.mark.parametrize(
    "model_type,test_case",
    get_parametrized_options(
        VLM_TEST_SETTINGS,
        test_type=VLMTestType.MULTI_IMAGE,
        create_new_process_for_each_test=False,
    ),
)
def test_multi_image_models(
    tmp_path: PosixPath,
    model_type: str,
    test_case: ExpandableVLMTestArgs,
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    image_assets: ImageTestAssets,
):
    model_test_info = VLM_TEST_SETTINGS[model_type]
    runners.run_multi_image_test(
        tmp_path=tmp_path,
        model_test_info=model_test_info,
        test_case=test_case,
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        image_assets=image_assets,
    )


@pytest.mark.parametrize(
    "model_type,test_case",
    get_parametrized_options(
        VLM_TEST_SETTINGS,
        test_type=VLMTestType.EMBEDDING,
        create_new_process_for_each_test=False,
    ),
)
def test_image_embedding_models(
    model_type: str,
    test_case: ExpandableVLMTestArgs,
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    image_assets: ImageTestAssets,
):
    model_test_info = VLM_TEST_SETTINGS[model_type]
    runners.run_embedding_test(
        model_test_info=model_test_info,
        test_case=test_case,
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        image_assets=image_assets,
    )


@pytest.mark.parametrize(
    "model_type,test_case",
    get_parametrized_options(
        VLM_TEST_SETTINGS,
        test_type=VLMTestType.VIDEO,
        create_new_process_for_each_test=False,
    ),
)
def test_video_models(
    model_type: str,
    test_case: ExpandableVLMTestArgs,
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    video_assets: VideoTestAssets,
):
    model_test_info = VLM_TEST_SETTINGS[model_type]
    runners.run_video_test(
        model_test_info=model_test_info,
        test_case=test_case,
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        video_assets=video_assets,
    )


@pytest.mark.parametrize(
    "model_type,test_case",
    get_parametrized_options(
        VLM_TEST_SETTINGS,
        test_type=VLMTestType.AUDIO,
        create_new_process_for_each_test=False,
    ),
)
def test_audio_models(
    model_type: str,
    test_case: ExpandableVLMTestArgs,
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    audio_assets: AudioTestAssets,
):
    model_test_info = VLM_TEST_SETTINGS[model_type]
    runners.run_audio_test(
        model_test_info=model_test_info,
        test_case=test_case,
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        audio_assets=audio_assets,
    )


@pytest.mark.parametrize(
    "model_type,test_case",
    get_parametrized_options(
        VLM_TEST_SETTINGS,
        test_type=VLMTestType.CUSTOM_INPUTS,
        create_new_process_for_each_test=False,
    ),
)
def test_custom_inputs_models(
    model_type: str,
    test_case: ExpandableVLMTestArgs,
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
):
    model_test_info = VLM_TEST_SETTINGS[model_type]
    runners.run_custom_inputs_test(
        model_test_info=model_test_info,
        test_case=test_case,
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
    )


#### Tests filtering for things running each test as a new process
@pytest.mark.parametrize(
    "model_type,test_case",
    get_parametrized_options(
        VLM_TEST_SETTINGS,
        test_type=VLMTestType.IMAGE,
        create_new_process_for_each_test=True,
    ),
)
@create_new_process_for_each_test()
def test_single_image_models_heavy(
    tmp_path: PosixPath,
    model_type: str,
    test_case: ExpandableVLMTestArgs,
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    image_assets: ImageTestAssets,
):
    model_test_info = VLM_TEST_SETTINGS[model_type]
    runners.run_single_image_test(
        tmp_path=tmp_path,
        model_test_info=model_test_info,
        test_case=test_case,
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        image_assets=image_assets,
    )


@pytest.mark.parametrize(
    "model_type,test_case",
    get_parametrized_options(
        VLM_TEST_SETTINGS,
        test_type=VLMTestType.MULTI_IMAGE,
        create_new_process_for_each_test=True,
    ),
)
@create_new_process_for_each_test()
def test_multi_image_models_heavy(
    tmp_path: PosixPath,
    model_type: str,
    test_case: ExpandableVLMTestArgs,
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    image_assets: ImageTestAssets,
):
    model_test_info = VLM_TEST_SETTINGS[model_type]
    runners.run_multi_image_test(
        tmp_path=tmp_path,
        model_test_info=model_test_info,
        test_case=test_case,
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        image_assets=image_assets,
    )


@pytest.mark.parametrize(
    "model_type,test_case",
    get_parametrized_options(
        VLM_TEST_SETTINGS,
        test_type=VLMTestType.EMBEDDING,
        create_new_process_for_each_test=True,
    ),
)
@create_new_process_for_each_test()
def test_image_embedding_models_heavy(
    model_type: str,
    test_case: ExpandableVLMTestArgs,
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    image_assets: ImageTestAssets,
):
    model_test_info = VLM_TEST_SETTINGS[model_type]
    runners.run_embedding_test(
        model_test_info=model_test_info,
        test_case=test_case,
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        image_assets=image_assets,
    )


@pytest.mark.parametrize(
    "model_type,test_case",
    get_parametrized_options(
        VLM_TEST_SETTINGS,
        test_type=VLMTestType.VIDEO,
        create_new_process_for_each_test=True,
    ),
)
def test_video_models_heavy(
    model_type: str,
    test_case: ExpandableVLMTestArgs,
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    video_assets: VideoTestAssets,
):
    model_test_info = VLM_TEST_SETTINGS[model_type]
    runners.run_video_test(
        model_test_info=model_test_info,
        test_case=test_case,
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        video_assets=video_assets,
    )


@pytest.mark.parametrize(
    "model_type,test_case",
    get_parametrized_options(
        VLM_TEST_SETTINGS,
        test_type=VLMTestType.AUDIO,
        create_new_process_for_each_test=True,
    ),
)
def test_audio_models_heavy(
    model_type: str,
    test_case: ExpandableVLMTestArgs,
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    audio_assets: AudioTestAssets,
):
    model_test_info = VLM_TEST_SETTINGS[model_type]
    runners.run_audio_test(
        model_test_info=model_test_info,
        test_case=test_case,
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        audio_assets=audio_assets,
    )


@pytest.mark.parametrize(
    "model_type,test_case",
    get_parametrized_options(
        VLM_TEST_SETTINGS,
        test_type=VLMTestType.CUSTOM_INPUTS,
        create_new_process_for_each_test=True,
    ),
)
@create_new_process_for_each_test()
def test_custom_inputs_models_heavy(
    model_type: str,
    test_case: ExpandableVLMTestArgs,
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
):
    model_test_info = VLM_TEST_SETTINGS[model_type]
    runners.run_custom_inputs_test(
        model_test_info=model_test_info,
        test_case=test_case,
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
    )
