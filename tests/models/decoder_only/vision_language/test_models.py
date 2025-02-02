"""Common tests for testing .generate() functionality for single / multiple
image, embedding, and video support for different VLMs in vLLM.
"""
import math
import os
from collections import defaultdict
from pathlib import PosixPath
from typing import Type

import pytest
from transformers import AutoModelForVision2Seq
from transformers import __version__ as TRANSFORMERS_VERSION

from vllm.platforms import current_platform
from vllm.utils import identity

from ....conftest import (IMAGE_ASSETS, HfRunner, VllmRunner, _ImageAssets,
                          _VideoAssets)
from ....utils import (fork_new_process_for_each_test, large_gpu_mark,
                       multi_gpu_marks)
from ...utils import check_outputs_equal
from .vlm_utils import custom_inputs, model_utils, runners
from .vlm_utils.case_filtering import get_parametrized_options
from .vlm_utils.types import (CustomTestOptions, ExpandableVLMTestArgs,
                              VLMTestInfo, VLMTestType)

# This hack is needed for phi3v & paligemma models
# ROCm Triton FA can run into shared memory issues with these models,
# use other backends in the meantime
# FIXME (mattwong, gshtrasb, hongxiayan)
if current_platform.is_rocm():
    os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"

# yapf: disable
COMMON_BROADCAST_SETTINGS = {
    "test_type": VLMTestType.IMAGE,
    "dtype": "half",
    "max_tokens": 5,
    "tensor_parallel_size": 2,
    "hf_model_kwargs": {"device_map": "auto"},
    "image_size_factors": [(.25, 0.5, 1.0)],
    "distributed_executor_backend": (
        "ray",
        "mp",
    )
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
        test_type=(
            VLMTestType.EMBEDDING,
            VLMTestType.IMAGE,
            VLMTestType.CUSTOM_INPUTS
        ),
        prompt_formatter=lambda img_prompt: f"USER: {img_prompt}\nASSISTANT:",
        convert_assets_to_embeddings=model_utils.get_llava_embeddings,
        max_model_len=4096,
        auto_cls=AutoModelForVision2Seq,
        vllm_output_post_proc=model_utils.llava_image_vllm_to_hf_output,
        custom_test_opts=[CustomTestOptions(
            inputs=custom_inputs.multi_image_multi_aspect_ratio_inputs(
                formatter=lambda img_prompt: f"USER: {img_prompt}\nASSISTANT:"
            ),
            limit_mm_per_prompt={"image": 4},
        )],
        marks=[pytest.mark.core_model, pytest.mark.cpu_model],
    ),
    "paligemma": VLMTestInfo(
        models=["google/paligemma-3b-mix-224"],
        test_type=VLMTestType.IMAGE,
        prompt_formatter=identity,
        img_idx_to_prompt = lambda idx: "",
        # Paligemma uses its own sample prompts because the default one fails
        single_image_prompts=IMAGE_ASSETS.prompts({
            "stop_sign": "caption es",
            "cherry_blossom": "What is in the picture?",
        }),
        auto_cls=AutoModelForVision2Seq,
        postprocess_inputs=model_utils.cast_dtype_post_processor(
            "pixel_values"
        ),
        vllm_output_post_proc=model_utils.paligemma_vllm_to_hf_output,
        dtype=("half" if current_platform.is_cpu() or current_platform.is_rocm()
               else ("half", "float")),
        marks=[pytest.mark.core_model],
    ),
    "qwen2_vl": VLMTestInfo(
        models=["Qwen/Qwen2-VL-2B-Instruct"],
        test_type=(
            VLMTestType.IMAGE,
            VLMTestType.MULTI_IMAGE,
            VLMTestType.VIDEO
        ),
        prompt_formatter=lambda img_prompt: f"<|im_start|>User\n{img_prompt}<|im_end|>\n<|im_start|>assistant\n", # noqa: E501
        img_idx_to_prompt=lambda idx: "<|vision_start|><|image_pad|><|vision_end|>", # noqa: E501
        video_idx_to_prompt=lambda idx: "<|vision_start|><|video_pad|><|vision_end|>", # noqa: E501
        max_model_len=4096,
        max_num_seqs=2,
        auto_cls=AutoModelForVision2Seq,
        vllm_output_post_proc=model_utils.qwen2_vllm_to_hf_output,
        image_size_factors=[(), (0.25,), (0.25, 0.25, 0.25), (0.25, 0.2, 0.15)],
        marks=[pytest.mark.core_model, pytest.mark.cpu_model],
    ),
    #### Extended model tests
    "aria": VLMTestInfo(
        models=["rhymes-ai/Aria"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"<|im_start|>user\n{img_prompt}<|im_end|>\n<|im_start|>assistant\n ", # noqa: E501
        img_idx_to_prompt=lambda idx: "<fim_prefix><|img|><fim_suffix>\n",
        max_model_len=4096,
        max_num_seqs=2,
        single_image_prompts=IMAGE_ASSETS.prompts({
            "stop_sign": "<vlm_image>Please describe the image shortly.",
            "cherry_blossom": "<vlm_image>Please infer the season with reason.",
        }),
        multi_image_prompt="<vlm_image><vlm_image>Describe the two images shortly.",    # noqa: E501
        postprocess_inputs=model_utils.cast_dtype_post_processor("pixel_values"),
        stop_str=["<|im_end|>"],
        image_size_factors=[(0.10, 0.15)],
        max_tokens=64,
        marks=[
            pytest.mark.skipif(
                TRANSFORMERS_VERSION < "4.48.0",
                reason="HF model requires transformers>=4.48.0",
            ),
            large_gpu_mark(min_gb=64),
        ],
    ),
    "blip2": VLMTestInfo(
        models=["Salesforce/blip2-opt-2.7b"],
        test_type=VLMTestType.IMAGE,
        prompt_formatter=lambda img_prompt: f"Question: {img_prompt} Answer:",
        img_idx_to_prompt=lambda idx: "",
        auto_cls=AutoModelForVision2Seq,
        vllm_output_post_proc=model_utils.blip2_vllm_to_hf_output,
    ),
    "chameleon": VLMTestInfo(
        models=["facebook/chameleon-7b"],
        test_type=VLMTestType.IMAGE,
        prompt_formatter=lambda img_prompt: f"USER: {img_prompt}\nASSISTANT:",
        max_model_len=4096,
        max_num_seqs=2,
        auto_cls=AutoModelForVision2Seq,
        postprocess_inputs=model_utils.cast_dtype_post_processor(
            "pixel_values"
        ),
        # For chameleon, we only compare the sequences
        vllm_output_post_proc = lambda vllm_output, model: vllm_output[:2],
        hf_output_post_proc = lambda hf_output, model: hf_output[:2],
        comparator=check_outputs_equal,
        max_tokens=8,
        dtype="bfloat16",
    ),
    "deepseek_vl_v2": VLMTestInfo(
        models=["Isotr0py/deepseek-vl2-tiny"], # model repo using dynamic module
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"<|User|>: {img_prompt}\n\n<|Assistant|>: ", # noqa: E501
        max_model_len=4096,
        max_num_seqs=2,
        single_image_prompts=IMAGE_ASSETS.prompts({
            "stop_sign": "<image>\nWhat's the content in the center of the image?", # noqa: E501
            "cherry_blossom": "<image>\nPlease infer the season with reason in details.",   # noqa: E501
        }),
        multi_image_prompt="image_1:<image>\nimage_2:<image>\nWhich image can we see the car and the tower?",    # noqa: E501
        vllm_runner_kwargs={"hf_overrides": {"architectures": ["DeepseekVLV2ForCausalLM"]}},  # noqa: E501
        patch_hf_runner=model_utils.deepseekvl2_patch_hf_runner,
        postprocess_inputs=model_utils.cast_dtype_post_processor("images"),
        hf_output_post_proc=model_utils.deepseekvl2_trunc_hf_output,
        stop_str=["<｜end▁of▁sentence｜>", "<｜begin▁of▁sentence｜>"],  # noqa: E501
        image_size_factors=[(), (1.0, ), (1.0, 1.0, 1.0), (0.1, 0.5, 1.0)],
        marks=[
            pytest.mark.skipif(
                TRANSFORMERS_VERSION >= "4.48.0",
                reason="HF model is not compatible with transformers>=4.48.0",
            )
        ],
    ),
    "fuyu": VLMTestInfo(
        models=["adept/fuyu-8b"],
        test_type=VLMTestType.IMAGE,
        prompt_formatter=lambda img_prompt: f"{img_prompt}\n",
        img_idx_to_prompt=lambda idx: "",
        max_model_len=2048,
        max_num_seqs=2,
        use_tokenizer_eos=True,
        vllm_output_post_proc=model_utils.fuyu_vllm_to_hf_output,
        num_logprobs=10,
        image_size_factors=[(), (0.25,), (0.25, 0.25, 0.25), (0.25, 0.2, 0.15)],
    ),
    "glm4": VLMTestInfo(
        models=["THUDM/glm-4v-9b"],
        test_type=VLMTestType.IMAGE,
        prompt_formatter=identity,
        img_idx_to_prompt=lambda idx: "",
        max_model_len=2048,
        max_num_seqs=2,
        dtype="bfloat16",
        get_stop_token_ids=lambda tok: [151329, 151336, 151338],
        patch_hf_runner=model_utils.glm_patch_hf_runner,
        marks=[large_gpu_mark(min_gb=32)],
    ),
    "h2ovl": VLMTestInfo(
        models = [
            "h2oai/h2ovl-mississippi-800m",
            "h2oai/h2ovl-mississippi-2b",
        ],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"<|prompt|>{img_prompt}<|end|><|answer|>", # noqa: E501
        single_image_prompts=IMAGE_ASSETS.prompts({
            "stop_sign": "<image>\nWhat's the content in the center of the image?",  # noqa: E501
            "cherry_blossom": "<image>\nWhat is the season?",
        }),
        multi_image_prompt="Image-1: <image>\nImage-2: <image>\nDescribe the two images in short.",  # noqa: E501
        max_model_len=8192,
        dtype="bfloat16",
        use_tokenizer_eos=True,
        patch_hf_runner=model_utils.h2ovl_patch_hf_runner,
    ),
    "idefics3": VLMTestInfo(
        models=["HuggingFaceM4/Idefics3-8B-Llama3"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt:f"<|begin_of_text|>User:{img_prompt}<end_of_utterance>\nAssistant:",  # noqa: E501
        img_idx_to_prompt=lambda idx: "<image>",
        max_model_len=8192,
        max_num_seqs=2,
        auto_cls=AutoModelForVision2Seq,
        marks=[large_gpu_mark(min_gb=48)],
    ),
    "intern_vl": VLMTestInfo(
        models=[
            "OpenGVLab/InternVL2-1B",
            "OpenGVLab/InternVL2-2B",
            "OpenGVLab/Mono-InternVL-2B",
        ],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"<|im_start|>User\n{img_prompt}<|im_end|>\n<|im_start|>Assistant\n", # noqa: E501
        single_image_prompts=IMAGE_ASSETS.prompts({
            "stop_sign": "<image>\nWhat's the content in the center of the image?",  # noqa: E501
            "cherry_blossom": "<image>\nWhat is the season?",
        }),
        multi_image_prompt="Image-1: <image>\nImage-2: <image>\nDescribe the two images in short.",  # noqa: E501
        max_model_len=4096,
        # NOTE: Mono-InternVL-2B doesn't work with fp16,
        # it will result NaN during inference.
        # See: https://huggingface.co/OpenGVLab/Mono-InternVL-2B/discussions/9
        dtype="bfloat16",
        use_tokenizer_eos=True,
        patch_hf_runner=model_utils.internvl_patch_hf_runner,
        marks=[large_gpu_mark(min_gb=32)],
    ),
    "llava_next": VLMTestInfo(
        models=["llava-hf/llava-v1.6-mistral-7b-hf"],
        test_type=(VLMTestType.IMAGE, VLMTestType.CUSTOM_INPUTS),
        prompt_formatter=lambda img_prompt: f"[INST] {img_prompt} [/INST]",
        max_model_len=10240,
        auto_cls=AutoModelForVision2Seq,
        vllm_output_post_proc=model_utils.llava_image_vllm_to_hf_output,
        custom_test_opts=[CustomTestOptions(
            inputs=custom_inputs.multi_image_multi_aspect_ratio_inputs(
                formatter=lambda img_prompt: f"[INST] {img_prompt} [/INST]"
            ),
            limit_mm_per_prompt={"image": 4},
        )],
    ),
    "llava_onevision": VLMTestInfo(
        models=["llava-hf/llava-onevision-qwen2-0.5b-ov-hf"],
        test_type=VLMTestType.CUSTOM_INPUTS,
        prompt_formatter=lambda vid_prompt: f"<|im_start|>user\n{vid_prompt}<|im_end|>\n<|im_start|>assistant\n",   # noqa: E501
        num_video_frames=16,
        max_model_len=16384,
        postprocess_inputs=model_utils.cast_dtype_post_processor(
            "pixel_values_videos"
        ),
        auto_cls=AutoModelForVision2Seq,
        vllm_output_post_proc=model_utils.llava_onevision_vllm_to_hf_output,
        custom_test_opts=[CustomTestOptions(
            inputs=custom_inputs.multi_video_multi_aspect_ratio_inputs(
                formatter=lambda vid_prompt: f"<|im_start|>user\n{vid_prompt}<|im_end|>\n<|im_start|>assistant\n",   # noqa: E501
            ),
            limit_mm_per_prompt={"video": 4},
            runner_mm_key="videos",
        )],
    ),
    "llava_next_video": VLMTestInfo(
        models=["llava-hf/LLaVA-NeXT-Video-7B-hf"],
        test_type=VLMTestType.VIDEO,
        prompt_formatter=lambda vid_prompt: f"USER: {vid_prompt} ASSISTANT:",
        num_video_frames=16,
        max_model_len=4096,
        auto_cls=AutoModelForVision2Seq,
        vllm_output_post_proc=model_utils.llava_video_vllm_to_hf_output,
    ),
    "mantis": VLMTestInfo(
        models=["TIGER-Lab/Mantis-8B-siglip-llama3"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"<|start_header_id|>user<|end_header_id|>\n\n{img_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",  # noqa: E501
        max_model_len=4096,
        postprocess_inputs=model_utils.cast_dtype_post_processor(
            "pixel_values"
        ),
        vllm_runner_kwargs={"hf_overrides": {"architectures": ["MantisForConditionalGeneration"]}},  # noqa: E501
        get_stop_token_ids=lambda tok: [128009],
        auto_cls=AutoModelForVision2Seq,
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
        postprocess_inputs=model_utils.wrap_inputs_post_processor,
        hf_output_post_proc=model_utils.minicpmv_trunc_hf_output,
    ),
    "minicpmo_26": VLMTestInfo(
        models=["openbmb/MiniCPM-o-2_6"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{img_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",  # noqa: E501
        img_idx_to_prompt=lambda idx: "(<image>./</image>)\n",
        max_model_len=4096,
        max_num_seqs=2,
        get_stop_token_ids=lambda tok: tok.convert_tokens_to_ids(['<|im_end|>', '<|endoftext|>']),  # noqa: E501
        postprocess_inputs=model_utils.ignore_inputs_post_processor(
            "image_sizes"
        ),
        hf_output_post_proc=model_utils.minicpmv_trunc_hf_output,
        patch_hf_runner=model_utils.minicpmo_patch_hf_runner
    ),
    "minicpmv_26": VLMTestInfo(
        models=["openbmb/MiniCPM-V-2_6"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{img_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",  # noqa: E501
        img_idx_to_prompt=lambda idx: "(<image>./</image>)\n",
        max_model_len=4096,
        max_num_seqs=2,
        get_stop_token_ids=lambda tok: tok.convert_tokens_to_ids(['<|im_end|>', '<|endoftext|>']),  # noqa: E501
        postprocess_inputs=model_utils.ignore_inputs_post_processor(
            "image_sizes"
        ),
        hf_output_post_proc=model_utils.minicpmv_trunc_hf_output,
    ),
    "molmo": VLMTestInfo(
        models=["allenai/Molmo-7B-D-0924"],
        test_type=(VLMTestType.IMAGE),
        prompt_formatter=lambda img_prompt:"User: " + img_prompt + " Assistant:", # noqa: E501
        max_model_len=4096,
        max_num_seqs=2,
        image_size_factors=[(),(1.0, 1.0, 1.0)],
        patch_hf_runner=model_utils.mlomo_patch_hf_runner,
        postprocess_inputs=model_utils.molmo_post_processor,
    ),
    # Tests for phi3v currently live in another file because of a bug in
    # transformers. Once this issue is fixed, we can enable them here instead.
    # https://github.com/huggingface/transformers/issues/34307
    # "phi3v": VLMTestInfo(
    #     models=["microsoft/Phi-3.5-vision-instruct"],
    #     test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
    #     prompt_formatter=lambda img_prompt: f"<|user|>\n{img_prompt}<|end|>\n<|assistant|>\n", # noqa: E501
    #     img_idx_to_prompt=lambda idx: f"<|image_{idx}|>\n",
    #     max_model_len=4096,
    #     max_num_seqs=2,
    #     task="generate",
    #     # use eager mode for hf runner since phi3v didn't work with flash_attn
    #     hf_model_kwargs={"_attn_implementation": "eager"},
    #     use_tokenizer_eos=True,
    #     vllm_output_post_proc=model_utils.phi3v_vllm_to_hf_output,
    #     num_logprobs=10,
    # ),
    "pixtral_hf": VLMTestInfo(
        models=["nm-testing/pixtral-12b-FP8-dynamic"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=lambda img_prompt: f"<s>[INST]{img_prompt}[/INST]",
        img_idx_to_prompt=lambda idx: "[IMG]",
        max_model_len=8192,
        max_num_seqs=2,
        auto_cls=AutoModelForVision2Seq,
        marks=[large_gpu_mark(min_gb=48)],
    ),
    "qwen": VLMTestInfo(
        models=["Qwen/Qwen-VL"],
        test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
        prompt_formatter=identity,
        img_idx_to_prompt=lambda idx: f"Picture {idx}: <img></img>\n",
        max_model_len=1024,
        max_num_seqs=2,
        vllm_output_post_proc=model_utils.qwen_vllm_to_hf_output,
        prompt_path_encoder=model_utils.qwen_prompt_path_encoder,
    ),
    ### Tensor parallel / multi-gpu broadcast tests
    "chameleon-broadcast": VLMTestInfo(
        models=["facebook/chameleon-7b"],
        prompt_formatter=lambda img_prompt: f"USER: {img_prompt}\nASSISTANT:",
        max_model_len=4096,
        auto_cls=AutoModelForVision2Seq,
        postprocess_inputs=model_utils.cast_dtype_post_processor(
            "pixel_values"
        ),
        vllm_output_post_proc = lambda vllm_output, model: vllm_output[:2],
        hf_output_post_proc = lambda hf_output, model: hf_output[:2],
        comparator=check_outputs_equal,
        marks=multi_gpu_marks(num_gpus=2),
        **COMMON_BROADCAST_SETTINGS # type: ignore
    ),
    "llava-broadcast": VLMTestInfo(
        models=["llava-hf/llava-1.5-7b-hf"],
        prompt_formatter=lambda img_prompt: f"USER: {img_prompt}\nASSISTANT:",
        max_model_len=4096,
        auto_cls=AutoModelForVision2Seq,
        vllm_output_post_proc=model_utils.llava_image_vllm_to_hf_output,
        marks=multi_gpu_marks(num_gpus=2),
        **COMMON_BROADCAST_SETTINGS # type: ignore
    ),
    "llava_next-broadcast": VLMTestInfo(
        models=["llava-hf/llava-v1.6-mistral-7b-hf"],
        prompt_formatter=lambda img_prompt: f"[INST] {img_prompt} [/INST]",
        max_model_len=10240,
        auto_cls=AutoModelForVision2Seq,
        vllm_output_post_proc=model_utils.llava_image_vllm_to_hf_output,
        marks=multi_gpu_marks(num_gpus=2),
        **COMMON_BROADCAST_SETTINGS # type: ignore
    ),
    ### Custom input edge-cases for specific models
    "intern_vl-diff-patches": VLMTestInfo(
        models=["OpenGVLab/InternVL2-2B"],
        prompt_formatter=lambda img_prompt: f"<|im_start|>User\n{img_prompt}<|im_end|>\n<|im_start|>Assistant\n", # noqa: E501
        test_type=VLMTestType.CUSTOM_INPUTS,
        max_model_len=4096,
        use_tokenizer_eos=True,
        patch_hf_runner=model_utils.internvl_patch_hf_runner,
        custom_test_opts=[
            CustomTestOptions(
                inputs=inp,
                limit_mm_per_prompt={"image": 2},
            ) for inp in custom_inputs.different_patch_input_cases_internvl()
        ],
    ),
    "llava_onevision-multiple-images": VLMTestInfo(
        models=["llava-hf/llava-onevision-qwen2-0.5b-ov-hf"],
        test_type=VLMTestType.CUSTOM_INPUTS,
        max_model_len=16384,
        max_num_seqs=2,
        postprocess_inputs=model_utils.cast_dtype_post_processor(
            "pixel_values"
        ),
        auto_cls=AutoModelForVision2Seq,
        vllm_output_post_proc=model_utils.llava_onevision_vllm_to_hf_output,
        custom_test_opts=[CustomTestOptions(
            inputs=custom_inputs.multi_image_multi_aspect_ratio_inputs(
                formatter=lambda vid_prompt: f"<|im_start|>user\n{vid_prompt}<|im_end|>\n<|im_start|>assistant\n",  # noqa: E501
            ),
            limit_mm_per_prompt={"image": 4},
        )],
    ),
}
# yapf: enable


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
        models_in_group = models[i * split_size:(i + 1) * split_size]

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
# - custom inputs
@pytest.mark.parametrize(
    "model_type,test_case",
    get_parametrized_options(
        VLM_TEST_SETTINGS,
        test_type=VLMTestType.IMAGE,
        fork_new_process_for_each_test=False,
    ))
def test_single_image_models(tmp_path: PosixPath, model_type: str,
                             test_case: ExpandableVLMTestArgs,
                             hf_runner: Type[HfRunner],
                             vllm_runner: Type[VllmRunner],
                             image_assets: _ImageAssets):
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
        fork_new_process_for_each_test=False,
    ))
def test_multi_image_models(tmp_path: PosixPath, model_type: str,
                            test_case: ExpandableVLMTestArgs,
                            hf_runner: Type[HfRunner],
                            vllm_runner: Type[VllmRunner],
                            image_assets: _ImageAssets):
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
        fork_new_process_for_each_test=False,
    ))
def test_image_embedding_models(model_type: str,
                                test_case: ExpandableVLMTestArgs,
                                hf_runner: Type[HfRunner],
                                vllm_runner: Type[VllmRunner],
                                image_assets: _ImageAssets):
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
        fork_new_process_for_each_test=False,
    ))
def test_video_models(model_type: str, test_case: ExpandableVLMTestArgs,
                      hf_runner: Type[HfRunner], vllm_runner: Type[VllmRunner],
                      video_assets: _VideoAssets):
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
        test_type=VLMTestType.CUSTOM_INPUTS,
        fork_new_process_for_each_test=False,
    ))
def test_custom_inputs_models(
    model_type: str,
    test_case: ExpandableVLMTestArgs,
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
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
        fork_new_process_for_each_test=True,
    ))
@fork_new_process_for_each_test
def test_single_image_models_heavy(tmp_path: PosixPath, model_type: str,
                                   test_case: ExpandableVLMTestArgs,
                                   hf_runner: Type[HfRunner],
                                   vllm_runner: Type[VllmRunner],
                                   image_assets: _ImageAssets):
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
        fork_new_process_for_each_test=True,
    ))
@fork_new_process_for_each_test
def test_multi_image_models_heavy(tmp_path: PosixPath, model_type: str,
                                  test_case: ExpandableVLMTestArgs,
                                  hf_runner: Type[HfRunner],
                                  vllm_runner: Type[VllmRunner],
                                  image_assets: _ImageAssets):
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
        fork_new_process_for_each_test=True,
    ))
@fork_new_process_for_each_test
def test_image_embedding_models_heavy(model_type: str,
                                      test_case: ExpandableVLMTestArgs,
                                      hf_runner: Type[HfRunner],
                                      vllm_runner: Type[VllmRunner],
                                      image_assets: _ImageAssets):
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
        fork_new_process_for_each_test=True,
    ))
def test_video_models_heavy(model_type: str, test_case: ExpandableVLMTestArgs,
                            hf_runner: Type[HfRunner],
                            vllm_runner: Type[VllmRunner],
                            video_assets: _VideoAssets):
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
        test_type=VLMTestType.CUSTOM_INPUTS,
        fork_new_process_for_each_test=True,
    ))
@fork_new_process_for_each_test
def test_custom_inputs_models_heavy(
    model_type: str,
    test_case: ExpandableVLMTestArgs,
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
):
    model_test_info = VLM_TEST_SETTINGS[model_type]
    runners.run_custom_inputs_test(
        model_test_info=model_test_info,
        test_case=test_case,
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
    )
