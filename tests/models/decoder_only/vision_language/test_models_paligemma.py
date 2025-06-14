# SPDX-License-Identifier: Apache-2.0
"""Common tests for testing .generate() functionality for single / multiple
image, embedding, and video support for different VLMs in vLLM.
"""
import math
import os
from collections import defaultdict
from pathlib import PosixPath

import pytest
from transformers import AutoModelForVision2Seq

from vllm.platforms import current_platform
from vllm.utils import identity

from ....conftest import IMAGE_ASSETS, HfRunner, VllmRunner, _ImageAssets
from .vlm_utils import model_utils, runners
from .vlm_utils.case_filtering import get_parametrized_options
from .vlm_utils.types import ExpandableVLMTestArgs, VLMTestInfo, VLMTestType

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
    "paligemma": VLMTestInfo(
        #models=["google/paligemma-3b-mix-224"],
        models=["google/paligemma2-3b-ft-docci-448"],
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
        marks=[pytest.mark.skip],
        max_model_len=8192,
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
                             hf_runner: type[HfRunner],
                             vllm_runner: type[VllmRunner],
                             image_assets: _ImageAssets):
    model_test_info = VLM_TEST_SETTINGS[model_type]
    print(f"{test_case=}\n{model_test_info=}\n{model_test_info.get_non_parametrized_runner_kwargs()}")
    #assert model_type==1 #(f"Failed ({model_type='paligemma'}, {test_case=}, {model_test_info=}")
    runners.run_single_image_test(
        tmp_path=tmp_path,
        model_test_info=model_test_info,
        test_case=test_case,
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        image_assets=image_assets,
    )

# if __name__ == 'main':
#     model_type,test_case = get_parametrized_options(
#         VLM_TEST_SETTINGS,
#         test_type=VLMTestType.IMAGE,
#         fork_new_process_for_each_test=False,
#     )
#     print(model_type,test_case, VLMTestType.IMAGE)

# @pytest.mark.parametrize(
#    "model_type,test_case",
#    get_parametrized_options(
#        VLM_TEST_SETTINGS,
#        test_type=VLMTestType.MULTI_IMAGE,
#        fork_new_process_for_each_test=False,
#    ))
# def test_multi_image_models(tmp_path: PosixPath, model_type: str,
#                            test_case: ExpandableVLMTestArgs,
#                            hf_runner: type[HfRunner],
#                            vllm_runner: type[VllmRunner],
#                            image_assets: _ImageAssets):
#    model_test_info = VLM_TEST_SETTINGS[model_type]
#    runners.run_multi_image_test(
#        tmp_path=tmp_path,
#        model_test_info=model_test_info,
#        test_case=test_case,
#        hf_runner=hf_runner,
#        vllm_runner=vllm_runner,
#        image_assets=image_assets,
#    )


# @pytest.mark.parametrize(
#    "model_type,test_case",
#    get_parametrized_options(
#        VLM_TEST_SETTINGS,
#        test_type=VLMTestType.EMBEDDING,
#        fork_new_process_for_each_test=False,
#    ))
# def test_image_embedding_models(model_type: str,
#                                test_case: ExpandableVLMTestArgs,
#                                hf_runner: type[HfRunner],
#                                vllm_runner: type[VllmRunner],
#                                image_assets: _ImageAssets):
#    model_test_info = VLM_TEST_SETTINGS[model_type]
#    runners.run_embedding_test(
#        model_test_info=model_test_info,
#        test_case=test_case,
#        hf_runner=hf_runner,
#        vllm_runner=vllm_runner,
#        image_assets=image_assets,
#    )


# @pytest.mark.parametrize(
#    "model_type,test_case",
#    get_parametrized_options(
#        VLM_TEST_SETTINGS,
#        test_type=VLMTestType.VIDEO,
#        fork_new_process_for_each_test=False,
#    ))
# def test_video_models(model_type: str, test_case: ExpandableVLMTestArgs,
#                      hf_runner: type[HfRunner], vllm_runner: type[VllmRunner],
#                      video_assets: _VideoAssets):
#    model_test_info = VLM_TEST_SETTINGS[model_type]
#    runners.run_video_test(
#        model_test_info=model_test_info,
#        test_case=test_case,
#        hf_runner=hf_runner,
#        vllm_runner=vllm_runner,
#        video_assets=video_assets,
#    )


# @pytest.mark.parametrize(
#    "model_type,test_case",
#    get_parametrized_options(
#        VLM_TEST_SETTINGS,
#        test_type=VLMTestType.CUSTOM_INPUTS,
#        fork_new_process_for_each_test=False,
#    ))
# def test_custom_inputs_models(
#    model_type: str,
#    test_case: ExpandableVLMTestArgs,
#    hf_runner: type[HfRunner],
#    vllm_runner: type[VllmRunner],
# ):
#    model_test_info = VLM_TEST_SETTINGS[model_type]
#    runners.run_custom_inputs_test(
#        model_test_info=model_test_info,
#        test_case=test_case,
#        hf_runner=hf_runner,
#        vllm_runner=vllm_runner,
#    )


# #### Tests filtering for things running each test as a new process
# @pytest.mark.parametrize(
#    "model_type,test_case",
#    get_parametrized_options(
#        VLM_TEST_SETTINGS,
#        test_type=VLMTestType.IMAGE,
#        fork_new_process_for_each_test=True,
#    ))
# @fork_new_process_for_each_test
# def test_single_image_models_heavy(tmp_path: PosixPath, model_type: str,
#                                   test_case: ExpandableVLMTestArgs,
#                                   hf_runner: type[HfRunner],
#                                   vllm_runner: type[VllmRunner],
#                                   image_assets: _ImageAssets):
#    model_test_info = VLM_TEST_SETTINGS[model_type]
#    runners.run_single_image_test(
#        tmp_path=tmp_path,
#        model_test_info=model_test_info,
#        test_case=test_case,
#        hf_runner=hf_runner,
#        vllm_runner=vllm_runner,
#        image_assets=image_assets,
#    )


# @pytest.mark.parametrize(
#    "model_type,test_case",
#    get_parametrized_options(
#        VLM_TEST_SETTINGS,
#        test_type=VLMTestType.MULTI_IMAGE,
#        fork_new_process_for_each_test=True,
#    ))
# @fork_new_process_for_each_test
# def test_multi_image_models_heavy(tmp_path: PosixPath, model_type: str,
#                                  test_case: ExpandableVLMTestArgs,
#                                  hf_runner: type[HfRunner],
#                                  vllm_runner: type[VllmRunner],
#                                  image_assets: _ImageAssets):
#    model_test_info = VLM_TEST_SETTINGS[model_type]
#    runners.run_multi_image_test(
#        tmp_path=tmp_path,
#        model_test_info=model_test_info,
#        test_case=test_case,
#        hf_runner=hf_runner,
#        vllm_runner=vllm_runner,
#        image_assets=image_assets,
#    )


# @pytest.mark.parametrize(
#    "model_type,test_case",
#    get_parametrized_options(
#        VLM_TEST_SETTINGS,
#        test_type=VLMTestType.EMBEDDING,
#        fork_new_process_for_each_test=True,
#    ))
# @fork_new_process_for_each_test
# def test_image_embedding_models_heavy(model_type: str,
#                                      test_case: ExpandableVLMTestArgs,
#                                      hf_runner: type[HfRunner],
#                                      vllm_runner: type[VllmRunner],
#                                      image_assets: _ImageAssets):
#    model_test_info = VLM_TEST_SETTINGS[model_type]
#    runners.run_embedding_test(
#        model_test_info=model_test_info,
#        test_case=test_case,
#        hf_runner=hf_runner,
#        vllm_runner=vllm_runner,
#        image_assets=image_assets,
#    )


# @pytest.mark.parametrize(
#    "model_type,test_case",
#    get_parametrized_options(
#        VLM_TEST_SETTINGS,
#        test_type=VLMTestType.VIDEO,
#        fork_new_process_for_each_test=True,
#    ))
# def test_video_models_heavy(model_type: str, test_case: ExpandableVLMTestArgs,
#                            hf_runner: type[HfRunner],
#                            vllm_runner: type[VllmRunner],
#                            video_assets: _VideoAssets):
#    model_test_info = VLM_TEST_SETTINGS[model_type]
#    runners.run_video_test(
#        model_test_info=model_test_info,
#        test_case=test_case,
#        hf_runner=hf_runner,
#        vllm_runner=vllm_runner,
#        video_assets=video_assets,
#    )


# @pytest.mark.parametrize(
#    "model_type,test_case",
#    get_parametrized_options(
#        VLM_TEST_SETTINGS,
#        test_type=VLMTestType.CUSTOM_INPUTS,
#        fork_new_process_for_each_test=True,
#    ))
# @fork_new_process_for_each_test
# def test_custom_inputs_models_heavy(
#    model_type: str,
#    test_case: ExpandableVLMTestArgs,
#    hf_runner: type[HfRunner],
#    vllm_runner: type[VllmRunner],
# ):
#    model_test_info = VLM_TEST_SETTINGS[model_type]
#    runners.run_custom_inputs_test(
#        model_test_info=model_test_info,
#        test_case=test_case,
#        hf_runner=hf_runner,
#        vllm_runner=vllm_runner,
#    )
