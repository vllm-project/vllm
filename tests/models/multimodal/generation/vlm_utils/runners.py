# SPDX-License-Identifier: Apache-2.0
"""Entrypoints for wrapping the core run_test implementation for specific test
types / modalities.
"""
from pathlib import PosixPath

from .....conftest import (AudioTestAssets, HfRunner, ImageTestAssets,
                           VideoTestAssets, VllmRunner)
from . import builders, core
from .types import ExpandableVLMTestArgs, VLMTestInfo


####### Entrypoints for running different test types
def run_single_image_test(*, tmp_path: PosixPath, model_test_info: VLMTestInfo,
                          test_case: ExpandableVLMTestArgs,
                          hf_runner: type[HfRunner],
                          vllm_runner: type[VllmRunner],
                          image_assets: ImageTestAssets):
    assert test_case.size_wrapper is not None
    inputs = builders.build_single_image_inputs_from_test_info(
        model_test_info, image_assets, test_case.size_wrapper, tmp_path)

    core.run_test(
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        inputs=inputs,
        model=test_case.model,
        dtype=test_case.dtype,
        max_tokens=test_case.max_tokens,
        num_logprobs=test_case.num_logprobs,
        limit_mm_per_prompt={"image": 1},
        distributed_executor_backend=test_case.distributed_executor_backend,
        **model_test_info.get_non_parametrized_runner_kwargs())


def run_multi_image_test(*, tmp_path: PosixPath, model_test_info: VLMTestInfo,
                         test_case: ExpandableVLMTestArgs,
                         hf_runner: type[HfRunner],
                         vllm_runner: type[VllmRunner],
                         image_assets: ImageTestAssets):
    assert test_case.size_wrapper is not None
    inputs = builders.build_multi_image_inputs_from_test_info(
        model_test_info, image_assets, test_case.size_wrapper, tmp_path)

    core.run_test(
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        inputs=inputs,
        model=test_case.model,
        dtype=test_case.dtype,
        max_tokens=test_case.max_tokens,
        num_logprobs=test_case.num_logprobs,
        limit_mm_per_prompt={"image": len(image_assets)},
        distributed_executor_backend=test_case.distributed_executor_backend,
        **model_test_info.get_non_parametrized_runner_kwargs())


def run_embedding_test(*, model_test_info: VLMTestInfo,
                       test_case: ExpandableVLMTestArgs,
                       hf_runner: type[HfRunner],
                       vllm_runner: type[VllmRunner],
                       image_assets: ImageTestAssets):
    assert test_case.size_wrapper is not None
    inputs, vllm_embeddings = builders.build_embedding_inputs_from_test_info(
        model_test_info, image_assets, test_case.size_wrapper)

    core.run_test(
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        inputs=inputs,
        model=test_case.model,
        dtype=test_case.dtype,
        max_tokens=test_case.max_tokens,
        num_logprobs=test_case.num_logprobs,
        limit_mm_per_prompt={"image": 1},
        vllm_embeddings=vllm_embeddings,
        distributed_executor_backend=test_case.distributed_executor_backend,
        **model_test_info.get_non_parametrized_runner_kwargs())


def run_video_test(
    *,
    model_test_info: VLMTestInfo,
    test_case: ExpandableVLMTestArgs,
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    video_assets: VideoTestAssets,
):
    assert test_case.size_wrapper is not None
    assert test_case.num_video_frames is not None
    inputs = builders.build_video_inputs_from_test_info(
        model_test_info, video_assets, test_case.size_wrapper,
        test_case.num_video_frames)

    core.run_test(
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        inputs=inputs,
        model=test_case.model,
        dtype=test_case.dtype,
        max_tokens=test_case.max_tokens,
        num_logprobs=test_case.num_logprobs,
        limit_mm_per_prompt={"video": len(video_assets)},
        distributed_executor_backend=test_case.distributed_executor_backend,
        **model_test_info.get_non_parametrized_runner_kwargs())


def run_audio_test(
    *,
    model_test_info: VLMTestInfo,
    test_case: ExpandableVLMTestArgs,
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    audio_assets: AudioTestAssets,
):
    inputs = builders.build_audio_inputs_from_test_info(
        model_test_info, audio_assets)

    core.run_test(
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        inputs=inputs,
        model=test_case.model,
        dtype=test_case.dtype,
        max_tokens=test_case.max_tokens,
        num_logprobs=test_case.num_logprobs,
        limit_mm_per_prompt={"audio": 1},
        distributed_executor_backend=test_case.distributed_executor_backend,
        **model_test_info.get_non_parametrized_runner_kwargs())


def run_custom_inputs_test(*, model_test_info: VLMTestInfo,
                           test_case: ExpandableVLMTestArgs,
                           hf_runner: type[HfRunner],
                           vllm_runner: type[VllmRunner]):
    # Custom test cases can provide inputs directly, but they need to
    # explicitly provided a CustomTestConfig, which wraps the inputs and
    # the limit_mm_per_prompt
    assert test_case.custom_test_opts is not None

    inputs = test_case.custom_test_opts.inputs
    limit_mm_per_prompt = test_case.custom_test_opts.limit_mm_per_prompt
    # Inputs and limit_mm_per_prompt should all be set
    assert inputs is not None
    assert limit_mm_per_prompt is not None

    core.run_test(
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        inputs=inputs,
        model=test_case.model,
        dtype=test_case.dtype,
        max_tokens=test_case.max_tokens,
        num_logprobs=test_case.num_logprobs,
        limit_mm_per_prompt=limit_mm_per_prompt,
        distributed_executor_backend=test_case.distributed_executor_backend,
        **model_test_info.get_non_parametrized_runner_kwargs())
