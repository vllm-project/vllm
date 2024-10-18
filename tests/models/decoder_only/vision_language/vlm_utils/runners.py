"""Entrypoints for wrapping the core run_test implementation for specific test
types / modalities.
"""
from pathlib import PosixPath
from typing import Optional, Type

from .....conftest import HfRunner, VllmRunner, _ImageAssets, _VideoAssets
from . import builders, core
from .types import CustomTestOptions, ImageSizeWrapper, VLMTestInfo


####### Entrypoints for running different test types
def run_single_image_test(
        *, tmp_path: PosixPath, test_info: VLMTestInfo, model: str,
        max_tokens: int, num_logprobs: int, dtype: str,
        distributed_executor_backend: Optional[str],
        size_wrapper: ImageSizeWrapper, hf_runner: Type[HfRunner],
        vllm_runner: Type[VllmRunner], image_assets: _ImageAssets):

    inputs = builders.build_single_image_inputs_from_test_info(
        test_info, image_assets, size_wrapper, tmp_path)

    core.run_test(hf_runner=hf_runner,
                  vllm_runner=vllm_runner,
                  inputs=inputs,
                  model=model,
                  dtype=dtype,
                  max_tokens=max_tokens,
                  num_logprobs=num_logprobs,
                  limit_mm_per_prompt={"image": 1},
                  distributed_executor_backend=distributed_executor_backend,
                  **test_info.get_non_parametrized_runner_kwargs())


def run_multi_image_test(
        *, tmp_path: PosixPath, test_info: VLMTestInfo, model: str,
        max_tokens: int, num_logprobs: int, dtype: str,
        distributed_executor_backend: Optional[str],
        size_wrapper: ImageSizeWrapper, hf_runner: Type[HfRunner],
        vllm_runner: Type[VllmRunner], image_assets: _ImageAssets):

    inputs = builders.build_multi_image_inputs_from_test_info(
        test_info, image_assets, size_wrapper, tmp_path)

    core.run_test(hf_runner=hf_runner,
                  vllm_runner=vllm_runner,
                  inputs=inputs,
                  model=model,
                  dtype=dtype,
                  max_tokens=max_tokens,
                  num_logprobs=num_logprobs,
                  limit_mm_per_prompt={"image": len(image_assets)},
                  distributed_executor_backend=distributed_executor_backend,
                  **test_info.get_non_parametrized_runner_kwargs())


def run_embedding_test(*, test_info: VLMTestInfo, model: str, max_tokens: int,
                       num_logprobs: int, dtype: str,
                       distributed_executor_backend: Optional[str],
                       size_wrapper: ImageSizeWrapper,
                       hf_runner: Type[HfRunner],
                       vllm_runner: Type[VllmRunner],
                       image_assets: _ImageAssets):

    inputs, vllm_embeddings = builders.build_embedding_inputs_from_test_info(
        test_info, image_assets, size_wrapper)

    core.run_test(hf_runner=hf_runner,
                  vllm_runner=vllm_runner,
                  inputs=inputs,
                  model=model,
                  dtype=dtype,
                  max_tokens=max_tokens,
                  num_logprobs=num_logprobs,
                  limit_mm_per_prompt={"image": 1},
                  vllm_embeddings=vllm_embeddings,
                  distributed_executor_backend=distributed_executor_backend,
                  **test_info.get_non_parametrized_runner_kwargs())


def run_video_test(
    *,
    test_info: VLMTestInfo,
    model: str,
    num_frames: int,
    max_tokens: int,
    num_logprobs: int,
    dtype: str,
    distributed_executor_backend: Optional[str],
    size_wrapper: ImageSizeWrapper,
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    video_assets: _VideoAssets,
):
    inputs = builders.build_video_inputs_from_test_info(
        test_info, video_assets, size_wrapper, num_frames)

    core.run_test(hf_runner=hf_runner,
                  vllm_runner=vllm_runner,
                  inputs=inputs,
                  model=model,
                  dtype=dtype,
                  max_tokens=max_tokens,
                  num_logprobs=num_logprobs,
                  limit_mm_per_prompt={"video": len(video_assets)},
                  distributed_executor_backend=distributed_executor_backend,
                  **test_info.get_non_parametrized_runner_kwargs())


def run_custom_inputs_test(*, test_info: VLMTestInfo, model: str,
                           max_tokens: int, num_logprobs: int,
                           distributed_executor_backend: Optional[str],
                           dtype: str, custom_test_opts: CustomTestOptions,
                           hf_runner: Type[HfRunner],
                           vllm_runner: Type[VllmRunner]):
    # Custom test cases can provide inputs directly, but they need to
    # explicitly provided a CustomTestConfig, which wraps the inputs and
    # the limit_mm_per_prompt
    assert custom_test_opts is not None

    inputs = custom_test_opts.inputs

    limit_mm_per_prompt = custom_test_opts.limit_mm_per_prompt

    assert inputs is not None and limit_mm_per_prompt is not None
    core.run_test(hf_runner=hf_runner,
                  vllm_runner=vllm_runner,
                  inputs=inputs,
                  model=model,
                  dtype=dtype,
                  max_tokens=max_tokens,
                  num_logprobs=num_logprobs,
                  limit_mm_per_prompt=limit_mm_per_prompt,
                  distributed_executor_backend=distributed_executor_backend,
                  **test_info.get_non_parametrized_runner_kwargs())
