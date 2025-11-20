# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that we handle a startup Error and shutdown."""

import pytest

from tests.utils import (
    check_gpu_memory_usage,
    create_new_process_for_each_test,
    wait_for_gpu_memory_to_clear,
)
from tests.v1.shutdown.utils import (
    SHUTDOWN_TEST_THRESHOLD_BYTES,
    SHUTDOWN_TEST_TIMEOUT_SEC,
)
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.utils.torch_utils import cuda_device_count_stateless
from vllm.v1.engine.async_llm import AsyncLLM

MODELS = ["hmellor/tiny-random-LlamaForCausalLM"]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tensor_parallel_size", [2, 1])
@pytest.mark.parametrize("send_one_request", [False, True])
async def test_async_llm_delete(
    model: str, tensor_parallel_size: int, send_one_request: bool
) -> None:
    """Test that AsyncLLM frees GPU memory upon deletion.
    AsyncLLM always uses an MP client.

    Args:
      model: model under test
      tensor_parallel_size: degree of tensor parallelism
      send_one_request: send one request to engine before deleting
    """
    if cuda_device_count_stateless() < tensor_parallel_size:
        pytest.skip(reason="Not enough CUDA devices")

    devices = list(range(tensor_parallel_size))
    check_gpu_memory_usage(devices)

    engine_args = AsyncEngineArgs(
        model=model, enforce_eager=True, tensor_parallel_size=tensor_parallel_size
    )

    # Instantiate AsyncLLM; make request to complete any deferred
    # initialization; then delete instance
    async_llm = AsyncLLM.from_engine_args(engine_args)
    if send_one_request:
        async for _ in async_llm.generate(
            "Hello my name is",
            request_id="abc",
            sampling_params=SamplingParams(
                max_tokens=1, output_kind=RequestOutputKind.DELTA
            ),
        ):
            pass
    del async_llm

    # Confirm all the processes are cleaned up.
    wait_for_gpu_memory_to_clear(
        devices=devices,
        threshold_bytes=SHUTDOWN_TEST_THRESHOLD_BYTES,
        timeout_s=SHUTDOWN_TEST_TIMEOUT_SEC,
    )


def _test_llm_delete(
    model: str,
    tensor_parallel_size: int,
    send_one_request: bool,
) -> None:
    devices = list(range(tensor_parallel_size))
    check_gpu_memory_usage(devices)

    # Instantiate LLM; make request to complete any deferred
    # initialization; then delete instance
    llm = LLM(
        model=model, enforce_eager=True, tensor_parallel_size=tensor_parallel_size
    )
    if send_one_request:
        llm.generate("Hello my name is", sampling_params=SamplingParams(max_tokens=1))
    del llm

    # Confirm all the processes are cleaned up.
    wait_for_gpu_memory_to_clear(
        devices=devices,
        threshold_bytes=SHUTDOWN_TEST_THRESHOLD_BYTES,
        timeout_s=SHUTDOWN_TEST_TIMEOUT_SEC,
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tensor_parallel_size", [2, 1])
@pytest.mark.parametrize("send_one_request", [False, True])
def test_llm_delete(
    monkeypatch,
    model: str,
    tensor_parallel_size: int,
    send_one_request: bool,
) -> None:
    """Test that LLM frees GPU memory upon deletion.

    Args:
      model: model under test
      tensor_parallel_size: degree of tensor parallelism
      send_one_request: send one request to engine before deleting
    """
    if cuda_device_count_stateless() < tensor_parallel_size:
        pytest.skip(reason="Not enough CUDA devices")

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")

    _test_llm_delete(model, tensor_parallel_size, send_one_request)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tensor_parallel_size", [2, 1])
@pytest.mark.parametrize("send_one_request", [False, True])
@create_new_process_for_each_test()  # Avoid initing CUDA in this process with TP=1
def test_llm_delete_without_multiprocessing(
    monkeypatch,
    model: str,
    tensor_parallel_size: int,
    send_one_request: bool,
) -> None:
    """Test that LLM frees GPU memory upon deletion, without multiprocessing.

    Args:
      model: model under test
      tensor_parallel_size: degree of tensor parallelism
      send_one_request: send one request to engine before deleting
    """
    if cuda_device_count_stateless() < tensor_parallel_size:
        pytest.skip(reason="Not enough CUDA devices")

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    _test_llm_delete(model, tensor_parallel_size, send_one_request)
