# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
import uuid
from typing import Dict, List, Optional

import pytest
from transformers import AutoTokenizer

from tests.utils import fork_new_process_for_each_test
from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.core_client import (AsyncMPClient, EngineCoreClient,
                                        SyncMPClient)
from vllm.v1.executor.abstract import Executor

if not current_platform.is_cuda():
    pytest.skip(reason="V1 currently only supported on CUDA.",
                allow_module_level=True)

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
PROMPT = "Hello my name is Robert and I love quantization kernels"
PROMPT_TOKENS = TOKENIZER(PROMPT).input_ids


def make_request(params: SamplingParams) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=str(uuid.uuid4()),
        prompt=PROMPT,
        prompt_token_ids=PROMPT_TOKENS,
        mm_inputs=None,
        mm_hashes=None,
        mm_placeholders=None,
        sampling_params=params,
        eos_token_id=None,
        arrival_time=time.time(),
        lora_request=None,
    )


def loop_until_done(client: EngineCoreClient, outputs: Dict):

    while True:
        engine_core_outputs = client.get_output().outputs

        if len(engine_core_outputs) == 0:
            break

        all_finished = True
        for out in engine_core_outputs:
            outputs[out.request_id].append(out)
            if not out.finished:
                all_finished = False

        if all_finished:
            break


async def loop_until_done_async(client: EngineCoreClient, outputs: Dict):

    while True:
        engine_core_outputs = (await client.get_output_async()).outputs

        if len(engine_core_outputs) == 0:
            break

        all_finished = True
        for out in engine_core_outputs:
            outputs[out.request_id].append(out)
            if not out.finished:
                all_finished = False

        if all_finished:
            break


# Dummy utility function to monkey-patch into engine core.
def echo(self, msg: str, err_msg: Optional[str] = None) -> str:
    print(f"echo util function called: {msg}, {err_msg}")
    if err_msg is not None:
        raise ValueError(err_msg)
    return msg


@fork_new_process_for_each_test
@pytest.mark.parametrize("multiprocessing_mode", [True, False])
def test_engine_core_client(monkeypatch, multiprocessing_mode: bool):

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        # Monkey-patch core engine utility function to test.
        m.setattr(EngineCore, "echo", echo, raising=False)

        engine_args = EngineArgs(model=MODEL_NAME, enforce_eager=True)
        vllm_config = engine_args.create_engine_config(
            UsageContext.UNKNOWN_CONTEXT)
        executor_class = Executor.get_class(vllm_config)
        client = EngineCoreClient.make_client(
            multiprocess_mode=multiprocessing_mode,
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=False,
        )

        MAX_TOKENS = 20
        params = SamplingParams(max_tokens=MAX_TOKENS)
        """Normal Request Cycle."""
        requests = [make_request(params) for _ in range(10)]
        request_ids = [req.request_id for req in requests]

        # Add requests to the engine.
        for request in requests:
            client.add_request(request)
            time.sleep(0.01)

        outputs: Dict[str, List] = {req_id: [] for req_id in request_ids}
        loop_until_done(client, outputs)

        for req_id in request_ids:
            assert len(outputs[req_id]) == MAX_TOKENS, (
                f"{outputs[req_id]=}, {MAX_TOKENS=}")
        """Abort Request Cycle."""

        # Note: this code pathway will only work for multiprocessing
        # since we have to call get_output() explicitly

        # Add requests to the engine.
        for idx, request in enumerate(requests):
            client.add_request(request)
            time.sleep(0.01)
            if idx % 2 == 0:
                client.abort_requests([request.request_id])

        outputs = {req_id: [] for req_id in request_ids}
        loop_until_done(client, outputs)

        for idx, req_id in enumerate(request_ids):
            if idx % 2 == 0:
                assert len(outputs[req_id]) < MAX_TOKENS, (
                    f"{len(outputs[req_id])=}, {MAX_TOKENS=}")
            else:
                assert len(outputs[req_id]) == MAX_TOKENS, (
                    f"{len(outputs[req_id])=}, {MAX_TOKENS=}")
        """Abort after request is finished."""

        # Note: this code pathway will only work for multiprocessing
        # since we have to call get_output() explicitly

        request = requests[0]
        client.add_request(request)
        time.sleep(10.)

        client.abort_requests([request.request_id])

        if multiprocessing_mode:
            """Utility method invocation"""

            core_client: SyncMPClient = client

            result = core_client._call_utility("echo", "testarg")
            assert result == "testarg"

            with pytest.raises(Exception) as e_info:
                core_client._call_utility("echo", None, "help!")

            assert str(e_info.value) == "Call to echo method failed: help!"


@pytest.mark.asyncio(loop_scope="function")
async def test_engine_core_client_asyncio(monkeypatch):

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        # Monkey-patch core engine utility function to test.
        m.setattr(EngineCore, "echo", echo, raising=False)

        engine_args = EngineArgs(model=MODEL_NAME, enforce_eager=True)
        vllm_config = engine_args.create_engine_config(
            usage_context=UsageContext.UNKNOWN_CONTEXT)
        executor_class = Executor.get_class(vllm_config)
        client = EngineCoreClient.make_client(
            multiprocess_mode=True,
            asyncio_mode=True,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=True,
        )

        MAX_TOKENS = 20
        params = SamplingParams(max_tokens=MAX_TOKENS)
        """Normal Request Cycle."""

        requests = [make_request(params) for _ in range(10)]
        request_ids = [req.request_id for req in requests]

        # Add requests to the engine.
        for request in requests:
            await client.add_request_async(request)
            await asyncio.sleep(0.01)

        outputs: Dict[str, List] = {req_id: [] for req_id in request_ids}
        await loop_until_done_async(client, outputs)

        for req_id in request_ids:
            assert len(outputs[req_id]) == MAX_TOKENS, (
                f"{outputs[req_id]=}, {MAX_TOKENS=}")
        """Abort Request Cycle."""

        # Add requests to the engine.
        for idx, request in enumerate(requests):
            await client.add_request_async(request)
            await asyncio.sleep(0.01)
            if idx % 2 == 0:
                await client.abort_requests_async([request.request_id])

        outputs = {req_id: [] for req_id in request_ids}
        await loop_until_done_async(client, outputs)

        for idx, req_id in enumerate(request_ids):
            if idx % 2 == 0:
                assert len(outputs[req_id]) < MAX_TOKENS, (
                    f"{len(outputs[req_id])=}, {MAX_TOKENS=}")
            else:
                assert len(outputs[req_id]) == MAX_TOKENS, (
                    f"{len(outputs[req_id])=}, {MAX_TOKENS=}")
        """Utility method invocation"""

        core_client: AsyncMPClient = client

        result = await core_client._call_utility_async("echo", "testarg")
        assert result == "testarg"

        with pytest.raises(Exception) as e_info:
            await core_client._call_utility_async("echo", None, "help!")

        assert str(e_info.value) == "Call to echo method failed: help!"
