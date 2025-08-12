# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
import uuid
from threading import Thread
from typing import Optional

import psutil
import pytest
from transformers import AutoTokenizer

from vllm import SamplingParams
from vllm.distributed.kv_events import BlockStored, KVEventBatch
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.core_client import (AsyncMPClient, EngineCoreClient,
                                        SyncMPClient)
from vllm.v1.executor.abstract import Executor

from ...distributed.conftest import MockSubscriber
from ...utils import create_new_process_for_each_test

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
        prompt_token_ids=PROMPT_TOKENS,
        mm_inputs=None,
        mm_hashes=None,
        mm_placeholders=None,
        sampling_params=params,
        eos_token_id=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
    )


def loop_until_done(client: EngineCoreClient, outputs: dict):

    while True:
        engine_core_outputs = client.get_output().outputs

        if len(engine_core_outputs) == 0:
            continue

        all_finished = True
        for out in engine_core_outputs:
            outputs[out.request_id].append(out)
            if not out.finished:
                all_finished = False

        if all_finished:
            break


async def loop_until_done_async(client: EngineCoreClient, outputs: dict):

    while True:
        engine_core_outputs = (await client.get_output_async()).outputs

        if len(engine_core_outputs) == 0:
            continue

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


@create_new_process_for_each_test()
@pytest.mark.parametrize("multiprocessing_mode", [True, False])
def test_engine_core_client(monkeypatch: pytest.MonkeyPatch,
                            multiprocessing_mode: bool):

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

        outputs: dict[str, list] = {req_id: [] for req_id in request_ids}
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

            result = core_client.call_utility("echo", "testarg")
            assert result == "testarg"

            with pytest.raises(Exception) as e_info:
                core_client.call_utility("echo", None, "help!")

            assert str(e_info.value) == "Call to echo method failed: help!"


@pytest.mark.asyncio(loop_scope="function")
async def test_engine_core_client_asyncio(monkeypatch: pytest.MonkeyPatch):

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

        try:
            MAX_TOKENS = 20
            params = SamplingParams(max_tokens=MAX_TOKENS)
            """Normal Request Cycle."""

            requests = [make_request(params) for _ in range(10)]
            request_ids = [req.request_id for req in requests]

            # Add requests to the engine.
            for request in requests:
                await client.add_request_async(request)
                await asyncio.sleep(0.01)

            outputs: dict[str, list] = {req_id: [] for req_id in request_ids}
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

            result = await core_client.call_utility_async("echo", "testarg")
            assert result == "testarg"

            with pytest.raises(Exception) as e_info:
                await core_client.call_utility_async("echo", None, "help!")

            assert str(e_info.value) == "Call to echo method failed: help!"
        finally:
            client.shutdown()


@pytest.mark.parametrize(
    "multiprocessing_mode,publisher_config",
    [(True, "tcp"), (False, "inproc")],
    indirect=["publisher_config"],
)
def test_kv_cache_events(
    monkeypatch: pytest.MonkeyPatch,
    multiprocessing_mode: bool,
    publisher_config,
):

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        block_size = 16
        num_blocks = 2

        engine_args = EngineArgs(model=MODEL_NAME,
                                 enforce_eager=True,
                                 enable_prefix_caching=True,
                                 block_size=block_size)
        engine_args.kv_events_config = publisher_config

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
        endpoint = publisher_config.endpoint.replace("*", "127.0.0.1")
        time.sleep(0.1)
        subscriber = MockSubscriber(endpoint,
                                    topic=publisher_config.topic,
                                    decode_type=KVEventBatch)

        try:
            custom_tokens = list(range(num_blocks * block_size))
            request = EngineCoreRequest(
                request_id=str(uuid.uuid4()),
                prompt_token_ids=custom_tokens,
                mm_inputs=None,
                mm_hashes=None,
                mm_placeholders=None,
                sampling_params=SamplingParams(
                    max_tokens=1),  # Short completion for speed
                eos_token_id=None,
                arrival_time=time.time(),
                lora_request=None,
                cache_salt=None,
            )
            client.add_request(request)

            outputs: dict[str, list] = {request.request_id: []}
            loop_until_done(client, outputs)

            result = subscriber.receive_one(timeout=1000)
            assert result is not None, "No message received"

            seq, received = result

            assert seq == 0, "Sequence number mismatch"
            assert len(received.events) == 1, (
                "We should have exactly one BlockStored event")
            event = received.events[0]
            assert isinstance(
                event, BlockStored), ("We should have a BlockStored event")
            assert len(event.block_hashes) == num_blocks, (
                "We should have a BlockStored event with 2 block_hashes")
            assert event.block_size == block_size, (
                "Block size should be the same as the block size")
            assert event.parent_block_hash is None, (
                "Parent block hash should be None")
            assert event.lora_id is None, "Lora id should be None"
            assert len(event.token_ids) == num_blocks * block_size, (
                "Token ids should be the same as the custom tokens")
            assert event.token_ids == custom_tokens, (
                "Token ids should be the same as the custom tokens")
        finally:
            client.shutdown()
        return


@pytest.mark.timeout(10)
def test_startup_failure(monkeypatch: pytest.MonkeyPatch):

    with monkeypatch.context() as m, pytest.raises(Exception) as e_info:
        m.setenv("VLLM_USE_V1", "1")

        engine_args = EngineArgs(model=MODEL_NAME)
        vllm_config = engine_args.create_engine_config(
            usage_context=UsageContext.UNKNOWN_CONTEXT)
        executor_class = Executor.get_class(vllm_config)

        # Start another thread to wait for engine core process to start
        # and kill it - simulate fatal uncaught process exit.
        this_proc = psutil.Process()
        children_before = set(this_proc.children())

        def kill_first_child():
            while True:
                time.sleep(0.5)
                children = set(this_proc.children()) - children_before
                if children:
                    child = children.pop()
                    print("Killing child core process", child.pid)
                    child.kill()
                    break

        Thread(target=kill_first_child, daemon=True).start()

        _core_client = EngineCoreClient.make_client(
            multiprocess_mode=True,
            asyncio_mode=True,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=True,
        )

    assert "Engine core initialization failed" in str(e_info.value)
