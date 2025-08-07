# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os
import signal
import time
import uuid
from dataclasses import dataclass
from threading import Thread
from typing import Optional, Union
from unittest.mock import MagicMock

import pytest
import torch
from transformers import AutoTokenizer

from tests.utils import multi_gpu_test
from vllm import SamplingParams
from vllm.distributed.kv_events import (BlockStored, KVEventBatch,
                                        ZmqEventPublisher)
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.usage.usage_lib import UsageContext
from vllm.utils import set_default_torch_num_threads
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.core_client import (AsyncMPClient, EngineCoreClient,
                                        SyncMPClient)
from vllm.v1.engine.utils import CoreEngineProcManager
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


def make_request(
        params: SamplingParams,
        prompt_tokens_ids: Optional[list[int]] = None) -> EngineCoreRequest:
    if not prompt_tokens_ids:
        prompt_tokens_ids = PROMPT_TOKENS

    return EngineCoreRequest(
        request_id=str(uuid.uuid4()),
        prompt_token_ids=prompt_tokens_ids,
        mm_inputs=None,
        mm_hashes=None,
        mm_placeholders=None,
        sampling_params=params,
        pooling_params=None,
        eos_token_id=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
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


async def loop_until_fully_done_async(client: EngineCoreClient, outputs: dict):

    while True:
        engine_core_outputs = (await client.get_output_async()).outputs

        if len(engine_core_outputs) == 0:
            continue

        # Add outputs to the dict
        for out in engine_core_outputs:
            outputs[out.request_id].append(out)

        # Check if all request IDs in outputs have finished
        if all(outs and outs[-1].finished for outs in outputs.values()):
            break

        await asyncio.sleep(0.1)


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

        with set_default_torch_num_threads(1):
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

        with set_default_torch_num_threads(1):
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


@dataclass
class MyDataclass:
    message: str


# Dummy utility function to monkey-patch into engine core.
def echo_dc(
    self,
    msg: str,
    return_list: bool = False,
) -> Union[MyDataclass, list[MyDataclass]]:
    print(f"echo dc util function called: {msg}")
    val = None if msg is None else MyDataclass(msg)
    # Return dataclass to verify support for returning custom types
    # (for which there is special handling to make it work with msgspec).
    return [val for _ in range(3)] if return_list else val


@pytest.mark.asyncio(loop_scope="function")
async def test_engine_core_client_util_method_custom_return(
        monkeypatch: pytest.MonkeyPatch):

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        # Must set insecure serialization to allow returning custom types.
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        # Monkey-patch core engine utility function to test.
        m.setattr(EngineCore, "echo_dc", echo_dc, raising=False)

        engine_args = EngineArgs(model=MODEL_NAME, enforce_eager=True)
        vllm_config = engine_args.create_engine_config(
            usage_context=UsageContext.UNKNOWN_CONTEXT)
        executor_class = Executor.get_class(vllm_config)

        with set_default_torch_num_threads(1):
            client = EngineCoreClient.make_client(
                multiprocess_mode=True,
                asyncio_mode=True,
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=True,
            )

        try:
            # Test utility method returning custom / non-native data type.
            core_client: AsyncMPClient = client

            result = await core_client.call_utility_async(
                "echo_dc", "testarg2", False)
            assert isinstance(result,
                              MyDataclass) and result.message == "testarg2"
            result = await core_client.call_utility_async(
                "echo_dc", "testarg2", True)
            assert isinstance(result, list) and all(
                isinstance(r, MyDataclass) and r.message == "testarg2"
                for r in result)

            # Test returning None and list of Nones
            result = await core_client.call_utility_async(
                "echo_dc", None, False)
            assert result is None
            result = await core_client.call_utility_async(
                "echo_dc", None, True)
            assert isinstance(result, list) and all(r is None for r in result)

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

        engine_args = EngineArgs(
            model=MODEL_NAME,
            enforce_eager=True,
            enable_prefix_caching=True,
            block_size=block_size,
        )
        engine_args.kv_events_config = publisher_config

        vllm_config = engine_args.create_engine_config(
            UsageContext.UNKNOWN_CONTEXT)

        executor_class = Executor.get_class(vllm_config)
        with set_default_torch_num_threads(1):
            client = EngineCoreClient.make_client(
                multiprocess_mode=multiprocessing_mode,
                asyncio_mode=False,
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=False,
            )
        endpoint = publisher_config.endpoint.replace("*", "127.0.0.1")
        subscriber = MockSubscriber(endpoint,
                                    topic=publisher_config.topic,
                                    decode_type=KVEventBatch)

        try:
            custom_tokens = list(range(num_blocks * block_size))
            sampling_params = SamplingParams(max_tokens=1)
            request = make_request(sampling_params, custom_tokens)
            client.add_request(request)

            outputs: dict[str, list] = {request.request_id: []}
            loop_until_done(client, outputs)

            result = subscriber.receive_one(timeout=1000)
            assert result is not None, "No message received"

            seq, received = result

            assert seq == 0, "Sequence number mismatch"
            assert (len(received.events) == 1
                    ), "We should have exactly one BlockStored event"
            event = received.events[0]
            assert isinstance(
                event, BlockStored), "We should have a BlockStored event"
            assert (len(event.block_hashes) == num_blocks
                    ), "We should have a BlockStored event with 2 block_hashes"
            assert (event.block_size == block_size
                    ), "Block size should be the same as the block size"
            assert (event.parent_block_hash
                    is None), "Parent block hash should be None"
            assert event.lora_id is None, "Lora id should be None"
            assert (len(event.token_ids) == num_blocks * block_size
                    ), "Token ids should be the same as the custom tokens"
            assert (event.token_ids == custom_tokens
                    ), "Token ids should be the same as the custom tokens"
        finally:
            client.shutdown()
            subscriber.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "multiprocessing_mode,publisher_config",
    [(True, "tcp")],
    indirect=["publisher_config"],
)
@multi_gpu_test(num_gpus=4)
async def test_kv_cache_events_dp(
    monkeypatch: pytest.MonkeyPatch,
    multiprocessing_mode: bool,
    publisher_config,
):

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        block_size = 16
        num_blocks = 2
        dp_size = 2
        tp_size = 2

        engine_args = EngineArgs(
            model=MODEL_NAME,
            enforce_eager=True,
            enable_prefix_caching=True,
            data_parallel_size=dp_size,
            tensor_parallel_size=tp_size,
            block_size=block_size,
        )
        engine_args.kv_events_config = publisher_config

        vllm_config = engine_args.create_engine_config(
            UsageContext.UNKNOWN_CONTEXT)

        executor_class = Executor.get_class(vllm_config)
        with set_default_torch_num_threads(1):
            client = EngineCoreClient.make_client(
                multiprocess_mode=multiprocessing_mode,
                asyncio_mode=True,
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=False,
            )
        await asyncio.sleep(1)

        # Build endpoints for all DP ranks
        base_endpoint = publisher_config.endpoint.replace("*", "127.0.0.1")
        endpoints = []
        for i in range(dp_size):
            offset_endpoint = ZmqEventPublisher.offset_endpoint_port(
                base_endpoint, i)
            endpoints.append(offset_endpoint)

        subscriber = MockSubscriber(endpoints,
                                    topic=publisher_config.topic,
                                    decode_type=KVEventBatch)

        try:
            custom_tokens = list(range(num_blocks * block_size))
            sampling_params = SamplingParams(max_tokens=1)
            all_request_ids = []

            # Create and add 25 requests
            # NOTE: attempts to force routing to both dp groups but can be flaky
            for i in range(25):
                await asyncio.sleep(0.01)
                request = make_request(sampling_params, custom_tokens)
                await client.add_request_async(request)
                all_request_ids.append(request.request_id)

            await asyncio.sleep(0.1)

            # Initialize outputs dict for all requests
            outputs: dict[str, list] = {
                req_id: []
                for req_id in all_request_ids
            }

            print("processing requests...")
            await asyncio.wait_for(loop_until_fully_done_async(
                client, outputs),
                                   timeout=20.0)

            # Receive from subscriber until no more messages
            print("collecting results...")
            results = []
            while True:
                result = subscriber.receive_one(timeout=1)
                print(result)
                if result is None:
                    break
                results.append(result)

            # Collect all events and data_parallel_ranks from all results
            all_dp_ranks = [
                received.data_parallel_rank for (_, received) in results
            ]
            unique_dps = set(all_dp_ranks)
            assert (
                len(unique_dps) == 2
            ), f"Expected 2 unique data_parallel_ranks, got {len(unique_dps)}"

        finally:
            client.shutdown()
            subscriber.close()


@pytest.mark.timeout(20)
def test_startup_failure(monkeypatch: pytest.MonkeyPatch):

    with monkeypatch.context() as m, pytest.raises(Exception) as e_info:
        m.setenv("VLLM_USE_V1", "1")

        # Monkey-patch to extract core process pid while it's starting.
        core_proc_pid = [None]
        cepm_ctor = CoreEngineProcManager.__init__

        def patched_cepm_ctor(self: CoreEngineProcManager, *args, **kwargs):
            cepm_ctor(self, *args, **kwargs)
            core_proc_pid[0] = self.processes[0].pid

        m.setattr(CoreEngineProcManager, "__init__", patched_cepm_ctor)

        t = time.time()
        engine_args = EngineArgs(model=MODEL_NAME)
        vllm_config = engine_args.create_engine_config(
            usage_context=UsageContext.UNKNOWN_CONTEXT)
        executor_class = Executor.get_class(vllm_config)
        print(f"VllmConfig creation took {time.time() - t:.2f} seconds.")

        # Start another thread to wait for engine core process to start
        # and kill it - simulate fatal uncaught process exit.

        def kill_first_child():
            while (child_pid := core_proc_pid[0]) is None:
                time.sleep(0.5)
            print(f"Killing child core process {child_pid}")
            assert isinstance(child_pid, int)
            os.kill(child_pid, signal.SIGKILL)

        Thread(target=kill_first_child, daemon=True).start()

        _core_client = EngineCoreClient.make_client(
            multiprocess_mode=True,
            asyncio_mode=True,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=True,
        )

    assert "Engine core initialization failed" in str(e_info.value)


@create_new_process_for_each_test()
def test_engine_core_proc_instantiation_cuda_empty(
        monkeypatch: pytest.MonkeyPatch):
    """
    Test that EngineCoreProc can be instantiated when CUDA_VISIBLE_DEVICES
    is empty. This ensures the engine frontend does not need access to GPUs.
    """

    from vllm.v1.engine.core import EngineCoreProc
    from vllm.v1.executor.abstract import Executor

    # Create a simple mock executor instead of a complex custom class
    mock_executor_class = MagicMock(spec=Executor)

    def create_mock_executor(vllm_config):
        mock_executor = MagicMock()

        # Only implement the methods that are actually called during init
        from vllm.v1.kv_cache_interface import FullAttentionSpec
        mock_spec = FullAttentionSpec(block_size=16,
                                      num_kv_heads=1,
                                      head_size=64,
                                      dtype=torch.float16,
                                      use_mla=False)

        mock_executor.get_kv_cache_specs.return_value = [{
            "default": mock_spec
        }]
        mock_executor.determine_available_memory.return_value = [
            1024 * 1024 * 1024
        ]
        mock_executor.initialize_from_config.return_value = None
        mock_executor.max_concurrent_batches = 1

        return mock_executor

    mock_executor_class.side_effect = create_mock_executor

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        m.setenv("CUDA_VISIBLE_DEVICES", "")  # No CUDA devices

        from vllm.v1.engine.utils import EngineZmqAddresses

        def mock_startup_handshake(self, handshake_socket, local_client,
                                   headless, parallel_config):
            return EngineZmqAddresses(inputs=["tcp://127.0.0.1:5555"],
                                      outputs=["tcp://127.0.0.1:5556"],
                                      coordinator_input=None,
                                      coordinator_output=None)

        # Background processes are not important here
        m.setattr(EngineCoreProc, "startup_handshake", mock_startup_handshake)

        vllm_config = EngineArgs(
            model="deepseek-ai/DeepSeek-V2-Lite",
            trust_remote_code=True).create_engine_config()
        engine_core_proc = EngineCoreProc(
            vllm_config=vllm_config,
            local_client=True,
            handshake_address="tcp://127.0.0.1:12345",
            executor_class=mock_executor_class,
            log_stats=False,
            engine_index=0,
        )

        engine_core_proc.shutdown()
