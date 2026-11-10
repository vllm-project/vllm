# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref

import pytest

from vllm import LLM, SamplingParams
from vllm.exceptions import VLLMValidationError

MODEL_NAME = "distilbert/distilgpt2"

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

TOKEN_IDS = [
    [0],
    [0, 1],
    [0, 2, 1],
    [0, 3, 1, 2],
]


@pytest.fixture(scope="module")
def llm(vllm_runner):
    with vllm_runner(
        MODEL_NAME,
        max_num_batched_tokens=4096,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.10,
        enforce_eager=True,
    ) as runner:
        # pytest caches yielded fixtures until after teardown, so use a proxy to
        # avoid retaining the LLM while VllmRunner.__exit__ releases ROCm memory.
        yield weakref.proxy(runner.llm)


@pytest.mark.skip_global_cleanup
def test_multiple_sampling_params(llm: LLM):
    sampling_params = [
        SamplingParams(temperature=0.01, top_p=0.95),
        SamplingParams(temperature=0.3, top_p=0.95),
        SamplingParams(temperature=0.7, top_p=0.95),
        SamplingParams(temperature=0.99, top_p=0.95),
    ]

    # Multiple SamplingParams should be matched with each prompt
    outputs = llm.generate(PROMPTS, sampling_params=sampling_params)
    assert len(PROMPTS) == len(outputs)

    # Exception raised, if the size of params does not match the size of prompts
    with pytest.raises(VLLMValidationError):
        outputs = llm.generate(PROMPTS, sampling_params=sampling_params[:3])

    # Single SamplingParams should be applied to every prompt
    single_sampling_params = SamplingParams(temperature=0.3, top_p=0.95)
    outputs = llm.generate(PROMPTS, sampling_params=single_sampling_params)
    assert len(PROMPTS) == len(outputs)

    # sampling_params is None, default params should be applied
    outputs = llm.generate(PROMPTS, sampling_params=None)
    assert len(PROMPTS) == len(outputs)


def test_multiple_priority(llm: LLM):
    # Generate works when priority is None
    outputs = llm.generate(PROMPTS, sampling_params=None, priority=None)
    assert len(PROMPTS) == len(outputs)

    # Generate works when length of priority is same as the len(PROMPTS)
    outputs = llm.generate(PROMPTS, sampling_params=None, priority=[0] * len(PROMPTS))
    assert len(PROMPTS) == len(outputs)

    # Generate works when priority is a single value applied to every prompt
    outputs = llm.generate(PROMPTS, sampling_params=None, priority=1)
    assert len(PROMPTS) == len(outputs)

    # Exception raised, if the length of priority does not match the length of prompts
    with pytest.raises(VLLMValidationError):
        outputs = llm.generate(
            PROMPTS, sampling_params=None, priority=[0] * (len(PROMPTS) - 1)
        )

    # Exception raised, if the priority list is empty
    with pytest.raises(VLLMValidationError):
        outputs = llm.generate(PROMPTS, sampling_params=None, priority=[])


def test_single_prompt_priority(llm: LLM):
    # Single string prompts should be normalized to one request.
    outputs = llm.generate(PROMPTS[0], sampling_params=None, priority=[0])
    assert len(outputs) == 1

    # Single string prompts should also accept a single priority value.
    outputs = llm.generate(PROMPTS[0], sampling_params=None, priority=1)
    assert len(outputs) == 1


def test_max_model_len(vllm_runner):
    max_model_len = 20
    with vllm_runner(
        MODEL_NAME,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.10,
        enforce_eager=True,  # reduce test time
    ) as runner:
        sampling_params = SamplingParams(max_tokens=max_model_len + 10)
        outputs = runner.llm.generate(PROMPTS, sampling_params)
        for output in outputs:
            num_total_tokens = len(output.prompt_token_ids) + len(
                output.outputs[0].token_ids
            )
            # Total tokens must not exceed max_model_len.
            # It can be less if generation finishes due to other reasons (e.g., EOS)
            # before reaching the absolute model length limit.
            assert num_total_tokens <= max_model_len


def test_log_stats(vllm_runner):
    with vllm_runner(
        MODEL_NAME,
        disable_log_stats=False,
        gpu_memory_utilization=0.10,
        enforce_eager=True,  # reduce test time
    ) as runner:
        outputs = runner.llm.generate(PROMPTS, sampling_params=None)

        # disable_log_stats is False, every output should have metrics
        assert all(output.metrics is not None for output in outputs)


@pytest.mark.parametrize("metadata", [None, False, True])
def test_kv_event_metadata_live_and_replay(vllm_runner, metadata):
    """Real generation honors metadata opt-in and replays the original bytes."""
    import time

    import msgspec
    import zmq

    from vllm.distributed.kv_events import (
        BlockStored,
        KVEventBatch,
        KVEventSource,
        ZmqEventPublisher,
    )
    from vllm.utils.network_utils import get_open_ports_list

    pub_port, replay_port = get_open_ports_list(2)
    endpoint = f"tcp://127.0.0.1:{pub_port}"
    replay_endpoint = f"tcp://127.0.0.1:{replay_port}"
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"metadata")
    subscriber.setsockopt(zmq.LINGER, 0)
    subscriber.connect(endpoint)
    replay = context.socket(zmq.DEALER)
    replay.setsockopt(zmq.LINGER, 0)
    replay.connect(replay_endpoint)
    try:
        with vllm_runner(
            MODEL_NAME,
            max_model_len=256,
            block_size=16,
            enable_prefix_caching=True,
            gpu_memory_utilization=0.10,
            enforce_eager=True,
            kv_events_config={
                "enable_kv_cache_events": True,
                "publisher": "zmq",
                "endpoint": f"tcp://*:{pub_port}",
                "replay_endpoint": f"tcp://*:{replay_port}",
                "topic": "metadata",
                **({} if metadata is None else {"enable_kv_event_metadata": metadata}),
            },
        ) as runner:
            # A received warm-up event proves the PUB subscription is active.
            # The target generation below runs exactly once after this exchange.
            deadline = time.monotonic() + 30
            warmup_token = 100
            while time.monotonic() < deadline:
                runner.llm.generate(
                    {"prompt_token_ids": [warmup_token] * 32},
                    SamplingParams(temperature=0, max_tokens=1),
                )
                warmup_token += 1
                if subscriber.poll(100):
                    subscriber.recv_multipart()
                    break
            else:
                pytest.fail("KV event subscription did not become ready")
            outputs = runner.llm.generate(
                {"prompt_token_ids": list(range(64))},
                SamplingParams(temperature=0, max_tokens=2),
            )
            assert len(outputs) == 1 and outputs[0].outputs[0].token_ids
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if not subscriber.poll(100):
                    continue
                topic, sequence, payload = subscriber.recv_multipart()
                batch = msgspec.msgpack.decode(payload, type=KVEventBatch)
                stored = [
                    event
                    for event in batch.events
                    if isinstance(event, BlockStored)
                    and event.token_ids[:16] == list(range(16))
                ]
                if stored:
                    break
            else:
                pytest.fail("Target generation did not emit a live prefix-cache store")
            assert topic == b"metadata"
            assert batch.data_parallel_rank == 0
            assert all(
                event.source == (KVEventSource("cache_manager") if metadata else None)
                for event in batch.events
            )
            assert all(
                event.medium == "GPU"
                and event.sharing_scope == ("instance" if metadata else None)
                for event in stored
            )

            if not metadata:
                from tests.distributed.test_kv_cache_events import (
                    _STRICT_PRE_METADATA_EVENT,
                )

                for event_payload in msgspec.msgpack.decode(payload)[1]:
                    assert "source" not in event_payload
                    assert "sharing_scope" not in event_payload
                    msgspec.msgpack.decode(
                        msgspec.msgpack.encode(event_payload),
                        type=_STRICT_PRE_METADATA_EVENT,
                    )

            replay.send_multipart([b"", sequence])
            replayed = {}
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                assert replay.poll(1000), "KV replay stalled"
                delimiter, replay_topic, replay_sequence, replay_payload = (
                    replay.recv_multipart()
                )
                assert delimiter == b""
                if replay_sequence == ZmqEventPublisher.END_SEQ:
                    break
                assert replay_sequence not in replayed
                assert replay_topic == topic
                replayed[replay_sequence] = replay_payload
            else:
                pytest.fail("KV replay did not finish")
            assert replayed[sequence] == payload
    finally:
        subscriber.close(linger=0)
        replay.close(linger=0)
        context.term()
