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


@pytest.fixture(scope="module", params=[False, True], ids=["disabled", "enabled"])
def kv_event_model(vllm_runner, request):
    from vllm.utils.network_utils import get_open_port

    replay_endpoint = f"tcp://127.0.0.1:{get_open_port()}"
    config = {
        "enable_kv_cache_events": True,
        "endpoint": "tcp://127.0.0.1:0",
        "replay_endpoint": replay_endpoint,
        "buffer_steps": 100,
    }
    if request.param:
        config["enable_source_attribution"] = True
    with vllm_runner(
        "facebook/opt-125m",
        enforce_eager=True,
        block_size=32,
        max_model_len=128,
        max_num_seqs=1,
        max_num_batched_tokens=128,
        kv_cache_memory_bytes=16 * 1024 * 1024,
        enable_prefix_caching=True,
        kv_events_config=config,
    ) as model:
        yield model, replay_endpoint, request.param


def test_kv_event_sources_preserves_legacy_defaults(kv_event_model):
    """Default discovery still decodes with the pre-attribution strict schema."""
    import msgspec
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm.entrypoints.serve.instrumentator.basic import router

    model, replay_endpoint, enabled = kv_event_model

    class LegacyConfig(
        msgspec.Struct,
        forbid_unknown_fields=True,  # type: ignore[call-arg]
    ):
        enable_kv_cache_events: bool
        publisher: str
        endpoint: str
        replay_endpoint: str | None
        buffer_steps: int
        hwm: int
        max_queue_size: int
        topic: str

    app = FastAPI()
    app.include_router(router)
    app.state.engine_client = model.llm.llm_engine.engine_core
    with TestClient(app) as client:
        discovery = client.get("/kv_event_sources")
        assert discovery.status_code == 200
        fields = discovery.json()["0"]
        if enabled:
            assert fields.pop("enable_source_attribution") is True
        legacy = msgspec.json.decode(msgspec.json.encode(fields), type=LegacyConfig)
        assert legacy.enable_kv_cache_events
        assert legacy.publisher == "zmq"
        assert legacy.replay_endpoint == replay_endpoint
        assert legacy.buffer_steps == 100


def test_native_source_attribution_through_llm(kv_event_model):
    """Real engine execution publishes native stores with optional source in replay."""
    import time

    import msgspec
    import zmq

    from vllm.distributed.kv_events import BlockStored, KVEventBatch, ZmqEventPublisher

    model, replay_endpoint, enabled = kv_event_model
    output = model.generate_greedy([[42] * 64], max_tokens=1)
    assert len(output[0][0]) == 65
    replay = zmq.Context.instance().socket(zmq.DEALER)
    replay.connect(replay_endpoint)
    try:
        stores: list[BlockStored] = []
        deadline = time.monotonic() + 10
        # Wait for the asynchronous publisher to drain its event queue.
        while not stores and time.monotonic() < deadline:
            replay.send_multipart([b"", (0).to_bytes(8, "big")])
            while time.monotonic() < deadline:
                assert replay.poll(1000), "publisher replay did not respond"
                delimiter, topic, sequence, payload = replay.recv_multipart()
                assert delimiter == b""
                if sequence == ZmqEventPublisher.END_SEQ:
                    break
                batch = msgspec.msgpack.decode(payload, type=KVEventBatch)
                stores.extend(e for e in batch.events if isinstance(e, BlockStored))
        assert stores, "native BlockStored event missing"
        if enabled:
            assert all(
                e.source is not None
                and e.source.source_id == "cache_manager"
                and e.source.kind == "native"
                for e in stores
            )
        else:
            assert all(getattr(e, "source", None) is None for e in stores)
    finally:
        replay.close(linger=0)
