# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import socket
import time

import msgspec
import msgspec.msgpack
import pytest
import zmq
from tqdm import tqdm

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import KVEventsConfig, KVTransferConfig
from vllm.distributed.kv_events import BlockStored, KVEventBatch
from vllm.platforms import current_platform

CPU_BLOCK_SIZES = [48]
ATTN_BACKENDS = []

if current_platform.is_cuda():
    ATTN_BACKENDS = ["FLASH_ATTN", "FLASHINFER", "TRITON_ATTN"]
elif current_platform.is_rocm():
    ATTN_BACKENDS = ["TRITON_ATTN"]

# Maximum time (seconds) to wait for the async CPU offload transfer
# to complete before giving up.
_RESET_CACHE_TIMEOUT = 30 if current_platform.is_rocm() else 10

# ZMQ poll timeout (ms) for the first event.
_FIRST_EVENT_POLL_MS = 10_000 if current_platform.is_rocm() else 1000

# Hard ceiling (seconds) on how long get_new_cpu_stored_events may loop,
# to prevent hangs if non-CPU events keep arriving indefinitely.
_EVENT_DRAIN_TIMEOUT = 60


class MockSubscriber:
    """Helper class to receive and verify published events"""

    def __init__(
        self,
        endpoint: str,
        topic: str,
    ):
        self.ctx = zmq.Context.instance()
        self.topic_bytes = topic.encode("utf-8")

        # Set up subscriber socket
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, self.topic_bytes)
        self.sub.connect(endpoint)

        self.decoder = msgspec.msgpack.Decoder(type=KVEventBatch)

    def get_new_cpu_stored_events(self) -> list[BlockStored]:
        cpu_stored_events: list[BlockStored] = []

        poller = zmq.Poller()
        poller.register(self.sub, zmq.POLLIN)

        poll_ms = _FIRST_EVENT_POLL_MS
        deadline = time.monotonic() + _EVENT_DRAIN_TIMEOUT
        while time.monotonic() < deadline:
            events = dict(poller.poll(poll_ms))

            if events.get(self.sub) != zmq.POLLIN:
                return cpu_stored_events

            topic_bytes, _, payload = self.sub.recv_multipart()

            assert topic_bytes == self.topic_bytes

            event_batch = self.decoder.decode(payload)
            assert isinstance(event_batch, KVEventBatch)
            for event in event_batch.events:
                if isinstance(event, BlockStored) and event.medium == "CPU":
                    cpu_stored_events.append(event)
                    poll_ms = 100

        return cpu_stored_events

    def close(self):
        """Clean up resources"""
        self.sub.close()


def _wait_for_prefix_cache_reset(llm: LLM) -> None:
    """Wait for async offload transfers to finish so prefix cache can reset.

    The GPU-to-CPU offload runs on a CUDA stream asynchronously.  While blocks
    are still held by the offload worker, ``reset_prefix_cache`` returns
    ``False``.  Retry with a short sleep until it succeeds or we time out.
    """
    deadline = time.monotonic() + _RESET_CACHE_TIMEOUT
    while not llm.reset_prefix_cache():
        if time.monotonic() > deadline:
            raise TimeoutError(
                "reset_prefix_cache did not succeed within "
                f"{_RESET_CACHE_TIMEOUT}s - async offload may be stuck"
            )
        time.sleep(0.1)


def _latency_test(llm: LLM, subscriber: MockSubscriber):
    sampling_params = SamplingParams(max_tokens=1)

    num_times_cpu_better_than_cold = 0
    num_tests = 10
    total_cold_time = 0.0
    total_gpu_hit_time = 0.0
    total_cpu_hit_time = 0.0
    prompt_token_ids = [0] * 10001
    for i in tqdm(range(num_tests), desc="Running tests"):
        prompt_token_ids[0] = i
        prompts = [TokensPrompt(prompt_token_ids=prompt_token_ids)]

        # run generation - this should trigger saving KV cache
        start_time = time.time()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        cold_time = time.time() - start_time
        total_cold_time += cold_time

        # run generation again - should hit the GPU prefix cache
        start_time = time.time()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        gpu_hit_time = time.time() - start_time
        total_gpu_hit_time += gpu_hit_time

        # Wait for the async CPU offload to finish, then reset prefix cache
        # so the next generate() must reload from CPU rather than GPU.
        _wait_for_prefix_cache_reset(llm)

        # Verify CPU stored events arrived (offload is done before we
        # attempt to load from CPU).
        assert subscriber.get_new_cpu_stored_events(), (
            f"No CPU stored events received on iteration {i}; "
            "async offload may not have completed in time"
        )

        # run generation again - this should trigger loading from CPU
        start_time = time.time()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        cpu_hit_time = time.time() - start_time
        total_cpu_hit_time += cpu_hit_time

        if cpu_hit_time < cold_time:
            num_times_cpu_better_than_cold += 1

    print("Average times:")
    print(f"    Cold: {total_cold_time * 1000 / num_tests:.2f}ms")
    print(f"    GPU hit: {total_gpu_hit_time * 1000 / num_tests:.2f}ms")
    print(f"    CPU hit: {total_cpu_hit_time * 1000 / num_tests:.2f}ms")

    assert num_times_cpu_better_than_cold >= 0.8 * num_tests


def _accuracy_test(llm: LLM, subscriber: MockSubscriber):
    sampling_params = SamplingParams(max_tokens=1)
    cpu_block_size = (
        llm.llm_engine.vllm_config.kv_transfer_config.kv_connector_extra_config[
            "block_size"
        ]
    )

    subscriber.get_new_cpu_stored_events()

    # prepend prompt to be cpu block aligned
    prompt = "Let's count to 10. One, two, three, four,"
    while (
        len(llm.generate(prompt, use_tqdm=False)[0].prompt_token_ids) % cpu_block_size
        != 0
    ):
        prompt = ". " + prompt

    assert subscriber.get_new_cpu_stored_events()

    test_count = 100
    success_count = 0
    for i in range(test_count):
        if (
            llm.generate(prompt, sampling_params, use_tqdm=False)[0].outputs[0].text
            == " five"
        ):
            success_count += 1

    assert success_count >= 0.5 * test_count


@pytest.mark.parametrize("cpu_block_size", CPU_BLOCK_SIZES)
@pytest.mark.parametrize("attn_backend", ATTN_BACKENDS)
def test_cpu_offloading(cpu_block_size: int, attn_backend: str) -> None:
    """
    Tests OffloadingConnector with CPUOffloadingSpec.
    """

    # configure OffloadingConnector (spec_name=CPUOffloadingSpec by default)
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "cpu_bytes_to_use": 500 << 20,
            "block_size": cpu_block_size,
        },
    )

    port: int
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        port = s.getsockname()[1]

    events_endpoint = f"tcp://*:{port}"
    kv_events_config = KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint=events_endpoint,
        topic="test",
    )

    llm = LLM(
        model="meta-llama/Llama-3.2-1B-Instruct",
        gpu_memory_utilization=0.5,
        kv_events_config=kv_events_config,
        kv_transfer_config=kv_transfer_config,
        attention_config={"backend": attn_backend},
        # ROCm: batch size 1 to reduce variability
        **({"max_num_seqs": 1} if current_platform.is_rocm() else {}),
    )

    events_endpoint = events_endpoint.replace("*", "127.0.0.1")
    subscriber = MockSubscriber(events_endpoint, topic=kv_events_config.topic)

    try:
        _latency_test(llm, subscriber)
        _accuracy_test(llm, subscriber)
    finally:
        subscriber.close()
        del llm
