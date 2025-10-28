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

CPU_BLOCK_SIZES = [16, 48]


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

        timeout = 1000  # 1 second
        while True:
            events = dict(poller.poll(timeout))

            if events.get(self.sub) != zmq.POLLIN:
                return cpu_stored_events

            topic_bytes, _, payload = self.sub.recv_multipart()

            assert topic_bytes == self.topic_bytes

            event_batch = self.decoder.decode(payload)
            assert isinstance(event_batch, KVEventBatch)
            for event in event_batch.events:
                if isinstance(event, BlockStored) and event.medium == "CPU":
                    cpu_stored_events.append(event)
                    timeout = 100

    def close(self):
        """Clean up resources"""
        self.sub.close()


@pytest.mark.parametrize("cpu_block_size", CPU_BLOCK_SIZES)
def test_cpu_offloading(cpu_block_size: int) -> None:
    """
    Tests OffloadingConnector with CPUOffloadingSpec.
    """

    # configure OffloadingConnector (spec_name=CPUOffloadingSpec by default)
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "num_cpu_blocks": 1000,
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
    )

    sampling_params = SamplingParams(temperature=0, max_tokens=1)

    events_endpoint = events_endpoint.replace("*", "127.0.0.1")
    subscriber = MockSubscriber(events_endpoint, topic=kv_events_config.topic)

    try:
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

            # reset prefix cache to avoid GPU hit.
            llm.reset_prefix_cache()

            assert subscriber.get_new_cpu_stored_events()

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
    finally:
        subscriber.close()
        del llm
