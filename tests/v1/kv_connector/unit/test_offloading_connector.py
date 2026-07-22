# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import socket
import time

import msgspec
import msgspec.msgpack
import prometheus_client
import pytest
import zmq
from tqdm import tqdm

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import KVEventsConfig, KVTransferConfig
from vllm.distributed.kv_events import BlockStored, KVEventBatch
from vllm.platforms import current_platform

CPU_BLOCK_SIZES: int = 64 if current_platform.is_xpu() else 48
_ATTN_BACKENDS: list[str] = []
if current_platform.is_cuda():
    _ATTN_BACKENDS = ["FLASH_ATTN", "FLASHINFER", "TRITON_ATTN"]
elif current_platform.is_rocm():
    _ATTN_BACKENDS = ["TRITON_ATTN"]
elif current_platform.is_xpu():
    _ATTN_BACKENDS = ["FLASH_ATTN", "TRITON_ATTN"]

# (model, attn_backend | None, block_size | None, uses_hma)
#
# - Llama: tested with each attention backend and a custom block_size.
# - Gemma-3: HMA (sliding window + full attention), default backend.
# - Mamba-130m: HMA (attention-free, acts like sliding_window=1),
#   default backend.  Prefix caching must be force-enabled.
# - Falcon-H1-0.5B-Instruct: HMA (parallel SSM/attention in every layer).
#   After page-size unification the mamba and attention groups have
#   different block sizes.
MODEL_PARAMS: list[tuple[str, str | None, int | None, bool]] = [
    ("meta-llama/Llama-3.2-1B-Instruct", backend, CPU_BLOCK_SIZES, False)
    for backend in _ATTN_BACKENDS
]
# HMA / Mamba models are only tested on CUDA (not ROCm).
if current_platform.is_cuda():
    MODEL_PARAMS += [
        ("google/gemma-3-1b-it", None, CPU_BLOCK_SIZES, True),
        ("state-spaces/mamba-130m-hf", None, CPU_BLOCK_SIZES, True),
        # Falcon-H1: parallel hybrid (every layer has both attention and SSM).
        # The mamba and attention groups end up with different GPU block sizes
        # after page-size unification, so we leave cpu_block_size=None
        # (blocks_per_chunk stays 1).
        ("tiiuae/Falcon-H1-0.5B-Instruct", None, None, True),
    ]

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

    The GPU-to-CPU offload runs on a CUDA stream asynchronously. While blocks
    are still held by the offload worker, ``reset_prefix_cache`` returns
    ``False``. Between retries we send a dummy single-token prefill to force
    the engine to step, which polls the worker for completed transfers and
    frees GPU blocks.
    """
    _dummy_params = SamplingParams(max_tokens=1)
    deadline = time.monotonic() + _RESET_CACHE_TIMEOUT
    while not llm.reset_prefix_cache():
        if time.monotonic() > deadline:
            raise TimeoutError(
                "reset_prefix_cache did not succeed within "
                f"{_RESET_CACHE_TIMEOUT}s - async offload may be stuck"
            )
        # Force an engine step so the scheduler polls get_finished()
        # and releases GPU blocks held by in-flight async stores.
        llm.generate(
            [TokensPrompt(prompt_token_ids=[0])],
            _dummy_params,
            use_tqdm=False,
        )


def _latency_test(llm: LLM, subscriber: MockSubscriber | None):
    # TODO: Reintroduce latency test on ROCm once MRV2 supports cross
    # layer KV Cache. See https://github.com/vllm-project/vllm/pull/45947
    if current_platform.is_rocm():
        return
    sampling_params = SamplingParams(max_tokens=1)

    num_times_cpu_better_than_cold = 0
    num_tests = 10
    total_cold_time = 0.0
    total_gpu_hit_time = 0.0
    total_cpu_hit_time = 0.0
    max_model_len = llm.llm_engine.vllm_config.model_config.max_model_len
    # Use a long prompt that fits within the model's context window.
    prompt_len = min(10001, max_model_len - 1)
    prompt_token_ids = [0] * prompt_len
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
        if subscriber is not None:
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


def _accuracy_test(llm: LLM, subscriber: MockSubscriber | None):
    sampling_params = SamplingParams(max_tokens=1)
    extra_config = (
        llm.llm_engine.vllm_config.kv_transfer_config.kv_connector_extra_config
    )
    cpu_block_size = extra_config.get("block_size")
    if cpu_block_size is None:
        # No custom offloaded block_size: offloaded blocks match GPU blocks.
        # Use the hash block_size (cache_config.block_size) for alignment.
        cpu_block_size = llm.llm_engine.vllm_config.cache_config.block_size

    if subscriber is not None:
        subscriber.get_new_cpu_stored_events()

    # Pad prompt so its token count is a multiple of cpu_block_size.
    # Use the tokenizer directly to avoid expensive llm.generate() calls.
    tokenizer = llm.get_tokenizer()
    prompt = "Let's count to 10. One, two, three, four,"
    while len(tokenizer.encode(prompt)) % cpu_block_size != 0:
        prompt = ". " + prompt

    # Seed the CPU cache with the prompt.
    llm.generate(prompt, sampling_params, use_tqdm=False)

    if subscriber is not None:
        assert subscriber.get_new_cpu_stored_events()

    test_count = 20
    results = llm.generate([prompt] * test_count, sampling_params, use_tqdm=False)
    success_count = sum(1 for r in results if r.outputs[0].text == " five")
    assert success_count >= 0.5 * test_count


@pytest.mark.parametrize("model, attn_backend, cpu_block_size, uses_hma", MODEL_PARAMS)
def test_cpu_offloading(
    model: str,
    attn_backend: str | None,
    cpu_block_size: int | None,
    uses_hma: bool,
) -> None:
    """
    Tests OffloadingConnector with CPUOffloadingSpec.
    """
    # configure OffloadingConnector (spec_name=CPUOffloadingSpec by default)
    extra_config: dict = {"cpu_bytes_to_use": 500 << 20}
    if cpu_block_size is not None:
        extra_config["block_size"] = cpu_block_size
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra_config,
    )

    subscriber: MockSubscriber | None = None
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

    # Attention-free / hybrid models disable prefix caching by default
    # (ModelConfig.is_prefix_caching_supported returns False).  Without it,
    # mamba_block_size falls back to max_model_len, making GPU blocks too
    # large for any reasonable offloaded block_size.  Force-enable it.
    force_prefix_caching = uses_hma

    llm = LLM(
        model=model,
        max_model_len=4096,
        gpu_memory_utilization=0.5,
        kv_events_config=kv_events_config,
        kv_transfer_config=kv_transfer_config,
        **({"attention_config": {"backend": attn_backend}} if attn_backend else {}),
        # Keep HMA explicitly enabled for HMA model coverage.
        **({"disable_hybrid_kv_cache_manager": False} if uses_hma else {}),
        **({"enable_prefix_caching": True} if force_prefix_caching else {}),
        # ROCm: batch size 1 to reduce variability
        **({"max_num_seqs": 1} if current_platform.is_rocm() else {}),
    )

    events_endpoint = events_endpoint.replace("*", "127.0.0.1")
    subscriber = MockSubscriber(events_endpoint, topic=kv_events_config.topic)

    try:
        _latency_test(llm, subscriber)
        _accuracy_test(llm, subscriber)
    finally:
        if subscriber is not None:
            subscriber.close()
        del llm


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_cpu_offloading_metrics() -> None:
    """Verify that offloading Prometheus metrics (new flat and deprecated
    labeled) are emitted after stores and loads."""
    extra_config: dict = {
        "cpu_bytes_to_use": 500 << 20,
        "block_size": CPU_BLOCK_SIZES,
    }
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra_config,
    )

    llm = LLM(
        model="meta-llama/Llama-3.2-1B-Instruct",
        max_model_len=4096,
        gpu_memory_utilization=0.5,
        kv_transfer_config=kv_transfer_config,
        disable_log_stats=False,
    )

    try:
        prompt_token_ids = list(range(500))

        # First generate: cold run, triggers a store to CPU.
        # Use max_tokens>1 so the request is still producing output
        # tokens when the async store completes and stats get drained.
        # (The LLMEngine only records stats on steps with request outputs.)
        llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_token_ids)],
            SamplingParams(max_tokens=10),
            use_tqdm=False,
        )

        # Wait for the async offload to finish, then reset GPU prefix cache
        # so the next generate must load from CPU.
        _wait_for_prefix_cache_reset(llm)

        # Second generate: triggers a load from CPU.
        # Send a short filler alongside the load prompt so there's always
        # a request producing output tokens when the load stats get drained
        # (the LLMEngine only records stats on steps with request outputs).
        filler = TokensPrompt(prompt_token_ids=[0])
        llm.generate(
            [filler, TokensPrompt(prompt_token_ids=prompt_token_ids)],
            SamplingParams(max_tokens=50),
            use_tqdm=False,
        )

        # Metric helpers.
        registry = prometheus_client.REGISTRY

        def _get_counter_value(
            name: str, labels: dict[str, str] | None = None
        ) -> float:
            total = 0.0
            for metric in registry.collect():
                if metric.name == name:
                    for sample in metric.samples:
                        if sample.name != name + "_total":
                            continue
                        if labels and not all(
                            sample.labels.get(k) == v for k, v in labels.items()
                        ):
                            continue
                        total += sample.value
            return total

        def _get_histogram_count(
            name: str, labels: dict[str, str] | None = None
        ) -> float:
            total = 0.0
            for metric in registry.collect():
                if metric.name == name:
                    for sample in metric.samples:
                        if sample.name != name + "_count":
                            continue
                        if labels and not all(
                            sample.labels.get(k) == v for k, v in labels.items()
                        ):
                            continue
                        total += sample.value
            return total

        # New flat counter metrics
        store_bytes = _get_counter_value("vllm:kv_offload_store_bytes")
        assert store_bytes > 0, f"Expected store_bytes > 0, got {store_bytes}"
        load_bytes = _get_counter_value("vllm:kv_offload_load_bytes")
        assert load_bytes > 0, f"Expected load_bytes > 0, got {load_bytes}"
        store_time = _get_counter_value("vllm:kv_offload_store_time")
        assert store_time > 0, f"Expected store_time > 0, got {store_time}"
        load_time = _get_counter_value("vllm:kv_offload_load_time")
        assert load_time > 0, f"Expected load_time > 0, got {load_time}"

        # New flat histogram metrics
        store_size_count = _get_histogram_count("vllm:kv_offload_store_size")
        assert store_size_count > 0, (
            f"Expected store_size histogram observations > 0, got {store_size_count}"
        )
        load_size_count = _get_histogram_count("vllm:kv_offload_load_size")
        assert load_size_count > 0, (
            f"Expected load_size histogram observations > 0, got {load_size_count}"
        )

        # Deprecated labeled metrics — verify per transfer_type label.
        load_label = {"transfer_type": "CPU_to_GPU"}
        store_label = {"transfer_type": "GPU_to_CPU"}

        dep_load_bytes = _get_counter_value("vllm:kv_offload_total_bytes", load_label)
        assert dep_load_bytes > 0, (
            f"Expected deprecated load bytes > 0, got {dep_load_bytes}"
        )
        dep_store_bytes = _get_counter_value("vllm:kv_offload_total_bytes", store_label)
        assert dep_store_bytes > 0, (
            f"Expected deprecated store bytes > 0, got {dep_store_bytes}"
        )
        dep_load_time = _get_counter_value("vllm:kv_offload_total_time", load_label)
        assert dep_load_time > 0, (
            f"Expected deprecated load time > 0, got {dep_load_time}"
        )
        dep_store_time = _get_counter_value("vllm:kv_offload_total_time", store_label)
        assert dep_store_time > 0, (
            f"Expected deprecated store time > 0, got {dep_store_time}"
        )
        dep_load_size = _get_histogram_count("vllm:kv_offload_size", load_label)
        assert dep_load_size > 0, (
            f"Expected deprecated load size observations > 0, got {dep_load_size}"
        )
        dep_store_size = _get_histogram_count("vllm:kv_offload_size", store_label)
        assert dep_store_size > 0, (
            f"Expected deprecated store size observations > 0, got {dep_store_size}"
        )

        # Flat and deprecated metrics must be consistent (dual-write).
        assert store_bytes == dep_store_bytes
        assert load_bytes == dep_load_bytes
        assert store_time == dep_store_time
        assert load_time == dep_load_time
        assert store_size_count == dep_store_size
        assert load_size_count == dep_load_size
    finally:
        del llm


def test_tiering_offloading() -> None:
    """Tests OffloadingConnector with TieringOffloadingSpec."""
    extra_config: dict = {
        "cpu_bytes_to_use": 500 << 20,
        "block_size": CPU_BLOCK_SIZES,
        "spec_name": "TieringOffloadingSpec",
        "secondary_tiers": [{"type": "example"}],
    }
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra_config,
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
        max_model_len=4096,
        gpu_memory_utilization=0.5,
        kv_events_config=kv_events_config,
        kv_transfer_config=kv_transfer_config,
    )
    subscriber = MockSubscriber(
        events_endpoint.replace("*", "127.0.0.1"),
        topic=kv_events_config.topic,
    )
    try:
        _latency_test(llm, subscriber)
        _accuracy_test(llm, subscriber)
    finally:
        subscriber.close()
        del llm


def test_fs_tiering_offloading(tmp_path) -> None:
    """Tests OffloadingConnector with TieringOffloadingSpec
    + fs secondary tier."""
    extra_config: dict = {
        "cpu_bytes_to_use": 1 << 30,
        "block_size": CPU_BLOCK_SIZES,
        "spec_name": "TieringOffloadingSpec",
        "secondary_tiers": [{"type": "fs", "root_dir": str(tmp_path)}],
    }
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra_config,
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
        max_model_len=4096,
        gpu_memory_utilization=0.5,
        kv_events_config=kv_events_config,
        kv_transfer_config=kv_transfer_config,
    )
    subscriber = MockSubscriber(
        events_endpoint.replace("*", "127.0.0.1"),
        topic=kv_events_config.topic,
    )
    try:
        _latency_test(llm, subscriber)
        _accuracy_test(llm, subscriber)
    finally:
        subscriber.close()
        del llm


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="HMA mamba-align CPU offload test is CUDA-only",
)
@pytest.mark.parametrize(
    "model,block_size,tp_size",
    [
        # ("Qwen/Qwen3.6-35B-A3B", 1056, 2),
        # ("tiiuae/falcon-mamba-7b", 16, 1),
        ("state-spaces/mamba-1.4b-hf", 16, 1)
    ],
)
def test_mamba_align_cpu_offload(model: str, block_size: int, tp_size: int):
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "cpu_bytes_to_use": 4 << 30,
            "block_size": block_size,
        },
    )
    llm = LLM(
        model=model,
        max_model_len=block_size * 10,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=tp_size,
        kv_transfer_config=kv_transfer_config,
        language_model_only=True,
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        disable_hybrid_kv_cache_manager=False,
    )

    _PROMPT_SIZE: int = block_size * 2
    _PROMPT_TEXT = "Hi. Give me a set of trivia questions and their answers "

    # build prompt ids to match prompt_size
    tokenizer = llm.get_tokenizer()
    raw_ids: list[int] = tokenizer.encode(_PROMPT_TEXT)
    while len(raw_ids) < _PROMPT_SIZE:
        raw_ids = tokenizer.encode("....") + raw_ids
    initial_ids: list[int] = raw_ids[:_PROMPT_SIZE]

    sampling_params = SamplingParams(max_tokens=128, temperature=0, ignore_eos=True)

    failures: list[str] = []

    def _get_output_str(outputs):
        return outputs[0].outputs[0].text

    def _verify(llm, prompt, label: str):
        cold_outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        _wait_for_prefix_cache_reset(llm)
        cpu_outputs = llm.generate([prompt], sampling_params, use_tqdm=False)

        cold_text = _get_output_str(cold_outputs)
        cpu_text = _get_output_str(cpu_outputs)
        print(f"{label} : cold outputs\n{cold_text}")
        print(f"{label} : cpu outputs\n{cpu_text}")

        if cold_text != cpu_text:
            failures.append(
                f"{label}: mismatch\n  cold: {cold_text!r}\n  cpu:  {cpu_text!r}"
            )

    try:
        # Mamba has only a single state. The CPU cache stores are triggered
        # at offload block boundaries. When the prompt is exactly at the boundary,
        # The CPU offload should not load the cached block.
        # This is because we'd use that state to recompute the last token. This
        # does not work for mamba as there is only one KV value and that is for
        # for the token at the boundary.
        # This is fine for other attention types as we have all the necessary
        # token KV values in the hit blocks.
        prompt = TokensPrompt(prompt_token_ids=initial_ids)
        _verify(llm, prompt, "block-boundary-prompt")

        # Test for prompt token ids at non-block boundaries.
        # Reuse is okay for this case.
        prompt = TokensPrompt(prompt_token_ids=[0] + initial_ids)
        _verify(llm, prompt, "block-mid-prompt")

        assert not failures, "\n\n".join(failures)

    finally:
        del llm
