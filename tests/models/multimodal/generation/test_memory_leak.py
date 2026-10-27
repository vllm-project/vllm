# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import random
import string
import sys
import weakref

import pytest
import torch
from PIL import Image

from tests.models.registry import HF_EXAMPLE_MODELS
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform
from vllm.renderers.chat_utils import ChatCompletionMessageParam
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils.mem_constants import KiB_bytes, MiB_bytes
from vllm.utils.mem_utils import format_mib

MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
RANDOM_PREFIX_LEN = 100
TEST_IMAGE_NAMES = [
    "2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
    "Grayscale_8bits_palette_sample_image.png",
]
MAX_MODEL_LEN = 8192
REQUESTS_PER_ROUND = 4
WARMUP_ROUNDS = 2
MEASURED_ROUNDS = 16
GPU_GROWTH_THRESHOLD_MIB = 0
CPU_PEAK_GROWTH_THRESHOLD_MIB = 0

SAMPLING_PARAMS = SamplingParams(
    temperature=0.0,
    max_tokens=16,
)


def _make_messages(image_url: str) -> list[ChatCompletionMessageParam]:
    # Avoid obscuring memory leaks because of prefix caching
    random_text = "".join(random.choices(string.ascii_uppercase, k=RANDOM_PREFIX_LEN))

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Ignore this random string: {random_text}",
                },
                {"type": "image_url", "image_url": {"url": image_url}},
                {
                    "type": "text",
                    "text": "Describe this image in one short sentence.",
                },
            ],
        }
    ]


def _build_request_batch(
    image_urls: list[str],
) -> list[list[ChatCompletionMessageParam]]:
    return [
        _make_messages(image_urls[i % len(image_urls)])
        for i in range(REQUESTS_PER_ROUND)
    ]


def _ru_maxrss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss <= 0:
        return 0

    # Linux reports kilobytes, macOS reports bytes.
    return rss if sys.platform == "darwin" else rss * KiB_bytes


def _gpu_used_bytes() -> int:
    torch.accelerator.synchronize()
    free_bytes, total_bytes = torch.accelerator.get_memory_info()
    return int(total_bytes - free_bytes)


def _format_mib(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "n/a"

    return f"{format_mib(num_bytes)} MiB"


@pytest.fixture(scope="function")
def llm(monkeypatch, request):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # pytest caches the fixture so we use weakref.proxy to
    # enable garbage collection
    llm_kwargs = dict(
        model=maybe_model_redirect(MODEL_NAME),
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=REQUESTS_PER_ROUND,
        limit_mm_per_prompt={"image": 1},
        seed=0,
        disable_log_stats=True,
        gpu_memory_utilization=0.8,
    )
    if current_platform.is_rocm():
        llm_kwargs["attention_backend"] = "TRITON_ATTN"
    llm_kwargs.update(getattr(request, "param", {}))

    llm = LLM(**llm_kwargs)

    yield weakref.proxy(llm)

    del llm

    cleanup_dist_env_and_memory()


@pytest.mark.core_model
@pytest.mark.parametrize("image_urls", [TEST_IMAGE_NAMES], indirect=True)
def test_no_memory_leak(llm, image_urls: list[str]) -> None:
    model_info = HF_EXAMPLE_MODELS.find_hf_info(MODEL_NAME)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    request_batch = _build_request_batch(image_urls)

    # Establish a warmup baseline after model load and the first multimodal
    # requests complete. Later rounds should remain near this steady state.
    for _ in range(WARMUP_ROUNDS):
        outputs = llm.chat(request_batch, sampling_params=SAMPLING_PARAMS)
        assert len(outputs) == len(request_batch)
        assert llm.llm_engine.get_num_unfinished_requests() == 0
        del outputs

    gc.collect()
    warmup_gpu = _gpu_used_bytes()
    warmup_cpu_peak = _ru_maxrss_bytes()

    post_warmup_gpu_samples: list[int] = []
    post_warmup_cpu_peak_samples: list[int] = []

    for _ in range(MEASURED_ROUNDS):
        outputs = llm.chat(request_batch, sampling_params=SAMPLING_PARAMS)
        assert len(outputs) == len(request_batch)
        assert llm.llm_engine.get_num_unfinished_requests() == 0
        del outputs

        gc.collect()
        post_warmup_gpu_samples.append(_gpu_used_bytes())
        cpu_peak = _ru_maxrss_bytes()
        if cpu_peak is not None:
            post_warmup_cpu_peak_samples.append(cpu_peak)

    gpu_growth = max(post_warmup_gpu_samples) - warmup_gpu
    gpu_threshold = GPU_GROWTH_THRESHOLD_MIB * MiB_bytes

    assert gpu_growth <= gpu_threshold, (
        "Qwen3-VL GPU memory kept growing after warmup. "
        f"warmup_baseline={_format_mib(warmup_gpu)}, "
        f"post_warmup_samples={[_format_mib(x) for x in post_warmup_gpu_samples]}, "
        f"gpu_growth={_format_mib(gpu_growth)}, "
        f"gpu_threshold={GPU_GROWTH_THRESHOLD_MIB} MiB"
    )

    if warmup_cpu_peak is not None and post_warmup_cpu_peak_samples:
        cpu_peak_growth = max(post_warmup_cpu_peak_samples) - warmup_cpu_peak
        cpu_threshold = CPU_PEAK_GROWTH_THRESHOLD_MIB * MiB_bytes

        assert cpu_peak_growth <= cpu_threshold, (
            "Qwen3-VL CPU peak RSS kept growing after warmup. "
            f"warmup_ru_maxrss={_format_mib(warmup_cpu_peak)}, "
            f"post_warmup_ru_maxrss={[_format_mib(x) for x in post_warmup_cpu_peak_samples]}, "  # noqa: E501
            f"cpu_peak_growth={_format_mib(cpu_peak_growth)}, "
            f"cpu_peak_threshold={CPU_PEAK_GROWTH_THRESHOLD_MIB} MiB"
        )


def _encoder_storage_sizes(worker) -> tuple[int, int, int]:
    outputs = worker.model_runner.encoder_cache.encoder_outputs
    storages = {
        output.untyped_storage().data_ptr(): output.untyped_storage().nbytes()
        for output in outputs.values()
    }
    return len(outputs), sum(t.nbytes for t in outputs.values()), sum(storages.values())


@pytest.mark.core_model
@pytest.mark.parametrize(
    "llm",
    [
        {
            "max_model_len": 2048,
            "max_num_batched_tokens": 4096,
            "max_num_seqs": 16,
            "mm_processor_cache_gb": 0,
            "mm_processor_kwargs": {"max_pixels": 512 * 512},
            "limit_mm_per_prompt": {"image": {"count": 1, "width": 512, "height": 512}},
        }
    ],
    indirect=True,
)
def test_batched_encoder_cache_bounds_retained_storage(llm) -> None:
    """Hot images must not retain unbounded backing from older encoder batches."""
    params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
    anchors: list[int] = []
    for round_id in range(10):
        for kind, ids in (
            ("reuse", anchors),
            ("new", range(round_id * 4, round_id * 4 + 4)),
        ):
            if not ids:
                continue
            prompts = [
                {
                    "prompt": f"<|im_start|>user\nRound {round_id}-{kind}: "
                    "<|vision_start|><|image_pad|><|vision_end|>"
                    "Describe the image.<|im_end|>\n<|im_start|>assistant\n",
                    "multi_modal_data": {
                        "image": Image.new("RGB", (512, 512), (i * 11, i * 7, i * 3))
                    },
                }
                for i in ids
            ]
            outputs = llm.generate(prompts, params, use_tqdm=False)
            assert all(len(o.outputs[0].token_ids) == 8 for o in outputs)
        anchors.append(round_id * 4)
    snapshots = llm.collective_rpc(_encoder_storage_sizes)
    # Forty images entered a sixteen-image cache, including hot peers.
    assert all(count == 16 for count, _, _ in snapshots), snapshots
    # Compaction waits until it reclaims at least as many bytes as it copies.
    assert all(
        logical <= storage <= 2 * logical for _, logical, storage in snapshots
    ), snapshots


def _live_mm_payloads(worker) -> dict[str, list[bool]]:
    cache = worker.model_runner.encoder_cache
    return {
        req_id: [feature.data is not None for feature in features]
        for req_id, features in cache.mm_features.items()
    }


def _arena_registered_bytes(worker) -> list[int]:
    from vllm.distributed.device_communicators.shm_tensor_arena import _TENSOR_ARENAS

    return [sum(arena._pinned_sizes) for arena in _TENSOR_ARENAS.values()]


@pytest.mark.core_model
@pytest.mark.parametrize(
    "llm",
    [{}, {"enable_shm_tensor_arena": True, "distributed_executor_backend": "mp"}],
    indirect=True,
    ids=["default", "arena"],
)
def test_consumed_cpu_inputs_released_during_generation(llm) -> None:
    """Release consumed pixels; arena transport must register only the used pages."""
    assert llm.llm_engine.vllm_config.use_v2_model_runner
    engine = llm.llm_engine
    arena = engine.vllm_config.parallel_config.enable_shm_tensor_arena
    image_size = 1536 if arena else 64
    engine.add_request(
        "cpu-input-lifetime",
        {
            "prompt": (
                "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                "Describe the image.<|im_end|>\n<|im_start|>assistant\n"
            ),
            "multi_modal_data": {
                "image": Image.new("RGB", (image_size, image_size), "red")
            },
        },
        SamplingParams(temperature=0, max_tokens=16, ignore_eos=True),
    )
    try:
        for _ in range(3):
            engine.step()
        assert engine.has_unfinished_requests()
        snapshots = llm.collective_rpc(_live_mm_payloads)
        assert all(snapshot for snapshot in snapshots)
        assert all(
            payloads == [False]
            for snapshot in snapshots
            for payloads in snapshot.values()
        ), snapshots
        if arena:
            sizes = llm.collective_rpc(_arena_registered_bytes)
            assert sizes and all(len(rank) == 1 for rank in sizes), sizes
            assert all(0 < rank[0] <= 32 * MiB_bytes for rank in sizes), sizes
        while engine.has_unfinished_requests():
            outputs = engine.step()
        assert len(outputs[0].outputs[0].token_ids) == 16
    finally:
        engine.abort_request(["cpu-input-lifetime"])
