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
def llm(monkeypatch):
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


def _live_mm_payloads(worker) -> dict[str, list[bool]]:
    cache = worker.model_runner.encoder_cache
    return {
        req_id: [feature.data is not None for feature in features]
        for req_id, features in cache.mm_features.items()
    }


@pytest.mark.core_model
def test_consumed_cpu_inputs_released_during_generation(llm) -> None:
    """An M-RoPE model must release pixels while its request is still decoding."""
    assert llm.llm_engine.vllm_config.use_v2_model_runner
    engine = llm.llm_engine
    engine.add_request(
        "cpu-input-lifetime",
        {
            "prompt": (
                "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                "Describe the image.<|im_end|>\n<|im_start|>assistant\n"
            ),
            "multi_modal_data": {"image": Image.new("RGB", (64, 64), "red")},
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
        while engine.has_unfinished_requests():
            outputs = engine.step()
        assert len(outputs[0].outputs[0].token_ids) == 16
    finally:
        engine.abort_request(["cpu-input-lifetime"])
