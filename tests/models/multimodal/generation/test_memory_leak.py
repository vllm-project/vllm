# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import random
import string
import sys
from pathlib import Path

import pytest
import regex as re
import torch
from PIL import Image

from tests.models.registry import HF_EXAMPLE_MODELS
from vllm import SamplingParams
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

if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx9

    ATTENTION_BACKENDS = [
        "TRITON_ATTN",
        "ROCM_ATTN",
        pytest.param(
            "ROCM_AITER_FA",
            marks=pytest.mark.skipif(not on_gfx9(), reason="AITER needs gfx9"),
        ),
    ]
else:
    ATTENTION_BACKENDS = [None]

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


_DRM_SIZE_UNITS = {"": 1, "KiB": KiB_bytes, "MiB": MiB_bytes}


def _rocm_process_vram_bytes() -> int:
    """VRAM of this process's amdgpu clients, from DRM client usage stats.

    Unlike PyTorch allocator stats, this also counts device memory allocated
    outside PyTorch. See https://docs.kernel.org/gpu/drm-usage-stats.html.
    """
    vram_by_client: dict[str, int] = {}
    for fdinfo in Path("/proc/self/fdinfo").iterdir():
        try:
            stats = dict(re.findall(r"^([\w-]+):\s*(.*)$", fdinfo.read_text(), re.M))
        except OSError:
            continue
        if stats.get("drm-driver") != "amdgpu":
            continue
        # drm-memory-vram counts only resident memory, so eviction would move it.
        size, _, unit = stats["drm-total-vram"].partition(" ")
        # Duplicated fds share one client, identified by its GPU VM.
        vram_by_client[stats["pasid"]] = int(size) * _DRM_SIZE_UNITS[unit]
    assert vram_by_client, "no amdgpu DRM client in /proc/self/fdinfo"
    return sum(vram_by_client.values())


def _gpu_memory_bytes() -> tuple[int, int]:
    # Process-local: device-wide used memory also counts other processes on
    # the GPU, and even a bare HIP context adds 2 MiB to it.
    torch.accelerator.synchronize()
    if current_platform.is_rocm():
        return torch.accelerator.memory_allocated(), _rocm_process_vram_bytes()
    return torch.accelerator.memory_allocated(), torch.accelerator.memory_reserved()


def _format_mib(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "n/a"

    return f"{format_mib(num_bytes)} MiB"


@pytest.fixture
def attention_backend() -> str | None:
    return None


@pytest.fixture(scope="function")
def llm(vllm_runner, monkeypatch, attention_backend):
    # In process, so the tests can measure memory and read the worker's state.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    with vllm_runner(
        maybe_model_redirect(MODEL_NAME),
        enforce_eager=True,
        enable_chunked_prefill=None,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=REQUESTS_PER_ROUND,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.8,
        attention_backend=attention_backend,
    ) as runner:
        yield runner.llm


@pytest.mark.core_model
@pytest.mark.parametrize("attention_backend", ATTENTION_BACKENDS)
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
    warmup_gpu = _gpu_memory_bytes()
    warmup_cpu_peak = _ru_maxrss_bytes()

    post_warmup_gpu_samples: list[tuple[int, int]] = []
    post_warmup_cpu_peak_samples: list[int] = []

    for _ in range(MEASURED_ROUNDS):
        outputs = llm.chat(request_batch, sampling_params=SAMPLING_PARAMS)
        assert len(outputs) == len(request_batch)
        assert llm.llm_engine.get_num_unfinished_requests() == 0
        del outputs

        gc.collect()
        post_warmup_gpu_samples.append(_gpu_memory_bytes())
        cpu_peak = _ru_maxrss_bytes()
        if cpu_peak is not None:
            post_warmup_cpu_peak_samples.append(cpu_peak)

    gpu_threshold = GPU_GROWTH_THRESHOLD_MIB * MiB_bytes
    for i, kind in enumerate(("allocated", "reserved")):
        samples = [sample[i] for sample in post_warmup_gpu_samples]
        gpu_growth = max(samples) - warmup_gpu[i]

        assert gpu_growth <= gpu_threshold, (
            f"Qwen3-VL {kind} GPU memory kept growing after warmup. "
            f"warmup_baseline={_format_mib(warmup_gpu[i])}, "
            f"post_warmup_samples={[_format_mib(x) for x in samples]}, "
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
