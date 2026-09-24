# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the renderer's spawn/IPC boundary without downloading model weights."""

import asyncio
import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from vllm.config import MultiModalConfig
from vllm.inputs import mm_input
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing import ProcessorInputs
from vllm.renderers.hf import HfRenderer
from vllm.renderers.mm_process import (
    initialize_mm_process,
    validate_mm_process_inputs,
)

pytestmark = pytest.mark.skip_global_cleanup


@dataclass
class _ModelConfig:
    multimodal_config: MultiModalConfig
    renderer_num_workers: int = 1
    supports_multimodal_inputs: bool = True
    hf_config: SimpleNamespace = field(default_factory=SimpleNamespace)

    def get_multimodal_config(self):
        return self.multimodal_config


class _Processor:
    def __init__(self, barrier, started):
        self.barrier = barrier
        self.started = started
        self.warmed_up = False
        self.info = SimpleNamespace(
            allowed_mm_limits={"image": 1},
            parse_mm_data=MultiModalDataParser().parse_mm_data,
        )

    def get_dummy_mm_inputs(self, *args, **kwargs):
        self.warmed_up = True

    def apply(self, inputs, timing):
        assert self.warmed_up
        assert inputs.cache is None
        kwargs = inputs.hf_processor_mm_kwargs
        if kwargs.get("fail"):
            raise ValueError("invalid test media")
        if kwargs.get("wait_for_peer"):
            self.started.set()
            self.barrier.wait(timeout=30)
        with timing.record("apply_hf_processor"):
            pixels = torch.from_numpy(
                np.asarray(inputs.mm_data_items["image"].get(0)).copy()
            )
            if kwargs.get("non_cpu_output"):
                pixels = pixels.to("meta")
            item = MultiModalKwargsItem(
                {
                    "pixel_values": MultiModalFieldElem(
                        data=pixels, field=MultiModalBatchedField()
                    ),
                    "worker_pid": MultiModalFieldElem(
                        data=torch.tensor(os.getpid()), field=MultiModalBatchedField()
                    ),
                }
            )
        return mm_input(
            inputs.prompt,
            MultiModalKwargsItems({"image": [item]}),
            inputs.get_mm_hashes("test-model", "sha256"),
            {"image": [PlaceholderRange(offset=0, length=1)]},
        )


def _initialize_test_worker(config, tokenizer, warmup_barrier):
    with patch(
        "vllm.renderers.mm_process.MULTIMODAL_REGISTRY.create_processor",
        return_value=_Processor(config.barrier, config.started),
    ):
        initialize_mm_process(config, tokenizer, warmup_barrier)
    with config.initialized_workers.get_lock():
        config.initialized_workers.value += 1


def _initialize_failing_worker(config, tokenizer, warmup_barrier):
    with config.initialized_workers.get_lock():
        first_worker = config.initialized_workers.value == 0
        config.initialized_workers.value += 1
    if not first_worker:
        raise RuntimeError("worker initialization failed")
    _initialize_test_worker(config, tokenizer, warmup_barrier)
    config.started.set()


@contextmanager
def _renderer(
    workers,
    *,
    prefix_caching=False,
    initializer=_initialize_test_worker,
    processor_kwargs=None,
):
    config = SimpleNamespace(
        model_config=_ModelConfig(
            MultiModalConfig(
                mm_processor_num_workers=workers,
                mm_processor_cache_gb=0,
                mm_processor_kwargs=processor_kwargs,
            )
        ),
        parallel_config=SimpleNamespace(_api_process_rank=3, _api_process_count=1),
        cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching),
        observability_config=SimpleNamespace(enable_mm_processor_stats=True),
        scheduler_config=None,
        barrier=multiprocessing.get_context("spawn").Barrier(2),
        started=multiprocessing.get_context("spawn").Event(),
        initialized_workers=multiprocessing.get_context("spawn").Value("i", 0),
    )
    with (
        patch(
            "vllm.renderers.base.mm_registry.create_processor",
            return_value=_Processor(config.barrier, config.started),
        ),
        patch("vllm.renderers.base.initialize_mm_process", initializer),
    ):
        renderer = HfRenderer(config, None)
        try:
            renderer.warmup_mm()
            if workers > 1:
                assert config.initialized_workers.value == workers
            yield renderer
        finally:
            renderer.shutdown()


def _prompt(**kwargs):
    return {
        "prompt_token_ids": [1, 2],
        "prompt": "image prompt",
        "cache_salt": "tenant",
        "multi_modal_data": {"image": Image.new("RGB", (8, 8), color="red")},
        **kwargs,
    }


def _pid(result):
    return result["mm_kwargs"]["image"][0]["worker_pid"].data.item()


@pytest.mark.asyncio
async def test_process_workers_run_concurrently_and_preserve_request_identity():
    with _renderer(2) as renderer:
        prompt = _prompt(mm_processor_kwargs={"wait_for_peer": True})
        first, second = await asyncio.wait_for(
            asyncio.gather(
                renderer._process_tokens_async(prompt),
                renderer._process_tokens_async(prompt, skip_mm_cache=True),
            ),
            timeout=60,
        )
        assert _pid(first) != _pid(second)
        assert os.getpid() not in (_pid(first), _pid(second))
        assert first["mm_hashes"] != second["mm_hashes"]
        for result in (first, second):
            assert result["prompt"] == prompt["prompt"]
            assert result["cache_salt"] == prompt["cache_salt"]
            assert result["prompt_token_ids"] == [1, 2]
            assert result["mm_placeholders"]["image"] == [
                PlaceholderRange(offset=0, length=1)
            ]
            pixels = result["mm_kwargs"]["image"][0]["pixel_values"].data
            assert pixels.shape == (8, 8, 3)
            assert pixels[0, 0].tolist() == [255, 0, 0]
        stats = renderer._mm_timing_registry.stat()
        assert len(stats) == 2
        assert all(s["apply_hf_processor_secs"] > 0 for s in stats.values())


@pytest.mark.parametrize("prefix_caching", [False, True])
def test_process_workers_preserve_hashes_and_tensor_outputs(prefix_caching):
    prompts = [
        _prompt(media_io_kwargs={"image": {"format": "RGB"}}),
        _prompt(multi_modal_uuids={"image": ["explicit-id"]}),
    ]
    with _renderer(1, prefix_caching=prefix_caching) as serial:
        expected = [serial._process_tokens(prompt) for prompt in prompts]
        assert all(_pid(result) == os.getpid() for result in expected)
    with _renderer(2, prefix_caching=prefix_caching) as parallel:
        for prompt, reference in zip(prompts, expected):
            actual = parallel._process_tokens(prompt)
            assert _pid(actual) != os.getpid()
            assert actual["mm_hashes"] == reference["mm_hashes"]
            assert actual["mm_placeholders"] == reference["mm_placeholders"]
            assert (
                actual["mm_kwargs"]["image"][0]["pixel_values"]
                == reference["mm_kwargs"]["image"][0]["pixel_values"]
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"fail": True}, "invalid test media"),
        ({"non_cpu_output": True}, "CPU output tensors"),
    ],
)
async def test_worker_exception_propagates_without_breaking_subsequent_requests(
    kwargs, error
):
    with _renderer(2) as renderer:
        with pytest.raises(ValueError, match=error):
            await renderer._process_tokens_async(_prompt(mm_processor_kwargs=kwargs))
        result = await renderer._process_tokens_async(_prompt())
        assert result["prompt_token_ids"] == [1, 2]
        await renderer.clear_mm_cache_async()
        assert renderer.stat_mm_cache() is None


@pytest.mark.asyncio
async def test_cancelling_a_request_does_not_break_other_workers():
    with _renderer(2) as renderer:
        prompt = _prompt(mm_processor_kwargs={"wait_for_peer": True})
        first = asyncio.create_task(renderer._process_tokens_async(prompt))
        assert await asyncio.to_thread(renderer.config.started.wait, 30)
        second = asyncio.create_task(renderer._process_tokens_async(prompt))
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        result = await asyncio.wait_for(second, timeout=60)
        assert result["prompt_token_ids"] == [1, 2]


def test_shutdown_reaps_workers_and_rejects_new_work():
    with _renderer(2) as renderer:
        pid = _pid(renderer._process_tokens(_prompt()))
        assert pid in {child.pid for child in multiprocessing.active_children()}
    assert pid not in {child.pid for child in multiprocessing.active_children()}
    renderer.shutdown()
    with pytest.raises(RuntimeError, match="shutdown"):
        renderer._process_tokens(_prompt())


def test_initializer_failure_reaps_workers_waiting_for_warmup(monkeypatch):
    spawn_process = ProcessPoolExecutor.__dict__["_spawn_process"]
    connection_wait = multiprocessing.connection.wait
    spawning_second = threading.Event()
    manager_waiting = threading.Event()

    def wait_with_first_worker_only(objects, timeout=None):
        if spawning_second.is_set() and len(objects) == 3 and not objects[1].poll():
            manager_waiting.set()
        return connection_wait(objects, timeout)

    def spawn_after_manager_waits(executor):
        if executor._processes:
            config = executor._initializer.args[0]
            assert config.started.wait(timeout=30)
            spawning_second.set()
            executor._executor_manager_thread_wakeup.wakeup()
            assert manager_waiting.wait(timeout=30)
        spawn_process(executor)

    monkeypatch.setattr(multiprocessing.connection, "wait", wait_with_first_worker_only)
    monkeypatch.setattr(
        ProcessPoolExecutor, "_spawn_process", spawn_after_manager_waits
    )
    before = {child.pid for child in multiprocessing.active_children()}
    with (
        pytest.raises(BrokenProcessPool),
        _renderer(2, initializer=_initialize_failing_worker),
    ):
        pytest.fail("Failed worker initialization must not report readiness")
    assert {child.pid for child in multiprocessing.active_children()} == before


def test_process_workers_reject_non_cpu_input_tensors():
    inputs = ProcessorInputs(
        prompt=[1],
        mm_data_items=MultiModalDataParser().parse_mm_data(
            {"image": torch.empty((1, 2, 8), device="meta")}
        ),
    )
    with pytest.raises(ValueError, match="CPU input tensors"):
        validate_mm_process_inputs(inputs)


@pytest.mark.parametrize(
    "kwargs", [{"device": "cuda"}, {"images_kwargs": {"device": "cuda:0"}}]
)
def test_process_workers_reject_accelerator_request_overrides(kwargs):
    with (
        _renderer(2) as renderer,
        pytest.raises(ValueError, match="CPU"),
    ):
        renderer._process_tokens(_prompt(mm_processor_kwargs=kwargs))


@pytest.mark.parametrize("modality", ["images_kwargs", "videos_kwargs", "audio_kwargs"])
def test_renderer_rejects_accelerator_defaults_before_starting_workers(modality):
    with (
        pytest.raises(ValueError, match="CPU"),
        _renderer(2, processor_kwargs={modality: {"device": "cuda"}}),
    ):
        pytest.fail("Accelerator defaults must be rejected during initialization")


@pytest.mark.cpu_test
@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen3.5-4B"])
async def test_qwen_processor_outputs_match_across_process_boundary(model):
    from vllm.config import ModelConfig, VllmConfig
    from vllm.tokenizers.registry import cached_tokenizer_from_config

    results = []
    for workers in (1, 2):
        model_config = ModelConfig(
            model=model,
            max_model_len=4096,
            mm_processor_cache_gb=0,
            mm_processor_num_workers=workers,
            mm_processor_kwargs={"min_pixels": 28 * 28, "max_pixels": 28 * 28 * 4},
        )
        tokenizer = cached_tokenizer_from_config(model_config)
        renderer = HfRenderer(VllmConfig(model_config=model_config), tokenizer)
        try:
            results.append(
                await renderer._process_tokens_async(
                    _prompt(
                        prompt_token_ids=tokenizer.encode(
                            "<|vision_start|><|image_pad|><|vision_end|>",
                            add_special_tokens=False,
                        ),
                        multi_modal_uuids={"image": ["test-image"]},
                    )
                )
            )
        finally:
            renderer.shutdown()

    assert results[0] == results[1]
