# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from PIL import Image, ImageDraw, ImageFont

from examples.generate.unlimited_ocr2 import document_pages, page_windows
from vllm import SamplingParams


def _attention_class_names(model):
    from vllm.model_executor.layers.attention import Attention

    return [
        type(module).__name__
        for module in model.modules()
        if isinstance(module, Attention)
    ]


def _dash_backends(model):
    return [
        getattr(module, "dash_backend", "TRITON_ATTN")
        for module in model.modules()
        if type(module).__name__ == "UnlimitedOCR2Attention"
    ]


def _prefill_graph_segments(worker):
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper

    manager = getattr(worker.model_runner, "cudagraph_manager", None)
    if manager is not None:
        model = manager.breakable_cg_runner
    else:
        model = worker.model_runner.model
        while not isinstance(model, BreakableCUDAGraphWrapper):
            model = model.runnable
    assert isinstance(model, BreakableCUDAGraphWrapper)
    return [
        (
            entry.batch_descriptor.num_tokens,
            entry.capture.num_graphs,
            entry.capture.num_eager_breaks,
        )
        for entry in model.entries.values()
        if entry.capture is not None
    ]


@pytest.mark.core_model
@pytest.mark.parametrize("crop_mode", [False, True])
@pytest.mark.parametrize("backend", ["TRITON_ATTN", "FLASH_ATTN", "FLASHINFER"])
@pytest.mark.parametrize("graph_mode", ["eager", "decode", "prefill"])
def test_unlimited_ocr2_batched_pages(
    vllm_runner, graph_mode, backend, crop_mode, monkeypatch
):
    """Load DASH weights and preserve page content across batched requests."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    enforce_eager = graph_mode == "eager"
    prefill_graph = graph_mode == "prefill"
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", str(int(prefill_graph)))
    pages = []
    texts = ["The access code is ORANGE 2468.", "The invoice total is 125 dollars."]
    font = ImageFont.load_default(size=40)
    for title, text in zip(["Page One", "Page Two"], texts):
        image = Image.new("RGB", (1920, 1080) if crop_mode else (1024, 1024), "white")
        draw = ImageDraw.Draw(image)
        draw.text((70, 90), title, fill="black", font=font)
        draw.text((70, 200), text, fill="black", font=font)
        pages.append(image)

    with vllm_runner(
        "baidu/Unlimited-OCR-2",
        max_model_len=8192,
        max_num_batched_tokens=8192,
        max_num_seqs=2,
        kv_cache_memory_bytes=1024**3,
        gpu_memory_utilization=0.1,
        enforce_eager=enforce_eager,
        attention_config={"backend": backend},
        kernel_config={"enable_jit_warmup": False},
        compilation_config={
            "cudagraph_capture_sizes": [2, 1024, 4096] if prefill_graph else [2],
            **({"cudagraph_mode": "FULL_AND_PIECEWISE"} if prefill_graph else {}),
        },
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        mm_encoder_attn_backend="TORCH_SDPA",
        limit_mm_per_prompt={"image": 2},
        mm_processor_kwargs={"crop_mode": crop_mode},
    ) as runner:
        for specs in runner.llm.apply_model(_attention_class_names):
            assert specs and set(specs) == {"RSWAAttention"}
        for backends in runner.llm.apply_model(_dash_backends):
            assert backends and set(backends) == {backend}
        requests = [
            {
                "prompt": "<PAGE><image>" * len(images) + "Multi page parsing.",
                "multi_modal_data": {"image": images},
            }
            for images in [pages[:1], pages]
        ]
        sampling = SamplingParams(
            temperature=0, max_tokens=180, skip_special_tokens=False
        )
        outputs = runner.llm.generate(requests, sampling)
        if crop_mode:
            # Official 16-crop preset selects a 5x3 grid for 16:9 pages.
            assert len(outputs[1].prompt_token_ids) == 3613
        for output, expected in zip(outputs, [texts[:1], texts]):
            result = output.outputs[0].text
            assert result.startswith("<PAGE>")
            for text in expected:
                assert text in result

        # Reuse scheduler storage for a new request with a different first page.
        requests[0]["multi_modal_data"] = {"image": pages[1:]}
        result = runner.llm.generate(requests[:1], sampling)[0].outputs[0].text
        assert texts[1] in result
        assert "ORANGE" not in result

        # Three document pages traverse two bounded windows with one overlap.
        emitted = []
        image_map = {"first": pages[0], "second": pages[1]}
        for paths, context in page_windows(["first", "second", "first"], 2, 1):
            images = [image_map[path] for path in paths]
            request = {
                "prompt": "<PAGE><image>" * len(images) + "Multi page parsing.",
                "multi_modal_data": {"image": images},
            }
            output = runner.llm.generate([request], sampling)[0].outputs[0].text
            emitted.extend(document_pages(output, len(images), context))
        assert len(emitted) == 3
        for page, text in zip(emitted, [texts[0], texts[1], texts[0]]):
            assert text in page

        # Decode beyond the 128-token output window through the real scheduler.
        long_output = runner.llm.generate(
            requests[:1],
            SamplingParams(
                temperature=0,
                max_tokens=320,
                min_tokens=320,
                ignore_eos=True,
                skip_special_tokens=False,
            ),
        )[0].outputs[0]
        assert len(long_output.token_ids) == 320
        assert texts[1] in long_output.text

        if prefill_graph and not enforce_eager:
            segments = runner.llm.collective_rpc(_prefill_graph_segments)
            for entries in segments:
                assert any(
                    n >= 1024 and graphs > 1 and breaks >= 12
                    for n, graphs, breaks in entries
                )


@pytest.mark.core_model
@pytest.mark.parametrize("use_v2", [False, True])
def test_unlimited_ocr2_encoder_pressure_keeps_complete_prefills(
    vllm_runner, monkeypatch, use_v2
):
    """Queue distinct cropped pages beyond the encoder cache's token budget."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", str(int(use_v2)))
    requests = []
    expected = ["The invoice total is 125 dollars.", "The access code is ORANGE 2468."]
    font = ImageFont.load_default(size=40)
    for idx in range(8):
        pages = []
        for text in expected:
            image = Image.new("RGB", (1024, 1024), "white")
            draw = ImageDraw.Draw(image)
            draw.text((70, 200), text, fill="black", font=font)
            # Distinct cache keys while preserving the same visible document.
            draw.point((0, 0), fill=(idx, 0, 0))
            pages.append(image)
        requests.append(
            {
                "prompt": "<PAGE><image><PAGE><image>Multi page parsing.",
                "multi_modal_data": {"image": pages},
            }
        )
    with vllm_runner(
        "baidu/Unlimited-OCR-2",
        max_model_len=4096,
        max_num_batched_tokens=4096,
        max_num_seqs=8,
        kv_cache_memory_bytes=1024**3,
        gpu_memory_utilization=0.1,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        mm_encoder_attn_backend="TORCH_SDPA",
        limit_mm_per_prompt={"image": 2},
        mm_processor_kwargs={"crop_mode": True},
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8]},
        kernel_config={"enable_jit_warmup": False},
    ) as runner:
        outputs = runner.llm.generate(
            requests,
            SamplingParams(temperature=0, max_tokens=180, skip_special_tokens=False),
        )
        assert len(outputs) == len(requests)
        for output in outputs:
            for text in expected:
                assert text in output.outputs[0].text
