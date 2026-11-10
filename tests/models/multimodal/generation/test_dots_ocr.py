# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from PIL import Image, ImageDraw, ImageFont

from tests.models.utils import check_logprobs_close
from tests.utils import multi_gpu_test
from vllm.model_executor.models.dots_ocr import DotsVisionTransformer
from vllm.transformers_utils.configs.dotsocr import DotsVisionConfig
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backends.registry import AttentionBackendEnum


@pytest.mark.core_model
@torch.inference_mode()
def test_dots_ocr_vision_without_host_sync(dist_init):
    """Catch metadata copies and per-layer scalar reads under strict sync checks."""
    torch.manual_seed(0)
    grid_thw = [[2, 4, 6], [1, 6, 8]]
    config = DotsVisionConfig(
        embed_dim=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
    )
    with set_default_torch_dtype(torch.bfloat16):
        model = DotsVisionTransformer(config).cuda().eval()
    assert model.attn_backend in {
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.ROCM_AITER_FA,
        AttentionBackendEnum.TRITON_ATTN,
    }
    for parameter in model.parameters():
        parameter.normal_(mean=0, std=0.02)
    num_patches = sum(t * h * w for t, h, w in grid_thw)
    pixels = torch.randn(
        num_patches, 3 * config.patch_size**2, device="cuda", dtype=model.dtype
    )
    model(pixels, grid_thw)
    previous_mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        actual = model(pixels, grid_thw)
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)

    assert actual.shape == (
        num_patches // config.spatial_merge_size**2,
        config.hidden_size,
    )
    assert actual.isfinite().all()


@multi_gpu_test(num_gpus=2)
def test_dots_ocr_tensor_parallel_without_host_sync(vllm_runner, monkeypatch):
    """OCR must survive strict sync checks and preserve results across TP sizes."""
    monkeypatch.setenv("VLLM_GPU_SYNC_CHECK", "error")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    images = []
    texts = ["HELLO 1234", "TOTAL 56"]
    font = ImageFont.load_default(size=40)
    for size, text in zip([(448, 224), (224, 336)], texts):
        image = Image.new("RGB", size, "white")
        ImageDraw.Draw(image).text((20, 80), text, fill="black", font=font)
        images.append(image)

    prompt = (
        "<|user|><|img|><|imgpad|><|endofimg|>"
        "Read the text in this image.<|endofuser|><|assistant|>"
    )
    outputs = []
    for tp_size in (1, 2):
        with vllm_runner(
            "rednote-hilab/dots.ocr",
            tensor_parallel_size=tp_size,
            distributed_executor_backend="mp",
            dtype="bfloat16",
            max_model_len=1024,
            max_num_seqs=2,
            gpu_memory_utilization=0.35,
            kv_cache_memory_bytes=256 * 1024**2,
            enforce_eager=True,
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            limit_mm_per_prompt={"image": 1},
            mm_processor_kwargs={
                "size": {"shortest_edge": 56 * 56, "longest_edge": 448 * 448}
            },
        ) as runner:
            results = runner.generate_greedy_logprobs(
                [prompt] * len(images),
                max_tokens=32,
                num_logprobs=5,
                images=images,
            )
            for result, text in zip(results, texts):
                assert text in result[1]
            outputs.append(results)

    check_logprobs_close(
        outputs_0_lst=outputs[0],
        outputs_1_lst=outputs[1],
        name_0="TP=1",
        name_1="TP=2",
    )
