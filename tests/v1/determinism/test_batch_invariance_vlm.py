# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-invariance regression tests for vision-language models (VLMs)."""

import contextlib
import os

import numpy as np
import pytest
import torch
from PIL import Image
from utils import _extract_step_logprobs, skip_if_not_cuda

from vllm import LLM, SamplingParams

VLM_TEST_MODEL = os.getenv("VLLM_VLM_TEST_MODEL", "Qwen/Qwen3-VL-2B-Instruct")

IMAGE_PROMPT = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "Describe this image.<|im_end|><|im_start|>assistant\n"
)
VIDEO_PROMPT = (
    "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
    "Describe this video.<|im_end|><|im_start|>assistant\n"
)


def _make_image(seed: int, size: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


def _make_inputs(input_type: str, num_reqs: int) -> list[dict]:
    inputs = []
    for i in range(num_reqs):
        if input_type == "image":
            prompt = IMAGE_PROMPT
            mm_data = {"image": _make_image(1000 + i, size=256)}
        else:
            frames = [_make_image(1000 + i + f, size=128) for f in range(4)]
            prompt = VIDEO_PROMPT
            mm_data = {
                "video": (
                    frames,
                    {
                        "total_num_frames": len(frames),
                        "fps": 1.0,
                        "frames_indices": list(range(len(frames))),
                        "do_sample_frames": False,
                    },
                )
            }
        inputs.append({"prompt": prompt, "multi_modal_data": mm_data})
    return inputs


def _assert_batch_invariant(llm: LLM, inputs: list[dict], sampling) -> None:
    bs1 = [
        _extract_step_logprobs(llm.generate([inp], sampling, use_tqdm=False)[0])
        for inp in inputs
    ]
    bsN = [_extract_step_logprobs(out) for out in llm.generate(inputs, sampling)]

    failures = []
    for i, ((lp1, t1), (lpN, tN)) in enumerate(zip(bs1, bsN)):
        if lp1 is None or lpN is None:
            pytest.skip(
                "Logits are not available on RequestOutput; "
                "enable logprobs return to run this test."
            )
        if t1 != tN:
            failures.append(f"req {i}: token mismatch bs1={t1} bsN={tN}")
        elif not torch.equal(lp1, lpN):
            d = (lp1 - lpN).abs().max().item()
            failures.append(f"req {i}: logprob mismatch max_diff={d:.3e}")

    if failures:
        pytest.fail(
            f"Batch invariance violated for {len(failures)}/{len(inputs)} "
            f"requests:\n" + "\n".join(failures)
        )


@skip_if_not_cuda
@pytest.mark.parametrize("input_type", ["image", "video"])
@pytest.mark.parametrize("mm_encoder_attn_backend", ["FLASH_ATTN", "TORCH_SDPA"])
def test_vlm_batch_invariance_bs1_vs_bsN(input_type: str, mm_encoder_attn_backend: str):
    inputs = _make_inputs(input_type, num_reqs=4)
    sampling = SamplingParams(temperature=0.0, max_tokens=8, seed=1234, logprobs=5)

    llm = LLM(
        model=VLM_TEST_MODEL,
        dtype="bfloat16",
        enforce_eager=True,
        max_num_seqs=16,
        gpu_memory_utilization=0.85,
        mm_encoder_attn_backend=mm_encoder_attn_backend,
    )
    try:
        _assert_batch_invariant(llm, inputs, sampling)
    finally:
        with contextlib.suppress(Exception):
            llm.shutdown()
