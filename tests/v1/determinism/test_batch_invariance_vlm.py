# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-invariance regression tests for vision-language models (VLMs).

Complements the text-only coverage in ``test_batch_invariance.py`` with
multimodal inputs (single image and video). The VLM vision tower is composed
almost entirely of linear layers and attention, so it shares the same cuBLAS
split-k and attention sources of non-determinism as the language model. These
tests assert that re-batching requests does not change the sampled tokens or
their logprobs when batch invariance is enabled (see ``conftest.py``).
"""

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


def _make_image(seed: int, size: int = 256) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


def _make_video(seed: int, num_frames: int = 4, size: int = 128) -> list[Image.Image]:
    return [_make_image(seed * 1000 + f, size) for f in range(num_frames)]


def _video_metadata(num_frames: int) -> dict:
    return {
        "total_num_frames": num_frames,
        "fps": 1.0,
        "frames_indices": list(range(num_frames)),
        "do_sample_frames": False,
    }


def _make_inputs(input_type: str, num_reqs: int) -> list[dict]:
    """Build ``num_reqs`` distinct prompts of the requested modality."""
    inputs = []
    for i in range(num_reqs):
        if input_type == "image":
            mm_data = {"image": _make_image(1000 + i)}
            prompt = IMAGE_PROMPT
        else:
            frames = _make_video(seed=1000 + i)
            mm_data = {"video": (frames, _video_metadata(len(frames)))}
            prompt = VIDEO_PROMPT
        inputs.append({"prompt": prompt, "multi_modal_data": mm_data})
    return inputs


def _extract(outputs) -> list[tuple[torch.Tensor, list[int]]]:
    """Return per-request (logprobs tensor, token ids) for all outputs."""
    results = []
    for out in outputs:
        step_logprobs, token_ids = _extract_step_logprobs(out)
        if step_logprobs is None:
            pytest.skip(
                "Logits are not available on RequestOutput; "
                "enable logprobs return to run this test."
            )
        results.append((step_logprobs, token_ids))
    return results


def _assert_batch_invariant(llm: LLM, inputs: list[dict], sampling) -> None:
    """Assert BS=1 and BS=N produce bitwise-identical logprobs and tokens."""
    bs1 = [
        _extract([llm.generate([inp], sampling, use_tqdm=False)[0]])[0]
        for inp in inputs
    ]
    bsN = _extract(llm.generate(inputs, sampling, use_tqdm=False))

    failures = []
    for i, ((lp1, t1), (lpN, tN)) in enumerate(zip(bs1, bsN)):
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


def _new_llm(**kwargs) -> LLM:
    return LLM(
        model=VLM_TEST_MODEL,
        dtype="bfloat16",
        enforce_eager=True,
        max_num_seqs=16,
        gpu_memory_utilization=0.85,
        **kwargs,
    )


@skip_if_not_cuda
@pytest.mark.parametrize("input_type", ["image", "video"])
def test_vlm_batch_invariance_bs1_vs_bsN(input_type: str):
    num_reqs = 8 if input_type == "image" else 4
    inputs = _make_inputs(input_type, num_reqs)
    sampling = SamplingParams(temperature=0.0, max_tokens=8, seed=1234, logprobs=5)

    llm = _new_llm()
    try:
        _assert_batch_invariant(llm, inputs, sampling)
    finally:
        with contextlib.suppress(Exception):
            llm.shutdown()


@skip_if_not_cuda
@pytest.mark.parametrize("mm_encoder_attn_backend", ["FLASH_ATTN", "TORCH_SDPA"])
def test_vlm_batch_invariance_vision_backends(mm_encoder_attn_backend: str):
    inputs = _make_inputs("image", 8)
    sampling = SamplingParams(temperature=0.0, max_tokens=8, seed=1234, logprobs=5)

    llm = _new_llm(mm_encoder_attn_backend=mm_encoder_attn_backend)
    try:
        _assert_batch_invariant(llm, inputs, sampling)
    finally:
        with contextlib.suppress(Exception):
            llm.shutdown()
