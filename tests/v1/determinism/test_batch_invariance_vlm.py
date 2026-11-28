# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-invariance regression tests for vision-language models (VLMs)."""

import contextlib
import os

import numpy as np
import pytest
import torch
from PIL import Image
from utils import _extract_step_logprobs, skip_unsupported

from vllm import LLM, SamplingParams
from vllm.inputs import TextPrompt
from vllm.platforms import current_platform
from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
from vllm.v1.attention.ops.vit_attn_wrappers import vit_flash_attn_wrapper

VLM_TEST_MODEL = os.getenv("VLLM_VLM_TEST_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
skip_if_not_vision_fa3 = pytest.mark.skipif(
    not current_platform.is_cuda() or get_flash_attn_version(head_size=64) != 3,
    reason="Requires vision FlashAttention 3 on Hopper",
)

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


def _make_inputs(input_type: str, num_reqs: int) -> list[TextPrompt]:
    inputs: list[TextPrompt] = []
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


def _assert_batch_invariant(llm: LLM, inputs: list[TextPrompt], sampling) -> None:
    bs1 = [
        _extract_step_logprobs(llm.generate([inp], sampling, use_tqdm=False)[0])
        for inp in inputs
    ]
    bsN_outputs = llm.generate(inputs, sampling, use_tqdm=False)
    assert len(bs1) == len(inputs)
    assert len(bsN_outputs) == len(inputs)
    bsN = [_extract_step_logprobs(out) for out in bsN_outputs]

    failures = []
    for i, ((lp1, t1), (lpN, tN)) in enumerate(zip(bs1, bsN)):
        if lp1 is None or lpN is None:
            failures.append(f"req {i}: selected-token logprobs are unavailable")
            continue
        if len(t1) != sampling.max_tokens or len(tN) != sampling.max_tokens:
            failures.append(
                f"req {i}: expected {sampling.max_tokens} tokens, "
                f"got bs1={len(t1)} bsN={len(tN)}"
            )
            continue
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


@skip_if_not_vision_fa3
def test_vision_fa3_batch_invariance() -> None:
    """Packing a shorter sequence must preserve the target's attention bits."""
    target_len = 1024
    packed_len = target_len + 256
    rng = torch.Generator(device="cuda").manual_seed(1234)
    qkv = [
        torch.randn(
            (1, packed_len, 16, 64), dtype=torch.bfloat16, device="cuda", generator=rng
        )
        for _ in range(3)
    ]
    max_seqlen = torch.tensor(target_len, dtype=torch.int32, device="cpu")

    outputs = []
    for offsets in ([0, target_len], [0, target_len, packed_len]):
        q, k, v = (tensor[:, : offsets[-1]] for tensor in qkv)
        outputs.append(
            vit_flash_attn_wrapper(
                q,
                k,
                v,
                batch_size=1,
                is_rocm_aiter=False,
                fa_version=3,
                cu_seqlens=torch.tensor(offsets, dtype=torch.int32, device="cuda"),
                max_seqlen=max_seqlen,
            )
        )
    solo, packed = outputs
    torch.testing.assert_close(
        solo.view(torch.int16), packed[:, :target_len].view(torch.int16), rtol=0, atol=0
    )


@skip_if_not_vision_fa3
def test_vision_fa3_mixed_image_batch_invariance(monkeypatch: pytest.MonkeyPatch):
    """A shorter image must preserve the larger image request's logprobs."""
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
    inputs: list[TextPrompt] = [
        {"prompt": IMAGE_PROMPT, "multi_modal_data": {"image": _make_image(seed, size)}}
        for seed, size in [(42, 512), (256, 256)]
    ]
    _run_vlm_batch_invariance(inputs, "FLASH_ATTN", enforce_eager=True)


@skip_unsupported
@pytest.mark.parametrize("input_type", ["image", "video"])
@pytest.mark.parametrize("mm_encoder_attn_backend", ["FLASH_ATTN", "TORCH_SDPA"])
def test_vlm_batch_invariance_bs1_vs_bsN(
    input_type: str,
    mm_encoder_attn_backend: str,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
    _run_vlm_batch_invariance(
        _make_inputs(input_type, num_reqs=4),
        mm_encoder_attn_backend,
        enforce_eager=True,
    )


@skip_unsupported
def test_vlm_batch_invariance_default_execution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
    _run_vlm_batch_invariance(
        _make_inputs("image", num_reqs=4), "FLASH_ATTN", enforce_eager=False
    )


def _run_vlm_batch_invariance(
    inputs: list[TextPrompt], mm_encoder_attn_backend: str, enforce_eager: bool
) -> None:
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=8,
        seed=1234,
        ignore_eos=True,
        logprobs=5,
    )

    llm = LLM(
        model=VLM_TEST_MODEL,
        dtype="bfloat16",
        enforce_eager=enforce_eager,
        max_num_seqs=16,
        max_model_len=4096,
        max_num_batched_tokens=2048,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        mm_encoder_attn_backend=mm_encoder_attn_backend,
    )
    try:
        _assert_batch_invariant(llm, inputs, sampling)
    finally:
        with contextlib.suppress(Exception):
            llm.llm_engine.engine_core.shutdown()
