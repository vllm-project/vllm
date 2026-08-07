# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
from collections.abc import Iterable
from typing import Any

import pytest
import torch

from tests.evals.gsm8k.gsm8k_eval import _build_gsm8k_prompts, evaluate_gsm8k_offline
from vllm import LLM, SamplingParams
from vllm.assets.base import VLLM_S3_BUCKET_URL
from vllm.assets.image import VLM_IMAGES_DIR
from vllm.v1.metrics.reader import Metric


def _skip_if_insufficient_gpus_for_tp(tp_size: int):
    """Skip test if available GPUs < tp_size on ROCm."""
    available_gpus = torch.accelerator.device_count()
    if available_gpus < tp_size:
        pytest.skip(
            f"Test requires {tp_size} GPUs, but only {available_gpus} available"
        )


Messages = list[dict[str, Any]]


def get_test_prompts(mm_enabled: bool, num_prompts: int = 100) -> list[Messages]:
    prompt_types = ["repeat", "gsm8k"]
    if mm_enabled:
        prompt_types.append("mm")
    prompts: list[Messages] = []

    num_repeat_prompts = num_prompts // len(prompt_types)
    if mm_enabled:
        num_gsm8k_prompts = num_prompts // len(prompt_types)
        num_mm_prompts = num_prompts - num_repeat_prompts - num_gsm8k_prompts
    else:
        num_mm_prompts = 0
        num_gsm8k_prompts = num_prompts - num_repeat_prompts

    # Generate a mixed batch of prompts, some of which can be easily
    # predicted by n-gram matching and some which likely cannot.
    random.seed(0)
    for _ in range(num_repeat_prompts):
        word_choices = ["test", "temp", "hello", "where"]
        word = random.choice(word_choices)
        prompts.append(
            [
                {
                    "role": "user",
                    "content": f"""
        please repeat the word '{word}' 10 times.
        give no other output than the word at least ten times in a row,
        in lowercase with spaces between each word and without quotes.
        """,
                }
            ]
        )
    prompts.extend(
        [{"role": "user", "content": prompt}]
        for prompt in _build_gsm8k_prompts(
            num_questions=num_gsm8k_prompts, num_shots=5
        )[0]
    )
    for _ in range(num_mm_prompts):
        placeholders = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"{VLLM_S3_BUCKET_URL}/{VLM_IMAGES_DIR}/stop_sign.jpg"
                },
            }
        ]
        prompt = [
            *placeholders,
            {"type": "text", "text": "The meaning of the image is"},
        ]
        prompts.append([{"role": "user", "content": prompt}])

    return prompts


def get_instruct_coder_messages(n: int) -> list[Messages]:
    from vllm.benchmarks.datasets import InstructCoderDataset

    dataset = InstructCoderDataset(
        dataset_path="likaixin/InstructCoder", dataset_split="train"
    )
    prompts: Iterable[str] = dataset.sample_prompts(n=n)
    return [[{"role": "user", "content": prompt}] for prompt in prompts]


def greedy_sampling() -> SamplingParams:
    return SamplingParams(temperature=0, max_tokens=10, ignore_eos=False)


def stochastic_sampling() -> SamplingParams:
    return SamplingParams(temperature=1.0, max_tokens=10, ignore_eos=False)


def evaluate_llm_for_gsm8k(llm: LLM, expected_accuracy_threshold: float = 0.70) -> None:
    """Evaluate the LLM on GSM8K and check that accuracy is above a sanity threshold.

    The default threshold assumes the LLM uses the same target model as the "model_name"
    fixture, with max model len == 4096. Precomputed reference value is 75% to 80%
    on GSM8K with greedy decoding, so we check that it's above a sanity threshold of 70%
    to verify that the model is correct.
    """
    if expected_accuracy_threshold <= 0.0:
        print("Skipping GSM8K evaluation")
        return
    results = evaluate_gsm8k_offline(llm)
    accuracy = results["accuracy"]
    print(f"GSM8K accuracy: {accuracy:.3f}")
    assert accuracy >= expected_accuracy_threshold, (
        f"Expected GSM8K accuracy >= {expected_accuracy_threshold}, got {accuracy:.3f}"
    )


def compute_acceptance_rate(
    metrics: list[Metric], prev_metrics: list[Metric] | None = None
) -> float:
    name2metric = {metric.name: metric for metric in metrics}
    n_draft_toks = name2metric["vllm:spec_decode_num_draft_tokens"].value
    if n_draft_toks == 0:
        return float("nan")
    n_accepted_toks = name2metric["vllm:spec_decode_num_accepted_tokens"].value
    if prev_metrics is not None:
        prev_name2metric = {metric.name: metric for metric in prev_metrics}
        n_draft_toks -= prev_name2metric["vllm:spec_decode_num_draft_tokens"].value
        n_accepted_toks -= prev_name2metric[
            "vllm:spec_decode_num_accepted_tokens"
        ].value
        if n_draft_toks <= 0:
            return float("nan")
    return n_accepted_toks / n_draft_toks


def compute_acceptance_len(
    metrics: list[Metric], prev_metrics: list[Metric] | None = None
) -> float:
    name2metric = {metric.name: metric for metric in metrics}
    n_drafts = name2metric["vllm:spec_decode_num_drafts"].value
    n_accepted_toks = name2metric["vllm:spec_decode_num_accepted_tokens"].value
    if n_drafts == 0:
        return 1
    if prev_metrics is not None:
        prev_name2metric = {metric.name: metric for metric in prev_metrics}
        n_drafts -= prev_name2metric["vllm:spec_decode_num_drafts"].value
        n_accepted_toks -= prev_name2metric[
            "vllm:spec_decode_num_accepted_tokens"
        ].value
        if n_drafts <= 0:
            return 1
    return 1 + (n_accepted_toks / n_drafts)
