# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest

from vllm import SamplingParams
from vllm.config import CompilationConfig

from ..utils import compute_acceptance_len


def load_and_process_dataset(data_name: str):
    from datasets import load_dataset

    if data_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        prompt_fmt = (
            "{question}\nPlease reason step by step,"
            " and put your final answer within \\boxed{{}}."
        )
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    elif data_name == "mt-bench":
        dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        dataset = dataset.map(lambda x: {"turns": x["prompt"]})
    elif data_name == "humaneval":
        dataset = load_dataset("openai/openai_humaneval", split="test")
        prompt_fmt = (
            "Write a solution to the following problem and make sure"
            " that it passes the tests:\n```python\n{prompt}\n```"
        )
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    return dataset


def run_acceptance_length_eval(
    monkeypatch: pytest.MonkeyPatch,
    vllm_runner,
    spec_config: dict[str, Any],
    expected_acceptance_lengths: dict[str, float],
    chat_template_kwargs: dict[str, Any],
    use_mrv2: bool,
):
    """
    E2E acceptance-rate validation for speculative decoding.

    Drives one or more datasets (keyed in ``expected_acceptance_lengths``)
    through the spec decode engine and asserts the mean acceptance length
    stays within tolerance of the reference figure for each dataset.
    """
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1" if use_mrv2 else "0")
    runner_config = spec_config.copy()
    model = runner_config.pop("model")
    runner_config.setdefault("block_size", None)
    runner_config.setdefault("enable_chunked_prefill", None)
    runner_config.setdefault("compilation_config", CompilationConfig())
    with vllm_runner(model, **runner_config) as spec_runner:
        spec_llm = spec_runner.llm
        # MT-Bench has 80 prompts and HumanEval has 164; truncate GSM8K.
        max_prompts_per_dataset = 200

        tokenizer = spec_llm.get_tokenizer()
        for dataset_name, expected_len in expected_acceptance_lengths.items():
            dataset = load_and_process_dataset(dataset_name)
            prev_metrics = None
            acceptance_lengths = []
            for i in range(min(max_prompts_per_dataset, len(dataset))):
                user_content = dataset[i]["turns"][0]
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_content}],
                    tokenize=False,
                    add_generation_prompt=True,
                    **chat_template_kwargs,
                )

                # Greedy (temp=0) so acceptance length is deterministic and comparable
                # across runs.
                spec_llm.generate(
                    [prompt_text],
                    SamplingParams(temperature=0, max_tokens=2048),
                    use_tqdm=False,
                )
                current_metrics = spec_llm.get_metrics()
                acceptance_len = compute_acceptance_len(current_metrics, prev_metrics)
                prev_metrics = current_metrics
                acceptance_lengths.append(acceptance_len)

            mean_acceptance_length = sum(acceptance_lengths) / len(acceptance_lengths)
            # Fairly tight tolerance of 95% against the reference figures,
            # watching for regressions. Can be relaxed if test is flaky but be sure to
            # check for genuine issues such as #40727.
            expected_len = expected_len * 0.95
            print(
                f"acceptance_len for {dataset_name}: {mean_acceptance_length:.2f}"
                f" (expected at least {expected_len:.2f})"
            )

            assert mean_acceptance_length >= expected_len, (
                f"acceptance_len for {dataset_name} is below expected threshold: "
                f"{mean_acceptance_length:.2f} < {expected_len:.2f}"
            )
