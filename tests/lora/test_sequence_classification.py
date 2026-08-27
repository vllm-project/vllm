# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from vllm import LLM, PoolingParams
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.repo_utils import hf_api

MODEL_NAME = "Qwen/Qwen3-0.6B"
NATIVE_MODEL_NAME = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
ADAPTERS = {
    "star_trek": "geoffmunn/Qwen3Guard-StarTrek-Classification-0.6B",
    "new_zealand": "geoffmunn/Qwen3Guard-NewZealand-Classification-0.6B",
    "native": "AmirMohseni/skywork-qwen3-0.6b-reward-lora",
}
PROMPTS = {
    "star_trek": "Does warp drive appear in Star Trek?",
    "new_zealand": "Wellington is the capital of New Zealand.",
}


def _classify_logits(
    llm: LLM,
    prompts: list[str],
    requests: LoRARequest | list[LoRARequest],
) -> torch.Tensor:
    params = [PoolingParams(task="classify", use_activation=False) for _ in prompts]
    outputs = llm.classify(
        prompts,
        pooling_params=params,
        lora_request=requests,
        use_tqdm=False,
    )
    return torch.tensor([output.outputs.probs for output in outputs])


def _hf_logits(
    model_name: str,
    adapter_path: str,
    prompt: str,
    num_labels: int | None = None,
) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = {"num_labels": num_labels} if num_labels is not None else {}
    base = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        dtype=torch.float16,
        **kwargs,
    )
    model = PeftModel.from_pretrained(base, adapter_path).eval().to("cuda")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        logits = model(**inputs).logits.float().cpu()[0]
    del model, base
    torch.accelerator.empty_cache()
    return logits


@pytest.fixture(scope="module")
def adapter_paths() -> dict[str, str]:
    return {
        name: hf_api().snapshot_download(
            repo_id=repo_id,
            allow_patterns=["adapter_config.json", "adapter_model.safetensors"],
        )
        for name, repo_id in ADAPTERS.items()
    }


def test_converted_model_with_modules_to_save(
    vllm_runner,
    adapter_paths: dict[str, str],
) -> None:
    prompt = PROMPTS["star_trek"]
    expected = _hf_logits(
        MODEL_NAME,
        adapter_paths["star_trek"],
        prompt,
        num_labels=2,
    )

    with vllm_runner(
        MODEL_NAME,
        runner="pooling",
        convert="classify",
        hf_overrides={"num_labels": 2},
        dtype="float16",
        enable_lora=True,
        max_lora_rank=16,
        enforce_eager=True,
        max_model_len=512,
    ) as runner:
        actual = _classify_logits(
            runner.llm,
            [prompt],
            LoRARequest("star-trek", 1, adapter_paths["star_trek"]),
        )[0]

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_native_classification_model_with_modules_to_save(
    vllm_runner,
    adapter_paths: dict[str, str],
) -> None:
    prompt = "Which response is more helpful and correct?"
    expected = _hf_logits(
        NATIVE_MODEL_NAME,
        adapter_paths["native"],
        prompt,
    )

    with vllm_runner(
        NATIVE_MODEL_NAME,
        runner="pooling",
        dtype="float16",
        enable_lora=True,
        max_lora_rank=4,
        enforce_eager=True,
        max_model_len=512,
    ) as runner:
        actual = _classify_logits(
            runner.llm,
            [prompt],
            LoRARequest("native", 1, adapter_paths["native"]),
        )[0]

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_multiple_classification_loras_in_one_batch(
    vllm_runner,
    adapter_paths: dict[str, str],
) -> None:
    names = ["star_trek", "new_zealand"]
    prompts = [PROMPTS[name] for name in names]
    expected = torch.stack(
        [
            _hf_logits(
                MODEL_NAME,
                adapter_paths[name],
                PROMPTS[name],
                num_labels=2,
            )
            for name in names
        ]
    )
    requests = [
        LoRARequest(name, index, adapter_paths[name])
        for index, name in enumerate(names, start=1)
    ]

    with vllm_runner(
        MODEL_NAME,
        runner="pooling",
        convert="classify",
        hf_overrides={"num_labels": 2},
        dtype="float16",
        enable_lora=True,
        max_loras=2,
        max_cpu_loras=2,
        max_lora_rank=16,
        enforce_eager=True,
        max_model_len=512,
    ) as runner:
        actual = _classify_logits(runner.llm, prompts, requests)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
