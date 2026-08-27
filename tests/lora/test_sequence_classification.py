# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import LLM, PoolingParams
from vllm.lora.request import LoRARequest

MODEL_NAME = "Qwen/Qwen3-0.6B"
NATIVE_MODEL_NAME = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
PROMPTS = {
    "star_trek": "Does warp drive appear in Star Trek?",
    "new_zealand": "Wellington is the capital of New Zealand.",
}
EXPECTED_OUT = {
    "star_trek": [13.6640625, -13.6640625],
    "new_zealand": [-4.80078125, 8.6171875],
    "native": [-1.52734375],
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


def test_converted_model_with_modules_to_save(
    qwen3_guard_star_trek_lora_files: str,
) -> None:
    prompt = PROMPTS["star_trek"]
    llm = LLM(
        model=MODEL_NAME,
        runner="pooling",
        convert="classify",
        hf_overrides={"num_labels": 2},
        dtype="float16",
        enable_lora=True,
        max_lora_rank=16,
        enforce_eager=True,
        max_model_len=512,
        gpu_memory_utilization=0.5,
    )
    actual = _classify_logits(
        llm,
        [prompt],
        LoRARequest("star-trek", 1, qwen3_guard_star_trek_lora_files),
    )[0]

    torch.testing.assert_close(
        actual,
        torch.tensor(EXPECTED_OUT["star_trek"]),
        atol=2e-2,
        rtol=2e-2,
    )


def test_native_classification_model_with_modules_to_save(
    skywork_qwen3_reward_lora_files: str,
) -> None:
    prompt = "Which response is more helpful and correct?"
    llm = LLM(
        model=NATIVE_MODEL_NAME,
        runner="pooling",
        dtype="float16",
        enable_lora=True,
        max_lora_rank=4,
        enforce_eager=True,
        max_model_len=512,
        gpu_memory_utilization=0.5,
    )
    actual = _classify_logits(
        llm,
        [prompt],
        LoRARequest("native", 1, skywork_qwen3_reward_lora_files),
    )[0]

    torch.testing.assert_close(
        actual,
        torch.tensor(EXPECTED_OUT["native"]),
        atol=2e-2,
        rtol=2e-2,
    )


def test_multiple_classification_loras_in_one_batch(
    qwen3_guard_star_trek_lora_files: str,
    qwen3_guard_new_zealand_lora_files: str,
) -> None:
    names = ["star_trek", "new_zealand"]
    prompts = [PROMPTS[name] for name in names]
    adapter_paths = [
        qwen3_guard_star_trek_lora_files,
        qwen3_guard_new_zealand_lora_files,
    ]
    requests = [
        LoRARequest(name, index, adapter_path)
        for index, (name, adapter_path) in enumerate(zip(names, adapter_paths), start=1)
    ]

    llm = LLM(
        model=MODEL_NAME,
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
        gpu_memory_utilization=0.5,
    )
    actual = _classify_logits(llm, prompts, requests)

    torch.testing.assert_close(
        actual,
        torch.tensor([EXPECTED_OUT[name] for name in names]),
        atol=2e-2,
        rtol=2e-2,
    )
