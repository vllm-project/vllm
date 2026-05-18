# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModel

from vllm.platforms import current_platform

from ....conftest import HfRunner
from ....utils import VLLM_PATH
from ...registry import HF_EXAMPLE_MODELS

if TYPE_CHECKING:
    from _typeshed import StrPath


FIXTURES_PATH = VLLM_PATH / "tests/models/fixtures"
assert FIXTURES_PATH.exists()
FIXTURE_REWARD_RESULT = {
    "Qwen/Qwen2.5-Math-PRM-7B": FIXTURES_PATH / "qwen2_5_math_prm_reward_step.json",
}


@pytest.fixture
def math_step_prompts():
    # ruff: noqa: E501
    data = {
        "system": "Please reason step by step, and put your final answer within \\boxed{}. ",
        "query": "Sue lives in a fun neighborhood.  One weekend, the neighbors decided to play a prank on Sue.  On Friday morning, the neighbors placed 18 pink plastic flamingos out on Sue's front yard.  On Saturday morning, the neighbors took back one third of the flamingos, painted them white, and put these newly painted white flamingos back out on Sue's front yard.  Then, on Sunday morning, they added another 18 pink plastic flamingos to the collection. At noon on Sunday, how many more pink plastic flamingos were out than white plastic flamingos?",
        "response": [
            "To find out how many more pink plastic flamingos were out than white plastic flamingos at noon on Sunday, we can break down the problem into steps. First, on Friday, the neighbors start with 18 pink plastic flamingos.",
            "On Saturday, they take back one third of the flamingos. Since there were 18 flamingos, (1/3 \\times 18 = 6) flamingos are taken back. So, they have (18 - 6 = 12) flamingos left in their possession. Then, they paint these 6 flamingos white and put them back out on Sue's front yard. Now, Sue has the original 12 pink flamingos plus the 6 new white ones. Thus, by the end of Saturday, Sue has (12 + 6 = 18) pink flamingos and 6 white flamingos.",
            "On Sunday, the neighbors add another 18 pink plastic flamingos to Sue's front yard. By the end of Sunday morning, Sue has (18 + 18 = 36) pink flamingos and still 6 white flamingos.",
            "To find the difference, subtract the number of white flamingos from the number of pink flamingos: (36 - 6 = 30). Therefore, at noon on Sunday, there were 30 more pink plastic flamingos out than white plastic flamingos. The answer is (\\boxed{30}).",
        ],
    }
    answer = "<extra_0>".join(data["response"]) + "<extra_0>"
    prompt = f"<im_start>system\n{data['system']}<im_end>\n<im_start>user\n{data['query']}<im_end>\n<im_start>assistant\n{answer}<im_end><|endoftext|>"
    return [prompt]


def step_reward_patch_hf_model(hf_model: HfRunner):
    # Patch the hf_runner to use the step reward function
    def make_step_rewards(
        logits: torch.Tensor, token_masks: torch.Tensor
    ) -> list[list[float]]:
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)

        all_scores_res: list[list[float]] = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]  # seq_len, num_labels
            positive_probs = sample[sample != 0].view(-1, 2)
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res

    def reward(prompts: list[str]) -> list[list[float]]:
        input_ids = hf_model.tokenizer(prompts, return_tensors="pt").input_ids
        input_ids = hf_model.wrap_device(input_ids)
        outputs = hf_model.model(input_ids=input_ids)

        step_sep_id = hf_model.tokenizer.encode("<extra_0>")[0]
        token_masks = input_ids == step_sep_id
        return make_step_rewards(outputs[0], token_masks)

    hf_model.reward = reward  # type: ignore[attr-defined]

    return hf_model


def dump_reward_outputs(outputs: list[list[float]], filename: "StrPath"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(outputs, f)


def load_reward_outputs(filename: "StrPath") -> list[list[float]]:
    with open(filename, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(
            "Qwen/Qwen2.5-Math-PRM-7B",
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
    ],
)
@pytest.mark.parametrize("dtype", ["half"])
def test_prm_models(
    hf_runner,
    vllm_runner,
    math_step_prompts,
    model: str,
    dtype: str,
) -> None:
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_transformers_version(on_fail="skip")

    if current_platform.is_cpu():
        pytest.skip("CPU only supports V1")

    with vllm_runner(model, max_model_len=1024, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.reward(math_step_prompts)

    with hf_runner(model, dtype=dtype, auto_cls=AutoModel) as hf_model:
        hf_model = step_reward_patch_hf_model(hf_model)
        hf_outputs = hf_model.reward(math_step_prompts)

    dump_reward_outputs(
        hf_outputs,
        FIXTURE_REWARD_RESULT[model],
    )

    # check logits difference
    for hf_output, vllm_output in zip(hf_outputs, vllm_outputs):
        hf_output = torch.tensor(hf_output).float()
        vllm_output = torch.tensor(vllm_output).float()

        assert torch.allclose(hf_output, vllm_output, 1.5e-2)


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(
            "Qwen/Qwen2.5-Math-PRM-7B",
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
    ],
)
@pytest.mark.parametrize("dtype", ["half"])
def test_prm_models_with_golden_outputs(
    vllm_runner,
    math_step_prompts,
    model: str,
    dtype: str,
) -> None:
    if not FIXTURE_REWARD_RESULT.get(model):
        pytest.skip(f"No available golden outputs for {model}.")

    with vllm_runner(model, max_model_len=1024, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.reward(math_step_prompts)

    golden_outputs = load_reward_outputs(FIXTURE_REWARD_RESULT[model])

    # check logits difference
    for golden_output, vllm_output in zip(golden_outputs, vllm_outputs):
        golden_output = torch.tensor(golden_output).float()
        vllm_output = torch.tensor(vllm_output).float()

        assert torch.allclose(golden_output, vllm_output, 1.5e-2)
