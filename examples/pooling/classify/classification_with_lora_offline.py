# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example of offline classification with a LoRA adapter."""

from vllm import LLM
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.repo_utils import hf_api


def main():
    lora_path = hf_api().snapshot_download(
        repo_id="AmirMohseni/skywork-qwen3-0.6b-reward-lora"
    )
    llm = LLM(
        model="Skywork/Skywork-Reward-V2-Qwen3-0.6B",
        runner="pooling",
        enable_lora=True,
        max_lora_rank=8,
        max_model_len=512,
    )

    prompt = "Which response is more helpful and correct?"
    (output,) = llm.classify(
        prompt,
        lora_request=LoRARequest("reward-lora", 1, lora_path),
    )
    print(f"Prompt: {prompt!r}")
    print(f"Class Probabilities: {output.outputs.probs}")


if __name__ == "__main__":
    main()
