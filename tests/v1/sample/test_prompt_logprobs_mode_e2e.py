# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E check that prompt_logprobs respects logprobs_mode.

Kept in its own module: the engines here must not compete for GPU memory
with module-scoped LLM fixtures elsewhere in the suite.
"""

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

MODEL = "hmellor/tiny-random-LlamaForCausalLM"


def test_prompt_logprobs_mode_respected():
    """Prompt logprobs should not be identical for raw_logits vs raw_logprobs."""
    prompt = "Hello world"
    per_mode_value = {}

    for mode in ("raw_logits", "raw_logprobs"):
        llm = LLM(
            MODEL,
            enforce_eager=True,
            logprobs_mode=mode,
            gpu_memory_utilization=0.15,
        )
        output = llm.generate(
            prompt, SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
        )[0]
        assert output.prompt_logprobs is not None
        assert output.prompt_logprobs[1] is not None
        prompt_token_id = output.prompt_token_ids[1]
        per_mode_value[mode] = output.prompt_logprobs[1][prompt_token_id].logprob
        del llm
        cleanup_dist_env_and_memory()

    assert per_mode_value["raw_logits"] != per_mode_value["raw_logprobs"], (
        "prompt_logprobs should reflect logprobs_mode (logits vs logprobs)."
    )
