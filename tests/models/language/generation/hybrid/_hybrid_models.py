# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared by hybrid/test_hybrid.py and ../test_granite_4_hybrid.py.

The leading underscore keeps pytest from collecting this module.
"""

from tests.models.registry import HF_EXAMPLE_MODELS
from tests.models.utils import check_logprobs_close
from vllm.platforms import current_platform

# Avoid OOM
MAX_NUM_SEQS = 4

ATTN_BACKEND = "TRITON_ATTN" if current_platform.is_rocm() else "auto"


def check_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    try:
        model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
        model_info.check_available_online(on_fail="skip")
        model_info.check_transformers_version(on_fail="skip")
    except ValueError:
        pass

    with hf_runner(model) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs
        )

    with vllm_runner(
        model,
        max_num_seqs=MAX_NUM_SEQS,
        attention_backend=ATTN_BACKEND,
        enable_chunked_prefill=True,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs
        )

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
