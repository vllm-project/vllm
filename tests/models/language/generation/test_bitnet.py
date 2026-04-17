# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for BitNet b1.58 model generation.

BitNet requires special handling because:
1. The HF model uses AutoBitLinear (with ActQuant + WeightQuant) which is
   triggered by quantization_config in the model's config.json.
2. vLLM implements these quantization ops natively and must strip the
   quantization_config via hf_overrides to prevent vLLM from trying to
   use its own quantization framework.
3. The model's config.json contains auto_map references to custom Python
   files (configuration_bitnet.py, modeling_bitnet.py) that no longer
   exist in the HuggingFace repo. These must be stripped to prevent
   download errors.
"""

import pytest

from ...utils import check_logprobs_close

MODEL = "microsoft/bitnet-b1.58-2B-4T-bf16"


def strip_bitnet_config(config):
    """Remove quantization_config and auto_map from config.

    - quantization_config: prevents vLLM from using its quantization
      framework (BitNet handles quantization natively in the model).
    - auto_map: references custom files (configuration_bitnet.py,
      modeling_bitnet.py) that no longer exist in the HF repo.
    """
    if hasattr(config, 'quantization_config'):
        delattr(config, 'quantization_config')
    if hasattr(config, 'auto_map'):
        delattr(config, 'auto_map')
    return config


@pytest.mark.parametrize("model", [MODEL])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_bitnet_generation(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    """Test that BitNet vLLM output matches HuggingFace reference."""
    with hf_runner(model, dtype="bfloat16", trust_remote_code=False) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs
        )

    with vllm_runner(
        model,
        dtype="bfloat16",
        hf_overrides=strip_bitnet_config,
        enforce_eager=True,
        max_model_len=2048,
        trust_remote_code=False,
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
