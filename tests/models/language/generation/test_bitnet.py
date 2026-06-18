# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for BitNet b1.58 model generation.

Two variants are supported:
  - BF16 (online): microsoft/bitnet-b1.58-2B-4T-bf16
      Weights stored as bfloat16, quantized at load time via weight_quant().
      Config has no quantization_config, so we strip auto_map only.
  - Packed (offline): microsoft/bitnet-b1.58-2B-4T
      Weights stored as packed uint8 (4 ternary values per byte).
      Config has quantization_config with quant_method="bitnet",
      routed to BitNetBitBLASConfig by vLLM's quantization framework.
      We only strip auto_map (keep quantization_config for the backend).
"""

import pytest

from ...utils import check_logprobs_close

BF16_MODEL = "microsoft/bitnet-b1.58-2B-4T-bf16"
PACKED_MODEL = "microsoft/bitnet-b1.58-2B-4T"


def strip_bf16_config(config):
    """Strip quantization_config and auto_map for BF16 variant.

    - quantization_config: prevents vLLM from using its quantization
      framework (BF16 variant handles quantization in the model).
    - auto_map: references custom files that no longer exist in HF repo.
    """
    if hasattr(config, 'quantization_config'):
        delattr(config, 'quantization_config')
    if hasattr(config, 'auto_map'):
        delattr(config, 'auto_map')
    return config


def strip_packed_config(config):
    """Strip only auto_map for packed variant.

    Keep quantization_config so vLLM routes to BitNetBitBLASConfig.
    Only strip auto_map (custom files no longer exist in HF repo).
    """
    if hasattr(config, 'auto_map'):
        delattr(config, 'auto_map')
    return config


@pytest.mark.parametrize("model", [BF16_MODEL])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_bitnet_bf16_generation(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    """Test that BitNet BF16 vLLM output matches HuggingFace reference."""
    with hf_runner(model, dtype="bfloat16",
                   trust_remote_code=False) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs
        )

    with vllm_runner(
        model,
        dtype="bfloat16",
        hf_overrides=strip_bf16_config,
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


@pytest.mark.parametrize("model", [PACKED_MODEL])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_bitnet_packed_generation(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    """Test that BitNet packed (uint8) vLLM output matches HuggingFace.

    The packed model stores ternary weights in 2-bit packed uint8 format.
    vLLM uses BitNetBitBLASConfig to unpack at load and apply weight_scale.
    """
    with hf_runner(model, dtype="bfloat16",
                   trust_remote_code=False) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs
        )

    with vllm_runner(
        model,
        dtype="bfloat16",
        hf_overrides=strip_packed_config,
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
