# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests BitNet b1.58 checkpoints against the HF Transformers reference."""

import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm.model_executor.layers.quantization.bitnet import unpack_ternary_weight

from ..utils import check_logprobs_close


def test_unpack_matches_hf_packing():
    """The HF packing interleaves output rows; unpacking must invert it."""
    from transformers.integrations.bitnet import pack_weights

    weight = torch.randint(-1, 2, (64, 32), dtype=torch.int8)
    packed = pack_weights(weight.clone())
    assert packed.shape == (16, 32)
    torch.testing.assert_close(unpack_ternary_weight(packed), weight)


@pytest.mark.skipif(
    not is_quant_method_supported("bitnet"),
    reason="bitnet is not supported on this GPU type.",
)
@pytest.mark.parametrize(
    "model",
    [
        # BitNetForCausalLM with `AutoBitLinear` (scale multiplies the output).
        "microsoft/bitnet-b1.58-2B-4T",
        # LlamaForCausalLM with `BitLinear` (scale divides the output).
        "tiiuae/Falcon3-1B-Instruct-1.58bit",
    ],
)
@pytest.mark.parametrize("max_tokens", [16])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    # bitnet-b1.58-2B-4T's `auto_map` points at remote code that doesn't exist.
    with hf_runner(model, trust_remote_code=False) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs
        )
    with vllm_runner(
        model, max_model_len=2048, enforce_eager=True, trust_remote_code=False
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
