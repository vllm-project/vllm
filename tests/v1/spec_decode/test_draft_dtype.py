# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`SpeculativeConfig.dtype` lets a DFlash draft run in its own dtype.

Only config files are fetched from the (public) repos below.
"""

import pytest
import torch

from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig

DFLASH_TARGET = "Qwen/Qwen3-4B"
DFLASH_DRAFT = "z-lab/Qwen3-4B-DFlash-b16"  # bfloat16 checkpoint
EAGLE3_TARGET = "unsloth/Meta-Llama-3.1-8B-Instruct"
EAGLE3_DRAFT = "vllm-project/EAGLE3-LLaMA3.1-Instruct-8B"


def _dflash_spec(**kwargs) -> SpeculativeConfig:
    return SpeculativeConfig(
        target_model_config=ModelConfig(DFLASH_TARGET, dtype="float16"),
        target_parallel_config=ParallelConfig(),
        model=DFLASH_DRAFT,
        method="dflash",
        num_speculative_tokens=4,
        **kwargs,
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (None, torch.float16),  # the target's
        ("bfloat16", torch.bfloat16),
        ("auto", torch.bfloat16),  # the draft checkpoint's
    ],
)
def test_draft_dtype(dtype: str | None, expected: torch.dtype):
    spec = _dflash_spec(dtype=dtype)
    assert spec.target_model_config.dtype == torch.float16
    assert spec.draft_model_config.dtype == expected


@pytest.mark.cpu_test
def test_draft_dtype_rejected_for_other_methods():
    with pytest.raises(ValueError, match="dtype is not supported"):
        SpeculativeConfig(
            target_model_config=ModelConfig(EAGLE3_TARGET),
            target_parallel_config=ParallelConfig(),
            model=EAGLE3_DRAFT,
            method="eagle3",
            num_speculative_tokens=3,
            dtype="bfloat16",
        )
