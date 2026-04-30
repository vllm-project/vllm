# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E smoke tests for Qwen3-Next MTP speculative decoding on Blackwell (B200).

Qwen3-Next interleaves Gated Delta-Net (GDN) linear-attention layers with
full-attention layers.  The MTP drafter shares its hidden state with the last
full-attention block, exercising:
  * hybrid attention dispatch
  * GDN linear-attn TP-aware shard loading
  * Qwen3NextMTP weight-loading (stacked qkv/gate_up, expert params, GDN shards)
  * mamba_cache_mode="align" prefix-cache path

The 80B FP8 footprint fits only on a single B200 (>=140 GiB); every test here
is gated on compute-capability family 10.x (Blackwell) and >=140 GiB VRAM.

Model Runner V2 does not yet support hybrid-model MTP.  Each test skips when
VLLM_USE_V2_MODEL_RUNNER=1 so CI catches regressions on the V1 path without
spurious failures on the V2 path.

Historical context (recurring GDN / MTP bug surface):
  vllm#36242, vllm#34489, vllm#34697, vllm#28202, vllm#30433, vllm#25743
"""

import os

import pytest
import torch

from tests.utils import large_gpu_mark
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform

QWEN3_NEXT_FP8_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"

# Blackwell = compute-capability family 10.x (B200, B100, ...)
_IS_BLACKWELL = current_platform.is_device_capability_family(100)

_skip_non_blackwell = pytest.mark.skipif(
    not _IS_BLACKWELL,
    reason="Test requires Blackwell B200 (compute capability 10.x)",
)

# Hybrid-model MTP is not yet wired through the V2 Model Runner path.
_skip_v2_runner = pytest.mark.skipif(
    bool(int(os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "0"))),
    reason=(
        "Qwen3-Next hybrid-MTP is not yet supported on Model Runner V2; "
        "skip until hybrid-MTP support lands in that path"
    ),
)


@_skip_non_blackwell
@_skip_v2_runner
@large_gpu_mark(min_gb=140)
def test_mtp_hybrid_qwen3_next_smoke():
    """
    Smoke-load Qwen3-Next-80B-A3B-Instruct-FP8 with MTP speculative decoding
    and assert that exactly one token is produced without error.

    Exercises the full Qwen3-Next drafter stack:
    - Qwen3NextForCausalLM + Qwen3NextMTP weight loading
    - GDN linear-attn TP-aware shard loading (in_proj_qkvz, in_proj_ba)
    - Hybrid attention dispatch through the MTP forward pass
    - mamba_cache_mode="align" Mamba-state management
    """
    llm = None
    try:
        llm = LLM(
            model=QWEN3_NEXT_FP8_MODEL,
            speculative_config={
                "method": "mtp",
                "num_speculative_tokens": 1,
            },
            mamba_cache_mode="align",
            max_model_len=256,
            # Avoid CUDA-graph compilation overhead in a smoke test.
            enforce_eager=True,
            gpu_memory_utilization=0.95,
        )

        outputs = llm.generate(
            ["Hello"],
            SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
        )

        assert len(outputs) == 1, "Expected exactly one output sequence"
        assert len(outputs[0].outputs) == 1, "Expected exactly one completion"
        assert len(outputs[0].outputs[0].token_ids) == 1, (
            "Expected exactly 1 generated token from the 1-token MTP smoke test"
        )
    finally:
        if llm is not None:
            del llm
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()
