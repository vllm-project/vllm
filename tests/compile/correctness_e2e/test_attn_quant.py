# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import logging

import pytest
import regex as re

from tests.utils import compare_two_settings
from vllm._aiter_ops import is_aiter_found_and_supported
from vllm.config import CompilationMode
from vllm.platforms import current_platform
from vllm.transformers_utils.repo_utils import hf_api


@pytest.mark.skipif(
    not current_platform.is_rocm() or not is_aiter_found_and_supported(),
    reason="Requires ROCm AITER attention and quantization",
)
def test_rocm_aiter_static_attention_output_correctness(caplog_mp_spawn):
    """Trained FP8 outputs agree when actual AITER output quantization is fused."""
    model = hf_api().snapshot_download(
        "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8",
        revision="870a1177ed082c0ba5dbefd62573ab8f91803f6a",
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"],
    )
    common_args = [
        "--dtype",
        "bfloat16",
        "--load-format",
        "auto",
        "--max-model-len",
        "512",
        "--max-num-batched-tokens",
        "512",
        "--max-num-seqs",
        "8",
        "--kv-cache-memory-bytes",
        str(256 * 1024**2),
        "--gpu-memory-utilization",
        "0.01",
        "--attention-backend",
        "ROCM_AITER_UNIFIED_ATTN",
    ]

    def settings(enabled):
        return [
            *common_args,
            "--compilation-config",
            json.dumps(
                {
                    "mode": CompilationMode.VLLM_COMPILE,
                    "cudagraph_mode": "NONE",
                    "splitting_ops": [],
                    "inductor_compile_config": {"force_disable_caches": True},
                    "custom_ops": ["+quant_fp8", "-rms_norm"],
                    "pass_config": {
                        "fuse_attn_quant": enabled,
                        "fuse_norm_quant": False,
                        "fuse_act_quant": False,
                        "fuse_allreduce_rms": False,
                    },
                }
            ),
        ]

    env = {
        "VLLM_ROCM_USE_AITER": "1",
        "VLLM_ROCM_USE_AITER_LINEAR": "1",
        "VLLM_DISABLE_COMPILE_CACHE": "1",
    }
    with caplog_mp_spawn(logging.DEBUG) as logs:
        compare_two_settings(
            model,
            settings(False),
            settings(True),
            env1=env,
            env2=env,
            method="generate",
            force_v1_runner=True,
        )
    matches = [
        int(count) for count in re.findall(r"'attn_quant_fusion': (\d+)", logs.text)
    ]
    print(f"Attention output quantization rewrite counts: {matches}")
    assert matches and all(count == 32 for count in matches), (
        f"Expected output quantization fused in all 32 Llama layers, found {matches}"
    )
