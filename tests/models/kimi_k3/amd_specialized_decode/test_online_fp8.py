# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online FP8 linears with the Kimi-K3 specialized decode: the weight check
refuses FP8 weights in the layers K1 / K2 read, so the deployment needs
`specialized_decode_bf16_copies`."""

import json

from tests.models.kimi_k3.amd_specialized_decode.test_engine_fault import (
    MODEL,
    SERVER_ARGS,
)
from tests.models.kimi_k3.amd_specialized_decode.test_numerics import needs_gfx950
from tests.utils import RemoteOpenAIServer, multi_gpu_test

STARTUP_TIMEOUT = 1800
MAX_TOKENS = 64
ONLINE_FP8 = {
    "linear": "fp8_per_channel",
    "ignore": ["*vision_tower*", "*mm_projector*", "*.mlp.gate"],
}
ENV = {
    "VLLM_ROCM_USE_AITER": "1",
    "VLLM_ROCM_AITER_MLA_ASM_PADDING": "asm",
    "VLLM_ROCM_USE_AITER_FP4BMM": "1",
}


def _server_args(additional_config: dict) -> list[str]:
    at = SERVER_ARGS.index("--additional-config")
    return [
        *SERVER_ARGS[: at + 1],
        json.dumps(additional_config),
        *SERVER_ARGS[at + 2 :],
        "--quantization-config",
        json.dumps(ONLINE_FP8),
    ]


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_online_fp8_serves_with_the_specialized_decode():
    args = _server_args(
        {
            "decode_backend": "latency_specialized",
            "specialized_decode_bf16_copies": True,
        }
    )
    with RemoteOpenAIServer(
        MODEL, args, env_dict=ENV, max_wait_seconds=STARTUP_TIMEOUT
    ) as server:
        completion = server.get_client().completions.create(
            model=MODEL,
            prompt="hello " * 64,
            max_tokens=MAX_TOKENS,
            temperature=0,
            extra_body={"ignore_eos": True},
        )
    assert completion.usage.completion_tokens == MAX_TOKENS
