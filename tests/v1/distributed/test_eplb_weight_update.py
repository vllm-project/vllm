# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight updates through the public API, with and without EPLB."""

import pytest

from tests.utils import multi_gpu_marks
from vllm import SamplingParams
from vllm.config import WeightTransferConfig

ENGINE_KWARGS = {
    "tensor_parallel_size": 2,
    "enable_expert_parallel": True,
    "block_size": None,
    "enable_prefix_caching": False,
    "enforce_eager": True,
    "trust_remote_code": True,
    "max_model_len": 2048,
    "gpu_memory_utilization": 0.85,
    "weight_transfer_config": WeightTransferConfig(backend="nccl"),
}
MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"
PROMPTS = ["The capital of France is", "def fibonacci(n):"]

pytestmark = multi_gpu_marks(num_gpus=2)


def test_expert_parallel_without_eplb_accepts_weight_updates(vllm_runner):
    with vllm_runner(MODEL, **ENGINE_KWARGS) as runner:
        llm = runner.llm
        sampling_params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)

        def generate():
            outputs = llm.generate(PROMPTS, sampling_params, use_tqdm=False)
            return [list(output.outputs[0].token_ids) for output in outputs]

        before = generate()
        llm.collective_rpc("reload_weights")
        assert generate() == before
        llm.collective_rpc("start_weight_update")
        llm.collective_rpc("finish_weight_update")
        assert generate() == before


def test_eplb_rejects_weight_updates(vllm_runner):
    with vllm_runner(
        MODEL,
        enable_eplb=True,
        eplb_config={
            "num_redundant_experts": 0,
            "use_async": False,
            "window_size": 2,
            "step_interval": 4,
        },
        **ENGINE_KWARGS,
    ) as runner:
        for method in ("reload_weights", "start_weight_update"):
            with pytest.raises(Exception, match="models managed by EPLB"):
                runner.llm.collective_rpc(method)
