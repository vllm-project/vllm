# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight updates through the public API, with and without EPLB."""

import pytest

from tests.utils import multi_gpu_marks
from vllm import SamplingParams
from vllm.config import WeightTransferConfig

MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"
PROMPTS = ["The capital of France is", "def fibonacci(n):"]
EPLB_KWARGS = {
    "enable_eplb": True,
    "eplb_config": {
        "num_redundant_experts": 0,
        "use_async": False,
        "window_size": 2,
        "step_interval": 4,
    },
}

pytestmark = multi_gpu_marks(num_gpus=2)


@pytest.fixture(params=[True, False], ids=["eplb", "ep_only"])
def enable_eplb(request):
    return request.param


def test_weight_updates_require_eplb_disabled(vllm_runner, enable_eplb):
    """Expert parallelism alone keeps updates; EPLB rejects them before any change."""
    with vllm_runner(
        MODEL,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        block_size=None,
        enable_prefix_caching=False,
        enforce_eager=True,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        weight_transfer_config=WeightTransferConfig(backend="nccl"),
        **(EPLB_KWARGS if enable_eplb else {}),
    ) as runner:
        llm = runner.llm
        sampling_params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)

        def generate():
            outputs = llm.generate(PROMPTS, sampling_params, use_tqdm=False)
            return [list(output.outputs[0].token_ids) for output in outputs]

        before = generate()
        if enable_eplb:
            for method in ("reload_weights", "start_weight_update"):
                with pytest.raises(Exception, match="models managed by EPLB"):
                    llm.collective_rpc(method)
            assert all(len(token_ids) == 8 for token_ids in generate())
        else:
            llm.collective_rpc("reload_weights")
            assert generate() == before
            llm.collective_rpc("start_weight_update")
            llm.collective_rpc("finish_weight_update")
            assert generate() == before
