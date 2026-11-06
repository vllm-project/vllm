# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPLB with MoE layers on more than one pipeline stage, end to end."""

import pytest

from tests.models.utils import check_logprobs_close
from tests.utils import multi_gpu_test
from vllm.config import CompilationConfig
from vllm.distributed.eplb.eplb_state import EplbState

# Layer 0 is dense and the other three are MoE, so the default 2 + 2 split puts
# one MoE layer on the first stage and two on the second.
MODEL = "vllm-project/tiny-random-DeepseekV2ForCausalLM"
PROMPTS = [
    "The capital of France is",
    "The future of AI is",
    "1 + 1 =",
    "def fibonacci(n):",
]
MAX_TOKENS = 16
NUM_LOGPROBS = 5


def _stage_eplb_state(worker) -> tuple[int, bool]:
    """Return this stage's MoE layer count and whether EPLB moved its experts."""
    (state,) = worker.model_runner.eplb_state.model_states.values()
    model = state.model
    initial_map = EplbState.build_initial_global_physical_to_logical_map(
        model.num_routed_experts, model.num_redundant_experts
    )
    moved = state.physical_to_logical_map.tolist() != (
        [initial_map] * model.num_moe_layers
    )
    return model.num_moe_layers, moved


@multi_gpu_test(num_gpus=4)
def test_eplb_with_moe_layers_on_two_pipeline_stages(
    vllm_runner, monkeypatch: pytest.MonkeyPatch
):
    # Required to send _stage_eplb_state to the workers.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    engine_args = dict(
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        enable_expert_parallel=True,
        max_model_len=2048,
        block_size=None,
        enable_chunked_prefill=None,
        enable_prefix_caching=False,
        enforce_eager=True,
        compilation_config=CompilationConfig(),
    )

    with vllm_runner(MODEL, **engine_args) as runner:
        expected = runner.generate_greedy_logprobs(PROMPTS, MAX_TOKENS, NUM_LOGPROBS)

    with vllm_runner(
        MODEL,
        **engine_args,
        enable_eplb=True,
        # A small window and interval make EPLB rebalance mid-generation;
        # synchronous torch_nccl keeps the test independent of NIXL.
        eplb_config={
            "num_redundant_experts": 2,
            "window_size": 2,
            "step_interval": 4,
            "use_async": False,
            "communicator": "torch_nccl",
        },
    ) as runner:
        outputs = runner.generate_greedy_logprobs(PROMPTS, MAX_TOKENS, NUM_LOGPROBS)
        # Ranks 0 and 1 are the first stage, ranks 2 and 3 the second.
        stages = runner.llm.collective_rpc(_stage_eplb_state)

    assert stages == [(1, True), (1, True), (2, True), (2, True)]
    check_logprobs_close(
        outputs_0_lst=expected,
        outputs_1_lst=outputs,
        name_0="no_eplb",
        name_1="eplb",
    )
