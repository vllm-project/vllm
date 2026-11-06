# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EP weight filtering with EPLB, end to end on a real checkpoint."""

import logging

import pytest

from tests.utils import multi_gpu_marks
from vllm.config import CompilationConfig
from vllm.platforms import current_platform

# Already used by other CI tests. Its 72 routed experts are stored per expert;
# with EP2 and 2 redundant experts each rank holds 37 slots, and rank 1's
# redundant slots replicate experts 0 and 1 from rank 0's plain EP shard.
MODEL = "vllm-project/deepseek_mtp_main_random"
PROMPTS = [
    "The capital of France is",
    "The future of AI is",
    "1 + 1 =",
    "def fibonacci(n):",
]
MAX_TOKENS = 16
FILTER_LOG = "EP weight filter: ep_size=2, ep_rank=0, loading 37/72 experts"

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Exact comparison needs VLLM_BATCH_INVARIANT, unsupported for MLA "
    "on this platform.",
)


@pytest.mark.parametrize(
    ("pipeline_parallel_size", "load_format"),
    [
        pytest.param(1, "auto", id="tp2", marks=multi_gpu_marks(num_gpus=2)),
        # Only the second stage has experts to filter, and InstantTensor must
        # not load collectively on the first stage alone.
        pytest.param(
            2, "instanttensor", id="tp2-pp2", marks=multi_gpu_marks(num_gpus=4)
        ),
    ],
)
def test_ep_weight_filter_matches_unfiltered_with_eplb(
    vllm_runner,
    caplog_mp_spawn,
    monkeypatch: pytest.MonkeyPatch,
    pipeline_parallel_size: int,
    load_format: str,
):
    # Batch invariance makes the two runs comparable token for token, and
    # spawned workers let caplog_mp_spawn read their logs.
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if pipeline_parallel_size > 1:
        # Only layer 0 is dense, so the first stage has no MoE layers and takes
        # no part in EPLB.
        monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "1,4")

    outputs = {}
    for enable_ep_weight_filter in (False, True):
        with (
            caplog_mp_spawn(logging.INFO) as log_holder,
            vllm_runner(
                MODEL,
                tensor_parallel_size=2,
                pipeline_parallel_size=pipeline_parallel_size,
                load_format=load_format,
                enable_expert_parallel=True,
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
                enable_ep_weight_filter=enable_ep_weight_filter,
                max_model_len=2048,
                block_size=None,
                enable_chunked_prefill=None,
                enable_prefix_caching=False,
                enforce_eager=True,
                compilation_config=CompilationConfig(),
            ) as runner,
        ):
            outputs[enable_ep_weight_filter] = runner.generate_greedy(
                PROMPTS, MAX_TOKENS
            )
        assert (FILTER_LOG in log_holder.text) == enable_ep_weight_filter, (
            f"expected the filter {'on' if enable_ep_weight_filter else 'off'}"
        )

    assert outputs[True] == outputs[False]
