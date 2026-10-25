# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test Dual Batch Overlap (DBO) with Data Parallelism + Expert Parallelism.

DBO is specifically designed for DP+EP scenarios to hide communication latency
by overlapping computation of two batches. This test validates that DBO works
correctly with the DeepSeek-V2-Lite model using GSM8K evaluation.
"""

import asyncio
import os

import pytest
import torch

from tests.models.utils import check_logprobs_close
from tests.utils import RemoteOpenAIServer
from vllm import SamplingParams
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.outputs import RequestOutput
from vllm.platforms import current_platform
from vllm.sampling_params import RequestOutputKind
from vllm.utils.import_utils import has_deep_ep
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.loggers import StatLoggerBase

# Detect Blackwell / B200 (compute capability 10.x)
try:
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        IS_BLACKWELL = cap[0] >= 10
    else:
        IS_BLACKWELL = False
except Exception:
    # Be conservative: if we can't detect, don't xfail by default
    IS_BLACKWELL = False

MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite-Chat"
DP_SIZE = 2

# GSM8K eval configuration
NUM_QUESTIONS = 256  # Fast eval for CI; but must be large enough to hit dbo thresholds
NUM_SHOTS = 5  # Few-shot examples

# Expected 0.64 with 2% buffer (based on vLLM test data)
# On ROCm, widened to 0.61 after observing 0.617 amd-ci nightly flake twice.
MIN_ACCURACY = 0.61 if current_platform.is_rocm() else 0.62

# Increase max_num_seqs to trigger DBO for decode batches
# With 64 seqs, decode batches should exceed the 32 token threshold
MAX_NUM_SEQS = 64  # Increased from 16 to trigger decode DBO

# DeepEP backends to test
DEEPEP_BACKENDS = [
    "deepep_low_latency",
    "deepep_high_throughput",
]


@pytest.mark.skipif(not has_deep_ep(), reason="These tests require deep_ep to run")
@pytest.mark.parametrize("all2all_backend", DEEPEP_BACKENDS)
@pytest.mark.xfail(
    IS_BLACKWELL,
    reason=(
        "Temporary: DBO accuracy unstable on Blackwell "
        "(doesn't meet expectation of MIN_ACCURACY = 0.62)"
    ),
)
def test_dbo_dp_ep_gsm8k(all2all_backend: str, num_gpus_available):
    """Test DBO with DP+EP using GSM8K evaluation."""
    lm_eval = pytest.importorskip("lm_eval")
    required_gpus = DP_SIZE

    if num_gpus_available < required_gpus:
        pytest.skip(f"Need at least {required_gpus} GPUs (DP={DP_SIZE})")

    # Server arguments for DBO + DP + EP
    server_args = [
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        str(MAX_NUM_SEQS),  # Use larger batch to trigger decode DBO
        "--trust-remote-code",
        # Note: Not using --enforce-eager to test DBO's alternate CUDA graph dispatching
        "--data-parallel-size",
        str(DP_SIZE),
        "--enable-expert-parallel",
        "--enable-dbo",
        # Fix threshold so we know we trigger DBO
        "--dbo-decode-token-threshold",
        "16",
        "--dbo-prefill-token-threshold",
        "32",
        "--all2all-backend",
        all2all_backend,
    ]

    with RemoteOpenAIServer(
        MODEL_NAME,
        server_args,
        max_wait_seconds=600,  # Allow time for model loading with DP+EP
    ) as remote_server:
        # Use host and port directly from RemoteOpenAIServer

        base_url = f"http://{remote_server.host}:{remote_server.port}/v1/completions"

        # Run GSM8K evaluation
        results = lm_eval.simple_evaluate(
            model="local-completions",
            model_args=(
                f"pretrained={MODEL_NAME},"
                f"base_url={base_url},"
                "num_concurrent=512,max_retries=3"
            ),
            tasks=["gsm8k"],
            num_fewshot=NUM_SHOTS,
            limit=NUM_QUESTIONS,
        )
        # Validate accuracy is reasonable
        gsm8k = results["results"]["gsm8k"]
        accuracy = gsm8k.get(
            "exact_match,strict-match",
            gsm8k.get("exact_match,flexible-extract"),
        )
        assert accuracy is not None, f"gsm8k exact_match missing: {gsm8k}"
        assert accuracy >= MIN_ACCURACY, (
            f"DBO+DP+EP accuracy too low ({all2all_backend}): "
            f"{accuracy:.3f} < {MIN_ACCURACY:.3f} "
        )


@pytest.mark.nixl_ep
@pytest.mark.asyncio
async def test_mrv2_imbalanced_decode_keeps_full_graph(monkeypatch):
    """Real decode rows survive graph staging with the eager model's logprobs."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    loads = (63, 97)

    async def run(enforce_eager):
        graph_stats: dict[int, list[CUDAGraphStat]] = {
            rank: [] for rank in range(DP_SIZE)
        }

        class GraphStats(StatLoggerBase):
            def __init__(self, vllm_config, engine_index=0):
                self.engine_index = engine_index

            def log_engine_initialized(self):
                pass

            def record(
                self,
                scheduler_stats,
                iteration_stats,
                mm_cache_stats=None,
                engine_idx=0,
            ):
                if scheduler_stats is not None and scheduler_stats.cudagraph_stats:
                    graph_stats[self.engine_index].append(
                        scheduler_stats.cudagraph_stats
                    )

        engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=os.getenv("DBO_TEST_MODEL", "deepseek-ai/DeepSeek-V2-Lite"),
                trust_remote_code=True,
                dtype="bfloat16",
                seed=0,
                skip_tokenizer_init=True,
                data_parallel_size=DP_SIZE,
                enable_expert_parallel=True,
                enable_dbo=True,
                all2all_backend="nixl_ep",
                moe_backend="batched_triton",
                attention_backend="FLASH_ATTN_MLA",
                enforce_eager=enforce_eager,
                max_model_len=256,
                max_num_seqs=128,
                max_num_batched_tokens=2048,
                kv_cache_memory_bytes=1 << 30,
                enable_prefix_caching=False,
                async_scheduling=False,
                cudagraph_metrics=True,
                compilation_config={
                    "mode": 0,
                    "cudagraph_mode": "FULL",
                    "cudagraph_capture_sizes": [128],
                },
            ),
            stat_loggers=[GraphStats],
        )
        try:
            await engine.pause_generation(mode="keep")
            params = SamplingParams(
                temperature=0,
                max_tokens=8,
                ignore_eos=True,
                logprobs=5,
                output_kind=RequestOutputKind.FINAL_ONLY,
            )
            collectors = [
                await engine.add_request(
                    f"{rank}-{i}",
                    {"prompt_token_ids": [1, 20 + i % 20, 37, 42] * (2 + i % 2)},
                    params,
                    data_parallel_rank=rank,
                )
                for rank, count in enumerate(loads)
                for i in range(count)
            ]
            await engine.resume_generation()
            outputs = await asyncio.wait_for(
                asyncio.gather(*(collector.get() for collector in collectors)), 180
            )
            completed = []
            for out in outputs:
                assert isinstance(out, RequestOutput)
                assert out.finished and len(out.outputs[0].token_ids) == 8
                completed.append(out.outputs[0])
            return [
                (
                    list(out.token_ids),
                    out.text,
                    out.logprobs,
                )
                for out in completed
            ], graph_stats
        finally:
            engine.shutdown()

    reference, _ = await run(enforce_eager=True)
    staged, graph_stats = await run(enforce_eager=False)
    for rank, count in enumerate(loads):
        assert any(
            stat.num_unpadded_tokens == count
            and stat.num_padded_tokens == 128
            and stat.runtime_mode == "FULL"
            for stat in graph_stats[rank]
        ), f"rank {rank} did not replay the imbalanced FULL graph"
    check_logprobs_close(
        outputs_0_lst=reference,
        outputs_1_lst=staged,
        name_0="eager DBO",
        name_1="staged FULL DBO",
    )
