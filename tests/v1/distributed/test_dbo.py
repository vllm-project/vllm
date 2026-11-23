# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test Dual Batch Overlap (DBO) with Data Parallelism + Expert Parallelism.

DBO is specifically designed for DP+EP scenarios to hide communication latency
by overlapping computation of two batches. This test validates that DBO works
correctly with the DeepSeek-V2-Lite model using GSM8K evaluation.
"""

import os
import random

import pytest
import torch
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from tests.utils import RemoteOpenAIServer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput
from vllm.platforms import current_platform
from vllm.sampling_params import RequestOutputKind
from vllm.utils.import_utils import has_deep_ep
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.engine.core_client import InprocClient
from vllm.v1.executor.uniproc_executor import ExecutorWithExternalLauncher

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
    _run_dbo_gsm8k(all2all_backend, num_gpus_available)


# Features combined with DBO. ngram verifies multi-token queries without a
# draft model; DCP shards the KV cache across each DP rank's TP group.
V2_FEATURES = {
    "spec_decode": (
        [
            "--speculative-config",
            (
                '{"method": "ngram", "num_speculative_tokens": 2, '
                '"prompt_lookup_max": 4, "prompt_lookup_min": 2}'
            ),
        ],
        1,
    ),
    "dcp": (["--tensor-parallel-size", "2", "--decode-context-parallel-size", "2"], 2),
}


@pytest.mark.skipif(not has_deep_ep(), reason="These tests require deep_ep to run")
@pytest.mark.parametrize("feature", list(V2_FEATURES))
def test_dbo_v2_features_gsm8k(feature: str, num_gpus_available):
    """Test DBO combined with spec decode or DCP."""
    extra_args, tp_size = V2_FEATURES[feature]
    _run_dbo_gsm8k(
        "deepep_low_latency",
        num_gpus_available,
        extra_args=extra_args,
        tp_size=tp_size,
    )


def _generate_with_hybrid_dbo(async_scheduling: bool | None) -> list[list[int]]:
    llm = LLM(
        model="ibm-granite/granite-4.0-h-tiny",
        skip_tokenizer_init=True,
        dtype="bfloat16",
        distributed_executor_backend="external_launcher",
        data_parallel_size=DP_SIZE,
        enable_expert_parallel=True,
        enable_dbo=True,
        all2all_backend="deepep_low_latency",
        dbo_prefill_token_threshold=512,
        dbo_decode_token_threshold=2,
        enable_prefix_caching=False,
        async_scheduling=async_scheduling,
        enforce_eager=True,
        max_model_len=4096,
        max_num_batched_tokens=2048,
        max_num_seqs=8,
        kv_cache_memory_bytes=3 * 1024**3,
        enable_extensible_kv_cache=False,
    )
    engine = llm.llm_engine
    client = engine.engine_core
    assert isinstance(client, InprocClient)
    core = client.engine_core
    assert isinstance(core.scheduler, AsyncScheduler) == (async_scheduling is None)
    assert isinstance(core.model_executor, ExecutorWithExternalLauncher)
    assert core.model_executor.driver_worker.model_runner.ubatch_runner is not None

    # The short request decodes while the long request finishes chunked prefill.
    generator = random.Random(2701 + int(os.environ["RANK"]))
    prompts = [
        TokensPrompt(
            prompt_token_ids=[generator.randrange(1024, 20000) for _ in range(n)]
        )
        for n in (256, 3072)
    ]
    params = SamplingParams(
        temperature=0,
        max_tokens=4,
        ignore_eos=True,
        detokenize=False,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    llm.enqueue(prompts, params, use_tqdm=False)
    assert not engine.step()
    previous = core.batch_queue[0][1] if core.batch_queue else None
    assert not engine.step()
    if async_scheduling is None:
        assert previous is not None and core.batch_queue
        current = core.batch_queue[0][1]
        assert current is not previous
        assert (
            previous.num_scheduled_tokens.keys() & current.num_scheduled_tokens.keys()
        )
    else:
        assert core.batch_queue is None

    outputs = llm.wait_for_completion(RequestOutput, use_tqdm=False)
    sequences = [list(output.outputs[0].token_ids) for output in outputs]
    assert len(sequences) == len(prompts)
    assert all(len(tokens) == params.max_tokens for tokens in sequences)
    client.shutdown()
    return sequences


@pytest.mark.skipif(
    not current_platform.is_cuda() or not has_deep_ep(),
    reason="Hybrid DBO requires CUDA and DeepEP",
)
def test_hybrid_dbo_async_matches_sync(num_gpus_available, monkeypatch):
    """Default async scheduling preserves mixed-batch generation under hybrid DBO."""
    if num_gpus_available < DP_SIZE:
        pytest.skip(f"Need {DP_SIZE} GPUs for hybrid DBO")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("NVSHMEM_REMOTE_TRANSPORT", "none")
    monkeypatch.setenv("NVSHMEM_HCA_LIST", "none")

    def run(async_scheduling: bool | None):
        config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=DP_SIZE,
            rdzv_backend="c10d",
            rdzv_endpoint="localhost:0",
            max_restarts=0,
        )
        return elastic_launch(config, _generate_with_hybrid_dbo)(async_scheduling)

    assert run(None) == run(False)


def _run_dbo_gsm8k(
    all2all_backend: str,
    num_gpus_available: int,
    extra_args: list[str] | None = None,
    tp_size: int = 1,
):
    lm_eval = pytest.importorskip("lm_eval")
    required_gpus = DP_SIZE * tp_size

    if num_gpus_available < required_gpus:
        pytest.skip(f"Need at least {required_gpus} GPUs (DP={DP_SIZE}, TP={tp_size})")

    # Server arguments for DBO + DP + EP
    server_args = [
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        str(MAX_NUM_SEQS),  # Use larger batch to trigger decode DBO
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
        *(extra_args or []),
    ]

    # Single-node test: keep NVSHMEM on NVLink and off host NICs to prevent
    # intermittent IB transport setup hangs (ibv_modify_qp) across NUMA sockets.
    env_dict = {
        "NVSHMEM_REMOTE_TRANSPORT": "none",
        "NVSHMEM_HCA_LIST": "none",
    }

    with RemoteOpenAIServer(
        MODEL_NAME,
        server_args,
        env_dict=env_dict,
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
