# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test Dual Batch Overlap (DBO) with Data Parallelism + Expert Parallelism.

DBO is specifically designed for DP+EP scenarios to hide communication latency
by overlapping computation of two batches. This test validates that DBO works
correctly with the DeepSeek-V2-Lite model using GSM8K evaluation.
"""

import pytest
import torch

from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k
from tests.utils import RemoteOpenAIServer
from vllm.utils.import_utils import has_deep_ep

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
MIN_ACCURACY = 0.62  # Expected 0.64 with 2% buffer (based on vLLM test data)

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
    """
    Test DBO with DP+EP using GSM8K evaluation.
    """
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
        "256",
        "--all2all-backend",
        all2all_backend,
    ]

    with RemoteOpenAIServer(
        MODEL_NAME,
        server_args,
        max_wait_seconds=600,  # Allow time for model loading with DP+EP
    ) as remote_server:
        # Use host and port directly from RemoteOpenAIServer
        host = f"http://{remote_server.host}"
        port = remote_server.port

        # Run GSM8K evaluation
        results = evaluate_gsm8k(
            num_questions=NUM_QUESTIONS,
            num_shots=NUM_SHOTS,
            host=host,
            port=port,
        )

        # Validate accuracy is reasonable
        accuracy = results["accuracy"]
        assert accuracy >= MIN_ACCURACY, (
            f"DBO+DP+EP accuracy too low ({all2all_backend}): "
            f"{accuracy:.3f} < {MIN_ACCURACY:.3f} "
        )
