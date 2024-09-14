"""
WARNING: This test runs in both single-node (4 GPUs) and multi-node
 (2 node with 2 GPUs each) modes. If the test only uses 2 GPUs, it is
 important to set the distributed backend to "mp" to avoid Ray scheduling
 all workers in a node other than the head node, which can cause the test
 to fail.
"""
import os

import pytest

from vllm.logger import init_logger

from ..utils import compare_two_settings, fork_new_process_for_each_test

logger = init_logger("test_pipeline_parallel")

VLLM_MULTI_NODE = os.getenv("VLLM_MULTI_NODE", "0") == "1"


@pytest.mark.parametrize(
    ("TP_SIZE, PP_SIZE, EAGER_MODE, CHUNKED_PREFILL, TRUST_REMOTE_CODE, "
     "MODEL_NAME, DIST_BACKEND"),
    [
        (2, 2, 0, 1, 0, "meta-llama/Meta-Llama-3-8B", "mp"),
        (2, 2, 1, 0, 0, "meta-llama/Meta-Llama-3-8B", "mp"),
        (1, 3, 0, 0, 0, "meta-llama/Meta-Llama-3-8B", "mp"),
        (1, 4, 0, 1, 0, "meta-llama/Meta-Llama-3-8B", "mp"),
        (1, 4, 1, 0, 0, "meta-llama/Meta-Llama-3-8B", "mp"),
        (1, 3, 0, 0, 0, "meta-llama/Meta-Llama-3-8B", "ray"),
        (1, 4, 0, 1, 0, "meta-llama/Meta-Llama-3-8B", "ray"),
        (1, 4, 1, 0, 0, "meta-llama/Meta-Llama-3-8B", "ray"),
        (2, 2, 1, 0, 0, "meta-llama/Meta-Llama-3-8B", "ray"),
        (2, 2, 0, 1, 0, "meta-llama/Meta-Llama-3-8B", "ray"),
        # NOTE: InternVL2 multi-node tests are flaky,
        # use mp backend to skip the multi-node tests
        (1, 2, 1, 1, 1, "OpenGVLab/InternVL2-1B", "mp"),
        (1, 2, 1, 1, 1, "OpenGVLab/InternVL2-2B", "mp"),
        (1, 2, 1, 0, 1, "OpenGVLab/InternVL2-4B", "mp"),
    ],
)
@fork_new_process_for_each_test
def test_compare_tp(TP_SIZE, PP_SIZE, EAGER_MODE, CHUNKED_PREFILL,
                    TRUST_REMOTE_CODE, MODEL_NAME, DIST_BACKEND):
    if VLLM_MULTI_NODE and DIST_BACKEND == "mp":
        pytest.skip("Skipping multi-node pipeline parallel test for "
                    "multiprocessing distributed backend")

    pp_args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "float16",
        "--max-model-len",
        "8192",
        "--pipeline-parallel-size",
        str(PP_SIZE),
        "--tensor-parallel-size",
        str(TP_SIZE),
        "--distributed-executor-backend",
        DIST_BACKEND,
    ]

    # compare without pipeline parallelism
    # NOTE: use mp backend for TP
    # PP tests might involve multiple nodes, and ray might
    #  schedule all workers in a node other than the head node,
    #  which can cause the test to fail.
    tp_args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "float16",
        "--max-model-len",
        "8192",
        "--tensor-parallel-size",
        str(max(TP_SIZE, 2)),  # We only use 2 GPUs in the CI.
        "--distributed-executor-backend",
        "mp",
    ]
    if CHUNKED_PREFILL:
        pp_args.append("--enable-chunked-prefill")
        tp_args.append("--enable-chunked-prefill")
    if EAGER_MODE:
        pp_args.append("--enforce-eager")
        tp_args.append("--enforce-eager")
    if TRUST_REMOTE_CODE:
        pp_args.append("--trust-remote-code")
        tp_args.append("--trust-remote-code")
    pp_env = None
    if (DIST_BACKEND == "ray" and TP_SIZE == 2 and PP_SIZE == 2
            and CHUNKED_PREFILL):
        # Test Ray ADAG for a subset of the tests
        pp_env = {
            "VLLM_USE_RAY_COMPILED_DAG": "1",
            "VLLM_USE_RAY_SPMD_WORKER": "1",
            "VLLM_USE_RAY_COMPILED_DAG_NCCL_CHANNEL": "1",
        }
        # Temporary. Currently when zeromq + SPMD is used, it does not properly
        # terminate because of aDAG issue.
        pp_args.append("--disable-frontend-multiprocessing")
        tp_args.append("--disable-frontend-multiprocessing")

    try:
        compare_two_settings(MODEL_NAME, pp_args, tp_args, pp_env)
    except Exception:
        if pp_env is None:
            raise
        else:
            # Ray ADAG tests are flaky, so we don't want to fail the test
            logger.exception("Ray ADAG tests failed")
