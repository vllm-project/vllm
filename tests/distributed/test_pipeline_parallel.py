import os

import pytest

from ..utils import compare_two_settings

VLLM_MULTI_NODE = os.getenv("VLLM_MULTI_NODE", "0") == "1"


@pytest.mark.parametrize(
    "TP_SIZE, PP_SIZE, EAGER_MODE, CHUNKED_PREFILL, MODEL_NAME, DIST_BACKEND",
    [
        (2, 2, 0, 1, "meta-llama/Meta-Llama-3-8B", "ray"),
        (2, 2, 1, 0, "meta-llama/Meta-Llama-3-8B", "ray"),
        (1, 3, 0, 0, "meta-llama/Meta-Llama-3-8B", "ray"),
        (1, 4, 0, 1, "meta-llama/Meta-Llama-3-8B", "ray"),
        (1, 4, 1, 0, "meta-llama/Meta-Llama-3-8B", "ray"),
        (2, 2, 0, 1, "meta-llama/Meta-Llama-3-8B", "mp"),
        (2, 2, 1, 0, "meta-llama/Meta-Llama-3-8B", "mp"),
        (1, 3, 0, 0, "meta-llama/Meta-Llama-3-8B", "mp"),
        (1, 4, 0, 1, "meta-llama/Meta-Llama-3-8B", "mp"),
        (1, 4, 1, 0, "meta-llama/Meta-Llama-3-8B", "mp"),
    ])
def test_compare_tp(TP_SIZE, PP_SIZE, EAGER_MODE, CHUNKED_PREFILL, MODEL_NAME,
                    DIST_BACKEND):
    if VLLM_MULTI_NODE and DIST_BACKEND == "mp":
        pytest.skip("Skipping multi-node pipeline parallel test for "
                    "multiprocessing distributed backend")

    pp_args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "float16",
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
        "bfloat16",
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

    compare_two_settings(MODEL_NAME, pp_args, tp_args)


@pytest.mark.parametrize("PP_SIZE, MODEL_NAME", [
    (2, "JackFram/llama-160m"),
])
@pytest.mark.parametrize("ATTN_BACKEND", [
    "FLASH_ATTN",
    "FLASHINFER",
])
def test_pp_cudagraph(PP_SIZE, MODEL_NAME, ATTN_BACKEND):
    cudagraph_args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "float16",
        "--pipeline-parallel-size",
        str(PP_SIZE),
        "--distributed-executor-backend",
        "ray",
    ]
    os.environ["VLLM_ATTENTION_BACKEND"] = ATTN_BACKEND

    eager_args = cudagraph_args + ["--enforce-eager"]

    compare_two_settings(MODEL_NAME, eager_args, cudagraph_args)
