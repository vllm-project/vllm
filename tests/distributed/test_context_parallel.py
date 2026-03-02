# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
WARNING: This test runs in both single-node (4 GPUs) and multi-node
 (2 node with 2 GPUs each) modes. If the test only uses 2 GPUs, it is
 important to set the distributed backend to "mp" to avoid Ray scheduling
 all workers in a node other than the head node, which can cause the test
 to fail.
"""

import json
import os
from dataclasses import dataclass
from typing import Literal, NamedTuple

import pytest
import torch

from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k
from tests.utils import RemoteOpenAIServer, create_new_process_for_each_test
from vllm.config.model import RunnerOption
from vllm.logger import init_logger

from ..models.registry import HF_EXAMPLE_MODELS

logger = init_logger("test_context_parallel")

VLLM_MULTI_NODE = os.getenv("VLLM_MULTI_NODE", "0") == "1"

CP_TEST_MODELS = [
    # TODO support other models
    # [LANGUAGE GENERATION]
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "Qwen/Qwen2.5-1.5B-Instruct",
]

# GSM8K eval configuration
NUM_QUESTIONS = 256  # Fast eval for CI
NUM_SHOTS = 5  # Few-shot examples
# tp accuracy with 2% buffer
MIN_ACCURACY = {
    # .buildkite/lm-eval-harness/configs/DeepSeek-V2-Lite-Chat.yaml
    "deepseek-ai/DeepSeek-V2-Lite-Chat": 0.64,
    # .buildkite/lm-eval-harness/configs/Qwen2.5-1.5B-Instruct.yaml
    "Qwen/Qwen2.5-1.5B-Instruct": 0.52,
}


class ParallelSetup(NamedTuple):
    tp_size: int
    pp_size: int
    dcp_size: int
    pcp_size: int
    dcp_kv_cache_interleave_size: int
    eager_mode: bool
    chunked_prefill: bool


class CPTestOptions(NamedTuple):
    multi_node_only: bool
    attn_backend: str | None = None


@dataclass
class CPTestSettings:
    parallel_setups: list[ParallelSetup]
    distributed_backends: list[str]
    runner: RunnerOption
    test_options: CPTestOptions

    @staticmethod
    def detailed(
        *,
        tp_base: int = 4,
        pp_base: int = 1,
        dcp_multipliers: list[float] | None = None,
        pcp_base: int = 1,
        dcp_kv_cache_interleave_size: int = 1,
        multi_node_only: bool = False,
        runner: RunnerOption = "auto",
        attn_backend: str | None = None,
    ):
        parallel_setups = []
        if dcp_multipliers is None:
            dcp_multipliers = [
                0.5,
            ]
        for eager_mode_val in [False]:
            for pp_multiplier in [1]:
                for dcp_multiplier in dcp_multipliers:
                    for chunked_prefill_val in [True]:
                        parallel_setups.append(
                            ParallelSetup(
                                tp_size=tp_base,
                                pp_size=pp_multiplier * pp_base,
                                dcp_size=max(1, int(dcp_multiplier * tp_base)),
                                pcp_size=pcp_base,
                                dcp_kv_cache_interleave_size=dcp_kv_cache_interleave_size,
                                eager_mode=eager_mode_val,
                                chunked_prefill=chunked_prefill_val,
                            )
                        )
        return CPTestSettings(
            parallel_setups=parallel_setups,
            distributed_backends=["mp"],
            runner=runner,
            test_options=CPTestOptions(
                multi_node_only=multi_node_only,
                attn_backend=attn_backend,
            ),
        )

    def iter_params(self, model_id: str):
        opts = self.test_options

        for parallel_setup in self.parallel_setups:
            for backend in self.distributed_backends:
                yield (
                    model_id,
                    parallel_setup,
                    backend,
                    self.runner,
                    opts,
                )


CP_TEXT_GENERATION_MODELS = {
    "deepseek-ai/DeepSeek-V2-Lite-Chat": [
        CPTestSettings.detailed(dcp_multipliers=[1]),
        CPTestSettings.detailed(
            dcp_multipliers=[0.5],
            dcp_kv_cache_interleave_size=64,
            attn_backend="FLASHMLA",
        ),
        CPTestSettings.detailed(tp_base=1, pcp_base=4, dcp_kv_cache_interleave_size=64),
        CPTestSettings.detailed(tp_base=2, pcp_base=2, dcp_kv_cache_interleave_size=64),
    ],
    "Qwen/Qwen2.5-1.5B-Instruct": [
        CPTestSettings.detailed(
            dcp_kv_cache_interleave_size=16, attn_backend="FLASH_ATTN"
        ),
        CPTestSettings.detailed(
            dcp_kv_cache_interleave_size=16, attn_backend="FLASHINFER"
        ),
    ],
}


def _test_cp_gsm8k(
    model_id: str,
    parallel_setup: ParallelSetup,
    distributed_backend: str,
    runner: RunnerOption,
    test_options: CPTestOptions,
    num_gpus_available: int,
    *,
    method: Literal["generate"],
    is_multimodal: bool,
):
    (
        tp_size,
        pp_size,
        dcp_size,
        pcp_size,
        dcp_kv_cache_interleave_size,
        eager_mode,
        chunked_prefill,
    ) = parallel_setup

    multi_node_only, attn_backend = test_options

    model_info = HF_EXAMPLE_MODELS.find_hf_info(model_id)
    model_info.check_transformers_version(on_fail="skip")

    trust_remote_code = model_info.trust_remote_code
    tokenizer_mode = model_info.tokenizer_mode
    hf_overrides = model_info.hf_overrides

    model_info.check_available_online(on_fail="skip")

    if num_gpus_available < tp_size * pp_size:
        pytest.skip(f"Need at least {tp_size} x {pp_size} GPUs")
    if VLLM_MULTI_NODE and distributed_backend == "mp":
        pytest.skip(
            "Skipping multi-node pipeline parallel test for "
            "multiprocessing distributed backend"
        )
    if multi_node_only and not VLLM_MULTI_NODE:
        pytest.skip("Not in multi-node setting")

    server_args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "64",
    ]
    if chunked_prefill:
        server_args.append("--enable-chunked-prefill")
    if eager_mode:
        server_args.append("--enforce-eager")
    if runner != "auto":
        server_args.extend(["--runner", runner])
    if trust_remote_code:
        server_args.append("--trust-remote-code")
    if tokenizer_mode:
        server_args.extend(["--tokenizer-mode", tokenizer_mode])
    if hf_overrides:
        server_args.extend(["--hf-overrides", json.dumps(hf_overrides)])

    server_args.extend(
        [
            "--tensor-parallel-size",
            str(tp_size),
            "--pipeline-parallel-size",
            str(pp_size),
            "--decode-context-parallel-size",
            str(dcp_size),
            "--prefill-context-parallel-size",
            str(pcp_size),
            "--dcp-kv-cache-interleave-size",
            str(dcp_kv_cache_interleave_size),
            "--distributed-executor-backend",
            distributed_backend,
        ]
    )

    if attn_backend:
        server_args.append(f"--attention-backend={attn_backend}")

    with RemoteOpenAIServer(
        model_id,
        server_args,
        max_wait_seconds=720,
    ) as remote_server:
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
        min_accuracy = MIN_ACCURACY[model_id]
        assert accuracy >= min_accuracy, (
            f"TP+DCP accuracy too low: {accuracy:.3f} < {min_accuracy:.3f}"
        )


@pytest.mark.parametrize(
    (
        "model_id",
        "parallel_setup",
        "distributed_backend",
        "runner",
        "test_options",
    ),
    [
        params
        for model_id, settings in CP_TEXT_GENERATION_MODELS.items()
        for setting in settings
        for params in setting.iter_params(model_id)
        if model_id in CP_TEST_MODELS
    ],
)
@create_new_process_for_each_test()
def test_cp_generation(
    model_id: str,
    parallel_setup: ParallelSetup,
    distributed_backend: str,
    runner: RunnerOption,
    test_options: CPTestOptions,
    num_gpus_available,
):
    if (
        model_id == "deepseek-ai/DeepSeek-V2-Lite-Chat"
        and torch.cuda.get_device_capability() < (9, 0)
    ):
        pytest.skip(reason="MLA+DCP requires compute capability of 9.0 or higher")
    if (
        model_id == "Qwen/Qwen2.5-1.5B-Instruct"
        and torch.cuda.get_device_capability() != (9, 0)
    ):
        pytest.skip(reason="GQA+DCP currently requires compute capability of 9.0")

    _test_cp_gsm8k(
        model_id,
        parallel_setup,
        distributed_backend,
        runner,
        test_options,
        num_gpus_available,
        method="generate",
        is_multimodal=False,
    )
