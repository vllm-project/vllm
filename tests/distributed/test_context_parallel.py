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

from vllm.config.model import RunnerOption
from vllm.logger import init_logger

from ..models.registry import HF_EXAMPLE_MODELS
from ..utils import compare_two_settings, create_new_process_for_each_test

logger = init_logger("test_context_parallel")

VLLM_MULTI_NODE = os.getenv("VLLM_MULTI_NODE", "0") == "1"


class ParallelSetup(NamedTuple):
    tp_size: int
    pp_size: int
    dcp_size: int
    dcp_kv_cache_interleave_size: int
    eager_mode: bool
    chunked_prefill: bool


class CPTestOptions(NamedTuple):
    multi_node_only: bool
    load_format: str | None = None
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
        dcp_base: int = 1,
        dcp_kv_cache_interleave_size: int = 1,
        multi_node_only: bool = False,
        runner: RunnerOption = "auto",
        load_format: str | None = None,
        attn_backend: str | None = None,
    ):
        parallel_setups = []
        for eager_mode_val in [False]:
            for pp_multiplier in [1]:
                for dcp_multiplier in [0.5, 1]:
                    for chunked_prefill_val in [True]:
                        parallel_setups.append(
                            ParallelSetup(
                                tp_size=tp_base,
                                pp_size=pp_multiplier * pp_base,
                                dcp_size=int(dcp_multiplier * tp_base),
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
                load_format=load_format,
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


def _compare_cp_with_tp(
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
        dcp_kv_cache_interleave_size,
        eager_mode,
        chunked_prefill,
    ) = parallel_setup

    multi_node_only, load_format, attn_backend = test_options

    model_info = HF_EXAMPLE_MODELS.find_hf_info(model_id)
    model_info.check_transformers_version(on_fail="skip")

    trust_remote_code = model_info.trust_remote_code
    tokenizer_mode = model_info.tokenizer_mode
    hf_overrides = model_info.hf_overrides

    if load_format == "dummy":
        # Avoid OOM
        text_overrides = {
            "num_hidden_layers": 4,
            "hidden_size": 512,
            "intermediate_size": 800,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
        }

        if is_multimodal:
            hf_overrides.update({"text_config": text_overrides})
        else:
            hf_overrides.update(text_overrides)
    else:
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

    common_args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "8",
    ]
    if chunked_prefill:
        common_args.append("--enable-chunked-prefill")
    if eager_mode:
        common_args.append("--enforce-eager")
    if runner != "auto":
        common_args.extend(["--runner", runner])
    if trust_remote_code:
        common_args.append("--trust-remote-code")
    if tokenizer_mode:
        common_args.extend(["--tokenizer-mode", tokenizer_mode])
    if load_format:
        common_args.extend(["--load-format", load_format])
    if hf_overrides:
        common_args.extend(["--hf-overrides", json.dumps(hf_overrides)])

    if not attn_backend:
        cp_env = tp_env = {}
    else:
        cp_env = tp_env = {
            "VLLM_ATTENTION_BACKEND": attn_backend,
        }

    cp_args = [
        *common_args,
        "--tensor-parallel-size",
        str(tp_size),
        "--pipeline-parallel-size",
        str(pp_size),
        "--decode-context-parallel-size",
        str(dcp_size),
        "--dcp-kv-cache-interleave-size",
        str(dcp_kv_cache_interleave_size),
        "--distributed-executor-backend",
        distributed_backend,
    ]

    tp_args = [
        *common_args,
        "--tensor-parallel-size",
        str(tp_size),
        "--pipeline-parallel-size",
        str(pp_size),
        "--distributed-executor-backend",
        distributed_backend,
    ]

    compare_two_settings(
        model_id,
        cp_args,
        tp_args,
        cp_env,
        tp_env,
        method=method,
        max_wait_seconds=720,
    )


CP_TEXT_GENERATION_MODELS = {
    "deepseek-ai/DeepSeek-V2-Lite-Chat": [
        CPTestSettings.detailed(),
        CPTestSettings.detailed(tp_base=2),
        CPTestSettings.detailed(tp_base=2, dcp_kv_cache_interleave_size=64),
    ],
    "bigcode/gpt_bigcode-santacoder": [
        CPTestSettings.detailed(),
        CPTestSettings.detailed(tp_base=2),
    ],
}

CP_TEST_MODELS = [
    # TODO support other models
    # [LANGUAGE GENERATION]
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "bigcode/gpt_bigcode-santacoder",
]


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
        model_id == "bigcode/gpt_bigcode-santacoder"
        and torch.cuda.get_device_capability() != (9, 0)
    ):
        pytest.skip(reason="GQA+DCP currently requires compute capability of 9.0")

    _compare_cp_with_tp(
        model_id,
        parallel_setup,
        distributed_backend,
        runner,
        test_options,
        num_gpus_available,
        method="generate",
        is_multimodal=False,
    )
