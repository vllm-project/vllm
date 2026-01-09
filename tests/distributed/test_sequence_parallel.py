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
from typing import NamedTuple

import pytest

from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k
from tests.utils import RemoteOpenAIServer, create_new_process_for_each_test
from vllm.config.compilation import CompilationMode
from vllm.config.model import RunnerOption
from vllm.logger import init_logger
from vllm.utils.torch_utils import is_torch_equal_or_newer

from ..models.registry import HF_EXAMPLE_MODELS

logger = init_logger("test_sequence_parallel")

VLLM_MULTI_NODE = os.getenv("VLLM_MULTI_NODE", "0") == "1"

# min accuracy for each model
# Baseline TP=1 (no SP): 72.66%, with SP: ~71%, threshold with 2% buffer
SP_TEST_MODELS = {
    # .buildkite/lm-eval-harness/configs/Meta-Llama-3-8B-Instruct.yaml
    "meta-llama/Meta-Llama-3-8B-Instruct": 0.70,
}

# GSM8K eval configuration
NUM_QUESTIONS = 256  # Fast eval for CI
NUM_SHOTS = 5  # Few-shot examples


class ParallelSetup(NamedTuple):
    tp_size: int
    pp_size: int
    fuse_norm_quant: bool
    fuse_act_quant: bool
    eager_mode: bool
    chunked_prefill: bool


class SPTestOptions(NamedTuple):
    multi_node_only: bool


@dataclass
class SPTestSettings:
    parallel_setups: list[ParallelSetup]
    distributed_backends: list[str]
    runner: RunnerOption
    test_options: SPTestOptions

    @staticmethod
    def detailed(
        *,
        tp_base: int = 2,
        pp_base: int = 1,
        pp_multipliers: list[int] | None = None,
        multi_node_only: bool = False,
        runner: RunnerOption = "auto",
    ):
        parallel_setups = []
        if pp_multipliers is None:
            pp_multipliers = [1, 2]
        for eager_mode_val in [False, True]:
            for pp_multiplier in pp_multipliers:
                for chunked_prefill_val in [False, True]:
                    parallel_setups.append(
                        ParallelSetup(
                            tp_size=tp_base,
                            pp_size=pp_multiplier * pp_base,
                            fuse_norm_quant=False,
                            fuse_act_quant=False,
                            eager_mode=eager_mode_val,
                            chunked_prefill=chunked_prefill_val,
                        )
                    )
        return SPTestSettings(
            parallel_setups=parallel_setups,
            distributed_backends=["mp", "ray"],
            runner=runner,
            test_options=SPTestOptions(multi_node_only=multi_node_only),
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


def _test_sp_gsm8k(
    model_id: str,
    parallel_setup: ParallelSetup,
    distributed_backend: str,
    runner: RunnerOption,
    test_options: SPTestOptions,
    num_gpus_available: int,
    use_inductor_graph_partition: bool,
    fuse_gemm_comms: bool,
):
    (
        tp_size,
        pp_size,
        fuse_norm_quant,
        fuse_act_quant,
        eager_mode,
        chunked_prefill,
    ) = parallel_setup

    (multi_node_only,) = test_options

    model_info = HF_EXAMPLE_MODELS.find_hf_info(model_id)
    model_info.check_transformers_version(on_fail="skip")

    trust_remote_code = model_info.trust_remote_code
    tokenizer_mode = model_info.tokenizer_mode
    hf_overrides = model_info.hf_overrides

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

    compilation_config = {
        "mode": CompilationMode.VLLM_COMPILE,
        "compile_sizes": [4, 8],
        "pass_config": {
            "enable_sp": True,
            "fuse_gemm_comms": fuse_gemm_comms,
            "fuse_norm_quant": fuse_norm_quant,
            "fuse_act_quant": fuse_act_quant,
            "eliminate_noops": True,
        },
        "use_inductor_graph_partition": use_inductor_graph_partition,
    }

    server_args.extend(
        [
            "--tensor-parallel-size",
            str(tp_size),
            "--pipeline-parallel-size",
            str(pp_size),
            "--distributed-executor-backend",
            distributed_backend,
            "--compilation_config",
            json.dumps(compilation_config),
        ]
    )

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
        min_accuracy = SP_TEST_MODELS[model_id]
        assert accuracy >= min_accuracy, (
            f"TP+SP accuracy too low: {accuracy:.3f} < {min_accuracy:.3f}"
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
        param
        for model_id in SP_TEST_MODELS
        for param in SPTestSettings.detailed().iter_params(model_id)
    ],
)
@pytest.mark.parametrize("use_inductor_graph_partition", [True, False])
@pytest.mark.parametrize("fuse_gemm_comms", [False])  # TODO: enable async TP
@create_new_process_for_each_test()
def test_tp_sp_generation(
    model_id: str,
    parallel_setup: ParallelSetup,
    distributed_backend: str,
    runner: RunnerOption,
    test_options: SPTestOptions,
    num_gpus_available,
    use_inductor_graph_partition: bool,
    fuse_gemm_comms: bool,
):
    if use_inductor_graph_partition and not is_torch_equal_or_newer("2.9.0.dev"):
        pytest.skip("inductor graph partition is only available in PyTorch 2.9+")

    _test_sp_gsm8k(
        model_id,
        parallel_setup,
        distributed_backend,
        runner,
        test_options,
        num_gpus_available,
        use_inductor_graph_partition,
        fuse_gemm_comms=fuse_gemm_comms,
    )
