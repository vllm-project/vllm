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

from vllm.config.compilation import CompilationMode
from vllm.config.model import RunnerOption
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import is_torch_equal_or_newer

from ..models.registry import HF_EXAMPLE_MODELS
from ..utils import compare_two_settings, create_new_process_for_each_test

logger = init_logger("test_sequence_parallel")

VLLM_MULTI_NODE = os.getenv("VLLM_MULTI_NODE", "0") == "1"


class ParallelSetup(NamedTuple):
    tp_size: int
    pp_size: int
    enable_fusion: bool
    eager_mode: bool
    chunked_prefill: bool


class SPTestOptions(NamedTuple):
    multi_node_only: bool
    load_format: str | None = None


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
        multi_node_only: bool = False,
        runner: RunnerOption = "auto",
        load_format: str | None = None,
    ):
        parallel_setups = []
        for eager_mode_val in [False, True]:
            for pp_multiplier in [1, 2]:
                for chunked_prefill_val in [False, True]:
                    parallel_setups.append(
                        ParallelSetup(
                            tp_size=tp_base,
                            pp_size=pp_multiplier * pp_base,
                            enable_fusion=False,
                            eager_mode=eager_mode_val,
                            chunked_prefill=chunked_prefill_val,
                        )
                    )
        return SPTestSettings(
            parallel_setups=parallel_setups,
            distributed_backends=["mp", "ray"],
            runner=runner,
            test_options=SPTestOptions(
                multi_node_only=multi_node_only, load_format=load_format
            ),
        )

    @staticmethod
    def fast(
        *,
        tp_base: int = 2,
        pp_base: int = 1,
        runner: RunnerOption = "auto",
        multi_node_only: bool = False,
        load_format: str | None = None,
    ):
        parallel_setups = []
        for eager_mode_val in [False, True]:
            for pp_multiplier in [1, 2]:
                for chunked_prefill_val in [False, True]:
                    parallel_setups.append(
                        ParallelSetup(
                            tp_size=tp_base,
                            pp_size=pp_multiplier * pp_base,
                            enable_fusion=False,
                            eager_mode=eager_mode_val,
                            chunked_prefill=chunked_prefill_val,
                        )
                    )
        return SPTestSettings(
            parallel_setups=parallel_setups,
            distributed_backends=["mp", "ray"],
            runner=runner,
            test_options=SPTestOptions(
                multi_node_only=multi_node_only, load_format=load_format
            ),
        )

    @staticmethod
    def fp8_quant(
        *,
        tp_base: int = 2,
        pp_base: int = 1,
        runner: RunnerOption = "auto",
        multi_node_only: bool = False,
        load_format: str | None = None,
    ):
        parallel_setups = []
        for fusion_val in [False, True]:
            parallel_setups.append(
                ParallelSetup(
                    tp_size=tp_base,
                    pp_size=pp_base,
                    enable_fusion=fusion_val,
                    eager_mode=True,
                    chunked_prefill=False,
                )
            )
        return SPTestSettings(
            parallel_setups=parallel_setups,
            distributed_backends=["mp", "ray"],
            runner=runner,
            test_options=SPTestOptions(
                multi_node_only=multi_node_only, load_format=load_format
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


def _compare_sp(
    model_id: str,
    parallel_setup: ParallelSetup,
    distributed_backend: str,
    runner: RunnerOption,
    test_options: SPTestOptions,
    num_gpus_available: int,
    use_inductor_graph_partition: bool,
    enable_async_tp: bool,
    *,
    method: Literal["generate", "encode"],
    is_multimodal: bool,
):
    (
        tp_size,
        pp_size,
        enable_fusion,
        eager_mode,
        chunked_prefill,
    ) = parallel_setup

    multi_node_only, load_format = test_options

    model_info = HF_EXAMPLE_MODELS.find_hf_info(model_id)
    model_info.check_transformers_version(on_fail="skip")

    trust_remote_code = model_info.trust_remote_code
    tokenizer_mode = model_info.tokenizer_mode
    hf_overrides = model_info.hf_overrides
    require_embed_inputs = model_info.require_embed_inputs

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
        "float16",
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
    if require_embed_inputs:
        common_args.extend(
            [
                "--skip-tokenizer-init",
                "--enable-prompt-embeds",
                "--enable-mm-embeds",
            ]
        )

    compilation_config = {
        "mode": CompilationMode.VLLM_COMPILE,
        "compile_sizes": [4, 8],
        "pass_config": {
            "enable_sequence_parallelism": True,
            "enable_async_tp": enable_async_tp,
            "enable_fusion": enable_fusion,
            "enable_noop": True,
        },
        "use_inductor_graph_partition": use_inductor_graph_partition,
    }

    tp_sp_args = [
        *common_args,
        "--tensor-parallel-size",
        str(tp_size),
        "--pipeline-parallel-size",
        str(pp_size),
        "--distributed-executor-backend",
        distributed_backend,
        "--compilation_config",
        json.dumps(compilation_config),
    ]

    tp_args = [
        *common_args,
        "--tensor-parallel-size",
        str(tp_size),
        "--distributed-executor-backend",
        "mp",
    ]

    compare_two_settings(model_id, tp_sp_args, tp_args, method=method)


SP_TEXT_GENERATION_MODELS = {
    # [Decoder-only]
    "hmellor/tiny-random-LlamaForCausalLM": SPTestSettings.fast(),
    "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8": SPTestSettings.fp8_quant(),
}

SP_TEST_MODELS = [
    # TODO support other models
    # [LANGUAGE GENERATION]
    "hmellor/tiny-random-LlamaForCausalLM",
    "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8",
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
        for model_id, settings in SP_TEXT_GENERATION_MODELS.items()
        for params in settings.iter_params(model_id)
        if model_id in SP_TEST_MODELS
    ],
)
@pytest.mark.parametrize("use_inductor_graph_partition", [True, False])
@pytest.mark.parametrize("enable_async_tp", [False])  # TODO: enable async TP
@create_new_process_for_each_test()
def test_tp_sp_generation(
    model_id: str,
    parallel_setup: ParallelSetup,
    distributed_backend: str,
    runner: RunnerOption,
    test_options: SPTestOptions,
    num_gpus_available,
    use_inductor_graph_partition: bool,
    enable_async_tp: bool,
):
    if use_inductor_graph_partition and not is_torch_equal_or_newer("2.9.0.dev"):
        pytest.skip("inductor graph partition is only available in PyTorch 2.9+")

    # Skip FP8 SP-only test on sm89 (compute capability 8.9)
    if (
        "fp8" in model_id.lower()
        and current_platform.get_device_capability() < (9, 0)
        and (not enable_async_tp)
    ):
        pytest.skip("FP8 reduction support begins with sm90 capable devices.")

    _compare_sp(
        model_id,
        parallel_setup,
        distributed_backend,
        runner,
        test_options,
        num_gpus_available,
        use_inductor_graph_partition,
        enable_async_tp=enable_async_tp,
        method="generate",
        is_multimodal=False,
    )
