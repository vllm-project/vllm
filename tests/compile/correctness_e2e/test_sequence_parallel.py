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

from ...models.registry import HF_EXAMPLE_MODELS, _HfExamplesInfo
from ...utils import compare_all_settings, create_new_process_for_each_test

logger = init_logger("test_sequence_parallel")

VLLM_MULTI_NODE = os.getenv("VLLM_MULTI_NODE", "0") == "1"
NVFP4_MODEL_ID = "nvidia/Llama-3.1-8B-Instruct-NVFP4"
NVFP4_MODEL_INFO = _HfExamplesInfo(NVFP4_MODEL_ID)


class ParallelSetup(NamedTuple):
    tp_size: int
    pp_size: int
    fuse_norm_quant: bool
    fuse_act_quant: bool
    eager_mode: bool


class SPTestOptions(NamedTuple):
    multi_node_only: bool
    load_format: str | None = None
    model_info: _HfExamplesInfo | None = None


@dataclass
class SPTestSettings:
    parallel_setups: list[ParallelSetup]
    distributed_backends: list[str]
    runner: RunnerOption
    test_options: SPTestOptions

    @staticmethod
    def fast(
        *,
        tp_base: int = 2,
        pp_base: int = 1,
        runner: RunnerOption = "auto",
        multi_node_only: bool = False,
        load_format: str | None = None,
    ):
        parallel_setups = [
            ParallelSetup(
                tp_size=tp_base,
                pp_size=pp_multiplier * pp_base,
                fuse_norm_quant=False,
                fuse_act_quant=False,
                eager_mode=False,
            )
            for pp_multiplier in [1, 2]
        ]
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
                    fuse_norm_quant=fusion_val,
                    fuse_act_quant=fusion_val,
                    eager_mode=True,
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


def _build_sp_args(
    model_id: str,
    parallel_setup: ParallelSetup,
    distributed_backend: str,
    runner: RunnerOption,
    test_options: SPTestOptions,
    num_gpus_available: int,
    use_inductor_graph_partition: bool,
    fuse_gemm_comms: bool,
    enable_prompt_embeds: bool,
    *,
    is_multimodal: bool,
    custom_ops: list[str] | None = None,
) -> tuple[list[str], list[str]] | None:
    (
        tp_size,
        pp_size,
        fuse_norm_quant,
        fuse_act_quant,
        eager_mode,
    ) = parallel_setup

    multi_node_only = test_options.multi_node_only
    load_format = test_options.load_format

    model_info = test_options.model_info or HF_EXAMPLE_MODELS.find_hf_info(model_id)
    model_info.check_transformers_version(on_fail="skip")

    trust_remote_code = model_info.trust_remote_code
    tokenizer_mode = model_info.tokenizer_mode
    hf_overrides = dict(model_info.hf_overrides)
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
        return None
    if VLLM_MULTI_NODE and distributed_backend == "mp":
        return None
    if multi_node_only and not VLLM_MULTI_NODE:
        return None

    common_args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "8",
    ]
    if eager_mode:
        common_args.append("-cc.cudagraph_mode=none")
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
    elif enable_prompt_embeds:
        common_args.append("--enable-prompt-embeds")

    compilation_config = {
        "mode": CompilationMode.VLLM_COMPILE,
        "compile_sizes": [4, 8],
        "pass_config": {
            "enable_sp": True,
            "fuse_gemm_comms": fuse_gemm_comms,
            "fuse_norm_quant": fuse_norm_quant,
            "fuse_act_quant": fuse_act_quant,
            "fuse_allreduce_rms": False,
            "eliminate_noops": True,
            "sp_min_token_num": 0,
        },
        "use_inductor_graph_partition": use_inductor_graph_partition,
    }
    if custom_ops is not None:
        compilation_config["custom_ops"] = custom_ops
    if not use_inductor_graph_partition:
        compilation_config["splitting_ops"] = []

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

    return tp_args, tp_sp_args


def _compare_sp_settings(
    model_id: str,
    settings: list[tuple[list[str], list[str]]],
    *,
    method: Literal["generate", "encode"],
) -> None:
    if not settings:
        pytest.skip("No supported sequence-parallel configurations")

    settings_by_tp_args: dict[tuple[str, ...], list[list[str]]] = {}
    for tp_args, tp_sp_args in settings:
        settings_by_tp_args.setdefault(tuple(tp_args), []).append(tp_sp_args)

    for tp_args_key, grouped_tp_sp_args in settings_by_tp_args.items():
        all_args = [list(tp_args_key), *grouped_tp_sp_args]
        compare_all_settings(
            model_id,
            all_args,
            [None] * len(all_args),
            method=method,
            force_v1_runner=True,
        )


SP_TEXT_GENERATION_MODELS = {
    # [Decoder-only]
    "hmellor/tiny-random-LlamaForCausalLM": SPTestSettings.fast(),
    "RedHatAI/Llama-3.2-1B-Instruct-FP8": SPTestSettings.fp8_quant(),
}


@pytest.mark.parametrize(
    ("model_id", "settings"),
    [
        pytest.param(model_id, settings, id=model_id)
        for model_id, settings in SP_TEXT_GENERATION_MODELS.items()
    ],
)
@create_new_process_for_each_test()
def test_tp_sp_generation(
    model_id: str,
    settings: SPTestSettings,
    num_gpus_available,
):
    # Skip FP8 SP-only test on sm89 (compute capability 8.9).
    # An unknown capability (None) is left to fail in the test body rather than
    # being silently skipped here.
    capability = current_platform.get_device_capability()
    if "fp8" in model_id.lower() and capability is not None and capability < (9, 0):
        pytest.skip("FP8 reduction support begins with sm90 capable devices.")

    graph_partition_options = [False]
    if is_torch_equal_or_newer("2.9.0.dev"):
        graph_partition_options.append(True)

    comparisons = []
    for _, parallel_setup, backend, runner, test_options in settings.iter_params(
        model_id
    ):
        for use_inductor_graph_partition in graph_partition_options:
            comparison = _build_sp_args(
                model_id,
                parallel_setup,
                backend,
                runner,
                test_options,
                num_gpus_available,
                use_inductor_graph_partition,
                fuse_gemm_comms=False,  # TODO: enable async TP
                enable_prompt_embeds=False,
                is_multimodal=False,
            )
            if comparison is not None:
                comparisons.append(comparison)

    _compare_sp_settings(model_id, comparisons, method="generate")


# Focused regression test for the SP + prompt_embeds graph-rewrite path.
# Covers pp_size=1 (SP only) and pp_size=2 (SP + PP); kept small on purpose so
# we don't double the matrix of `test_tp_sp_generation` above.
SP_PROMPT_EMBEDS_PARALLEL_SETUPS = [
    ParallelSetup(
        tp_size=2,
        pp_size=pp_size,
        fuse_norm_quant=False,
        fuse_act_quant=False,
        eager_mode=False,
    )
    for pp_size in [1, 2]
]


@create_new_process_for_each_test()
def test_tp_sp_generation_prompt_embeds(
    num_gpus_available,
):
    model_id = "hmellor/tiny-random-LlamaForCausalLM"
    graph_partition_options = [False]
    if is_torch_equal_or_newer("2.9.0.dev"):
        graph_partition_options.append(True)

    comparisons = []
    for parallel_setup in SP_PROMPT_EMBEDS_PARALLEL_SETUPS:
        for use_inductor_graph_partition in graph_partition_options:
            comparison = _build_sp_args(
                model_id,
                parallel_setup,
                distributed_backend="mp",
                runner="auto",
                test_options=SPTestOptions(multi_node_only=False, load_format=None),
                num_gpus_available=num_gpus_available,
                use_inductor_graph_partition=use_inductor_graph_partition,
                fuse_gemm_comms=False,
                enable_prompt_embeds=True,
                is_multimodal=False,
            )
            if comparison is not None:
                comparisons.append(comparison)

    _compare_sp_settings(model_id, comparisons, method="generate")


@create_new_process_for_each_test()
def test_tp_sp_pp_generation_native_rms_norm(num_gpus_available):
    model_id = "hmellor/tiny-random-LlamaForCausalLM"
    comparison = _build_sp_args(
        model_id,
        ParallelSetup(
            tp_size=2,
            pp_size=2,
            fuse_norm_quant=False,
            fuse_act_quant=False,
            eager_mode=False,
        ),
        distributed_backend="mp",
        runner="auto",
        test_options=SPTestOptions(multi_node_only=False, load_format=None),
        num_gpus_available=num_gpus_available,
        use_inductor_graph_partition=False,
        fuse_gemm_comms=False,
        enable_prompt_embeds=False,
        is_multimodal=False,
        custom_ops=["-rms_norm"],
    )

    _compare_sp_settings(
        model_id,
        [] if comparison is None else [comparison],
        method="generate",
    )


@create_new_process_for_each_test()
def test_tp_sp_nvfp4_generation(num_gpus_available: int):
    if (
        not current_platform.is_cuda()
        or not current_platform.is_device_capability_family(100)
    ):
        pytest.skip("NVFP4 requires Blackwell")

    comparison = _build_sp_args(
        NVFP4_MODEL_ID,
        ParallelSetup(
            tp_size=2,
            pp_size=1,
            fuse_norm_quant=True,
            fuse_act_quant=True,
            eager_mode=True,
        ),
        "mp",
        "auto",
        SPTestOptions(
            multi_node_only=False,
            load_format="dummy",
            model_info=NVFP4_MODEL_INFO,
        ),
        num_gpus_available,
        use_inductor_graph_partition=False,
        fuse_gemm_comms=False,
        enable_prompt_embeds=False,
        is_multimodal=False,
    )
    _compare_sp_settings(
        NVFP4_MODEL_ID,
        [] if comparison is None else [comparison],
        method="generate",
    )
