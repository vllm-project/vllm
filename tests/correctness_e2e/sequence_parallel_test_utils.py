# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared infrastructure for e2e correctness tests of model-level sequence
parallelism (``--enable-sequence-parallel``) and its fused comm+GEMM kernels
(``--enable-async-tp``) against a plain TP baseline.

Each model's own ``test_sequence_parallel.py`` (e.g.
``tests/models/qwen3/test_sequence_parallel.py``) defines its model list and
calls ``run_tp_sp_generation_test`` from a parametrized test.
"""

from dataclasses import dataclass
from typing import Literal, NamedTuple

import pytest

from vllm.config.model import RunnerOption

from ..models.registry import HF_EXAMPLE_MODELS, _HfExamplesInfo
from ..utils import compare_all_settings


class ParallelSetup(NamedTuple):
    tp_size: int
    pp_size: int
    enable_async_tp: bool


class SPTestOptions(NamedTuple):
    load_format: str | None = None
    model_info: _HfExamplesInfo | None = None
    quantization: str | None = None


@dataclass
class SPTestSettings:
    parallel_setups: list[ParallelSetup]
    runner: RunnerOption
    test_options: SPTestOptions

    @staticmethod
    def default(
        *,
        tp_base: int = 2,
        pp_base: int = 1,
        runner: RunnerOption = "auto",
        load_format: str | None = None,
        model_info: _HfExamplesInfo | None = None,
        quantization: str | None = None,
    ):
        # Both the unfused (plain all_gather/reduce_scatter) and fused
        # (symmetric-memory AllGather+GEMM / GEMM+ReduceScatter) SP kernels,
        # each vs. a no-SP TP baseline. Always run in eager mode (see
        # ``build_sp_args``) since the fused kernels rely on symmetric
        # memory, whose CUDA graph behavior isn't the concern of this test.
        parallel_setups = [
            ParallelSetup(tp_size=tp_base, pp_size=pp_base, enable_async_tp=async_tp)
            for async_tp in [False, True]
        ]
        return SPTestSettings(
            parallel_setups=parallel_setups,
            runner=runner,
            test_options=SPTestOptions(
                load_format=load_format,
                model_info=model_info,
                quantization=quantization,
            ),
        )

    def iter_params(self, model_id: str):
        opts = self.test_options

        for parallel_setup in self.parallel_setups:
            yield (model_id, parallel_setup, self.runner, opts)


def build_sp_args(
    model_id: str,
    parallel_setup: ParallelSetup,
    runner: RunnerOption,
    test_options: SPTestOptions,
    num_gpus_available: int,
) -> tuple[list[str], list[str]] | None:
    tp_size, pp_size, enable_async_tp = parallel_setup

    if num_gpus_available < tp_size * pp_size:
        return None

    load_format = test_options.load_format

    model_info = test_options.model_info or HF_EXAMPLE_MODELS.find_hf_info(model_id)
    model_info.check_transformers_version(on_fail="skip")

    trust_remote_code = model_info.trust_remote_code
    tokenizer_mode = model_info.tokenizer_mode

    if load_format != "dummy":
        model_info.check_available_online(on_fail="skip")

    common_args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "8",
        # Hardcode eager mode: the fused kernels rely on symmetric memory,
        # and CUDA graph interaction with it isn't the concern of this
        # correctness test.
        "--enforce-eager",
    ]
    if runner != "auto":
        common_args.extend(["--runner", runner])
    if trust_remote_code:
        common_args.append("--trust-remote-code")
    if tokenizer_mode:
        common_args.extend(["--tokenizer-mode", tokenizer_mode])
    if load_format:
        common_args.extend(["--load-format", load_format])
    if test_options.quantization:
        common_args.extend(["--quantization", test_options.quantization])

    tp_args = [
        *common_args,
        "--tensor-parallel-size",
        str(tp_size),
        "--pipeline-parallel-size",
        str(pp_size),
        "--distributed-executor-backend",
        "mp",
    ]

    tp_sp_args = [
        *tp_args,
        "--enable-sequence-parallel",
        "--sequence-parallel-min-tokens",
        "0",
    ]
    if enable_async_tp:
        tp_sp_args.append("--enable-async-tp")

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
        )


def run_tp_sp_generation_test(
    model_id: str,
    settings: SPTestSettings,
    num_gpus_available: int,
) -> None:
    """Compare ``--enable-sequence-parallel``/``--enable-async-tp`` text
    generation against a plain TP baseline for every parallel setup in
    ``settings``."""
    comparisons = []
    for _, parallel_setup, runner, test_options in settings.iter_params(model_id):
        comparison = build_sp_args(
            model_id,
            parallel_setup,
            runner,
            test_options,
            num_gpus_available,
        )
        if comparison is not None:
            comparisons.append(comparison)

    _compare_sp_settings(model_id, comparisons, method="generate")
