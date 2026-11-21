# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import runpy
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2] / "benchmarks/kernels/benchmark_fused_topk.py"
)


def test_kernel_benchmark_defaults_are_preserved():
    parse_args = runpy.run_path(str(SCRIPT))["parse_args"]
    args = parse_args([])
    assert args.mode == "kernel"
    assert args.scoring_func == "softmax"
    assert args.dtype == "bfloat16"
    assert args.save_path == "./configs/fused_topk/"


@pytest.mark.parametrize(
    "options, message",
    [
        (["--num-tokens", "0"], "positive"),
        (["--num-layers", "0"], "positive"),
        (["--num-experts", "0"], "positive"),
        (["--hidden-size", "-1"], "positive"),
        (["--repeats", "0"], "positive"),
        (["--eager-iters", "0"], "positive"),
        (["--replays", "0"], "positive"),
        (["--warmup-ms", "-1"], "nonnegative"),
        (["--num-experts", "4", "--topk", "8"], "--topk"),
    ],
)
def test_invalid_chain_arguments_fail_before_cuda(options, message, capsys):
    parse_args = runpy.run_path(str(SCRIPT))["parse_args"]
    with pytest.raises(SystemExit) as exc:
        parse_args(["--mode", "chain", *options])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_chain_accepts_separate_gemm_and_gating_dtypes():
    parse_args = runpy.run_path(str(SCRIPT))["parse_args"]
    args = parse_args(
        [
            "--mode",
            "chain",
            "--num-tokens",
            "1",
            "16",
            "48",
            "--num-experts",
            "256",
            "--topk",
            "8",
            "--hidden-size",
            "4096",
            "--num-layers",
            "48",
            "--dtype",
            "float32",
            "--gemm-dtype",
            "bfloat16",
            "--warmup-ms",
            "0",
        ]
    )
    assert args.num_tokens == [1, 16, 48]
    assert (args.num_experts, args.topk, args.hidden_size, args.num_layers) == (
        256,
        8,
        4096,
        48,
    )
    assert args.dtype == "float32"
    assert args.gemm_dtype == "bfloat16"
    assert args.warmup_ms == 0
