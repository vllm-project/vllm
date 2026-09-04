# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from argparse import Namespace


def test_compute_result_filename_for_dataset_stats_plot(tmp_path):
    from vllm.benchmarks.serve import compute_result_filename

    args = Namespace(
        append_result=False,
        backend="openai",
        max_concurrency=None,
        plot_dataset_stats=True,
        plot_timeline=False,
        ramp_up_strategy=None,
        request_rate=float("inf"),
        result_dir=str(tmp_path),
        result_filename=None,
        save_result=False,
    )

    file_name = compute_result_filename(
        args=args,
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        label=None,
        current_dt="20260101-010203",
    )

    assert file_name == str(
        tmp_path / "openai-infqps-Llama-3.2-1B-Instruct-20260101-010203.json"
    )
