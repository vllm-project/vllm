# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import subprocess
from unittest.mock import Mock

import pytest

from vllm.benchmarks import latency
from vllm.utils.argparse_utils import FlexibleArgumentParser

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("fails", [False, True])
def test_proton_profile_excludes_warmup_and_stops_on_error(
    tmp_path, monkeypatch, fails
):
    from vllm import LLM

    parser = FlexibleArgumentParser()
    latency.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--profile",
            "--num-iters-warmup",
            "2",
            "--profiler-config.profiler",
            "proton",
            "--profiler-config.proton_profiler_dir",
            str(tmp_path),
            "--profiler-config.proton_graph_attribution",
            "true",
            "--profiler-config.proton_mode",
            "periodic_flushing",
            "--profiler-config.proton_flush_interval",
            "2",
        ]
    )
    llm = Mock()
    llm.llm_engine.model_config.max_model_len = 1024
    factory = Mock(return_value=llm)
    monkeypatch.setattr(LLM, "from_engine_args", factory)
    llm.generate.side_effect = [
        None,
        None,
        RuntimeError("generation failed") if fails else None,
    ]

    if fails:
        with pytest.raises(RuntimeError, match="generation failed"):
            latency.main(args)
    else:
        latency.main(args)

    config = factory.call_args.args[0].profiler_config
    assert config.profiler == "proton"
    assert config.proton_graph_attribution
    assert config.proton_mode == "periodic_flushing"
    assert config.proton_flush_interval == 2
    assert config.proton_profiler_dir == str(tmp_path)
    assert [c[0] for c in llm.mock_calls] == [
        "generate",
        "generate",
        "start_profile",
        "generate",
        "stop_profile",
    ]


@pytest.mark.benchmark
def test_bench_latency():
    command = [
        "vllm",
        "bench",
        "latency",
        "--model",
        MODEL_NAME,
        "--input-len",
        "32",
        "--output-len",
        "1",
        "--enforce-eager",
        "--load-format",
        "dummy",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"
