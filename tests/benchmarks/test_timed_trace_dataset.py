# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import subprocess
import sys
import textwrap

import pytest


@pytest.fixture(scope="module")
def sampled_trace_prompts(tmp_path_factory):
    trace = tmp_path_factory.mktemp("timed-trace") / "trace.jsonl"
    entries = [
        {"ts": 0, "input_length": 24, "output_length": 3, "hash_ids": [7, 11]},
        {"ts": 1, "input_length": 32, "output_length": 3, "hash_ids": [7, 19]},
        {"ts": 2, "input_length": 32, "output_length": 3, "hash_ids": [7, 11]},
        {"ts": 3, "input_length": 16, "output_length": 3, "hash_ids": [69235]},
        {"ts": 4, "input_length": 16, "output_length": 3, "hash_ids": [95303]},
    ]
    trace.write_text(
        "\n".join(json.dumps(entry) for entry in entries), encoding="utf-8"
    )
    script = textwrap.dedent(
        """
        import json
        import sys
        from tokenizers import Tokenizer, models
        from transformers import TokenizersBackend
        from vllm.benchmarks.datasets import add_dataset_parser, get_samples
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        tokenizer = TokenizersBackend(
            tokenizer_object=Tokenizer(models.WordLevel(
                {"[UNK]": 0, **{str(i): i for i in range(1, 128)}},
                unk_token="[UNK]",
            )),
            unk_token="[UNK]",
        )
        parser = FlexibleArgumentParser()
        add_dataset_parser(parser)
        args = parser.parse_args([
            "--dataset-name", "timed_trace", "--dataset-path", sys.argv[1],
            "--num-prompts", "5", "--timed-trace-label-timestamp", "ts",
            "--seed", "42",
        ])
        samples = get_samples(args, tokenizer)
        print(json.dumps([sample.prompt for sample in samples]))
        """
    )
    results = []
    for hash_seed in ("1", "2"):
        output = subprocess.check_output(
            [sys.executable, "-c", script, str(trace)],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            text=True,
            timeout=180,
        )
        results.append(json.loads(output.splitlines()[-1]))

    return results


def test_timed_trace_prompts_are_independent_of_python_hash_seed(sampled_trace_prompts):
    assert sampled_trace_prompts[0] == sampled_trace_prompts[1]


def test_timed_trace_partial_chunk_shares_full_prefix(sampled_trace_prompts):
    prompts = sampled_trace_prompts[0]
    assert [len(prompt) for prompt in prompts] == [24, 32, 32, 16, 16]
    assert prompts[0] == prompts[2][:24]
    assert prompts[0][:16] == prompts[1][:16]
    assert prompts[1][16:] != prompts[2][16:]


def test_timed_trace_distinct_hashes_do_not_share_truncated_seed(sampled_trace_prompts):
    prompts = sampled_trace_prompts[0]
    assert prompts[3] != prompts[4]
