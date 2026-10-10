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
        {"ts": 0, "input_length": 12, "output_length": 3, "hash_ids": [7, 11]},
        {"ts": 1, "input_length": 16, "output_length": 3, "hash_ids": [7, 19]},
        {"ts": 2, "input_length": 16, "output_length": 3, "hash_ids": [7, 11]},
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
        from vllm.benchmarks.datasets.datasets import TimedTrace

        tokenizer = TokenizersBackend(
            tokenizer_object=Tokenizer(models.WordLevel(
                {"[UNK]": 0, **{str(i): i for i in range(1, 100)}},
                unk_token="[UNK]",
            )),
            unk_token="[UNK]",
        )
        dataset = TimedTrace(
            dataset_path=sys.argv[1], random_seed=42,
            timed_trace_chunk_hash_size=8,
            timed_trace_label_timestamp="ts",
            timed_trace_label_input_length="input_length",
            timed_trace_label_output_length="output_length",
            timed_trace_label_hash_ids="hash_ids",
        )
        samples = dataset.sample(tokenizer, num_requests=3)
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
    assert [len(prompt) for prompt in prompts] == [12, 16, 16]
    assert prompts[0] == prompts[2][:12]
    assert prompts[0][:8] == prompts[1][:8]
    assert prompts[1][8:] != prompts[2][8:]
