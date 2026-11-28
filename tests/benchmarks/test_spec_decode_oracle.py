# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import numpy as np
import pytest

from benchmarks.benchmark_spec_decode_oracle import OracleProposer, spec_decode_metrics
from vllm import SamplingParams
from vllm.config import ModelConfig, SpeculativeConfig, VllmConfig
from vllm.inputs import TokensPrompt

MODEL = "facebook/opt-125m"
SPEC_CONFIG = {
    "method": "custom_class",
    "model": "benchmarks.benchmark_spec_decode_oracle.OracleProposer",
    "num_speculative_tokens": 3,
}


@pytest.fixture
def reference_file(tmp_path, monkeypatch):
    path = tmp_path / "ref.json"
    monkeypatch.setenv("ORACLE_REF_FILE", str(path))
    monkeypatch.setenv("ORACLE_STRICT", "0")
    monkeypatch.setenv("ORACLE_RESYNC_NGRAM", "3")
    monkeypatch.setenv("ORACLE_STATS_DIR", str(tmp_path))
    return path


@pytest.mark.parametrize(
    "output_len, expected",
    [(5, [3, 4, 5]), (11, [6, 7, 8]), (17, [9, 20, 21]), (8, [3, 4, 5])],
)
def test_oracle_uses_nearest_match(reference_file, output_len, expected):
    """Repeated suffixes use the nearest match; ties favor the earlier one."""
    prompt = [2, 100]
    ref = [1, 2, 10, 11, 12, 3, 4, 5, 10, 11, 12, 6, 7, 8, 10, 11, 12, 9, 20, 21]
    reference_file.write_text(
        json.dumps([{"prompt_token_ids": prompt, "output_token_ids": ref}])
    )
    proposer = OracleProposer(
        VllmConfig(
            model_config=ModelConfig(model=MODEL, max_model_len=128),
            speculative_config=SpeculativeConfig(**SPEC_CONFIG),
        )
    )
    output = [1] + [99] * (output_len - 4) + [10, 11, 12]
    tokens = np.array([prompt + output], dtype=np.int32)
    assert proposer.propose([[12]], np.array([tokens.shape[1]]), tokens) == [expected]


def test_oracle_resumes_drafts_after_divergence(reference_file, vllm_runner):
    """A later copy at the reference end must not prevent useful drafts."""
    prompt = [2, 100]
    reference_file.write_text(
        json.dumps([{"prompt_token_ids": prompt, "output_token_ids": [43] + [42] * 63}])
    )
    # One allowed token makes the output exact, independent of batch shape.
    params = SamplingParams(
        temperature=0, max_tokens=32, ignore_eos=True, allowed_token_ids=[42]
    )
    with vllm_runner(
        MODEL,
        max_model_len=128,
        max_num_seqs=1,
        enforce_eager=True,
        gpu_memory_utilization=0.4,
        speculative_config=SPEC_CONFIG,
        async_scheduling=False,
        disable_log_stats=False,
    ) as runner:
        llm = runner.get_llm()
        outputs = llm.generate([TokensPrompt(prompt_token_ids=prompt)], params)
        assert list(outputs[0].outputs[0].token_ids) == [42] * 32
        metrics = spec_decode_metrics(llm)
        assert metrics.get("vllm:spec_decode_num_accepted_tokens", 0) > 0
