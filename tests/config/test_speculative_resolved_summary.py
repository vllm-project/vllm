# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Snapshot tests of the resolved speculative config logged at startup.
Only config files are fetched, no weights."""

from dataclasses import dataclass
from typing import Any

import pytest

from vllm.engine.arg_utils import EngineArgs


@dataclass
class Case:
    """`EngineArgs` for a speculative setup and the summary it must log."""

    test_id: str
    engine_args: dict[str, Any]
    expected_summary: str


# The EAGLE-3 draft config has no rope keys, so the draft runs RoPE base 10000
# while the target runs llama3 RoPE with base 500000 (#61231). The draft runs
# without TP.
case_eagle3 = Case(
    test_id="eagle3",
    engine_args={
        "model": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "tensor_parallel_size": 2,
        "speculative_config": {
            "method": "eagle3",
            "model": "vllm-project/EAGLE3-LLaMA3.1-Instruct-8B",
            "num_speculative_tokens": 3,
            "draft_tensor_parallel_size": 1,
        },
    },
    expected_summary="""\
Resolved speculative config:
  method: eagle3
  num_speculative_tokens: 3
  model: draft=vllm-project/EAGLE3-LLaMA3.1-Instruct-8B, target=unsloth/Meta-Llama-3.1-8B-Instruct
  revision: draft=main, target=main
  max_position_embeddings: draft=8192, target=131072
  rope_parameters: draft={'rope_theta': 10000.0, 'rope_type': 'default'}, target={'factor': 8.0, 'high_freq_factor': 4.0, 'low_freq_factor': 1.0, 'original_max_position_embeddings': 8192, 'rope_type': 'llama3', 'rope_theta': 500000.0}
  tensor_parallel_size: draft=1, target=2
  decode_context_parallel_size: draft=1, target=1""",
)

# The MTP draft inherits the target's YaRN override (#58080) and TP, but not
# its --revision (#61197). The override is merged into the checkpoint's
# rope_parameters, so rope_theta and the M-RoPE keys come from the checkpoint.
case_mtp = Case(
    test_id="mtp",
    engine_args={
        "model": "Qwen/Qwen3.5-0.8B",
        "revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "tensor_parallel_size": 2,
        "hf_overrides": {
            "text_config": {
                "rope_parameters": {
                    "rope_type": "yarn",
                    "factor": 4.0,
                    "original_max_position_embeddings": 262144,
                }
            }
        },
        "speculative_config": {"method": "mtp", "num_speculative_tokens": 2},
    },
    expected_summary="""\
Resolved speculative config:
  method: mtp
  num_speculative_tokens: 2
  model: draft=Qwen/Qwen3.5-0.8B, target=Qwen/Qwen3.5-0.8B
  revision: draft=main, target=2fc06364715b967f1860aea9cf38778875588b17
  max_position_embeddings: draft=262144, target=262144
  rope_parameters: draft={'mrope_interleaved': True, 'mrope_section': [11, 11, 10], 'rope_type': 'yarn', 'rope_theta': 10000000, 'partial_rotary_factor': 0.25, 'factor': 4.0, 'original_max_position_embeddings': 262144}, target={'mrope_interleaved': True, 'mrope_section': [11, 11, 10], 'rope_type': 'yarn', 'rope_theta': 10000000, 'partial_rotary_factor': 0.25, 'factor': 4.0, 'original_max_position_embeddings': 262144}
  tensor_parallel_size: draft=2, target=2
  decode_context_parallel_size: draft=1, target=1""",
)

# No draft model, so there are no draft settings to compare.
case_ngram = Case(
    test_id="ngram",
    engine_args={
        "model": "Qwen/Qwen3.5-0.8B",
        "speculative_config": {
            "method": "ngram",
            "num_speculative_tokens": 4,
            "prompt_lookup_max": 4,
        },
    },
    expected_summary="""\
Resolved speculative config:
  method: ngram
  num_speculative_tokens: 4""",
)

all_cases = [case_eagle3, case_mtp, case_ngram]


@pytest.mark.cpu_test
@pytest.mark.parametrize("case", all_cases, ids=lambda case: case.test_id)
def test_resolved_summary(case: Case):
    engine_args = EngineArgs(
        max_model_len=8192,
        # Explicit, so TP > 1 doesn't check the GPU count.
        distributed_executor_backend="mp",
        **case.engine_args,
    )
    vllm_config = engine_args.create_engine_config()
    assert (spec_config := vllm_config.speculative_config) is not None
    assert spec_config.resolved_summary(vllm_config) == case.expected_summary
