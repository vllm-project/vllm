# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.inputs import tokens_input

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("spans", [[(0, 2)], []])
def test_beams_keep_the_bidirectional_spans_of_their_prompt(spans):
    """Every beam starts with the prompt the spans index."""
    prompt = tokens_input([1, 2, 3])
    prompt["bidirectional_spans"] = spans
    beam = BeamSearchSequence(orig_prompt=prompt, tokens=[1, 2, 3, 4], logprobs=[])
    assert beam.get_prompt()["bidirectional_spans"] == spans


def test_beams_of_prompts_without_spans_carry_none():
    beam = BeamSearchSequence(
        orig_prompt=tokens_input([1, 2, 3]), tokens=[1, 2, 3, 4], logprobs=[]
    )
    assert "bidirectional_spans" not in beam.get_prompt()
