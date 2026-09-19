# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression: crash dumps must not leak structured-output schemas.

``NewRequestData.anon_repr`` obfuscates the prompt but used to emit
``sampling_params`` verbatim, so the full json/regex/grammar/structural_tag
was written to the logs at ERROR level on an engine crash.
"""

from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.core.sched.output import NewRequestData


def test_structured_outputs_anonymized_redacts_content():
    schema = '{"properties": {"buyerCompanyName": {"type": "string"}}}'
    anon = StructuredOutputsParams(json=schema).anonymized()
    assert schema not in repr(anon)
    assert anon.json == f"<redacted len={len(schema)}>"
    # exactly-one-constraint invariant still holds
    assert anon.regex is None and anon.grammar is None


def test_anonymized_preserves_non_content_flags():
    anon = StructuredOutputsParams(
        regex="secret.*pattern", disable_any_whitespace=True
    ).anonymized()
    assert "secret" not in anon.regex
    assert anon.disable_any_whitespace is True


def test_new_request_data_anon_repr_hides_schema():
    schema = '{"properties": {"ssn": {"type": "string"}}}'
    req = NewRequestData(
        req_id="r0",
        prompt_token_ids=[1, 2, 3],
        mm_features=[],
        sampling_params=SamplingParams(
            structured_outputs=StructuredOutputsParams(json=schema)
        ),
        pooling_params=None,
        block_ids=([],),
        num_computed_tokens=0,
        lora_request=None,
    )
    dumped = req.anon_repr()
    assert "ssn" not in dumped
    assert schema not in dumped
    # the non-anonymized repr still carries it (debug-only, opt-in)
    assert "ssn" in repr(req)
