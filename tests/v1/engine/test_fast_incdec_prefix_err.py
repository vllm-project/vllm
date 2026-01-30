# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import AutoTokenizer

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.detokenizer import IncrementalDetokenizer

# ruff: noqa: E501


def test_fast_inc_detok_invalid_utf8_err_case():
    """
    Test edge case where tokenizer can produce non-monotonic,
    invalid UTF-8 output, which breaks the internal state of
    tokenizers' DecodeStream.
    See https://github.com/vllm-project/vllm/issues/17448.

    Thanks to reproducer from @fpaupier:
    https://gist.github.com/fpaupier/0ed1375bd7633c5be6c894b1c7ac1be3.
    """
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

    # Create a test request
    prompt_token_ids = [107, 4606, 236787, 107]
    params = SamplingParams(skip_special_tokens=True)
    request = EngineCoreRequest(
        request_id="test",
        external_req_id="test-ext",
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        eos_token_id=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )

    detokenizer = IncrementalDetokenizer.from_new_request(tokenizer, request)

    assert detokenizer.__class__.__name__ == "FastIncrementalDetokenizer", (
        "Should use FastIncrementalDetokenizer by default"
    )

    # Process tokens incrementally
    test_tokens = [
        236840,
        107,
        138,
        236782,
        107,
        140,
        236775,
        6265,
        1083,
        623,
        121908,
        147418,
        827,
        107,
        140,
        236775,
        6265,
        236779,
        2084,
        1083,
        623,
        203292,
        827,
        107,
        140,
        236775,
        6265,
        236779,
        7777,
        1083,
        623,
        121908,
        147418,
        569,
        537,
        236789,
        65880,
        569,
        537,
        236789,
        62580,
        853,
        115693,
        210118,
        35178,
        16055,
        1270,
        759,
        215817,
        4758,
        1925,
        1117,
        827,
        107,
        140,
        236775,
        5654,
        1083,
        623,
        110733,
        46291,
        827,
        107,
        140,
        236775,
        5654,
        236779,
        2084,
        1083,
        623,
        136955,
        56731,
        827,
        107,
        140,
        236775,
        5654,
        236779,
        7777,
        1083,
        623,
        194776,
        2947,
        496,
        109811,
        1608,
        890,
        215817,
        4758,
        1925,
        1117,
        2789,
        432,
        398,
        602,
        31118,
        569,
        124866,
        134772,
        509,
        19478,
        1640,
        33779,
        236743,
        236770,
        236819,
        236825,
        236771,
        432,
        398,
        432,
        237167,
        827,
        107,
        140,
        236775,
        77984,
        1083,
        623,
        2709,
        236745,
        2555,
        513,
        236789,
        602,
        31118,
        569,
    ]

    output = ""
    for i, token_id in enumerate(test_tokens):
        detokenizer.update([token_id], False)

        finished = i == len(test_tokens) - 1
        output += detokenizer.get_next_output_text(finished, delta=True)

    assert (
        output
        == r"""[
  {
    "source": "Résultats",
    "source_type": "CONCEPT",
    "source_description": "Résultats de l'analyse de l'impact des opérations israéliennes sur la frontière libanaise",
    "target": "Israël",
    "target_type": "ORGANIZATION",
    "target_description": "Pays qui a obtenu à sa frontière libanaise « un niveau de calme inédit depuis les années 1960 »",
    "relationship": "Obtention d'un niveau de"""
    )
