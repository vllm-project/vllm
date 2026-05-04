# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).parents[2] / "tools" / "compare_vllm_http_logprobs_oracle.py"
)
spec = importlib.util.spec_from_file_location(
    "compare_vllm_http_logprobs_oracle", SCRIPT_PATH
)
assert spec is not None
oracle_compare = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(oracle_compare)


def _response(tokens, top_logprobs):
    token_ids = [
        int(token.split(":", 1)[1])
        for token in tokens
        if isinstance(token, str) and token.startswith("token_id:")
    ]
    return {
        "choices": [
            {
                "text": "",
                "logprobs": {
                    "tokens": tokens,
                    "token_logprobs": [-0.1 * (i + 1) for i in range(len(tokens))],
                    "top_logprobs": top_logprobs,
                },
                "token_ids": token_ids,
                "prompt_token_ids": [1, 2, 3],
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": len(tokens)},
    }


def test_compare_response_accepts_identical_top_logprobs():
    top_logprobs = [
        {"token_id:10": -0.1, "token_id:20": -1.0},
        {"token_id:11": -0.2, "token_id:21": -1.2},
    ]
    report = oracle_compare.compare_response(
        "case0",
        _response(["token_id:10", "token_id:11"], top_logprobs),
        _response(["token_id:10", "token_id:11"], top_logprobs),
        top_n=2,
    )

    assert report["tokens_match"] is True
    assert report["first_token_mismatch"] is None
    assert report["top1_matches"] == 2
    assert report["topk_overlap_mean"] == 1.0
    assert report["max_common_logprob_abs_error"] == 0.0


def test_compare_response_reports_first_generated_token_divergence():
    oracle = _response(
        ["token_id:10", "token_id:11"],
        [
            {"token_id:10": -0.1, "token_id:20": -1.0},
            {"token_id:11": -0.2, "token_id:21": -1.2},
        ],
    )
    actual = _response(
        ["token_id:10", "token_id:99"],
        [
            {"token_id:10": -0.1, "token_id:20": -1.0},
            {"token_id:99": -0.2, "token_id:21": -1.2},
        ],
    )

    report = oracle_compare.compare_response("case0", oracle, actual, top_n=2)

    assert report["tokens_match"] is False
    assert report["first_token_mismatch"] == {
        "step": 1,
        "oracle": "token_id:11",
        "actual": "token_id:99",
    }
    assert report["matching_prefix_tokens"] == 1
    assert report["top1_matches"] == 1


def test_compare_response_can_decode_oracle_token_id_keys():
    normalizer = oracle_compare.TokenNormalizer(
        lambda token_id: {10: '","', 11: "title", 20: " What"}[token_id]
    )
    oracle = _response(
        ["token_id:10", "token_id:11"],
        [
            {"token_id:10": -0.1, "token_id:20": -1.0},
            {"token_id:11": -0.2, "token_id:20": -1.2},
        ],
    )
    actual = _response(
        ['","', "title"],
        [
            {'","': -0.11, " What": -1.1},
            {"title": -0.19, " What": -1.3},
        ],
    )

    report = oracle_compare.compare_response(
        "case0", oracle, actual, top_n=2, normalizer=normalizer
    )

    assert report["tokens_match"] is True
    assert report["top1_matches"] == 2
    assert report["topk_overlap_mean"] == 1.0
    assert report["max_common_logprob_abs_error"] == 0.10000000000000009
