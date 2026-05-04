#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare current vLLM HTTP logprobs with a captured oracle bundle.

The expected oracle format is a directory with request_*.json and matching
response_*.json files produced from /v1/completions with token-id logprob keys.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

Json = dict[str, Any]


class TokenNormalizer:
    """Normalize token-id placeholders to decoded token strings."""

    def __init__(self, decode_token_id):
        self._decode_token_id = decode_token_id

    def token_key(self, token: Any) -> str:
        key = _token_key(token)
        if not key.startswith("token_id:"):
            return key
        try:
            token_id = int(key.removeprefix("token_id:"))
        except ValueError:
            return key
        return self._decode_token_id(token_id)


def _token_key(token: Any) -> str:
    if isinstance(token, str):
        return token
    if isinstance(token, int):
        return f"token_id:{token}"
    return str(token)


def _choice(response: Json) -> Json:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response has no choices[0]")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("response choices[0] is not an object")
    return choice


def _normalize_token(token: Any, normalizer: TokenNormalizer | None) -> str:
    if normalizer is None:
        return _token_key(token)
    return normalizer.token_key(token)


def _generated_tokens(
    response: Json, normalizer: TokenNormalizer | None = None
) -> list[str]:
    choice = _choice(response)
    logprobs = choice.get("logprobs") or {}
    tokens = logprobs.get("tokens")
    if isinstance(tokens, list) and tokens:
        return [_normalize_token(token, normalizer) for token in tokens]
    token_ids = choice.get("token_ids")
    if isinstance(token_ids, list):
        return [_normalize_token(token, normalizer) for token in token_ids]
    return []


def _prompt_token_ids(response: Json) -> list[int] | None:
    token_ids = _choice(response).get("prompt_token_ids")
    if not isinstance(token_ids, list):
        return None
    return [int(token) for token in token_ids]


def _token_logprobs(response: Json) -> list[float | None]:
    logprobs = _choice(response).get("logprobs") or {}
    values = logprobs.get("token_logprobs")
    if not isinstance(values, list):
        return []
    out: list[float | None] = []
    for value in values:
        out.append(float(value) if value is not None else None)
    return out


def _top_logprobs(
    response: Json, normalizer: TokenNormalizer | None = None
) -> list[dict[str, float]]:
    logprobs = _choice(response).get("logprobs") or {}
    values = logprobs.get("top_logprobs")
    if not isinstance(values, list):
        return []
    out: list[dict[str, float]] = []
    for step in values:
        if not isinstance(step, dict):
            out.append({})
            continue
        out.append(
            {
                _normalize_token(token, normalizer): float(logprob)
                for token, logprob in step.items()
            }
        )
    return out


def _first_mismatch(oracle_tokens: list[str], actual_tokens: list[str]) -> Json | None:
    for step, (oracle_token, actual_token) in enumerate(
        zip(oracle_tokens, actual_tokens)
    ):
        if oracle_token != actual_token:
            return {"step": step, "oracle": oracle_token, "actual": actual_token}
    if len(oracle_tokens) != len(actual_tokens):
        step = min(len(oracle_tokens), len(actual_tokens))
        return {
            "step": step,
            "oracle": oracle_tokens[step] if step < len(oracle_tokens) else None,
            "actual": actual_tokens[step] if step < len(actual_tokens) else None,
        }
    return None


def _matching_prefix(oracle_tokens: list[str], actual_tokens: list[str]) -> int:
    count = 0
    for oracle_token, actual_token in zip(oracle_tokens, actual_tokens):
        if oracle_token != actual_token:
            break
        count += 1
    return count


def _top_keys(top_logprobs: dict[str, float], top_n: int) -> list[str]:
    return list(top_logprobs.keys())[:top_n]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _max(values: list[float]) -> float | None:
    return max(values) if values else None


def compare_response(
    case_name: str,
    oracle_response: Json,
    actual_response: Json,
    *,
    top_n: int = 50,
    normalizer: TokenNormalizer | None = None,
) -> Json:
    oracle_tokens = _generated_tokens(oracle_response, normalizer)
    actual_tokens = _generated_tokens(actual_response, normalizer)
    oracle_top = _top_logprobs(oracle_response, normalizer)
    actual_top = _top_logprobs(actual_response, normalizer)
    oracle_token_logprobs = _token_logprobs(oracle_response)
    actual_token_logprobs = _token_logprobs(actual_response)

    steps = min(
        len(oracle_tokens), len(actual_tokens), len(oracle_top), len(actual_top)
    )
    top1_matches = 0
    topk_overlaps: list[float] = []
    common_logprob_errors: list[float] = []
    chosen_token_logprob_errors: list[float] = []

    for step in range(steps):
        oracle_keys = _top_keys(oracle_top[step], top_n)
        actual_keys = _top_keys(actual_top[step], top_n)
        if oracle_keys and actual_keys and oracle_keys[0] == actual_keys[0]:
            top1_matches += 1

        oracle_set = set(oracle_keys)
        actual_set = set(actual_keys)
        if oracle_set:
            topk_overlaps.append(len(oracle_set & actual_set) / len(oracle_set))
        for token in oracle_set & actual_set:
            common_logprob_errors.append(
                abs(oracle_top[step][token] - actual_top[step][token])
            )

        if (
            oracle_tokens[step] == actual_tokens[step]
            and step < len(oracle_token_logprobs)
            and step < len(actual_token_logprobs)
            and oracle_token_logprobs[step] is not None
            and actual_token_logprobs[step] is not None
        ):
            chosen_token_logprob_errors.append(
                abs(oracle_token_logprobs[step] - actual_token_logprobs[step])
            )

    oracle_prompt_ids = _prompt_token_ids(oracle_response)
    actual_prompt_ids = _prompt_token_ids(actual_response)
    prompt_ids_match = (
        None
        if oracle_prompt_ids is None or actual_prompt_ids is None
        else oracle_prompt_ids == actual_prompt_ids
    )

    first_mismatch = _first_mismatch(oracle_tokens, actual_tokens)
    return {
        "case": case_name,
        "tokens_match": first_mismatch is None,
        "prompt_token_ids_match": prompt_ids_match,
        "first_token_mismatch": first_mismatch,
        "matching_prefix_tokens": _matching_prefix(oracle_tokens, actual_tokens),
        "oracle_token_count": len(oracle_tokens),
        "actual_token_count": len(actual_tokens),
        "compared_steps": steps,
        "top1_matches": top1_matches,
        "top1_match_rate": top1_matches / steps if steps else None,
        "topk_overlap_mean": _mean(topk_overlaps),
        "topk_overlap_min": min(topk_overlaps) if topk_overlaps else None,
        "max_common_logprob_abs_error": _max(common_logprob_errors),
        "mean_common_logprob_abs_error": _mean(common_logprob_errors),
        "max_chosen_token_logprob_abs_error": _max(chosen_token_logprob_errors),
        "mean_chosen_token_logprob_abs_error": _mean(chosen_token_logprob_errors),
    }


def _load_json(path: Path) -> Json:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def load_oracle_cases(oracle_dir: Path) -> list[tuple[str, Json, Json]]:
    cases: list[tuple[str, Json, Json]] = []
    request_paths = sorted(oracle_dir.glob("request_*.json"))
    if not request_paths:
        raise ValueError(f"{oracle_dir} has no request_*.json files")
    for request_path in request_paths:
        suffix = request_path.stem.removeprefix("request_")
        response_path = oracle_dir / f"response_{suffix}.json"
        if not response_path.exists():
            raise ValueError(f"missing {response_path.name} for {request_path.name}")
        cases.append((suffix, _load_json(request_path), _load_json(response_path)))
    return cases


def post_completion(base_url: str, payload: Json, timeout: float) -> Json:
    url = f"{base_url.rstrip('/')}/v1/completions"
    encoded = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError(f"{url} returned non-object JSON")
    return data


def load_token_normalizer(
    tokenizer: str,
    *,
    tokenizer_mode: str,
    trust_remote_code: bool,
) -> TokenNormalizer:
    from vllm.tokenizers import get_tokenizer

    hf_tokenizer = get_tokenizer(
        tokenizer,
        tokenizer_mode=tokenizer_mode,
        trust_remote_code=trust_remote_code,
    )

    def decode_token_id(token_id: int) -> str:
        return hf_tokenizer.decode([token_id])

    return TokenNormalizer(decode_token_id)


def summarize_reports(reports: list[Json]) -> Json:
    return {
        "case_count": len(reports),
        "all_tokens_match": all(report["tokens_match"] for report in reports),
        "all_prompt_token_ids_match": all(
            report["prompt_token_ids_match"] is not False for report in reports
        ),
        "min_top1_match_rate": min(
            (
                report["top1_match_rate"]
                for report in reports
                if report["top1_match_rate"] is not None
            ),
            default=None,
        ),
        "min_topk_overlap_mean": min(
            (
                report["topk_overlap_mean"]
                for report in reports
                if report["topk_overlap_mean"] is not None
            ),
            default=None,
        ),
        "max_common_logprob_abs_error": max(
            (
                report["max_common_logprob_abs_error"]
                for report in reports
                if report["max_common_logprob_abs_error"] is not None
            ),
            default=None,
        ),
        "max_chosen_token_logprob_abs_error": max(
            (
                report["max_chosen_token_logprob_abs_error"]
                for report in reports
                if report["max_chosen_token_logprob_abs_error"] is not None
            ),
            default=None,
        ),
    }


def _fails_thresholds(
    summary: Json, reports: list[Json], args: argparse.Namespace
) -> bool:
    failed = False
    if args.strict_tokens and not summary["all_tokens_match"]:
        failed = True
    if args.strict_prompt_token_ids and not summary["all_prompt_token_ids_match"]:
        failed = True
    if args.min_top1_match_rate is not None:
        value = summary["min_top1_match_rate"]
        failed = failed or value is None or value < args.min_top1_match_rate
    if args.min_topk_overlap_mean is not None:
        value = summary["min_topk_overlap_mean"]
        failed = failed or value is None or value < args.min_topk_overlap_mean
    if args.max_common_logprob_abs_error is not None:
        value = summary["max_common_logprob_abs_error"]
        failed = failed or value is None or value > args.max_common_logprob_abs_error
    if args.max_chosen_token_logprob_abs_error is not None:
        value = summary["max_chosen_token_logprob_abs_error"]
        failed = (
            failed or value is None or value > args.max_chosen_token_logprob_abs_error
        )
    if args.fail_on_first_mismatch:
        failed = failed or any(report["first_token_mismatch"] for report in reports)
    return failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", help="Override the model field from request_*.json")
    parser.add_argument(
        "--tokenizer",
        help=(
            "Tokenizer used to decode token_id:<id> logprob keys before "
            "comparing them with text token keys."
        ),
    )
    parser.add_argument("--tokenizer-mode", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--strict-tokens", action="store_true")
    parser.add_argument("--strict-prompt-token-ids", action="store_true")
    parser.add_argument("--fail-on-first-mismatch", action="store_true")
    parser.add_argument("--min-top1-match-rate", type=float)
    parser.add_argument("--min-topk-overlap-mean", type=float)
    parser.add_argument("--max-common-logprob-abs-error", type=float)
    parser.add_argument("--max-chosen-token-logprob-abs-error", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = load_oracle_cases(args.oracle_dir)
    normalizer = None
    if args.tokenizer:
        normalizer = load_token_normalizer(
            args.tokenizer,
            tokenizer_mode=args.tokenizer_mode,
            trust_remote_code=args.trust_remote_code,
        )
    reports: list[Json] = []
    for suffix, request_payload, oracle_response in cases:
        payload = dict(request_payload)
        if args.model:
            payload["model"] = args.model
        actual_response = post_completion(args.base_url, payload, args.timeout)
        reports.append(
            compare_response(
                f"request_{suffix}",
                oracle_response,
                actual_response,
                top_n=args.top_n,
                normalizer=normalizer,
            )
        )

    summary = summarize_reports(reports)
    result: Json = {"summary": summary, "cases": reports}
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 1 if _fails_thresholds(summary, reports, args) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
