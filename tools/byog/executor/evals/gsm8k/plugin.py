# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real GSM8K few-shot eval plugin using vLLM's OpenAI `/v1/completions` API."""

from __future__ import annotations

import concurrent.futures
import http.client
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from tools.byog.executor.interfaces import (
    EvalRecordWriter,
    EvalRunner,
    PointResult,
    ServerRunContext,
    WorkloadPoint,
)
from tools.byog.schemas.result import ErrorKind, ExecutionStatus

GSM8K_STOP_STRINGS = ["Question", "Assistant:", "<|separator|>"]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _clean_number_token(token: str) -> str | None:
    """Normalize a candidate numeric token (strip currency/commas/trailing dot)."""
    s = token.strip().lstrip("$").rstrip(".,;:!?%")
    s = s.replace(",", "")
    if not s:
        return None
    sign = ""
    if s[0] in ("+", "-"):
        sign = "-" if s[0] == "-" else ""
        s = s[1:]
    if not s:
        return None
    dots = 0
    for ch in s:
        if ch == ".":
            dots += 1
        elif not ch.isdigit():
            return None
    if dots > 1 or s == ".":
        return None
    try:
        val = float(sign + s)
    except ValueError:
        return None
    if val.is_integer():
        return str(int(val))
    return f"{val:g}"


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract the final numeric answer from GSM8K reference or model text."""
    if not text:
        return None
    # 1. Canonical GSM8K `#### <number>` marker.
    marker_idx = text.rfind("####")
    if marker_idx != -1:
        tail = text[marker_idx + 4 :].strip()
        for tok in tail.split():
            num = _clean_number_token(tok)
            if num is not None:
                return num

    # 2. `The answer is <number>` phrasing (case-insensitive).
    lower = text.lower()
    phrase = "the answer is"
    p_idx = lower.rfind(phrase)
    if p_idx != -1:
        tail = text[p_idx + len(phrase) :].strip()
        for tok in tail.split():
            num = _clean_number_token(tok)
            if num is not None:
                return num

    # 3. Fallback: last valid numeric token in the response.
    last_num: str | None = None
    for tok in text.split():
        num = _clean_number_token(tok)
        if num is not None:
            last_num = num
    return last_num


def build_few_shot_prompt(
    train_rows: list[dict[str, Any]],
    question: str,
    num_shots: int,
) -> str:
    parts: list[str] = []
    for ex in train_rows[:num_shots]:
        ex_q = str(ex["question"]).strip()
        ex_a = str(ex["answer"]).strip()
        parts.append(f"Question: {ex_q}\nAnswer: {ex_a}")
    parts.append(f"Question: {question.strip()}\nAnswer:")
    return "\n\n".join(parts)


class GSM8KEvalRunner(EvalRunner):
    """Real few-shot GSM8K evaluator over an OpenAI-compatible vLLM server."""

    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]:
        concurrencies = params.get("concurrency", [1])
        if isinstance(concurrencies, int):
            concurrencies = [concurrencies]
        return [
            WorkloadPoint(
                point_id=f"c{int(c)}",
                params={**params, "concurrency": int(c)},
            )
            for c in concurrencies
        ]

    def run_point(
        self,
        point: WorkloadPoint,
        ctx: ServerRunContext,
        record_writer: EvalRecordWriter,
    ) -> PointResult:
        p = point.params
        concurrency = int(p.get("concurrency", 1))
        repeats = int(p.get("repeats", 1))
        num_shots = int(p.get("num_shots", 5))
        num_questions = int(p.get("num_questions", 20))
        max_tokens = int(p.get("max_tokens", 256))
        temperature = float(p.get("temperature", 0.0))

        train_asset = ctx.assets.resolve(ctx.workload_assets["gsm8k_train"])
        test_asset = ctx.assets.resolve(ctx.workload_assets["gsm8k_test"])

        train_rows = _load_jsonl(train_asset.path)
        test_rows = _load_jsonl(test_asset.path)[:num_questions]

        deadline = time.monotonic() + max(1, ctx.timeout_s)
        tasks: list[tuple[int, int, dict[str, Any]]] = [
            (q_idx, rep, row)
            for rep in range(repeats)
            for q_idx, row in enumerate(test_rows)
        ]

        def _eval_one(task: tuple[int, int, dict[str, Any]]) -> dict[str, Any]:
            q_idx, rep, row = task
            question = str(row["question"])
            ref_raw = str(row["answer"])
            ref_num = extract_gsm8k_answer(ref_raw) or ""
            prompt = build_few_shot_prompt(train_rows, question, num_shots)
            payload = {
                "model": ctx.model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop": GSM8K_STOP_STRINGS,
            }
            req_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{ctx.base_url}/v1/completions",
                data=req_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            per_req_timeout = min(60.0, max(1.0, deadline - time.monotonic()))
            t0 = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=per_req_timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                latency_s = time.monotonic() - t0
                choices = body.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise ValueError("Missing 'choices' in completion response.")
                text = str(choices[0].get("text", ""))
                usage = body.get("usage") or {}
                out_toks = usage.get("completion_tokens")
                pred = extract_gsm8k_answer(text)
                is_correct = pred is not None and pred == ref_num
                return {
                    "sample_id": f"gsm8k-{q_idx:04d}",
                    "repeat": rep,
                    "question": question,
                    "reference_answer": ref_num,
                    "rendered_input": prompt,
                    "model_response": text,
                    "extracted_answer": pred,
                    "score": 1.0 if is_correct else 0.0,
                    "correct": is_correct,
                    "error_kind": ErrorKind.NONE,
                    "error_message": None,
                    "latency_s": latency_s,
                    "output_tokens": int(out_toks)
                    if isinstance(out_toks, int)
                    else None,
                }
            except urllib.error.HTTPError as err:
                err_kind = ErrorKind.MODEL if 400 <= err.code < 500 else ErrorKind.INFRA
                return {
                    "sample_id": f"gsm8k-{q_idx:04d}",
                    "repeat": rep,
                    "question": question,
                    "reference_answer": ref_num,
                    "rendered_input": prompt,
                    "model_response": "",
                    "extracted_answer": None,
                    "score": None,
                    "correct": None,
                    "error_kind": err_kind,
                    "error_message": f"HTTP {err.code}: {err.reason}",
                    "latency_s": time.monotonic() - t0,
                    "output_tokens": None,
                }
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                TimeoutError,
                OSError,
            ) as err:
                return {
                    "sample_id": f"gsm8k-{q_idx:04d}",
                    "repeat": rep,
                    "question": question,
                    "reference_answer": ref_num,
                    "rendered_input": prompt,
                    "model_response": "",
                    "extracted_answer": None,
                    "score": None,
                    "correct": None,
                    "error_kind": ErrorKind.INFRA,
                    "error_message": f"{type(err).__name__}: {err}",
                    "latency_s": time.monotonic() - t0,
                    "output_tokens": None,
                }
            except (ValueError, KeyError, TypeError) as err:
                return {
                    "sample_id": f"gsm8k-{q_idx:04d}",
                    "repeat": rep,
                    "question": question,
                    "reference_answer": ref_num,
                    "rendered_input": prompt,
                    "model_response": "",
                    "extracted_answer": None,
                    "score": None,
                    "correct": None,
                    "error_kind": ErrorKind.MODEL,
                    "error_message": f"{type(err).__name__}: {err}",
                    "latency_s": time.monotonic() - t0,
                    "output_tokens": None,
                }

        results: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, concurrency)
        ) as pool:
            results = list(pool.map(_eval_one, tasks))

        attempted = len(results)
        completed = 0
        correct_count = 0
        invalid_count = 0
        model_errors = 0
        infra_errors = 0
        latencies: list[float] = []

        for item in results:
            ek: ErrorKind = item["error_kind"]
            if ek == ErrorKind.NONE:
                completed += 1
                if item["correct"]:
                    correct_count += 1
                if item["extracted_answer"] is None:
                    invalid_count += 1
                if isinstance(item["latency_s"], float):
                    latencies.append(item["latency_s"])
            elif ek == ErrorKind.MODEL:
                model_errors += 1
            else:
                infra_errors += 1

            record_writer.write_response(
                dataset_id="gsm8k_test",
                dataset_revision=test_asset.resolved_revision,
                sample_id=item["sample_id"],
                question=item["question"],
                reference_answer=item["reference_answer"],
                rendered_input=item["rendered_input"],
                point_id=point.point_id,
                repeat=item["repeat"],
                concurrency=concurrency,
                model_response=item["model_response"],
                extracted_answer=item["extracted_answer"],
                score=item["score"],
                correct=item["correct"],
                error_kind=ek,
                error_message=item["error_message"],
                latency_s=item["latency_s"],
                output_tokens=item["output_tokens"],
            )

        scoring_denom = completed + model_errors
        accuracy = (correct_count / scoring_denom) if scoring_denom > 0 else None
        invalid_rate = (invalid_count / completed) if completed > 0 else None
        mean_lat = (sum(latencies) / len(latencies)) if latencies else None

        summary_path = ctx.output_dir / f"summary_{point.point_id}.json"
        summary_payload = {
            "point_id": point.point_id,
            "generation": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "num_shots": num_shots,
                "stop": GSM8K_STOP_STRINGS,
            },
            "assets": {
                "gsm8k_train": {
                    "resolved_revision": train_asset.resolved_revision,
                    "sha256": train_asset.sha256,
                    "is_override": train_asset.is_override,
                },
                "gsm8k_test": {
                    "resolved_revision": test_asset.resolved_revision,
                    "sha256": test_asset.sha256,
                    "is_override": test_asset.is_override,
                },
            },
            "attempted": attempted,
            "completed": completed,
            "scoring_denominator": scoring_denom,
            "correct": correct_count,
            "invalid": invalid_count,
            "model_errors": model_errors,
            "infra_errors": infra_errors,
            "accuracy": accuracy,
            "invalid_rate": invalid_rate,
        }
        summary_path.write_text(
            json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        status = ExecutionStatus.COMPLETED if completed > 0 else ExecutionStatus.FAILED
        return PointResult(
            status=status,
            metrics={
                "accuracy": accuracy,
                "invalid_rate": invalid_rate,
                "mean_latency_s": mean_lat,
            },
            attempted=attempted,
            completed=completed,
            scoring_denominator=scoring_denom,
            model_errors=model_errors,
            infra_errors=infra_errors,
            error_message=(
                f"{infra_errors} infra error(s) during GSM8K"
                if infra_errors and completed == 0
                else None
            ),
            artifacts=[summary_path],
        )
