# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example real 5-shot GSM8K eval plugin over OpenAI `/v1/completions`."""

from __future__ import annotations

import concurrent.futures
import hashlib
import http.client
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from tools.byog.interfaces import (
    EvalRecordWriter,
    EvalRunner,
    PointResult,
    ServerRunContext,
    WorkloadPoint,
)
from tools.byog.schemas.execution import ErrorKind, ExecutionStatus

GSM8K_STOP = ["Question", "Assistant:", "<|separator|>"]


def _load_jsonl(
    source: str, expected_sha256: str, timeout_s: float = 60.0
) -> list[dict[str, Any]]:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=timeout_s) as resp:
            raw = resp.read()
    else:
        raw = Path(source).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"SHA256 mismatch for '{source}': expected {expected_sha256}, got {actual}."
        )
    return [json.loads(ln) for ln in raw.decode("utf-8").splitlines() if ln.strip()]


def _parse_num(token: str) -> str | None:
    s = token.strip().lstrip("$").rstrip(".,;:!?%").replace(",", "")
    if not s:
        return None
    sign = "-" if s[0] == "-" else ""
    body = s[1:] if s[0] in ("+", "-") else s
    if not body or body == "." or body.count(".") > 1:
        return None
    if not all(ch.isdigit() or ch == "." for ch in body):
        return None
    val = float(sign + body)
    return str(int(val)) if val.is_integer() else f"{val:g}"


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract final numeric answer from GSM8K reference or model response."""
    for marker in ("####", "the answer is"):
        idx = text.lower().rfind(marker)
        if idx != -1:
            for tok in text[idx + len(marker) :].split():
                if (num := _parse_num(tok)) is not None:
                    return num
    last: str | None = None
    for tok in text.split():
        if (num := _parse_num(tok)) is not None:
            last = num
    return last


class GSM8KEvalRunner(EvalRunner):
    """5-shot GSM8K evaluator emitting deduplicated per-question JSONL records."""

    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]:
        concs = params.get("concurrency", [1])
        if isinstance(concs, int):
            concs = [concs]
        return [
            WorkloadPoint(f"c{int(c)}", {**params, "concurrency": int(c)})
            for c in concs
        ]

    def run_point(
        self,
        point: WorkloadPoint,
        ctx: ServerRunContext,
        record_writer: EvalRecordWriter,
    ) -> PointResult:
        p = point.params
        conc = int(p.get("concurrency", 1))
        repeats = int(p.get("repeats", 1))
        shots = int(p.get("num_shots", 5))
        n_q = int(p.get("num_questions", 20))
        max_toks = int(p.get("max_tokens", 256))
        temp = float(p.get("temperature", 0.0))

        train_src = str(p["train_source"])
        test_src = str(p["test_source"])
        train_sha = str(p["train_sha256"])
        test_sha = str(p["test_sha256"])
        ds_rev = str(p.get("dataset_revision", "vllm-ci-gsm8k-v1"))
        train_rows = _load_jsonl(train_src, train_sha)[:shots]
        test_rows = _load_jsonl(test_src, test_sha)[:n_q]
        prefix = "\n\n".join(
            f"Question: {str(r['question']).strip()}\n"
            f"Answer: {str(r['answer']).strip()}"
            for r in train_rows
        )
        deadline = time.monotonic() + max(1, ctx.timeout_s)
        tasks = [
            (q_idx, rep, row)
            for rep in range(repeats)
            for q_idx, row in enumerate(test_rows)
        ]

        def _eval_one(task: tuple[int, int, dict[str, Any]]) -> dict[str, Any]:
            q_idx, rep, row = task
            q = str(row["question"])
            ref = extract_gsm8k_answer(str(row["answer"])) or ""
            prompt = f"{prefix}\n\nQuestion: {q.strip()}\nAnswer:"
            body = json.dumps(
                {
                    "model": ctx.model,
                    "prompt": prompt,
                    "temperature": temp,
                    "max_tokens": max_toks,
                    "stop": GSM8K_STOP,
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                f"{ctx.base_url}/v1/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            base = {
                "sample_id": f"gsm8k-{q_idx:04d}",
                "repeat": rep,
                "question": q,
                "reference_answer": ref,
                "rendered_input": prompt,
            }
            t0 = time.monotonic()
            try:
                timeout = min(60.0, max(1.0, deadline - time.monotonic()))
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    parsed = json.loads(resp.read().decode("utf-8"))
                text = str(parsed["choices"][0]["text"])
                pred = extract_gsm8k_answer(text)
                ok = pred is not None and pred == ref
                return {
                    **base,
                    "model_response": text,
                    "extracted_answer": pred,
                    "score": 1.0 if ok else 0.0,
                    "correct": ok,
                    "error_kind": ErrorKind.NONE,
                    "error_message": None,
                    "latency_s": time.monotonic() - t0,
                }
            except urllib.error.HTTPError as err:
                # 401/403/404/429 (e.g. wrong model name, auth, rate-limit) are infra.
                ek = ErrorKind.MODEL if err.code in (400, 422) else ErrorKind.INFRA
                return {
                    **base,
                    "model_response": "",
                    "extracted_answer": None,
                    "score": None,
                    "correct": None,
                    "error_kind": ek,
                    "error_message": f"HTTP {err.code}: {err.reason}",
                    "latency_s": time.monotonic() - t0,
                }
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                TimeoutError,
                OSError,
            ) as err:
                return {
                    **base,
                    "model_response": "",
                    "extracted_answer": None,
                    "score": None,
                    "correct": None,
                    "error_kind": ErrorKind.INFRA,
                    "error_message": str(err),
                    "latency_s": time.monotonic() - t0,
                }
            except (ValueError, KeyError, IndexError, TypeError) as err:
                return {
                    **base,
                    "model_response": "",
                    "extracted_answer": None,
                    "score": None,
                    "correct": None,
                    "error_kind": ErrorKind.MODEL,
                    "error_message": str(err),
                    "latency_s": time.monotonic() - t0,
                }

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, conc)) as pool:
            rows = list(pool.map(_eval_one, tasks))

        completed = correct = invalid = m_err = i_err = 0
        for item in rows:
            ek: ErrorKind = item["error_kind"]
            if ek == ErrorKind.NONE:
                completed += 1
                correct += int(bool(item["correct"]))
                invalid += int(item["extracted_answer"] is None)
            elif ek == ErrorKind.MODEL:
                m_err += 1
            else:
                i_err += 1
            record_writer.write_response(
                dataset_id="gsm8k_test",
                dataset_revision=ds_rev,
                sample_id=item["sample_id"],
                question=item["question"],
                reference_answer=item["reference_answer"],
                rendered_input=item["rendered_input"],
                point_id=point.point_id,
                repeat=item["repeat"],
                concurrency=conc,
                model_response=item["model_response"],
                extracted_answer=item["extracted_answer"],
                score=item["score"],
                correct=item["correct"],
                error_kind=ek,
                error_message=item["error_message"],
                latency_s=item["latency_s"],
            )

        denom = completed + m_err
        point_ok = completed > 0 and i_err == 0
        return PointResult(
            status=ExecutionStatus.COMPLETED if point_ok else ExecutionStatus.FAILED,
            metrics={
                "accuracy": (correct / denom) if denom else None,
                "invalid_rate": (invalid / completed) if completed else None,
            },
            attempted=len(rows),
            completed=completed,
            scoring_denominator=denom,
            model_errors=m_err,
            infra_errors=i_err,
            error_message=f"{i_err} infra error(s)" if i_err else None,
        )
