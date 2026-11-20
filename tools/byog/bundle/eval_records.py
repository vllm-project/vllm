# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deduplicated eval record writer enforcing catalog PublicationPolicy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tools.byog.schemas.catalog import PublicationPolicy
from tools.byog.schemas.result import (
    ErrorKind,
    EvalRenderedInputRecord,
    EvalResponseRecord,
    EvalSampleRecord,
)


def _canonical_hash(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


class DeduplicatedEvalRecordWriter:
    """Writes deduplicated `samples.jsonl`, `inputs.jsonl`, and `responses.jsonl`.

    Enforces an immutable copy of the catalog's `PublicationPolicy` so
    unreviewed or restricted text (questions, references, model responses,
    extracted answers, and free-text error messages) is never leaked.
    """

    def __init__(
        self,
        *,
        workload: str,
        output_dir: Path,
        policy: PublicationPolicy,
    ):
        self._workload = workload
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        # Snapshot an immutable copy of the catalog policy.
        self._policy = policy.model_copy(deep=True)
        self._samples_path = self._output_dir / "samples.jsonl"
        self._inputs_path = self._output_dir / "inputs.jsonl"
        self._responses_path = self._output_dir / "responses.jsonl"

        self._seen_samples: set[tuple[str, str, str]] = set()
        self._seen_inputs: set[str] = set()
        self._written_count: int = 0
        self._withheld_fields: set[str] = set()

        self._samples_fh = self._samples_path.open("w", encoding="utf-8")
        self._inputs_fh = self._inputs_path.open("w", encoding="utf-8")
        self._responses_fh = self._responses_path.open("w", encoding="utf-8")
        self._closed = False

    @property
    def policy(self) -> PublicationPolicy:
        return self._policy

    @property
    def written_count(self) -> int:
        return self._written_count

    def withheld_summary(self) -> list[dict[str, str]]:
        return [
            {
                "workload": self._workload,
                "field": field_name,
                "reason": self._withheld_reason(),
            }
            for field_name in sorted(self._withheld_fields)
        ]

    def _withheld_reason(self) -> str:
        if not self._policy.reviewed:
            return "publication_policy_unreviewed"
        return "publication_policy_restricted"

    def write_response(
        self,
        *,
        dataset_id: str,
        dataset_revision: str,
        sample_id: str,
        question: str,
        reference_answer: str,
        rendered_input: Any,
        point_id: str,
        repeat: int,
        concurrency: int,
        model_response: str,
        extracted_answer: Any,
        score: float | None,
        correct: bool | None,
        error_kind: ErrorKind = ErrorKind.NONE,
        error_message: str | None = None,
        latency_s: float | None = None,
        output_tokens: int | None = None,
        media_refs: list[str] | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("Cannot write to a closed EvalRecordWriter.")

        pub_q = self._policy.can_publish_question()
        pub_ref = self._policy.can_publish_reference()
        pub_resp = self._policy.can_publish_response()
        reason = self._withheld_reason()

        if not pub_q:
            self._withheld_fields.update({"question", "rendered_input"})
        if not pub_ref:
            self._withheld_fields.add("reference_answer")
        if not pub_resp:
            self._withheld_fields.update(
                {"model_response", "extracted_answer", "error_message"}
            )

        sample_key = (dataset_id, dataset_revision, sample_id)
        if sample_key not in self._seen_samples:
            self._seen_samples.add(sample_key)
            sample_rec = EvalSampleRecord(
                workload=self._workload,
                dataset_id=dataset_id,
                dataset_revision=dataset_revision,
                sample_id=sample_id,
                question=question if pub_q else None,
                reference_answer=reference_answer if pub_ref else None,
                media_refs=list(media_refs or []),
                withheld_reason=None if (pub_q and pub_ref) else reason,
            )
            self._samples_fh.write(sample_rec.model_dump_json() + "\n")

        input_hash = _canonical_hash(
            {
                "dataset_id": dataset_id,
                "dataset_revision": dataset_revision,
                "sample_id": sample_id,
                "rendered_input": rendered_input,
            }
        )
        if input_hash not in self._seen_inputs:
            self._seen_inputs.add(input_hash)
            input_rec = EvalRenderedInputRecord(
                workload=self._workload,
                dataset_id=dataset_id,
                input_hash=input_hash,
                dataset_revision=dataset_revision,
                sample_id=sample_id,
                rendered_input=rendered_input if pub_q else None,
                withheld_reason=None if pub_q else reason,
            )
            self._inputs_fh.write(input_rec.model_dump_json() + "\n")

        resp_rec = EvalResponseRecord(
            workload=self._workload,
            point_id=point_id,
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            sample_id=sample_id,
            input_hash=input_hash,
            repeat=repeat,
            concurrency=concurrency,
            model_response=model_response if pub_resp else None,
            extracted_answer=extracted_answer if pub_resp else None,
            score=score,
            correct=correct,
            error_kind=error_kind,
            error_message=error_message if pub_resp else None,
            latency_s=latency_s,
            output_tokens=output_tokens,
            withheld_reason=None if pub_resp else reason,
        )
        self._responses_fh.write(resp_rec.model_dump_json() + "\n")
        self._written_count += 1

    def close(self) -> list[Path]:
        if not self._closed:
            self._samples_fh.close()
            self._inputs_fh.close()
            self._responses_fh.close()
            self._closed = True
        return [self._samples_path, self._inputs_path, self._responses_path]
