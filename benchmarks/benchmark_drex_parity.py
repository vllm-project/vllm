"""Compare vLLM's Drex v1.5 decisions with the reference ``code/kev``.

Each input line is a ``/v1/systemone`` request body::

    {"state": ..., "questions": {"q0": {"type": "choice", "instructions": "",
     "criteria": {"a": "...", "b": "..."}}}}

An optional top-level ``labels`` mapping (question id -> the correct option key)
enables expected calibration error; it is stripped before the request is sent.

The reference probabilities come from ``code/kev`` in ``--reference-dir`` (a
checkpoint snapshot that holds ``head.pt``, the shards and ``code/kev``); the
vLLM probabilities come from a running ``/v1/systemone`` server. The script
prints per-option error, argmax agreement, ECE and, with ``--flip``, the
option-order flip rate, and exits non-zero when the bars are missed.

Usage on a GPU box::

    python benchmarks/benchmark_drex_parity.py \
        --reference-dir /models/drex-v1.5 --cases parity.jsonl \
        --url http://localhost:8000
"""

import argparse
import json
import sys
from pathlib import Path

import torch

NoulTypes = {"noul"}


def _read_cases(path: str) -> list[dict]:
    cases = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def _normalize_vllm(question_type: str, answer: dict) -> dict[str, float]:
    """Map a ``/v1/systemone`` answer onto the reference's key order.

    ``noul`` reads as ``(no, yes)`` in both implementations, but the reference
    names the keys ``false``/``true`` while the OpenAI answer uses ``no``/``yes``.
    """
    probabilities = answer["probabilities"]
    if question_type == "noul":
        return {
            "false": float(probabilities["no"]),
            "true": float(probabilities["yes"]),
        }
    return {str(k): float(v) for k, v in probabilities.items()}


def _reference_vector(meta: dict, probs: torch.Tensor) -> dict[str, float]:
    return {str(key): float(p) for key, p in zip(meta["keys"], probs.tolist())}


def _argmax(vector: dict[str, float]) -> str:
    return max(vector, key=vector.get)


def _ece(rows: list[tuple[float, bool]], bins: int = 10) -> float:
    """Expected calibration error over the top-1 confidence."""
    if not rows:
        return 0.0
    total = len(rows)
    error = 0.0
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        bucket = [
            (c, hit)
            for c, hit in rows
            if low <= c < high or (b == bins - 1 and c == 1.0)
        ]
        if not bucket:
            continue
        confidence = sum(c for c, _ in bucket) / len(bucket)
        accuracy = sum(1 for _, hit in bucket if hit) / len(bucket)
        error += len(bucket) / total * abs(confidence - accuracy)
    return error


def _post(url: str, model: str, body: dict) -> dict:
    import urllib.request

    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/systemone",
        data=json.dumps({**body, "model": model}).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())["answers"]


def _flip_body(body: dict) -> dict:
    """Reverse the option order of choice/score questions, keeping ids."""
    flipped = {"state": body["state"], "questions": {}}
    for qid, spec in body["questions"].items():
        spec = dict(spec)
        if spec["type"] == "choice":
            spec["criteria"] = dict(reversed(list(spec["criteria"].items())))
        elif spec["type"] == "score" and isinstance(spec["criteria"], list):
            spec["criteria"] = list(reversed(spec["criteria"]))
        flipped["questions"][qid] = spec
    return flipped


def _flipped_answer(question_type: str, criteria, answer: dict) -> str:
    """Name the content a flipped request chose the original way."""
    moved = _argmax(_normalize_vllm(question_type, answer))
    if question_type == "score" and isinstance(criteria, list):
        return str(len(criteria) - 1 - int(moved))
    return moved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="drex-v1.5")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--max-diff", type=float, default=1e-3)
    args = parser.parse_args()

    sys.path.insert(0, args.reference_dir)
    from code.kev import api as ref_api
    from code.kev.checkpoint import load as ref_load

    dtype = getattr(torch, args.dtype) if args.dtype else None
    tokenizer, model = ref_load(args.reference_dir, device=args.device, dtype=dtype)

    cases = _read_cases(args.cases)
    diffs: list[float] = []
    hits = 0
    total = 0
    calibration: list[tuple[float, bool]] = []
    flips = 0
    flip_total = 0

    for index, case in enumerate(cases):
        labels = case.pop("labels", {})
        reference_request = ref_api.SystemOneRequest(**case)
        record, meta = ref_api.to_record(reference_request)
        reference = model.probs(model.encode(tokenizer, record))

        answers = _post(args.url, args.model, case)
        for meta_q, probs in zip(meta, reference):
            qid = meta_q["id"]
            wanted = _reference_vector(meta_q, probs)
            got = _normalize_vllm(meta_q["type"], answers[qid])
            if set(wanted) != set(got):
                print(f"[{index}] {qid}: keys differ {sorted(wanted)} != {sorted(got)}")
                return 2
            diff = max(abs(wanted[key] - got[key]) for key in wanted)
            diffs.append(diff)
            total += 1
            agree = _argmax(wanted) == _argmax(got)
            hits += agree
            if qid in labels:
                confidence = max(got.values())
                calibration.append((confidence, _argmax(got) == str(labels[qid])))
            if not agree:
                print(f"[{index}] {qid}: argmax {_argmax(wanted)} != {_argmax(got)}")

        if args.flip:
            flipped = _post(args.url, args.model, _flip_body(case))
            for qid, spec in case["questions"].items():
                if spec["type"] not in ("choice", "score"):
                    continue
                original = _argmax(_normalize_vllm(spec["type"], answers[qid]))
                moved = _flipped_answer(spec["type"], spec["criteria"], flipped[qid])
                flip_total += 1
                if original != moved:
                    flips += 1

    max_diff = max(diffs, default=0.0)
    agreement = hits / total if total else 1.0
    ece = _ece(calibration)
    print(f"cases={len(cases)} questions={total}")
    print(f"max-abs-diff={max_diff:.6g} (bar {args.max_diff:g})")
    print(f"argmax-agreement={agreement:.2%} (bar 100%)")
    if calibration:
        print(f"ece={ece:.6g}")
    if args.flip:
        print(f"option-order-flip-rate={flips / flip_total:.2%} of {flip_total}")

    if max_diff > args.max_diff or agreement < 1.0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
