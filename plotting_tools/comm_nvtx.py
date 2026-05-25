"""Parse vLLM inner comm NVTX markers (authoritative message records)."""

from __future__ import annotations

from typing import Any

# Keep in sync with vllm/distributed/iteration_phase_nvtx.py comm ops.
COMM_NVTX_OPS = frozenset({
    "send",
    "recv",
    "pp_send",
    "pp_recv",
    "pp_isend",
    "pp_irecv",
    "all_reduce",
    "all_to_all_dispatch",
    "all_to_all_combine",
})

ITERATION_PHASES = frozenset({"prefill", "decode", "mixed", "idle", "unknown"})

# Phases/sources safe for prefill-vs-decode conclusions (plots, summaries).
CONCLUSIVE_COMM_PHASES = frozenset({"prefill", "decode"})
TRUSTED_COMM_PHASE_SOURCES = frozenset({
    "comm_nvtx_label",
    "iter_id",
    "iteration_nvtx_overlap",
})
UNTRUSTED_COMM_PHASE_SOURCES = frozenset({"iteration_nvtx_nearest"})

PP_COMM_OPS = frozenset({"pp_send", "pp_recv", "pp_isend", "pp_irecv"})
LOW_LEVEL_COMM_OPS = frozenset({"send", "recv"})
LOW_LEVEL_TO_PP: dict[str, tuple[str, ...]] = {
    "send": ("pp_send", "pp_isend"),
    "recv": ("pp_recv", "pp_irecv"),
}

BYTES_LABEL = "logical tensor bytes"


def _looks_like_shape(token: str) -> bool:
    if token in ("-", ""):
        return True
    if "x" in token:
        return True
    return token.isdigit()


def parse_iter_nvtx_label(name: str) -> dict[str, Any] | None:
    """Parse ``iter|id=…|phase=…|rank=…|…`` outer worker-step NVTX labels."""
    if not name.startswith("iter|"):
        return None
    fields: dict[str, str] = {}
    for part in name.split("|")[1:]:
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        fields[key.strip()] = val.strip()
    phase = fields.get("phase", "").lower()
    if phase not in ("prefill", "decode", "mixed", "idle"):
        return None
    try:
        iter_id = int(fields.get("id", "0"))
    except ValueError:
        return None
    return {
        "name": phase,
        "scope": "iteration",
        "iter_id": iter_id,
        "phase": phase,
        "rank": int(fields.get("rank", "0")),
        "pp": int(fields.get("pp", "0")),
        "tp": int(fields.get("tp", "0")),
        "ctx_tokens": int(fields.get("ctx", "0")),
        "gen_tokens": int(fields.get("gen", "0")),
        "reqs": int(fields.get("reqs", "0")),
        "nvtx_name": name,
    }


def _parse_comm_nvtx_label_legacy(name: str) -> dict[str, Any] | None:
    """
    Parse legacy ``op:phase[:tensor_key]:shape:logical_tensor_bytes[:pN]``.
    """
    parts = name.split(":")
    if len(parts) < 4 or parts[0] not in COMM_NVTX_OPS:
        return None

    phase = parts[1].lower().strip()
    if phase not in ITERATION_PHASES:
        return None

    remaining = parts[2:]
    peer: int | None = None
    if (
        remaining
        and remaining[-1].startswith("p")
        and len(remaining[-1]) > 1
        and remaining[-1][1:].isdigit()
    ):
        peer = int(remaining[-1][1:])
        remaining = remaining[:-1]

    if len(remaining) < 2:
        return None

    nbytes_str = remaining[-1]
    shape = remaining[-2]
    prefix = remaining[:-2]
    tensor_key: str | None = None
    if len(prefix) == 1:
        if not _looks_like_shape(prefix[0]):
            tensor_key = prefix[0]
        else:
            return None
    elif len(prefix) != 0:
        return None

    logical_tensor_bytes: int | None = None
    if nbytes_str.isdigit():
        logical_tensor_bytes = int(nbytes_str)

    return {
        "op": parts[0],
        "phase": phase,
        "iter_id": None,
        "tensor_key": tensor_key,
        "shape": shape,
        "logical_tensor_bytes": logical_tensor_bytes,
        "peer": peer,
        "nvtx_name": name,
    }


def _parse_comm_nvtx_label_pipe(name: str) -> dict[str, Any] | None:
    """Parse ``comm|iter=…|op=…|phase=…|…`` labels."""
    if not name.startswith("comm|"):
        return None
    fields: dict[str, str] = {}
    for part in name.split("|")[1:]:
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        fields[key.strip()] = val.strip()
    op = fields.get("op", "")
    phase = fields.get("phase", "unknown").lower()
    if not op or phase not in ITERATION_PHASES:
        return None
    peer: int | None = None
    if fields.get("peer", "").isdigit():
        peer = int(fields["peer"])
    nbytes: int | None = None
    if fields.get("bytes", "").isdigit():
        nbytes = int(fields["bytes"])
    try:
        iter_id = int(fields.get("iter", "0"))
    except ValueError:
        iter_id = 0
    return {
        "op": op,
        "phase": phase,
        "iter_id": iter_id,
        "tensor_key": fields.get("key"),
        "shape": fields.get("shape", "-"),
        "logical_tensor_bytes": nbytes,
        "peer": peer,
        "rank": int(fields.get("rank", "0")),
        "pp": int(fields.get("pp", "0")),
        "tp": int(fields.get("tp", "0")),
        "nvtx_name": name,
    }


def parse_comm_nvtx_label(name: str) -> dict[str, Any] | None:
    """
    Parse vLLM inner comm NVTX markers (pipe or legacy colon format).

    Returns a comm message record dict, or None if not a vLLM comm marker.
    """
    if name.startswith("comm|"):
        return _parse_comm_nvtx_label_pipe(name)
    return _parse_comm_nvtx_label_legacy(name)


def _intervals_overlap(
    a_start: int, a_end: int, b_start: int, b_end: int
) -> bool:
    return a_start < b_end and a_end > b_start


def _same_message_identity(
    left: dict[str, Any],
    right: dict[str, Any],
) -> bool:
    return (
        left.get("shape") == right.get("shape")
        and left.get("logical_tensor_bytes") == right.get("logical_tensor_bytes")
        and left.get("peer") == right.get("peer")
    )


def dedupe_comm_nvtx_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Drop nested low-level send/recv when a PP marker covers the same message.

    Keeps pp_send / pp_recv / pp_isend / pp_irecv and collectives; removes
    redundant send/recv that overlap a matching PP op (same shape, bytes, peer).
    """
    pp_records = [r for r in records if r.get("op") in PP_COMM_OPS]
    kept: list[dict[str, Any]] = []
    removed = 0

    for record in records:
        op = record.get("op")
        if op not in LOW_LEVEL_COMM_OPS:
            kept.append(record)
            continue

        pp_ops = LOW_LEVEL_TO_PP.get(op, ())
        duplicate = False
        for pp in pp_records:
            if pp.get("op") not in pp_ops:
                continue
            if not _same_message_identity(record, pp):
                continue
            if _intervals_overlap(
                int(record["ts"]),
                int(record["end"]),
                int(pp["ts"]),
                int(pp["end"]),
            ):
                duplicate = True
                break
        if duplicate:
            removed += 1
        else:
            kept.append(record)

    stats = {
        "raw_count": len(records),
        "deduped_count": len(kept),
        "removed_nested_send_recv": removed,
    }
    return kept, stats


def is_conclusive_comm_record(record: dict[str, Any]) -> bool:
    """
    True only for prefill/decode comm with a trusted phase source.

    Excludes ``mixed`` (step-level batching) and ``iteration_nvtx_nearest``.
    """
    phase = (record.get("phase") or "unknown").lower().strip()
    if phase not in CONCLUSIVE_COMM_PHASES:
        return False
    source = (record.get("phase_source") or "comm_nvtx_label").strip()
    return source in TRUSTED_COMM_PHASE_SOURCES


def filter_conclusive_comm_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [r for r in records if is_conclusive_comm_record(r)]


def comm_phase_exclusion_counts(
    records: list[dict[str, Any]],
) -> dict[str, int]:
    """Counts of comm records excluded from prefill/decode conclusions."""
    counts: dict[str, int] = {
        "total": len(records),
        "conclusive": 0,
        "mixed": 0,
        "unknown": 0,
        "idle": 0,
        "untrusted_nearest": 0,
        "other": 0,
    }
    for record in records:
        if is_conclusive_comm_record(record):
            counts["conclusive"] += 1
            continue
        phase = (record.get("phase") or "unknown").lower().strip()
        source = record.get("phase_source") or ""
        if source in UNTRUSTED_COMM_PHASE_SOURCES:
            counts["untrusted_nearest"] += 1
        elif phase == "mixed":
            counts["mixed"] += 1
        elif phase == "unknown":
            counts["unknown"] += 1
        elif phase == "idle":
            counts["idle"] += 1
        else:
            counts["other"] += 1
    return counts


def _stamp_comm_phase_sources(records: list[dict[str, Any]]) -> None:
    """Label phases parsed from ``comm|…|phase=…`` at capture time."""
    for record in records:
        if record.get("phase_source") is not None:
            continue
        phase = (record.get("phase") or "unknown").lower().strip()
        if phase != "unknown":
            record["phase_source"] = "comm_nvtx_label"


def assign_comm_phases_from_iteration(
    records: list[dict[str, Any]],
    iteration_ranges: list[dict[str, Any]],
) -> int:
    """
    For comm records with phase=unknown, assign prefill/decode/mixed.

    1. Match ``iter_id`` when present on both comm and iteration records.
    2. Prefer narrowest overlapping outer iteration NVTX range.

    Does **not** use nearest-gap fallback (unreliable; left as unknown).
    """
    iter_id_to_phase: dict[int, str] = {}
    phase_ranges: list[tuple[str, int, int, int, int | None]] = []
    for row in iteration_ranges:
        name = (row.get("name") or row.get("phase") or "").lower().strip()
        if name not in ("prefill", "decode", "mixed"):
            continue
        start = int(row["ts"])
        end = int(row["end"])
        iter_id = row.get("iter_id")
        iter_id_int = int(iter_id) if iter_id is not None else None
        if iter_id_int is not None:
            iter_id_to_phase[iter_id_int] = name
        phase_ranges.append((name, start, end, end - start, iter_id_int))

    assigned = 0
    for record in records:
        if record.get("phase") != "unknown":
            continue
        record_iter = record.get("iter_id")
        if record_iter is not None:
            phase = iter_id_to_phase.get(int(record_iter))
            if phase is not None:
                record["phase"] = phase
                record["phase_source"] = "iter_id"
                assigned += 1
                continue
        start = int(record["ts"])
        end = int(record["end"])
        best: str | None = None
        best_span: int | None = None
        for phase, range_start, range_end, span, _iter_id in phase_ranges:
            if _intervals_overlap(start, end, range_start, range_end):
                if best_span is None or span < best_span:
                    best = phase
                    best_span = span
        if best is not None:
            record["phase"] = best
            record["phase_source"] = "iteration_nvtx_overlap"
            assigned += 1
    return assigned


def finalize_comm_nvtx_records(
    records: list[dict[str, Any]],
    iteration_ranges: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Dedupe nested markers, then assign unknown phases from iteration NVTX."""
    deduped, stats = dedupe_comm_nvtx_records(records)
    _stamp_comm_phase_sources(deduped)
    assigned = assign_comm_phases_from_iteration(deduped, iteration_ranges)
    stats["phases_assigned_from_iteration"] = assigned
    stats["unknown_phase_remaining"] = sum(
        1 for r in deduped if r.get("phase") == "unknown"
    )
    exclusion = comm_phase_exclusion_counts(deduped)
    stats.update({f"comm_{k}": v for k, v in exclusion.items()})
    return deduped, stats


def comm_nvtx_message_stats(
    records: list[dict[str, Any]],
    *,
    phase: str | None = None,
    op: str | None = None,
    conclusive_only: bool = False,
) -> dict[str, Any]:
    """Aggregate inner comm NVTX records (not CUPTI/kernel events)."""
    pool = filter_conclusive_comm_records(records) if conclusive_only else records
    filtered = [
        r
        for r in pool
        if (phase is None or r.get("phase") == phase)
        and (op is None or r.get("op") == op)
    ]
    sizes = [
        int(r["logical_tensor_bytes"])
        for r in filtered
        if r.get("logical_tensor_bytes") is not None
        and int(r["logical_tensor_bytes"]) > 0
    ]
    return {
        "count": len(filtered),
        "logical_tensor_bytes_list": sizes,
        "avg_logical_tensor_bytes": float(sum(sizes) / len(sizes)) if sizes else 0.0,
        "total_logical_tensor_bytes": int(sum(sizes)),
        "bytes_label": BYTES_LABEL,
        # Deprecated aliases for older summary consumers.
        "sizes_bytes": sizes,
        "avg_bytes": float(sum(sizes) / len(sizes)) if sizes else 0.0,
        "total_bytes": int(sum(sizes)),
    }
