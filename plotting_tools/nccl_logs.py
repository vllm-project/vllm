"""NCCL INFO log parsing for communication-size summaries."""

from __future__ import annotations

import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

_NCCL_OP_TO_BUCKET = {
    "AllReduce": "all_reduce",
    "AllGather": "all_gather",
    "ReduceScatter": "reduce_scatter",
    "AllToAll": "all_to_all",
    "AllToAllv": "all_to_all",
    "Broadcast": "broadcast",
    "Reduce": "reduce",
    "Send": "point_to_point",
    "Recv": "point_to_point",
}

_COLLECTIVE_OPS = {
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "all_to_all",
    "broadcast",
    "reduce",
}

# NCCL enum values. The parser primarily trusts explicit "N Bytes" lines, and
# only uses this table when a log has opCount/count/datatype but no size line.
_NCCL_DTYPE_BYTES = {
    0: 1,   # int8 / char
    1: 1,   # uint8
    2: 4,   # int32
    3: 4,   # uint32
    4: 8,   # int64
    5: 8,   # uint64
    6: 2,   # float16
    7: 4,   # float32
    8: 8,   # float64
    9: 2,   # bfloat16
    10: 1,  # float8 e4m3
    11: 1,  # float8 e5m2
}

_RANK_RE = re.compile(r"\[(?P<rank>\d+)\]\s+NCCL INFO")
_INIT_RE = re.compile(
    r"ncclCommInitRank(?:Config)?\s+comm\s+(?P<comm>\S+)\s+"
    r"rank\s+(?P<rank>\d+)\s+nranks\s+(?P<nranks>\d+)"
)
_COMM_RE = re.compile(
    r"\bcomm\s+(?P<comm>\S+)\s+rank\s+(?P<rank>\d+)\s+"
    r"nRanks\s+(?P<nranks>\d+)\s+nNodes\s+(?P<nodes>\d+)\s+"
    r"localRanks\s+(?P<local_ranks>\d+)\s+localRank\s+(?P<local_rank>\d+)"
)
_OP_RE = re.compile(
    r"NCCL INFO\s+(?P<op>AllReduce|AllGather|ReduceScatter|AllToAllv?|"
    r"Broadcast|Reduce|Send|Recv):\s+opCount\s+(?P<op_count>\d+).*?"
    r"\bcount\s+(?P<count>\d+)\s+datatype\s+(?P<datatype>\d+).*?"
    r"\[nranks=(?P<nranks>\d+)\]"
)
_BYTES_RE = re.compile(
    r"NCCL INFO\s+(?P<op>AllReduce|AllGather|ReduceScatter|AllToAllv?|"
    r"Broadcast|Reduce|Send|Recv):\s+(?P<bytes>\d+)\s+Bytes\s+->"
)
_PEER_RE = re.compile(r"\bpeer\s+(?P<peer>\d+)\b")
_CHANNEL_LINK_RE = re.compile(
    r"NCCL INFO\s+Channel\s+(?P<channel>\d+)/(?:\d+)\s+:\s+"
    r"(?P<src>\d+)\[(?P<src_local>\d+)\]\s+->\s+"
    r"(?P<dst>\d+)\[(?P<dst_local>\d+)\]\s+via\s+(?P<transport>.+)"
)
_CHANNEL_RING_RE = re.compile(
    r"NCCL INFO\s+Channel\s+(?P<channel>\d+)/(?:\d+)\s+:\s+"
    r"(?P<ranks>\d+(?:\s+\d+)+)\s*$"
)


def _line_rank(line: str, default: int | None = None) -> int | None:
    match = _RANK_RE.search(line)
    if match:
        return int(match.group("rank"))
    return default


def _fallback_bytes(count: int | None, datatype: int | None) -> int | None:
    if count is None or datatype is None:
        return None
    dtype_bytes = _NCCL_DTYPE_BYTES.get(datatype)
    if dtype_bytes is None:
        return None
    return count * dtype_bytes


def _algorithmic_rank_bytes(op: str, logical_bytes: float, nranks: int) -> float:
    """Approximate per-rank transfer from NCCL op size, not measured link bytes."""
    if logical_bytes <= 0 or nranks <= 1:
        return logical_bytes
    if op == "all_reduce":
        return 2.0 * (nranks - 1) / nranks * logical_bytes
    if op == "all_gather":
        return (nranks - 1) * logical_bytes
    if op in {"reduce_scatter", "all_to_all"}:
        return (nranks - 1) / nranks * logical_bytes
    if op in {"broadcast", "reduce"}:
        return logical_bytes
    return logical_bytes


def _new_op_record(
    *,
    path: Path,
    line_no: int,
    rank: int | None,
    op_raw: str,
    op_count: int | None = None,
    count: int | None = None,
    datatype: int | None = None,
    nranks: int | None = None,
    peer: int | None = None,
    bytes_value: int | None = None,
    bytes_source: str = "unknown",
) -> dict[str, Any]:
    op = _NCCL_OP_TO_BUCKET.get(op_raw, "nccl_other")
    if bytes_value is None:
        bytes_value = _fallback_bytes(count, datatype)
        if bytes_value is not None:
            bytes_source = "count_x_dtype"
    return {
        "file": str(path),
        "line": line_no,
        "rank": rank,
        "op": op,
        "op_raw": op_raw,
        "op_count": op_count,
        "count": count,
        "datatype": datatype,
        "nranks": nranks,
        "peer": peer,
        "bytes": bytes_value,
        "bytes_source": bytes_source,
    }


def parse_nccl_logs(log_paths: list[Path]) -> dict[str, Any]:
    """Parse NCCL INFO logs into operation and channel records."""
    ops: list[dict[str, Any]] = []
    channels: list[dict[str, Any]] = []
    rings: list[dict[str, Any]] = []
    comms: dict[str, dict[str, Any]] = {}
    pending: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    line_counts: dict[str, int] = {}

    for path in sorted(log_paths):
        current_rank: int | None = None
        with path.open(errors="replace") as f:
            for line_no, line in enumerate(f, 1):
                line_counts[str(path)] = line_no
                current_rank = _line_rank(line, current_rank)

                init = _INIT_RE.search(line)
                if init:
                    comm = init.group("comm")
                    comms.setdefault(comm, {})
                    comms[comm].update({
                        "rank": int(init.group("rank")),
                        "nranks": int(init.group("nranks")),
                        "file": str(path),
                    })
                    continue

                comm_match = _COMM_RE.search(line)
                if comm_match:
                    comm = comm_match.group("comm")
                    comms.setdefault(comm, {})
                    comms[comm].update({
                        "rank": int(comm_match.group("rank")),
                        "nranks": int(comm_match.group("nranks")),
                        "nodes": int(comm_match.group("nodes")),
                        "local_ranks": int(comm_match.group("local_ranks")),
                        "local_rank": int(comm_match.group("local_rank")),
                        "file": str(path),
                    })
                    continue

                op_match = _OP_RE.search(line)
                if op_match:
                    op_raw = op_match.group("op")
                    peer_match = _PEER_RE.search(line)
                    record = _new_op_record(
                        path=path,
                        line_no=line_no,
                        rank=current_rank,
                        op_raw=op_raw,
                        op_count=int(op_match.group("op_count")),
                        count=int(op_match.group("count")),
                        datatype=int(op_match.group("datatype")),
                        nranks=int(op_match.group("nranks")),
                        peer=(
                            int(peer_match.group("peer"))
                            if peer_match else None
                        ),
                    )
                    ops.append(record)
                    pending[op_raw].append(record)
                    continue

                bytes_match = _BYTES_RE.search(line)
                if bytes_match:
                    op_raw = bytes_match.group("op")
                    bytes_value = int(bytes_match.group("bytes"))
                    if pending[op_raw]:
                        record = pending[op_raw].popleft()
                        record["bytes"] = bytes_value
                        record["bytes_source"] = "explicit_nccl_log"
                        continue
                    ops.append(_new_op_record(
                        path=path,
                        line_no=line_no,
                        rank=current_rank,
                        op_raw=op_raw,
                        bytes_value=bytes_value,
                        bytes_source="explicit_nccl_log_without_opcount",
                    ))
                    continue

                channel_match = _CHANNEL_LINK_RE.search(line)
                if channel_match:
                    channels.append({
                        "file": str(path),
                        "line": line_no,
                        "logger_rank": current_rank,
                        "channel": int(channel_match.group("channel")),
                        "src": int(channel_match.group("src")),
                        "src_local": int(channel_match.group("src_local")),
                        "dst": int(channel_match.group("dst")),
                        "dst_local": int(channel_match.group("dst_local")),
                        "transport": channel_match.group("transport").strip(),
                    })
                    continue

                ring_match = _CHANNEL_RING_RE.search(line)
                if ring_match:
                    rank_order = [
                        int(token)
                        for token in ring_match.group("ranks").split()
                    ]
                    rings.append({
                        "file": str(path),
                        "line": line_no,
                        "logger_rank": current_rank,
                        "channel": int(ring_match.group("channel")),
                        "rank_order": rank_order,
                    })

    return {
        "log_files": [str(p) for p in sorted(log_paths)],
        "line_counts": line_counts,
        "comms": comms,
        "ops": ops,
        "channels": channels,
        "rings": rings,
    }


def summarize_nccl_log_ops(parsed: dict[str, Any]) -> dict[str, Any]:
    ops = parsed.get("ops") or []
    summary: dict[str, dict[str, Any]] = {}
    per_rank: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for rec in ops:
        op = rec.get("op") or "nccl_other"
        rank = rec.get("rank")
        bytes_value = int(rec.get("bytes") or 0)
        nranks = int(rec.get("nranks") or 0)
        algo_bytes = _algorithmic_rank_bytes(op, bytes_value, nranks)
        bucket = summary.setdefault(op, {
            "count": 0,
            "logical_bytes": 0,
            "algorithmic_rank_bytes_estimated": 0.0,
            "bytes_missing": 0,
            "bytes_source_counts": {},
            "datatypes": {},
            "counts": {},
        })
        bucket["count"] += 1
        bucket["logical_bytes"] += bytes_value
        bucket["algorithmic_rank_bytes_estimated"] += algo_bytes
        if not bytes_value:
            bucket["bytes_missing"] += 1
        source_counts = Counter(bucket["bytes_source_counts"])
        source_counts[str(rec.get("bytes_source") or "unknown")] += 1
        bucket["bytes_source_counts"] = dict(source_counts)
        if rec.get("datatype") is not None:
            dtypes = Counter(bucket["datatypes"])
            dtypes[str(rec["datatype"])] += 1
            bucket["datatypes"] = dict(dtypes)
        if rec.get("count") is not None:
            counts = Counter(bucket["counts"])
            counts[str(rec["count"])] += 1
            bucket["counts"] = dict(counts)

        if rank is not None:
            rank_bucket = per_rank[str(rank)].setdefault(op, {
                "count": 0,
                "logical_bytes": 0,
                "algorithmic_rank_bytes_estimated": 0.0,
                "bytes_missing": 0,
            })
            rank_bucket["count"] += 1
            rank_bucket["logical_bytes"] += bytes_value
            rank_bucket["algorithmic_rank_bytes_estimated"] += algo_bytes
            if not bytes_value:
                rank_bucket["bytes_missing"] += 1

    return {
        "ops": summary,
        "per_rank_ops": {rank: ops for rank, ops in sorted(per_rank.items())},
    }


def infer_nccl_world_size(parsed: dict[str, Any], fallback: int = 1) -> int:
    ranks: set[int] = set()
    nranks_seen: list[int] = []
    for rec in parsed.get("ops") or []:
        if rec.get("rank") is not None:
            ranks.add(int(rec["rank"]))
        if rec.get("nranks") is not None:
            nranks_seen.append(int(rec["nranks"]))
    for rec in parsed.get("channels") or []:
        ranks.add(int(rec["src"]))
        ranks.add(int(rec["dst"]))
    for comm in (parsed.get("comms") or {}).values():
        if comm.get("rank") is not None:
            ranks.add(int(comm["rank"]))
        if comm.get("nranks") is not None:
            nranks_seen.append(int(comm["nranks"]))
    inferred = max(ranks) + 1 if ranks else fallback
    if nranks_seen:
        inferred = max(inferred, max(nranks_seen))
    return max(inferred, fallback, 1)


def build_nccl_log_rank_matrices(
    parsed: dict[str, Any],
    *,
    n_ranks: int | None = None,
) -> dict[str, np.ndarray]:
    """Build logical and algorithmic matrices from parsed NCCL op-size lines."""
    n = n_ranks or infer_nccl_world_size(parsed)
    logical = np.zeros((n, n), dtype=np.float64)
    algorithmic = np.zeros((n, n), dtype=np.float64)
    for rec in parsed.get("ops") or []:
        src = rec.get("rank")
        if src is None or src < 0 or src >= n:
            continue
        op = rec.get("op") or "nccl_other"
        bytes_value = float(rec.get("bytes") or 0)
        if bytes_value <= 0:
            continue
        if op == "point_to_point" and rec.get("peer") is not None:
            peer = int(rec["peer"])
            if 0 <= peer < n and peer != src:
                if rec.get("op_raw") == "Recv":
                    logical[peer, src] += bytes_value
                    algorithmic[peer, src] += bytes_value
                else:
                    logical[src, peer] += bytes_value
                    algorithmic[src, peer] += bytes_value
            continue
        nranks = int(rec.get("nranks") or n)
        peers = [r for r in range(min(nranks, n)) if r != src]
        if not peers:
            continue
        logical_share = bytes_value / len(peers)
        algo_total = _algorithmic_rank_bytes(op, bytes_value, nranks)
        algo_share = algo_total / len(peers)
        if op in _COLLECTIVE_OPS:
            for dst in peers:
                logical[src, dst] += logical_share
                algorithmic[src, dst] += algo_share
        else:
            for dst in peers:
                logical[src, dst] += logical_share
                algorithmic[src, dst] += algo_share
    return {
        "logical_bytes": logical,
        "algorithmic_bytes_estimated": algorithmic,
    }


def build_nccl_channel_matrix(
    parsed: dict[str, Any],
    *,
    n_ranks: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a rank matrix from NCCL channel setup lines."""
    n = n_ranks or infer_nccl_world_size(parsed)
    matrix = np.zeros((n, n), dtype=np.float64)
    by_transport: Counter[str] = Counter()
    for rec in parsed.get("channels") or []:
        src = int(rec["src"])
        dst = int(rec["dst"])
        if 0 <= src < n and 0 <= dst < n:
            matrix[src, dst] += 1
        by_transport[str(rec.get("transport") or "unknown")] += 1

    ring_matrix = np.zeros((n, n), dtype=np.float64)
    for rec in parsed.get("rings") or []:
        order = rec.get("rank_order") or []
        if len(order) < 2:
            continue
        for i, src in enumerate(order):
            dst = order[(i + 1) % len(order)]
            if 0 <= src < n and 0 <= dst < n:
                ring_matrix[src, dst] += 1

    meta = {
        "channel_link_count": int(matrix.sum()),
        "ring_edge_count": int(ring_matrix.sum()),
        "transport_counts": dict(by_transport),
    }
    return matrix, {"matrix": matrix, "ring_matrix": ring_matrix, "meta": meta}


def nccl_rank_summary(parsed: dict[str, Any]) -> dict[str, Any]:
    ranks = sorted({
        int(rec["rank"])
        for rec in parsed.get("ops") or []
        if rec.get("rank") is not None
    })
    comm_nranks = sorted({
        int(comm["nranks"])
        for comm in (parsed.get("comms") or {}).values()
        if comm.get("nranks") is not None
    })
    return {
        "ranks_seen": ranks,
        "nranks_seen": comm_nranks,
        "log_file_count": len(parsed.get("log_files") or []),
        "op_record_count": len(parsed.get("ops") or []),
        "channel_link_record_count": len(parsed.get("channels") or []),
        "ring_record_count": len(parsed.get("rings") or []),
    }
