"""Where one weight-update transaction spends its time, seen from the publisher.

Pure functions over the per-bucket records the publisher collects, so they can
be tested without torch or vLLM.

Stages of one transaction (``start -> buckets -> finish``):

- ``start_rpc``: ``/start_weight_update`` round trip (server-side preparation).
- per bucket, sender side:
  - ``read``: safetensors reads (``source_get_tensor_seconds``);
  - ``h2d``: host-to-device copies (``source_to_device_seconds``);
  - ``pack``: CPU packing for direct-file expert copies;
  - ``materialize_other``: the rest of building the bucket;
  - ``nccl_send``: packed broadcast until the sender's device is done;
- per bucket, server side:
  - ``server_wait``: after the send, waiting for ``/update_weights`` to
    return, i.e. the part of the server's ``load_weights`` that did not
    overlap the transfer;
- ``finish_rpc``: ``/finish_weight_update`` round trip (post-load processing).
- ``other``: what the stages above do not cover (metadata, bucket planning,
  Python overhead).
"""

from __future__ import annotations

from typing import Any

GiB = 1024**3

STAGES = (
    "start_rpc",
    "read",
    "h2d",
    "pack",
    "materialize_other",
    "nccl_send",
    "server_wait",
    "finish_rpc",
    "other",
)


def summarize_update_timing(
    buckets: list[dict[str, Any]],
    *,
    start_rpc_seconds: float,
    finish_rpc_seconds: float,
    total_seconds: float,
) -> dict[str, Any]:
    """Aggregate per-bucket records into one breakdown of the transaction."""
    read = sum(b.get("source_get_tensor_seconds", 0.0) for b in buckets)
    h2d = sum(b.get("source_to_device_seconds", 0.0) for b in buckets)
    pack = sum(b.get("source_cpu_pack_seconds", 0.0) for b in buckets)
    materialize = sum(b.get("materialize_seconds", 0.0) for b in buckets)
    stages = {
        "start_rpc": start_rpc_seconds,
        "read": read,
        "h2d": h2d,
        "pack": pack,
        # reads and copies happen inside materialize; never negative
        "materialize_other": max(0.0, materialize - read - h2d - pack),
        "nccl_send": sum(b.get("nccl_send_seconds", 0.0) for b in buckets),
        "server_wait": sum(b.get("server_wait_seconds", 0.0) for b in buckets),
        "finish_rpc": finish_rpc_seconds,
    }
    stages["other"] = max(0.0, total_seconds - sum(stages.values()))
    total_bytes = sum(b.get("total_bytes", 0) for b in buckets)
    return {
        "total_seconds": total_seconds,
        "stages_seconds": stages,
        "bucket_count": len(buckets),
        "tensor_count": sum(b.get("tensor_count", 0) for b in buckets),
        "total_bytes": total_bytes,
        "throughput_gib_per_second": (
            total_bytes / GiB / total_seconds if total_seconds > 0 else 0.0
        ),
    }


def format_timing_table(timing: dict[str, Any], title: str = "update") -> str:
    """A fixed-width table: one row per stage, seconds and share of total."""
    total = timing["total_seconds"]
    lines = [
        f"{title}: {total:.2f} s, {timing['total_bytes'] / GiB:.1f} GiB in "
        f"{timing['bucket_count']} buckets "
        f"({timing['throughput_gib_per_second']:.2f} GiB/s)",
        f"  {'stage':<18}{'seconds':>10}{'share':>8}",
    ]
    for stage in STAGES:
        seconds = timing["stages_seconds"].get(stage, 0.0)
        share = seconds / total if total > 0 else 0.0
        lines.append(f"  {stage:<18}{seconds:>10.2f}{share:>8.1%}")
    return "\n".join(lines)
