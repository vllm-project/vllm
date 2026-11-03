from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).parents[1] / "scripts" / "vllm_weight_update_client")
)

from update_timing import (  # noqa: E402
    STAGES,
    GiB,
    format_timing_table,
    summarize_update_timing,
)


def _bucket(**seconds: float) -> dict:
    return {"tensor_count": 3, "total_bytes": GiB, **seconds}


class TestSummarizeUpdateTiming:
    def test_breakdown(self):
        buckets = [
            _bucket(
                source_get_tensor_seconds=1.0,
                source_to_device_seconds=0.5,
                materialize_seconds=2.0,
                nccl_send_seconds=0.75,
                server_wait_seconds=0.25,
            ),
            _bucket(
                source_get_tensor_seconds=1.0,
                source_to_device_seconds=0.25,
                source_cpu_pack_seconds=0.25,
                materialize_seconds=1.5,
                nccl_send_seconds=0.75,
                server_wait_seconds=1.25,
            ),
        ]
        timing = summarize_update_timing(
            buckets, start_rpc_seconds=0.5, finish_rpc_seconds=3.0, total_seconds=10.0
        )
        assert timing["stages_seconds"] == pytest.approx(
            {
                "start_rpc": 0.5,
                "read": 2.0,
                "h2d": 0.75,
                "pack": 0.25,
                "materialize_other": 0.5,
                "nccl_send": 1.5,
                "server_wait": 1.5,
                "finish_rpc": 3.0,
                "other": 0.0,
            }
        )
        assert (timing["bucket_count"], timing["tensor_count"]) == (2, 6)
        assert timing["throughput_gib_per_second"] == pytest.approx(0.2)

    def test_uncovered_time_goes_to_other_and_never_negative(self):
        timing = summarize_update_timing(
            [_bucket(materialize_seconds=1.0, source_get_tensor_seconds=2.0)],
            start_rpc_seconds=0.0,
            finish_rpc_seconds=0.0,
            total_seconds=1.0,
        )
        assert timing["stages_seconds"]["materialize_other"] == 0.0
        assert timing["stages_seconds"]["other"] == 0.0

    def test_table_lists_every_stage(self):
        timing = summarize_update_timing(
            [_bucket(server_wait_seconds=1.0)],
            start_rpc_seconds=0.0,
            finish_rpc_seconds=1.0,
            total_seconds=4.0,
        )
        table = format_timing_table(timing, title="main update")
        assert table.splitlines()[0].startswith(
            "main update: 4.00 s, 1.0 GiB in 1 buckets"
        )
        rows = {line.split()[0]: line.split()[1:] for line in table.splitlines()[2:]}
        assert list(rows) == list(STAGES)
        assert rows["server_wait"] == ["1.00", "25.0%"]
        assert rows["other"] == ["2.00", "50.0%"]
