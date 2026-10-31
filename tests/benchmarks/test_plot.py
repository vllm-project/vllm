# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path

import pytest

from vllm.benchmarks.plot import construct_timeline_data, generate_timeline_plot


@pytest.mark.parametrize(
    ("ttft", "expected_end"),
    [
        (1.2344, "00:00:01.234"),
        (59.9994, "00:00:59.999"),
        (59.9999, "00:01:00.000"),
        (3599.9999, "01:00:00.000"),
    ],
)
def test_timeline_rounding_carries_into_minutes_and_hours(ttft, expected_end):
    segments = construct_timeline_data(
        [
            {
                "start_time": 10.0,
                "ttft": ttft,
                "itl": [0.025],
                "latency": ttft + 0.025,
                "prompt_len": 8,
                "output_tokens": 2,
            }
        ],
        [0.025, 0.050],
        ["fast", "medium", "slow"],
    )

    assert len(segments) == 2
    assert segments[0]["start"] == "00:00:00.000"
    assert segments[0]["end"] == expected_end
    assert segments[1]["start"] == expected_end
    for segment in segments:
        assert segment["segment_start"] == segment["start"]
        assert segment["segment_end"] == segment["end"]
        assert segment["req_finish_time"] == segments[1]["end"]
        assert segment["prompt_tokens"] == 8
        assert segment["output_tokens"] == 2
    assert segments[1]["type"] == "medium"


def test_timeline_html_export_at_minute_boundary(tmp_path: Path):
    output = tmp_path / "timeline.html"
    generate_timeline_plot(
        [
            {
                "start_time": 10.0,
                "ttft": 59.9999,
                "itl": [0.025],
                "latency": 60.0249,
                "prompt_len": 8,
                "output_tokens": 2,
            }
        ],
        output,
    )

    html = output.read_text()
    assert "Plotly.newPlot" in html
    assert "00:01:00.000" in html
    assert "00:00:60.000" not in html
