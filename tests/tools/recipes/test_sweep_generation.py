# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import sys
from pathlib import Path

import pytest

RECIPES_DIR = Path(__file__).parents[3] / "tools" / "recipes"
sys.path.insert(0, str(RECIPES_DIR))

from runtime_tuning import WorkloadHints  # noqa: E402
from sweep_generation import (  # noqa: E402
    ModelParallelMetadata,
    build_parallel_layout_plan,
    model_parallel_metadata_from_config,
    write_parallel_layout_sweep_files,
)

WORKLOAD = WorkloadHints(input_tokens=128, output_tokens=128, concurrency=32)
CONFIG = {
    "model": "test/model",
    "max-num-seqs": 32,
    "max-num-batched-tokens": 2048,
}

# Real google/gemma-4-26B-A4B-it text-config values. These are regression-test
# fixtures only; production values are loaded from the selected model config.
GEMMA4_26B_NUM_EXPERTS = 128
GEMMA4_26B_MOE_INTERMEDIATE_SIZE = 704


def _layout_sizes(candidates):
    return [
        (item["tensor_parallel_size"], item["data_parallel_size"])
        for item in candidates
    ]


def test_extracts_gemma4_moe_metadata_from_nested_mapping():
    metadata = model_parallel_metadata_from_config(
        {
            "model_type": "multimodal_wrapper",
            "text_config": {
                "num_experts": GEMMA4_26B_NUM_EXPERTS,
                "moe_intermediate_size": GEMMA4_26B_MOE_INTERMEDIATE_SIZE,
            },
        }
    )

    assert metadata == ModelParallelMetadata(
        is_moe=True,
        num_experts=GEMMA4_26B_NUM_EXPERTS,
        moe_intermediate_size=GEMMA4_26B_MOE_INTERMEDIATE_SIZE,
    )


@pytest.mark.parametrize(
    ("detected_numa_nodes", "expected_layouts"),
    [
        (2, [(2, 1), (1, 2)]),
        (4, [(4, 1), (2, 2), (1, 4)]),
        (6, [(4, 1), (2, 3), (1, 6)]),
        (8, [(8, 1), (4, 2), (2, 4), (1, 8)]),
    ],
)
def test_dense_layouts_follow_detected_numa_count(
    detected_numa_nodes, expected_layouts
):
    candidates, skipped = build_parallel_layout_plan(
        CONFIG,
        WORKLOAD,
        detected_numa_nodes,
        model_metadata=ModelParallelMetadata(is_moe=False),
        cpu_moe_dp_supported=False,
    )

    assert _layout_sizes(candidates) == expected_layouts
    assert skipped == []


@pytest.mark.parametrize(
    ("detected_numa_nodes", "expected_tp_sizes"),
    [(2, [2, 1]), (4, [4, 2, 1]), (6, [4, 2, 1]), (8, [8, 4, 2, 1])],
)
def test_moe_dp_one_replacements_follow_detected_numa_count(
    detected_numa_nodes, expected_tp_sizes
):
    candidates, _ = build_parallel_layout_plan(
        CONFIG,
        WORKLOAD,
        detected_numa_nodes,
        model_metadata=ModelParallelMetadata(is_moe=True),
        cpu_moe_dp_supported=False,
    )

    assert _layout_sizes(candidates) == [(tp_size, 1) for tp_size in expected_tp_sizes]


def test_moe_without_cpu_collectives_uses_dp_one_and_keeps_tp_coverage():
    candidates, skipped = build_parallel_layout_plan(
        CONFIG,
        WORKLOAD,
        6,
        model_metadata=ModelParallelMetadata(
            is_moe=True,
            num_experts=GEMMA4_26B_NUM_EXPERTS,
            moe_intermediate_size=GEMMA4_26B_MOE_INTERMEDIATE_SIZE,
        ),
        cpu_moe_dp_supported=False,
    )

    assert _layout_sizes(candidates) == [(4, 1), (2, 1), (1, 1)]
    assert [item["_benchmark_name"] for item in candidates] == [
        "tp4_dp1_numa4of6",
        "tp2_dp1_numa2of6",
        "tp1_dp1_numa1of6",
    ]
    assert _layout_sizes(skipped) == [(2, 3), (1, 6)]
    assert all(item["policy"] == "cpu_moe_dp_collectives" for item in skipped)
    assert all("variable-size collectives" in item["reason"] for item in skipped)


def test_known_moe_width_rejects_invalid_effective_parallel_size():
    candidates, skipped = build_parallel_layout_plan(
        CONFIG,
        WORKLOAD,
        6,
        model_metadata=ModelParallelMetadata(
            is_moe=True, num_experts=16, moe_intermediate_size=64
        ),
        cpu_moe_dp_supported=True,
    )

    assert _layout_sizes(candidates) == [(4, 1)]
    assert _layout_sizes(skipped) == [(2, 3), (1, 6)]
    assert all(item["policy"] == "moe_effective_parallel_size" for item in skipped)
    assert all(item["effective_parallel_size"] == 6 for item in skipped)


def test_expert_parallel_does_not_use_flattened_width_rule():
    config = {**CONFIG, "enable-expert-parallel": True}
    candidates, skipped = build_parallel_layout_plan(
        config,
        WORKLOAD,
        6,
        model_metadata=ModelParallelMetadata(
            is_moe=True, num_experts=16, moe_intermediate_size=64
        ),
        cpu_moe_dp_supported=True,
    )

    assert _layout_sizes(candidates) == [(4, 1), (2, 3), (1, 6)]
    assert skipped == []


def test_writer_records_skips_and_keeps_failure_isolation(tmp_path, monkeypatch):
    import sweep_generation

    monkeypatch.setattr(
        sweep_generation,
        "inspect_model_parallel_metadata",
        lambda _config: ModelParallelMetadata(
            is_moe=True,
            num_experts=GEMMA4_26B_NUM_EXPERTS,
            moe_intermediate_size=GEMMA4_26B_MOE_INTERMEDIATE_SIZE,
        ),
    )
    monkeypatch.setattr(
        sweep_generation, "cpu_supports_moe_dp_collectives", lambda: False
    )

    files = write_parallel_layout_sweep_files(
        str(tmp_path / "sweep"),
        config_path=str(tmp_path / "config.yml"),
        env_path=str(tmp_path / "env.sh"),
        config=CONFIG,
        workload=WORKLOAD,
        numa_node_count=6,
    )

    output_paths = {path.name: path for path in files}
    skips = json.loads(output_paths["parallel_layout_skips.json"].read_text())
    runner = output_paths["run_parallel_layout_sweep.sh"].read_text()
    report = output_paths["report.py"]
    assert len(skips) == 2
    assert "--continue-on-error" in runner
    assert "--resume" in runner
    assert report.read_text().startswith("#!/usr/bin/env python3\n")
    assert report.stat().st_mode & 0o111
