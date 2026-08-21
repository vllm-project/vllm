# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace

import pytest
import torch

from benchmarks.benchmark_recirculation import (
    DEFAULT_MODEL_REVISION,
    PeakGpuMemory,
    build_recirculation_config,
    parse_args,
)
from benchmarks.validate_recirculation_adapters import FAMILIES

EXPECTED_FAMILIES = {
    "deepseek-v3",
    "gemma3",
    "gemma4",
    "glm4-moe",
    "glm4-moe-lite",
    "gpt-oss",
    "llama",
    "llama4",
    "mimo-v2",
    "minimax-m2",
    "minimax-m3",
    "mistral",
    "mixtral",
    "qwen2",
    "qwen3",
    "qwen3-moe",
    "qwen3-next",
    "qwen3.5",
    "qwen3.5-moe",
    "step3.5",
}


def test_benchmark_accepts_model_specific_revision_and_beta(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_recirculation.py",
            "--mode",
            "recirculation",
            "--output",
            "result.json",
            "--windows-file",
            "windows.json",
            "--model",
            "google/gemma-3-4b-pt",
            "--beta",
            "1.0",
        ],
    )

    args = parse_args()

    assert args.model_revision is None
    assert build_recirculation_config(args)["beta"] == 1.0


def test_benchmark_pins_only_default_gemma_revision(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_recirculation.py",
            "--mode",
            "baseline",
            "--output",
            "result.json",
            "--windows-file",
            "windows.json",
        ],
    )

    assert parse_args().model_revision == DEFAULT_MODEL_REVISION


def test_gpu_memory_monitor_falls_back_to_torch(monkeypatch) -> None:
    properties = SimpleNamespace(total_memory=1000, name="test accelerator")
    device_module = SimpleNamespace(get_device_properties=lambda index: properties)
    monkeypatch.setattr(PeakGpuMemory, "_load_pynvml", staticmethod(lambda: None))
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: "cuda")
    monkeypatch.setattr(torch.accelerator, "memory_allocated", lambda index: 10)
    monkeypatch.setattr(
        torch.accelerator, "reset_peak_memory_stats", lambda index: None
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda index: None)
    monkeypatch.setattr(torch.accelerator, "max_memory_allocated", lambda index: 25)
    monkeypatch.setattr(torch, "get_device_module", lambda device: device_module)

    with PeakGpuMemory(0) as monitor:
        pass

    assert monitor.source == "torch"
    assert monitor.start_bytes == 10
    assert monitor.peak_bytes == 25
    assert monitor.total_bytes == 1000


def test_adapter_validation_covers_every_documented_family() -> None:
    assert FAMILIES.keys() == EXPECTED_FAMILIES


@pytest.mark.parametrize("family", sorted(EXPECTED_FAMILIES))
def test_adapter_validation_uses_tiny_local_config(family: str) -> None:
    architecture, build_config, _ = FAMILIES[family]

    config = build_config()

    assert architecture
    assert config.num_hidden_layers == 4
    assert config.vocab_size == 256
