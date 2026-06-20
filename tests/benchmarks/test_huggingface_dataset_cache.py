# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest

import vllm.benchmarks.datasets.datasets as datasets_module
from vllm.benchmarks.datasets import HuggingFaceDataset, MTBenchDataset


def test_hf_dataset_does_not_check_latest_revision_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class DummyDataset(HuggingFaceDataset):
        def sample(self, **kwargs: Any) -> list[Any]:
            return []

    def fail_if_called(self: HuggingFaceDataset) -> str | None:
        raise AssertionError("revision lookup should be opt-in")

    def fake_load_dataset(**kwargs: Any) -> list[Any]:
        calls.append(kwargs)
        return []

    monkeypatch.setattr(DummyDataset, "_get_latest_revision", fail_if_called)
    monkeypatch.setattr(datasets_module, "load_dataset", fake_load_dataset)

    DummyDataset(
        dataset_path="dummy/dataset",
        dataset_split="train",
        no_stream=True,
        disable_shuffle=True,
    )

    assert calls
    assert "revision" not in calls[0]
    assert "download_mode" not in calls[0]


def test_mtbench_uses_latest_revision_without_forcing_redownload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_load_dataset(**kwargs: Any) -> list[Any]:
        calls.append(kwargs)
        return []

    monkeypatch.setattr(
        MTBenchDataset,
        "_get_latest_revision",
        lambda self: "latest-revision",
    )
    monkeypatch.setattr(datasets_module, "load_dataset", fake_load_dataset)

    MTBenchDataset(
        dataset_path="philschmid/mt-bench",
        dataset_split="train",
        no_stream=True,
        disable_shuffle=True,
    )

    assert calls
    assert calls[0]["revision"] == "latest-revision"
    assert "download_mode" not in calls[0]
