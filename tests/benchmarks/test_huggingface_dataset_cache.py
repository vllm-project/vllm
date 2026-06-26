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
    revision_calls: list[dict[str, Any]] = []

    class DummyDataset(HuggingFaceDataset):
        def sample(self, **kwargs: Any) -> list[Any]:
            return []

    def fake_load_dataset(**kwargs: Any) -> list[Any]:
        calls.append(kwargs)
        return []

    def fake_resolve_revision(*args: Any, **kwargs: Any) -> None:
        revision_calls.append({"args": args, "kwargs": kwargs})
        return None

    monkeypatch.setattr(
        datasets_module,
        "maybe_resolve_latest_hf_revision",
        fake_resolve_revision,
    )
    monkeypatch.setattr(datasets_module, "load_dataset", fake_load_dataset)

    DummyDataset(
        dataset_path="dummy/dataset",
        dataset_split="train",
        no_stream=True,
        disable_shuffle=True,
    )

    assert calls
    assert revision_calls == [
        {
            "args": ("dummy/dataset", None),
            "kwargs": {"repo_type": "dataset", "ensure_latest": False},
        }
    ]
    assert "revision" not in calls[0]
    assert "download_mode" not in calls[0]


def test_mtbench_uses_latest_revision_without_forcing_redownload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    revision_calls: list[dict[str, Any]] = []

    def fake_load_dataset(**kwargs: Any) -> list[Any]:
        calls.append(kwargs)
        return []

    def fake_resolve_revision(*args: Any, **kwargs: Any) -> str:
        revision_calls.append({"args": args, "kwargs": kwargs})
        return "latest-revision"

    monkeypatch.setattr(
        datasets_module,
        "maybe_resolve_latest_hf_revision",
        fake_resolve_revision,
    )
    monkeypatch.setattr(datasets_module, "load_dataset", fake_load_dataset)

    MTBenchDataset(
        dataset_path="philschmid/mt-bench",
        dataset_split="train",
        no_stream=True,
        disable_shuffle=True,
    )

    assert calls
    assert revision_calls == [
        {
            "args": ("philschmid/mt-bench", None),
            "kwargs": {"repo_type": "dataset", "ensure_latest": True},
        }
    ]
    assert calls[0]["revision"] == "latest-revision"
    assert "download_mode" not in calls[0]
