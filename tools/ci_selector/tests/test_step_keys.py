# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Emission is the one place a bug makes CI run LESS, so every test here pins a
failure resolving to "send nothing", which means everything runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ci_selector.codemap.step_keys import emit
from ci_selector.handwritten import ONLY_STEP_KEYS_ENV


@dataclass
class FakeStep:
    step_id: str
    key: str | None = None
    label: str = "A Step"
    mirror_hw: str | None = None

    @property
    def buildkite_key(self) -> str:
        from ci_selector.codemap.pipeline.step import derive_step_key

        if not self.key:
            return derive_step_key(self.label)
        if not self.mirror_hw:
            return self.key
        return f"{self.mirror_hw}-{self.key.removesuffix(f'-{self.mirror_hw}')}"


@dataclass
class FakeConfig:
    name: str = "vllm_ci"


@dataclass
class FakePipeline:
    config: FakeConfig
    steps: list


@dataclass
class FakeState:
    pipelines: list


@dataclass
class FakeSelection:
    selected: dict = field(default_factory=dict)
    run_all: dict = field(default_factory=dict)


def state_of(*steps):
    return FakeState([FakePipeline(FakeConfig(), list(steps))])


def test_normal_selection_emits_sorted_keys():
    state = state_of(FakeStep("vllm_ci:b", "b-key"), FakeStep("vllm_ci:a", "a-key"))
    got = emit(state, FakeSelection({"vllm_ci:a": [], "vllm_ci:b": []}))
    assert not got.omit
    assert got.keys == ["a-key", "b-key"]
    assert json.loads(got.as_env()[ONLY_STEP_KEYS_ENV]) == ["a-key", "b-key"]


def test_run_all_omits_the_variable():
    state = state_of(FakeStep("vllm_ci:a", "a-key"))
    got = emit(
        state, FakeSelection({"vllm_ci:a": []}, {"vllm_ci": "a .buildkite change"})
    )
    assert got.omit
    assert "run-all" in got.reason
    assert got.as_env() == {}


def test_empty_selection_omits_rather_than_emitting_an_empty_list():
    # The whole point: an empty list would read as "deliberately run nothing".
    state = state_of(FakeStep("vllm_ci:a", "a-key"))
    got = emit(state, FakeSelection({}))
    assert got.omit
    assert got.as_env() == {}
    assert ONLY_STEP_KEYS_ENV not in got.as_env()


def test_a_kept_step_we_cannot_name_forces_omission():
    state = state_of(FakeStep("vllm_ci:a", "a-key"))
    got = emit(state, FakeSelection({"vllm_ci:a": [], "vllm_ci:ghost": []}))
    assert got.omit
    assert got.unnameable == ["vllm_ci:ghost"]
    assert got.as_env() == {}


def test_a_step_from_another_pipeline_is_not_ours_to_name():
    state = state_of(FakeStep("vllm_ci:a", "a-key"))
    got = emit(state, FakeSelection({"vllm_ci:a": [], "vllm_rocm_ci:x": []}))
    assert not got.omit
    assert got.keys == ["a-key"]


def test_mirror_keeps_the_generators_spelling():
    # We mint `<key>-amd`; the generator publishes `amd-<key>`. Emitting our
    # spelling would name a step that does not exist, and it would not run.
    state = state_of(FakeStep("vllm_ci:lora:amd", "lora-amd", mirror_hw="amd"))
    got = emit(state, FakeSelection({"vllm_ci:lora:amd": []}))
    assert got.keys == ["amd-lora"]


def test_a_keyless_step_uses_the_derived_key():
    state = state_of(
        FakeStep(
            "vllm_ci:Rust Frontend Serve/Admin Coverage",
            None,
            "Rust Frontend Serve/Admin Coverage",
        )
    )
    got = emit(state, FakeSelection({"vllm_ci:Rust Frontend Serve/Admin Coverage": []}))
    assert got.keys == ["rust-frontend-serve-admin-coverage"]


def test_omission_is_absence_not_emptiness():
    state = state_of(FakeStep("vllm_ci:a", "a-key"))
    for sel in (FakeSelection({}), FakeSelection({"vllm_ci:a": []}, {"vllm_ci": "x"})):
        env = emit(state, sel).as_env()
        assert env == {}, "must not set the variable at all"
