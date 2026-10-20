# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hooks that let a speculator own verification (tree or other non-chain drafts).

Every hook is a no-op by default: a speculator that does not override it gets
the behavior vLLM had before the hook existed.
"""

from types import SimpleNamespace

import pytest

import vllm.v1.worker.gpu.model_runner as model_runner_module
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.ngram.speculator import NgramGPUSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import (
    BaseSpeculator,
)

# --- make_rejection_sampler -------------------------------------------------


class _FakeRejectionSampler:
    def __init__(self, sampler, speculative_config, device, watermark_key=None):
        self.args = (sampler, speculative_config, device, watermark_key)


def _runner(speculator, monkeypatch) -> GPUModelRunner:
    monkeypatch.setattr(model_runner_module, "RejectionSampler", _FakeRejectionSampler)
    monkeypatch.setattr(
        model_runner_module, "speculative_target_watermark_key", lambda config: "key"
    )
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.sampler = object()
    runner.speculator = speculator
    runner.speculative_config = object()
    runner.device = "cpu"
    runner.vllm_config = SimpleNamespace(watermark_config=None)
    return runner


class _Declines:
    def make_rejection_sampler(self, sampler):
        return None


@pytest.mark.parametrize("speculator", [_Declines(), None])
def test_default_rejection_sampler_is_built_unless_the_speculator_supplies_one(
    speculator, monkeypatch
):
    runner = _runner(speculator, monkeypatch)
    built = GPUModelRunner._build_rejection_sampler(runner)
    assert isinstance(built, _FakeRejectionSampler)
    assert built.args == (runner.sampler, runner.speculative_config, "cpu", "key")


def test_speculator_supplied_rejection_sampler_replaces_the_default(monkeypatch):
    custom = object()
    seen = []

    class Supplies:
        def make_rejection_sampler(self, sampler):
            seen.append(sampler)
            return custom

    runner = _runner(Supplies(), monkeypatch)
    assert GPUModelRunner._build_rejection_sampler(runner) is custom
    assert seen == [runner.sampler]


# --- prepare_verify_inputs / post_accept ------------------------------------


class _MinimalSpeculator(BaseSpeculator):
    def init_cudagraph_manager(self, cudagraph_mode):
        pass

    def capture(self):
        pass

    def propose(self, *args, **kwargs):  # pragma: no cover - not called
        raise NotImplementedError


class _VerifySpeculator(_MinimalSpeculator):
    def prepare_verify_inputs(self, input_batch, model_inputs, *, dummy_run=False):
        pass

    def post_accept(self, input_batch, sampler_output, num_sampled, num_rejected):
        pass


def test_only_overridden_hooks_are_called():
    assert _MinimalSpeculator.__new__(_MinimalSpeculator).overridden_hooks() == set()
    assert NgramGPUSpeculator.__new__(NgramGPUSpeculator).overridden_hooks() == set()
    assert _VerifySpeculator.__new__(_VerifySpeculator).overridden_hooks() == {
        "prepare_verify_inputs",
        "post_accept",
    }
    # A runner built without a speculator calls none.
    assert GPUModelRunner.__new__(GPUModelRunner).speculator_hooks == set()
