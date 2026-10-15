# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The conformance kit against the reference spec: the library's own test."""

import pytest
import torch

from tests.models.common.mono import reference
from tests.models.common.mono.kit import MonoConformance, forward
from vllm.config.compilation import CUDAGraphMode
from vllm.models.common.mono.guards import _select_spec, select_mono_spec
from vllm.models.common.mono.runtime import MonoRuntime
from vllm.models.common.mono.spec import MonoUnsupported

SPECS = reference.MONO_SPECS


class TestReferenceSpec(MonoConformance):
    specs = SPECS
    specs_module = "tests.models.common.mono.reference"

    def make_config(self, **overrides):
        return reference.make_config(**overrides)

    def make_model(self, config):
        return reference.make_model(config)

    def decode_metadata(self, config, model):
        return {"prefill": False}

    def skipped_metadata(self, config, model):
        return {"prefill": True}

    def bad_caches(self, config, model):
        return {
            "wrong": reference.make_cache(dtype=torch.float16, ndim=2),
        }


def test_switch_off_means_no_spec():
    config = reference.make_config()
    config.kernel_config.monokernel = False
    assert select_mono_spec(config, SPECS) is None


def test_the_first_servable_variant_wins():
    config = reference.make_config()
    config.scheduler_config.max_num_seqs = 4
    spec, refusals = _select_spec(reference.MONO_SPECS, config)
    assert spec is reference.REFERENCE
    assert refusals["reference-wide"] == ["needs max_num_seqs >= 8"]

    config.scheduler_config.max_num_seqs = 8
    spec, refusals = _select_spec(reference.MONO_SPECS, config)
    assert spec is reference.REFERENCE_WIDE
    assert refusals == {}


def _make_mono(config, model):
    mono = MonoRuntime(model, config, select_mono_spec(config, SPECS))
    mono.bind_caches()
    return mono


def test_skipped_layers_keep_vllm_path():
    config = reference.make_config()
    model = reference.make_model(config)
    mono = _make_mono(config, model)
    assert set(mono.kernel.weights) == {1, 2, 3}


def test_kernel_layers_follow_the_pipeline_slice():
    config = reference.make_config()
    model = reference.make_model(config)
    model.start_layer, model.end_layer = 2, 4
    mono = _make_mono(config, model)
    assert set(mono.kernel.weights) == {2, 3}


def test_the_kernel_runs_the_layers_it_replaces():
    config = reference.make_config()
    model = reference.make_model(config)
    mono = _make_mono(config, model)
    width = min(mono.widths)
    positions = torch.zeros(width, dtype=torch.long)
    embeds = torch.zeros(width, model.config.hidden_size)
    with forward(config, CUDAGraphMode.NONE, {"prefill": False}):
        step = mono.plan(positions)
        assert step is not None
        hidden = mono.run(step, None, embeds)
    assert torch.equal(hidden, model(None, positions, None, embeds))
    assert mono.kernel.launched == [width]


def test_model_refusal_is_raised():
    config = reference.make_config()
    with pytest.raises(MonoUnsupported, match="not a ReferenceModel"):
        MonoRuntime(torch.nn.Linear(2, 2), config, select_mono_spec(config, SPECS))
