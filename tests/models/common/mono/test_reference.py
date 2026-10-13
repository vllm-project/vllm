# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The conformance kit against the reference spec: what the kit asserts is the
library's behaviour, so this is also the library's own test."""

import pytest
import torch

from tests.models.common.mono import reference
from tests.models.common.mono.kit import MonoConformance, forward, forward_layer
from vllm.config.compilation import CUDAGraphMode
from vllm.models.common.mono.guards import _enabled, _select_spec
from vllm.models.common.mono.layer import attach
from vllm.models.common.mono.spec import MonoUnsupported

SPECS_PATH = "tests.models.common.mono.reference:MONO_SPECS"


class TestReferenceSpec(MonoConformance):
    specs_path = SPECS_PATH

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


def test_switch_off_means_no_attach():
    config = reference.make_config()
    config.kernel_config.monokernel = False
    assert not _enabled(config)
    assert attach(reference.make_model(config), config, SPECS_PATH) is None


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


def test_skipped_layers_keep_vllm_path():
    config = reference.make_config()
    model = reference.make_model(config)
    mono = attach(model, config, SPECS_PATH)
    assert {i for i, _ in mono.kernels} == {1, 2, 3}


def test_kernel_layers_follow_the_pipeline_slice():
    config = reference.make_config()
    model = reference.make_model(config)
    model.start_layer, model.end_layer = 2, 4
    mono = attach(model, config, SPECS_PATH)
    assert {i for i, _ in mono.kernels} == {2, 3}


def test_kernels_run_the_layers_they_replace():
    config = reference.make_config()
    model = reference.make_model(config)
    mono = attach(model, config, SPECS_PATH)
    width = min(mono.widths)
    positions = torch.zeros(width, dtype=torch.long)
    hidden = torch.zeros(width, model.config.hidden_size)
    with forward(config, CUDAGraphMode.NONE, {"prefill": False}):
        for i, layer in enumerate(model.layers):
            hidden = forward_layer(mono, i, layer, positions, hidden)
    assert torch.equal(hidden, torch.full_like(hidden, 0 + 1 + 2 + 3))
    assert [k.launched for k in mono.kernels.values() if k.launched] == [[width]] * 3


def test_model_refusal_is_raised():
    config = reference.make_config()
    with pytest.raises(MonoUnsupported, match="not a ReferenceModel"):
        attach(torch.nn.Linear(2, 2), config, SPECS_PATH)
