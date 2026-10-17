# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the auxiliary-readout model metadata and request validation."""

import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch

from vllm import ModelRegistry, SamplingParams
from vllm.exceptions import VLLMValidationError
from vllm.model_executor.models.interfaces import supports_aux_readout
from vllm.model_executor.models.opt import OPTForCausalLM
from vllm.model_executor.models.registry import _ModelInfo

pytestmark = pytest.mark.skip_global_cleanup


class _ReadoutA(OPTForCausalLM):
    supports_aux_readout = True
    aux_readout_dims = {"decision": 3, "score": 1}

    def compute_aux_readouts(self, hidden_states):
        return torch.zeros(len(hidden_states), 4)


class _ReadoutB(OPTForCausalLM):
    supports_aux_readout = True
    aux_readout_dims = {"gate": 2}

    def compute_aux_readouts(self, hidden_states):
        return torch.zeros(len(hidden_states), 2)


def test_model_info_readout_dims():
    info = _ModelInfo.from_model_cls(_ReadoutA)
    assert info.aux_readout_dims == (("decision", 3), ("score", 1))
    # The JSON model-info cache returns tuples as lists; the field must stay
    # hashable after the round trip.
    loaded = _ModelInfo(**json.loads(json.dumps(dataclasses.asdict(info))))
    assert loaded.aux_readout_dims == info.aux_readout_dims
    hash(loaded.aux_readout_dims)
    assert _ModelInfo.from_model_cls(OPTForCausalLM).aux_readout_dims == ()
    assert supports_aux_readout(_ReadoutA)
    assert not supports_aux_readout(OPTForCausalLM)


def test_registry_inspects_in_process_classes():
    """In-process registration hashes _ModelInfo (lru_cache)."""
    ModelRegistry.register_model("AuxReadoutTestA", _ReadoutA)
    ModelRegistry.register_model("AuxReadoutTestA", _ReadoutA)
    ModelRegistry.register_model("AuxReadoutTestB", _ReadoutB)
    a = ModelRegistry._try_inspect_model_cls("AuxReadoutTestA")
    b = ModelRegistry._try_inspect_model_cls("AuxReadoutTestB")
    assert a is not None and b is not None
    assert dict(a.aux_readout_dims) == {"decision": 3, "score": 1}
    assert dict(b.aux_readout_dims) == {"gate": 2}


def _model_config(enabled=True):
    return SimpleNamespace(
        enable_aux_readouts=enabled, aux_readout_dims={"decision": 3, "score": 1}
    )


def test_validate_column_slices_follow_declared_layout():
    sp = SamplingParams(aux_readouts=["score", "decision"])
    sp._validate_aux_readouts(_model_config())
    assert sp._aux_readout_slices == [("score", 3, 4), ("decision", 0, 3)]
    plain = SamplingParams()
    plain._validate_aux_readouts(_model_config(enabled=False))
    assert plain._aux_readout_slices is None


@pytest.mark.parametrize(
    ("names", "enabled", "match"),
    [
        ([], True, "non-empty"),
        (["nope"], True, "non-empty"),
        (["decision", "decision"], True, "distinct"),
        (["decision"], False, "enable_aux_readouts"),
    ],
    ids=["empty", "unknown", "dup", "engine-off"],
)
def test_validate_rejects(names, enabled, match):
    with pytest.raises(VLLMValidationError, match=match):
        SamplingParams(aux_readouts=names)._validate_aux_readouts(
            _model_config(enabled)
        )
