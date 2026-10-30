# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the auxiliary-readout model metadata and request validation."""

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm import ModelRegistry, SamplingParams
from vllm.config import VllmConfig
from vllm.exceptions import VLLMValidationError
from vllm.model_executor.models.interfaces import supports_aux_readout
from vllm.model_executor.models.opt import OPTForCausalLM
from vllm.model_executor.models.registry import _ModelInfo
from vllm.sampling_params import RequestOutputKind
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

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


def test_validate_rejects_delta_output():
    # Readouts arrive with the final output; a DELTA final output carries only
    # its last tokens, so the row contract would not hold.
    sp = SamplingParams(aux_readouts=["decision"], output_kind=RequestOutputKind.DELTA)
    with pytest.raises(VLLMValidationError, match="DELTA"):
        sp._validate_aux_readouts(_model_config())
    for kind in (RequestOutputKind.CUMULATIVE, RequestOutputKind.FINAL_ONLY):
        SamplingParams(
            aux_readouts=["decision"], output_kind=kind
        )._validate_aux_readouts(_model_config())


def _vllm_config(**parallel):
    sizes = dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
    )
    model_config = SimpleNamespace(
        enable_aux_readouts=True,
        runner_type="generate",
        is_diffusion=False,
        aux_readout_dims={"decision": 3},
        architecture="AuxReadoutTest",
    )
    return SimpleNamespace(
        model_config=model_config,
        use_v2_model_runner=True,
        speculative_config=None,
        parallel_config=SimpleNamespace(**(sizes | parallel)),
    )


@pytest.mark.parametrize(
    ("parallel", "match"),
    [
        ({"tensor_parallel_size": 2}, "tensor parallelism"),
        ({"pipeline_parallel_size": 2}, "pipeline parallelism"),
        ({"decode_context_parallel_size": 2}, "context parallelism"),
    ],
    ids=["tp", "pp", "dcp"],
)
def test_engine_refuses_unvalidated_parallelism(parallel, match):
    VllmConfig._verify_aux_readout_config(_vllm_config())
    with pytest.raises(ValueError, match=match):
        VllmConfig._verify_aux_readout_config(_vllm_config(**parallel))


class _Head:
    def __init__(self, width):
        self.width, self.calls = width, []

    def compute_aux_readouts(self, hidden_states):
        self.calls.append(hidden_states.shape[0])
        return hidden_states[:, : self.width]


def _compute(head, asking, cu, logits_indices, width=2):
    runner = SimpleNamespace(
        model=head, aux_readout_req_ids=set(asking), aux_readout_width=width
    )
    batch = SimpleNamespace(
        req_ids=["a", "b", "c"],
        cu_num_logits_np=np.array(cu),
        logits_indices=torch.tensor(logits_indices),
    )
    hidden = torch.arange(12.0).reshape(6, 2)
    return GPUModelRunner._compute_aux_readouts(runner, hidden, batch), hidden


def test_runner_readouts_follow_logits_rows():
    head = _Head(2)
    out, hidden = _compute(head, {"a", "c"}, [0, 1, 2, 4], [1, 3, 4, 5])
    assert set(out) == {"a", "c"} and head.calls == [3]
    torch.testing.assert_close(out["a"], hidden[[1]])
    torch.testing.assert_close(out["c"], hidden[[4, 5]])


def test_runner_skips_requests_without_logits_rows():
    head = _Head(2)
    out, _ = _compute(head, {"a"}, [0, 0, 1, 2], [3, 5])
    assert out is None and head.calls == []
    out, hidden = _compute(head, {"a", "c"}, [0, 0, 1, 2], [3, 5])
    assert set(out) == {"c"} and head.calls == [1]
    torch.testing.assert_close(out["c"], hidden[[5]])


def test_runner_rejects_wrong_readout_shape():
    with pytest.raises(ValueError, match=r"expected \(1, 2\)"):
        _compute(_Head(1), {"a"}, [0, 1, 2, 3], [1, 3, 5])
