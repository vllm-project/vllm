# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The runner side of the deferred RecoverSSM (`DeferredRecoverSSMState`) and how
the specialized decode installs it on `KimiK3ModelState` and the KDA metadata
builders: CPU only, the hybrid state's base class mocked."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadataBuilder
from vllm.models.kimi_k3.nvidia.model_state import KimiK3ModelState
from vllm.models.kimi_k3.nvidia.specialized_decode.recoverssm.state import (
    DeferredRecoverSSMState,
    install_model_state,
)
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState


def _kda_builder(use_recoverssm: bool = True) -> KimiK3KDAMetadataBuilder:
    builder = KimiK3KDAMetadataBuilder.__new__(KimiK3KDAMetadataBuilder)
    builder.use_recoverssm = use_recoverssm
    builder.recoverssm_context = None
    builder.device = torch.device("cpu")
    builder.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=6)
    )
    return builder


class _Group:
    def __init__(self, builder) -> None:
        self.builder = builder

    def get_metadata_builder(self, ubatch_id: int = 0):
        assert ubatch_id == 0
        return self.builder


def _groups():
    deferring = [_kda_builder(), _kda_builder()]
    others = [SimpleNamespace(), _kda_builder(use_recoverssm=False)]
    groups = [
        [_Group(deferring[0]), _Group(others[0])],
        [_Group(deferring[1]), _Group(others[1])],
    ]
    return groups, deferring, others


def test_add_request_zeroes_the_pending_count() -> None:
    state = DeferredRecoverSSMState(4, torch.device("cpu"))
    assert isinstance(state, RecoverSSMState)
    assert state.pending.dtype == torch.int32 and state.pending.tolist() == [0] * 4
    state.pending.copy_(torch.tensor([3, 2, 4, 1], dtype=torch.int32))
    state.add_request("r", 2)
    assert state.pending.tolist() == [3, 2, 0, 1]


def test_bind_installs_only_on_recoverssm_kda_builders() -> None:
    state = DeferredRecoverSSMState(4, torch.device("cpu"))
    groups, deferring, others = _groups()
    idx_mapping = torch.tensor([2, 0], dtype=torch.int32)
    state.bind(groups, idx_mapping)
    for builder in deferring:
        assert builder.recoverssm_pending is state.pending
        assert builder.recoverssm_request_slots is idx_mapping
        assert builder.recoverssm_record_rows.tolist() == [-1] * 6
        # Per-instance overrides; the class is unchanged.
        assert "build" in vars(builder)
        assert "_get_recoverssm_context" in vars(builder)
        assert type(builder) is KimiK3KDAMetadataBuilder
    for other in others:
        assert not hasattr(other, "recoverssm_pending")
        assert not hasattr(other, "recoverssm_request_slots")
        assert "build" not in vars(other)

    # Capture / dummy batches own no request: no slots, the same pending buffer,
    # installed once.
    builds = [builder.build for builder in deferring]
    state.bind(groups, None)
    for builder, build in zip(deferring, builds):
        assert builder.recoverssm_pending is state.pending
        assert builder.recoverssm_request_slots is None
        assert builder.build is build


def _fake_base_init(use_recoverssm: bool):
    def init(self, vllm_config, model, encoder_cache, device):
        self.max_num_reqs = 8
        self.device = device
        self.recoverssm = RecoverSSMState(8, device) if use_recoverssm else None

    return init


def _make_state(use_recoverssm: bool, model=None) -> KimiK3ModelState:
    model = model or SimpleNamespace(specialized_decode_config=None)
    with patch.object(
        MambaHybridModelState, "__init__", _fake_base_init(use_recoverssm)
    ):
        return KimiK3ModelState(None, model, None, torch.device("cpu"))


def test_model_state_hands_itself_to_the_specialized_decode() -> None:
    bound: list = []
    decode = SimpleNamespace(bind_model_state=bound.append)
    model = SimpleNamespace(specialized_decode_config=object())
    model.build_specialized_decode = lambda: setattr(
        model, "specialized_decode", decode
    )
    state = _make_state(use_recoverssm=True, model=model)
    assert bound == [state]
    # Without the specialized decode the stock state is left as it is.
    assert type(_make_state(use_recoverssm=True).recoverssm) is RecoverSSMState


def test_install_model_state_only_with_recoverssm() -> None:
    state = _make_state(use_recoverssm=True)
    install_model_state(state)
    assert type(state.recoverssm) is DeferredRecoverSSMState
    assert state.recoverssm.pending.shape == (8,)
    plain = _make_state(use_recoverssm=False)
    install_model_state(plain)
    assert plain.recoverssm is None
    assert "add_request" not in vars(plain) and "prepare_attn" not in vars(plain)
