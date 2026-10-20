# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for Uno's draft LoRA mapping through the public LoRA API."""

import pytest

from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.v1.worker.gpu.spec_decode.uno import UNO_LORA_ID, UnoSpeculator
from vllm.v1.worker.gpu.spec_decode.uno_lora import (
    base_prompt_mapping,
    draft_lora_mapping,
)


@pytest.mark.parametrize("k", [1, 4, 8])
def test_draft_mapping_has_base_seed_and_padded_base_rows(k):
    mapping = draft_lora_mapping(3 * k + 2, 3, k, 1_000_003)
    assert mapping[:k] == (0,) + (1_000_003,) * (k - 1)
    assert mapping[k : 2 * k] == mapping[:k]
    assert mapping[3 * k :] == (0, 0)


def test_draft_mapping_rejects_truncation_and_zero_dimensions():
    with pytest.raises(ValueError):
        draft_lora_mapping(7, 2, 4, 1_000_003)
    with pytest.raises(ValueError):
        draft_lora_mapping(4, 0, 4, 1_000_003)


class _RecordingWorkerLoRAManager:
    """Records the public calls a WorkerLoRAManager receives."""

    def __init__(self):
        self.calls: list[tuple[set, LoRAMapping | None]] = []
        self.registered: dict[int, str] = {}
        self.added: list = []
        self.removed: list[int] = []

    def set_active_adapters(self, requests, mapping):
        self.calls.append((set(requests), mapping))
        for request in requests:
            self.registered.setdefault(request.lora_int_id, request.lora_path)

    def list_adapters(self):
        return set(self.registered)

    def add_adapter(self, request):
        self.added.append(request)
        self.registered[request.lora_int_id] = request.lora_path
        return True

    def remove_adapter(self, adapter_id):
        self.removed.append(adapter_id)
        return self.registered.pop(adapter_id, None) is not None


def _bound_proposer(k: int = 4):
    proposer = object.__new__(UnoSpeculator)
    proposer.k = k
    proposer.lora_request = LoRARequest("uno", UNO_LORA_ID, "/uno/adapter")
    proposer._lora_hook = None
    manager = _RecordingWorkerLoRAManager()
    proposer.set_lora_manager(manager)
    return proposer, manager


def test_speculator_lora_hook_routes_only_the_draft_pass():
    """Binding loads Uno's adapter; each draft pass installs only its mapping.

    The draft mapping goes through the public ``set_active_adapters`` with
    Uno's own request alone, so no other adapter is added or removed, and
    nothing is installed after the draft: the runner's next target step
    installs its own request mapping, which the manager rebuilds because it
    differs from the draft mapping it last saw.
    """
    proposer, manager = _bound_proposer(k=4)
    assert manager.calls == [({proposer.lora_request}, None)]

    with proposer._draft_lora(2, 8):
        requests, mapping = manager.calls[-1]
        assert requests == {proposer.lora_request}
        assert mapping.index_mapping == draft_lora_mapping(8, 2, 4, UNO_LORA_ID)
        assert mapping.prompt_mapping == base_prompt_mapping(8)
        assert mapping.is_prefill
    assert len(manager.calls) == 2


def test_target_request_reusing_unos_id_gets_its_own_adapter():
    """No request can select Uno's adapter for its target rows.

    A prebuilt request skips the frontend's LoRA refusal and can carry any id.
    The runner's target call reaches the manager through Uno's hook; when it
    asks for the id Uno's adapter is registered under, Uno moves its adapter to
    a fresh id first, so the request's own adapter loads under the id it asked
    for and the runner's call goes through unchanged. Drafts follow Uno's new
    id.
    """
    proposer, manager = _bound_proposer(k=4)
    user = LoRARequest("user", UNO_LORA_ID, "/user/adapter")
    target_mapping = LoRAMapping((UNO_LORA_ID,) * 3, (UNO_LORA_ID,), is_prefill=True)

    manager.set_active_adapters({user}, target_mapping)

    moved_id = proposer.lora_request.lora_int_id
    assert moved_id != UNO_LORA_ID
    assert manager.removed == [UNO_LORA_ID]
    assert [(r.lora_int_id, r.lora_path) for r in manager.added] == [
        (moved_id, "/uno/adapter")
    ]
    assert manager.calls[-1] == ({user}, target_mapping)
    assert manager.registered[UNO_LORA_ID] == "/user/adapter"
    assert manager.registered[moved_id] == "/uno/adapter"

    with proposer._draft_lora(1, 4):
        requests, draft_mapping = manager.calls[-1]
        assert requests == {proposer.lora_request}
        assert draft_mapping.index_mapping == draft_lora_mapping(4, 1, 4, moved_id)


def test_ordinary_requests_and_unos_own_request_leave_unos_id_alone():
    """Only a request reusing Uno's id moves the adapter; nothing else does."""
    proposer, manager = _bound_proposer(k=4)
    ordinary = LoRARequest("user", 7, "/user/adapter")
    mapping = LoRAMapping((7, 0), (7,), is_prefill=True)

    manager.set_active_adapters({ordinary}, mapping)
    manager.set_active_adapters(set(), LoRAMapping((0,), (0,), is_prefill=True))
    manager.set_active_adapters(
        {proposer.lora_request}, LoRAMapping((0,), (0,), is_prefill=True)
    )

    assert proposer.lora_request.lora_int_id == UNO_LORA_ID
    assert manager.added == [] and manager.removed == []
    assert manager.calls[1] == ({ordinary}, mapping)


def test_draft_pass_without_a_bound_manager_refuses():
    proposer = object.__new__(UnoSpeculator)
    proposer._lora_hook = None
    with (
        pytest.raises(RuntimeError, match="adapter routing has not been initialized"),
        proposer._draft_lora(1, 4),
    ):
        pass
