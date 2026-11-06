# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""How an EPLB communicator lets go of the ranks a reconfiguration removed.

Scaling down leaves the survivors connected to agents that are gone. The
outgoing communicator cannot drop those itself, since it shares its NIXL agent
with the incoming one, so the incoming communicator does it after taking over.
"""

import logging
from types import SimpleNamespace

from vllm.distributed.eplb.eplb_communicator import (
    NixlEplbCommunicator,
    TorchDistNcclEplbCommunicator,
)
from vllm.distributed.eplb.eplb_state import EplbState

_LOGGER_NAME = "vllm.distributed.eplb.eplb_communicator"


class _FakeNixlWrapper:
    """Records the agents it was asked to disconnect from."""

    def __init__(self, failing: set[str] | None = None):
        self.removed: list[str] = []
        self._failing = failing or set()

    def remove_remote_agent(self, agent_name: str) -> None:
        self.removed.append(agent_name)
        if agent_name in self._failing:
            raise RuntimeError(f"cannot disconnect from {agent_name}")


def _communicator(wrapper, agents: dict[int, str]) -> NixlEplbCommunicator:
    """A communicator carrying only the state release_departed_peers reads."""
    communicator = object.__new__(NixlEplbCommunicator)
    communicator._nixl_wrapper = wrapper
    communicator._remote_agents = dict(agents)
    return communicator


def test_departed_peers_are_dropped_and_survivors_kept():
    wrapper = _FakeNixlWrapper()
    previous = _communicator(wrapper, {1: "eplb-1", 2: "eplb-2", 3: "eplb-3"})
    current = _communicator(wrapper, {1: "eplb-1"})

    current.release_departed_peers(previous)

    assert sorted(wrapper.removed) == ["eplb-2", "eplb-3"]


def test_renumbered_survivors_are_matched_by_name():
    """Scaling down renumbers the ranks, so comparing indices would be wrong."""
    wrapper = _FakeNixlWrapper()
    previous = _communicator(wrapper, {1: "eplb-b", 2: "eplb-c", 3: "eplb-d"})
    # The worker that was rank 3 is rank 1 now.
    current = _communicator(wrapper, {1: "eplb-d"})

    current.release_departed_peers(previous)

    assert sorted(wrapper.removed) == ["eplb-b", "eplb-c"]


def test_an_agent_we_do_not_share_is_left_alone():
    previous = _communicator(_FakeNixlWrapper(), {1: "eplb-1", 2: "eplb-2"})
    wrapper = _FakeNixlWrapper()
    current = _communicator(wrapper, {1: "eplb-1"})

    current.release_departed_peers(previous)

    assert wrapper.removed == []


def test_another_backend_is_left_alone():
    wrapper = _FakeNixlWrapper()
    previous = object.__new__(TorchDistNcclEplbCommunicator)
    current = _communicator(wrapper, {1: "eplb-1"})

    current.release_departed_peers(previous)

    assert wrapper.removed == []


def test_taking_over_releases_the_departed():
    """The hook only matters if update_communicator reaches it."""
    wrapper = _FakeNixlWrapper()
    previous = _communicator(wrapper, {1: "eplb-1", 2: "eplb-2"})
    current = _communicator(wrapper, {1: "eplb-1"})
    model_config = SimpleNamespace(compute_hash=lambda: "model")

    state = object.__new__(EplbState)
    state.model_states = {"model": SimpleNamespace(communicator=previous)}
    state.update_communicator(model_config, current)

    assert state.model_states["model"].communicator is current
    assert wrapper.removed == ["eplb-2"]


def test_one_failed_disconnect_does_not_stop_the_others(caplog):
    wrapper = _FakeNixlWrapper(failing={"eplb-2"})
    previous = _communicator(wrapper, {1: "eplb-1", 2: "eplb-2", 3: "eplb-3"})
    current = _communicator(wrapper, {1: "eplb-1"})

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        current.release_departed_peers(previous)

    assert sorted(wrapper.removed) == ["eplb-2", "eplb-3"]
    assert "eplb-2" in caplog.text
