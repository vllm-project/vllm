# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch

from vllm.distributed.device_communicators import custom_all_reduce


@pytest.mark.parametrize(
    ("peer_access", "expected", "checked_peers"),
    [
        ({0: True, 2: True, 3: True}, True, [0, 2, 3]),
        ({0: True, 2: False, 3: True}, False, [0, 2]),
        ({0: False, 2: True, 3: True}, False, [0]),
    ],
)
def test_can_p2p_skip_check_checks_each_peer(
    monkeypatch: pytest.MonkeyPatch,
    peer_access: dict[int, bool],
    expected: bool,
    checked_peers: list[int],
):
    can_device_access_peer = Mock(side_effect=lambda _src, dst: peer_access[dst - 10])
    monkeypatch.setattr(
        custom_all_reduce,
        "envs",
        SimpleNamespace(VLLM_SKIP_P2P_CHECK=True),
    )
    monkeypatch.setattr(
        custom_all_reduce,
        "current_platform",
        SimpleNamespace(
            logical_device_id_to_visible_device_id=lambda device: device + 10
        ),
    )
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", can_device_access_peer)

    assert custom_all_reduce._can_p2p(rank=1, world_size=4) is expected
    assert can_device_access_peer.call_args_list == [
        call(11, peer + 10) for peer in checked_peers
    ]
