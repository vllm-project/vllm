# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock

from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)


def test_offloading_connector_supports_hma() -> None:
    assert issubclass(OffloadingConnector, SupportsHMA)


def test_request_finished_all_groups_delegates_to_scheduler() -> None:
    connector = object.__new__(OffloadingConnector)
    connector.connector_scheduler = MagicMock()
    request = MagicMock()
    block_ids = ([1, 2], [3, 4])

    connector.connector_scheduler.request_finished.return_value = (False, None)

    assert connector.request_finished_all_groups(request, block_ids) == (False, None)
    connector.connector_scheduler.request_finished.assert_called_once_with(
        request, block_ids
    )
