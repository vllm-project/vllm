# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (  # noqa: E501
    ExampleHiddenStatesConnector,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _connector(storage_path: Path) -> ExampleHiddenStatesConnector:
    connector = object.__new__(ExampleHiddenStatesConnector)
    connector._storage_path = str(storage_path)
    return connector


@pytest.mark.parametrize(
    "req_id",
    [
        "x/../../outside",
        "chatcmpl-x/../../outside-deadbeef",
        "cmpl-x/../../outside-0-deadbeef",
    ],
)
def test_default_hidden_states_path_rejects_request_id_escape(
    tmp_path: Path, req_id: str
):
    connector = _connector(tmp_path / "root")

    with pytest.raises(ValueError, match="escapes shared_storage_path"):
        connector._generate_default_hidden_states_path(req_id)


def test_default_hidden_states_path_stays_under_storage_root(tmp_path: Path):
    connector = _connector(tmp_path / "root")

    filename = Path(connector._generate_default_hidden_states_path("chatcmpl-safe"))

    assert filename == tmp_path / "root" / "chatcmpl-safe.safetensors"
