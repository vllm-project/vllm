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


@pytest.mark.parametrize("req_id", ["chatcmpl-safe", "team/job", "é" * 256])
def test_default_hidden_states_path_uses_a_bounded_basename(
    tmp_path: Path, req_id: str
):
    connector = _connector(tmp_path)

    filename = Path(connector._generate_default_hidden_states_path(req_id))

    assert filename.parent == tmp_path.resolve()
    assert filename.suffix == ".safetensors"
    assert len(bytes.fromhex(filename.stem)) == 32
    assert filename == Path(connector._generate_default_hidden_states_path(req_id))
    assert filename != Path(connector._generate_default_hidden_states_path("other"))
