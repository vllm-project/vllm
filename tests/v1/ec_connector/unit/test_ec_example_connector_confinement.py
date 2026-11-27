# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path

import pytest
import torch

from vllm.config import VllmConfig
from vllm.config.ec_transfer import ECTransferConfig
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.example_connector import (
    ECExampleConnector,
    ECExampleConnectorMetadata,
    MMMeta,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _connector(storage_path: Path) -> ECExampleConnector:
    config = VllmConfig(
        ec_transfer_config=ECTransferConfig(
            ec_connector="ECExampleConnector",
            ec_role="ec_both",
            ec_connector_extra_config={"shared_storage_path": str(storage_path)},
        ),
    )
    return ECExampleConnector(config, ECConnectorRole.WORKER)


@pytest.mark.parametrize("identifier", ["team/item", "é" * 256, "id-reserved"])
def test_encoder_cache_uses_bounded_directory_names(
    tmp_path: Path, identifier: str
) -> None:
    connector = _connector(tmp_path)

    assert not connector.has_cache_item(identifier)
    assert list(tmp_path.iterdir()) == []
    expected = torch.tensor([[1.0, 2.0]])
    connector.save_caches({identifier: expected}, identifier)

    (directory,) = tmp_path.iterdir()
    assert directory.name.startswith("id-")
    assert len(bytes.fromhex(directory.name.removeprefix("id-"))) == 32
    assert connector.has_cache_item(identifier)
    assert not connector.has_cache_item(directory.name)

    mapped_file = directory / "encoder_cache.v2.safetensors"
    legacy_file = mapped_file.rename(directory / "encoder_cache.safetensors")
    assert not connector.has_cache_item(identifier)
    legacy_file.rename(mapped_file)

    consumer = _connector(tmp_path)
    metadata = ECExampleConnectorMetadata()
    metadata.add_mm_data(MMMeta(identifier, num_token=1))
    consumer.bind_connector_metadata(metadata)
    loaded: dict[str, torch.Tensor] = {}
    consumer.start_load_caches(loaded)
    torch.testing.assert_close(loaded[identifier], expected)


def test_encoder_cache_retains_existing_safe_paths(tmp_path: Path) -> None:
    identifier = "a" * 128
    connector = _connector(tmp_path)

    connector.save_caches({identifier: torch.ones(1)}, identifier)

    assert (tmp_path / identifier / "encoder_cache.safetensors").is_file()
