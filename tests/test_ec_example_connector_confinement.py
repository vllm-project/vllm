# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path

import pytest
import torch

from vllm.distributed.ec_transfer.ec_connector.example_connector import (
    ECExampleConnector,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _connector(storage_path: Path) -> ECExampleConnector:
    connector = object.__new__(ECExampleConnector)
    connector._storage_path = str(storage_path)
    return connector


@pytest.mark.parametrize("mm_hash", ["../escaped", "/tmp/escaped"])
def test_mm_hash_cannot_escape_shared_storage_path(tmp_path: Path, mm_hash: str):
    storage_path = tmp_path / "root"
    connector = _connector(storage_path)

    with pytest.raises(ValueError, match="escapes shared_storage_path"):
        connector._generate_filename_debug(mm_hash)

    assert not (tmp_path / "escaped").exists()


def test_unsafe_mm_hash_is_cache_miss_without_creating_outside_path(tmp_path: Path):
    storage_path = tmp_path / "root"
    connector = _connector(storage_path)

    assert connector.has_cache_item("../escaped") is False
    assert not (tmp_path / "escaped").exists()


def test_unsafe_mm_hash_save_is_skipped_without_creating_outside_path(tmp_path: Path):
    storage_path = tmp_path / "root"
    connector = _connector(storage_path)
    connector._is_producer = True

    connector.save_caches({"../escaped": torch.empty(0)}, "../escaped")

    assert not (tmp_path / "escaped").exists()


def test_safe_mm_hash_stays_inside_shared_storage_path(tmp_path: Path):
    storage_path = tmp_path / "root"
    connector = _connector(storage_path)

    filename = Path(connector._generate_filename_debug("safe/hash"))

    assert filename == storage_path / "safe" / "hash" / "encoder_cache.safetensors"
    assert filename.parent.is_dir()
