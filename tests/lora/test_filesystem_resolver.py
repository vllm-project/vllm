# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from pathlib import Path

import pytest

from vllm.plugins.lora_resolvers.filesystem_resolver import FilesystemResolver

BASE_MODEL_NAME = "base-model"


def _write_adapter(path: Path, base_model_name: str = BASE_MODEL_NAME) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": base_model_name})
    )


@pytest.mark.asyncio
async def test_filesystem_resolver_loads_adapter_inside_cache(tmp_path):
    cache_dir = tmp_path / "lora-cache"
    adapter = cache_dir / "safe-adapter"
    _write_adapter(adapter)

    resolver = FilesystemResolver(str(cache_dir))

    request = await resolver.resolve_lora(BASE_MODEL_NAME, "safe-adapter")

    assert request is not None
    assert request.lora_name == "safe-adapter"
    assert os.path.realpath(request.lora_path) == os.path.realpath(adapter)


@pytest.mark.asyncio
async def test_filesystem_resolver_rejects_absolute_path_escape(tmp_path):
    cache_dir = tmp_path / "lora-cache"
    outside_adapter = tmp_path / "outside-adapter"
    cache_dir.mkdir()
    _write_adapter(outside_adapter)

    resolver = FilesystemResolver(str(cache_dir))

    request = await resolver.resolve_lora(BASE_MODEL_NAME, str(outside_adapter))

    assert request is None


@pytest.mark.asyncio
async def test_filesystem_resolver_rejects_parent_directory_escape(tmp_path):
    cache_dir = tmp_path / "lora-cache"
    outside_adapter = tmp_path / "outside-adapter"
    cache_dir.mkdir()
    _write_adapter(outside_adapter)

    resolver = FilesystemResolver(str(cache_dir))
    escape_name = os.path.relpath(outside_adapter, cache_dir)

    request = await resolver.resolve_lora(BASE_MODEL_NAME, escape_name)

    assert escape_name.startswith("..")
    assert request is None


@pytest.mark.asyncio
async def test_filesystem_resolver_rejects_symlink_escape(tmp_path):
    cache_dir = tmp_path / "lora-cache"
    outside_adapter = tmp_path / "outside-adapter"
    cache_dir.mkdir()
    _write_adapter(outside_adapter)
    (cache_dir / "linked-adapter").symlink_to(outside_adapter, target_is_directory=True)

    resolver = FilesystemResolver(str(cache_dir))

    request = await resolver.resolve_lora(BASE_MODEL_NAME, "linked-adapter")

    assert request is None
