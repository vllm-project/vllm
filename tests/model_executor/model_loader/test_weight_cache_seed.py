# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for daemon-to-daemon weight-cache metadata and copying."""

from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.model_loader.weight_cache.protocol import TensorEntry
from vllm.model_executor.model_loader.weight_cache.seed import (
    PeerIpcSeedSource,
    build_manifest,
    manifest_dtype,
    manifest_nbytes,
)


def _entries():
    return {
        "weight": TensorEntry.from_tensor(
            torch.zeros(2, 3, dtype=torch.float16), "param"
        ),
        "scale": TensorEntry.from_tensor(torch.ones(4, dtype=torch.float32), "buffer"),
    }


def test_manifest_describes_all_exported_tensors():
    manifest = build_manifest(_entries())
    assert manifest == {
        "weight": {"shape": [2, 3], "dtype": "float16", "is_param": True},
        "scale": {"shape": [4], "dtype": "float32", "is_param": False},
    }
    assert manifest_nbytes(manifest) == 2 * 3 * 2 + 4 * 4
    assert manifest_dtype("bfloat16") is torch.bfloat16


def test_unknown_manifest_dtype_fails_loudly():
    with pytest.raises(RuntimeError, match="unsupported dtype"):
        manifest_dtype("not_a_torch_dtype")


def test_peer_seed_copy_owns_new_tensors():
    entries = _entries()
    source = PeerIpcSeedSource()
    manifest = build_manifest(entries)
    with patch.object(torch.accelerator, "synchronize"):
        result = source.fill(
            manifest,
            {"source_device_index": 0, "entries": entries},
            torch.device("cpu"),
        )
    assert torch.equal(result["weight"], torch.zeros(2, 3, dtype=torch.float16))
    assert torch.equal(result["scale"], torch.ones(4, dtype=torch.float32))
    assert result["weight"].data_ptr() != entries["weight"].cpu_tensor.data_ptr()
