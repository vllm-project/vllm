# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for offline EPLB support — pure-Python checks for the two
public surfaces added by the offline-EPLB feature:

  1. `EPLBConfig.save_path` / `load_path` are mutually exclusive.
  2. `EplbState._save_logical_load` writes a safetensors file that
     round-trips, creates parent directories on demand, and tags the file
     with `version: "1"` metadata.

No GPU, no distributed setup — these run in the regular pytest suite."""

from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from vllm.config.parallel import EPLBConfig
from vllm.distributed.eplb.eplb_state import EplbState


def test_eplb_config_save_load_paths_are_mutually_exclusive(tmp_path):
    """Only one of `save_path` / `load_path` may be set at a time:
    a run either records stats or replays them, never both."""
    save_only = tmp_path / "save.safetensors"
    load_only = tmp_path / "load.safetensors"

    # Each side alone is fine.
    EPLBConfig(save_path=save_only)
    EPLBConfig(load_path=load_only)
    # Neither side set is also fine (back-compat with existing configs).
    EPLBConfig()

    # Both at once is rejected by the model validator.
    with pytest.raises(ValueError, match="save_path and load_path"):
        EPLBConfig(save_path=save_only, load_path=load_only)


def test_save_logical_load_roundtrip_creates_parents_and_tags_version(tmp_path):
    """`_save_logical_load` must:
      - create missing parent directories,
      - write a safetensors file containing every input tensor unchanged,
      - tag the file with `version: "1"` metadata so a future reader can
        gate on format compatibility."""
    tensors: dict[str, torch.Tensor] = {
        "model_a": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "model_b": torch.full((2, 5), 7.0, dtype=torch.float32),
    }

    # Parent directory deliberately does not exist — the helper is supposed
    # to create it.
    out: Path = tmp_path / "nested" / "subdir" / "load.safetensors"
    assert not out.parent.exists()

    EplbState._save_logical_load(tensors, out)

    assert out.exists()
    assert out.parent.is_dir()

    loaded = load_file(str(out))
    assert set(loaded.keys()) == set(tensors.keys())
    for name, expected in tensors.items():
        assert torch.equal(loaded[name], expected)
        assert loaded[name].dtype == expected.dtype

    # Metadata tag is part of the on-disk contract — readers may key on it
    # for format-version checks in future refactors.
    with safe_open(str(out), framework="pt") as f:
        assert f.metadata() == {"version": "1"}
