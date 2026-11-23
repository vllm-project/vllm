# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm.model_executor.model_loader.checkpoint_rows import CheckpointRows


def test_checkpoint_rows_read_each_shard_at_its_offset(tmp_path):
    """Rows come from each shard's file offset, including a view that starts
    inside its tensor, as a tensor-parallel rank's slice does."""
    first = torch.randint(0, 256, (20, 24), dtype=torch.uint8)
    second = torch.randint(0, 256, (16, 24), dtype=torch.uint8)
    path = tmp_path / "model.safetensors"
    save_file({"first": first, "second": second}, str(path))
    rows = CheckpointRows("test", 32, torch.uint8)
    with safe_open(path, framework="pt") as checkpoint:
        rows.add(0, checkpoint.get_tensor("first").narrow(0, 4, 16), 16)
        rows.add(1, checkpoint.get_tensor("second"), 16)
    rows.bind()

    ids = torch.tensor([0, 3, 15, 16, 31])
    out = torch.empty(len(ids), 24, dtype=torch.uint8)
    rows.read(ids, out)
    assert torch.equal(out, torch.cat([first[4:], second])[ids])
    with pytest.raises(IndexError, match="out of range"):
        rows.read(torch.tensor([32]), out[:1])
