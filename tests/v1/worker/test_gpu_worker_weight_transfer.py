# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import SparseWeightPatch
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine
from vllm.v1.worker.gpu_worker import Worker


def _make_nccl_engine() -> NCCLWeightTransferEngine:
    parallel_config = MagicMock(spec=ParallelConfig)
    parallel_config.rank = 0
    parallel_config.world_size = 1
    parallel_config.data_parallel_rank = 0
    parallel_config.data_parallel_index = 0
    return NCCLWeightTransferEngine(
        WeightTransferConfig(backend="nccl"),
        parallel_config,
        MagicMock(spec=torch.nn.Module),
    )


def test_update_weights_sparse_dispatches_to_sparse_receive(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    worker = object.__new__(Worker)
    worker.parallel_config = SimpleNamespace(world_size=1)
    worker.weight_transfer_engine = _make_nccl_engine()
    worker._weight_update_active = True
    worker._is_checkpoint_format = False

    applied_patches = []

    def apply_sparse_weight_patches(patches):
        applied_patches.extend(patches)

    worker.model_runner = SimpleNamespace(
        apply_sparse_weight_patches=apply_sparse_weight_patches,
    )

    received_kinds = []

    def receive_sparse_weights(update_info, apply_patches):
        received_kinds.append(update_info.update_kind)
        apply_patches(
            [
                SparseWeightPatch(
                    name="layer.weight",
                    indices=torch.tensor([1], dtype=torch.int32),
                    values=torch.tensor([2.0], dtype=torch.float32),
                )
            ]
        )

    worker.weight_transfer_engine.receive_sparse_weights = receive_sparse_weights

    Worker.update_weights(
        worker,
        {
            "names": ["layer.weight"],
            "dtype_names": ["float32"],
            "shapes": [[4]],
            "num_updates_list": [1],
            "update_kind": "sparse_flat",
        },
    )

    assert received_kinds == ["sparse_flat"]
    assert len(applied_patches) == 1
    assert torch.equal(applied_patches[0].indices, torch.tensor([1], dtype=torch.int32))


def test_update_weights_sparse_rejects_tp_or_pp(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    worker = object.__new__(Worker)
    worker.parallel_config = SimpleNamespace(world_size=2)
    worker.weight_transfer_engine = _make_nccl_engine()
    worker._weight_update_active = True
    worker._is_checkpoint_format = False
    worker.model_runner = SimpleNamespace(apply_sparse_weight_patches=lambda _: None)

    with pytest.raises(NotImplementedError, match="TP=1 and PP=1"):
        Worker.update_weights(
            worker,
            {
                "names": ["layer.weight"],
                "dtype_names": ["float32"],
                "shapes": [[4]],
                "num_updates_list": [1],
                "update_kind": "sparse_flat",
            },
        )


def test_update_weights_sparse_rejects_checkpoint_format(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    worker = object.__new__(Worker)
    worker.parallel_config = SimpleNamespace(world_size=1)
    worker.weight_transfer_engine = _make_nccl_engine()
    worker._weight_update_active = True
    worker._is_checkpoint_format = True
    worker.model_runner = SimpleNamespace(model=MagicMock())

    with pytest.raises(ValueError, match="start_weight_update"):
        Worker.update_weights(
            worker,
            {
                "names": ["layer.weight"],
                "dtype_names": ["float32"],
                "shapes": [[4]],
                "num_updates_list": [1],
                "update_kind": "sparse_flat",
            },
        )
