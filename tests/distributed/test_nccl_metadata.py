# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for NCCL metadata transfer via NCCL broadcast.

Tests that the trainer can send weight metadata (names, dtypes, shapes) via NCCL
and the receiver auto-discovers it, eliminating out-of-band metadata passing.
"""

import multiprocessing as mp
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
    NCCLWeightTransferUpdateInfo,
)
from vllm.utils.network_utils import get_open_port


def _create_test_model(device):
    """Create a small test model with known weights."""
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32, bias=True, device=device),
        torch.nn.Linear(32, 8, bias=False, device=device),
    )
    for param in model.parameters():
        param.data.fill_(1.0)
    return model


def _sender_process(host, port, packed, send_metadata, result_queue):
    """Trainer process that sends weights via NCCL."""
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    group = NCCLWeightTransferEngine.trainer_init({
        "master_address": host,
        "master_port": port,
        "world_size": 2,
    })

    model = _create_test_model(device)
    metadata = None
    if send_metadata:
        params = list(model.named_parameters())
        metadata = {
            "names": [n for n, _ in params],
            "dtype_names": [str(p.dtype).replace("torch.", "") for _, p in params],
            "shapes": [list(p.shape) for _, p in params],
        }
    args = NCCLTrainerSendWeightsArgs(
        group=group,
        packed=packed,
        metadata=metadata,
    )
    NCCLWeightTransferEngine.trainer_send_weights(
        iterator=((n, p) for n, p in model.named_parameters()),
        trainer_args=args,
    )
    torch.cuda.synchronize()
    result_queue.put(True)


def _receiver_process(host, port, packed, receive_metadata, expected_names, result_queue):
    """Inference process that receives weights via NCCL."""
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)

    config = WeightTransferConfig(backend="nccl")
    parallel_config = MagicMock(spec=ParallelConfig)
    parallel_config.rank = 0
    parallel_config.world_size = 1
    parallel_config.data_parallel_rank = 0

    engine = NCCLWeightTransferEngine(config, parallel_config)
    engine.init_transfer_engine(engine.parse_init_info({
        "master_address": host,
        "master_port": port,
        "rank_offset": 1,
        "world_size": 2,
    }))

    received = []

    def load_weights(weights):
        received.extend([(n, t.clone()) for n, t in weights])

    if receive_metadata:
        update_info = NCCLWeightTransferUpdateInfo(
            receive_metadata=True,
            packed=packed,
        )
    else:
        # Backward compat: provide metadata explicitly
        model = _create_test_model(torch.device("meta"))
        update_info = NCCLWeightTransferUpdateInfo(
            names=[n for n, _ in model.named_parameters()],
            dtype_names=[str(p.dtype).replace("torch.", "") for _, p in model.named_parameters()],
            shapes=[list(p.shape) for _, p in model.named_parameters()],
            packed=packed,
            is_checkpoint_format=False,
        )

    engine.receive_weights(update_info, load_weights)
    torch.cuda.synchronize()
    engine.shutdown()

    # Verify
    received_names = [n for n, _ in received]
    all_ones = all(t.allclose(torch.ones_like(t)) for _, t in received)
    names_match = received_names == expected_names

    result_queue.put({
        "received_names": received_names,
        "all_ones": all_ones,
        "names_match": names_match,
        "num_received": len(received),
    })


def _run_transfer_test(packed, send_metadata, receive_metadata):
    """Run a sender/receiver test pair."""
    host = "localhost"
    port = get_open_port()

    model = _create_test_model(torch.device("cpu"))
    expected_names = [n for n, _ in model.named_parameters()]
    del model

    sender_q = mp.Queue()
    receiver_q = mp.Queue()

    sender = mp.Process(target=_sender_process, args=(host, port, packed, send_metadata, sender_q))
    receiver = mp.Process(target=_receiver_process, args=(host, port, packed, receive_metadata, expected_names, receiver_q))

    receiver.start()
    sender.start()

    sender.join(timeout=30)
    receiver.join(timeout=30)

    assert sender.exitcode == 0, f"Sender exited with code {sender.exitcode}"
    assert receiver.exitcode == 0, f"Receiver exited with code {receiver.exitcode}"

    sender_ok = sender_q.get()
    assert sender_ok

    result = receiver_q.get()
    assert result["names_match"], f"Names mismatch: got {result['received_names']}, expected {expected_names}"
    assert result["all_ones"], "Weight values don't match (expected all ones)"
    assert result["num_received"] == len(expected_names), f"Expected {len(expected_names)} params, got {result['num_received']}"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need 2 GPUs")
def test_nccl_metadata_transfer_unpacked():
    """Test metadata transfer via NCCL with unpacked broadcasting."""
    _run_transfer_test(packed=False, send_metadata=True, receive_metadata=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need 2 GPUs")
def test_nccl_metadata_transfer_packed():
    """Test metadata transfer via NCCL with packed broadcasting."""
    _run_transfer_test(packed=True, send_metadata=True, receive_metadata=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need 2 GPUs")
def test_backward_compat_no_metadata():
    """Test that the old flow (metadata via update_info, no NCCL metadata) still works."""
    _run_transfer_test(packed=False, send_metadata=False, receive_metadata=False)


def test_receive_metadata_skips_validation():
    """Test that receive_metadata=True skips field validation (allows empty lists)."""
    info = NCCLWeightTransferUpdateInfo(receive_metadata=True, packed=True)
    assert info.names == []
    assert info.dtype_names == []
    assert info.shapes == []


def test_explicit_metadata_still_validates():
    """Test that explicit metadata (no receive_metadata) still validates."""
    with pytest.raises(ValueError, match="dtype_names"):
        NCCLWeightTransferUpdateInfo(
            names=["a", "b"],
            dtype_names=["float32"],
            shapes=[[1], [2]],
        )
