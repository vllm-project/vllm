# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from vllm.v1.worker.pp.boundary import DynamicPPConfig, PPBoundaryController
from vllm.v1.worker.pp.buffer import PPSendBuffer


def _gloo_worker(rank, rendezvous, first_two_posted):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        controller = PPBoundaryController(DynamicPPConfig(redundant_layers=2), 6, 12)
        torch.manual_seed(7)
        layers = [(torch.randn(8, 8) / 4, torch.randn(8)) for _ in range(12)]

        def execute(value, start, end):
            for weight, bias in layers[start:end]:
                value = torch.tanh(value @ weight + bias)
            return value

        if rank == 0:
            buffer = PPSendBuffer(2)
            output = torch.empty(7, 4, 8)
            buffer.reserve({"hidden_states": output})
            for batch in range(12):
                rows = 3 if batch % 2 else 7
                boundary = 4 + batch % 5
                buffer.acquire()
                output[:rows].copy_(
                    execute(torch.full_like(output[:rows], batch), 0, boundary)
                )

                def send(payload):
                    header = torch.tensor(payload["header"], dtype=torch.int64)
                    # Gloo Work retains the header allocation until completion.
                    return [
                        dist.isend(header, 1),
                        dist.isend(payload["hidden_states"], 1),
                    ]

                buffer.send(
                    {
                        "hidden_states": output[:rows],
                        "header": (batch, boundary, batch, rows),
                    },
                    send,
                )
                output.fill_(-100)
                if batch == 1:
                    first_two_posted.set()
            buffer.drain()
            assert buffer.num_sends == 12
        else:
            assert first_two_posted.wait(30), (
                "producer waited for receive before filling two slots"
            )
            for batch in range(12):
                wire = torch.empty(4, dtype=torch.int64)
                dist.recv(wire, 0)
                batch_id, cut, version, rows = wire.tolist()
                header = controller.accept((batch_id, cut, version), batch)
                value = torch.empty(rows, 4, 8)
                dist.recv(value, 0)
                actual = execute(value, *controller.execution_range(1, header))
                expected = execute(torch.full_like(value, batch), 0, 12)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_delayed_receiver_and_changing_boundaries_preserve_each_layer(tmp_path):
    context = torch.multiprocessing.get_context("spawn")
    torch.multiprocessing.spawn(
        _gloo_worker,
        args=(f"file://{tmp_path / 'rendezvous'}", context.Event()),
        nprocs=2,
        join=True,
    )


def test_reserved_storage_rejects_an_unbudgeted_shape():
    buffer = PPSendBuffer(2)
    buffer.reserve({"hidden_states": torch.empty(7, 4, 8)})
    buffer.acquire()
    with pytest.raises(ValueError, match="exceeds its reserved"):
        buffer.send({"hidden_states": torch.empty(8, 4, 8)}, lambda tensors: ())
    buffer.cancel()
    assert buffer.reserved_bytes == 2 * 7 * 4 * 8 * 4


def test_snapshot_separates_output_and_metadata_from_the_next_forward():
    buffer = PPSendBuffer(1)
    output = torch.ones(2, 4, 8)
    header = [0, 4, 0]
    received = []
    buffer.acquire()

    def send(tensors):
        received.append(tensors)
        return ()

    buffer.send({"hidden_states": output, "header": header}, send)
    output.zero_()
    header[1] = 5
    assert received[0]["header"] == [0, 4, 0]
    torch.testing.assert_close(received[0]["hidden_states"], torch.ones_like(output))
    buffer.drain()
