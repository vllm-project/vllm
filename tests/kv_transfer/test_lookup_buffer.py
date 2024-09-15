import os
import random

import torch
from tqdm import tqdm

import vllm.distributed.kv_transfer.kv_lookup_buffer.simple_buffer as sklb
import vllm.distributed.kv_transfer.kv_pipe.torch_distributed_pipe as tdp

# TODO: the test depends on a lot of fields in the current implementation.
# We should have standard interface instead direct field access


def test_run(my_rank, buffer, device):

    # buffer should be empty in the beginning
    if my_rank == 0:
        assert buffer.buffer_size == 0
        assert len(buffer.buffer) == 0

    # insert
    tokens = torch.tensor([1, 2, 3]).to(device)
    roi = (tokens > 0)
    if my_rank == 0:
        key = 2.0 * torch.ones([5, 6]).to(device)
        value = 3.0 * torch.ones([5, 6]).to(device)

        placeholder = torch.tensor([1]).to(device)

        buffer.insert(tokens, roi, key, value, placeholder)

    torch.distributed.barrier()

    # drop_select
    if my_rank == 1:
        tok, roi_, key, value, hidden = buffer.drop_select(tokens, roi)
        assert torch.allclose(tokens, tok)
        assert torch.allclose(roi, roi_)
        assert torch.allclose(key, 2.0 * torch.ones([5, 6], device=device))
        assert torch.allclose(value, 3.0 * torch.ones([5, 6], device=device))
    torch.distributed.barrier()

    if my_rank == 0:
        assert buffer.buffer_size == 0
        assert len(buffer.buffer) == 0

    print("Test run passed!")


def stress_test(my_rank, buf, device):

    torch.distributed.barrier()
    torch.manual_seed(100)

    reqs = [
        (
            torch.rand(100).to(device),  # tokens
            torch.ones(100).bool().to(device),  # roi
            torch.rand(100).to(device),  # key
            torch.rand(100).to(device),  # value
            torch.rand(100).to(device),  # hidden
        ) for i in tqdm(range(200))
    ]

    random.seed(my_rank)
    random.shuffle(reqs)

    torch.distributed.barrier()

    n = 0

    # the buffer size can only store 100 reqs
    # so the sender will occasionally block to wait for the receiver.
    for req in tqdm(reqs):
        if my_rank == 0:
            buf.insert(*req)
        else:
            tok, roi, k, v, h = req
            tok_, roi_, k_, v_, h_ = buf.drop_select(tok, roi)

            if tok_ is None:
                assert roi_ is None
                assert k_ is None
                assert v_ is None
                assert h_ is None
                n += 1
            else:
                assert torch.allclose(tok, tok_)
                assert torch.allclose(roi, roi_)
                assert torch.allclose(k, k_)
                assert torch.allclose(v, v_)
                assert torch.allclose(h, h_)
    print('Rank %d done' % my_rank)
    torch.distributed.barrier()

    if my_rank == 0:
        x = torch.tensor([0])
        torch.distributed.recv(x, 1)
        # the # of None received is the kv that are not selected
        assert x.item() == len(buf.buffer)
        # and the size of the buffer should be 2000 * buffer len
        print(buf.buffer_size)
        assert buf.buffer_size == 1700 * len(buf.buffer)
    else:
        torch.distributed.send(torch.tensor([n]), 0)

    print("Passed stress test!")


if __name__ == "__main__":

    my_rank = int(os.environ['RANK'])

    torch.distributed.init_process_group(init_method="tcp://127.0.0.1:23456",
                                         world_size=2,
                                         rank=my_rank)

    print("initialized! My rank is %d" % my_rank)

    pipe = tdp.TorchDistributedPipe([[0, 1]], my_rank, "nccl")
    cpu_pipe = tdp.TorchDistributedPipe([[0, 1]], my_rank, "gloo")
    buffer = sklb.SimpleKVLookupBuffer(cpu_pipe, pipe, 170000)

    test_run(my_rank, buffer, pipe.device)

    stress_test(my_rank, buffer, pipe.device)

    buffer.close()
    pipe.close()
    cpu_pipe.close()
    print('Done')
