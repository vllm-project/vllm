import os
import time
from typing import List

import torch
from tqdm import tqdm

from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.pynccl_connector import pipe


def test_run(my_rank, pipe):
    # test run
    x = torch.tensor([1]).to(pipe.device)
    y = torch.tensor([[2., 3., 4., 8.]]).to(pipe.device)
    if my_rank == 0:
        pipe.send_tensor(x)
        print("sent tensor x")
        pipe.send_tensor(y)
        print("sent tensor y")
        x2 = pipe.recv_tensor()
        print("received x2 = ", x2)
        y2 = pipe.recv_tensor()
        print("received y2 = ", x2)

    else:
        x2 = pipe.recv_tensor()
        print("received x2 = ", x2)
        y2 = pipe.recv_tensor()
        print("received y2 = ", x2)
        pipe.send_tensor(x)
        print("sent tensor x")
        pipe.send_tensor(y)
        print("sent tensor y")

    assert torch.allclose(x, x2)
    assert torch.allclose(y, y2)


def stress_test(my_rank, pipe):

    torch.distributed.barrier()

    tensors: List[torch.Tensor] = []

    torch.manual_seed(0)

    for i in tqdm(range(500)):
        mean = torch.rand(1).item() * 100
        std = torch.rand(1).item() * 100
        size = torch.randint(900, 1000, (2, ))
        x = torch.normal(mean * 1.0, std * 1.0,
                         size=size.tolist()).to(pipe.device)

        # 5% probability of sending a None
        if torch.rand(1).item() < 0.05:
            tensors.append(None)
            tensors.append(None)
            tensors.append(None)
        else:
            tensors.append(x)
            tensors.append(x.mean().unsqueeze(0))
            tensors.append(x.std().unsqueeze(0))

    torch.distributed.barrier()

    for i in tqdm(range(500)):
        if my_rank == int((i % 10) > 3):
            pipe.send_tensor(tensors[3 * i])
            pipe.send_tensor(tensors[3 * i + 1])
            pipe.send_tensor(tensors[3 * i + 2])
        else:
            x = pipe.recv_tensor()
            mean = pipe.recv_tensor()
            std = pipe.recv_tensor()

            if x is None:
                assert mean is None
                assert std is None
            else:
                assert torch.allclose(x, tensors[3 * i])
                assert x.mean() == mean[0]
                assert x.std() == std[0]

        torch.distributed.barrier()


def latency_test(my_rank, pipe, nelement, ntensor):

    latencies = []

    torch.distributed.barrier()

    for i in tqdm(range(500)):

        tensors = []

        if my_rank == 0:
            # create tensor
            tensors = [
                torch.rand(nelement).to(pipe.device) for _ in range(ntensor)
            ]

        torch.distributed.barrier()

        if my_rank == 0:
            t = torch.tensor([time.time()],
                             dtype=torch.float64).to(pipe.device)
            for tensor in tensors:
                pipe.send_tensor(tensor)
            pipe.send_tensor(t)
        else:
            for _ in range(ntensor):
                pipe.recv_tensor()
            t = pipe.recv_tensor()
            latencies.append(time.time() - t.item())

    torch.distributed.barrier()

    print('Latency test passed.')
    print('Latency:', torch.tensor(latencies).mean().item() * 1000, 'ms')


if __name__ == "__main__":

    my_rank = int(os.environ['RANK'])

    torch.distributed.init_process_group(
        backend='gloo',
        init_method='tcp://localhost:12398',
        world_size=2,
        rank=my_rank,
    )

    config = KVTransferConfig(
        kv_connector='PyNcclConnector',
        kv_buffer_device='cuda',
        kv_buffer_size=1e9,
        kv_rank=my_rank,
        kv_role="kv_both",  # this arg doesn't matter in this test
        kv_parallel_size=2,
        kv_ip="127.0.0.1",
        kv_port=12345,
    )

    pipe = pipe.PyNcclPipe(
        local_rank=my_rank,
        config=config,
    )

    test_run(my_rank, pipe)
    stress_test(my_rank, pipe)

    # Use this function if you want to test the latency of pipe impl.
    # latency_test(my_rank, pipe, 1024 * 8 * 128, 80)
