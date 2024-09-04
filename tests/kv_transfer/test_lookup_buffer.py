
import vllm.distributed.kv_transfer.kv_pipe.torch_distributed_pipe as tdp
import vllm.distributed.kv_transfer.kv_lookup_buffer.simple_kv_lookup_buffer as sklb
import torch
import os
import random
from tqdm import tqdm
import time


def test_run(my_rank, buffer):
    # test run
    tokens = torch.tensor([1,2,3]).to(buffer.pipe.device)
    
    if my_rank == 0:
        key = 2.0 * torch.ones([5, 6]).to(buffer.pipe.device)
        value = 3.0 * torch.ones([5, 6]).to(buffer.pipe.device)

        placeholder = torch.tensor([1]).to(buffer.pipe.device)

        buffer.insert(tokens, placeholder, key, value, placeholder)
        
    else:
        placeholder = torch.tensor([1]).to(buffer.pipe.device)
        tok, roi, key, value, hidden = buffer.drop_select(tokens, placeholder)
        assert torch.allclose(tokens, tok)
        assert torch.allclose(key, 2.0 * torch.ones([5, 6]))
        assert torch.allclose(value, 3.0 * torch.ones([5, 6]))
        
    torch.distributed.barrier()
    
    if my_rank == 0:
        assert buffer.buffer_size == 0
        assert len(buffer.buffer) == 0


def stress_test(my_rank, pipe):
    
    torch.distributed.barrier()
    
    tensors = []
    
    for i in tqdm(range(2000)):
        mean = random.randint(1, 10)
        std = random.randint(1, 10)
        size = [random.randint(900, 1000), random.randint(900, 1000)]
        x = torch.normal(mean * 1.0, std * 1.0, size=size).to(pipe.device)
        
        # 5% probability of sending a None
        if random.randint(1, 100) < 5:
            tensors.append(None)
            tensors.append(None)
            tensors.append(None)
        else:
            tensors.append(x)
            tensors.append(x.mean())
            tensors.append(x.std())
        
    torch.distributed.barrier()
    
    for i in tqdm(range(2000)):
        if my_rank == int((i % 10) > 3):
            pipe.send_tensor(tensors[3*i])
            pipe.send_tensor(tensors[3*i+1])
            pipe.send_tensor(tensors[3*i+2])
        else:
            x = pipe.recv_tensor()
            mean = pipe.recv_tensor()
            std = pipe.recv_tensor()
            if x is None:
                assert mean is None
                assert std is None
            else:
                assert x.mean() == mean
                assert x.std() == std

    torch.distributed.barrier()

    print("Stress test passed.")
    
    
    
def latency_test(my_rank, pipe, nelement, ntensor):
    
    latencies = []
    
    torch.distributed.barrier()
    
    for i in tqdm(range(1000)):
        
        tensors = []
        
        if my_rank == 0:
            # create tensor
            tensors = [torch.rand(nelement).to(pipe.device) for _ in range(ntensor)]
        
        torch.distributed.barrier()
        
        if my_rank == 0:
            t = torch.tensor(time.time(), dtype=torch.float64).to(pipe.device)
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
                init_method="tcp://127.0.0.1:23456",
                world_size=2,
                rank=my_rank)

    print("initialized! My rank is %d" % my_rank)


    pipe = tdp.TorchDistributedPipe([[0,1]], my_rank, "nccl")
    buffer = sklb.SimpleKVLookupBuffer(pipe)

    test_run(my_rank, buffer)
