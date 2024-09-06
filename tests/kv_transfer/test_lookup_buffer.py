
import vllm.distributed.kv_transfer.kv_pipe.torch_distributed_pipe as tdp
import vllm.distributed.kv_transfer.kv_lookup_buffer.simple_kv_lookup_buffer as sklb
import torch
import os
import random
from tqdm import tqdm
import time


def test_run(my_rank, buffer):
    
    # buffer should be empty in the beginning    
    if my_rank == 0:
        assert buffer.buffer_size == 0
        assert len(buffer.buffer) == 0


    # insert
    tokens = torch.tensor([1,2,3]).to(buffer.pipe.device)
    roi = (tokens > 0)
    if my_rank == 0:
        key = 2.0 * torch.ones([5, 6]).to(buffer.pipe.device)
        value = 3.0 * torch.ones([5, 6]).to(buffer.pipe.device)

        placeholder = torch.tensor([1]).to(buffer.pipe.device)

        buffer.insert(tokens, roi, key, value, placeholder)
    torch.distributed.barrier()
        
    # drop_select
    if my_rank == 1:
        tok, roi_, key, value, hidden = buffer.drop_select(tokens, roi)
        assert torch.allclose(tokens, tok)
        assert torch.allclose(roi, roi_)
        assert torch.allclose(key, 2.0 * torch.ones([5, 6]))
        assert torch.allclose(value, 3.0 * torch.ones([5, 6]))
    torch.distributed.barrier()
    
    if my_rank == 0:
        assert buffer.buffer_size == 0
        assert len(buffer.buffer) == 0


def stress_test(my_rank, buf):
    
    torch.distributed.barrier()
    torch.manual_seed(100)

    device = buf.pipe.device
    
    reqs = [
        (
         torch.rand(100).to(device),   # tokens
         torch.ones(100).bool().to(device),    # roi
         torch.rand(100).to(device),   # key
         torch.rand(100).to(device),   # value
         torch.rand(100).to(device),   # hidden
         ) for i in range(200)]

    random.seed(my_rank)
    random.shuffle(reqs)
    
    torch.distributed.barrier()
    
    n = 0
    
    # the buffer size can only store 100 reqs
    # so the sender will occasionally block.needs to wait for the receiver.
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
    print('Rand %d done' % my_rank)
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

        
    
            
            
    
    
    

if __name__ == "__main__":

    my_rank = int(os.environ['RANK'])


    torch.distributed.init_process_group(
                init_method="tcp://127.0.0.1:23456",
                world_size=2,
                rank=my_rank)

    print("initialized! My rank is %d" % my_rank)


    pipe = tdp.TorchDistributedPipe([[0,1]], my_rank, "nccl")
    buffer = sklb.SimpleKVLookupBuffer(pipe, 170000)

    test_run(my_rank, buffer)
    
    stress_test(my_rank, buffer)
    
    print('Done')
