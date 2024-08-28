
import vllm.distributed.kv_transfer.kv_pipe.torch_distributed_pipe as tdp
import torch
import os
import random
from tqdm import tqdm

my_rank = int(os.environ['RANK'])


torch.distributed.init_process_group(
            init_method="tcp://127.0.0.1:23456",
            world_size=2,
            rank=my_rank)

print("initialized! My rank is %d" % my_rank)


pipe = tdp.TorchDistributedPipe([[0,1]], my_rank, "nccl")

print('My device is ', pipe.device, ' default: ', torch.cuda.current_device())
print(pipe.target_rank_for_send, pipe.target_rank_for_recv)

# test run

if my_rank == 0:
    x = torch.tensor([1]).to(pipe.device)
    pipe.send(x, 1)
    
    
else:
    y = torch.tensor([0]).to(pipe.device)
    y = pipe.recv(y.shape, y.dtype)
    
    assert y.item() == 1

# if my_rank == 0:
#     x = torch.tensor([1]).to(pipe.device)
#     torch.distributed.send(x, dst=1, group=pipe.device_group)
# else:
#     x = torch.tensor([0]).to(pipe.device)
#     torch.distributed.recv(x, src=0, group=pipe.device_group)
#     assert x.item() == 1

print(my_rank, 'Test run successed! ')

if my_rank == 0:
    # send a tensor 1000 times
    for i in range(3):
        
        mean = random.randint(10, 100)
        std = random.randint(10, 100)
        size = [random.randint(10, 100), random.randint(10, 100)]
        x = torch.normal(mean, std, size=size).to(pipe.device)
        
        if i % 10 == 0:
            pipe.send_tensor(None)
            pipe.send_tensor(None)
            pipe.send_tensor(None)
        else:
            pipe.send_tensor(x)
            pipe.send_tensor(x.mean())
            pipe.send_tensor(x.std())
            
else:
    # recv a tensor 1000 times
    for i in tqdm(range(2)):
        
        x = pipe.recv_tensor()
        mean = pipe.recv_tensor()
        std = pipe.recv_tensor()
        
        if x is None:
            assert mean is None, std is None
        else:
            assert x.mean() == mean
            assert x.std() == std

        

        