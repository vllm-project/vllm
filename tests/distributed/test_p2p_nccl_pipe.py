# SPDX-License-Identifier: Apache-2.0

import multiprocessing
import time

import torch

from vllm.distributed.kv_transfer.kv_pipe.p2p_nccl_pipe import P2pNcclPipe


def process_task1(name):
    pipe = P2pNcclPipe(0, "127.0.0.1", 10001)
    device = torch.device(f"cuda:{0}")

    # send to node1
    tensor_id = "111111111"
    tensor1 = torch.ones([100, 100], dtype=torch.float16, device=device)
    remote_address1 = "127.0.0.1:10002"
    pipe.send_tensor(tensor1, tensor_id, remote_address1)

    # send to node2
    tensor_id = "222222222"
    tensor2 = torch.zeros([100, 100], dtype=torch.float16, device=device)
    remote_address2 = "127.0.0.1:10003"
    pipe.send_tensor(tensor2, tensor_id, remote_address2)

    # send to node1
    tensor_id = "333333333"
    tensor3 = torch.rand([100, 100], dtype=torch.float16, device=device)
    pipe.send_tensor(tensor3, tensor_id, remote_address1)

    time.sleep(100000)


def process_task2(name):
    P2pNcclPipe(1, "127.0.0.1", 10002)
    time.sleep(100000)


def process_task3(name):
    P2pNcclPipe(2, "127.0.0.1", 10003)
    time.sleep(100000)


if __name__ == "__main__":
    process1 = multiprocessing.Process(target=process_task1, args=("Task1", ))
    process2 = multiprocessing.Process(target=process_task2, args=("Task2", ))
    process3 = multiprocessing.Process(target=process_task3, args=("Task3", ))

    process1.start()
    process2.start()
    process3.start()

    process1.join()
    process2.join()
    process3.join()
