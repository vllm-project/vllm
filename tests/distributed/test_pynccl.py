import pytest
import torch

from vllm.implementations.launcher.mp_launcher import MPLauncher
from vllm.implementations.coordinator import CoordinatorType
from vllm.implementations.communicator import CommunicatorType
from vllm.implementations.distributed_tasks.global_coordinator_task import GlobalCoordinatorDistributedTask

class AllReduceDistributedTask(GlobalCoordinatorDistributedTask):
    def post_init_distributed(self, **kwargs):
        tensor = torch.ones(16, 1024, 1024, dtype=torch.float32).cuda(self.coordinator.get_local_rank())
        self.communicator.all_reduce(tensor_in=tensor)
        result = tensor.mean().cpu().item()
        assert result == self.coordinator.get_local_world_size()    

@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
def test_pynccl():
    MPLauncher(n_tasks=2).launch(
        task_type=AllReduceDistributedTask,
        coordinator_type=CoordinatorType.TORCH_DISTRIBUTED,
        communicator_type=CommunicatorType.PYNCCL,
    )


class CUDAGraphAllReduceDistributedTask(GlobalCoordinatorDistributedTask):
    def post_init_distributed(self, **kwargs):
        graph = torch.cuda.CUDAGraph()
        device=f'cuda:{self.coordinator.get_rank()}'
        stream = torch.cuda.Stream(device=device)

        # run something in the default stream to initialize torch engine
        a = torch.ones((4, 4), device=device)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            # operation during the graph capture is recorded but not executed
            # see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#creating-a-graph-using-stream-capture # noqa
            self.communicator.all_reduce(a, stream=stream)
        stream.synchronize()
        assert a.mean().cpu().item() == self.coordinator.get_world_size() ** 0
        graph.replay()
        stream.synchronize()
        assert a.mean().cpu().item() == self.coordinator.get_world_size() ** 1

@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
def test_pynccl_with_cudagraph():
    MPLauncher(n_tasks=2).launch(
        task_type=CUDAGraphAllReduceDistributedTask,
        coordinator_type=CoordinatorType.TORCH_DISTRIBUTED,
        communicator_type=CommunicatorType.PYNCCL,
    )
