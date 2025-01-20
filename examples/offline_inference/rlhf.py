"""
a simple demonstration of RLHF with vLLM, inspired by
the OpenRLHF framework https://github.com/OpenRLHF/OpenRLHF .
It follows the design that, training processes and inference processes
are different, and they live on different GPUs.
Training processes send prompts to inference processes to generate data,
and also synchronize the weights of the model by broadcasting the weights
from the training process to the inference process.
Note that this is a simple demonstration of one training instance and one
inference instance. In practice, there could be multiple training instances
and multiple inference instances. For the full implementation, please refer
to the OpenRLHF framework.
"""
import os

import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.utils import get_ip, get_open_port
from vllm.worker.worker import Worker


def stateless_init_process_group(master_address, master_port, rank, world_size,
                                 device):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes) 
    and vLLM workers.
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    pg = StatelessProcessGroup.create(host=master_address,
                                      port=master_port,
                                      rank=rank,
                                      world_size=world_size)
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl


class MyWorker(Worker):
    """
    The `MyWorker` class inherits from `Worker` to provide custom functions.
    For simplicity, we define the `MyWorker` class in this self-contained 
    script. Normally, we should define the `MyWorker` class in a separate 
    file and pass the qualified name of the class to the `worker_cls` 
    parameter.
    """

    def init_weight_update_group(self, master_address, master_port,
                                 rank_offset, world_size):
        from vllm.distributed.parallel_state import get_world_group
        rank = get_world_group().rank + rank_offset
        self.model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            self.device,
        )

    def update_weight(self, name, dtype, shape):
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        self.model_update_group.broadcast(weight,
                                          src=0,
                                          stream=torch.cuda.current_stream())

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight

    def check_weights_changed(self):
        """
        Check if the weights are updated to 0.
        """
        weights_updated = True
        for name, p in self.model_runner.model.named_parameters():
            weights_updated = weights_updated and torch.allclose(
                p, torch.zeros_like(p))
        return weights_updated


class MyLLM(LLM):

    def __init__(self, *args, **kwargs):
        # a hack to make the script work.
        # stop ray from manipulating CUDA_VISIBLE_DEVICES
        # at the top-level
        del os.environ["CUDA_VISIBLE_DEVICES"]
        super().__init__(*args, **kwargs)


"""
Start the training process, here we use huggingface transformers 
as an example to hold a model on GPU 0.
"""

train_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
train_model.to("cuda:0")
"""
Start the inference process, here we use vLLM to hold a model on GPU 1 and 
GPU 2. For the details on how to use ray, please refer to the ray 
documentation https://docs.ray.io/en/latest/ .
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
ray.init()

pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * 2)
ray.get(pg_inference.ready())
scheduling_inference = PlacementGroupSchedulingStrategy(
    placement_group=pg_inference,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=0,
)
"""
launch the vLLM inference engine.
here we use `enforce_eager` to reduce the start time.
"""
llm = ray.remote(
    num_cpus=0,
    num_gpus=0,
    scheduling_strategy=scheduling_inference,
)(MyLLM).remote(
    model="facebook/opt-125m",
    enforce_eager=True,
    worker_cls=MyWorker,
    tensor_parallel_size=2,
    distributed_executor_backend="ray",
)

# Generate texts from the prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0)

outputs = ray.get(llm.generate.remote(prompts, sampling_params))

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, "
          f"Generated text: {generated_text!r}")

# set up the communication between the training process
# and the inference engine.
master_address = get_ip()
master_port = get_open_port()

handle = llm.collective_rpc.remote("init_weight_update_group",
                                   args=(master_address, master_port, 1, 3))
model_update_group = stateless_init_process_group(master_address, master_port,
                                                  0, 3, torch.device("cuda:0"))
ray.get(handle)

# simulate training, modify the weights of the model.
for name, p in train_model.named_parameters():
    p.data.zero_()

# sync weight from the training process to the inference engine.
for name, p in train_model.named_parameters():
    handle = llm.collective_rpc.remote("update_weight",
                                       args=(name, p.dtype, p.shape))
    model_update_group.broadcast(p, src=0, stream=torch.cuda.current_stream())
    ray.get(handle)

# check if the weights are updated.
assert all(ray.get(llm.collective_rpc.remote("check_weights_changed")))

# use the updated model to generate texts, they will be nonsense
# because the weights are all zeros.
outputs_updated = ray.get(llm.generate.remote(prompts, sampling_params))
for output in outputs_updated:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, "
          f"Generated text: {generated_text!r}")
