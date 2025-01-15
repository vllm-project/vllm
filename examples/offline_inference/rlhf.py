# a simple demonstration of RLHF with VLLM.
import cloudpickle
import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams, configure_as_vllm_process
from vllm.utils import get_ip, get_open_port
from vllm.worker.worker import Worker


# recommended way to create data-plane communication
# between external (train processes) and VLLM workers.
def stateless_init_process_group(master_address, master_port, rank, world_size,
                                 device):
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    pg = StatelessProcessGroup.create(host=master_address,
                                      port=master_port,
                                      rank=rank,
                                      world_size=world_size)
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl


# inference code, inherit from Worker to provide custom functions
class MyWorker(Worker):
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

    def get_weight_square_sum(self):
        sum_value = 0.0
        for name, p in self.model_runner.model.named_parameters():
            sum_value += p.square().sum().item()
        return sum_value

class MyLLM(LLM):
    def __init__(self, *args, **kwargs):
        import os
        del os.environ["CUDA_VISIBLE_DEVICES"]
        super().__init__(*args, **kwargs)

# current process is a training process, and it takes 1 GPU.
# important: set some common environment variables the same as vLLM workers.
configure_as_vllm_process()

train_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m").to("cuda:0")

ray.init()

pg_train = placement_group([{"GPU": 1, "CPU": 0}])
ray.get(pg_train.ready())

scheduling_train = PlacementGroupSchedulingStrategy(
    placement_group=pg_train,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=0,)

pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * 2)
ray.get(pg_inference.ready())
scheduling_inference = PlacementGroupSchedulingStrategy(
    placement_group=pg_inference,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=0,)


class PlaceHolder:
    pass


# a place holder to reserve 1 GPU for the training process.
place_holder = ray.remote(
    num_cpus=0,
    num_gpus=1,
    scheduling_strategy=scheduling_train,
)(PlaceHolder).remote()

# inferencing engine, it takes 2 GPUs.
# for simplicity, we define the MyWorker class in this self-contained script,
# and the MyWorker class does not have qualified name in the global scope,
# so we need to pass the worker_cls through `cloudpickle.dumps(MyWorker)`.
# normally, we should define the MyWorker class in a separate file and pass
# the qualified name of the class to the worker_cls parameter.
# here we use `enforce_eager` to reduce test time.
llm = ray.remote(
    num_cpus=0,
    num_gpus=0,
    scheduling_strategy=scheduling_inference,
)(MyLLM).remote(
    model="facebook/opt-125m",
    enforce_eager=True,
    worker_cls=cloudpickle.dumps(MyWorker),
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

outputs_original = ray.get(llm.generate.remote(prompts, sampling_params))

master_address = get_ip()
master_port = get_open_port()

# set up the connection between the training process and the inference engine.
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
weight_square_sum_values = ray.get(
    llm.collective_rpc.remote("get_weight_square_sum"))
for x in weight_square_sum_values:
    assert x == 0.0

# use the updated model to generate texts.
outputs_updated = ray.get(llm.generate.remote(prompts, sampling_params))

# they should be different.
for output_original, output_updated in zip(outputs_original, outputs_updated):
    generated_text_original = output_original.outputs[0].text
    generated_text_updated = output_updated.outputs[0].text
    assert generated_text_original != generated_text_updated
