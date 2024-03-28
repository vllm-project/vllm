import torch
import contextlib
import gc
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import os
import socket
from collections import defaultdict
import copy

from vllm.engine.ray_utils import RayWorkerVllm, initialize_ray_cluster

from vllm.lora.models import convert_mapping
from tests.lora.test_layers import get_random_id_to_index, create_random_inputs
from vllm.lora.fully_sharded_layers import *
from vllm.model_executor.parallel_utils.parallel_state import (
    destroy_model_parallel)
from vllm import EngineArgs

from vllm.model_executor.parallel_utils.parallel_state import (
    ensure_model_parallel_initialized)
from vllm.model_executor.parallel_utils.communication_op import (broadcast)
from vllm.lora.fully_sharded_layers import *

from vllm.lora.layers import (LoRAMapping, ColumnParallelLinearWithLoRA,
                              RowParallelLinearWithLoRA,
                              MergedColumnParallelLinearWithLoRA,
                              QKVParallelLinearWithLora)

from vllm.config import LoRAConfig
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               RowParallelLinear,
                                               QKVParallelLinear)
from vllm.model_executor.utils import set_random_seed

TOLERANCES = {
    torch.float16: (5e-3, 5e-3),
    torch.float32: (5e-3, 5e-3),
    torch.bfloat16: (3e-2, 2e-2),
}

HIGH, LOW = 0.3, -0.3


class Worker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        layer,
        parallel_config,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config: Optional[LoRAConfig] = None,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
    ) -> None:
        self.parallel_config = parallel_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = lora_config
        self.is_driver_worker = is_driver_worker
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

    def init_model(self) -> None:
        # torch.distributed.all_reduce does not free the input tensor until
        # the synchronization point. This causes the memory usage to grow
        # as the number of all_reduce calls increases. This env var disables
        # this behavior.
        # Related issue:
        # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        # Initialize the distributed environment.
        init_distributed_environment(self.parallel_config, self.rank,
                                     self.distributed_init_method)

    @torch.inference_mode()
    def create_column_parallel_packed_layer(self,
                                            repeats,
                                            dtype,
                                            device,
                                            linear_method=None,
                                            fully_sharded=True):
        if repeats == 2:
            linear = MergedColumnParallelLinear(4096, [4096] * repeats,
                                                bias=False,
                                                linear_method=linear_method)
            linear.weight.data = torch.rand_like(
                linear.weight.data, dtype=dtype, device=device) * HIGH + LOW
            lora_linear = MergedColumnParallelLinearWithShardedLoRA(
                linear
            ) if fully_sharded else MergedColumnParallelLinearWithLoRA(linear)
        else:
            linear = QKVParallelLinear(4096,
                                       64,
                                       32,
                                       bias=False,
                                       linear_method=linear_method)
            linear.weight.data = torch.rand_like(
                linear.weight.data, dtype=dtype, device=device) * HIGH + LOW
            lora_linear = QKVParallelLinearWithShardedLora(
                linear) if fully_sharded else QKVParallelLinearWithLora(linear)

        linear.weight.data = broadcast(linear.weight.data, src=0)

        class FakeConfig:
            hidden_size = 4096
            num_key_value_heads = 32
            num_attention_heads = 32

        lora_linear.create_lora_weights(self.lora_config.max_loras,
                                        self.lora_config,
                                        model_config=FakeConfig())
        return linear, lora_linear

    @torch.inference_mode()
    def test_column_parallel_packed_lora(self, num_loras, repeats, dtype):
        set_random_seed(2028374)
        device = torch.device(f'cuda:{torch.distributed.get_rank()}')
        id_to_index = get_random_id_to_index(num_loras,
                                             self.lora_config.max_loras)
        linear, lora_linear = self.create_column_parallel_packed_layer(
            repeats, dtype, device)
        assert lora_linear.lora_a_stacked[0].device == device

        inputs, index_mapping, prompt_mapping = create_random_inputs(
            active_lora_ids=id_to_index,
            num_inputs=32 * num_loras,
            input_size=(1, 4096),
            input_range=(LOW, HIGH),
            input_type=dtype,
        )
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping)
        mapping_info = convert_mapping(
            lora_mapping,
            id_to_index,
            self.lora_config.max_loras,
            512,
            self.lora_config.lora_extra_vocab_size,
        )
        lora_linear.set_mapping(*mapping_info, )

        loras = []
        for i in range(repeats):
            lora_a = torch.rand((4096, self.lora_config.max_lora_rank),
                                dtype=dtype,
                                device=device) * HIGH + LOW
            if repeats == 2:
                lora_b = torch.rand((self.lora_config.max_lora_rank, 4096),
                                    dtype=dtype,
                                    device=device) * HIGH + LOW
            else:
                out_size = lora_linear.q_proj_shard_size if i == 0 else \
                           lora_linear.kv_proj_shard_size
                out_size *= get_tensor_model_parallel_world_size()
                lora_b = torch.rand((self.lora_config.max_lora_rank, out_size),
                                    dtype=dtype,
                                    device=device) * HIGH + LOW

            lora_a = broadcast(lora_a, src=0)
            lora_b = broadcast(lora_b, src=0)
            loras.append((lora_a, lora_b))
        for lora_idx in range(self.lora_config.max_loras):
            lora_linear.set_lora(lora_idx, [lora[0] for lora in loras],
                                 [lora[1] for lora in loras], None)

        inputs = [inp.to(device=device, non_blocking=True) for inp in inputs]
        inputs = torch.cat(inputs)
        inputs = broadcast(inputs, src=0)

        # for better comparison, keep this the same as the dtype
        # used to compute the expected result
        lora_result = lora_linear(inputs)[0]

        # the lora and the test run the same linear layer,
        # exclude this from the testing.
        # Also, running this batched vs per example has different results,
        # so it introduces unnecessary uncertainty
        linear_result = linear(inputs)[0]
        lora_result -= linear_result
        expected_results = []
        world_size = get_tensor_model_parallel_world_size()
        for input_, lora_idx in zip(inputs, mapping_info[0]):
            sub_result = []
            rank = get_tensor_model_parallel_rank()
            input_ = input_.view(-1, input_.shape[-1])
            for r in range(repeats):
                out_size = loras[r][1].shape[1] // world_size
                lora_a = loras[r][0]
                lora_b = loras[r][1][:, out_size * rank:out_size * (rank + 1)]
                # on my gpu's, this computation in fp16 leads to
                # diff results on each.
                # Off by exactly 0.0625 or 0.125.
                # fp32 removes those differences.
                # Inherently less comparable to punica, but,
                # better than having this issue
                result = (input_.to(dtype=torch.float32) @ lora_a.to(
                    dtype=torch.float32)) @ lora_b.to(dtype=torch.float32)
                sub_result.append(result)
            expected_results.append(torch.cat(sub_result, dim=-1))

        expected_result = torch.cat(expected_results).to(dtype=torch.float16)
        rtol, atol = TOLERANCES[lora_result.dtype]
        assert torch.allclose(lora_result,
                              expected_result,
                              rtol=rtol,
                              atol=atol)
        print('Complete. All tests passed!')

    @torch.inference_mode()
    def create_random_linear_parallel_layer(self,
                                            orientation,
                                            dtype,
                                            device,
                                            linear_method=None,
                                            fully_sharded=True):
        if orientation == "row":
            linear = RowParallelLinear(4096,
                                       4096,
                                       input_is_parallel=False,
                                       reduce_results=True,
                                       bias=False,
                                       linear_method=linear_method)
            linear.weight.data = torch.rand_like(
                linear.weight.data, dtype=dtype, device=device) * HIGH + LOW
            lora_linear = RowParallelLinearWithShardedLoRA(
                linear) if fully_sharded else RowParallelLinearWithLoRA(linear)
        else:
            linear = ColumnParallelLinear(4096,
                                          4096,
                                          gather_output=True,
                                          bias=False,
                                          linear_method=linear_method)
            linear.weight.data = torch.rand_like(
                linear.weight.data, dtype=dtype, device=device) * HIGH + LOW
            lora_linear = ColumnParallelLinearWithShardedLoRA(
                linear) if fully_sharded else ColumnParallelLinearWithLoRA(
                    linear)
        linear.weight.data = broadcast(linear.weight.data, src=0)
        lora_linear.create_lora_weights(self.lora_config.max_loras,
                                        self.lora_config)

        return linear, lora_linear

    @torch.inference_mode()
    def test_lora(self, num_loras, orientation, dtype):
        set_random_seed(129383)
        device = torch.device(f'cuda:{torch.distributed.get_rank()}')
        id_to_index = get_random_id_to_index(num_loras,
                                             self.lora_config.max_loras)
        linear, lora_linear = self.create_random_linear_parallel_layer(
            orientation, dtype, device)
        assert lora_linear.lora_a_stacked.device == device

        inputs, index_mapping, prompt_mapping = create_random_inputs(
            active_lora_ids=id_to_index,
            num_inputs=32 * num_loras,
            input_size=(1, 4096),
            input_range=(LOW, HIGH),
            input_type=dtype,
        )
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping)
        mapping_info = convert_mapping(
            lora_mapping,
            id_to_index,
            self.lora_config.max_loras,
            512,
            self.lora_config.lora_extra_vocab_size,
        )
        lora_linear.set_mapping(*mapping_info, )

        loras = []
        lora_a = torch.rand(
            (4096, self.lora_config.max_lora_rank), dtype=dtype,
            device=device) * HIGH + LOW
        lora_b = torch.rand(
            (self.lora_config.max_lora_rank, 4096), dtype=dtype,
            device=device) * HIGH + LOW

        lora_a = broadcast(lora_a, src=0)
        lora_b = broadcast(lora_b, src=0)
        for lora_idx in range(self.lora_config.max_loras):
            shard = 4096 // get_tensor_model_parallel_world_size()
            start, end = shard * get_tensor_model_parallel_rank(), shard * (
                get_tensor_model_parallel_rank() + 1)
            if orientation == 'col':
                lora_linear.set_lora(lora_idx,
                                     lora_a,
                                     lora_b[:, start:end],
                                     embeddings_tensor=None)
            else:
                lora_linear.set_lora(lora_idx,
                                     lora_a,
                                     lora_b,
                                     embeddings_tensor=None)
            loras.append((lora_a, lora_b))

        inputs = [inp.to(device=device, non_blocking=True) for inp in inputs]
        inputs = torch.cat(inputs)
        inputs = broadcast(inputs, src=0)

        lora_result = lora_linear(inputs)[0]

        linear_result = linear(inputs)[0]
        lora_result -= linear_result
        expected_results = []
        for input_, lora_idx in zip(inputs, mapping_info[0]):
            input_ = input_.view(-1, input_.shape[-1])
            lora = loras[lora_idx]
            result = (input_.to(dtype=torch.float32) @ lora[0].to(
                dtype=torch.float32)) @ lora[1].to(dtype=torch.float32)
            expected_results.append(result)

        expected_result = torch.cat(expected_results).to(dtype=torch.float16)
        rtol, atol = TOLERANCES[lora_result.dtype]
        assert torch.allclose(lora_result,
                              expected_result,
                              rtol=rtol,
                              atol=atol)
        print('Complete. All tests passed!')

    @torch.inference_mode()
    def profile_sampler(self, sampler, linear, inputs, steps):
        # warm up
        for _ in range(100):
            result = sampler._get_logits(hidden_states=inputs,
                                         embedding=linear.weight,
                                         embedding_bias=None)

        start_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(steps)
        ]
        end_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(steps)
        ]

        for i in range(steps):
            start_events[i].record()
            result = sampler._get_logits(hidden_states=inputs,
                                         embedding=linear.weight,
                                         embedding_bias=None)
            end_events[i].record()

        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        return times

    @torch.inference_mode()
    def profile_linear(self, lora_linear, inputs, steps):
        # warm up
        for _ in range(100):
            lora_result = lora_linear(inputs)

        start_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(steps)
        ]
        end_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(steps)
        ]

        for i in range(steps):
            start_events[i].record()
            lora_result = lora_linear(inputs)
            end_events[i].record()

        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        return times

    @torch.inference_mode()
    def speed(self, num_loras, inputs_per_lora):
        set_random_seed(1792834)
        device = torch.device(f'cuda:{torch.distributed.get_rank()}')
        id_to_index = get_random_id_to_index(num_loras, num_loras)

        inputs, index_mapping, prompt_mapping = create_random_inputs(
            active_lora_ids=id_to_index,
            num_inputs=inputs_per_lora * num_loras,
            input_size=(1, 4096),
            input_range=(LOW, HIGH),
            input_type=torch.float16,
        )
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping)
        mapping_info = convert_mapping(
            lora_mapping,
            id_to_index,
            self.lora_config.max_loras,
            512,
            self.lora_config.lora_extra_vocab_size,
        )
        inputs = [inp.to(device=device, non_blocking=True) for inp in inputs]
        inputs = torch.cat(inputs)
        inputs = broadcast(inputs, src=0)

        steps = 1000

        ############     TEST FULLY SHARDED COL     ############
        linear, lora_linear = self.create_random_linear_parallel_layer(
            'col', torch.float16, device, fully_sharded=True)
        lora_linear.set_mapping(*mapping_info, )
        fs_times = self.profile_linear(lora_linear, inputs, steps)

        linear, lora_linear = self.create_random_linear_parallel_layer(
            'col', torch.float16, device, fully_sharded=False)
        lora_linear.set_mapping(*mapping_info, )
        times = self.profile_linear(lora_linear, inputs, steps)
        if get_tensor_model_parallel_rank() == 0:
            print(f'num loras, inputs per lora, rank [{num_loras}, ' +
                  f'{inputs_per_lora}, ' +
                  f'{self.lora_config.max_lora_rank}], ' +
                  f'fully sharded col: {sum(fs_times)/steps} ms | ' +
                  f'partially sharded col: {sum(times)/steps} ms')

        ############     TEST FULLY SHARDED ROW     ############
        linear, lora_linear = self.create_random_linear_parallel_layer(
            'row', torch.float16, device, fully_sharded=True)
        lora_linear.set_mapping(*mapping_info, )
        fs_times = self.profile_linear(lora_linear, inputs, steps)

        linear, lora_linear = self.create_random_linear_parallel_layer(
            'row', torch.float16, device, fully_sharded=False)
        lora_linear.set_mapping(*mapping_info, )
        times = self.profile_linear(lora_linear, inputs, steps)
        if get_tensor_model_parallel_rank() == 0:
            print(f'num loras, inputs per lora, rank [{num_loras}, ' +
                  f'{inputs_per_lora}, ' +
                  f'{self.lora_config.max_lora_rank}], ' +
                  f'fully sharded row: {sum(fs_times)/steps} ms | ' +
                  f'partially sharded row: {sum(times)/steps} ms')

        ############     TEST FULLY SHARDED MERGED COL     ############
        linear, lora_linear = self.create_column_parallel_packed_layer(
            2, torch.float16, device, fully_sharded=True)
        linear.gather_output = True
        lora_linear.set_mapping(*mapping_info, )
        fs_times = self.profile_linear(lora_linear, inputs, steps)

        linear, lora_linear = self.create_column_parallel_packed_layer(
            2, torch.float16, device, fully_sharded=False)
        linear.gather_output = True
        lora_linear.set_mapping(*mapping_info, )
        times = self.profile_linear(lora_linear, inputs, steps)

        if get_tensor_model_parallel_rank() == 0:
            print(f'num loras, inputs per lora, rank [{num_loras}, ' +
                  f'{inputs_per_lora}, ' +
                  f'{self.lora_config.max_lora_rank}], ' +
                  f'fully sharded merged col: {sum(fs_times)/steps} ms | ' +
                  f'partially sharded merged col: {sum(times)/steps} ms')

        ############     TEST FULLY SHARDED QKV     ############
        linear, lora_linear = self.create_column_parallel_packed_layer(
            3, torch.float16, device, fully_sharded=True)
        linear.gather_output = True
        lora_linear.set_mapping(*mapping_info, )
        fs_times = self.profile_linear(lora_linear, inputs, steps)

        linear, lora_linear = self.create_column_parallel_packed_layer(
            3, torch.float16, device, fully_sharded=False)
        linear.gather_output = True
        lora_linear.set_mapping(*mapping_info, )
        times = self.profile_linear(lora_linear, inputs, steps)
        if get_tensor_model_parallel_rank() == 0:
            print(f'num loras, inputs per lora, rank [{num_loras}, ' +
                  f'{inputs_per_lora}, ' +
                  f'{self.lora_config.max_lora_rank}], ' +
                  f'fully sharded qkv: {sum(fs_times)/steps} ms | ' +
                  f'partially sharded qkv: {sum(times)/steps} ms')


def init_distributed_environment(
    parallel_config,
    rank: int,
    distributed_init_method: Optional[str] = None,
) -> None:
    """Initialize the distributed environment."""
    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != parallel_config.world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size "
                f"({torch_world_size} vs. {parallel_config.world_size}).")
    elif not distributed_init_method:
        raise ValueError(
            "distributed_init_method must be set if torch.distributed "
            "is not already initialized")
    else:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)


def get_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
    return s.getsockname()[0]


def get_distributed_init_method(ip: str, port: int) -> str:
    return f"tcp://{ip}:{port}"


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def set_cuda_visible_devices(device_ids: List[int]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))


def cleanup():
    destroy_model_parallel()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()


def test_sharded_layers(rank,
                        tp_size,
                        max_loras,
                        orientation='col',
                        repeats=None,
                        speed=False):
    engine_args = EngineArgs(
        model=
        'mistralai/Mistral-7B-Instruct-v0.2',  # "meta-llama/Llama-2-7b-hf",
        enable_lora=True,
        tensor_parallel_size=tp_size,
        max_loras=max_loras,
        max_lora_rank=rank,
        max_cpu_loras=max_loras,
        max_num_seqs=256,
        max_model_len=512,
        enforce_eager=True)

    engine_args.lora_dtype = torch.float16
    engine_configs = engine_args.create_engine_configs()
    parallel_config = engine_configs[2]
    lora_config = engine_configs[-1]
    # Initialize the cluster.
    initialize_ray_cluster(parallel_config)

    driver_dummy_worker = None
    workers = []

    driver_ip = get_ip()
    for bundle_id, bundle in enumerate(
            parallel_config.placement_group.bundle_specs):
        if not bundle.get("GPU", 0):
            continue
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=parallel_config.placement_group,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=bundle_id,
        )
        worker = ray.remote(
            num_cpus=0,
            num_gpus=1,
            scheduling_strategy=scheduling_strategy,
        )(RayWorkerVllm).remote(True)

        worker_ip = ray.get(worker.get_node_ip.remote())
        if worker_ip == driver_ip and driver_dummy_worker is None:
            # If the worker is on the same node as the driver, we use it
            # as the resource holder for the driver process.
            driver_dummy_worker = worker
        else:
            workers.append(worker)

    if driver_dummy_worker is None:
        raise ValueError(
            "Ray does not allocate any GPUs on the driver node. Consider "
            "adjusting the Ray placement group or running the driver on a "
            "GPU node.")

    driver_node_id, driver_gpu_ids = ray.get(
        driver_dummy_worker.get_node_and_gpu_ids.remote())
    worker_node_and_gpu_ids = ray.get(
        [worker.get_node_and_gpu_ids.remote() for worker in workers])

    node_workers = defaultdict(list)
    node_gpus = defaultdict(list)

    node_workers[driver_node_id].append(0)
    node_gpus[driver_node_id].extend(driver_gpu_ids)
    for i, (node_id, gpu_ids) in enumerate(worker_node_and_gpu_ids, start=1):
        node_workers[node_id].append(i)
        node_gpus[node_id].extend(gpu_ids)
    for node_id, gpu_ids in node_gpus.items():
        node_gpus[node_id] = sorted(gpu_ids)

    # Set CUDA_VISIBLE_DEVICES for the driver.
    set_cuda_visible_devices(node_gpus[driver_node_id])
    for worker, (node_id, _) in zip(workers, worker_node_and_gpu_ids):
        worker.set_cuda_visible_devices.remote(node_gpus[node_id])

    distributed_init_method = get_distributed_init_method(
        driver_ip, get_open_port())

    # Lazy import the Worker to avoid importing torch.cuda/xformers
    # before CUDA_VISIBLE_DEVICES is set in the Worker
    # from vllm.worker.worker import Worker

    # Initialize torch distributed process group for the workers.
    parallel_config = copy.deepcopy(parallel_config)

    for rank, (worker, (node_id, _)) in enumerate(zip(workers,
                                                      worker_node_and_gpu_ids),
                                                  start=1):
        local_rank = node_workers[node_id].index(rank)
        worker.init_worker.remote(
            lambda rank=rank, local_rank=local_rank: Worker(
                None,  # layer
                parallel_config,
                local_rank,
                rank,
                distributed_init_method,
                lora_config=lora_config))

    driver_rank = 0
    driver_local_rank = node_workers[driver_node_id].index(driver_rank)
    driver_worker = Worker(
        None,  # layer
        parallel_config,
        driver_local_rank,
        driver_rank,
        distributed_init_method,
        lora_config=lora_config,
        is_driver_worker=True)

    run_workers(driver_worker, workers, 'init_model')
    if speed:
        run_workers(driver_worker,
                    workers,
                    'speed',
                    num_loras=max_loras,
                    inputs_per_lora=32)
    elif not repeats:
        run_workers(driver_worker,
                    workers,
                    'test_lora',
                    num_loras=max_loras,
                    orientation=orientation,
                    dtype=torch.float16)
    else:
        run_workers(driver_worker,
                    workers,
                    'test_column_parallel_packed_lora',
                    num_loras=max_loras,
                    repeats=repeats,
                    dtype=torch.float16)

    cleanup()


def run_workers(driver, workers, method, *args, **kwargs):
    ray_worker_outputs = [
        worker.execute_method.remote(method, *args, **kwargs)
        for worker in workers
    ]
    driver_worker_output = getattr(driver, method)(*args, **kwargs)

    # Get the results of the ray workers.
    ray_worker_outputs = ray.get(ray_worker_outputs)

    return [driver_worker_output] + ray_worker_outputs


def profile_expand(S, R):
    out = torch.zeros((S, 4096), dtype=torch.float16, device='cuda')
    x = torch.rand((S, R), dtype=torch.float32, device='cuda')
    w = torch.rand((S, 1, 4096, R), dtype=torch.float16, device='cuda')
    indices = torch.arange(S, dtype=torch.long, device='cuda')

    steps = 5000
    # warm up
    for _ in range(100):
        lora_result = bgmv(out, x, w, indices, 0, 1.0)
        # return

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]

    for i in range(steps):
        start_events[i].record()
        lora_result = bgmv(out, x, w, indices, 0, 1.0)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    print(f'[s, rank] [{S}, {R}]: ', sum(times) / steps)


def profile_shrink(S, R):
    out = torch.zeros((S, R), dtype=torch.float16, device='cuda')
    x = torch.rand((S, 4096), dtype=torch.float32, device='cuda')
    w = torch.rand((S, 1, R, 4096), dtype=torch.float16, device='cuda')
    indices = torch.arange(S, dtype=torch.long, device='cuda')

    steps = 5000
    # warm up
    for _ in range(100):
        lora_result = bgmv(out, x, w, indices, 0, 1.0)
        # return

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]

    for i in range(steps):
        start_events[i].record()
        lora_result = bgmv(out, x, w, indices, 0, 1.0)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    print(f'[s, rank] [{S}, {R}]: ', sum(times) / steps)


def main():
    ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
    if ray_usage != "1":
        os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

    # test_sharded_layers(rank=8, tp_size=2, max_loras=8, orientation='col')
    # test_sharded_layers(rank=32, tp_size=2, max_loras=8, repeats=3)
    # test_sharded_layers(rank=32, tp_size=2, max_loras=8, repeats=2)

    # test_sharded_layers(rank=8, tp_size=2, max_loras=16, speed=True)
    test_sharded_layers(rank=16, tp_size=2, max_loras=16, speed=True)
    # test_sharded_layers(rank=32, tp_size=2, max_loras=16, speed=True)
    # test_sharded_layers(rank=64, tp_size=2, max_loras=16, speed=True)

    # profile_expand(128, 16)
    # profile_expand(256, 16)
    # profile_expand(512, 16)
    # profile_expand(4096, 16)

    # profile_shrink(128, 16)
    # profile_shrink(256, 16)
    # profile_shrink(512, 16)
    # profile_shrink(4096, 16)


if __name__ == '__main__':
    main()
