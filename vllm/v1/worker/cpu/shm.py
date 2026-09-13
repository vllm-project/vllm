# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# isort: skip_file
# ruff: noqa: E402
# mypy: disable-error-code="misc, assignment"

from typing import Any

import numpy as np

# Patch torch APIs
import torch


def noop(*args: Any, **kwargs: Any) -> None:
    pass


# Distinct no-op so empty_cache does not alias synchronize: Dynamo's
# handle_synchronize is keyed on that object and asserts on CPU-only hosts.
def empty_cache_noop(*args: Any, **kwargs: Any) -> None:
    pass


def fake_pin_memory(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    return self


class _EventPlaceholder:
    def __init__(self, *args, **kwargs) -> None:
        self.record = noop
        self.wait = noop
        self.synchronize = noop


class _StreamPlaceholder:
    def __init__(self, *args, **kwargs) -> None:
        self.wait_stream = noop
        self.wait_event = noop
        self.record_event = noop
        self.synchronize = noop
        self.query = lambda: True
        self.device = torch.device("cpu")

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


from vllm.utils.cpu_resource_utils import get_memory_node_info


def get_memory_info(*args: Any, **kwargs: Any) -> tuple[int, int]:
    meminfo = get_memory_node_info()
    return meminfo.available_memory, meminfo.total_memory


torch.Event = _EventPlaceholder
torch.cuda.Event = _EventPlaceholder
torch.cuda.Stream = _StreamPlaceholder
torch.cuda.set_stream = noop
torch.cuda.current_stream = lambda *args, **kwargs: _StreamPlaceholder()
torch.cuda.stream = lambda *args, **kwargs: _StreamPlaceholder()
torch.accelerator.synchronize = noop
torch.accelerator.empty_cache = empty_cache_noop
torch.Tensor.pin_memory = fake_pin_memory
torch.Tensor.record_stream = noop
torch.accelerator.get_memory_info = get_memory_info

# Patch vLLM torch utils
import vllm.utils.torch_utils as torch_utils


def async_tensor_h2d(
    data: list | np.ndarray | torch.Tensor,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype)
    return torch.tensor(data, dtype=dtype, device="cpu")


torch_utils.async_tensor_h2d = async_tensor_h2d

# Patch model runner APIs
import vllm.v1.worker.gpu.buffer_utils as gpu_buffer_utils
import vllm.v1.worker.cpu.buffer_utils as cpu_buffer_utils

gpu_buffer_utils.UvaBuffer = cpu_buffer_utils.UvaBuffer

# Patch Triton
from vllm.triton_utils import HAS_TRITON, tl, triton

if HAS_TRITON:
    tl.debug_barrier = noop

from vllm.v1.worker.cpu.kernel_shim import TorchKernel, next_power_of_2

# The Triton placeholder omits this, and the entry points below call it before
# reaching the kernel.
if not hasattr(triton, "next_power_of_2"):
    triton.next_power_of_2 = next_power_of_2

# Patch input batch kernels. The kernel object is the interception point: the
# entry points resolve it as a module global per call, so this reaches every
# caller, whereas they themselves are imported by name and cannot be replaced.
import vllm.v1.worker.cpu.input_batch as cpu_input_batch
import vllm.v1.worker.gpu.input_batch as gpu_input_batch

gpu_input_batch._prepare_prefill_inputs_kernel = TorchKernel(
    cpu_input_batch.prepare_prefill_inputs
)
gpu_input_batch._prepare_pos_seq_lens_kernel = TorchKernel(
    cpu_input_batch.prepare_pos_seq_lens
)
gpu_input_batch._combine_sampled_and_draft_tokens_kernel = TorchKernel(
    cpu_input_batch.combine_sampled_and_draft_tokens
)
gpu_input_batch._get_num_sampled_and_rejected_kernel = TorchKernel(
    cpu_input_batch.get_num_sampled_and_rejected
)
gpu_input_batch._post_update_kernel = TorchKernel(cpu_input_batch.post_update)
gpu_input_batch._post_update_num_computed_tokens_kernel = TorchKernel(
    cpu_input_batch.post_update_num_computed_tokens
)
gpu_input_batch._expand_idx_mapping_kernel = TorchKernel(
    cpu_input_batch.expand_idx_mapping
)

# Patch block table and staged writes at the method level: their kernels take
# device pointer arrays, which a torch implementation cannot resolve back to
# the tensors they refer to.
import vllm.v1.worker.cpu.block_table as cpu_block_table
import vllm.v1.worker.gpu.block_table as gpu_block_table

gpu_block_table.BlockTables.gather_block_tables = cpu_block_table.gather_block_tables
gpu_block_table.BlockTables.compute_slot_mappings = (
    cpu_block_table.compute_slot_mappings
)

gpu_buffer_utils.StagedWriteTensor.apply_write = cpu_buffer_utils.apply_write
gpu_buffer_utils.FusedStagedWriter.apply = cpu_buffer_utils.fused_apply

# Patch sampler kernels.
import vllm.v1.worker.cpu.sample.gumbel as cpu_gumbel
import vllm.v1.worker.gpu.sample.gumbel as gpu_gumbel

gpu_gumbel._temperature_kernel = TorchKernel(cpu_gumbel.apply_temperature)
gpu_gumbel._gumbel_sample_kernel = TorchKernel(cpu_gumbel.gumbel_sample)

import vllm.v1.worker.cpu.profiling as cpu_profiling

if cpu_profiling.ENABLED:
    import vllm.v1.worker.gpu.model_runner as gpu_model_runner
    import vllm.v1.worker.gpu.sample.sampler as gpu_sampler

    for _method, _name in (
        ("finish_requests", "mrv2::state_update"),
        ("free_states", "mrv2::state_update"),
        ("add_requests", "mrv2::state_update"),
        ("update_requests", "mrv2::state_update"),
        ("prepare_inputs", "mrv2::prepare_inputs"),
        ("postprocess_sampled", "mrv2::postprocess"),
        ("postprocess_num_computed_tokens", "mrv2::postprocess"),
    ):
        cpu_profiling.label(gpu_model_runner.GPUModelRunner, _method, _name)
    cpu_profiling.label(gpu_sampler.Sampler, "__call__", "mrv2::sample")
    cpu_profiling.label(
        gpu_block_table.BlockTables, "apply_staged_writes", "mrv2::staged_writes"
    )

    # Label the whole step and the forward in both runners: the host cost of a
    # step is then step minus forward, which is comparable between them rather
    # than a sum over whichever phases happen to carry a label.
    import vllm.v1.sample.sampler as v1_sampler
    import vllm.v1.worker.cpu.model_runner as cpu_model_runner_v2
    import vllm.v1.worker.cpu_model_runner as v1_cpu_model_runner
    import vllm.v1.worker.gpu_model_runner as v1_model_runner

    cpu_profiling.label(gpu_model_runner.GPUModelRunner, "execute_model", "mrv2::step")
    cpu_profiling.label_model_forward(
        cpu_model_runner_v2.CPUModelRunner, "mrv2::forward"
    )

    for _method, _name in (
        ("execute_model", "v1::step"),
        ("_prepare_inputs", "v1::prepare_inputs"),
        ("_update_states", "v1::state_update"),
        ("_update_states_after_model_execute", "v1::state_update"),
    ):
        cpu_profiling.label(v1_model_runner.GPUModelRunner, _method, _name)
    cpu_profiling.label(v1_sampler.Sampler, "__call__", "v1::sample")
    cpu_profiling.label_model_forward(v1_cpu_model_runner.CPUModelRunner, "v1::forward")
