# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# isort: skip_file
# ruff: noqa: E402
# mypy: disable-error-code="misc, assignment"
"""Torch stand-ins for the V2 model runner's Triton kernels.

Imported only where Triton cannot run, so the kernels are launched for real
wherever triton-cpu is available. See `shm.py` for the gate.
"""

from vllm.triton_utils import triton

import vllm.v1.worker.cpu.buffer_utils as cpu_buffer_utils
import vllm.v1.worker.gpu.buffer_utils as gpu_buffer_utils

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

# Patch multi-dimensional RoPE position setup.
import vllm.v1.worker.cpu.mm.rope as cpu_rope
import vllm.v1.worker.gpu.mm.rope as gpu_rope

gpu_rope._prepare_rope_positions_kernel = TorchKernel(cpu_rope.prepare_rope_positions)

import vllm.v1.worker.cpu.kv_zero as cpu_kv_zero
import vllm.v1.worker.utils as worker_utils

worker_utils.KVBlockZeroer.__init__ = cpu_kv_zero.init
worker_utils.KVBlockZeroer.zero_block_ids = cpu_kv_zero.zero_block_ids

# Patch the hybrid model state's per-step accepted-token bookkeeping.
import vllm.v1.worker.cpu.model_states.mamba_hybrid as cpu_mamba_hybrid
import vllm.v1.worker.gpu.model_states.mamba_hybrid as gpu_mamba_hybrid

gpu_mamba_hybrid._scatter_num_accepted_kernel = TorchKernel(
    cpu_mamba_hybrid.scatter_num_accepted
)
gpu_mamba_hybrid._fill_num_accepted_kernel = TorchKernel(
    cpu_mamba_hybrid.fill_num_accepted
)

# Patch align-mode state migration. The copy kernels address state tensors
# through device pointer arrays, so they go at the method level, and
# _populate_metadata is extended to keep the tensors those addresses describe.
import vllm.v1.worker.cpu.mamba_utils as cpu_mamba_utils
import vllm.v1.worker.mamba_utils as gpu_mamba_utils

gpu_mamba_utils.MambaSpecDecodeGPUContext._populate_metadata = (
    cpu_mamba_utils.populate_metadata
)
gpu_mamba_utils.MambaSpecDecodeGPUContext.run_fused_precopy = (
    cpu_mamba_utils.run_fused_precopy
)
gpu_mamba_utils.MambaSpecDecodeGPUContext.run_fused_postprocess_align = (
    cpu_mamba_utils.run_fused_postprocess_align
)
# This kernel is imported by name into the hybrid state, so the binding that
# the caller resolves is the one there, not the one in mamba_utils.
gpu_mamba_hybrid.preprocess_mamba_align_fused_kernel = TorchKernel(
    cpu_mamba_utils.preprocess_mamba_align
)

# Patch sampler kernels.
import vllm.v1.worker.cpu.sample.gumbel as cpu_gumbel
import vllm.v1.worker.gpu.sample.gumbel as gpu_gumbel

gpu_gumbel._temperature_kernel = TorchKernel(cpu_gumbel.apply_temperature)
gpu_gumbel._gumbel_sample_kernel = TorchKernel(cpu_gumbel.gumbel_sample)
