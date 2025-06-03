# SPDX-License-Identifier: Apache-2.0
import bisect
import gc
import time
from typing import TYPE_CHECKING, Optional, cast
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
# TPU XLA related
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr

import vllm.envs as envs
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
from vllm.config import ParallelConfig, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.lora.layers import BaseLayerWithLoRA
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.tpu import TPUModelLoader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (BatchedTensorInputs, MultiModalKwargs,
                                    PlaceholderRange)
from vllm.multimodal.utils import group_mm_inputs_by_modality
from vllm.sequence import IntermediateTensors
from vllm.utils import LayerBlockType, cdiv, is_pin_memory_available
from vllm.v1.attention.backends.pallas import (PallasAttentionBackend,
                                               PallasMetadata)
from vllm.v1.core.encoder_cache_manager import compute_encoder_budget
from vllm.v1.kv_cache_interface import (AttentionSpec, FullAttentionSpec,
                                        KVCacheConfig, KVCacheSpec,
                                        SlidingWindowSpec)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, LogprobsTensors,
                             ModelRunnerOutput)
from vllm.v1.sample.tpu.metadata import TPUSupportedSamplingMetadata
from vllm.v1.sample.tpu.sampler import Sampler as TPUSampler
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

from .utils import sanity_check_mm_encoder_outputs

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

# Here we utilize the behavior that out-of-bound index is ignored.
# FIXME(woosuk): Find a more reliable way to prevent possible bugs.
_PAD_SLOT_ID = 1_000_000_000
INVALID_TOKEN_ID = -1
# Smallest output size
MIN_NUM_SEQS = 8


#########################################################
# Ways to avoid recompilation
#########################################################
#
# The model executor has two primary components:
# 1. preparing the model and sampler inputs
# 2. executing the model and sampler.
# The core idea is to avoid any TPU computation during input preparation. For
# better compilation tracking and increased flexibility, the model execution and
# sampler are divided into several distinct components.
#
# Below are the detailed steps:
#
# Step 1
# It is recommended to avoid TPU operations when preparing the model and sampler
# inputs. CPU tensors can be prepared and transferred to the XLA device using
# cpu_tensor.to(xla_device), which only triggers CPU to TPU transfers and avoids
# compilation.
#
# Step 2
# The TPU execution should be decomposed into subgraphs (4 at the moment):
# 1. the main model
# 2. selecting hidden states for each request
# 3. sampler
# 4. encoder.
# Each subgraph should be decorated in a torch.compile. This is used to make
# sure that we have the same subgraph topology in both dummy_run and
# xecute_model. The results from these subgraphs should either be passed to
# other subgraphs, or transferred from TPU to CPU using xla_tensor.cpu() for
# subsequent processing on the CPU.
#
# Step 3
# The dummy_run should be comprehensive, ensuring all potential input shapes and
# branch predictions are included as subgraph inputs to facilitate
# pre-compilation.
class TPUModelRunner(LoRAModelRunnerMixin):

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        original_parallel_config: Optional[ParallelConfig] = None,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.original_parallel_config = original_parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config
        self.device_config = vllm_config.device_config

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.check_recompilation = envs.VLLM_XLA_CHECK_RECOMPILATION

        # SPMD Related
        self.use_spmd = envs.VLLM_XLA_USE_SPMD
        if self.use_spmd:
            num_devices = xr.global_runtime_device_count()
            mesh_shape = (num_devices, 1)
            device_ids = np.array(range(num_devices))
            self.mesh = xs.Mesh(device_ids, mesh_shape, ('x', 'y'))

        self.enforce_eager = model_config.enforce_eager

        self.num_xla_graphs = 0
        self._update_num_xla_graphs("init")

        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        self._hidden_states_dtype = self.dtype

        self.is_multimodal_model = model_config.is_multimodal_model
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_model_len = model_config.max_model_len
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        # InputBatch needs to work with sampling tensors greater than padding
        # to avoid dynamic shapes. Also, avoid suboptimal alignment.
        self.max_num_reqs = max(scheduler_config.max_num_seqs, MIN_NUM_SEQS)
        self.num_tokens_paddings = _get_token_paddings(
            min_token_size=16,
            max_token_size=scheduler_config.max_num_batched_tokens,
            padding_gap=envs.VLLM_TPU_BUCKET_PADDING_GAP)
        # In case `max_num_tokens < max(num_tokens_paddings)` use the actual
        # padded max value to pre-allocate data structures and pre-compile.
        self.max_num_tokens = self.num_tokens_paddings[-1]

        # Model-related.
        self.num_attn_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        self.num_query_heads = model_config.get_num_attention_heads(
            parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.head_size = model_config.get_head_size()
        self.hidden_size = model_config.get_hidden_size()
        self.vocab_size = model_config.get_vocab_size()

        if self.lora_config is not None:
            self.vocab_size += self.lora_config.lora_extra_vocab_size

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = model_config.uses_mrope
        # TODO: Support M-RoPE (e.g, Qwen2-VL)
        assert not self.uses_mrope, "TPU does not support M-RoPE yet."

        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=model_config,
            scheduler_config=scheduler_config,
            mm_registry=self.mm_registry,
        )
        self.max_num_encoder_input_tokens = encoder_compute_budget
        self.encoder_cache_size = encoder_cache_size

        # Lazy initialization
        self.model: nn.Module  # Set after load_model
        self.kv_caches: list[torch.Tensor] = []
        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: dict[str, dict[int, torch.Tensor]] = {}

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}

        # Initialize input batch early to avoid AttributeError in _update_states
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_size=self.block_size,
        )

        # Cached torch/numpy tensor
        # The pytorch tensor and numpy array share the same buffer.
        # Sometimes the numpy op is faster so we create both.
        self.input_ids_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int32,
                                         device="cpu")

        self.positions_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int32,
                                         device="cpu")
        self.positions_np = self.positions_cpu.numpy()

        self.block_table_cpu = torch.zeros(
            (self.max_num_reqs, self.max_num_blocks_per_req),
            dtype=torch.int32,
            device="cpu")

        self.query_start_loc_cpu = torch.zeros(self.max_num_tokens + 1,
                                               dtype=torch.int32,
                                               device="cpu",
                                               pin_memory=self.pin_memory)
        self.query_start_loc_np = self.query_start_loc_cpu.numpy()

        self.seq_lens_cpu = torch.zeros(self.max_num_tokens,
                                        dtype=torch.int32,
                                        device="cpu",
                                        pin_memory=self.pin_memory)
        self.seq_lens_np = self.seq_lens_cpu.numpy()

        # Range tensor with values [0 .. self.max_num_tokens - 1].
        # Used to initialize positions / context_lens / seq_lens
        # Keep in int64 to avoid overflow with long context
        self.arange_np = np.arange(self.max_num_tokens, dtype=np.int64)
        self.num_reqs_paddings = _get_req_paddings(
            min_req_size=MIN_NUM_SEQS, max_req_size=self.max_num_reqs)

        # tensors for structured decoding
        self.grammar_bitmask_cpu = torch.zeros(
            (self.max_num_reqs, cdiv(self.vocab_size, 32)),
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory)
        self.require_structured_out_cpu = torch.zeros(
            (self.max_num_reqs, 1),
            dtype=torch.bool,
            device="cpu",
            pin_memory=self.pin_memory)
        self.structured_decode_arange = torch.arange(
            0, 32, device="cpu", pin_memory=self.pin_memory)

        # Get maximum number of mm items per modality (batch size).
        self.max_num_mm_items_by_modality = dict()
        if (self.is_multimodal_model and self.max_num_encoder_input_tokens > 0
                and self.encoder_cache_size > 0):
            max_tokens_by_modality_dict = (
                MULTIMODAL_REGISTRY.
                get_max_tokens_per_item_by_nonzero_modality(self.model_config))
            for modality, max_tokens in max_tokens_by_modality_dict.items():
                # Check how many items of this modality can be supported by
                # the encoder budget.
                encoder_budget = min(self.max_num_encoder_input_tokens,
                                     self.encoder_cache_size)

                max_num_mm_items_encoder_budget = cdiv(encoder_budget,
                                                       max_tokens)

                # Check how many items of this modality can be supported by
                # the decoder budget.
                max_mm_items_per_req = self.mm_registry.\
                    get_mm_limits_per_prompt(self.model_config)[modality]

                # NOTE: We do not consider max_num_batched_tokens on purpose
                # because the multimodal embeddings can be generated in advance
                # and chunked prefilled.
                max_num_mm_items_decoder_budget = self.max_num_reqs * \
                    max_mm_items_per_req

                max_num_mm_items = min(max_num_mm_items_encoder_budget,
                                       max_num_mm_items_decoder_budget)
                self.max_num_mm_items_by_modality[modality] = max_num_mm_items

        if not self.use_spmd:
            self.sample_from_logits_func = torch.compile(
                self.sample_from_logits,
                backend="openxla",
                fullgraph=True,
                dynamic=False)
        else:
            self.sample_from_logits_func = self.sample_from_logits

    def _update_num_xla_graphs(self, case_str):
        check_comp = self.check_recompilation and not self.enforce_eager
        if not check_comp:
            return

        total_cached_graphs = xr.get_num_cached_compilation_graph()
        new_compiled_graphs = total_cached_graphs - self.num_xla_graphs
        if new_compiled_graphs == 0:
            return

        logger.info("Add new %d compiled XLA graphs due to %s",
                    new_compiled_graphs, case_str)
        self.num_xla_graphs += new_compiled_graphs

    def _verify_num_xla_graphs(self, case_str):
        check_comp = self.check_recompilation and not self.enforce_eager
        if not check_comp:
            return

        curr_cached_graph = xr.get_num_cached_compilation_graph()
        assert self.num_xla_graphs == curr_cached_graph, (
            "Recompilation after warm up is detected during {}."
            " num_xla_graphs = {} curr_cached_graph = {}".format(
                case_str, self.num_xla_graphs, curr_cached_graph))

    def _update_states(self, scheduler_output: "SchedulerOutput") -> bool:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        Returns:
            True if there is a new/resumed/paused/finished request.
            If False, we can skip copying SamplingMetadata to the GPU.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.encoder_cache.pop(req_id, None)

        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        removed_req_indices: list[int] = []
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Free the cached encoder outputs.
        for req_id, input_id in scheduler_output.free_encoder_input_ids:
            encoder_outputs = self.encoder_cache.get(req_id)
            if encoder_outputs is not None:
                encoder_outputs.pop(input_id, None)
                if not encoder_outputs:
                    self.encoder_cache.pop(req_id, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            assert req_index is not None
            removed_req_indices.append(req_index)

        req_ids_to_add: list[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                generator=None,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        for req_data in scheduler_output.scheduled_cached_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            # Update the cached states.
            req_state.num_computed_tokens = req_data.num_computed_tokens
            if not req_data.resumed_from_preemption:
                # Append the new blocks to the existing block IDs.
                req_state.block_ids.extend(req_data.new_block_ids)
            else:
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = req_data.new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                req_data.num_computed_tokens)
            self.input_batch.block_table.append_row(req_data.new_block_ids,
                                                    req_index)

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            if removed_req_indices:
                # Fill the empty index.
                req_index = removed_req_indices.pop()
            else:
                # Append to the end.
                req_index = None
            self.input_batch.add_request(req_state, req_index)

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

        return len(unscheduled_req_ids) > 0 or len(req_ids_to_add) > 0

    def get_model(self) -> nn.Module:
        return self.model

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        block_size = self.vllm_config.cache_config.block_size
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        for layer_name, attn_module in layers.items():
            if attn_module.attn_type == AttentionType.DECODER:
                if attn_module.sliding_window is not None:
                    kv_cache_spec[layer_name] = SlidingWindowSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=attn_module.dtype,
                        sliding_window=attn_module.sliding_window,
                        use_mla=False,
                    )
                else:
                    kv_cache_spec[layer_name] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=attn_module.dtype,
                        use_mla=False,
                    )
            elif attn_module.attn_type in (AttentionType.ENCODER,
                                           AttentionType.ENCODER_ONLY):
                # encoder-only attention does not need KV cache.
                continue
            elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                raise NotImplementedError
            else:
                raise ValueError(
                    f"Unknown attention type: {attn_module.attn_type}")

        return kv_cache_spec

    def _prepare_inputs(self, scheduler_output: "SchedulerOutput"):
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # Get the number of scheduled tokens for each request.
        num_scheduled_tokens_per_req = []
        max_num_scheduled_tokens_all_reqs = 0
        for req_id in self.input_batch.req_ids[:num_reqs]:
            assert req_id is not None
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_scheduled_tokens_per_req.append(num_tokens)
            max_num_scheduled_tokens_all_reqs = max(
                max_num_scheduled_tokens_all_reqs, num_tokens)
        num_scheduled_tokens_per_req = np.array(num_scheduled_tokens_per_req,
                                                dtype=np.int32)
        assert max_num_scheduled_tokens_all_reqs > 0

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        # For each scheduled token, what are the corresponding req index.
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens_per_req)

        # Get batched arange.
        # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # For each scheduled token, what is its position in corresponding req.
        arange = np.concatenate(
            [self.arange_np[:n] for n in num_scheduled_tokens_per_req])

        # Get positions.
        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:total_num_scheduled_tokens])

        # Calculate the slot mapping.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
        # where K is the max_num_blocks_per_req and the block size is 2.
        # NOTE(woosuk): We can't simply use `token_indices // block_size` here
        # because M (max_model_len) is not necessarily divisible by block_size.
        # req_indices: # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        block_table_indices = (req_indices * self.max_num_blocks_per_req +
                               positions_np // self.block_size)
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        block_table_cpu = self.input_batch.block_table[0].get_cpu_tensor()
        block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()
        block_offsets = positions_np % self.block_size
        np.add(block_numbers * self.block_size,
               block_offsets,
               out=self.input_batch.block_table[0].
               slot_mapping_np[:total_num_scheduled_tokens])

        # Prepare the attention metadata.
        self.query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens_per_req,
                  out=self.query_start_loc_np[1:num_reqs + 1])
        self.query_start_loc_np[num_reqs + 1:] = 1

        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens_per_req)

        # Do the padding and copy the tensors to the TPU.
        padded_total_num_scheduled_tokens = _get_padded_token_len(
            self.num_tokens_paddings, total_num_scheduled_tokens)
        # Zero out to avoid spurious values from prev iteration (last cp chunk)
        self.input_ids_cpu[
            total_num_scheduled_tokens:padded_total_num_scheduled_tokens] = 0
        self.input_ids = self.input_ids_cpu[:
                                            padded_total_num_scheduled_tokens].to(
                                                self.device)
        self.position_ids = self.positions_cpu[:
                                               padded_total_num_scheduled_tokens].to(
                                                   self.device)
        self.input_batch.block_table[0].slot_mapping_cpu[
            total_num_scheduled_tokens:] = _PAD_SLOT_ID
        slot_mapping = (
            self.input_batch.block_table[0].
            slot_mapping_cpu[:padded_total_num_scheduled_tokens].to(
                self.device))
        block_tables = self.block_table_cpu[:self.max_num_reqs]
        block_tables[:num_reqs, :self.max_num_blocks_per_req] = (
            self.input_batch.block_table[0].get_cpu_tensor()[:num_reqs])
        block_tables = block_tables.to(self.device)
        query_start_loc = self.query_start_loc_cpu[:self.max_num_reqs + 1].to(
            self.device)
        seq_lens = self.seq_lens_cpu[:self.max_num_reqs].to(self.device)

        if self.lora_config is not None:
            # We need to respect padding when activating LoRA adapters
            padded_num_scheduled_tokens_per_req = np.copy(
                num_scheduled_tokens_per_req
            )  # Copying to avoid accidental state corruption bugs
            padded_num_scheduled_tokens_per_req[-1] += \
                padded_total_num_scheduled_tokens - total_num_scheduled_tokens

            self.set_active_loras(self.input_batch,
                                  padded_num_scheduled_tokens_per_req)

        attn_metadata = PallasMetadata(
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=seq_lens,
            query_start_loc=query_start_loc,
            num_seqs=torch.tensor([num_reqs],
                                  dtype=torch.int32,
                                  device=self.device),
        )
        # NOTE(woosuk): Due to chunked prefills, there can be at most 1 partial
        # request in the batch. While we should not sample any token from this
        # partial request, we do so for simplicity. We will ignore the sampled
        # token from the partial request.
        # TODO: Support prompt logprobs.
        padded_num_reqs = _get_padded_num_reqs_with_upper_limit(
            num_reqs, self.max_num_reqs)
        # Indices at which we sample (positions of last token in the sequence).
        # Padded to avoid recompiling when `num_reqs` varies.
        logits_indices = self.query_start_loc_cpu[1:padded_num_reqs + 1] - 1
        logits_indices = logits_indices.to(self.device)

        if self.lora_config is not None:
            # We need to respect padding when activating LoRA adapters
            padded_num_scheduled_tokens_per_req = np.copy(
                num_scheduled_tokens_per_req
            )  # Copying to avoid accidental state corruption bugs
            padded_num_scheduled_tokens_per_req[-1] += \
                padded_total_num_scheduled_tokens - total_num_scheduled_tokens

            self.set_active_loras(self.input_batch,
                                  padded_num_scheduled_tokens_per_req)

        layer_names = get_layers_from_vllm_config(self.vllm_config,
                                                  Attention).keys()
        per_layer_attn_metadata = {
            layer_name: attn_metadata
            for layer_name in layer_names
        }
        return per_layer_attn_metadata, logits_indices, padded_num_reqs

    def _scatter_placeholders(
        self,
        embeds: torch.Tensor,
        is_embed: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if is_embed is None:
            return embeds

        placeholders = embeds.new_full(
            (is_embed.shape[0], embeds.shape[-1]),
            fill_value=torch.nan,
        )
        placeholders[is_embed] = embeds
        return placeholders

    def _gather_placeholders(
        self,
        placeholders: torch.Tensor,
        is_embed: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if is_embed is None:
            return placeholders

        return placeholders[is_embed]

    def _execute_mm_encoder(self, scheduler_output: "SchedulerOutput"):
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return

        # Batch the multi-modal inputs.
        mm_inputs = list[MultiModalKwargs]()
        req_ids_pos = list[tuple[str, int, PlaceholderRange]]()
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]

            for mm_input_id in encoder_input_ids:
                mm_inputs.append(req_state.mm_inputs[mm_input_id])
                req_ids_pos.append(
                    (req_id, mm_input_id, req_state.mm_positions[mm_input_id]))

        # Batch mm inputs as much as we can: if a request in the batch has
        # multiple modalities or a different modality than the previous one,
        # we process it separately to preserve item order.
        # FIXME(ywang96): This is a hacky way to deal with multiple modalities
        # in the same batch while still being able to benefit from batching
        # multimodal inputs. The proper solution should be reordering the
        # encoder outputs.
        grouped_mm_inputs_list = group_mm_inputs_by_modality(mm_inputs)

        encoder_outputs = []
        for grouped_mm_inputs in grouped_mm_inputs_list:
            batched_mm_inputs = MultiModalKwargs.batch(grouped_mm_inputs)
            batched_mm_inputs = MultiModalKwargs.as_kwargs(
                batched_mm_inputs,
                dtype=self.model_config.dtype,
                device=self.device,
            )

            # Run the encoder.
            # `curr_group_outputs` is either of the following:
            # 1. A tensor of shape (num_items, feature_size, hidden_size)
            # in case feature_size is fixed across all multimodal items.
            # 2. A list or tuple (length: num_items) of tensors, each of shape
            # (feature_size, hidden_size) in case the feature size is dynamic
            # depending on the input multimodal items.
            xm.mark_step()
            curr_group_outputs = self.model.get_multimodal_embeddings(
                **batched_mm_inputs)
            xm.mark_step()

            sanity_check_mm_encoder_outputs(
                curr_group_outputs,
                expected_num_items=len(grouped_mm_inputs),
            )

            if isinstance(curr_group_outputs, torch.Tensor):
                encoder_outputs.append(curr_group_outputs)
            else:
                assert isinstance(curr_group_outputs, (list, tuple))
                for output in curr_group_outputs:
                    encoder_outputs.append(output)

        # Cache the encoder outputs.
        # NOTE (NickLucche) here we diverge from logic in other runners, as we
        # assume to only have whole mm items to process. Hence we avoid the
        # intrinsic dynamism that `scatter_mm_placeholders` introduces.
        for (req_id, input_id, pos_info), output in zip(
                req_ids_pos,
                encoder_outputs,
        ):
            if req_id not in self.encoder_cache:
                self.encoder_cache[req_id] = {}
            assert pos_info.is_embed is None, "Expected all positions to be"\
                " contiguous and embeddings."
            self.encoder_cache[req_id][input_id] = output

    def _gather_mm_embeddings(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> list[torch.Tensor]:
        mm_embeds: list[torch.Tensor] = []
        for req_id in self.input_batch.req_ids:
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = req_state.num_computed_tokens
            mm_positions = req_state.mm_positions
            # TODO unroll loop and assume/enforce --disable_chunked_mm_input
            # NOTE (NickLucche) here we diverge from logic in other runners, as
            # we assume to only have whole mm items to process. Hence we avoid
            # the intrinsic dynamism that `gather_mm_placeholders` introduces.
            for i, pos_info in enumerate(mm_positions):
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                # The encoder output is needed if the two ranges overlap:
                # [num_computed_tokens,
                #  num_computed_tokens + num_scheduled_tokens) and
                # [start_pos, start_pos + num_encoder_tokens)
                if start_pos >= num_computed_tokens + num_scheduled_tokens:
                    # The encoder output is not needed in this step.
                    break
                if start_pos + num_encoder_tokens <= num_computed_tokens:
                    # The encoder output is already processed and stored
                    # in the decoder's KV cache.
                    continue

                assert req_id in self.encoder_cache
                assert i in self.encoder_cache[req_id]
                assert pos_info.is_embed is None, "Expected all positions to"\
                " be contiguous and embeddings."
                encoder_output = self.encoder_cache[req_id][i]
                mm_embeds.append(encoder_output)
        return mm_embeds

    def _get_model_inputs(self, input_ids: torch.Tensor,
                          mm_embeds: list[torch.Tensor]):
        if self.is_multimodal_model:
            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            if mm_embeds:
                inputs_embeds = self.model.get_input_embeddings(
                    input_ids, mm_embeds)
            else:
                inputs_embeds = self.model.get_input_embeddings(input_ids)
            return None, inputs_embeds
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            return input_ids, None

    @torch.no_grad()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> ModelRunnerOutput:
        # Update cached state
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            # Return empty ModelRunnerOutput if there's no work to do.
            return EMPTY_MODEL_RUNNER_OUTPUT

        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)
        else:
            mm_embeds = []
        xm.mark_step()
        # Prepare inputs
        attn_metadata, logits_indices, padded_num_reqs = self._prepare_inputs(
            scheduler_output)
        input_ids, inputs_embeds = self._get_model_inputs(
            self.input_ids, mm_embeds)
        xm.mark_step()
        num_reqs = self.input_batch.num_reqs
        # Run the decoder
        with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=scheduler_output.total_num_scheduled_tokens):
            hidden_states = self.model(
                input_ids=input_ids,
                positions=self.position_ids,
                inputs_embeds=inputs_embeds,
            )
        hidden_states = self.select_hidden_states(hidden_states,
                                                  logits_indices)
        logits = self.compute_logits(hidden_states)
        tpu_sampling_metadata = TPUSupportedSamplingMetadata.\
            from_input_batch(self.input_batch, padded_num_reqs, self.device)
        if scheduler_output.grammar_bitmask is not None:
            require_struct_decoding, grammar_bitmask_padded, arange = \
                self.prepare_structured_decoding_input(logits, scheduler_output)
            logits = self.structured_decode(require_struct_decoding,
                                            grammar_bitmask_padded, logits,
                                            arange)
        selected_token_ids = self.sample_from_logits_func(
            logits, tpu_sampling_metadata)
        # NOTE (NickLucche) Use the original logits (before any penalties or
        # temperature scaling) for the top-k logprobs. We can't enforce it due
        # to recompilations outside torch.compiled code, so just make sure
        # `sample_from_logits` does not modify the logits in-place.
        logprobs = self.gather_logprobs(logits, selected_token_ids) \
            if tpu_sampling_metadata.logprobs else None

        # Remove padding on cpu and keep dynamic op outside of xla graph.
        selected_token_ids = selected_token_ids.cpu()[:num_reqs]
        logprobs_lists = logprobs.tolists() \
            if tpu_sampling_metadata.logprobs else None

        # Update the cache state concurrently. Code above will not block until
        # we use `selected_token_ids`. Add mark_step if post-processing changes
        request_seq_lens: list[tuple[int, CachedRequestState, int]] = []
        discard_sampled_tokens_req_indices = []
        for i, req_id in zip(range(num_reqs), self.input_batch.req_ids):
            assert req_id is not None
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            if seq_len >= req_state.num_tokens:
                request_seq_lens.append((i, req_state, seq_len))
            else:
                # Ignore the sampled token from the partial request.
                # Rewind the generator state as if the token was not sampled.
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    # This relies on cuda-specific torch-internal impl details
                    generator.set_offset(generator.get_offset() - 4)

                # Record the index of the request that should not be sampled,
                # so that we could clear the sampled tokens before returning.
                discard_sampled_tokens_req_indices.append(i)

        assert all(
            req_id is not None for req_id in
            self.input_batch.req_ids[:num_reqs]), "req_ids contains None"
        req_ids = cast(list[str], self.input_batch.req_ids[:num_reqs])

        prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]] = {}
        for req_id in self.input_batch.req_ids[:num_reqs]:
            prompt_logprobs_dict[req_id] = None

        max_gen_len = selected_token_ids.shape[-1]
        if max_gen_len == 1:
            valid_sampled_token_ids = selected_token_ids.tolist()

            # Mask out the sampled tokens that should not be sampled.
            # TODO: Keep in sync with gpu_model_runner.py, in particular
            #       the "else" case here
            for i in discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[i].clear()

            # Append sampled tokens
            for i, req_state, seq_len in request_seq_lens:
                token_id = valid_sampled_token_ids[i][0]
                self.input_batch.token_ids_cpu[i, seq_len] = token_id
                req_state.output_token_ids.append(token_id)
                self.input_batch.num_tokens[i] += 1

        else:
            valid_mask = selected_token_ids != INVALID_TOKEN_ID
            gen_lens = valid_mask.sum(dim=1).tolist()
            valid_sampled_token_ids = [
                seq.tolist()
                for seq in selected_token_ids[valid_mask].split(gen_lens)
            ]
            self.input_batch.num_tokens[:num_reqs] += gen_lens
            for i, req_state, seq_len in request_seq_lens:
                target_slice = slice(seq_len - gen_lens[i] + 1, seq_len + 1)
                self.input_batch.token_ids_cpu[
                    i, target_slice] = valid_sampled_token_ids[i]
                req_state.output_token_ids.extend(valid_sampled_token_ids[i])

        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=None,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
        )

        # Check there are no new graphs compiled - all the graphs should be
        # captured and compiled during warm up.
        self._verify_num_xla_graphs("execute_model")

        return model_runner_output

    def load_model(self) -> None:
        self.device = self.device_config.device

        # NOTE(woosuk): While the executor assigns the TP ranks to the worker
        # process, the ranks can be different from the ranks internally assigned
        # by the xm runtime. Therefore, there is a mismatch in the rank
        # assignment between the gloo (cpu) runtime and the xm (tpu) runtime.
        # This is not a problem in linear layers because all-reduce is
        # rank-agnostic. However, it matters for all-gather as the ranks
        # determine the order of concatenating the output tensors.
        # As a workaround, we use the xm's rank assignment only when loading
        # the embedding weights.
        xm_tp_rank = xr.global_ordinal()
        with patch(
                "vllm.model_executor.layers.vocab_parallel_embedding."
                "get_tensor_model_parallel_rank",
                return_value=xm_tp_rank):
            if self.use_spmd:
                tpu_loader = TPUModelLoader(
                    load_config=self.vllm_config.load_config)
                model = tpu_loader.load_model(
                    vllm_config=self.vllm_config,
                    model_config=self.vllm_config.model_config,
                    mesh=self.mesh)
            else:
                # model = get_model(vllm_config=self.vllm_config)
                model_loader = get_model_loader(self.load_config)
                if not hasattr(self, "model"):
                    logger.info("Loading model from scratch...")
                    model = model_loader.load_model(
                        vllm_config=self.vllm_config,
                        model_config=self.model_config)
                else:
                    logger.info("Model was already initialized. \
                            Loading weights inplace...")
                    model_loader.load_weights(self.model,
                                              model_config=self.model_config)
        if self.lora_config is not None:
            model = self.load_lora_model(model, self.model_config,
                                         self.scheduler_config,
                                         self.lora_config, self.device)
            replace_set_lora(model)

        # Sync all pending XLA execution during model initialization and weight
        # loading.
        xm.mark_step()
        xm.wait_device_ops()
        if not hasattr(self, "model"):
            self.model = model
        self.sampler = TPUSampler()

    @torch.no_grad()
    def _dummy_run(self, num_tokens: int) -> None:
        if self.is_multimodal_model:
            input_ids = None
            inputs_embeds = torch.zeros((num_tokens, self.hidden_size),
                                        dtype=self.dtype,
                                        device=self.device)
        else:
            input_ids = torch.zeros((num_tokens),
                                    dtype=torch.int32).to(self.device)
            inputs_embeds = None
        actual_num_reqs = min(num_tokens, self.max_num_reqs)
        position_ids = torch.zeros(num_tokens,
                                   dtype=torch.int32).to(self.device)
        slot_mapping = torch.zeros(num_tokens,
                                   dtype=torch.int64).to(self.device)
        block_tables = torch.zeros(
            (self.max_num_reqs, self.block_table_cpu.shape[1]),
            dtype=torch.int32).to(self.device)
        query_lens = [1] * self.max_num_reqs
        query_start_loc = torch.cumsum(torch.tensor([0] + query_lens,
                                                    dtype=torch.int32),
                                       dim=0,
                                       dtype=torch.int32).to(self.device)
        context_lens = torch.ones((self.max_num_reqs, ),
                                  dtype=torch.int32).to(self.device)
        num_seqs = torch.tensor([actual_num_reqs],
                                dtype=torch.int32).to(self.device)
        attn_metadata = PallasMetadata(
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            query_start_loc=query_start_loc,
            num_seqs=num_seqs,
        )

        if self.is_multimodal_model:
            torch._dynamo.mark_dynamic(inputs_embeds, 0)
        else:
            torch._dynamo.mark_dynamic(input_ids, 0)
        torch._dynamo.mark_dynamic(position_ids, 0)
        torch._dynamo.mark_dynamic(attn_metadata.slot_mapping, 0)

        layer_names = get_layers_from_vllm_config(self.vllm_config,
                                                  Attention).keys()
        per_layer_attn_metadata = {
            layer_name: attn_metadata
            for layer_name in layer_names
        }

        with self.maybe_select_dummy_loras(
                self.lora_config,
                np.array([num_tokens], dtype=np.int32)), set_forward_context(
                    per_layer_attn_metadata, self.vllm_config, 0):
            out = self.model(input_ids=input_ids,
                             positions=position_ids,
                             inputs_embeds=inputs_embeds)
        self._hidden_states_dtype = out.dtype

    def _set_active_loras(self, prompt_lora_mapping, token_lora_mapping,
                          lora_requests) -> None:
        xm.mark_step()  # Captures input updates
        super()._set_active_loras(prompt_lora_mapping, token_lora_mapping,
                                  lora_requests)
        xm.mark_step()  # Captures metadata updates

    def _precompile_mm_encoder(self) -> None:
        # Pre-compile MM encoder for all supported data modalities.
        hf_config = self.vllm_config.model_config.hf_config
        for mode, max_items_by_mode in \
            self.max_num_mm_items_by_modality.items():
            logger.info(
                "Compiling Multimodal %s Encoder with different input"
                " shapes.", mode)
            start = time.perf_counter()
            # No padding for MM encoder just yet.
            for num_items in range(1, max_items_by_mode + 1):
                logger.info("  -- mode: %s items: %d", mode, num_items)
                batched_dummy_mm_inputs = self._get_mm_dummy_batch(
                    mode, num_items)
                # Run multimodal encoder.
                xm.mark_step()
                mm_embeds = self.model.\
                    get_multimodal_embeddings(**batched_dummy_mm_inputs)
                xm.mark_step()
                num_patches = mm_embeds[0].shape[0]
                items_size = num_patches * num_items

                # NOTE (NickLucche) pre-compile `get_input_embeddings` when mm
                # embeddings are present. We assume `--disable-mm-chunked`,
                # hence only whole items can be scheduled. This implies we just
                # need to compile when `num_items` fit the (padded) `input_ids`
                for num_tokens in self.num_tokens_paddings:
                    if num_tokens >= items_size:
                        # XLA Workaround: if torch.zeros(..device) is used, XLA
                        # compiles a scalar+expansion op, which won't match
                        # the graph generated at runtime. CPU->TPU must be used
                        placeholders_ids = torch.zeros(num_tokens,
                                                       dtype=torch.int32,
                                                       device="cpu")
                        # Align placeholders and actual num mm_embeddings.
                        placeholders_ids[:items_size] = \
                            hf_config.image_token_index

                        placeholders_ids = placeholders_ids.to(self.device)
                        # Assign outputs or the graph will be cut short.
                        a, b = self._get_model_inputs(placeholders_ids,
                                                      [mm_embeds])
                        assert a is None
                        xm.mark_step()

            # Pre-compile `get_input_embeddings` when mm_embeddings are not
            # present. Chunk is only made of text, no mm_placeholders.
            for num_tokens in self.num_tokens_paddings:
                placeholders_ids = torch.zeros(num_tokens,
                                               dtype=torch.int32,
                                               device="cpu")
                placeholders_ids = placeholders_ids.to(self.device)
                a, b = self._get_model_inputs(placeholders_ids, [])
                assert a is None
                xm.mark_step()

            xm.wait_device_ops()
            end = time.perf_counter()
            logger.info(
                "Multimodal %s Encoder compilation finished in in %.2f "
                "[secs].", mode, end - start)

    def _precompile_backbone(self) -> None:
        logger.info("Compiling the model with different input shapes.")
        start = time.perf_counter()
        for num_tokens in self.num_tokens_paddings:
            logger.info("  -- num_tokens: %d", num_tokens)
            self._dummy_run(num_tokens)
        xm.wait_device_ops()
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
        self._update_num_xla_graphs("model backbone")

    def _precompile_select_hidden_states(self) -> None:
        # Compile hidden state selection function for bucketed
        # n_tokens x max_num_reqs. Graph is really small so this is fine.
        logger.info(
            "Compiling select_hidden_states with different input shapes.")
        start = time.perf_counter()
        hsize = self.model_config.get_hidden_size()
        for num_tokens in self.num_tokens_paddings:
            dummy_hidden = torch.zeros((num_tokens, hsize),
                                       device=self.device,
                                       dtype=self._hidden_states_dtype)
            torch._dynamo.mark_dynamic(dummy_hidden, 0)
            for num_reqs in self.num_reqs_paddings:
                indices = torch.zeros(num_reqs,
                                      dtype=torch.int32,
                                      device=self.device)
                torch._dynamo.mark_dynamic(indices, 0)
                self.select_hidden_states(dummy_hidden, indices)
                logger.info("  -- num_tokens: %d, num_seqs: %d", num_tokens,
                            num_reqs)
                # Requests can't be more than tokens. But do compile for the
                # next bigger value in case num_tokens uses bucketed padding.
                if num_reqs >= min(num_tokens, self.max_num_reqs):
                    break
        xm.wait_device_ops()
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
        self._update_num_xla_graphs("select_hidden_states")

    def _precompile_compute_logits(self) -> None:
        logger.info("Compiling compute_logits with different input shapes.")
        start = time.perf_counter()
        hsize = self.model_config.get_hidden_size()
        for num_reqs in self.num_reqs_paddings:
            dummy_hidden = torch.zeros((num_reqs, hsize),
                                       device=self.device,
                                       dtype=self._hidden_states_dtype)
            torch._dynamo.mark_dynamic(dummy_hidden, 0)
            self.compute_logits(dummy_hidden)
            logger.info("  -- num_seqs: %d", num_reqs)
        xm.wait_device_ops()
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
        self._update_num_xla_graphs("compute_logits")

    def _precompile_structured_decoding(self) -> None:
        logger.info(
            "Compiling structured_decoding with different input shapes.")
        start = time.perf_counter()
        for num_reqs in self.num_reqs_paddings:
            dummy_logits = torch.zeros((num_reqs, self.vocab_size),
                                       device=self.device,
                                       dtype=self._hidden_states_dtype)
            dummy_require_struct_decoding = \
                self.require_structured_out_cpu[:num_reqs].to(self.device)
            dummy_grammar_bitmask = \
                self.grammar_bitmask_cpu[:num_reqs].to(self.device)
            # The first dimension of the above 3 dummy tensors cannot be
            # mark_dynamic because some operations in structured_decode require
            # them to be static.
            arange = self.structured_decode_arange.to(self.device)
            self.structured_decode(dummy_require_struct_decoding,
                                   dummy_grammar_bitmask, dummy_logits, arange)
            logger.info("  -- num_seqs: %d", num_reqs)
        xm.wait_device_ops()
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
        self._update_num_xla_graphs("structured_decoding")

    def _precompile_sample_from_logits(self) -> None:
        logger.info(
            "Compiling sample_from_logits with different input shapes.")
        start = time.perf_counter()
        for num_reqs in self.num_reqs_paddings:
            dummy_logits = torch.zeros((num_reqs, self.vocab_size),
                                       device=self.device,
                                       dtype=self._hidden_states_dtype)
            # The first dimension of dummy_logits cannot be mark_dynamic
            # because some operations in the sampler require it to be static.
            for all_greedy in [False, True]:
                generate_params_if_all_greedy = not all_greedy
                sampling_metadata = (
                    TPUSupportedSamplingMetadata.from_input_batch(
                        self.input_batch,
                        num_reqs,
                        self.device,
                        generate_params_if_all_greedy,
                    ))
                sampling_metadata.all_greedy = all_greedy
                with self.maybe_select_dummy_loras(
                        self.lora_config, np.array([num_reqs],
                                                   dtype=np.int32)):
                    self.sample_from_logits_func(dummy_logits,
                                                 sampling_metadata)
            logger.info("  -- num_seqs: %d", num_reqs)
        xm.wait_device_ops()
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
        self._update_num_xla_graphs("sample_from_logits")

    def _precompile_gather_logprobs(self) -> None:
        logger.info("Compiling gather_logprobs with different input shapes.")
        start = time.perf_counter()
        for num_reqs in self.num_reqs_paddings:
            dummy_logits = torch.zeros((num_reqs, self.vocab_size),
                                       device=self.device,
                                       dtype=self._hidden_states_dtype)
            dummy_tokens = torch.zeros((num_reqs, 1),
                                       dtype=torch.int64).to(self.device)
            with self.maybe_select_dummy_loras(
                    self.lora_config, np.array([num_reqs], dtype=np.int32)):
                self.gather_logprobs(dummy_logits, dummy_tokens)
            logger.info("  -- num_seqs: %d", num_reqs)
        xm.wait_device_ops()
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
        self._update_num_xla_graphs("gather_logprobs")

    def capture_model(self) -> None:
        """
        Precompile all the subgraphs with possible input shapes.
        """
        with self.maybe_setup_dummy_loras(self.lora_config):
            self._precompile_mm_encoder()
            self._precompile_backbone()
            self._precompile_select_hidden_states()
            self._precompile_compute_logits()
            self._precompile_structured_decoding()
            self._precompile_sample_from_logits()
            self._precompile_gather_logprobs()

    def profile_run(
        self,
        num_tokens: int,
    ) -> None:
        # Profile with multimodal encoder & encoder cache.
        # TODO: handle encoder-decoder models once we support them.
        if (self.is_multimodal_model and self.max_num_encoder_input_tokens > 0
                and self.encoder_cache_size > 0):

            # NOTE: Currently model is profiled with a single non-text
            # modality with the max possible input tokens even when
            # it supports multiple.
            dummy_data_modality, max_num_mm_items = max(
                self.max_num_mm_items_by_modality.items(), key=lambda t: t[1])

            encoder_budget = min(self.max_num_encoder_input_tokens,
                                 self.encoder_cache_size)

            logger.info(
                "Encoder cache will be initialized with a budget of %d tokens,"
                " and profiled with %s %s items of the maximum feature size.",
                encoder_budget, max_num_mm_items, dummy_data_modality)

            # Create dummy batch of multimodal inputs.
            batched_dummy_mm_inputs = self._get_mm_dummy_batch(
                dummy_data_modality, max_num_mm_items)

            # Run multimodal encoder.
            # Isolate encoder graph from post-processing to minimize
            # impact of recompilation until it's fixed.
            start = time.perf_counter()
            xm.mark_step()
            dummy_encoder_outputs = self.model.get_multimodal_embeddings(
                **batched_dummy_mm_inputs)
            xm.mark_step()
            xm.wait_device_ops()
            end = time.perf_counter()
            logger.info(
                "Multimodal Encoder profiling finished in in %.2f [secs].",
                end - start)

            assert len(dummy_encoder_outputs) == max_num_mm_items, (
                "Expected dimension 0 of encoder outputs to match the number "
                f"of multimodal data items: {max_num_mm_items}, got "
                f"{len(dummy_encoder_outputs)=} instead. This is most likely "
                "due to the 'get_multimodal_embeddings' method of the model "
                "not implemented correctly.")

            # Cache the dummy encoder outputs.
            self.encoder_cache["tmp"] = dict(enumerate(dummy_encoder_outputs))

        # Trigger compilation for general shape.
        self._dummy_run(num_tokens)

        xm.mark_step()
        xm.wait_device_ops()
        self.encoder_cache.clear()
        gc.collect()

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        if len(kv_cache_config.kv_cache_groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not "
                "supported yet.")

        if kv_cache_config.kv_cache_groups[
                0].kv_cache_spec.block_size != self.block_size:
            self.input_batch = InputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=self.max_model_len,
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_size=kv_cache_config.kv_cache_groups[0].kv_cache_spec.
                block_size,
            )
        # Verify dtype compatibility between block_table_cpu and input_batch
        assert self.block_table_cpu.dtype == self.input_batch.block_table[
            0].get_cpu_tensor().dtype

        kv_caches: dict[str, torch.Tensor] = {}

        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            for layer_name in kv_cache_group.layer_names:
                tensor_config = kv_cache_config.tensors[layer_name]
                assert tensor_config.size % kv_cache_spec.page_size_bytes == 0
                num_blocks = tensor_config.size // kv_cache_spec.page_size_bytes
                if isinstance(kv_cache_spec, AttentionSpec):
                    if self.use_spmd:
                        num_kv_heads = kv_cache_spec.num_kv_heads
                        assert self.original_parallel_config is not None
                        tp_size = \
                            self.original_parallel_config.tensor_parallel_size
                        # TODO: Handle kv cache duplication under SPMD mode.
                        assert num_kv_heads % tp_size == 0, (
                            f"num_kv_heads {num_kv_heads} must be divisible by "
                            f"tp_size {tp_size} under SPMD mode")
                    kv_cache_shape = PallasAttentionBackend.get_kv_cache_shape(
                        num_blocks, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype

                    tpu_kv_cache = torch.zeros(kv_cache_shape,
                                               dtype=dtype).to(self.device)

                    kv_caches[layer_name] = tpu_kv_cache
                else:
                    raise NotImplementedError

        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches)

        if self.use_spmd:
            # Shard KV Cache
            for cache in self.kv_caches:
                xs.mark_sharding(cache, self.mesh, (None, 'x', None, None))

    def reset_dynamo_cache(self):
        if self.is_multimodal_model:
            compiled_model = self.model.get_language_model().model
        else:
            compiled_model = self.model.model
        if isinstance(compiled_model, TorchCompileWrapperWithCustomDispatcher):
            logger.info("Clear dynamo cache and cached dynamo bytecode.")
            torch._dynamo.eval_frame.remove_from_cache(
                compiled_model.original_code_object)
            compiled_model.compiled_codes.clear()

    @torch.compile(backend="openxla", fullgraph=True, dynamic=False)
    def select_hidden_states(self, hidden_states, indices_do_sample):
        return hidden_states[indices_do_sample]

    @torch.compile(backend="openxla", fullgraph=True, dynamic=False)
    def compute_logits(self,
                       sample_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.compute_logits(sample_hidden_states, None)

    # TODO: Under SPMD mode, sample_from_logits has correctness issue.
    #       Re-enable the torch.compile once the issue is fixed in torchxla.
    # @torch.compile(backend="openxla", fullgraph=True, dynamic=False)
    def sample_from_logits(
            self, logits: torch.Tensor,
            sampling_metadata: TPUSupportedSamplingMetadata) -> torch.Tensor:
        """
        Sample with xla-friendly function. This function is to be traced 
        separately from `forward` for lighter compilation overhead.
        """
        if sampling_metadata.all_greedy:
            out_tokens = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            out_tokens = self.sampler(logits,
                                      sampling_metadata).sampled_token_ids
        return out_tokens

    @torch.compile(backend="openxla", fullgraph=True, dynamic=False)
    def gather_logprobs(self, logits: torch.Tensor,
                        sampled_tokens: torch.Tensor) -> LogprobsTensors:
        """
        Gather the top_logprobs with corresponding tokens. Use a fixed number
        of logprobs as an alternative to having multiple pre-compiled graphs.
        Select the number of logprobs actually demanded by each request on CPU.
        """
        logprobs = self.sampler.compute_logprobs(logits)
        return self.sampler.gather_logprobs(
            logprobs,
            self.model_config.max_logprobs,
            token_ids=sampled_tokens.squeeze(-1))

    @torch.compile(backend="openxla", fullgraph=True, dynamic=False)
    def structured_decode(self, require_struct_decoding: torch.Tensor,
                          grammar_bitmask: torch.Tensor, logits: torch.Tensor,
                          arange: torch.Tensor) -> torch.Tensor:
        return torch.where(
            require_struct_decoding,
            self.apply_grammar_bitmask(logits, grammar_bitmask, arange),
            logits)

    def apply_grammar_bitmask(self, logits: torch.Tensor,
                              grammar_bitmask: torch.Tensor,
                              arange: torch.Tensor):
        assert (logits.shape[0] == grammar_bitmask.shape[0])
        logits_cloned = logits.clone()
        for i in range(logits.shape[0]):
            unpacked_bitmask = (torch.bitwise_right_shift(
                grammar_bitmask[i][:, None], arange[None, :]) & 1) == 0
            unpacked_bitmask = unpacked_bitmask.reshape(-1)[:self.vocab_size]
            logits_cloned[i] = logits_cloned[i].masked_fill(
                unpacked_bitmask, -float("inf"))
        return logits_cloned

    def get_multimodal_embeddings(self, *args, **kwargs):
        return self.model.get_multimodal_embeddings(*args, **kwargs)

    def get_input_embeddings(self, *args, **kwargs):
        return self.model.get_input_embeddings(*args, **kwargs)

    def prepare_structured_decoding_input(
        self, logits: torch.Tensor, scheduler_output: "SchedulerOutput"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grammar_bitmask = scheduler_output.grammar_bitmask
        assert grammar_bitmask is not None
        num_reqs, _ = logits.shape

        # Reset pre-allocated tensors
        self.grammar_bitmask_cpu.zero_()
        self.require_structured_out_cpu.zero_()

        # We receive the structured output bitmask from the scheduler, but the
        # indices of the requests in the batch may not match the indices of
        # the bitmask since the scheduler doesn't know how the tpu runner is
        # ordering the requests in the batch. We need to match the order of
        # bitmask with the order of requests
        struct_out_indices: list[int] = []
        mask_indices: list[int] = []
        for req_id in self.input_batch.req_ids:
            mask_index = scheduler_output.structured_output_request_ids.get(
                req_id)
            if mask_index is None:
                continue
            batch_index = self.input_batch.req_id_to_index[req_id]
            struct_out_indices.append(batch_index)
            mask_indices.append(mask_index)
        self.grammar_bitmask_cpu[struct_out_indices] = torch.from_numpy(
            grammar_bitmask[mask_indices])
        # It's not guaranteed that all requests in this batch require
        # structured output, so create a bool tensor to represent
        # the requests that need structured output.
        struct_out_indices = torch.tensor(struct_out_indices, dtype=torch.long)
        self.require_structured_out_cpu[struct_out_indices] = True
        return self.require_structured_out_cpu[:num_reqs].to(logits.device), \
            self.grammar_bitmask_cpu[:num_reqs].to(logits.device), \
            self.structured_decode_arange.to(logits.device)

    def _get_mm_dummy_batch(self, modality: str,
                            batch_size: int) -> BatchedTensorInputs:
        # Dummy data for pre-compiling multimodal models.
        dummy_request_data = self.mm_registry.get_decoder_dummy_data(
            model_config=self.model_config,
            seq_len=self.max_num_tokens,
        )
        dummy_mm_data = dummy_request_data.multi_modal_data

        # Dummy data definition in V0 may contain multiple multimodal items
        # (e.g, multiple images) for a single request, therefore here we
        # always replicate first item by max_num_mm_items times since in V1
        # they are scheduled to be processed separately.
        assert isinstance(dummy_mm_data, MultiModalKwargs), (
            "Expected dummy multimodal data to be of type "
            f"MultiModalKwargs, got {type(dummy_mm_data)=} instead. "
            "This is most likely due to the model not having a merged "
            "processor.")

        # When models have a merged processor, their dummy data is
        # already batched `MultiModalKwargs`, therefore we take the first
        # `MultiModalKwargsItem` from the desired modality to profile on.
        dummy_mm_item = dummy_mm_data.get_item(modality=modality, item_index=0)
        dummy_mm_kwargs = MultiModalKwargs.from_items([dummy_mm_item])

        batched_dummy_mm_inputs = MultiModalKwargs.batch([dummy_mm_kwargs] *
                                                         batch_size)
        return MultiModalKwargs.as_kwargs(
            batched_dummy_mm_inputs,
            dtype=self.model_config.dtype,
            device=self.device,
        )


def _get_req_paddings(min_req_size: int, max_req_size: int) -> list[int]:
    logger.info("Preparing request paddings:")
    # assert min_req_size is power of 2
    assert (min_req_size & (min_req_size - 1) == 0) and min_req_size > 0
    paddings: list = []
    num = max(MIN_NUM_SEQS, min_req_size)
    while num <= max_req_size and (len(paddings) == 0 or paddings[-1] != num):
        paddings.append(num)
        logger.info("    %d", num)
        num = _get_padded_num_reqs_with_upper_limit(num + 1, max_req_size)
    return paddings


def _get_padded_num_reqs_with_upper_limit(x: int, upper_limit: int) -> int:
    res = MIN_NUM_SEQS if x <= MIN_NUM_SEQS else 1 << (x - 1).bit_length()
    return min(res, upper_limit)


def _get_token_paddings(min_token_size: int, max_token_size: int,
                        padding_gap: int) -> list[int]:
    """Generate a list of padding size, starting from min_token_size, 
    ending with a number that can cover max_token_size

    If padding_gap == 0 then:
        increase 2X each time (exponential)
    else:
        first increase the size to twice,
        then increase the padding size by padding_gap.
    """
    # assert min_token_size is power of 2
    assert (min_token_size & (min_token_size - 1) == 0) and min_token_size > 0
    paddings = []
    num = min_token_size

    if padding_gap == 0:
        logger.info("Using exponential token paddings:")
        while True:
            logger.info("    %d", num)
            paddings.append(num)
            if num >= max_token_size:
                break
            num *= 2
    else:
        logger.info("Using incremental token paddings:")
        while num <= padding_gap:
            logger.info("    %d", num)
            paddings.append(num)
            num *= 2
        num //= 2
        while num < max_token_size:
            num += padding_gap
            logger.info("    %d", num)
            paddings.append(num)

    return paddings


def _get_padded_token_len(paddings: list[int], x: int) -> int:
    """Return the first element in paddings list greater or equal to x.
    """
    index = bisect.bisect_left(paddings, x)
    assert index < len(paddings)
    return paddings[index]


def replace_set_lora(model):

    def _tpu_set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
        bias: Optional[torch.Tensor] = None,
    ):
        # TODO: The integer index leads to a recompilation, but converting it
        # to a tensor doesn't seem to work anymore. This might be fixed with a
        # later release of torch_xla.
        self._original_set_lora(index, lora_a, lora_b, embeddings_tensor, bias)
        xm.mark_step()

    def _tpu_reset_lora(self, index: int):
        self._original_reset_lora(index)
        xm.mark_step()

    for _, module in model.named_modules():
        if isinstance(module, BaseLayerWithLoRA):
            module._original_set_lora = module.set_lora
            module._original_reset_lora = module.reset_lora
            module.set_lora = _tpu_set_lora.__get__(module, module.__class__)
            module.reset_lora = _tpu_reset_lora.__get__(
                module, module.__class__)
