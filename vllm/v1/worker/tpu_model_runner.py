# SPDX-License-Identifier: Apache-2.0
import time
from typing import TYPE_CHECKING, Optional, cast
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
# TPU XLA related
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.sampling_params import SamplingType
from vllm.utils import LayerBlockType, cdiv, is_pin_memory_available
from vllm.v1.attention.backends.pallas import (NUM_KV_PAGES_PER_BLOCK,
                                               NUM_QUERIES_PER_BLOCK,
                                               PallasAttentionBackend,
                                               PallasMetadata)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import LogprobsTensors, ModelRunnerOutput
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput

logger = init_logger(__name__)

# Here we utilize the behavior that out-of-bound index is ignored.
# FIXME(woosuk): Find a more reliable way to prevent possible bugs.
_PAD_SLOT_ID = 1_000_000_000
INVALID_TOKEN_ID = -1


class TPUModelRunner:

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
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
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype

        self.is_multimodal_model = model_config.is_multimodal_model
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_model_len = model_config.max_model_len
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs

        # Model-related.
        self.num_attn_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        self.num_query_heads = model_config.get_num_attention_heads(
            parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.head_size = model_config.get_head_size()
        self.hidden_size = model_config.get_hidden_size()

        # Persistent batch.
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_blocks_per_req=self.max_num_blocks_per_req,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
        )

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}

        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: dict[str, dict[int, torch.Tensor]] = {}

        # KV caches for forward pass
        self.kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []

        # Cached torch/numpy tensor
        # The pytorch tensor and numpy array share the same buffer.
        # Sometimes the numpy op is faster so we create both.
        self.input_ids_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int32,
                                         device="cpu")
        self.input_ids_np = self.input_ids_cpu.numpy()

        self.positions_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int32,
                                         device="cpu")
        self.positions_np = self.positions_cpu.numpy()

        self.slot_mapping_cpu = torch.zeros(self.max_num_tokens,
                                            dtype=torch.int64,
                                            device="cpu")
        self.slot_mapping_np = self.slot_mapping_cpu.numpy()

        # self.input_batch.block_table has a shape of [max_num_reqs,
        # max_num_blocks_per_req]. To reduce the number of recompilation,
        # we want the block_table.shape[0] to be num_tokens.
        # To make the block_table to be compatible with the paged attention
        # kernel, we want the block_table[1] to be multiple of
        # NUM_KV_PAGES_PER_BLOCK.
        padded_max_num_blocks_per_req = _get_padded_number(
            self.max_num_blocks_per_req, NUM_KV_PAGES_PER_BLOCK)
        self.block_table_cpu = torch.zeros(
            (self.max_num_tokens, padded_max_num_blocks_per_req),
            dtype=self.input_batch.block_table.get_cpu_tensor().dtype,
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
        self.arange_np = np.arange(self.max_num_tokens, dtype=np.int32)

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
            if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt=new_req_data.prompt,
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                generator=generator,
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
        assert self.model is not None
        return self.model

    def get_kv_cache_spec(self) -> KVCacheSpec:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each 
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache 
            format. Layers that do not need KV cache are not included.
        """

        forward_ctx = self.vllm_config.compilation_config.static_forward_context
        block_size = self.vllm_config.cache_config.block_size
        kv_cache_spec: KVCacheSpec = {}
        for layer_name, attn_module in forward_ctx.items():
            # TODO: Support other attention modules, e.g., sliding window,
            # cross-attention, MLA.
            assert isinstance(attn_module, Attention)
            if attn_module.attn_type == AttentionType.DECODER:
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=attn_module.dtype,
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
        block_table_cpu = self.input_batch.block_table.get_cpu_tensor()
        block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()
        block_offsets = positions_np % self.block_size
        np.add(block_numbers * self.block_size,
               block_offsets,
               out=self.slot_mapping_np[:total_num_scheduled_tokens])

        # Prepare the attention metadata.
        self.query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens_per_req,
                  out=self.query_start_loc_np[1:num_reqs + 1])

        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens_per_req)

        # Do the padding and copy the tensors to the TPU.
        padded_total_num_scheduled_tokens = _get_padded_number(
            total_num_scheduled_tokens, NUM_QUERIES_PER_BLOCK)
        self.input_ids = self.input_ids_cpu[:
                                            padded_total_num_scheduled_tokens].to(
                                                self.device)
        self.position_ids = self.positions_cpu[:
                                               padded_total_num_scheduled_tokens].to(
                                                   self.device)
        self.slot_mapping_cpu[total_num_scheduled_tokens:] = _PAD_SLOT_ID
        slot_mapping = self.slot_mapping_cpu[:
                                             padded_total_num_scheduled_tokens].to(
                                                 self.device)
        padded_block_table = self.block_table_cpu[:
                                                  padded_total_num_scheduled_tokens]
        padded_block_table[:num_reqs, :self.max_num_blocks_per_req] = (
            self.input_batch.block_table.get_cpu_tensor()[:num_reqs])
        padded_block_table = padded_block_table.to(self.device)
        query_start_loc = self.query_start_loc_cpu[:
                                                   padded_total_num_scheduled_tokens
                                                   + 1].to(self.device)
        seq_lens = self.seq_lens_cpu[:padded_total_num_scheduled_tokens].to(
            self.device)

        attn_metadata = PallasMetadata(
            slot_mapping=slot_mapping,
            block_tables=padded_block_table,
            context_lens=seq_lens,
            query_start_loc=query_start_loc,
            num_seqs=num_reqs,
        )
        # NOTE(woosuk): Due to chunked prefills, there can be at most 1 partial
        # request in the batch. While we should not sample any token from this
        # partial request, we do so for simplicity. We will ignore the sampled
        # token from the partial request.
        # TODO: Support prompt logprobs.
        logits_indices = query_start_loc[1:] - 1
        return attn_metadata, logits_indices

    @torch.no_grad()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput:
        # Update cached state
        self._update_states(scheduler_output)

        # Prepare inputs
        attn_metadata, logits_indices = self._prepare_inputs(scheduler_output)
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

        # Run the decoder
        with set_forward_context(attn_metadata, self.vllm_config):
            hidden_states = self.model(
                token_ids=self.input_ids,
                position_ids=self.position_ids,
                kv_caches=self.kv_caches,
            )
        hidden_states = hidden_states[:total_num_scheduled_tokens]
        num_reqs = self.input_batch.num_reqs
        logits_indices = logits_indices[:num_reqs]
        hidden_states = hidden_states[logits_indices]
        logits = self.model.compute_logits(hidden_states, None)
        selected_token_ids = torch.argmax(logits, dim=-1, keepdim=True)

        # Then, let's update the cache state.
        request_seq_lens: list[tuple[int, CachedRequestState, int]] = []
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

        # num_reqs entries should be non-None
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
            logprobs=None,
            prompt_logprobs_dict=prompt_logprobs_dict,
        )
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
            model = get_model(vllm_config=self.vllm_config)
        model = model.eval()
        xm.mark_step()
        xm.wait_device_ops()
        model = ModelWrapperV1(model)
        self.model = torch.compile(model,
                                   backend="openxla",
                                   fullgraph=True,
                                   dynamic=False)

    def dummy_run(
        self,
        kv_caches,
        num_tokens: int,
    ) -> None:
        input_ids = torch.zeros(num_tokens,
                                dtype=torch.int32,
                                device=self.device)
        position_ids = torch.zeros(num_tokens,
                                   dtype=torch.int32,
                                   device=self.device)
        slot_mapping = torch.zeros(num_tokens,
                                   dtype=torch.int64,
                                   device=self.device)
        block_tables = torch.zeros((num_tokens, self.block_table_cpu.shape[1]),
                                   dtype=torch.int32,
                                   device=self.device)
        query_lens = [1] * num_tokens
        query_start_loc = torch.cumsum(torch.tensor([0] + query_lens,
                                                    dtype=torch.int32),
                                       dim=0,
                                       dtype=torch.int32).to(self.device)
        context_lens = torch.ones((num_tokens, ),
                                  dtype=torch.int32,
                                  device=self.device)
        attn_metadata = PallasMetadata(
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            query_start_loc=query_start_loc,
            num_seqs=num_tokens,
        )

        torch._dynamo.mark_dynamic(input_ids, 0)
        torch._dynamo.mark_dynamic(position_ids, 0)
        torch._dynamo.mark_dynamic(attn_metadata.slot_mapping, 0)
        torch._dynamo.mark_dynamic(attn_metadata.block_tables, 0)
        torch._dynamo.mark_dynamic(attn_metadata.query_start_loc, 0)
        torch._dynamo.mark_dynamic(attn_metadata.context_lens, 0)

        with set_forward_context(attn_metadata, self.vllm_config, 0):
            assert self.model is not None
            self.model(input_ids, position_ids, kv_caches)

    def capture_model(self) -> None:
        """Compile the model."""

        logger.info("Compiling the model with different input shapes.")

        start = time.perf_counter()
        num_tokens = 16
        while True:
            self.dummy_run(self.kv_caches, num_tokens)
            logger.info("  -- num_tokens: %d", num_tokens)
            xm.mark_step()
            xm.wait_device_ops()
            if num_tokens >= self.scheduler_config.max_num_batched_tokens:
                break
            num_tokens *= 2
        end = time.perf_counter()
        logger.info("Compilation finished in in %.2f [secs].", end - start)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV 
            cache size of each layer
        """
        if len(kv_cache_config.groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not "
                "supported yet.")

        kv_caches: dict[str, torch.Tensor] = {}

        for layer_name, layer_spec in kv_cache_config.kv_cache_spec.items():
            tensor_config = kv_cache_config.tensors[layer_name]
            assert tensor_config.size % layer_spec.page_size_bytes == 0
            num_blocks = tensor_config.size // layer_spec.page_size_bytes
            if isinstance(layer_spec, FullAttentionSpec):
                kv_cache_shape = PallasAttentionBackend.get_kv_cache_shape(
                    num_blocks, layer_spec.block_size, layer_spec.num_kv_heads,
                    layer_spec.head_size)
                dtype = layer_spec.dtype

                tpu_k_cache = torch.zeros(kv_cache_shape,
                                          dtype=dtype,
                                          device=self.device)
                tpu_v_cache = torch.zeros_like(tpu_k_cache)

                kv_caches[layer_name] = (tpu_k_cache, tpu_v_cache)
            else:
                raise NotImplementedError

        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches)


class ModelWrapperV1(nn.Module):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Executes the forward pass of the model and samples the next token.

        Args:
            token_ids: The input token IDs of shape [num_tokens].
            position_ids: The input position IDs of shape [num_tokens].
            kv_caches: The key and value caches. They can be None during the
                memory profiling at initialization.
        """
        # Skip this in memory profiling at initialization.
        if kv_caches[0][0].numel() > 0:
            attn_metadata = get_forward_context().attn_metadata
            # index_copy_(slot_mapping) only works when the inserted dimension
            # is 0. However, the KV cache in the Pallas backend has the shape
            # [num_kv_heads, num_blocks, block_size, head_size]. To make it
            # work, we need to flatten the first three dimensions and modify
            # the slot_mapping accordingly.
            # kv_caches: list[tuple[torch.Tensor, torch.Tensor]]
            num_kv_heads, num_blocks, block_size, _ = kv_caches[0][0].shape
            slot_mapping = attn_metadata.slot_mapping
            slot_mapping = slot_mapping.flatten()
            head_indicies = torch.arange(0,
                                         num_kv_heads,
                                         device=slot_mapping.device,
                                         dtype=slot_mapping.dtype)
            head_indicies *= block_size * num_blocks
            slot_mapping = slot_mapping.repeat_interleave(num_kv_heads).view(
                -1, num_kv_heads)
            slot_mapping = slot_mapping + head_indicies.view(1, -1)
            slot_mapping = slot_mapping.flatten()
            attn_metadata.slot_mapping = slot_mapping

        assert self.model is not None
        hidden_states = self.model(
            token_ids,
            position_ids,
            kv_caches,
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata,
    ) -> Optional[torch.Tensor]:
        logits = self.model.compute_logits(hidden_states, sampling_metadata)
        return logits


def _get_padded_number(n: int, multiple: int) -> int:
    return ((n + multiple - 1) // multiple) * multiple
