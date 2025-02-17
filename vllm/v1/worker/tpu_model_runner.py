# SPDX-License-Identifier: Apache-2.0
import enum
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, cast
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
# TPU XLA related
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from vllm.attention import AttentionMetadata
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.sampling_params import SamplingType
from vllm.utils import LayerBlockType, cdiv, is_pin_memory_available
from vllm.v1.attention.backends.pallas import (PallasAttentionBackend,
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
# FIXME(woosuk): Temporarily disabled top-p sampling since it's too slow.
_ENABLE_TOP_P = False
# FIXME(woosuk): A temporary hack to support `n > 1`.
# This can significantly affect the performance if too large.
_MAX_NUM_SAMPLES = 128


class TPUModelRunner():

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

        self.model: Optional[nn.Module] = None

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
        self.requests: Dict[str, CachedRequestState] = {}

        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: Dict[str, Dict[int, torch.Tensor]] = {}

        # KV caches for forward pass
        self.kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # Cached torch/numpy tensor
        # xw32: what's the numpy array (eg input_ids_np) for?
        self.input_ids_cpu = torch.empty(self.max_num_tokens,
                                     dtype=torch.int32,
                                     device="cpu")
        self.input_ids_np = self.input_ids_cpu.numpy()

        self.positions_cpu = torch.empty(self.max_num_tokens,
                                         dtype=torch.int32,
                                         device="cpu")
        self.positions_np = self.positions_cpu.numpy()

        # xw32: slot_mapping maps a token to its position in the block (=block_numbers * self.block_size+block_offset)
        self.slot_mapping_cpu = torch.empty(self.max_num_tokens,
                                            dtype=torch.int64,
                                            device="cpu")
        self.slot_mapping_np = self.slot_mapping_cpu.numpy()

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
            True if there is a new/resumed/paused/finished request in the batch.
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
        removed_req_indices: List[int] = []
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

        req_ids_to_add: List[str] = []
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
            start_index = len(req_state.block_ids) - len(
                req_data.new_block_ids)
            self.input_batch.block_table.append_row(req_index, start_index,
                                                    req_data.new_block_ids)

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
        print(f'xw32 _prepare_inputs begins. {scheduler_output=}')
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        # xw32q: Do we need this?
        self.input_batch.block_table.commit(num_reqs)

        # Get the number of scheduled tokens for each request.
        # TODO: The Python loop can be slow. Optimize.
        num_scheduled_tokens_per_req = []
        max_num_scheduled_tokens_all_reqs = 0
        for req_id in self.input_batch.req_ids[:num_reqs]:
            assert req_id is not None
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            # print(f'xw32 TPUModelRunner.prepare_input line148. {req_id=}, {num_tokens=}')
            # xw32 TPUModelRunner.prepare_input line148. req_id='0', num_tokens=5
            num_scheduled_tokens_per_req.append(num_tokens)
            max_num_scheduled_tokens_all_reqs = max(max_num_scheduled_tokens_all_reqs,
                                           num_tokens)
        num_scheduled_tokens_per_req = np.array(num_scheduled_tokens_per_req, dtype=np.int32)
        assert max_num_scheduled_tokens_all_reqs > 0

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens_per_req)

        # Get batched arange.
        # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = np.concatenate(
            [self.arange_np[:n] for n in num_scheduled_tokens_per_req])
        
        # Get positions.
        # TODO(xw32): add an example of the output positions_np.
        positions_np = self.positions_np[:total_num_scheduled_tokens]
        # print(f'xw32 TPUModelRunner.prepare_input. {total_num_scheduled_tokens=}, {self.input_batch.num_computed_tokens_cpu=}, {self.input_batch.num_reqs=}, {self.model_config.uses_mrope=}')
        # xw32 TPUModelRunner.prepare_input. total_num_scheduled_tokens=5, self.input_batch.num_computed_tokens_cpu=array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int32), self.input_batch.num_reqs=1, self.model_config.uses_mrope=False
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)
        
        # xw32: Do we need to check self.model_config.uses_mrope?

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])
        # print(f'xw32 TPUModelRunner.prepare_input line148. {positions_np=}, {req_indices=}, {self.input_batch.token_ids_cpu.shape=}, {token_indices}')
        # xw32 TPUModelRunner.prepare_input line148. positions_np=array([0, 1, 2, 3, 4]), req_indices=array([0, 0, 0, 0, 0], dtype=int32), self.input_batch.token_ids_cpu.shape=(16, 512), [0 1 2 3 4]
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        
        # print(f'xw32 TPUModelRunner.prepare_input line189 . {self.input_batch.token_ids_cpu_tensor=}') # prints a 2d tensor
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
        print(f'xw32 TPUModelRunner.prepare_input line214. {self.query_start_loc_np.shape=}')
        self.query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens_per_req,
                  out=self.query_start_loc_np[1:num_reqs + 1])
        
        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens_per_req)

        # Copy the tensors to the TPU.
        self.input_ids = self.input_ids_cpu[:total_num_scheduled_tokens].to(self.device)
        self.position_ids = self.positions_cpu[:total_num_scheduled_tokens].to(self.device)
        query_start_loc = self.query_start_loc_cpu[:total_num_scheduled_tokens+1].to(self.device)
        seq_lens = self.seq_lens_cpu[:total_num_scheduled_tokens].to(self.device)
        slot_mapping = self.slot_mapping_cpu[:total_num_scheduled_tokens].to(self.device)
        print(f'xw32 TPUModelRunner.prepare_input line230 . {self.input_batch.block_table.get_device_tensor().shape=}, {total_num_scheduled_tokens=}, {num_reqs=}')

        attn_metadata = PallasMetadata(
            slot_mapping=slot_mapping,
            block_tables=(
                self.input_batch.block_table.get_device_tensor()[:total_num_scheduled_tokens]),
            context_lens=seq_lens,
            query_start_loc=query_start_loc,
            num_seqs=num_reqs,
            # num_actual_tokens=total_num_scheduled_tokens,
            # max_query_len=max_num_scheduled_tokens,
            # max_seq_len=max_seq_len,
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
        logger.info(f"xw32 TPUModelRunner.execute_model. {scheduler_output=}")

        # Update cached state
        self._update_states(scheduler_output)

        # Prepare inputs
        attn_metadata, logits_indices = self._prepare_inputs(scheduler_output)
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        num_input_tokens = num_scheduled_tokens

        input_ids = self.input_ids[:num_input_tokens]

        # Run the decoder
        with set_forward_context(attn_metadata, self.vllm_config): 
            positions = self.position_ids[:num_input_tokens]
            selected_token_ids = self.model(
                token_ids=input_ids,
                position_ids=positions,
                kv_caches=self.kv_caches,
                # xw32q: Why in gpu_model_runner.py, attn_metadata is None https://github.com/vllm-project/vllm/blob/46fe9b46d83e733130ce952eb3967a9c96713583/vllm/v1/worker/gpu_model_runner.py#L455?
                attn_metadata=attn_metadata,
            )

        # Then, let's update the cache state.
        num_reqs = self.input_batch.num_reqs
        for i, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            assert req_id is not None
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            assert seq_len <= req_state.num_tokens
            if seq_len == req_state.num_tokens:
                # Append the sampled token to the output token ids.
                token_id = selected_token_ids[i]
                self.input_batch.token_ids_cpu[i, seq_len] = token_id
                self.input_batch.num_tokens[i] += 1
                req_state.output_token_ids.append(token_id)
            else:
                # xw32q: what are these from gpu_model_runner.py? I don't understand.
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
        req_ids = cast(List[str], self.input_batch.req_ids[:num_reqs])

        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=selected_token_ids,
            logprob_token_ids_cpu=None,
            logprobs_cpu=None,
        )
        return model_runner_output

        # old code begins
        # # Init
        # num_reqs = self.input_batch.num_reqs
        # assert num_reqs > 0
        # sampled_token_ids_list = [0] * num_reqs

        # # Run decodes (a single batch)
        # if len(decode_data.req_ids) > 0:
        #     # Forward
        #     with set_forward_context(decode_data.attn_metadata,
        #                              self.vllm_config):
        #         assert self.model is not None
        #         selected_token_ids = self.model(decode_data.input_tokens,
        #                                         decode_data.input_positions,
        #                                         decode_data.attn_metadata,
        #                                         self.kv_caches)

        #     # Transfer sampled tokens from TPU to CPU
        #     selected_token_ids_list = selected_token_ids.cpu().tolist()

        #     # Update cached state
        #     for i, req_id in enumerate(decode_data.req_ids):
        #         # xw32: what is the difference between req_index and req_id?
        #         req_index = self.input_batch.req_id_to_index[req_id]
        #         req_state = self.requests[req_id]

        #         seq_len = (req_state.num_computed_tokens +
        #                    scheduler_output.num_scheduled_tokens[req_id])

        #         token_id = selected_token_ids_list[i]

        #         self.input_batch.token_ids_cpu[req_index, seq_len] = token_id
        #         self.input_batch.num_tokens[req_index] += 1
        #         req_state.output_token_ids.append(token_id)

        #         sampled_token_ids_list[req_index] = token_id

        # # Run each prompt
        # for (req_id, prompt_len, input_tokens, input_positions,
        #      attn_metadata) in prompt_data.zipped():
        #     assert req_id is not None
        #     req_state = self.requests[req_id]
        #     req_index = self.input_batch.req_id_to_index[req_id]

        #     # Forward
        #     with set_forward_context(attn_metadata, self.vllm_config):
        #         assert self.model is not None
        #         selected_token_ids = self.model(input_tokens, input_positions,
        #                                         attn_metadata, self.kv_caches)

        #     seq_len = (req_state.num_computed_tokens +
        #                scheduler_output.num_scheduled_tokens[req_id])
        #     if seq_len >= len(req_state.prompt_token_ids):
        #         # Transfer sampled tokens from TPU to CPU
        #         token_id = selected_token_ids.cpu()[prompt_len - 1].item()
        #         sampled_token_ids_list[req_index] = token_id

        #         # Update cached state
        #         self.input_batch.token_ids_cpu[req_index, seq_len] = token_id
        #         self.input_batch.num_tokens[req_index] += 1
        #         req_state.output_token_ids.append(token_id)

        # # Get req_ids
        # assert all(
        #     req_id is not None for req_id in
        #     self.input_batch.req_ids[:num_reqs]), "req_ids contains None"
        # req_ids = cast(List[str], self.input_batch.req_ids[:num_reqs])

        # model_runner_output = ModelRunnerOutput(
        #     req_ids=req_ids,
        #     req_id_to_index=self.input_batch.req_id_to_index,
        #     sampled_token_ids=sampled_token_ids_list,
        #     logprob_token_ids_cpu=None,
        #     logprobs_cpu=None,
        # )

        # return model_runner_output
        # old code ends

    def load_model(self) -> None:
        logger.info("xw32 TPUModelRunner.load_model begins.")
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
        # TODO(xw32): turn on dynamo.
        # xw32 turns off dynamo
        self.model = model
        # self.model = torch.compile(model,
        #                            backend="openxla",
        #                            fullgraph=True,
        #                            dynamic=False)
        logger.info("xw32 TPUModelRunner.load_model ends.")

    # @torch.inference_mode() fails so I disabled it.
    # It's also not in the original v1 tpu_model_runner.py
    # @torch.inference_mode()
    def dummy_run(
        self,
        kv_caches,
        num_tokens: int,
        seq_len: Optional[int] = None,
    ) -> None:
        # logger.info(f"xw32 TPUModelRunner.dummy_run. {self.input_ids_cpu.shape=}, {self.positions_cpu.shape=}, {num_tokens=}, {self.input_ids_cpu.device=}")
        # xw32 qq: what are input_ids and positions and slot_mapping? What are their shapes? Here is the answer:
        # xw32 TPUModelRunner.dummy_run. self.input_ids.shape=torch.Size([8192]), self.positions.shape=torch.Size([8192]), num_tokens=16, 32, ..., self.input_ids.device=device(type='xla', index=0)
        input_ids = torch.zeros(num_tokens,
                                     dtype=torch.int32,
                                     device=self.device)
        position_ids = torch.zeros(num_tokens,
                                     dtype=torch.int64,
                                     device=self.device)
        slot_mapping = torch.zeros(num_tokens,
                                   dtype=torch.int64,
                                   device=self.device)
        block_tables = torch.zeros(
            (num_tokens, self.max_num_blocks_per_req),
            dtype=torch.int32,
            device=self.device)
        context_lens = torch.ones((num_tokens, ),
                                  dtype=torch.int32,
                                  device=self.device)
        block_tables = torch.zeros(
            (num_tokens, self.max_num_blocks_per_req),
            dtype=torch.int32,
            device=self.device)
        query_start_loc = torch.zeros(num_tokens+1, dtype=torch.int32, device=self.device)
        # how do I set torch._dynamo.mark_dynamic?
        # The attn_metadata is used in torch._dynamo.mark_dynamic.
        attn_metadata = PallasMetadata(
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            query_start_loc=query_start_loc,
            num_seqs=num_tokens,  # xw32: is it correct?
        )
        with set_forward_context(None, self.vllm_config):
            assert self.model is not None
            logger.info(f"xw32 TPUModelRunner.dummy_run. before calling self.model, {input_ids.shape=}, {position_ids.shape=}")
            self.model(input_ids, position_ids, None, kv_caches)
            logger.info(f"xw32 TPUModelRunner.dummy_run. after calling self.model")

    def capture_model(self) -> None:
        """Compile the model."""

        logger.info("xw32 TPUModelRunner.capture_model.")
        logger.info("Compiling the model with different input shapes.")

        # xw32 qq: is the compilation here for both torch.compile and the XLA compile?
        # xw32: may need to compile for num_seqs.
        start = time.perf_counter()
        num_tokens = 16
        while True:
            self.dummy_run(self.kv_caches, num_tokens)
            xm.wait_device_ops()
            logger.info("  -- num_tokens: %d", num_tokens)
            if num_tokens >= self.scheduler_config.max_num_batched_tokens:
                break
            num_tokens *= 2
        end = time.perf_counter()
        logger.info("Compilation finished in in %.2f [secs].",
                    end - start)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV 
            cache size of each layer
        """
        logger.info(f"xw32 TPUModelRunner.initialize_kv_cache. {kv_cache_config=}")
        if len(kv_cache_config.groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not "
                "supported yet.")

        kv_caches: Dict[str, torch.Tensor] = {}

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
        attn_metadata: AttentionMetadata,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Executes the forward pass of the model and samples the next token.

        Args:
            token_ids: The input token IDs of shape [batch_size, seq_len].
            position_ids: The input position IDs of shape [batch_size, seq_len].
            attn_metadata: The Pallas attention metadata.
            input_lens: The actual input lengths of shape [batch_size].
            t: The sampling temperature of shape [batch_size].
            p: The top-p probability of shape [batch_size].
            num_samples: Number of samples to draw from each logits vector.
            kv_caches: The key and value caches. They can be None during the
                memory profiling at initialization.
        """
        # Skip this in memory profiling at initialization.
        logger.info("xw32 ModelWrapperV1.forward.")
        print(f'xw32 ModelWrapperV1.forward', flush=True)
        print(f'xw32 ModelWrapperV1.forward {token_ids=}')
        print(f'xw32 ModelWrapperV1.forward {position_ids=}')
        print(f'xw32 ModelWrapperV1.forward {attn_metadata=}')
        print(f'xw32 ModelWrapperV1.forward {len(kv_caches)=}, {kv_caches[0][0].shape=}')
        if attn_metadata is not None:
            # index_copy_(slot_mapping) only works when the inserted dimension
            # is 0. However, the KV cache in the Pallas backend has the shape
            # [num_kv_heads, num_blocks, block_size, head_size]. To make it
            # work, we need to flatten the first three dimensions and modify
            # the slot_mapping accordingly.
            # kv_caches: List[Tuple[torch.Tensor, torch.Tensor]]
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
        print(f'xw32 ModelWrapperV1.forward, right before calling self.model, {token_ids=}', flush=True)
        hidden_states = self.model(
            token_ids,
            position_ids,
            kv_caches,
            attn_metadata,
        )
        print(f'xw32 ModelWrapperV1.forward, right after calling self.model, {hidden_states.shape=}', flush=True)

        # hidden_states = hidden_states.flatten(0, 1) is not needed because previously hidden_states has shape [bs, T, C] and we need to combine the first 2 dimensions.
        # hidden_states = hidden_states.flatten(0, 1)
        print(f'xw32 ModelWrapperV1.forward, right after calling hidden_states.flatten, {hidden_states.shape=}', flush=True)
        logits = self.model.compute_logits(hidden_states, None)
        print(f'xw32 ModelWrapperV1.forward, right after calling self.model.compute_logits', flush=True)

        # Greedy sampling.
        argmax_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
        print(f'xw32 ModelWrapperV1.forward, right after calling torch.argmax', flush=True)
        argmax_token_ids = argmax_token_ids.squeeze(dim=-1)
        print(f'xw32 ModelWrapperV1.forward, right after calling argmax_token_ids.squeeze', flush=True)
        return argmax_token_ids


def _get_padded_prefill_len(x: int) -> int:
    # NOTE(woosuk): The pallas FlashAttention kernel requires the sequence
    # length to be a multiple of 16. We pad the prompt length to the nearest
    # multiple of 16. This is also good for performance.
    if x <= 16:
        return 16
    return 1 << (x - 1).bit_length()


def _get_padded_batch_size(batch_size: int) -> int:
    # The GMM Pallas kernel requires num_tokens * topk to be a multiple of 16.
    # To meet this requirement in the simplest way, we set the minimal batch
    # size to 8.
    if batch_size <= 8:
        return 8
    else:
        return ((batch_size + 15) // 16) * 16
