# SPDX-License-Identifier: Apache-2.0

import time
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
import torch_xla.core.xla_model as xm

from vllm.attention import AttentionType
from vllm.attention.layer import Attention
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.inputs import INPUT_REGISTRY
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import MultiModalKwargs
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType, cdiv
from vllm.v1.attention.backends.neuron_attn import (NeuronAttentionBackend,
                                                    NeuronAttentionMetadata)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, LogprobsTensors,
                             ModelRunnerOutput)
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput

logger = init_logger(__name__)

B_P_SIZE = 128
LARGE_TILE_SZ = 2048
INVALID_TOKEN_ID = -1


def shift_bit_length(x):
    return 1 << (x - 1).bit_length()


class NeuronModelRunner:

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

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.pin_memory = False
        self.dtype = self.model_config.dtype
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                cache_config.cache_dtype]

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

        # Multi-modal data support
        self.input_registry = INPUT_REGISTRY
        self.uses_mrope = model_config.uses_mrope

        # Lazy initialization
        # self.model: nn.Module  # Set after load_model
        self.kv_caches: list[torch.Tensor] = []
        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: dict[str, dict[int, torch.Tensor]] = {}

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}
        # Persistent batch.
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_blocks_per_req=self.max_num_blocks_per_req,
            device="cpu",
            pin_memory=self.pin_memory,
            vocab_size=model_config.get_vocab_size(),
        )

        self.input_ids = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int32,
                                     device="cpu")
        self.positions = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device="cpu")
        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: Optional[IntermediateTensors] = None

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = torch.zeros((3, self.max_num_tokens + 1),
                                               dtype=torch.int64,
                                               device="cpu")
            self.mrope_positions_cpu = torch.zeros(
                (3, self.max_num_tokens + 1),
                dtype=torch.int64,
                device="cpu",
                pin_memory=self.pin_memory)

        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device="cpu")

        self.neuron_compilation_batch_sizes = list(
            reversed(
                self.vllm_config.compilation_config.cudagraph_capture_sizes))

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
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

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                image_grid_thw = []
                video_grid_thw = []
                second_per_grid_ts = []
                for mm_input in self.requests[req_id].mm_inputs:
                    if mm_input.get("image_grid_thw") is not None:
                        image_grid_thw.extend(
                            mm_input["image_grid_thw"].tolist())
                    if mm_input.get("video_grid_thw") is not None:
                        video_grid_thw.extend(
                            mm_input["video_grid_thw"].tolist())
                    if mm_input.get("second_per_grid_ts") is not None:
                        second_per_grid_ts.extend(
                            mm_input["second_per_grid_ts"])

                hf_config = self.model_config.hf_config

                self.requests[req_id].mrope_positions, \
                    self.requests[req_id].mrope_position_delta = \
                    MRotaryEmbedding.get_input_positions_tensor(
                        self.requests[req_id].prompt_token_ids,
                        hf_config=hf_config,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts,
                    )

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        for req_data in scheduler_output.scheduled_cached_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            # Update the cached states.
            num_computed_tokens = req_data.num_computed_tokens
            req_state.num_computed_tokens = num_computed_tokens
            # Add the sampled token(s) from the previous step (if any).
            # This doesn't include "unverified" tokens like spec decode tokens.
            num_new_tokens = (num_computed_tokens +
                              len(req_data.new_token_ids) -
                              req_state.num_tokens)
            if num_new_tokens == 1:
                # Avoid slicing list in most common case.
                req_state.output_token_ids.append(req_data.new_token_ids[-1])
            elif num_new_tokens > 0:
                req_state.output_token_ids.extend(
                    req_data.new_token_ids[-num_new_tokens:])
            # Update the block IDs.
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
                num_computed_tokens)
            self.input_batch.block_table.append_row(req_data.new_block_ids,
                                                    req_index)
            # Add new_token_ids to token_ids_cpu.
            start_token_index = num_computed_tokens
            end_token_index = num_computed_tokens + len(req_data.new_token_ids)
            self.input_batch.token_ids_cpu[
                req_index,
                start_token_index:end_token_index] = req_data.new_token_ids
            self.input_batch.num_tokens_no_spec[req_index] = end_token_index
            # Add spec_token_ids to token_ids_cpu.
            spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
                req_id, ())
            if spec_token_ids:
                start_index = end_token_index
                end_token_index += len(spec_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index, start_index:end_token_index] = spec_token_ids
            # NOTE(woosuk): `num_tokens` here may include spec decode tokens.
            self.input_batch.num_tokens[req_index] = end_token_index

        # Check if the batch has changed. If not, we can skip copying the
        # sampling metadata from CPU to GPU.
        batch_changed = len(removed_req_indices) > 0 or len(req_ids_to_add) > 0

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

        if batch_changed:
            self.input_batch.refresh_sampling_metadata()

    def _prepare_inputs(self, scheduler_output: "SchedulerOutput"):
        from vllm.attention.ops.nki_flash_attn import reorder_context_mask

        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        with torch.inference_mode():
            self.input_batch.block_table.commit(num_reqs)

        # Get the number of scheduled tokens for each request.
        # TODO: The Python loop can be slow. Optimize.
        num_scheduled_tokens = []
        max_num_scheduled_tokens = 0
        for req_id in self.input_batch.req_ids[:num_reqs]:
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_scheduled_tokens.append(num_tokens)
            max_num_scheduled_tokens = max(max_num_scheduled_tokens,
                                           num_tokens)
        num_scheduled_tokens = np.array(num_scheduled_tokens, dtype=np.int32)
        assert max_num_scheduled_tokens > 0

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        indices = np.arange(num_reqs)
        req_indices = np.repeat(indices, num_scheduled_tokens)

        # Get batched arange.
        # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange_matrix = np.tile(np.arange(max_num_scheduled_tokens),
                                (num_reqs, 1))
        mask = arange_matrix < np.expand_dims(num_scheduled_tokens, axis=1)
        arange = arange_matrix[mask]

        # Get positions.
        positions = torch.empty((total_num_scheduled_tokens, ),
                                dtype=torch.int32,
                                device="cpu",
                                pin_memory=self.pin_memory)
        positions_np = positions.numpy()
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])
        token_indices = torch.from_numpy(token_indices)
        input_ids = torch.empty((total_num_scheduled_tokens, ),
                                dtype=torch.int32,
                                device="cpu",
                                pin_memory=self.pin_memory)
        torch.index_select(torch.from_numpy(
            self.input_batch.token_ids_cpu).flatten(),
                           0,
                           token_indices,
                           out=input_ids)

        # Calculate the slot mapping.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
        # where K is the max_num_blocks_per_req and the block size is 2.
        # NOTE(woosuk): We can't simply use `token_indices // block_size` here
        # because M (max_model_len) is not necessarily divisible by block_size.
        block_numbers = self.input_batch.block_table.get_cpu_tensor().flatten(
        )[req_indices * self.max_num_blocks_per_req +
          positions_np // self.block_size]
        block_offsets = torch.from_numpy(positions_np % self.block_size)
        slot_mapping = torch.empty((total_num_scheduled_tokens, ),
                                   dtype=torch.int32,
                                   device="cpu",
                                   pin_memory=self.pin_memory)
        torch.add(block_numbers * self.block_size,
                  block_offsets,
                  out=slot_mapping)

        _PAD_SLOT_ID = self.num_blocks * self.block_size
        padded_num_tokens = self._get_padded_batch_size(
            total_num_scheduled_tokens)
        slot_mapping_pad_length = padded_num_tokens - slot_mapping.shape[0]
        slot_mapping = torch.nn.functional.pad(slot_mapping,
                                               (0, slot_mapping_pad_length),
                                               'constant', _PAD_SLOT_ID)

        # Prepare the attention metadata.
        query_start_loc = torch.empty((num_reqs + 1, ),
                                      dtype=torch.int32,
                                      device="cpu",
                                      pin_memory=self.pin_memory)
        query_start_loc_np = query_start_loc.numpy()
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1:])

        seq_lens = (self.input_batch.num_computed_tokens_cpu[:num_reqs] +
                    num_scheduled_tokens)
        max_seq_len = seq_lens.max()
        seq_start_loc = torch.empty((num_reqs + 1, ),
                                    dtype=torch.int32,
                                    device="cpu",
                                    pin_memory=self.pin_memory)
        seq_start_loc_np = seq_start_loc.numpy()
        seq_start_loc_np[0] = 0
        np.cumsum(seq_lens, out=seq_start_loc_np[1:])

        with torch.inference_mode():
            self.input_ids[:total_num_scheduled_tokens].copy_(input_ids)
            self.positions[:total_num_scheduled_tokens].copy_(positions)

        seq_lens = torch.diff(seq_start_loc)
        query_lens = torch.diff(query_start_loc)
        context_lens = seq_lens - query_lens
        num_active_blocks_shifted = shift_bit_length(
            ((context_lens + self.block_size - 1) //
             self.block_size).sum().item())
        num_active_blocks_factor = max(
            LARGE_TILE_SZ // self.block_size // num_active_blocks_shifted, 1)
        num_active_blocks = num_active_blocks_shifted * num_active_blocks_factor
        assert (num_active_blocks * self.block_size
                ) % LARGE_TILE_SZ == 0, "invalid {num_active_blocks=}"

        context_kv_len = num_active_blocks * self.block_size

        block_table = self.input_batch.block_table.get_cpu_tensor()[:num_reqs]
        active_block_table = get_active_block_tables(
            block_table,
            query_lens,
            seq_lens,
            self.block_size,
            num_active_blocks,
        )

        prior_mask, active_mask = (
            BlockDiagonalCausalFromBottomRightMask.from_seqlens(
                query_lens=query_lens.tolist(),
                seq_lens=seq_lens.tolist(),
                block_size=self.block_size))

        attn_mask = torch.concat(
            [
                nn.functional.pad(
                    prior_mask,
                    (
                        0,
                        max(context_kv_len, LARGE_TILE_SZ) -
                        prior_mask.shape[1],
                        0,
                        padded_num_tokens - prior_mask.shape[0],
                    ),
                    "constant",
                    0,
                ).bool(),
                nn.functional.pad(
                    active_mask,
                    (
                        0,
                        padded_num_tokens - active_mask.shape[1],
                        0,
                        padded_num_tokens - active_mask.shape[0],
                    ),
                    "constant",
                    0,
                ).bool(),
            ],
            dim=1,
        )
        attn_mask = reorder_context_mask(attn_mask,
                                         LARGE_TILE_SZ=LARGE_TILE_SZ,
                                         block_size=self.block_size)

        logits_indices = query_start_loc[1:] - 1
        query_start_loc = query_start_loc.to(self.device)
        seq_start_loc = seq_start_loc.to(self.device)
        slot_mapping = slot_mapping.long().to(self.device)
        active_block_table = active_block_table.to(torch.int32).to(self.device)
        attn_mask = attn_mask.to(self.device)
        attn_metadata = NeuronAttentionMetadata(
            num_actual_tokens=total_num_scheduled_tokens,
            max_query_len=max_num_scheduled_tokens,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_start_loc=seq_start_loc,
            block_table=self.input_batch.block_table.get_device_tensor()
            [:num_reqs],
            slot_mapping=slot_mapping,
            num_active_blocks=num_active_blocks,
            active_block_table=active_block_table,
            attn_mask=attn_mask)
        # NOTE(woosuk): Due to chunked prefills, there can be at most 1 partial
        # request in the batch. While we should not sample any token from this
        # partial request, we do so for simplicity. We will ignore the sampled
        # token from the partial request.
        # TODO: Support prompt logprobs.
        return attn_metadata, logits_indices

    def _execute_encoder(self, scheduler_output: "SchedulerOutput"):
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return

        # Batch the multi-modal inputs.
        mm_inputs: list[MultiModalKwargs] = []
        req_input_ids: list[tuple[str, int]] = []
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]
            for input_id in encoder_input_ids:
                mm_inputs.append(req_state.mm_inputs[input_id])
                req_input_ids.append((req_id, input_id))
        batched_mm_inputs = MultiModalKwargs.batch(mm_inputs)
        batched_mm_inputs = MultiModalKwargs.as_kwargs(batched_mm_inputs,
                                                       device=self.device)

        # Run the encoder.
        # `encoder_outputs` is either of the following:
        # 1. A tensor of shape [num_images, feature_size, hidden_size]
        # in case when feature_size is fixed across all images.
        # 2. A list (length: num_images) of tensors, each of shape
        # [feature_size, hidden_size] in case when the feature size is
        # dynamic depending on input images.
        encoder_outputs = self.model.get_multimodal_embeddings(
            **batched_mm_inputs)

        # Cache the encoder outputs.
        for (req_id, input_id), output in zip(req_input_ids, encoder_outputs):
            if req_id not in self.encoder_cache:
                self.encoder_cache[req_id] = {}
            self.encoder_cache[req_id][input_id] = output

    def _gather_encoder_outputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> list[torch.Tensor]:
        encoder_outputs: list[torch.Tensor] = []
        num_reqs = self.input_batch.num_reqs
        for req_id in self.input_batch.req_ids[:num_reqs]:
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = req_state.num_computed_tokens
            mm_positions = req_state.mm_positions
            for i, pos_info in enumerate(mm_positions):
                start_pos = pos_info["offset"]
                num_encoder_tokens = pos_info["length"]

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

                start_idx = max(num_computed_tokens - start_pos, 0)
                end_idx = min(
                    num_computed_tokens - start_pos + num_scheduled_tokens,
                    num_encoder_tokens)
                assert start_idx < end_idx
                assert req_id in self.encoder_cache
                assert i in self.encoder_cache[req_id]
                encoder_output = self.encoder_cache[req_id][i]
                encoder_outputs.append(encoder_output[start_idx:end_idx])
        return encoder_outputs

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> ModelRunnerOutput:
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            # Return empty ModelRunnerOuptut if there's no work to do.
            return EMPTY_MODEL_RUNNER_OUTPUT

        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            self._execute_encoder(scheduler_output)
            encoder_outputs = self._gather_encoder_outputs(scheduler_output)
        else:
            encoder_outputs = []

        # Prepare the decoder inputs.
        attn_metadata, logits_indices = self._prepare_inputs(scheduler_output)
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        num_input_tokens = self._get_padded_batch_size(num_scheduled_tokens)

        attn_metadata.num_input_tokens = num_input_tokens

        if self.is_multimodal_model:
            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            input_ids = self.input_ids[:num_scheduled_tokens]
            if encoder_outputs:
                inputs_embeds = self.model.get_input_embeddings(
                    input_ids, encoder_outputs)
            else:
                inputs_embeds = self.model.get_input_embeddings(input_ids)
            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds[:num_scheduled_tokens].copy_(inputs_embeds)
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None
        if self.uses_mrope:
            positions = self.mrope_positions[:, :num_input_tokens]
        else:
            positions = self.positions[:num_input_tokens]

        # Run the decoder.
        with set_forward_context(attn_metadata, self.vllm_config):
            hidden_states = self.model(
                input_ids=input_ids.unsqueeze(0).to(self.device),
                positions=positions[:num_input_tokens].unsqueeze(0).to(
                    self.device),
                intermediate_tensors=None,
                inputs_embeds=inputs_embeds.to(self.device)
                if inputs_embeds is not None else None,
            ).cpu()
        hidden_states = hidden_states[0, :num_scheduled_tokens]
        hidden_states = hidden_states[logits_indices.cpu()]

        if hasattr(self.model, "lm_head"):
            # Compute logits on CPU
            self.model.lm_head = self.model.lm_head.cpu()
        logits = self.model.compute_logits(hidden_states, None)

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        sampler_output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )

        sampled_token_ids = sampler_output.sampled_token_ids.tolist()
        # TODO(woosuk): The following loop can be slow since it iterates over
        # the requests one by one. Optimize.
        num_reqs = self.input_batch.num_reqs
        for i, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            assert seq_len <= req_state.num_tokens
            if seq_len == req_state.num_tokens:
                # Append the sampled token to the output token ids.
                token_id = sampled_token_ids[i]
                self.input_batch.token_ids_cpu[i, seq_len] = np.array(token_id)
                req_state.output_token_ids.append(token_id)
            else:
                # Ignore the sampled token from the partial request.
                # Rewind the generator state as if the token was not sampled.
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    generator.set_offset(generator.get_offset() - 4)

        # NOTE: device to host sync happens here.
        # Move as many CPU operations as possible before this sync point.
        logprobs_tensors = sampler_output.logprobs_tensors
        logprobs_lists = logprobs_tensors.tolists() \
            if logprobs_tensors is not None else None

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]] = {}
        for req_id in self.input_batch.req_ids[:num_reqs]:
            prompt_logprobs_dict[req_id] = None

        # Get the valid generated tokens.
        sampled_token_ids = sampler_output.sampled_token_ids
        max_gen_len = sampled_token_ids.shape[-1]
        if max_gen_len == 1:
            # No spec decode tokens.
            valid_sampled_token_ids = sampled_token_ids.tolist()
        else:
            # Includes spec decode tokens.
            valid_mask = sampled_token_ids != INVALID_TOKEN_ID
            gen_lens = valid_mask.sum(dim=1).tolist()
            # TODO(woosuk): Optimize this.
            valid_sampled_token_ids = [
                seq.tolist()
                for seq in sampled_token_ids[valid_mask].split(gen_lens)
            ]

        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=None,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
        )
        return model_runner_output

    def load_model(self) -> None:
        # TODO(gnovack) - Add memory profiler during model load
        with torch.inference_mode():
            logger.info("Starting to load model %s...",
                        self.model_config.model)
            model = get_model(vllm_config=self.vllm_config)
            model = model.eval().to(self.device)
            xm.mark_step()
            xm.wait_device_ops()
            self.model = torch.compile(model,
                                       backend="openxla",
                                       fullgraph=True,
                                       dynamic=False)

    @torch.inference_mode()
    def _dummy_run(self, kv_caches, num_tokens: int) -> torch.Tensor:
        num_active_blocks_shifted = shift_bit_length(
            (self.block_size - 1) // self.block_size)
        num_active_blocks_factor = (LARGE_TILE_SZ // self.block_size //
                                    num_active_blocks_shifted)
        num_active_blocks = num_active_blocks_shifted * num_active_blocks_factor
        block_table = torch.arange((num_tokens // self.block_size) +
                                   1).unsqueeze(0)
        active_block_table = get_active_block_tables(
            block_table,
            torch.tensor([num_tokens]),
            torch.tensor([num_tokens]),
            self.block_size,
            num_active_blocks,
        )

        attn_mask, _ = BlockDiagonalCausalFromBottomRightMask.from_seqlens(
            query_lens=[num_tokens], seq_lens=[num_tokens])
        attn_mask = nn.functional.pad(
            attn_mask,
            (
                0,
                LARGE_TILE_SZ + num_tokens - attn_mask.shape[1],
                0,
                B_P_SIZE - attn_mask.shape[0],
            ),
            "constant",
            0,
        ).bool()

        attn_metadata = NeuronAttentionMetadata(
            num_actual_tokens=num_tokens,
            max_query_len=num_tokens,
            query_start_loc=torch.tensor([0, num_tokens - 1
                                          ]).to(self.device,
                                                non_blocking=True),
            max_seq_len=num_tokens,
            seq_start_loc=torch.tensor([0,
                                        num_tokens - 1]).to(self.device,
                                                            non_blocking=True),
            block_table=block_table,
            slot_mapping=torch.arange(0,
                                      num_tokens).long().to(self.device,
                                                            non_blocking=True),
            num_active_blocks=num_active_blocks,
            active_block_table=active_block_table.to(torch.int32).to(
                self.device, non_blocking=True),
            attn_mask=attn_mask.to(self.device, non_blocking=True))

        if self.is_multimodal_model:
            input_ids = None
            inputs_embeds = self.inputs_embeds[:num_tokens]
        else:
            input_ids = self.input_ids[:num_tokens]
            inputs_embeds = None
        with set_forward_context(attn_metadata, self.vllm_config):
            hidden_states = self.model(
                input_ids=input_ids.unsqueeze(0).to(self.device)
                if input_ids is not None else None,
                positions=self.positions[:num_tokens].unsqueeze(0).to(
                    self.device),
                inputs_embeds=inputs_embeds.to(self.device)
                if inputs_embeds is not None else None,
            )
        return hidden_states

    def profile_run(self) -> None:
        num_tokens = max(self.neuron_compilation_batch_sizes)
        self._dummy_run(self.kv_caches, num_tokens)

    def capture_model(self) -> None:

        start_time = time.perf_counter()

        # Trigger Neuron compilation for specific shapes
        for num_tokens in reversed(self.neuron_compilation_batch_sizes):
            self._dummy_run(self.kv_caches, num_tokens)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logger.info("Neuron compilation finished in %.0f secs", elapsed_time)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        assert len(self.kv_caches) == 0
        self.num_blocks = kv_cache_config.num_blocks
        if len(kv_cache_config.kv_cache_groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not "
                "supported yet.")

        kv_caches: dict[str, torch.Tensor] = {}

        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            for layer_name in kv_cache_group.layer_names:
                tensor_config = kv_cache_config.tensors[layer_name]
                assert tensor_config.size % kv_cache_spec.page_size_bytes == 0
                if isinstance(kv_cache_spec, FullAttentionSpec):
                    kv_cache_shape = NeuronAttentionBackend.get_kv_cache_shape(
                        self.num_blocks + 1, self.block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype

                    neuron_kv_cache = torch.zeros(kv_cache_shape,
                                                  dtype=dtype,
                                                  device=self.device)

                    kv_caches[layer_name] = neuron_kv_cache
                else:
                    raise NotImplementedError

        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches)

    def _get_padded_batch_size(self,
                               batch_size: int,
                               B_P_SIZE: int = 128) -> Optional[int]:
        # Find the smallest compilation batch size that fits batch_size
        for size in self.neuron_compilation_batch_sizes:
            if batch_size <= size:
                # Round up to nearest multiple of B_P_SIZE
                padded_size = ((size + B_P_SIZE - 1) // B_P_SIZE) * B_P_SIZE
                return padded_size
        return None

    def get_model(self) -> nn.Module:
        assert self.model is not None
        return self.model

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each 
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache 
            format. Layers that do not need KV cache are not included.
        """

        forward_ctx = self.vllm_config.compilation_config.static_forward_context
        block_size = self.vllm_config.cache_config.block_size
        kv_cache_spec: dict[str, KVCacheSpec] = {}
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
                    use_mla=False,
                )
            else:
                raise NotImplementedError
        return kv_cache_spec


def get_active_block_tables(block_tables, query_lens, seq_lens, block_size,
                            num_blocks):
    context_lens = seq_lens - query_lens
    blocks_per_seq = (context_lens + block_size - 1) // block_size
    num_seqs = len(seq_lens)
    active_blocks: list[int] = []
    for seq_id in range(num_seqs):
        active_blocks = (
            active_blocks +
            block_tables[seq_id, :blocks_per_seq[seq_id]].tolist())
    return nn.functional.pad(
        torch.tensor(active_blocks),
        (0, num_blocks - len(active_blocks)),
        "constant",
        0,
    )


class BlockDiagonalCausalFromBottomRightMask:

    @staticmethod
    def _from_seqlens(query_lens, seq_lens, block_size=None):
        from torch import logical_and, logical_or

        contexted = block_size is None
        context_lens = torch.tensor(seq_lens) - torch.tensor(query_lens)
        n_queries = sum(query_lens)
        num_seqs = len(query_lens)
        if contexted:
            key_lens_blockaligned = seq_lens
        else:
            n_blocks_per_seq = (context_lens + block_size - 1) // block_size
            offset_per_seq = n_blocks_per_seq * block_size
            key_lens_blockaligned = offset_per_seq[:num_seqs].tolist()
        n_keys = sum(key_lens_blockaligned)

        a = (torch.arange(n_queries).reshape(n_queries,
                                             1).expand(n_queries, n_keys))
        b = torch.arange(n_keys).reshape(1, n_keys).expand(n_queries, n_keys)
        q_cumsum = torch.tensor([0] + query_lens).cumsum(dim=0)
        k_cumsum = torch.tensor([0] + key_lens_blockaligned).cumsum(dim=0)

        prior_mask = torch.zeros(n_queries, n_keys)
        new_masks: list[torch.Tensor] = []
        for seq_id in range(num_seqs):
            ri = q_cumsum[seq_id]
            ci = k_cumsum[seq_id]
            nr = query_lens[seq_id]

            if contexted:
                nc = seq_lens[seq_id]
                a_offset = ci + nc - ri - nr
                new_mask = (a + a_offset) >= b
            else:
                nc = context_lens[seq_id]
                a_offset = ci + nc - 1
                new_mask = a_offset >= b

            left_mask = b >= ci
            top_mask = a >= ri
            bottom_mask = a < (ri + nr)

            new_mask = logical_and(
                logical_and(logical_and(new_mask, left_mask), top_mask),
                bottom_mask,
            )
            prior_mask = logical_or(prior_mask, new_mask)
            new_masks = new_masks + [new_mask]
        return prior_mask

    @staticmethod
    def from_seqlens(query_lens, seq_lens, block_size=None):
        contexted = block_size is None
        if contexted:
            prior_mask = BlockDiagonalCausalFromBottomRightMask._from_seqlens(
                query_lens, seq_lens)
            active_mask = None
        else:
            prior_mask = BlockDiagonalCausalFromBottomRightMask._from_seqlens(
                query_lens, seq_lens, block_size)
            active_mask = BlockDiagonalCausalFromBottomRightMask._from_seqlens(
                query_lens, query_lens)
        return prior_mask, active_mask
