import gc
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.distributed
import torch.nn as nn

from vllm.compilation.compile_context import set_compile_context
from vllm.config import CompilationLevel, VllmConfig
from vllm.distributed.parallel_state import graph_capture
from vllm.forward_context import set_forward_context
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import MultiModalKwargs
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, DeviceMemoryProfiler, cdiv,
                        is_pin_memory_available)
from vllm.v1.attention.backends.flash_attn import (FlashAttentionBackend,
                                                   FlashAttentionMetadata)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.sample.metadata import SamplingMetadata

if TYPE_CHECKING:
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm.v1.core.scheduler import SchedulerOutput

logger = init_logger(__name__)


class GPUModelRunner:

    def __init__(
        self,
        vllm_config: VllmConfig,
        input_registry: InputRegistry = INPUT_REGISTRY,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                cache_config.cache_dtype]

        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_model_len = model_config.max_model_len
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        self.max_num_tokens = scheduler_config.max_num_batched_tokens

        # Model-related.
        self.num_attn_layers = model_config.get_num_attention_layers(
            parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.head_size = model_config.get_head_size()
        self.hidden_size = model_config.get_hidden_size()

        # Multi-modal data support
        self.input_registry = input_registry

        # Lazy initialization
        # self.model: nn.Module  # Set after load_model
        self.kv_caches: List[torch.Tensor] = []
        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: Dict[str, Dict[int, torch.Tensor]] = {}

        # Request states.
        self.requests: Dict[str, CachedRequestState] = {}
        # Persistent batch.
        self.input_batch = InputBatch(
            max_num_reqs=self.scheduler_config.max_num_seqs,
            max_model_len=self.max_model_len,
            max_num_blocks_per_req=self.max_num_blocks_per_req,
            device=self.device,
            pin_memory=self.pin_memory,
        )

        self.use_cuda_graph = (self.vllm_config.compilation_config.level
                               == CompilationLevel.PIECEWISE
                               and not self.model_config.enforce_eager)
        # TODO(woosuk): Provide an option to tune the max cudagraph batch size.
        self.cudagraph_batch_sizes = [1, 2, 4] + [i for i in range(8, 513, 8)]
        self.positions = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device=self.device)
        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device)

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        # Remove stopped requests from the cached states.
        # Keep the states of the pre-empted requests.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.encoder_cache.pop(req_id, None)

        # Free the cached encoder outputs.
        for req_id, input_id in scheduler_output.free_encoder_input_ids:
            encoder_outputs = self.encoder_cache.get(req_id)
            if encoder_outputs is not None:
                encoder_outputs.pop(input_id, None)
                if not encoder_outputs:
                    self.encoder_cache.pop(req_id, None)

        # Remove the requests from the persistent batch.
        stopped_req_ids = set().union(
            scheduler_output.preempted_req_ids,
            scheduler_output.finished_req_ids,
        )
        removed_req_indices: List[int] = []
        for req_id in stopped_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Update the states of the running requests.
        for req_data in scheduler_output.scheduled_running_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]
            req_index = self.input_batch.req_id_to_index[req_id]

            # Update the num_computed_tokens.
            req_state.num_computed_tokens = req_data.num_computed_tokens
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                req_data.num_computed_tokens)

            # Update the block table.
            num_new_blocks = len(req_data.new_block_ids)
            if num_new_blocks == 0:
                continue
            start_index = len(req_state.block_ids)
            end_index = start_index + num_new_blocks
            req_state.block_ids.extend(req_data.new_block_ids)
            self.input_batch.block_table_cpu[
                req_index, start_index:end_index] = req_data.new_block_ids

        req_ids_to_add: List[str] = []
        # Add new requests to the cached states.
        for req_data in scheduler_output.scheduled_new_reqs:
            req_id = req_data.req_id
            sampling_params = req_data.sampling_params
            if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=req_data.prompt_token_ids,
                prompt=req_data.prompt,
                mm_inputs=req_data.mm_inputs,
                mm_positions=req_data.mm_positions,
                sampling_params=sampling_params,
                generator=generator,
                block_ids=req_data.block_ids,
                num_computed_tokens=req_data.num_computed_tokens,
                output_token_ids=[],
            )
            req_ids_to_add.append(req_id)

        # Update the cached states of the resumed requests.
        for req_data in scheduler_output.scheduled_resumed_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            req_state.block_ids = req_data.block_ids
            req_state.num_computed_tokens = req_data.num_computed_tokens
            req_ids_to_add.append(req_id)

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

    def _prepare_inputs(self, scheduler_output: "SchedulerOutput"):
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table[:num_reqs].copy_(
            self.input_batch.block_table_cpu_tensor[:num_reqs],
            non_blocking=True)

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
        mask = arange_matrix < num_scheduled_tokens[:, np.newaxis]
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
        token_indices = positions_np + req_indices * self.max_model_len
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
        block_numbers = self.input_batch.block_table_cpu_tensor.flatten()[
            token_indices // self.block_size]
        block_offsets = token_indices % self.block_size
        slot_mapping = torch.empty((total_num_scheduled_tokens, ),
                                   dtype=torch.int32,
                                   device="cpu",
                                   pin_memory=self.pin_memory)
        torch.add(block_numbers * self.block_size,
                  block_offsets,
                  out=slot_mapping)

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

        input_ids = input_ids.to(self.device, non_blocking=True)
        self.positions[:total_num_scheduled_tokens].copy_(positions,
                                                          non_blocking=True)
        query_start_loc = query_start_loc.to(self.device, non_blocking=True)
        seq_start_loc = seq_start_loc.to(self.device, non_blocking=True)
        slot_mapping = slot_mapping.to(self.device, non_blocking=True).long()
        attn_metadata = FlashAttentionMetadata(
            num_actual_tokens=total_num_scheduled_tokens,
            max_query_len=max_num_scheduled_tokens,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_start_loc=seq_start_loc,
            block_table=self.input_batch.block_table[:num_reqs],
            slot_mapping=slot_mapping,
        )
        # NOTE(woosuk): Due to chunked prefills, there can be at most 1 partial
        # request in the batch. While we should not sample any token from this
        # partial request, we do so for simplicity. We will ignore the sampled
        # token from the partial request.
        # TODO: Support prompt logprobs.
        logits_indices = query_start_loc[1:] - 1
        return input_ids, attn_metadata, logits_indices

    def _prepare_sampling(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> SamplingMetadata:
        skip_copy = True
        if (scheduler_output.finished_req_ids
                or scheduler_output.preempted_req_ids):
            skip_copy = False
        if (scheduler_output.scheduled_new_reqs
                or scheduler_output.scheduled_resumed_reqs):
            skip_copy = False
        # Create the sampling metadata.
        sampling_metadata = self.input_batch.make_sampling_metadata(skip_copy)
        return sampling_metadata

    def _execute_encoder(self, scheduler_output: "SchedulerOutput"):
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return

        # Batch the multi-modal inputs.
        mm_inputs: List[MultiModalKwargs] = []
        req_input_ids: List[Tuple[int, int]] = []
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
    ) -> List[torch.Tensor]:
        encoder_outputs: List[torch.Tensor] = []
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
    ) -> ModelRunnerOutput:
        self._update_states(scheduler_output)

        # Run the encoder.
        self._execute_encoder(scheduler_output)
        encoder_outputs = self._gather_encoder_outputs(scheduler_output)

        # Prepare the decoder inputs.
        input_ids, attn_metadata, logits_indices = self._prepare_inputs(
            scheduler_output)
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        if (self.use_cuda_graph
                and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]):
            # Use piecewise CUDA graphs.
            # Add padding to the batch size.
            num_input_tokens = self._get_padded_batch_size(
                num_scheduled_tokens)
        else:
            # Eager mode.
            num_input_tokens = num_scheduled_tokens

        # Get the inputs embeds.
        if encoder_outputs:
            inputs_embeds = self.model.get_input_embeddings(
                input_ids, encoder_outputs)
        else:
            inputs_embeds = self.model.get_input_embeddings(input_ids)
        # NOTE(woosuk): To unify token ids and soft tokens (vision embeddings),
        # always use embeddings (rather than token ids) as input to the model.
        # TODO(woosuk): Avoid the copy. Optimize.
        self.inputs_embeds[:num_scheduled_tokens].copy_(inputs_embeds)

        # Run the decoder.
        # Use persistent buffers for CUDA graphs.
        with set_forward_context(attn_metadata, self.vllm_config):
            hidden_states = self.model(
                input_ids=None,
                positions=self.positions[:num_input_tokens],
                kv_caches=self.kv_caches,
                attn_metadata=None,
                inputs_embeds=self.inputs_embeds[:num_input_tokens],
            )
        hidden_states = hidden_states[:num_scheduled_tokens]
        hidden_states = hidden_states[logits_indices]
        logits = self.model.compute_logits(hidden_states, None)

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self._prepare_sampling(scheduler_output)
        sampler_output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )

        # NOTE: CPU-GPU synchronization happens here.
        sampled_token_ids = sampler_output.sampled_token_ids.cpu()
        sampled_token_ids_list = sampled_token_ids.tolist()
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
                token_id = sampled_token_ids_list[i]
                self.input_batch.token_ids_cpu[i, seq_len] = token_id
                req_state.output_token_ids.append(token_id)
            else:
                # Ignore the sampled token from the partial request.
                # Rewind the generator state as if the token was not sampled.
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    # This relies on cuda-specific torch-internal impl details
                    generator.set_offset(generator.get_offset() - 4)

        if sampler_output.logprob_token_ids is None:
            logprob_token_ids = None
        else:
            logprob_token_ids = sampler_output.logprob_token_ids.cpu()
        if sampler_output.logprobs is None:
            logprobs = None
        else:
            logprobs = sampler_output.logprobs.cpu()
        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids[:num_reqs],
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids_cpu=sampled_token_ids,
            logprob_token_ids_cpu=logprob_token_ids,
            logprobs_cpu=logprobs,
        )
        return model_runner_output

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        with DeviceMemoryProfiler() as m:  # noqa: SIM117
            self.model = get_model(vllm_config=self.vllm_config)

        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB",
                    self.model_memory_usage / float(2**30))

    @torch.inference_mode()
    def _dummy_run(
        self,
        model: nn.Module,
        num_tokens: int,
        kv_caches: List[torch.Tensor],
    ) -> torch.Tensor:
        with set_forward_context(None, self.vllm_config):
            hidden_states = model(
                input_ids=None,
                positions=self.positions[:num_tokens],
                kv_caches=kv_caches,
                attn_metadata=None,
                inputs_embeds=self.inputs_embeds[:num_tokens])
        return hidden_states

    def profile_run(self) -> None:
        # TODO(woosuk): Profile the max memory usage of the encoder and
        # the encoder cache.
        # use an empty tensor instead of `None`` to force Dynamo to pass
        # it by reference, rather by specializing on the value `None`.
        # the `dtype` argument does not matter, and we use `float32` as
        # a placeholder (it has wide hardware support).
        # it is important to create tensors inside the loop, rather than
        # multiplying the list, to avoid Dynamo from treating them as
        # tensor aliasing.
        dummy_kv_caches = [
            torch.tensor([], dtype=torch.float32, device=self.device)
            for _ in range(self.num_attn_layers)
        ]
        with set_compile_context(self.cudagraph_batch_sizes):
            # Trigger compilation for general shape.
            hidden_states = self._dummy_run(self.model, self.max_num_tokens,
                                            dummy_kv_caches)
        logits = self.model.compute_logits(hidden_states, None)
        logits = logits[:self.max_num_tokens]
        # TODO(woosuk): Consider the memory usage of the sampler.
        torch.cuda.synchronize()
        del hidden_states, logits
        gc.collect()

    def capture_model(self) -> None:
        if not self.use_cuda_graph:
            logger.warning(
                "Skipping CUDA graph capture. Please add "
                "-O %s to use CUDA graphs.", CompilationLevel.PIECEWISE)
            return

        start_time = time.perf_counter()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        with graph_capture():
            for num_tokens in reversed(self.cudagraph_batch_sizes):
                self._dummy_run(self.model, num_tokens, self.kv_caches)

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, cuda_graph_size / (1 << 30))

    def initialize_kv_cache(self, num_blocks: int) -> None:
        assert len(self.kv_caches) == 0
        kv_cache_shape = FlashAttentionBackend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        for _ in range(self.num_attn_layers):
            self.kv_caches.append(
                torch.zeros(kv_cache_shape,
                            dtype=self.kv_cache_dtype,
                            device=self.device))

    def _get_padded_batch_size(self, batch_size: int) -> Optional[int]:
        # TODO: Optimize this?
        for size in self.cudagraph_batch_sizes:
            if batch_size <= size:
                return size
        return None


@dataclass
class CachedRequestState:

    req_id: str
    prompt_token_ids: List[int]
    prompt: Optional[str]
    mm_inputs: List[MultiModalKwargs]
    mm_positions: List["PlaceholderRange"]
    sampling_params: SamplingParams
    generator: Optional[torch.Generator]

    block_ids: List[int]
    num_computed_tokens: int
    output_token_ids: List[int]

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)


class InputBatch:

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_blocks_per_req: int,
        device: torch.device,
        pin_memory: bool,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.device = device
        self.pin_memory = pin_memory

        self.req_ids: List[Optional[str]] = [None] * max_num_reqs
        self.req_id_to_index: Dict[str, int] = {}

        self.token_ids_cpu = np.empty((max_num_reqs, max_model_len),
                                      dtype=np.int32)
        self.num_computed_tokens_cpu = np.empty(max_num_reqs, dtype=np.int32)

        # Attention-related.
        self.block_table = torch.zeros((max_num_reqs, max_num_blocks_per_req),
                                       device=self.device,
                                       dtype=torch.int32)
        self.block_table_cpu_tensor = torch.zeros(
            (max_num_reqs, max_num_blocks_per_req),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self.block_table_cpu = self.block_table_cpu_tensor.numpy()

        # Sampling-related.
        self.temperature = torch.empty((max_num_reqs, ),
                                       dtype=torch.float32,
                                       device=device)
        self.temperature_cpu_tensor = torch.empty((max_num_reqs, ),
                                                  dtype=torch.float32,
                                                  device="cpu",
                                                  pin_memory=pin_memory)
        self.temperature_cpu = self.temperature_cpu_tensor.numpy()
        self.greedy_reqs: Set[str] = set()
        self.random_reqs: Set[str] = set()

        self.top_p = torch.empty((max_num_reqs, ),
                                 dtype=torch.float32,
                                 device=device)
        self.top_p_cpu_tensor = torch.empty((max_num_reqs, ),
                                            dtype=torch.float32,
                                            device="cpu",
                                            pin_memory=pin_memory)
        self.top_p_cpu = self.top_p_cpu_tensor.numpy()
        self.top_p_reqs: Set[str] = set()

        self.top_k = torch.empty((max_num_reqs, ),
                                 dtype=torch.int32,
                                 device=device)
        self.top_k_cpu_tensor = torch.empty((max_num_reqs, ),
                                            dtype=torch.int32,
                                            device="cpu",
                                            pin_memory=pin_memory)
        self.top_k_cpu = self.top_k_cpu_tensor.numpy()
        self.top_k_reqs: Set[str] = set()

        # req_index -> generator
        self.generators: Dict[int, torch.Generator] = {}

        self.num_logprobs: Dict[str, int] = {}
        self.prompt_logprob_reqs: Set[str] = set()

    def add_request(
        self,
        request: "CachedRequestState",
        req_index: Optional[int] = None,
    ) -> None:
        if req_index is None:
            req_index = self.num_reqs
        assert req_index < self.max_num_reqs

        req_id = request.req_id
        self.req_ids[req_index] = req_id
        self.req_id_to_index[req_id] = req_index

        # Copy the prompt token ids and output token ids.
        num_prompt_tokens = len(request.prompt_token_ids)
        self.token_ids_cpu[
            req_index, :num_prompt_tokens] = request.prompt_token_ids
        start_idx = num_prompt_tokens
        end_idx = start_idx + len(request.output_token_ids)
        self.token_ids_cpu[req_index,
                           start_idx:end_idx] = request.output_token_ids

        self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens
        num_blocks = len(request.block_ids)
        self.block_table_cpu[req_index, :num_blocks] = request.block_ids

        sampling_params = request.sampling_params
        self.temperature_cpu[req_index] = sampling_params.temperature
        if sampling_params.sampling_type == SamplingType.GREEDY:
            self.greedy_reqs.add(req_id)
        else:
            self.random_reqs.add(req_id)

        self.top_p_cpu[req_index] = sampling_params.top_p
        if sampling_params.top_p < 1:
            self.top_p_reqs.add(req_id)
        self.top_k_cpu[req_index] = sampling_params.top_k
        if sampling_params.top_k > 0:
            self.top_k_reqs.add(req_id)

        self.generators[req_index] = request.generator

        num_logprobs = sampling_params.logprobs
        if num_logprobs is not None and num_logprobs > 0:
            self.num_logprobs[req_id] = num_logprobs
        if sampling_params.prompt_logprobs:
            self.prompt_logprob_reqs.add(req_id)

    def remove_request(self, req_id: str) -> Optional[int]:
        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None
        self.req_ids[req_index] = None

        self.greedy_reqs.discard(req_id)
        self.random_reqs.discard(req_id)
        self.top_p_reqs.discard(req_id)
        self.top_k_reqs.discard(req_id)
        self.generators.pop(req_index, None)
        self.num_logprobs.pop(req_id, None)
        self.prompt_logprob_reqs.discard(req_id)
        return req_index

    def clear(self) -> None:
        self.req_ids = [None] * self.max_num_reqs
        self.req_id_to_index.clear()
        self.greedy_reqs.clear()
        self.random_reqs.clear()
        self.top_p_reqs.clear()
        self.top_k_reqs.clear()
        self.generators.clear()
        self.num_logprobs.clear()
        self.prompt_logprob_reqs.clear()

    def condense(self, empty_req_indices: List[int]) -> None:
        if self.num_reqs == 0:
            # The batched states are empty.
            return

        # NOTE(woosuk): This function assumes that the empty_req_indices
        # is sorted in descending order.
        last_req_index = self.num_reqs + len(empty_req_indices) - 1
        while empty_req_indices:
            # Find the largest non-empty index.
            while last_req_index in empty_req_indices:
                last_req_index -= 1

            # Find the smallest empty index.
            empty_index = empty_req_indices.pop()
            if empty_index >= last_req_index:
                break

            # Swap the states.
            req_id = self.req_ids[last_req_index]
            self.req_ids[empty_index] = req_id
            self.req_ids[last_req_index] = None
            self.req_id_to_index[req_id] = empty_index

            # TODO(woosuk): Optimize the copy of token_ids_cpu and
            # block_table_cpu.
            self.token_ids_cpu[empty_index] = self.token_ids_cpu[
                last_req_index]
            self.num_computed_tokens_cpu[
                empty_index] = self.num_computed_tokens_cpu[last_req_index]
            self.block_table_cpu[empty_index] = self.block_table_cpu[
                last_req_index]
            self.temperature_cpu[empty_index] = self.temperature_cpu[
                last_req_index]
            self.top_p_cpu[empty_index] = self.top_p_cpu[last_req_index]
            self.top_k_cpu[empty_index] = self.top_k_cpu[last_req_index]
            generator = self.generators.pop(last_req_index, None)
            if generator is not None:
                self.generators[empty_index] = generator

            # Decrement last_req_index since it is now empty.
            last_req_index -= 1

    def make_sampling_metadata(
        self,
        skip_copy: bool = False,
    ) -> SamplingMetadata:
        if not skip_copy:
            self.temperature[:self.num_reqs].copy_(
                self.temperature_cpu_tensor[:self.num_reqs], non_blocking=True)
            self.top_p[:self.num_reqs].copy_(
                self.top_p_cpu_tensor[:self.num_reqs], non_blocking=True)
            self.top_k[:self.num_reqs].copy_(
                self.top_k_cpu_tensor[:self.num_reqs], non_blocking=True)
        return SamplingMetadata(
            temperature=self.temperature[:self.num_reqs],
            all_greedy=self.all_greedy,
            all_random=self.all_random,
            top_p=self.top_p[:self.num_reqs],
            top_k=self.top_k[:self.num_reqs],
            no_top_p=self.no_top_p,
            no_top_k=self.no_top_k,
            generators=self.generators,
            max_num_logprobs=self.max_num_logprobs,
        )

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    @property
    def all_greedy(self) -> bool:
        return len(self.random_reqs) == 0

    @property
    def all_random(self) -> bool:
        return len(self.greedy_reqs) == 0

    @property
    def no_top_p(self) -> bool:
        return len(self.top_p_reqs) == 0

    @property
    def no_top_k(self) -> bool:
        return len(self.top_k_reqs) == 0

    @property
    def max_num_logprobs(self) -> int:
        return max(self.num_logprobs.values()) if self.num_logprobs else 0

    @property
    def no_logprob(self) -> bool:
        return len(self.num_logprobs) == 0

    @property
    def no_prompt_logprob(self) -> bool:
        return len(self.prompt_logprob_reqs) == 0
