import gc
import time
import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple, cast, Optional
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed
import torch.nn as nn

# TPU XLA related
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.inputs import INPUT_REGISTRY
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.sampling_params import SamplingType
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType, cdiv,
                        is_pin_memory_available)
from vllm.v1.attention.backends.pallas import PallasMetadata, PallasAttentionBackend
from vllm.v1.engine.mm_input_mapper import MMInputMapperClient
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.model_runner_base import ModelRunnerBase, ExecutionMode
from vllm.v1.sample.metadata import SamplingMetadata
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


@dataclass
class PrefillInputData:

    request_ids: List
    prompt_lens: List
    token_ids: List
    position_ids: List
    attn_metadata: List

    def zipped(self):
        return zip(self.request_ids, self.prompt_lens, self.token_ids,
                   self.position_ids, self.attn_metadata)


@dataclass
class DecodeInputData:

    num_decodes: int
    token_ids: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    attn_metadata: Optional[PallasMetadata] = None


class TPUModelRunner(ModelRunnerBase):

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(vllm_config, device)

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

        # Lazy initialization
        # self.model: nn.Module  # Set after load_model
        self.kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # Multi-modal data support
        self.input_registry = INPUT_REGISTRY
        self.mm_registry = MULTIMODAL_REGISTRY

        # NOTE: Initialized input mapper is only used for processing dummy
        # multimodal data into multimodal kwargs for GPU memory profiling.
        self.mm_input_mapper_profiling = MMInputMapperClient(self.model_config)
        self.mm_input_mapper_profiling.use_cache = False

        self.max_num_encoder_input_tokens = self.scheduler_config.max_num_encoder_input_tokens  # noqa: E501
        self.encoder_cache_size = self.scheduler_config.encoder_cache_size

        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: Dict[str, Dict[int, torch.Tensor]] = {}

        # Used to initialize positions for the individual prefills
        self.prefill_positions = torch.tensor(range(self.max_model_len),
                                              device="cpu",
                                              dtype=torch.int32).reshape(
                                                  1, -1)

        # Used to indicate how many prefills there are for each scheduler
        # iteration
        self.num_new_reqs: int = 0

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
            req_state.block_ids.extend(req_data.new_block_ids)
            self.input_batch.block_table.append_row(req_index, start_index,
                                                    req_data.new_block_ids)

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
            )
            req_ids_to_add.append(req_id)

        # Update the cached states of the resumed requests.
        for res_req_data in scheduler_output.scheduled_resumed_reqs:
            req_id = res_req_data.req_id
            req_state = self.requests[req_id]

            req_state.block_ids = res_req_data.block_ids
            req_state.num_computed_tokens = res_req_data.num_computed_tokens
            req_ids_to_add.append(req_id)

        # For TPU, we keep all of the decode requests before the
        # prefill requests in the batch sequence.
        #   1. First condense, so all decodes move to start
        #   2. Then add new prefills to the end of the batch
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            self.input_batch.add_request(req_state, None)  # Append last

        self.num_new_reqs = len(req_ids_to_add)

    def _prepare_prefill_inputs(
        self,
        num_scheduled_tokens: List[int],
    ) -> PrefillInputData:
        # Each prefill run separately with shape [1, padded_prompt_len].
        # So we create lists that will be used in execute_model().

        prefill_request_ids = []
        prefill_prompt_lens = []
        prefill_token_ids = []
        prefill_position_ids = []
        prefill_attn_metadata = []

        # DECODES are the first num_decodes REQUESTS.
        # PREFILLS are the next num_reqs - num_decodes REQUESTS.
        num_reqs = self.input_batch.num_reqs
        num_decodes = num_reqs - self.num_new_reqs
        for idx in range(num_decodes, num_reqs):
            req_id = self.input_batch.req_ids[idx]
            prefill_request_ids.append(req_id)

            prompt_len = num_scheduled_tokens[idx]
            prefill_prompt_lens.append(prompt_len)

            # STATIC SHAPE: prefills are padded to the next power of 2.
            padded_prompt_len = _get_padded_prefill_len(prompt_len)
            assert padded_prompt_len <= self.max_model_len

            # TOKEN_IDS.
            token_ids = torch.from_numpy(self.input_batch.token_ids_cpu[
                idx, :padded_prompt_len].reshape(1, -1))
            token_ids[:, prompt_len:] = 0
            prefill_token_ids.append(token_ids.to(self.device))

            # POSITIONS.
            positions = self.prefill_positions[:, :padded_prompt_len].clone()
            positions[:, prompt_len:] = 0
            prefill_position_ids.append(positions.to(self.device))

            # SLOT_MAPPING.
            # The "slot" is the "physical index" of a token in the KV cache.
            # Look up the block_idx in the block table (logical<>physical map)
            # to compute this.
            block_table_cpu_tensor = self.input_batch.block_table.get_cpu_tensor(
            )
            block_numbers = block_table_cpu_tensor[idx, positions //
                                                   self.block_size].reshape(
                                                       1, -1)

            block_offsets = positions % self.block_size
            slot_mapping = block_numbers * self.block_size + block_offsets
            # Set an out of range value for the padding tokens so that they
            # are ignored when inserting into the KV cache.
            slot_mapping[:, prompt_len:] = _PAD_SLOT_ID
            slot_mapping = slot_mapping.long()

            # TODO: Remove prompt_len param here
            prefill_attn_metadata.append(
                PallasMetadata(
                    num_prefills=1,
                    num_prefill_tokens=prompt_len,  # NOTE: This is not used.
                    num_decode_tokens=0,
                    slot_mapping=slot_mapping.to(self.device),
                    multi_modal_placeholder_index_maps=None,
                    block_tables=None,
                    context_lens=None,
                    effective_query_lens=None,
                ))

        return PrefillInputData(
            request_ids=prefill_request_ids,
            prompt_lens=prefill_prompt_lens,
            token_ids=prefill_token_ids,
            position_ids=prefill_position_ids,
            attn_metadata=prefill_attn_metadata,
        )

    def _prepare_decode_inputs(self) -> DecodeInputData:
        # Decodes run as one single padded batch with shape [batch, 1]
        #
        # We need to set _PAD_SLOT_ID for the padding tokens in the
        # slot_mapping, such that the attention KV cache insertion
        # logic knows to ignore those indicies. Otherwise, the
        # padding data can be dummy since we have a causal mask.

        # DECODES are the first num_decodes REQUESTS.
        # PREFILLS are the next num_reqs - num_decodes REQUESTS.
        num_reqs = self.input_batch.num_reqs
        num_decodes = num_reqs - self.num_new_reqs

        if num_decodes == 0:
            return DecodeInputData(num_decodes=0)

        # PAD FOR STATIC SHAPES.
        padded_batch_size = _get_padded_batch_size(num_decodes)

        # POSITIONS. [batch, 1]
        # We slice at the end, since we use the positions for gathering.
        positions = torch.from_numpy(
            self.input_batch.num_computed_tokens_cpu.reshape(-1, 1))
        index = positions.to(torch.int64)
        index[num_decodes:] = 0
        positions = positions[:padded_batch_size]
        positions[num_decodes:] = 0

        # TOKEN_IDS. [batch, 1]
        token_ids = torch.gather(
            input=torch.from_numpy(self.input_batch.token_ids_cpu),
            dim=1,
            index=index,
        )[:padded_batch_size].to(torch.int32)
        token_ids[num_decodes:] = 0

        # SLOT_MAPPING [batch, 1]
        # The "slot" is the "physical index" of a token in the KV cache.
        # Look up the block_idx in the block table (logical<>physical map)
        # to compute this.
        block_table_cpu_tensor = self.input_batch.block_table.get_cpu_tensor()
        block_number = torch.gather(input=block_table_cpu_tensor,
                                    dim=1,
                                    index=(index // self.block_size))
        block_offsets = index % self.block_size
        slot_mapping = block_number * self.block_size + block_offsets
        # Set an out of range value for the padding tokens so that they
        # are ignored when inserting into the KV cache.
        slot_mapping[num_decodes:] = _PAD_SLOT_ID
        slot_mapping = slot_mapping[:padded_batch_size]
        slot_mapping = slot_mapping.long()

        # BLOCK_TABLE [batch, max_num_blocks_per_req]
        block_table = block_table_cpu_tensor[:padded_batch_size]

        # CONTEXT_LENS [batch_size]
        context_lens = (positions.reshape(-1) + 1)
        context_lens[num_decodes:] = 0

        # CPU<>TPU sync happens here.
        return DecodeInputData(num_decodes=num_decodes,
                               token_ids=token_ids.to(self.device),
                               position_ids=positions.to(self.device),
                               attn_metadata=PallasMetadata(
                                   num_prefills=0,
                                   num_prefill_tokens=0,
                                   num_decode_tokens=padded_batch_size,
                                   slot_mapping=slot_mapping.to(self.device),
                                   multi_modal_placeholder_index_maps=None,
                                   block_tables=block_table.to(self.device),
                                   context_lens=context_lens.to(self.device),
                                   effective_query_lens=None,
                               ))

    def _prepare_inputs(self, scheduler_output: "SchedulerOutput"):
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        num_decodes = num_reqs - self.num_new_reqs

        # TODO: Resurrect
        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        # TODO: Verify this works with TPUs
        # self.input_batch.block_table.commit(num_reqs)

        # Get the number of scheduled tokens for each request.
        # TODO: The Python loop can be slow. Optimize.
        num_scheduled_tokens = []
        max_num_scheduled_tokens = 0
        for idx, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            assert req_id is not None
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_scheduled_tokens.append(num_tokens)
            max_num_scheduled_tokens = max(max_num_scheduled_tokens,
                                           num_tokens)

            # NOTE: Assert that all the decodes are "decodes".
            if idx < num_decodes:
                assert num_tokens == 1

        assert max_num_scheduled_tokens > 0

        return (
            self._prepare_prefill_inputs(num_scheduled_tokens),
            self._prepare_decode_inputs(),
        )

    def _prepare_sampling(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> SamplingMetadata:
        # TODO: Add sampler code to TPUs
        raise NotImplementedError()

    def _execute_encoder(self, scheduler_output: "SchedulerOutput"):
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return

        # Batch the multi-modal inputs.
        mm_inputs: List[MultiModalKwargs] = []
        req_input_ids: List[Tuple[str, int]] = []
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
            assert req_id is not None
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

    @torch.no_grad()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput:
        self._update_states(scheduler_output)

        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            self._execute_encoder(scheduler_output)
            encoder_outputs = self._gather_encoder_outputs(scheduler_output)
        else:
            encoder_outputs = []

        # Prepare the decoder inputs.
        prefill_data, decode_data = self._prepare_inputs(scheduler_output)

        num_reqs = self.input_batch.num_reqs

        ######################### DECODES #########################
        # Decodes run as one single batch with [padded_batch, 1]
        sampled_token_ids_list = []
        if decode_data.num_decodes > 0:
            # FORWARD.
            selected_token_ids = self.model(decode_data.token_ids,
                                            decode_data.position_ids,
                                            decode_data.attn_metadata,
                                            self.kv_caches)

            # NOTE: TPU<>CPU sync happens here.
            # We need to call .cpu() first to avoid recompilation.
            token_ids = selected_token_ids.cpu()[:decode_data.num_decodes]
            sampled_token_ids_list.extend(token_ids.tolist())

            # UPDATE REQUEST STATE.
            for i, req_id in enumerate(
                    self.input_batch.req_ids[:decode_data.num_decodes]):
                assert req_id is not None
                req_state = self.requests[req_id]

                # TODO: ASSERT NO CHUNKED PREFILL.
                assert scheduler_output.num_scheduled_tokens[req_id] == 1
                seq_len = (req_state.num_computed_tokens +
                           scheduler_output.num_scheduled_tokens[req_id])
                assert seq_len == req_state.num_tokens

                # TODO: Verify if req_id_to_index mapping is needed here!
                token_id = sampled_token_ids_list[i]
                self.input_batch.token_ids_cpu[i, seq_len] = token_id
                self.input_batch.num_tokens[i] += 1
                req_state.output_token_ids.append(token_id)

        ######################### PREFILLS #########################
        # Prefills run separately with shape [1, padded_prefill_len],
        # due to lack of variable length attention kernel so far.
        for (req_id, prompt_len, token_ids, position_ids,
             attn_metadata) in prefill_data.zipped():
            assert req_id is not None

            if self.is_multimodal_model:
                # NOTE(woosuk): To unify token ids and soft tokens (vision
                # embeddings), we always use embeddings (rather than token ids)
                # as input to the multimodal model, even when the input is text.
                input_ids = token_ids
                if encoder_outputs:
                    inputs_embeds = self.model.get_input_embeddings(
                        input_ids, encoder_outputs)
                else:
                    inputs_embeds = self.model.get_input_embeddings(input_ids)
                # TODO(woosuk): Avoid the copy. Optimize.
                # self.inputs_embeds.copy_(inputs_embeds)
                # inputs_embeds = self.inputs_embeds
                input_ids = None
            else:
                # For text-only models, we use token ids as input.
                # While it is possible to use embeddings as input just like the
                # multimodal models, it is not desirable for performance since
                # then the embedding layer is not included in the CUDA graph.
                input_ids = token_ids
                inputs_embeds = None

            # FORWARD.
            selected_token_ids = self.model(input_ids, position_ids,
                                            attn_metadata, self.kv_caches, inputs_embeds)

            print("PREFILL selected_token_ids.shape = {}".format(
                selected_token_ids.shape))
            # NOTE: TPU<>CPU sync happens here.
            # We need to call .cpu() first to avoid recompilation.
            token_id = selected_token_ids.cpu()[prompt_len - 1].item()
            sampled_token_ids_list.append(token_id)
            req_state = self.requests[req_id]

            # TODO: ASSERT NO PREFIX CACHING.
            assert req_state.num_computed_tokens == 0
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])

            # TODO: ASSERT NO CHUNKED PREFILL.
            assert seq_len == req_state.num_tokens
            assert prompt_len == seq_len

            # UPDATE REQUEST STATE.
            req_idx = self.input_batch.req_id_to_index[req_id]
            self.input_batch.token_ids_cpu[req_idx, seq_len] = token_id
            self.input_batch.num_tokens[req_idx] += 1
            req_state.output_token_ids.append(token_id)

        # num_reqs entries should be non-None
        assert all(
            req_id is not None for req_id in
            self.input_batch.req_ids[:num_reqs]), "req_ids contains None"
        req_ids = cast(List[str], self.input_batch.req_ids[:num_reqs])

        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=sampled_token_ids_list,
            logprob_token_ids_cpu=None,
            logprobs_cpu=None,
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
        xm.wait_device_ops()
        model = ModelWrapperV1(model)
        self.model = torch.compile(model,
                                   backend="openxla",
                                   fullgraph=True,
                                   dynamic=False)

    def dummy_run(
        self,
        kv_caches,
        batch_size: int,
        seq_len: Optional[int] = None,
        exec_mode: Optional[ExecutionMode] = None,
    ) -> None:
        assert seq_len is not None
        assert exec_mode is not None

        exec_mode = ExecutionMode(exec_mode)
        if exec_mode.is_prefill():
            seq_len = (seq_len + 15) // 16 * 16
            token_ids = torch.zeros((batch_size, seq_len),
                                    dtype=torch.int32,
                                    device=self.device)
            position_ids = torch.zeros((batch_size, seq_len),
                                       dtype=torch.int32,
                                       device=self.device)
            slot_mapping = torch.zeros((batch_size, seq_len),
                                       dtype=torch.int64,
                                       device=self.device)
            if exec_mode == ExecutionMode.PREFILL:
                attn_metadata = PallasMetadata(
                    num_prefills=batch_size,
                    num_prefill_tokens=batch_size * seq_len,
                    num_decode_tokens=0,
                    slot_mapping=slot_mapping,
                    multi_modal_placeholder_index_maps=None,
                    block_tables=None,
                    context_lens=None,
                    effective_query_lens=None,
                )

            else:
                context_lens = torch.ones((batch_size, ),
                                          dtype=torch.int32,
                                          device=self.device)

                block_tables = torch.zeros(
                    (batch_size, self.max_num_blocks_per_req),
                    dtype=torch.int32,
                    device=self.device)

                effective_query_lens = torch.ones_like(context_lens)

                attn_metadata = PallasMetadata(
                    num_prefills=batch_size,
                    num_prefill_tokens=batch_size * seq_len,
                    num_decode_tokens=0,
                    slot_mapping=slot_mapping,
                    multi_modal_placeholder_index_maps=None,
                    block_tables=block_tables,
                    context_lens=context_lens,
                    effective_query_lens=effective_query_lens,
                )
            inputs_embeds = None
        else:
            assert seq_len == 1
            token_ids = torch.zeros((batch_size, seq_len),
                                    dtype=torch.int32,
                                    device=self.device)
            position_ids = torch.zeros((batch_size, seq_len),
                                       dtype=torch.int32,
                                       device=self.device)
            slot_mapping = torch.zeros((batch_size, seq_len),
                                       dtype=torch.int64,
                                       device=self.device)
            block_tables = torch.zeros(
                (batch_size, self.max_num_blocks_per_req),
                dtype=torch.int32,
                device=self.device)
            context_lens = torch.ones((batch_size, ),
                                      dtype=torch.int32,
                                      device=self.device)
            attn_metadata = PallasMetadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decode_tokens=batch_size * seq_len,
                slot_mapping=slot_mapping,
                multi_modal_placeholder_index_maps=None,
                block_tables=block_tables,
                context_lens=context_lens,
            )
            inputs_embeds = None

        # NOTE(woosuk): There are two stages of compilation: torch.compile and
        # XLA compilation. Using `mark_dynamic` can reduce the torch.compile
        # overhead by reusing the FX graph for different shapes.
        # However, the XLA graph will still require static shapes and needs to
        # be re-compiled for every different shapes. This overhead is inevitable
        # in the first run, but can be skipped afterwards as we cache the XLA
        # graphs in the disk (VLLM_XLA_CACHE_PATH).
        if exec_mode.is_prefill():
            # Prefll
            torch._dynamo.mark_dynamic(token_ids, 1)
            torch._dynamo.mark_dynamic(position_ids, 1)
            torch._dynamo.mark_dynamic(attn_metadata.slot_mapping, 1)
        else:
            # Decode
            torch._dynamo.mark_dynamic(token_ids, 0)
            torch._dynamo.mark_dynamic(position_ids, 0)
            torch._dynamo.mark_dynamic(attn_metadata.slot_mapping, 0)
            torch._dynamo.mark_dynamic(attn_metadata.context_lens, 0)
            torch._dynamo.mark_dynamic(attn_metadata.block_tables, 0)

        self.model(token_ids, position_ids, attn_metadata, kv_caches, inputs_embeds)

    def profile_run(self) -> None:
        raise NotImplementedError()

    def capture_model(self) -> None:
        """Compile the model."""

        logger.info("Compiling the model with different input shapes.")

        # Capture prefill shapes
        start = time.perf_counter()
        for batch_size in [1]:
            seq_len = 16
            while True:
                self.dummy_run(self.kv_caches, batch_size, seq_len,
                               ExecutionMode.PREFILL)
                xm.wait_device_ops()
                logger.info("  -- batch_size: %d, seq_len: %d", batch_size,
                            seq_len)

                if seq_len >= self.model_config.max_model_len:
                    break

                num_tokens = batch_size * seq_len
                if num_tokens >= self.scheduler_config.max_num_batched_tokens:
                    break

                # Move to next seq_len
                seq_len = seq_len * 2

        end = time.perf_counter()
        logger.info("Compilation for prefill shapes is done in %.2f [secs].",
                    end - start)

        # Capture decode shapes.
        start = time.time()
        seq_len = 1
        batch_size = 8  # Must be in sync with _get_padded_batch_size()
        while True:
            self.dummy_run(self.kv_caches, batch_size, seq_len,
                           ExecutionMode.DECODE)
            xm.wait_device_ops()
            logger.info("  -- batch_size: %d, seq_len: %d, max_num_seqs = %d",
                        batch_size, seq_len,
                        self.scheduler_config.max_num_seqs)

            if batch_size >= self.scheduler_config.max_num_seqs:
                break

            batch_size = batch_size + 16 if batch_size >= 16 else batch_size * 2

        end = time.time()
        logger.info("Compilation for decode shapes is done in %.2f [secs].",
                    end - start)

    def initialize_kv_cache(self, num_tpu_blocks: int) -> None:
        assert len(self.kv_caches) == 0

        tpu_cache_shape = PallasAttentionBackend.get_kv_cache_shape(
            num_tpu_blocks, self.block_size, self.num_kv_heads, self.head_size)

        for _ in range(self.num_attn_layers):
            tpu_k_cache = torch.zeros(tpu_cache_shape,
                                      dtype=self.kv_cache_dtype,
                                      device=self.device)
            tpu_v_cache = torch.zeros_like(tpu_k_cache)
            self.kv_caches.append((tpu_k_cache, tpu_v_cache))


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
        inputs_embeds: Optional[torch.Tensor] = None,
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
        if kv_caches[0][0].numel() > 0:
            # index_copy_(slot_mapping) only works when the inserted dimension
            # is 0. However, the KV cache in the Pallas backend has the shape
            # [num_kv_heads, num_blocks, block_size, head_size]. To make it
            # work, we need to flatten the first three dimensions and modify
            # the slot_mapping accordingly.
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

        hidden_states = self.model(
            input_ids=token_ids,
            positions=position_ids,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = hidden_states.flatten(0, 1)
        logits = self.model.compute_logits(hidden_states, None)

        # Greedy sampling.
        argmax_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
        argmax_token_ids = argmax_token_ids.squeeze(dim=-1)
        return argmax_token_ids
    
    def get_multimodal_embeddings(self, *args, **kwargs):
        return self.model.get_multimodal_embeddings(*args, **kwargs)
    
    def get_input_embeddings(self, *args, **kwargs):
        return self.model.get_input_embeddings(*args, **kwargs)


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
