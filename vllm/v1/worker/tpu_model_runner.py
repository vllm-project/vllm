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


class ExecutionMode(enum.Enum):
    PREFILL = enum.auto()
    DECODE = enum.auto()
    PREFIX_PREFILL = enum.auto()

    def is_prefill(self) -> bool:
        return self in (ExecutionMode.PREFILL, ExecutionMode.PREFIX_PREFILL)


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
    attn_metadata: PallasMetadata = None


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
        self.mm_registry = MULTIMODAL_REGISTRY

        # NOTE: Initialized input mapper is only used for processing dummy
        # multimodal data into multimodal kwargs for GPU memory profiling.
        self.mm_input_mapper_profiling = MMInputMapperClient(self.model_config)
        self.mm_input_mapper_profiling.use_cache = False

        self.max_num_encoder_input_tokens = self.scheduler_config.max_num_encoder_input_tokens  # noqa: E501
        self.encoder_cache_size = self.scheduler_config.encoder_cache_size

        # Lazy initialization
        # self.model: nn.Module  # Set after load_model
        self.kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: Dict[str, Dict[int, torch.Tensor]] = {}

        # Request states.
        self.requests: Dict[str, CachedRequestState] = {}
        # Persistent batch.
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_blocks_per_req=self.max_num_blocks_per_req,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=model_config.get_vocab_size(),
        )

        self.prefill_positions = torch.tensor(range(self.max_model_len),
                                              device="cpu",
                                              dtype=torch.int32).reshape(
                                                  1, -1)

        self.num_new_reqs = None

        # TODO: Remove this
        # self.use_cuda_graph = (self.vllm_config.compilation_config.level
        #                        == CompilationLevel.PIECEWISE
        #                        and not self.model_config.enforce_eager)
        # # TODO(woosuk): Provide an option to tune the max cudagraph batch size.
        # # The convention is different.
        # # self.cudagraph_batch_sizes sorts in ascending order.
        # # The batch sizes in the config are in descending order.
        # self.cudagraph_batch_sizes = list(
        #     reversed(self.vllm_config.compilation_config.capture_sizes))

        # # Cache the device properties.
        # self.device_properties = torch.cuda.get_device_properties(self.device)
        # self.num_sms = self.device_properties.multi_processor_count

        # # Persistent buffers for CUDA graphs.
        # self.input_ids = torch.zeros(self.max_num_tokens,
        #                              dtype=torch.int32,
        #                              device=self.device)
        # self.positions = torch.zeros(self.max_num_tokens,
        #                              dtype=torch.int64,
        #                              device=self.device)
        # self.inputs_embeds = torch.zeros(
        #     (self.max_num_tokens, self.hidden_size),
        #     dtype=self.dtype,
        #     device=self.device)

        # # OPTIMIZATION: Cache the tensors rather than creating them every step.
        # self.arange_np = np.arange(max(self.max_num_reqs + 1,
        #                                self.max_model_len),
        #                            dtype=np.int32)
        # # NOTE(woosuk): These tensors are "stateless", i.e., they are literally
        # # a faster version of creating a new tensor every time. Thus, we should
        # # not make any assumptions about the values in these tensors.
        # self.input_ids_cpu = torch.zeros(self.max_num_tokens,
        #                                  dtype=torch.int32,
        #                                  device="cpu",
        #                                  pin_memory=self.pin_memory)
        # self.input_ids_np = self.input_ids_cpu.numpy()
        # self.positions_cpu = torch.zeros(self.max_num_tokens,
        #                                  dtype=torch.int64,
        #                                  device="cpu",
        #                                  pin_memory=self.pin_memory)
        # self.positions_np = self.positions_cpu.numpy()
        # self.slot_mapping_cpu = torch.zeros(self.max_num_tokens,
        #                                     dtype=torch.int32,
        #                                     device="cpu",
        #                                     pin_memory=self.pin_memory)
        # self.slot_mapping_np = self.slot_mapping_cpu.numpy()
        # self.query_start_loc_cpu = torch.zeros(self.max_num_reqs + 1,
        #                                        dtype=torch.int32,
        #                                        device="cpu",
        #                                        pin_memory=self.pin_memory)
        # self.query_start_loc_np = self.query_start_loc_cpu.numpy()
        # self.seq_start_loc_cpu = torch.zeros(self.max_num_reqs + 1,
        #                                      dtype=torch.int32,
        #                                      device="cpu",
        #                                      pin_memory=self.pin_memory)
        # self.seq_start_loc_np = self.seq_start_loc_cpu.numpy()

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
            print("prepare prefill idx = {}".format(idx))
            req_id = self.input_batch.req_ids[idx]
            prefill_request_ids.append(req_id)
            print("  req_id = {}".format(req_id))

            prompt_len = num_scheduled_tokens[idx]
            prefill_prompt_lens.append(prompt_len)
            print("  prompt_len = {}".format(prompt_len))

            # STATIC SHAPE: prefills are padded to the next power of 2.
            padded_prompt_len = _get_padded_prefill_len(prompt_len)
            assert padded_prompt_len <= self.max_model_len
            print("  padded_prompt_len = {}".format(padded_prompt_len))

            # TOKEN_IDS.
            token_ids = torch.from_numpy(self.input_batch.token_ids_cpu[
                idx, :padded_prompt_len].reshape(1, -1))
            token_ids[:, prompt_len:] = 0
            prefill_token_ids.append(token_ids.to(self.device))
            print("  token_ids.shape = {} token_ids.vals = {}".format(token_ids.shape, token_ids))

            # POSITIONS.
            positions = self.prefill_positions[:, :padded_prompt_len].clone()
            positions[:, prompt_len:] = 0
            prefill_position_ids.append(positions.to(self.device))
            print("  positions.shape = {} positions.vals = {}".format(positions.shape, positions))

            # SLOT_MAPPING.
            # The "slot" is the "physical index" of a token in the KV cache.
            # Look up the block_idx in the block table (logical<>physical map)
            # to compute this.
            block_table_cpu_tensor = self.input_batch.block_table.get_cpu_tensor(
            )
            block_numbers = block_table_cpu_tensor[idx, positions //
                                                   self.block_size].reshape(
                                                       1, -1)
            print("  block_numbers.shape = {} block_numbers.vals = {}".format(block_numbers.shape, block_numbers))

            block_offsets = positions % self.block_size
            slot_mapping = block_numbers * self.block_size + block_offsets
            # Set an out of range value for the padding tokens so that they
            # are ignored when inserting into the KV cache.
            slot_mapping[:, prompt_len:] = _PAD_SLOT_ID
            slot_mapping = slot_mapping.long()
            print("  slot_mapping.shape = {} slot_mapping.vals = {}".format(slot_mapping.shape, slot_mapping))

            # BLOCK_TABLE [batch, max_num_blocks_per_req]
            # block_table = block_table_cpu_tensor[idx:idx + 1, :]

            # context_lens_tensor = torch.tensor([prompt_len],
            #                                    dtype=torch.int32,
            #                                    device=self.device)
            # prompt_lens_tensor = torch.tensor([prompt_len],
            #                                   dtype=torch.int32,
            #                                   device=self.device)

            prefill_attn_metadata.append(
                PallasMetadata(
                    num_prefills=1,
                    num_prefill_tokens=prompt_len,  # NOTE: This is not used.
                    num_decode_tokens=0,
                    slot_mapping=slot_mapping.to(self.device),
                    multi_modal_placeholder_index_maps=None,
                    block_tables=None,  #block_table.to(self.device),
                    context_lens=None,  #context_lens_tensor,
                    effective_query_lens=None,  #prompt_lens_tensor,
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

        print("prepare num_decodes = {}".format(num_decodes))
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
        print("  positions.shape = {} positions.vals = {}".format(positions.shape, positions))

        # TOKEN_IDS. [batch, 1]
        token_ids = torch.gather(
            input=torch.from_numpy(self.input_batch.token_ids_cpu),
            dim=1,
            index=index,
        )[:padded_batch_size].to(torch.int32)
        token_ids[num_decodes:] = 0
        print("  token_ids.shape = {} token_ids.vals = {}".format(token_ids.shape, token_ids))

        # SLOT_MAPPING [batch, 1]
        # The "slot" is the "physical index" of a token in the KV cache.
        # Look up the block_idx in the block table (logical<>physical map)
        # to compute this.
        block_table_cpu_tensor = self.input_batch.block_table.get_cpu_tensor()
        block_number = torch.gather(input=block_table_cpu_tensor,
                                    dim=1,
                                    index=(index // self.block_size))
        print("  block_number.shape = {} block_number.vals = {}".format(block_number.shape, block_number))
        block_offsets = index % self.block_size
        slot_mapping = block_number * self.block_size + block_offsets
        # Set an out of range value for the padding tokens so that they
        # are ignored when inserting into the KV cache.
        slot_mapping[num_decodes:] = _PAD_SLOT_ID
        slot_mapping = slot_mapping[:padded_batch_size]
        slot_mapping = slot_mapping.long()

        print("  slot_mapping.shape = {} slot_mapping.vals = {}".format(slot_mapping.shape, slot_mapping))
        # BLOCK_TABLE [batch, max_num_blocks_per_req]
        block_table = block_table_cpu_tensor[:padded_batch_size]

        # CONTEXT_LENS [batch_size]
        context_lens = (positions.reshape(-1) + 1)
        context_lens[num_decodes:] = 0
        print("  context_lens.shape = {} context_lens.vals = {}".format(context_lens.shape, context_lens))

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

        # TODO: Ressurect
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

        # # OPTIMIZATION: Start copying the block table first.
        # # This way, we can overlap the copy with the following CPU operations.
        # self.input_batch.block_table.commit(num_reqs)

        # # Get the number of scheduled tokens for each request.
        # # TODO: The Python loop can be slow. Optimize.
        # num_scheduled_tokens = []
        # max_num_scheduled_tokens = 0
        # for req_id in self.input_batch.req_ids[:num_reqs]:
        #     assert req_id is not None
        #     num_tokens = scheduler_output.num_scheduled_tokens[req_id]
        #     num_scheduled_tokens.append(num_tokens)
        #     max_num_scheduled_tokens = max(max_num_scheduled_tokens,
        #                                    num_tokens)
        # num_scheduled_tokens = np.array(num_scheduled_tokens, dtype=np.int32)
        # assert max_num_scheduled_tokens > 0

        # # Get request indices.
        # # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        # req_indices = np.repeat(self.arange_np[:num_reqs],
        #                         num_scheduled_tokens)

        # # Get batched arange.
        # # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # arange = np.concatenate(
        #     [self.arange_np[:n] for n in num_scheduled_tokens])

        # # Get positions.
        # positions_np = self.positions_np[:total_num_scheduled_tokens]
        # np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
        #        arange,
        #        out=positions_np)

        # # Get token indices.
        # # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # # where M is the max_model_len.
        # token_indices = (positions_np +
        #                  req_indices * self.input_batch.token_ids_cpu.shape[1])
        # # NOTE(woosuk): We use torch.index_select instead of np.take here
        # # because torch.index_select is much faster than np.take for large
        # # tensors.
        # torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
        #                    0,
        #                    torch.from_numpy(token_indices),
        #                    out=self.input_ids_cpu[:total_num_scheduled_tokens])

        # # Calculate the slot mapping.
        # # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
        # # where K is the max_num_blocks_per_req and the block size is 2.
        # # NOTE(woosuk): We can't simply use `token_indices // block_size` here
        # # because M (max_model_len) is not necessarily divisible by block_size.
        # block_table_indices = (req_indices * self.max_num_blocks_per_req +
        #                        positions_np // self.block_size)
        # # NOTE(woosuk): We use torch.index_select instead of np.take here
        # # because torch.index_select is much faster than np.take for large
        # # tensors.
        # block_table_cpu = self.input_batch.block_table.get_cpu_tensor()
        # block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()
        # block_offsets = positions_np % self.block_size
        # np.add(block_numbers * self.block_size,
        #        block_offsets,
        #        out=self.slot_mapping_np[:total_num_scheduled_tokens])

        # # Prepare the attention metadata.
        # self.query_start_loc_np[0] = 0
        # np.cumsum(num_scheduled_tokens,
        #           out=self.query_start_loc_np[1:num_reqs + 1])

        # seq_lens = (self.input_batch.num_computed_tokens_cpu[:num_reqs] +
        #             num_scheduled_tokens)
        # max_seq_len = seq_lens.max()
        # self.seq_start_loc_np[0] = 0
        # np.cumsum(seq_lens, out=self.seq_start_loc_np[1:num_reqs + 1])

        # # Copy the tensors to the GPU.
        # self.input_ids[:total_num_scheduled_tokens].copy_(
        #     self.input_ids_cpu[:total_num_scheduled_tokens], non_blocking=True)
        # self.positions[:total_num_scheduled_tokens].copy_(
        #     self.positions_cpu[:total_num_scheduled_tokens], non_blocking=True)
        # query_start_loc = self.query_start_loc_cpu[:num_reqs + 1].to(
        #     self.device, non_blocking=True)
        # seq_start_loc = self.seq_start_loc_cpu[:num_reqs + 1].to(
        #     self.device, non_blocking=True)
        # slot_mapping = self.slot_mapping_cpu[:total_num_scheduled_tokens].to(
        #     self.device, non_blocking=True).long()

        # # Prepare for cascade attention if needed.
        # common_prefix_len = (scheduler_output.num_common_prefix_blocks *
        #                      self.block_size)
        # if common_prefix_len == 0:
        #     # Common case.
        #     use_cascade = False
        # else:
        #     # NOTE(woosuk): Cascade attention uses two attention kernels: one
        #     # for the common prefix and the other for the rest. For the first
        #     # kernel, we concatenate all the query tokens (possibly from
        #     # different requests) and treat them as if they are from the same
        #     # request. Then, we use bi-directional attention to process the
        #     # common prefix in the KV cache. Importantly, this means that the
        #     # first kernel does not do any masking.

        #     # Consider the following example:
        #     # Request 1's input query: [D, E, X]
        #     # Request 1's kv cache: [A, B, C, D, E, X]
        #     # Request 1's num_computed_tokens: 3 (i.e., [A, B, C])
        #     # Request 2's input query: [E, Y]
        #     # Request 2's kv cache: [A, B, C, D, E, Y]
        #     # Request 2's num_computed_tokens: 4 (i.e., [A, B, C, D])

        #     # If we use [A, B, C, D, E] as the common prefix, then the
        #     # first kernel will compute the bi-directional attention between
        #     # input query [D, E, X, E, Y] and common prefix [A, B, C, D, E].
        #     # However, this is wrong because D in Request 1 should not attend to
        #     # E in the common prefix (i.e., we need masking).
        #     # To avoid this, [A, B, C, D] should be the common prefix.
        #     # That is, the common prefix should be capped by the minimum
        #     # num_computed_tokens among the requests, and plus one to include
        #     # the first token of the query.

        #     # In practice, we use [A, B, C] as the common prefix, instead of
        #     # [A, B, C, D] (i.e., the common prefix is capped by the minimum
        #     # num_computed_tokens, without plus one).
        #     # This is because of an implementation detail: We want to always
        #     # use two kernels for cascade attention. Let's imagine:
        #     # Request 3's input query: [D]
        #     # Request 3's kv cache: [A, B, C, D]
        #     # Request 3's num_computed_tokens: 4 (i.e., [A, B, C, D])
        #     # If we use [A, B, C, D] as the common prefix for Request 1-3,
        #     # then Request 3 will be processed only by the first kernel,
        #     # and the second kernel will get an empty input. While this is not
        #     # a fundamental problem, our current implementation does not support
        #     # this case.
        #     common_prefix_len = min(
        #         common_prefix_len,
        #         self.input_batch.num_computed_tokens_cpu[:num_reqs].min())
        #     # common_prefix_len should be a multiple of the block size.
        #     common_prefix_len = (common_prefix_len // self.block_size *
        #                          self.block_size)
        #     use_cascade = FlashAttentionBackend.use_cascade_attention(
        #         common_prefix_len=common_prefix_len,
        #         query_lens=num_scheduled_tokens,
        #         num_query_heads=self.num_query_heads,
        #         num_kv_heads=self.num_kv_heads,
        #         use_alibi=False,  # FIXME
        #         use_sliding_window=self.sliding_window is not None,
        #         num_sms=self.num_sms,
        #     )

        # if use_cascade:
        #     # TODO: Optimize.
        #     cu_prefix_query_lens = torch.tensor(
        #         [0, total_num_scheduled_tokens],
        #         dtype=torch.int32,
        #         device=self.device)
        #     cu_prefix_kv_lens = torch.tensor([0, common_prefix_len],
        #                                      dtype=torch.int32,
        #                                      device=self.device)
        #     cu_suffix_kv_lens = (
        #         self.seq_start_loc_np[:num_reqs + 1] -
        #         self.arange_np[:num_reqs + 1] * common_prefix_len)
        #     cu_suffix_kv_lens = torch.from_numpy(cu_suffix_kv_lens).to(
        #         self.device)
        # else:
        #     cu_prefix_query_lens = None
        #     cu_prefix_kv_lens = None
        #     cu_suffix_kv_lens = None

        # attn_metadata = FlashAttentionMetadata(
        #     num_actual_tokens=total_num_scheduled_tokens,
        #     max_query_len=max_num_scheduled_tokens,
        #     query_start_loc=query_start_loc,
        #     max_seq_len=max_seq_len,
        #     seq_start_loc=seq_start_loc,
        #     block_table=(
        #         self.input_batch.block_table.get_device_tensor()[:num_reqs]),
        #     slot_mapping=slot_mapping,
        #     use_cascade=use_cascade,
        #     common_prefix_len=common_prefix_len,
        #     cu_prefix_query_lens=cu_prefix_query_lens,
        #     cu_prefix_kv_lens=cu_prefix_kv_lens,
        #     cu_suffix_kv_lens=cu_suffix_kv_lens,
        # )
        # # NOTE(woosuk): Due to chunked prefills, there can be at most 1 partial
        # # request in the batch. While we should not sample any token from this
        # # partial request, we do so for simplicity. We will ignore the sampled
        # # token from the partial request.
        # # TODO: Support prompt logprobs.
        # logits_indices = query_start_loc[1:] - 1
        # return attn_metadata, logits_indices

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
        req_id_output_token_ids: Dict[str, List[int]] = \
            {req_id: req.output_token_ids \
                for req_id, req in self.requests.items()}

        sampling_metadata = self.input_batch.make_sampling_metadata(
            req_id_output_token_ids, skip_copy)
        return sampling_metadata

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

        # TODO: Ressurect this code
        # if self.is_multimodal_model:
        #     # Run the multimodal encoder if any.
        #     self._execute_encoder(scheduler_output)
        #     encoder_outputs = self._gather_encoder_outputs(scheduler_output)
        # else:
        #     encoder_outputs = []

        # Prepare the decoder inputs.
        prefill_data, decode_data = self._prepare_inputs(scheduler_output)

        num_reqs = self.input_batch.num_reqs
        # sampled_token_ids = torch.empty(num_reqs, dtype=torch.int32)

        # attn_metadata, logits_indices = self._prepare_inputs(scheduler_output)
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        num_input_tokens = num_scheduled_tokens
        # attn_metadata.num_input_tokens = num_input_tokens

        # TODO: Resurrect this code
        # if self.is_multimodal_model:
        #     # NOTE(woosuk): To unify token ids and soft tokens (vision
        #     # embeddings), we always use embeddings (rather than token ids)
        #     # as input to the multimodal model, even when the input is text.
        #     input_ids = self.input_ids[:num_scheduled_tokens]
        #     if encoder_outputs:
        #         inputs_embeds = self.model.get_input_embeddings(
        #             input_ids, encoder_outputs)
        #     else:
        #         inputs_embeds = self.model.get_input_embeddings(input_ids)
        #     # TODO(woosuk): Avoid the copy. Optimize.
        #     self.inputs_embeds[:num_scheduled_tokens].copy_(inputs_embeds)
        #     inputs_embeds = self.inputs_embeds[:num_input_tokens]
        #     input_ids = None
        # else:
        #     # For text-only models, we use token ids as input.
        #     # While it is possible to use embeddings as input just like the
        #     # multimodal models, it is not desirable for performance since
        #     # then the embedding layer is not included in the CUDA graph.
        #     input_ids = self.input_ids[:num_input_tokens]
        #     inputs_embeds = None

        ######################### DECODES #########################
        # Decodes run as one single batch with [padded_batch, 1]
        sampled_token_ids_list = []
        if decode_data.num_decodes > 0:
            # FORWARD.
            selected_token_ids = self.model(decode_data.token_ids,
                                            decode_data.position_ids,
                                            decode_data.attn_metadata,
                                            self.kv_caches)

            print("DECODE selected_token_ids.shape = {}".format(selected_token_ids.shape))
            # NOTE: TPU<>CPU sync happens here.
            # We need to call .cpu() first to avoid recompilation.
            token_ids = selected_token_ids.cpu()[:decode_data.num_decodes]
            sampled_token_ids_list.extend(token_ids.tolist())
            # sampled_token_ids[:decode_data.num_decodes] = token_ids

            # UPDATE REQUEST STATE.
            for i, req_id in enumerate(
                    self.input_batch.req_ids[:decode_data.num_decodes]):
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
        for idx, (req_id, prompt_len, token_ids, position_ids,
                  attn_metadata) in enumerate(prefill_data.zipped()):
            # FORWARD.
            selected_token_ids = self.model(token_ids, position_ids,
                                            attn_metadata, self.kv_caches)

            print("PREFILL selected_token_ids.shape = {}".format(selected_token_ids.shape))
            # NOTE: TPU<>CPU sync happens here.
            # We need to call .cpu() first to avoid recompilation.
            token_id = selected_token_ids.cpu()[prompt_len-1].item()
            sampled_token_ids_list.append(token_id)
            # sampled_token_ids[decode_data.num_decodes + idx] = token_id
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

        # TODO: Remove
        # # Sample the next token and get logprobs if needed.
        # sampling_metadata = self._prepare_sampling(scheduler_output)
        # sampler_output = self.model.sample(
        #     logits=logits,
        #     sampling_metadata=sampling_metadata,
        # )

        # sampled_token_ids = sampler_output.sampled_token_ids
        # # TODO(woosuk): The following loop can be slow since it iterates over
        # # the requests one by one. Optimize.
        # num_reqs = self.input_batch.num_reqs
        # for i, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
        #     assert req_id is not None
        #     req_state = self.requests[req_id]
        #     seq_len = (req_state.num_computed_tokens +
        #                scheduler_output.num_scheduled_tokens[req_id])
        #     assert seq_len <= req_state.num_tokens
        #     if seq_len == req_state.num_tokens:
        #         # Append the sampled token to the output token ids.
        #         token_id = sampled_token_ids[i]
        #         self.input_batch.token_ids_cpu[i, seq_len] = token_id
        #         self.input_batch.num_tokens[i] += 1
        #         req_state.output_token_ids.append(token_id)
        #     else:
        #         # Ignore the sampled token from the partial request.
        #         # Rewind the generator state as if the token was not sampled.
        #         generator = self.input_batch.generators.get(i)
        #         if generator is not None:
        #             # This relies on cuda-specific torch-internal impl details
        #             generator.set_offset(generator.get_offset() - 4)

        # if sampler_output.logprob_token_ids is None:
        #     logprob_token_ids = None
        # else:
        #     logprob_token_ids = sampler_output.logprob_token_ids.cpu()
        # if sampler_output.logprobs is None:
        #     logprobs = None
        # else:
        #     logprobs = sampler_output.logprobs.cpu()

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
        model = ModelWrapper(model)
        self.model = torch.compile(model,
                                   backend="openxla",
                                   fullgraph=True,
                                   dynamic=False)

    # @torch.inference_mode()
    def _dummy_run(
        self,
        batch_size: int,
        seq_len: int,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        exec_mode: ExecutionMode,
    ) -> None:
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
            # input_lens = torch.ones((batch_size, ),
            #                         dtype=torch.int32,
            #                         device=self.device)
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
            # input_lens = torch.ones((batch_size, ),
            #                         dtype=torch.int32,
            #                         device=self.device)
            attn_metadata = PallasMetadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decode_tokens=batch_size * seq_len,
                slot_mapping=slot_mapping,
                multi_modal_placeholder_index_maps=None,
                block_tables=block_tables,
                context_lens=context_lens,
            )

        # t = torch.ones((batch_size, ), dtype=torch.float32, device=self.device)
        # p = torch.ones((batch_size, ), dtype=torch.float32, device=self.device)
        # num_samples = _MAX_NUM_SAMPLES if exec_mode.is_prefill() else 1

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
            # torch._dynamo.mark_dynamic(input_lens, 0)
            torch._dynamo.mark_dynamic(attn_metadata.slot_mapping, 0)
            torch._dynamo.mark_dynamic(attn_metadata.context_lens, 0)
            torch._dynamo.mark_dynamic(attn_metadata.block_tables, 0)
            # torch._dynamo.mark_dynamic(t, 0)
            # torch._dynamo.mark_dynamic(p, 0)

        # Dummy run.
        # TODO: Fix this!
        # self.model(token_ids, position_ids, attn_metadata, input_lens, t, p,
        #            num_samples, kv_caches)
        self.model(token_ids, position_ids, attn_metadata, kv_caches)

    # def profile_run(self) -> None:
    #     """Profile to measure peak memory during forward pass."""

    #     # use an empty tensor instead of `None`` to force Dynamo to pass
    #     # it by reference, rather by specializing on the value `None`.
    #     # the `dtype` argument does not matter, and we use `float32` as
    #     # a placeholder (it has wide hardware support).
    #     # it is important to create tensors inside the loop, rather than
    #     # multiplying the list, to avoid Dynamo from treating them as
    #     # tensor aliasing.
    #     dummy_kv_caches = [(
    #         torch.tensor([], dtype=torch.float32, device=self.device),
    #         torch.tensor([], dtype=torch.float32, device=self.device),
    #     ) for _ in range(self.num_attn_layers)]

    #     # Run empty forward.
    #     self._dummy_run(
    #         batch_size=1,
    #         seq_len=self.max_num_tokens,  # Will be rounded to 16 multiple
    #         kv_caches=dummy_kv_caches,
    #         exec_mode=ExecutionMode.PREFILL)

    def capture_model(self) -> None:
        """Compile the model."""

        logger.info("Compiling the model with different input shapes.")

        # Capture prefill shapes
        start = time.perf_counter()
        for batch_size in [1]:
            seq_len = 16
            while True:
                self._dummy_run(batch_size,
                                seq_len,
                                self.kv_caches,
                                exec_mode=ExecutionMode.PREFILL)
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
            self._dummy_run(batch_size,
                            seq_len,
                            self.kv_caches,
                            exec_mode=ExecutionMode.DECODE)
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


# TODO: This is duplicate from V0, refactor
class ModelWrapper(nn.Module):

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
        # batch_size, seq_len = token_ids.shape
        # Calculate the positions to sample from.
        # start_indicies = torch.arange(
        #     batch_size, dtype=torch.int32, device=input_lens.device) * seq_len
        # logits_indices = start_indicies + input_lens - 1

        # TODO: Ressurect
        # FIXME(woosuk): This is a temporary hack to avoid using the existing
        # sampler and sampling metadata.
        # sampling_metadata = SamplingMetadata(
        #     seq_groups=[],
        #     selected_token_indices=logits_indices,
        #     categorized_sample_indices={},
        #     num_prompts=attn_metadata.num_prefills,
        # )

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
            token_ids,
            position_ids,
            kv_caches,
            attn_metadata,
        )

        hidden_states = hidden_states.flatten(0, 1)
        logits = self.model.compute_logits(hidden_states, None)

        # Greedy sampling.
        # argmax_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
        # # argmax_token_ids = argmax_token_ids.repeat(1, num_samples)
        # return argmax_token_ids.squeeze(dim=-1)

        ######
        # Greedy sampling.
        argmax_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
        # argmax_token_ids = argmax_token_ids.repeat(1, 1)

        # Zero temperature means greedy decoding. Avoid division by zero.
        # nonzero_t = torch.where(t != 0, t, 1.0)
        # logits = logits / nonzero_t.unsqueeze(dim=1)
        # if _ENABLE_TOP_P:
        #     logits = _apply_top_p(logits, p.unsqueeze(dim=1))

        # # Random sampling.
        # probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        # sampled_token_ids = torch.multinomial(probs,
        #                                       num_samples,
        #                                       replacement=True)
        # if num_samples == 1:
        argmax_token_ids = argmax_token_ids.squeeze(dim=-1)
        # sampled_token_ids = sampled_token_ids.squeeze(dim=-1)
        # next_token_ids = torch.where(t != 0, sampled_token_ids,
        #                              argmax_token_ids)
        return argmax_token_ids
        ####

        # TODO: Ressurect this code
        # hidden_states = hidden_states.flatten(0, 1)
        # logits = self.model.compute_logits(hidden_states, sampling_metadata)

        # # Argmax sampling.
        # argmax_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
        # argmax_token_ids = argmax_token_ids.repeat(1, num_samples)

        # # Zero temperature means greedy decoding. Avoid division by zero.
        # nonzero_t = torch.where(t != 0, t, 1.0)
        # logits = logits / nonzero_t.unsqueeze(dim=1)
        # if _ENABLE_TOP_P:
        #     logits = _apply_top_p(logits, p.unsqueeze(dim=1))

        # # Random sampling.
        # probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        # sampled_token_ids = torch.multinomial(probs,
        #                                       num_samples,
        #                                       replacement=True)
        # if num_samples == 1:
        #     argmax_token_ids = argmax_token_ids.squeeze(dim=-1)
        #     sampled_token_ids = sampled_token_ids.squeeze(dim=-1)
        # next_token_ids = torch.where(t != 0, sampled_token_ids,
        #                              argmax_token_ids)
        # return next_token_ids


# TODO: Duplicate with V0, refactor
def _get_padded_prefill_len(x: int) -> int:
    # NOTE(woosuk): The pallas FlashAttention kernel requires the sequence
    # length to be a multiple of 16. We pad the prompt length to the nearest
    # multiple of 16. This is also good for performance.
    if x <= 16:
        return 16
    return 1 << (x - 1).bit_length()


# TODO: Duplicate with V0, refactor
def _get_padded_batch_size(batch_size: int) -> int:
    # The GMM Pallas kernel requires num_tokens * topk to be a multiple of 16.
    # To meet this requirement in the simplest way, we set the minimal batch
    # size to 8.
    if batch_size <= 8:
        return 8
    else:
        return ((batch_size + 15) // 16) * 16


# TODO: Duplicate with V0, refactor
def _apply_top_p(logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    logits_sorted = torch.sort(logits, dim=-1, descending=True).values
    sorted_cum_probs = torch.cumsum(logits_sorted.softmax(dim=-1), dim=-1)
    cutoff_index = torch.sum(sorted_cum_probs < p, dim=-1, keepdim=True)
    cutoff_logit = torch.gather(logits_sorted, -1, cutoff_index)
    logits = logits.masked_fill_(logits < cutoff_logit, -float("inf"))
    return logits
