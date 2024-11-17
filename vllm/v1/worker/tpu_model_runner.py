import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
import torch_xla.core.xla_model as xm

from vllm.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import MultiModalDataDict
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, cdiv, is_pin_memory_available
from vllm.v1.attention.backends.pallas import (PallasAttentionBackend,
                                               PallasAttentionMetadata)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.sample.metadata import SamplingMetadata

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput

logger = init_logger(__name__)

# Here we utilize the behavior that out-of-bound index is ignored.
# FIXME: Find a more reliable way to prevent possible bugs.
_PAD_SLOT_ID = 1_000_000_000


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
    attn_metadata: PallasAttentionMetadata = None


class TPUModelRunner:

    def __init__(
        self,
        vllm_config: VllmConfig,
    ):
        # TODO: use ModelRunnerBase.__init__(self, vllm_config=vllm_config)
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

        # List[k_cache, v_cache]
        self.kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []

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

        self.prefill_positions = torch.tensor(
            range(self.max_model_len),
            device="cpu",
        ).to(torch.int32).reshape(1, -1)

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        # Remove stopped requests from the cached states.
        # Keep the states of the pre-empted requests.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)

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
                multi_modal_data=req_data.multi_modal_data,
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

        # THIS MOVES ALL THE DECODES TO THE FIRST N IN BATCH.
        # Condense the batched states if there are empty indices.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

        # ALL THE PREFILLS ARE THE LAST M IN THE BATCH.
        # These are added at the end after the bacth is condensed.
        self.input_batch.num_prefills = len(req_ids_to_add)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            self.input_batch.add_request(req_state, None)

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
        num_decodes = self.input_batch.num_decodes
        for idx in range(num_decodes, num_reqs):
            prefill_request_ids.append(self.input_batch.req_ids[idx])

            # STATIC SHAPE: prefills are padded to the next power of 2.
            prompt_len = num_scheduled_tokens[idx]
            padded_prompt_len = _get_padded_prefill_len(prompt_len)
            prefill_prompt_lens.append(prompt_len)
            assert padded_prompt_len <= self.max_model_len

            # TOKEN_IDS.
            token_ids = torch.from_numpy(self.input_batch.token_ids_cpu[
                idx, :padded_prompt_len].reshape(1, -1))
            prefill_token_ids.append(token_ids.to(self.device))

            # POSITIONS.
            positions = self.prefill_positions[:, :padded_prompt_len]
            prefill_position_ids.append(positions.to(self.device))

            # SLOT_MAPPING.
            # The "slot" is the "physical index" of a token in the KV cache.
            # Look up the block_idx in the block table (logical<>physical map)
            # to compute this.
            block_numbers = self.input_batch.block_table_cpu_tensor[
                idx, positions // self.block_size].reshape(1, -1)
            block_offsets = positions % self.block_size
            slot_mapping = block_numbers * self.block_size + block_offsets
            # Set an out of range value for the padding tokens so that they
            # are ignored when inserting into the KV cache.
            slot_mapping[:, prompt_len:] = _PAD_SLOT_ID
            slot_mapping = slot_mapping.long()

            # ATTN_METADATA.
            prefill_attn_metadata.append(
                PallasAttentionMetadata(
                    is_prompt=True,
                    slot_mapping=slot_mapping.to(self.device),
                    block_tables=None,
                    context_lens=None,
                ))

        return PrefillInputData(
            request_ids=prefill_request_ids,
            prompt_lens=prefill_prompt_lens,
            token_ids=prefill_token_ids,
            position_ids=prefill_position_ids,
            attn_metadata=prefill_attn_metadata,
        )

    def _prepare_decode_inputs(self, num_decodes: int) -> DecodeInputData:
        # Decodes run as one single padded batch with shape [batch, 1]
        #
        # We need to set _PAD_SLOT_ID for the padding tokens in the
        # slot_mapping, such that the attention KV cache insertion
        # logic knows to ignore those indicies. Otherwise, the
        # padding data can be dummy since we have a causal mask.

        if num_decodes == 0:
            return DecodeInputData(num_decodes=0)

        # PAD FOR STATIC SHAPES.
        padded_batch_size = _get_padded_batch_size(num_decodes)

        # POSITIONS. [batch, 1]
        # We slice at the end, since we use the positions for gathering.
        positions = torch.from_numpy(
            self.input_batch.num_computed_tokens_cpu.reshape(-1, 1))
        index = positions.to(torch.int64)
        positions = positions[:padded_batch_size]

        # TOKEN_IDS. [batch, 1]
        token_ids = torch.gather(
            input=torch.from_numpy(self.input_batch.token_ids_cpu),
            dim=1,
            index=index,
        )[:padded_batch_size]

        # SLOT_MAPPING [batch, 1]
        # The "slot" is the "physical index" of a token in the KV cache.
        # Look up the block_idx in the block table (logical<>physical map)
        # to compute this.
        block_number = torch.gather(
            input=self.input_batch.block_table_cpu_tensor,
            dim=1,
            index=(index // self.block_size))
        block_offsets = index % self.block_size
        slot_mapping = block_number * self.block_size + block_offsets
        # Set an out of range value for the padding tokens so that they
        # are ignored when inserting into the KV cache.
        slot_mapping[num_decodes:] = _PAD_SLOT_ID
        slot_mapping = slot_mapping[:padded_batch_size]

        # BLOCK_TABLE [batch, max_num_blocks_per_req]
        block_table = self.input_batch.block_table_cpu_tensor[:
                                                              padded_batch_size]

        # CONTEXT_LENS [batch_size]
        context_lens = (positions.reshape(-1) + 1)

        # CPU<>TPU sync happens here.
        return DecodeInputData(num_decodes=num_decodes,
                               token_ids=token_ids.to(self.device),
                               position_ids=positions.to(self.device),
                               attn_metadata=PallasAttentionMetadata(
                                   is_prompt=False,
                                   slot_mapping=slot_mapping.to(self.device),
                                   block_tables=block_table.to(self.device),
                                   context_lens=context_lens.to(self.device),
                               ))

    def _prepare_inputs(
        self, scheduler_output: "SchedulerOutput"
    ) -> Tuple[PrefillInputData, Optional[DecodeInputData]]:

        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0

        num_reqs = self.input_batch.num_reqs
        num_decodes = self.input_batch.num_decodes

        # Get the number of scheduled tokens for each request.
        # TODO: The Python loop can be slow. Optimize.
        num_scheduled_tokens = []
        for idx, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_scheduled_tokens.append(num_tokens)

            # NOTE: assert that all the decodes are "decodes".
            if idx < num_decodes:
                assert num_tokens == 1

        return (
            self._prepare_prefill_inputs(num_scheduled_tokens),
            self._prepare_decode_inputs(num_decodes),
        )

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

    @torch.no_grad()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput:
        self._update_states(scheduler_output)
        prefill_data, decode_data = self._prepare_inputs(scheduler_output)
        num_reqs = self.input_batch.num_reqs
        sampled_token_ids = torch.empty(num_reqs, dtype=torch.int32)

        ######################### DECODES #########################
        # Decodes run as one single batch with [padded_batch, 1]
        if decode_data.num_decodes > 0:

            # FORWARD.
            selected_token_ids = self.model(decode_data.token_ids,
                                            decode_data.position_ids,
                                            decode_data.attn_metadata,
                                            self.kv_caches,
                                            is_prompt=False)

            # NOTE: TPU<>CPU sync happens here.
            # We need to call .cpu() first to avoid recompilation.
            token_ids = selected_token_ids.cpu()[:decode_data.num_decodes]
            sampled_token_ids_list = token_ids.tolist()
            sampled_token_ids[:decode_data.num_decodes] = token_ids

            # UPDATE REQUEST STATE.
            for i, req_id in enumerate(
                    self.input_batch.req_ids[:decode_data.num_decodes]):
                req_state = self.requests[req_id]

                # TODO: ASSERT NO CHUNKED PREFILL.
                assert scheduler_output.num_scheduled_tokens[req_id] == 1
                seq_len = (req_state.num_computed_tokens +
                           scheduler_output.num_scheduled_tokens[req_id])
                assert seq_len == req_state.num_tokens

                token_id = sampled_token_ids_list[i]
                self.input_batch.token_ids_cpu[i, seq_len] = token_id
                req_state.output_token_ids.append(token_id)

        ######################### PREFILLS #########################
        # Prefills run separately with shape [1, padded_prefill_len],
        # due to lack of variable length attention kernel so far.
        for idx, (req_id, prompt_len, token_ids, position_ids,
                  attn_metadata) in enumerate(prefill_data.zipped()):

            # FORWARD.
            selected_token_ids = self.model(token_ids,
                                            position_ids,
                                            attn_metadata,
                                            self.kv_caches,
                                            is_prompt=True)

            # NOTE: TPU<>CPU sync happens here.
            # We need to call .cpu() first to avoid recompilation.
            token_id = selected_token_ids.cpu()[prompt_len - 1].item()
            sampled_token_ids[decode_data.num_decodes + idx] = token_id
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
            req_state.output_token_ids.append(token_id)

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids[:num_reqs],
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids_cpu=sampled_token_ids,
            logprob_token_ids_cpu=None,
            logprobs_cpu=None,
        )

    def load_model(self) -> None:

        # NOTE(woosuk): While the executor assigns the TP ranks to the worker
        # process, the ranks can be different from the ranks internally assigned
        # by the xm runtime. Therefore, there is a mismatch in the rank
        # assignment between the gloo (cpu) runtime and the xm (tpu) runtime.
        # This is not a problem in linear layers because all-reduce is
        # rank-agnostic. However, it matters for all-gather as the ranks
        # determine the order of concatenating the output tensors.
        # As a workaround, we use the xm's rank assignment only when loading
        # the embedding weights.

        # xm_tp_rank = xr.global_ordinal()
        # with patch(
        #         "vllm.model_executor.layers.vocab_parallel_embedding."
        #         "get_tensor_model_parallel_rank",
        #         return_value=xm_tp_rank):
        #     model = get_model(vllm_config=self.vllm_config)
        model = get_model(vllm_config=self.vllm_config)
        model = model.eval()
        xm.wait_device_ops()
        self.model = ModelWrapper(model)

    def _dummy_run(self, batch_size: int, seq_len: int,
                   kv_caches: List[torch.Tensor], is_prompt: bool) -> None:
        """Dummy warmup run for memory usage and graph compilation."""

        input_ids = torch.zeros((batch_size, seq_len),
                                dtype=torch.int32,
                                device=self.device)
        position_ids = torch.zeros((batch_size, seq_len),
                                   dtype=torch.int32,
                                   device=self.device)
        slot_mapping = torch.zeros((batch_size, seq_len),
                                   dtype=torch.int64,
                                   device=self.device)
        block_tables = None if is_prompt else torch.zeros(
            (batch_size, self.max_num_blocks_per_req),
            dtype=torch.int32,
            device=self.device,
        )
        context_lens = None if is_prompt else torch.ones(
            (batch_size, ),
            dtype=torch.int32,
            device=self.device,
        )
        attn_metadata = PallasAttentionMetadata(
            is_prompt=is_prompt,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
        )

        # NOTE: There are two stages of compilation: torch.compile and
        # XLA compilation. Using `mark_dynamic` can reduce the torch.compile
        # overhead by reusing the FX graph for different shapes.
        # However, the XLA graph will still require static shapes and needs to
        # be re-compiled for every different shapes. This overhead is inevitable
        # in the first run, but can be skipped afterwards as we cache the XLA
        # graphs in the disk (VLLM_XLA_CACHE_PATH).
        if is_prompt:
            torch._dynamo.mark_dynamic(input_ids, 1)
            torch._dynamo.mark_dynamic(position_ids, 1)
            torch._dynamo.mark_dynamic(attn_metadata.slot_mapping, 1)
        else:
            torch._dynamo.mark_dynamic(input_ids, 0)
            torch._dynamo.mark_dynamic(position_ids, 0)
            torch._dynamo.mark_dynamic(attn_metadata.slot_mapping, 0)
            torch._dynamo.mark_dynamic(attn_metadata.context_lens, 0)
            torch._dynamo.mark_dynamic(attn_metadata.block_tables, 0)

        # Dummy run.
        self.model(input_ids,
                   position_ids,
                   attn_metadata,
                   kv_caches,
                   is_prompt=is_prompt)

    def profile_run(self) -> None:
        """Profile to measure peak memory during forward pass."""

        # use an empty tensor instead of `None`` to force Dynamo to pass
        # it by reference, rather by specializing on the value `None`.
        # the `dtype` argument does not matter, and we use `float32` as
        # a placeholder (it has wide hardware support).
        # it is important to create tensors inside the loop, rather than
        # multiplying the list, to avoid Dynamo from treating them as
        # tensor aliasing.
        dummy_kv_caches = [(
            torch.tensor([], dtype=torch.float32, device=self.device),
            torch.tensor([], dtype=torch.float32, device=self.device),
        ) for _ in range(self.num_attn_layers)]

        # Round to multiple of 16.
        seq_len = (self.max_num_tokens + 15) // 16 * 16

        # Run empty forward.
        self._dummy_run(batch_size=1,
                        seq_len=seq_len,
                        kv_caches=dummy_kv_caches,
                        is_prompt=True)

    def capture_model(self) -> None:
        """Compile the model."""

        logger.info("Compiling the model with different input shapes.")

        # Prefill shapes.
        start = time.perf_counter()
        for batch_size in [1]:
            seq_len = 16
            while True:
                self._dummy_run(batch_size,
                                seq_len,
                                self.kv_caches,
                                is_prompt=True)
                xm.wait_device_ops()
                logger.info("batch_size: %d, seq_len: %d", batch_size, seq_len)
                if seq_len >= self.model_config.max_model_len:
                    break
                num_tokens = batch_size * seq_len
                if num_tokens >= self.scheduler_config.max_num_batched_tokens:
                    break
                seq_len = seq_len * 2

        end = time.perf_counter()
        logger.info("Compilation for prefill done in %.2f s.", end - start)

        # Decode shapes.
        start = time.time()
        seq_len = 1
        batch_size = 8  # Must be in sync with _get_padded_batch_size()
        while True:
            self._dummy_run(batch_size,
                            seq_len,
                            self.kv_caches,
                            is_prompt=False)
            xm.wait_device_ops()
            logger.info("batch_size: %d, seq_len: %d", batch_size, seq_len)

            if batch_size >= self.scheduler_config.max_num_seqs:
                break
            batch_size = batch_size + 16 if batch_size >= 16 else batch_size * 2

        end = time.time()
        logger.info("Compilation for decode done in %.2f s.", end - start)

    def initialize_kv_cache(self, num_blocks: int) -> None:
        assert len(self.kv_caches) == 0
        kv_cache_shape = PallasAttentionBackend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        for _ in range(self.num_attn_layers):
            self.kv_caches.append((
                torch.zeros(kv_cache_shape,
                            dtype=self.kv_cache_dtype,
                            device=self.device),
                torch.zeros(kv_cache_shape,
                            dtype=self.kv_cache_dtype,
                            device=self.device),
            ))


@dataclass
class CachedRequestState:

    req_id: str
    prompt_token_ids: List[int]
    prompt: Optional[str]
    multi_modal_data: Optional["MultiModalDataDict"]
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

        self.token_ids_cpu = np.zeros((max_num_reqs, max_model_len),
                                      dtype=np.int32)
        self.num_computed_tokens_cpu = np.zeros(max_num_reqs, dtype=np.int32)

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

        self.num_prefills = 0

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
    def num_decodes(self) -> int:
        return self.num_reqs - self.num_prefills

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


class ModelWrapper(TorchCompileWrapperWithCustomDispatcher):

    def __init__(self, model: nn.Module):
        self.model = model
        compiled_callable = torch.compile(self.forward,
                                          backend="openxla",
                                          fullgraph=True,
                                          dynamic=False)
        super().__init__(compiled_callable)

    def __call__(self, *args, is_prompt: bool, **kwargs):
        if len(self.compiled_codes) < 3 or not self.use_custom_dispatcher:
            # not fully compiled yet, or not using the custom dispatcher,
            # let PyTorch handle it
            return self.compiled_callable(*args, **kwargs)
        # the 3 compiled codes are:
        # 0: for profiling
        # 1: for prompt
        # 2: for decode
        # dispatch to the compiled code directly, skip PyTorch
        if is_prompt:
            with self.dispatch_to_code(1):
                return self.forward(*args, **kwargs)
        else:
            with self.dispatch_to_code(2):
                return self.forward(*args, **kwargs)

    def forward(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: PallasAttentionMetadata,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Executes the forward pass of the model and samples the next token.

        Args:
            token_ids: The input token IDs of shape [batch_size, seq_len].
            position_ids: The input position IDs of shape [batch_size, seq_len].
            attn_metadata: The Pallas attention metadata.
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
            token_ids,
            position_ids,
            kv_caches,
            attn_metadata,
        )
        hidden_states = hidden_states.flatten(0, 1)
        logits = self.model.compute_logits(hidden_states, None)

        # Greedy sampling.
        argmax_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
        return argmax_token_ids.squeeze(dim=1)


def _get_padded_batch_size(batch_size: int) -> int:
    # The GMM Pallas kernel requires num_tokens * topk to be a multiple of 16.
    # To meet this requirement in the simplest way, we set the minimal batch
    # size to 8.
    if batch_size <= 8:
        return 8
    else:
        return ((batch_size + 15) // 16) * 16


def _get_padded_prefill_len(x: int) -> int:
    # NOTE(woosuk): The pallas FlashAttention kernel requires the sequence
    # length to be a multiple of 16. We pad the prompt length to the nearest
    # multiple of 16. This is also good for performance.
    if x <= 16:
        return 16
    return 1 << (x - 1).bit_length()
