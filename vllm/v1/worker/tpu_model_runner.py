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
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.attention.backends.pallas import (PallasAttentionBackend,
                                               PallasMetadata)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.gpu_input_batch import (CachedRequestState, InputBatch,
                                            ensure_decodes_first)
from vllm.v1.worker.model_runner_base import ExecutionMode, ModelRunnerBase

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput

logger = init_logger(__name__)

# Here we utilize the behavior that out-of-bound index is ignored.
# FIXME(woosuk): Find a more reliable way to prevent possible bugs.
_PAD_SLOT_ID = 1_000_000_000


@dataclass
class PromptDecodeInfo:
    prompt_req_ids: List[str]
    decode_req_ids: List[str]
    prompt_scheduled_tokens: List[int]


@dataclass
class PromptData:
    input_tokens: torch.Tensor
    input_positions: torch.Tensor
    attn_metadata: PallasMetadata


@dataclass
class DecodeData:
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    attn_metadata: Optional[PallasMetadata] = None


class TPUModelRunner(ModelRunnerBase):

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(vllm_config, device)

        # KV caches for forward pass
        self.kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # Cached torch/numpy tensors
        self.input_ids_cpu = torch.empty(self.max_num_tokens,
                                         dtype=torch.int32,
                                         device="cpu")
        self.input_ids_np = self.input_ids_cpu.numpy()

        self.input_positions_cpu = torch.empty(self.max_num_tokens,
                                               dtype=torch.int32,
                                               device="cpu")
        self.input_positions_np = self.input_positions_cpu.numpy()

        self.slot_mapping_cpu = torch.empty(self.max_num_tokens,
                                            dtype=torch.int64,
                                            device="cpu")
        self.slot_mapping_np = self.slot_mapping_cpu.numpy()

        self.prompt_context_lens_cpu = torch.empty((1),
                                                   dtype=torch.int32,
                                                   device="cpu")
        self.prompt_effective_query_lens_cpu = torch.empty((1),
                                                           dtype=torch.int32,
                                                           device="cpu")

        self.decode_context_lens_cpu = torch.empty(self.max_num_tokens,
                                                   dtype=torch.int32,
                                                   device="cpu")
        self.decode_context_lens_np = self.decode_context_lens_cpu.numpy()

        # Range tensor with values [0 .. self.max_num_tokens - 1].
        # Used to initialize positions / context_lens / seq_lens
        self.arange_np = np.arange(self.max_num_tokens, dtype=np.int32)

        # Cached lists
        self.req_ids = []
        self.prompt_token_ids = []
        self.sampled_token_ids = []

    def _get_prompts_and_decodes(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> PromptDecodeInfo:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # Traverse decodes first
        decode_req_ids = []
        for i in range(num_reqs):
            req_id = self.input_batch.req_ids[i]

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[i]
            num_prompt_tokens = self.input_batch.num_prompt_tokens[i]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]

            if num_computed_tokens < num_prompt_tokens:
                # This is prompt
                break

            # This is decode
            assert num_scheduled_tokens == 1
            decode_req_ids.append(req_id)

        # Traverse prompts
        prompt_req_ids = []
        prompt_scheduled_tokens = []
        for i in range(len(decode_req_ids), num_reqs):
            req_id = self.input_batch.req_ids[i]

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[i]
            num_prompt_tokens = self.input_batch.num_prompt_tokens[i]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]

            # Must be prompt
            assert num_computed_tokens < num_prompt_tokens

            prompt_req_ids.append(req_id)
            prompt_scheduled_tokens.append(num_scheduled_tokens)

        return PromptDecodeInfo(prompt_req_ids, decode_req_ids,
                                prompt_scheduled_tokens)

    def _prepare_prompt(self, req_index: int,
                        num_scheduled_tokens: int) -> PromptData:
        num_computed_tokens = self.input_batch.num_computed_tokens_cpu[
            req_index]
        num_prompt_tokens = self.input_batch.num_prompt_tokens[req_index]

        # Must be prompt
        assert num_computed_tokens < num_prompt_tokens

        # Prompt len
        prompt_len = num_scheduled_tokens
        padded_prompt_len = _get_padded_prompt_len(prompt_len)
        assert padded_prompt_len <= self.max_model_len

        # Seq len
        seq_len = num_computed_tokens + prompt_len
        padded_seq_len = num_computed_tokens + padded_prompt_len

        # Input tokens
        input_tokens_cpu = self.input_batch.token_ids_cpu_tensor[
            req_index, num_computed_tokens:padded_seq_len]
        input_tokens_cpu[prompt_len:] = 0

        # Input positions
        input_positions_np = self.input_positions_np[:padded_prompt_len]
        np.add(num_computed_tokens,
               self.arange_np[:padded_prompt_len],
               out=input_positions_np)
        input_positions_np[prompt_len:] = 0

        # Slot mapping
        block_table_np = \
            self.input_batch.block_table.get_numpy_array()
        block_numbers_np = block_table_np[req_index, input_positions_np //
                                          self.block_size]
        block_offsets_np = input_positions_np % self.block_size

        slot_mapping_np = self.slot_mapping_np[:padded_prompt_len]
        np.add(block_numbers_np * self.block_size,
               block_offsets_np,
               out=slot_mapping_np)
        slot_mapping_np[prompt_len:] = _PAD_SLOT_ID

        # Block table
        block_table_cpu = None
        if num_computed_tokens > 0:
            block_table_cpu = self.input_batch.block_table.get_cpu_tensor()
            block_table_cpu = block_table_cpu[req_index]

        # Context len
        self.prompt_context_lens_cpu[0] = 0
        if num_computed_tokens > 0:
            self.prompt_context_lens_cpu[0] = seq_len

        # Effective query len
        self.prompt_effective_query_lens_cpu[0] = prompt_len

        # Get final tensors
        input_tokens = input_tokens_cpu.reshape(1, -1).to(self.device)
        input_positions = self.input_positions_cpu[:padded_prompt_len].reshape(
            1, -1).to(self.device)
        slot_mapping = self.slot_mapping_cpu[:padded_prompt_len].reshape(
            1, -1).to(self.device)
        block_table = block_table_cpu.reshape(1, -1).to(
            self.device) if block_table_cpu is not None else None

        context_lens = self.prompt_context_lens_cpu.reshape(1,
                                                            -1).to(self.device)
        effective_query_lens = self.prompt_effective_query_lens_cpu.reshape(
            1, -1).to(self.device)

        # Attn metadata
        attn_metadata = PallasMetadata(
            num_prefills=1,
            num_prefill_tokens=0,  # NOTE: This is not used.
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=True,
            block_tables=block_table,
            context_lens=context_lens,
            effective_query_lens=effective_query_lens,
        )

        return PromptData(input_tokens, input_positions, attn_metadata)

    def _prepare_decode(
        self,
        decode_req_ids: List[str],
    ) -> DecodeData:
        # Batch size
        batch_size = len(decode_req_ids)
        padded_batch_size = _get_padded_batch_size(batch_size)
        assert padded_batch_size <= self.max_model_len

        # Input positions
        input_positions_np = self.input_positions_np[:padded_batch_size]
        np.add(self.input_batch.num_computed_tokens_cpu[:padded_batch_size],
               0,
               out=input_positions_np)
        input_positions_np[batch_size:] = 0
        input_positions_cpu = self.input_positions_cpu[:padded_batch_size]

        # Input tokens
        input_tokens_cpu = self.input_ids_cpu[:padded_batch_size]
        torch.index_select(self.input_batch.token_ids_cpu_tensor,
                           1,
                           input_positions_cpu,
                           out=input_tokens_cpu)
        input_tokens_cpu[:batch_size] = 0

        # Slot mapping
        block_table_cpu = self.input_batch.block_table.get_cpu_tensor()
        block_numbers_cpu = torch.index_select(
            block_table_cpu, 1, input_positions_cpu // self.block_size)
        block_numbers_np = block_numbers_cpu.numpy()

        block_offsets_np = input_positions_np % self.block_size

        slot_mapping_np = self.slot_mapping_np[:padded_batch_size]
        np.add(block_numbers_np * self.block_size,
               block_offsets_np,
               out=slot_mapping_np)
        slot_mapping_np[:, batch_size:] = _PAD_SLOT_ID

        block_table_cpu = block_table_cpu[:len(decode_req_ids)]

        # Context lens
        context_lens_np = self.decode_context_lens_np[:padded_batch_size]
        np.add(self.input_batch.num_computed_tokens_cpu[:padded_batch_size],
               1,
               out=context_lens_np)
        context_lens_np[batch_size:] = 0

        # Get final tensors
        input_tokens = input_tokens_cpu.to(self.device)
        input_positions = input_positions_cpu.to(self.device)
        slot_mapping = self.slot_mapping_cpu[:padded_batch_size].to(
            self.device)
        block_table = block_table_cpu.to(self.device)
        context_lens = self.decode_context_lens_cpu[:padded_batch_size].to(
            self.device)

        # Attn metadata
        attn_metadata = PallasMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=padded_batch_size,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=True,
            block_tables=block_table,
            context_lens=context_lens,
            effective_query_lens=None,
        )

        return DecodeData(input_tokens=input_tokens,
                          input_positions=input_positions,
                          attn_metadata=attn_metadata)

    @torch.no_grad()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput:
        # Update cached state
        self.update_states(scheduler_output)

        # If necessary, swap decodes/prompts to have all decodes on the start
        ensure_decodes_first(self.input_batch)

        # Prepare prompts/decodes info
        pd_info = self._get_prompts_and_decodes(scheduler_output)

        # Init
        num_prompts = len(pd_info.prompt_req_ids)
        num_decodes = len(pd_info.decode_req_ids)
        decode_token_ids_list = None
        decode_data = None
        self.req_ids.clear()
        self.prompt_token_ids.clear()
        self.sampled_token_ids.clear()

        # Run each prompt individually
        is_first = True
        for i in range(num_prompts):
            req_id = pd_info.prompt_req_ids[i]
            req_index = num_decodes + i
            req_state = self.requests[req_id]
            num_scheduled_tokens = pd_info.prompt_scheduled_tokens[i]
            prompt_len = num_scheduled_tokens
            seq_len = req_state.num_computed_tokens + num_scheduled_tokens

            # Prepare first prompt
            if is_first:
                prompt_data = self._prepare_prompt(req_index,
                                                   num_scheduled_tokens)
                is_first = False

            # Run forward pass
            with set_forward_context(prompt_data.attn_metadata,
                                     self.vllm_config):
                assert self.model is not None
                selected_token_ids = self.model(prompt_data.input_tokens,
                                                prompt_data.input_positions,
                                                prompt_data.attn_metadata,
                                                self.kv_caches)

            # In parallel to TPU execution, prepare the next iteration
            if i < num_prompts - 1:
                # There is next prompt => prepare it
                prompt_data = self._prepare_prompt(
                    req_index + 1, pd_info.prompt_scheduled_tokens[i + 1])
            elif i == num_prompts - 1 and num_decodes > 0:
                # There is next decode => prepare it
                decode_data = self._prepare_decode(pd_info.decode_req_ids)

            # Update cached state (if prompt is fully done)
            if seq_len >= len(req_state.prompt_token_ids):
                # Transfer sampled tokens from TPU to CPU
                selected_token_ids_cpu = selected_token_ids.cpu()

                # Get output token
                token_id = selected_token_ids_cpu[prompt_len - 1].item()
                self.prompt_token_ids.append(token_id)

                # Add output token to the request
                self.input_batch.token_ids_cpu[req_index, seq_len] = token_id
                self.input_batch.num_tokens[req_index] += 1
                req_state.output_token_ids.append(token_id)

        # Run decodes (a single batch)
        if num_decodes > 0:

            # Prepare decode (if was not yet prepared)
            if decode_data is None:
                decode_data = self._prepare_decode(pd_info.decode_req_ids)

            # Run forward pass
            with set_forward_context(decode_data.attn_metadata,
                                     self.vllm_config):
                assert self.model is not None
                selected_token_ids = self.model(decode_data.input_tokens,
                                                decode_data.input_positions,
                                                decode_data.attn_metadata,
                                                self.kv_caches)

            # Transfer sampled tokens from TPU to CPU
            decode_token_ids_cpu = selected_token_ids.cpu()
            # Convert to list
            decode_token_ids_list = decode_token_ids_cpu.tolist()

            # Update cached state for each decode request
            for i in range(num_decodes):
                req_id = pd_info.decode_req_ids[i]
                req_index = i
                req_state = self.requests[req_id]
                seq_len = req_state.num_computed_tokens + 1

                token_id = decode_token_ids_list[i]

                self.input_batch.token_ids_cpu[req_index, seq_len] = token_id
                self.input_batch.num_tokens[req_index] += 1
                req_state.output_token_ids.append(token_id)

        # Create final req_id => token lists.
        # This must match the actual batch index positions,
        # so we put decodes first and then prompts.
        self.req_ids.extend(pd_info.decode_req_ids)
        self.req_ids.extend(pd_info.prompt_req_ids)
        if decode_token_ids_list is not None:
            self.sampled_token_ids.extend(decode_token_ids_list)
        self.sampled_token_ids.extend(self.prompt_token_ids)

        # Create output
        model_runner_output = ModelRunnerOutput(
            req_ids=self.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=self.sampled_token_ids,
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
        num_tokens: int,
        seq_len: Optional[int] = None,
        exec_mode: Optional[ExecutionMode] = None,
    ) -> None:
        assert seq_len is not None
        assert exec_mode is not None

        exec_mode = ExecutionMode(exec_mode)
        if exec_mode.is_prefill():
            seq_len = (seq_len + 15) // 16 * 16
            token_ids = torch.zeros((num_tokens, seq_len),
                                    dtype=torch.int32,
                                    device=self.device)
            position_ids = torch.zeros((num_tokens, seq_len),
                                       dtype=torch.int32,
                                       device=self.device)
            slot_mapping = torch.zeros((num_tokens, seq_len),
                                       dtype=torch.int64,
                                       device=self.device)
            if exec_mode == ExecutionMode.PREFILL:
                attn_metadata = PallasMetadata(
                    num_prefills=num_tokens,
                    num_prefill_tokens=num_tokens * seq_len,
                    num_decode_tokens=0,
                    slot_mapping=slot_mapping,
                    multi_modal_placeholder_index_maps=None,
                    enable_kv_scales_calculation=True,
                    block_tables=None,
                    context_lens=None,
                    effective_query_lens=None,
                )

            else:
                context_lens = torch.ones((num_tokens, ),
                                          dtype=torch.int32,
                                          device=self.device)

                block_tables = torch.zeros(
                    (num_tokens, self.max_num_blocks_per_req),
                    dtype=torch.int32,
                    device=self.device)

                effective_query_lens = torch.ones_like(context_lens)

                attn_metadata = PallasMetadata(
                    num_prefills=num_tokens,
                    num_prefill_tokens=num_tokens * seq_len,
                    num_decode_tokens=0,
                    slot_mapping=slot_mapping,
                    multi_modal_placeholder_index_maps=None,
                    enable_kv_scales_calculation=True,
                    block_tables=block_tables,
                    context_lens=context_lens,
                    effective_query_lens=effective_query_lens,
                )
        else:
            assert seq_len == 1
            token_ids = torch.zeros((num_tokens, seq_len),
                                    dtype=torch.int32,
                                    device=self.device)
            position_ids = torch.zeros((num_tokens, seq_len),
                                       dtype=torch.int32,
                                       device=self.device)
            slot_mapping = torch.zeros((num_tokens, seq_len),
                                       dtype=torch.int64,
                                       device=self.device)
            block_tables = torch.zeros(
                (num_tokens, self.max_num_blocks_per_req),
                dtype=torch.int32,
                device=self.device)
            context_lens = torch.ones((num_tokens, ),
                                      dtype=torch.int32,
                                      device=self.device)
            attn_metadata = PallasMetadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decode_tokens=num_tokens * seq_len,
                slot_mapping=slot_mapping,
                multi_modal_placeholder_index_maps=None,
                enable_kv_scales_calculation=True,
                block_tables=block_tables,
                context_lens=context_lens,
            )

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

        # TODO: Remove the attn_metadata above
        with set_forward_context(None, self.vllm_config):
            assert self.model is not None
            self.model(token_ids, position_ids, None, kv_caches)

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
        if attn_metadata is not None:
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

        assert self.model is not None
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
        argmax_token_ids = argmax_token_ids.squeeze(dim=-1)
        return argmax_token_ids


def _get_padded_prompt_len(x: int) -> int:
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
