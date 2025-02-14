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
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.model_runner_base import ExecutionMode, ModelRunnerBase

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


class TPUModelRunner(ModelRunnerBase):

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(vllm_config, device)


        # Persistent batch.
        # TODO(xw32): delete this as the base class has already done it.
        # self.input_batch = InputBatch(
        #     max_num_reqs=self.max_num_reqs,
        #     max_model_len=self.max_model_len,
        #     max_num_blocks_per_req=self.max_num_blocks_per_req,
        #     device=self.device,
        #     pin_memory=self.pin_memory,
        #     vocab_size=self.model_config.get_vocab_size(),
        # )
        # print(f'xw32 {self.max_num_reqs=}, {self.max_model_len=}, {self.max_num_blocks_per_req=}, {self.model_config.get_vocab_size()=}') # xw32 self.max_num_reqs=16, self.max_model_len=512, self.max_num_blocks_per_req=32, self.model_config.get_vocab_size()=151936

        # Request states.
        # TODO(xw32): delete this as the base class has already done it.
        # self.requests: Dict[str, CachedRequestState] = {}

        # KV caches for forward pass
        self.kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # xw32: Don't need the `prefill_input_positions` because prefill_input_positions was used in _prepare_prompt_inputs previously and we don't need it with the new kernel.
        # Used to initialize positions for the individual prefills
        # self.prefill_input_positions = torch.tensor(range(self.max_model_len),
        #                                             device="cpu",
        #                                             dtype=torch.int32).reshape(
        #                                                 1, -1)

        # xw32: below are copied from the gpu_model_runner.py. Can be used for
        # capture the graph, etc.
        # Persistent buffers for XLA graphs.
        self.input_ids = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int32,
                                     device=self.device)
        self.positions = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device=self.device)
        
        # In gpu_model_runner.py, inputs_embeds is only used for multimodal. So we don't need it for now.
        # self.inputs_embeds = torch.zeros(
        #     (self.max_num_tokens, self.hidden_size),
        #     dtype=self.dtype,
        #     device=self.device)
        
        # OPTIMIZATION: Cache the tensors rather than creating them every step.
        self.arange_np = np.arange(max(self.max_num_reqs + 1,
                                       self.max_model_len),
                                   dtype=np.int32)
        # NOTE(woosuk): These tensors are "stateless", i.e., they are literally
        # a faster version of creating a new tensor every time. Thus, we should
        # not make any assumptions about the values in these tensors.
        self.input_ids_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int32,
                                         device="cpu",
                                         pin_memory=self.pin_memory)
        self.input_ids_np = self.input_ids_cpu.numpy()
        self.positions_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int64,
                                         device="cpu",
                                         pin_memory=self.pin_memory)
        self.positions_np = self.positions_cpu.numpy()
        self.slot_mapping_cpu = torch.zeros(self.max_num_tokens,
                                            dtype=torch.int64,
                                            device="cpu",
                                            pin_memory=self.pin_memory)
        self.slot_mapping_np = self.slot_mapping_cpu.numpy()
        self.query_start_loc_cpu = torch.zeros(self.max_num_reqs + 1,
                                               dtype=torch.int32,
                                               device="cpu",
                                               pin_memory=self.pin_memory)
        self.query_start_loc_np = self.query_start_loc_cpu.numpy()
        self.seq_lens_cpu = torch.zeros(self.max_num_reqs,
                                        dtype=torch.int32,
                                        device="cpu",
                                        pin_memory=self.pin_memory)
        self.seq_lens_np = self.seq_lens_cpu.numpy()

    def _prepare_inputs(self, scheduler_output: "SchedulerOutput"):

        
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit(num_reqs)

        # Get the number of scheduled tokens for each request.
        # TODO: The Python loop can be slow. Optimize.
        num_scheduled_tokens = []
        max_num_scheduled_tokens = 0
        for req_id in self.input_batch.req_ids[:num_reqs]:
            assert req_id is not None
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_scheduled_tokens.append(num_tokens)
            max_num_scheduled_tokens = max(max_num_scheduled_tokens,
                                           num_tokens)
        num_scheduled_tokens = np.array(num_scheduled_tokens, dtype=np.int32)
        assert max_num_scheduled_tokens > 0

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)

        # Get batched arange.
        # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = np.concatenate(
            [self.arange_np[:n] for n in num_scheduled_tokens])
        
        # Get positions.
        # TODO(xw32): add an example of the output positions_np.
        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)
        
        # Do we need to check self.model_config.uses_mrope?

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
        np.cumsum(num_scheduled_tokens,
                  out=self.query_start_loc_np[1:num_reqs + 1])
        
        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)
        max_seq_len = self.seq_lens_np[:num_reqs].max()

        # Copy the tensors to the TPU.
        input_ids = self.input_ids[:total_num_scheduled_tokens].to(self.device)
        position_ids = self.positions[:total_num_scheduled_tokens].to(self.device)
        query_start_loc = self.query_start_loc_cpu.to(self.device)
        seq_lens = self.seq_lens_cpu.to(self.device)
        slot_mapping = self.slot_mapping_cpu[:total_num_scheduled_tokens].to(self.device)

        attn_metadata = PallasMetadata(
            num_actual_tokens=total_num_scheduled_tokens,
            max_query_len=max_num_scheduled_tokens,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            num_seqs=num_reqs,
            block_table=(
                self.input_batch.block_table.get_device_tensor()[:num_reqs]),
            slot_mapping=slot_mapping,
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
        self.update_states(scheduler_output)

        # Prepare inputs
        # prompt_data = self._prepare_prompt_inputs(scheduler_output)
        # decode_data = self._prepare_decode_inputs(scheduler_output)
        attn_metadata, logits_indices = self._prepare_inputs(scheduler_output)
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        num_input_tokens = num_scheduled_tokens
        attn_metadata.num_input_tokens = num_input_tokens

        input_ids = self.input_ids[:num_input_tokens]
        inputs_embeds = None

        # Run the decoder
        with set_forward_context(attn_metadata, self.vllm_config): 
            positions = self.positions[:num_input_tokens]
            selected_token_ids = self.model(
                token_ids=input_ids,
                position_ids=positions,
                kv_caches=self.kv_caches,
                attn_metadata=None,
            )

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
        exec_mode: Optional[ExecutionMode] = None,
    ) -> None:
        logger.info(f"xw32 TPUModelRunner.dummy_run. {self.input_ids.shape=}, {self.positions.shape=}, {num_tokens=}, {self.input_ids.device=}")
        # xw32 qq: what are input_ids and positions and slot_mapping? What are their shapes? Here is the answer:
        # xw32 TPUModelRunner.dummy_run. self.input_ids.shape=torch.Size([8192]), self.positions.shape=torch.Size([8192]), num_tokens=16, 32, ..., self.input_ids.device=device(type='xla', index=0)
        input_ids = self.input_ids[:num_tokens]
        position_ids = self.positions[:num_tokens]
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
        # attn_metadata = PallasMetadata(
        #    num_prefills=num_tokens,
        #    num_prefill_tokens=num_tokens * seq_len,
        #    num_decode_tokens=0,
        #    slot_mapping=slot_mapping,
        #    multi_modal_placeholder_index_maps=None,
        #    enable_kv_scales_calculation=True,
        #    block_tables=None,
        #    context_lens=None,
        #    effective_query_lens=None,
        #)
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

        # old code begins. TODO(xw32): delete
        # exec_mode = ExecutionMode(exec_mode)
        # if exec_mode.is_prefill():
        #     seq_len = (seq_len + 15) // 16 * 16
        #     token_ids = torch.zeros((num_tokens, seq_len),
        #                             dtype=torch.int32,
        #                             device=self.device)
        #     position_ids = torch.zeros((num_tokens, seq_len),
        #                                dtype=torch.int32,
        #                                device=self.device)
        #     slot_mapping = torch.zeros((num_tokens, seq_len),
        #                                dtype=torch.int64,
        #                                device=self.device)
        #     if exec_mode == ExecutionMode.PREFILL:
        #         attn_metadata = PallasMetadata(
        #             num_prefills=num_tokens,
        #             num_prefill_tokens=num_tokens * seq_len,
        #             num_decode_tokens=0,
        #             slot_mapping=slot_mapping,
        #             multi_modal_placeholder_index_maps=None,
        #             enable_kv_scales_calculation=True,
        #             block_tables=None,
        #             context_lens=None,
        #             effective_query_lens=None,
        #         )

        #     else:  # PREFIX_PREFILL
        #         context_lens = torch.ones((num_tokens, ),
        #                                   dtype=torch.int32,
        #                                   device=self.device)

        #         block_tables = torch.zeros(
        #             (num_tokens, self.max_num_blocks_per_req),
        #             dtype=torch.int32,
        #             device=self.device)

        #         effective_query_lens = torch.ones_like(context_lens)

        #         attn_metadata = PallasMetadata(
        #             num_prefills=num_tokens,
        #             num_prefill_tokens=num_tokens * seq_len,
        #             num_decode_tokens=0,
        #             slot_mapping=slot_mapping,
        #             multi_modal_placeholder_index_maps=None,
        #             enable_kv_scales_calculation=True,
        #             block_tables=block_tables,
        #             context_lens=context_lens,
        #             effective_query_lens=effective_query_lens,
        #         )
        # else:
        #     assert seq_len == 1
        #     token_ids = torch.zeros((num_tokens, seq_len),
        #                             dtype=torch.int32,
        #                             device=self.device)
        #     position_ids = torch.zeros((num_tokens, seq_len),
        #                                dtype=torch.int32,
        #                                device=self.device)
        #     slot_mapping = torch.zeros((num_tokens, seq_len),
        #                                dtype=torch.int64,
        #                                device=self.device)
        #     block_tables = torch.zeros(
        #         (num_tokens, self.max_num_blocks_per_req),
        #         dtype=torch.int32,
        #         device=self.device)
        #     context_lens = torch.ones((num_tokens, ),
        #                               dtype=torch.int32,
        #                               device=self.device)
        #     attn_metadata = PallasMetadata(
        #         num_prefills=0,
        #         num_prefill_tokens=0,
        #         num_decode_tokens=num_tokens * seq_len,
        #         slot_mapping=slot_mapping,
        #         multi_modal_placeholder_index_maps=None,
        #         enable_kv_scales_calculation=True,
        #         block_tables=block_tables,
        #         context_lens=context_lens,
        #     )

        # # NOTE(woosuk): There are two stages of compilation: torch.compile and
        # # XLA compilation. Using `mark_dynamic` can reduce the torch.compile
        # # overhead by reusing the FX graph for different shapes.
        # # However, the XLA graph will still require static shapes and needs to
        # # be re-compiled for every different shapes. This overhead is inevitable
        # # in the first run, but can be skipped afterwards as we cache the XLA
        # # graphs in the disk (VLLM_XLA_CACHE_PATH).
        # if exec_mode.is_prefill():
        #     # Prefll
        #     torch._dynamo.mark_dynamic(token_ids, 1)
        #     torch._dynamo.mark_dynamic(position_ids, 1)
        #     torch._dynamo.mark_dynamic(attn_metadata.slot_mapping, 1)
        # else:
        #     # Decode
        #     torch._dynamo.mark_dynamic(token_ids, 0)
        #     torch._dynamo.mark_dynamic(position_ids, 0)
        #     torch._dynamo.mark_dynamic(attn_metadata.slot_mapping, 0)
        #     torch._dynamo.mark_dynamic(attn_metadata.context_lens, 0)
        #     torch._dynamo.mark_dynamic(attn_metadata.block_tables, 0)

        # # TODO: Remove the attn_metadata above
        # with set_forward_context(None, self.vllm_config):
        #     assert self.model is not None
        #     self.model(token_ids, position_ids, None, kv_caches)
        # old code ends

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
