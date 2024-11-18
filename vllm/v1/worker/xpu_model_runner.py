import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, List, Optional, Set

from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.attention.backends.ipex_attn import IPEXAttentionMetadata, IPEXAttentionBackend
if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput


class XPUModelRunner(GPUModelRunner):
    """A model runner for XPU devices."""

    def __init__(self, vllm_config):
        super().__init__(vllm_config)
        self.use_cuda_graph = False

    @torch.inference_mode()
    def profile_run(self) -> None:
        self._dummy_run(self.model, self.max_num_tokens)
        torch.xpu.synchronize()

    def initialize_kv_cache(self, num_blocks: int) -> None:
        assert len(self.kv_caches) == 0
        kv_cache_shape = IPEXAttentionBackend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        for _ in range(self.num_attn_layers):
            self.kv_caches.append(
                torch.zeros(kv_cache_shape,
                            dtype=self.kv_cache_dtype,
                            device=self.device))

    def _prepare_inputs(self, scheduler_output: "SchedulerOutput"):
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        print(f"aaaaa :{total_num_scheduled_tokens}")
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
        min_num_scheduled_tokens = 1e9
        for req_id in self.input_batch.req_ids[:num_reqs]:
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_scheduled_tokens.append(num_tokens)
            max_num_scheduled_tokens = max(max_num_scheduled_tokens,
                                           num_tokens)
            min_num_scheduled_tokens = min(min_num_scheduled_tokens,
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
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)
        max_seq_len = seq_lens.max()
        seq_start_loc = torch.empty((num_reqs + 1, ),
                                    dtype=torch.int32,
                                    device="cpu",
                                    pin_memory=self.pin_memory)
        seq_start_loc_np = seq_start_loc.numpy()
        seq_start_loc_np[0] = 0
        np.cumsum(seq_lens, out=seq_start_loc_np[1:])

        # self.input_ids[:total_num_scheduled_tokens].copy_(input_ids,
        #                                                   non_blocking=True)
        input_ids = input_ids.to(self.device, non_blocking=True)
        self.positions[:total_num_scheduled_tokens].copy_(positions,
                                                          non_blocking=True)

        query_start_loc = query_start_loc.to(self.device, non_blocking=True)
        seq_start_loc = seq_start_loc.to(self.device, non_blocking=True)
        slot_mapping = slot_mapping.to(self.device, non_blocking=True).long()

        attn_metadata = IPEXAttentionMetadata(
            is_prefill_only=min_num_scheduled_tokens > 1,
            is_decode_only=max_num_scheduled_tokens == 1,
            num_actual_tokens=total_num_scheduled_tokens,
            max_query_len=max_num_scheduled_tokens,
            query_start_loc=query_start_loc,
            seq_len_q=seq_lens_tensor,
            max_seq_len=max_seq_len,
            seq_start_loc=seq_start_loc,
            block_table=self.input_batch.block_table[:num_reqs],
            slot_mapping=slot_mapping,
        )
        print(attn_metadata)
        # NOTE(woosuk): Due to chunked prefills, there can be at most 1 partial
        # request in the batch. While we should not sample any token from this
        # partial request, we do so for simplicity. We will ignore the sampled
        # token from the partial request.
        # TODO: Support prompt logprobs.
        logits_indices = query_start_loc[1:] - 1
        return input_ids, attn_metadata, logits_indices
