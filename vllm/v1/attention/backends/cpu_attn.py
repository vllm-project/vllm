# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.backends.torch_sdpa import (TorchSDPABackendImpl,
                                                TorchSDPAMetadata)
from vllm.attention.backends.utils import CommonAttentionState
from vllm.attention.ops.ipex_attn import PagedAttention
from vllm.v1.attention.backends.utils import (AttentionMetadataBuilder,
                                              CommonAttentionMetadata)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.cpu_model_runner import CPUModelRunner
from vllm.v1.worker.gpu_input_batch import InputBatch


class TorchSDPABackend:
    accept_output_buffer: bool = False

    @staticmethod
    def get_name() -> str:
        return "TORCH_SDPA_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["TorchSDPABackendImpl"]:
        return TorchSDPABackendImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return TorchSDPAMetadata

    @staticmethod
    def get_state_cls() -> type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_builder_cls() -> type["TorchSDPAMetadataBuilderV1"]:
        return TorchSDPAMetadataBuilderV1

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                 num_kv_heads, head_size)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


class TorchSDPAMetadataBuilderV1(AttentionMetadataBuilder[TorchSDPAMetadata]):

    def __init__(self, runner: CPUModelRunner, kv_cache_spec: AttentionSpec,
                 block_table: BlockTable) -> None:
        self.runner = runner
        self.block_table = block_table

        # For reorder
        self.reorder_prompt_req_index_list = np.empty(self.runner.max_num_reqs,
                                                      dtype=np.int64)
        self.reorder_decode_req_index_list = np.empty(self.runner.max_num_reqs,
                                                      dtype=np.int64)
        self.num_prompt_req: int = 0

        self.seq_start_loc_cpu = torch.zeros(
            runner.max_num_reqs + 1,
            dtype=torch.int32,
            device="cpu",
        )
        self.seq_start_loc_np = self.seq_start_loc_cpu.numpy()

    def reorder_batch(self, input_batch: InputBatch,
                      scheduler_output: SchedulerOutput) -> bool:
        prompt_list_idx = 0
        decode_list_idx = 0
        for req_index in range(input_batch.num_reqs):
            if input_batch.num_computed_tokens_cpu[
                    req_index] < input_batch.num_prompt_tokens[req_index]:
                # prompt stage
                self.reorder_prompt_req_index_list[prompt_list_idx] = req_index
                prompt_list_idx += 1
            else:
                # decode stage
                self.reorder_decode_req_index_list[decode_list_idx] = req_index
                decode_list_idx += 1
        assert decode_list_idx + prompt_list_idx == input_batch.num_reqs

        # Update prompt requests number
        self.num_prompt_req = prompt_list_idx

        reorder_req_num = 0
        for req_index in range(decode_list_idx):
            if self.reorder_decode_req_index_list[req_index] < prompt_list_idx:
                reorder_req_num += 1
            else:
                break

        if reorder_req_num == 0:
            return False

        reorder_prompt_list = (
            self.reorder_prompt_req_index_list[:prompt_list_idx]
            [-reorder_req_num:])
        reorder_decode_list = (
            self.reorder_decode_req_index_list[:decode_list_idx]
            [:reorder_req_num])
        assert reorder_decode_list.size == reorder_prompt_list.size

        for idx in range(reorder_req_num):
            prompt_req_index = reorder_prompt_list[idx].item()
            decode_req_index = reorder_decode_list[idx].item()
            input_batch.swap_states(prompt_req_index, decode_req_index)

        return True

    def build(self, common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata):
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        runner = self.runner
        block_table = self.block_table
        seq_lens_np = runner.seq_lens_np[:num_reqs]
        num_prompt_req = self.num_prompt_req
        max_prefill_seq_len = seq_lens_np[:num_prompt_req].max().item(
        ) if num_prompt_req > 0 else 0
        max_decode_seq_len = seq_lens_np[num_prompt_req:num_reqs].max().item(
        ) if num_prompt_req < num_reqs else 0
        self.seq_start_loc_np[0] = 0
        np.cumsum(seq_lens_np, out=self.seq_start_loc_np[1:num_reqs + 1])
        num_prefill_tokens = runner.query_start_loc_np[num_prompt_req].item()
        num_decode_tokens = runner.query_start_loc_np[num_reqs].item(
        ) - num_prefill_tokens
        slot_mapping = block_table.slot_mapping_cpu[:num_actual_tokens].long()
        block_table_tensor = block_table.get_device_tensor()
        attn_metadata = TorchSDPAMetadata(
            num_prefills=num_prompt_req,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            slot_mapping=slot_mapping,
            seq_lens_tensor=runner.
            seq_lens_cpu[num_prompt_req:num_reqs],  # decode
            max_decode_seq_len=max_decode_seq_len,  # decode
            block_tables=block_table_tensor[num_prompt_req:num_reqs],  # decode
            chunked_prefill=True,
            max_query_len=max_query_len,
            max_kv_len=max_prefill_seq_len,
            prefill_query_start_loc=runner.
            query_start_loc_cpu[:num_prompt_req + 1],  # prefill
            kv_start_loc=self.seq_start_loc_cpu[:num_prompt_req +
                                                1],  # prefill
            prefill_block_tables=block_table_tensor[:
                                                    num_prompt_req],  # prefill
            query_start_loc=runner.query_start_loc_cpu[:num_reqs +
                                                       1],  # for logits index
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
        )

        return attn_metadata
