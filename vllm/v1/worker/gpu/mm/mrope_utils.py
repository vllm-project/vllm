# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.model_executor.models.interfaces import SupportsMRoPE
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor


class MRopeState:
    def __init__(
        self,
        max_num_reqs: int,
        max_num_tokens: int,
        max_model_len: int,
        device: torch.device,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens
        self.max_model_len = max_model_len
        self.device = device

        # NOTE(woosuk): This tensor can be extremely large (e.g., several GBs)
        # wasting a lot of CPU memory.
        self.prefill_mrope_positions = StagedWriteTensor(
            (max_num_reqs * 3, max_model_len),
            dtype=torch.int32,
            device=device,
            uva_instead_of_gpu=True,
        )
        self.prefill_mrope_delta = UvaBackedTensor(max_num_reqs, dtype=torch.int32)

        # NOTE: `mrope_positions` is implemented with one additional dummy
        # position on purpose to make it non-contiguous so that it can work
        # with torch compile.
        # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923
        # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
        # the modality of inputs. For text-only inputs, each dimension has
        # identical position IDs, making M-RoPE functionally equivalent to
        # 1D-RoPE.
        # See page 5 of https://arxiv.org/abs/2409.12191
        self.mrope_positions = torch.zeros(
            (3, max_num_tokens + 1), dtype=torch.int64, device=device
        )

    def init_prefill_mrope_positions(
        self,
        req_idx: int,
        mrope_model: SupportsMRoPE,
        prefill_token_ids: list[int],
        mm_features: list,
    ) -> None:
        prefill_mrope_positions, prefill_mrope_delta = (
            mrope_model.get_mrope_input_positions(prefill_token_ids, mm_features)
        )
        for i in range(3):
            pos = prefill_mrope_positions[i].tolist()
            self.prefill_mrope_positions.stage_write(3 * req_idx + i, 0, pos)
        self.prefill_mrope_delta.np[req_idx] = prefill_mrope_delta

    def apply_staged_writes(self) -> None:
        self.prefill_mrope_positions.apply_write()
        self.prefill_mrope_delta.copy_to_uva()

    def prepare_mrope_positions(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        prefill_lens: torch.Tensor,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        num_reqs = idx_mapping.shape[0]
        _prepare_mrope_positions_kernel[(num_reqs,)](
            self.mrope_positions,
            self.mrope_positions.stride(0),
            self.prefill_mrope_positions.gpu,
            3 * self.max_model_len,
            self.max_model_len,
            self.prefill_mrope_delta.gpu,
            idx_mapping,
            query_start_loc,
            prefill_lens,
            num_computed_tokens,
            BLOCK_SIZE=1024,
        )


@triton.jit
def _prepare_mrope_positions_kernel(
    mrope_positions_ptr,
    mrope_positions_stride,
    prefill_mrope_positions_ptr,
    prefill_mrope_positions_stride0,
    prefill_mrope_positions_stride1,
    prefill_mrope_delta_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    prefill_lens_ptr,
    num_computed_tokens_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    prefill_len = tl.load(prefill_lens_ptr + req_state_idx)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    is_prefill = num_computed < prefill_len

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    mrope_delta = tl.load(prefill_mrope_delta_ptr + req_state_idx)
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        orig_pos = num_computed + block

        for j in tl.static_range(3):
            if is_prefill:
                # Read from pre-computed M-RoPE positions.
                pos = tl.load(
                    prefill_mrope_positions_ptr
                    + req_state_idx * prefill_mrope_positions_stride0
                    + j * prefill_mrope_positions_stride1
                    + orig_pos,
                    mask=mask,
                )
            else:
                # Apply M-RoPE delta.
                pos = orig_pos + mrope_delta
            tl.store(
                mrope_positions_ptr + j * mrope_positions_stride + query_start + block,
                pos,
                mask=mask,
            )
