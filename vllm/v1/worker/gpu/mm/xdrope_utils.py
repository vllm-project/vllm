# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.model_executor.models.interfaces import SupportsXDRoPE
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor


class XDRopeState:
    def __init__(
        self,
        uses_xdrope_dim: int,
        max_num_reqs: int,
        max_num_tokens: int,
        max_model_len: int,
        device: torch.device,
    ):
        self.uses_xdrope_dim = uses_xdrope_dim
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens
        self.max_model_len = max_model_len
        self.device = device

        # NOTE(woosuk): This tensor can be extremely large (e.g., several GBs)
        # wasting a lot of CPU memory.
        self.prefill_xdrope_positions = StagedWriteTensor(
            (max_num_reqs * uses_xdrope_dim, max_model_len),
            dtype=torch.int32,
            device=device,
            uva_instead_of_gpu=True,
        )
        self.xdrope_positions = torch.zeros(
            (uses_xdrope_dim, max_num_tokens + 1), dtype=torch.int64, device=device
        )

    def init_prefill_xdrope_positions(
        self,
        req_idx: int,
        xdrope_model: SupportsXDRoPE,
        prefill_token_ids: list[int],
        mm_features: list,
    ) -> None:
        prefill_xdrope_positions = xdrope_model.get_xdrope_input_positions(
            prefill_token_ids, mm_features
        )
        for i in range(self.uses_xdrope_dim):
            pos = prefill_xdrope_positions[i].tolist()
            self.prefill_xdrope_positions.stage_write(
                self.uses_xdrope_dim * req_idx + i, 0, pos
            )

    def apply_staged_writes(self) -> None:
        self.prefill_xdrope_positions.apply_write()

    def prepare_xdrope_positions(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        prefill_lens: torch.Tensor,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        num_reqs = idx_mapping.shape[0]
        _prepare_xdrope_positions_kernel[(num_reqs,)](
            self.xdrope_positions,
            self.xdrope_positions.stride(0),
            self.prefill_xdrope_positions.gpu,
            self.uses_xdrope_dim * self.max_model_len,
            self.max_model_len,
            idx_mapping,
            query_start_loc,
            prefill_lens,
            num_computed_tokens,
            BLOCK_SIZE=1024,
            USES_XDROPE_DIM=self.uses_xdrope_dim,
        )


@triton.jit
def _prepare_xdrope_positions_kernel(
    xdrope_positions_ptr,
    xdrope_positions_stride,
    prefill_xdrope_positions_ptr,
    prefill_xdrope_positions_stride0,
    prefill_xdrope_positions_stride1,
    idx_mapping_ptr,
    query_start_loc_ptr,
    prefill_lens_ptr,
    num_computed_tokens_ptr,
    BLOCK_SIZE: tl.constexpr,
    USES_XDROPE_DIM: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    prefill_len = tl.load(prefill_lens_ptr + req_state_idx)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    is_prefill = num_computed < prefill_len

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        orig_pos = num_computed + block

        for j in tl.static_range(USES_XDROPE_DIM):
            if is_prefill:
                # Read from pre-computed XD-RoPE positions.
                pos = tl.load(
                    prefill_xdrope_positions_ptr
                    + req_state_idx * prefill_xdrope_positions_stride0
                    + j * prefill_xdrope_positions_stride1
                    + orig_pos,
                    mask=mask,
                )
            else:
                pos = orig_pos
            tl.store(
                xdrope_positions_ptr
                + j * xdrope_positions_stride
                + query_start
                + block,
                pos,
                mask=mask,
            )
