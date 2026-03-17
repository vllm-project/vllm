# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import cast

import torch
import torch.nn as nn

from vllm.config import ModelConfig
from vllm.model_executor.models.interfaces import SupportsMRoPE, SupportsXDRoPE
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor


class RopeState:
    """Unified state for multi-dimensional RoPE variants (M-RoPE, XD-RoPE).

    M-RoPE: 3 dims, uses position delta for decode.
    XD-RoPE: 3 or 4 dims, delta is 0 (decode uses orig_pos for all dims).

    NOTE: `positions` is implemented with one additional dummy position on
    purpose to make it non-contiguous so that it can work with torch compile.
    See detailed explanation in
    https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

    NOTE: When M-RoPE is enabled, position ids are 3D regardless of the
    modality of inputs. For text-only inputs, each dimension has identical
    position IDs, making M-RoPE functionally equivalent to 1D-RoPE.
    See page 5 of https://arxiv.org/abs/2409.12191
    """

    def __init__(
        self,
        num_dims: int,
        has_delta: bool,
        max_num_reqs: int,
        max_num_tokens: int,
        max_model_len: int,
        device: torch.device,
    ):
        self.num_dims = num_dims
        self.has_delta = has_delta
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens
        self.max_model_len = max_model_len
        self.device = device

        # NOTE(woosuk): This tensor can be extremely large (e.g., several GBs)
        # wasting a lot of CPU memory.
        self.prefill_positions = StagedWriteTensor(
            (max_num_reqs * num_dims, max_model_len),
            dtype=torch.int32,
            device=device,
            uva_instead_of_gpu=True,
        )
        self.positions = torch.zeros(
            (num_dims, max_num_tokens + 1), dtype=torch.int64, device=device
        )

        # Delta is non-zero for M-RoPE, always 0 for XD-RoPE.
        self.prefill_delta = UvaBackedTensor(max_num_reqs, dtype=torch.int32)

    def init_prefill_positions(
        self,
        req_idx: int,
        model: nn.Module,
        prefill_token_ids: list[int],
        mm_features: list,
    ) -> None:
        if self.has_delta:
            mrope_model = cast(SupportsMRoPE, model)
            prefill_positions, delta = mrope_model.get_mrope_input_positions(
                prefill_token_ids, mm_features
            )
            self.prefill_delta.np[req_idx] = delta
        else:
            xdrope_model = cast(SupportsXDRoPE, model)
            prefill_positions = xdrope_model.get_xdrope_input_positions(
                prefill_token_ids, mm_features
            )

        for i in range(self.num_dims):
            pos = prefill_positions[i].tolist()
            self.prefill_positions.stage_write(self.num_dims * req_idx + i, 0, pos)

    def apply_staged_writes(self) -> None:
        self.prefill_positions.apply_write()
        if self.has_delta:
            self.prefill_delta.copy_to_uva()

    def get_positions(self, num_tokens: int) -> torch.Tensor:
        return self.positions[:, :num_tokens]

    def prepare_positions(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        prefill_lens: torch.Tensor,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        num_reqs = idx_mapping.shape[0]
        _prepare_rope_positions_kernel[(num_reqs,)](
            self.positions,
            self.positions.stride(0),
            self.prefill_positions.gpu,
            self.num_dims * self.max_model_len,
            self.max_model_len,
            self.prefill_delta.gpu,
            idx_mapping,
            query_start_loc,
            prefill_lens,
            num_computed_tokens,
            BLOCK_SIZE=1024,
            NUM_DIMS=self.num_dims,
        )


def get_rope_state(
    model_config: ModelConfig,
    model: nn.Module,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
    device: torch.device,
) -> RopeState | None:
    """Create a RopeState if the model uses multi-dimensional RoPE."""
    if model_config.uses_mrope:
        assert isinstance(model, SupportsMRoPE)
        return RopeState(
            num_dims=3,
            has_delta=True,
            max_num_reqs=max_num_reqs,
            max_num_tokens=max_num_tokens,
            max_model_len=max_model_len,
            device=device,
        )
    if model_config.uses_xdrope_dim > 0:
        assert isinstance(model, SupportsXDRoPE)
        return RopeState(
            num_dims=model_config.uses_xdrope_dim,
            has_delta=False,
            max_num_reqs=max_num_reqs,
            max_num_tokens=max_num_tokens,
            max_model_len=max_model_len,
            device=device,
        )
    return None


@triton.jit
def _prepare_rope_positions_kernel(
    positions_ptr,
    positions_stride,
    prefill_positions_ptr,
    prefill_positions_stride0,
    prefill_positions_stride1,
    prefill_delta_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    prefill_lens_ptr,
    num_computed_tokens_ptr,
    BLOCK_SIZE: tl.constexpr,
    NUM_DIMS: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    prefill_len = tl.load(prefill_lens_ptr + req_state_idx)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    is_prefill = num_computed < prefill_len

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    delta = tl.load(prefill_delta_ptr + req_state_idx)

    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        orig_pos = num_computed + block

        for j in tl.static_range(NUM_DIMS):
            if is_prefill:
                pos = tl.load(
                    prefill_positions_ptr
                    + req_state_idx * prefill_positions_stride0
                    + j * prefill_positions_stride1
                    + orig_pos,
                    mask=mask,
                )
            else:
                pos = orig_pos + delta
            tl.store(
                positions_ptr + j * positions_stride + query_start + block,
                pos,
                mask=mask,
            )
