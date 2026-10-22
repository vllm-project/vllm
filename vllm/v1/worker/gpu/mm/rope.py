# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import cast

import torch
import torch.nn as nn

from vllm.config import ModelConfig
from vllm.model_executor.models.interfaces import SupportsMRoPE
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor


class RopeState:
    """State for multi-dimensional (M-RoPE) positions.

    `num_dims` is the number of position channels the model consumes, which
    the model config derives from its M-RoPE sections.

    NOTE: `positions` is implemented with one additional dummy position on
    purpose to make it non-contiguous so that it can work with torch compile.
    See detailed explanation in
    https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

    NOTE: For text-only inputs, each dimension has identical position IDs,
    making M-RoPE functionally equivalent to 1D-RoPE.
    See page 5 of https://arxiv.org/abs/2409.12191

    With `linear_prefill_positions`, every request is text-only, so the
    kernel computes the positions from the token index and no per-request
    prefill positions are staged.
    """

    def __init__(
        self,
        num_dims: int,
        max_num_reqs: int,
        max_num_tokens: int,
        max_model_len: int,
        device: torch.device,
        linear_prefill_positions: bool = False,
    ):
        self.num_dims = num_dims
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens
        self.max_model_len = max_model_len
        self.device = device
        self.linear_prefill_positions = linear_prefill_positions

        self.prefill_positions: StagedWriteTensor | None = None
        self.prefill_delta: UvaBackedTensor | None = None
        if not linear_prefill_positions:
            # NOTE(woosuk): This tensor can be extremely large (e.g., several GBs)
            # wasting a lot of CPU memory.
            self.prefill_positions = StagedWriteTensor(
                (max_num_reqs * num_dims, max_model_len),
                dtype=torch.int32,
                device=device,
                uva_instead_of_gpu=True,
            )
            self.prefill_delta = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.positions = torch.zeros(
            (num_dims, max_num_tokens + 1), dtype=torch.int64, device=device
        )

    def init_prefill_positions(
        self,
        req_idx: int,
        model: nn.Module,
        prefill_token_ids: list[int],
        mm_features: list,
    ) -> None:
        if self.linear_prefill_positions:
            if mm_features:
                raise RuntimeError(
                    "Linear M-RoPE positions cannot be used with multimodal inputs."
                )
            return
        assert self.prefill_positions is not None
        assert self.prefill_delta is not None
        mrope_model = cast(SupportsMRoPE, model)
        prefill_positions, delta = mrope_model.get_mrope_input_positions(
            prefill_token_ids, mm_features
        )
        self.prefill_delta.np[req_idx] = delta

        for i in range(self.num_dims):
            pos = prefill_positions[i].tolist()
            self.prefill_positions.stage_write(self.num_dims * req_idx + i, 0, pos)

    def apply_staged_writes(self) -> None:
        if self.linear_prefill_positions:
            return
        assert self.prefill_positions is not None
        assert self.prefill_delta is not None
        self.prefill_positions.apply_write()
        self.prefill_delta.copy_to_uva()

    def get_positions(self, num_tokens: int) -> torch.Tensor:
        return self.positions[:, :num_tokens]

    def read_prefill_positions(self, req_idx: int, length: int) -> torch.Tensor:
        """Return staged per-request prefill positions as [num_dims, length]."""
        assert self.prefill_positions is not None
        base = self.num_dims * req_idx
        return self.prefill_positions.gpu[base : base + self.num_dims, :length]

    def update_prefill_positions(
        self, req_idx: int, positions: torch.Tensor, delta: int
    ) -> None:
        """Overwrite a request's staged prefill positions with recomputed values."""
        assert self.prefill_positions is not None
        assert self.prefill_delta is not None
        base = self.num_dims * req_idx
        length = positions.shape[1]
        self.prefill_positions.gpu[base : base + self.num_dims, :length].copy_(
            positions
        )
        self.prefill_delta.np[req_idx] = delta

    def prepare_positions(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        prefill_lens: torch.Tensor,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        num_reqs = idx_mapping.shape[0]
        # The kernel does not read the staged buffers in linear mode; the
        # output tensor stands in for them.
        prefill_positions = self.positions
        prefill_delta = self.positions
        if not self.linear_prefill_positions:
            assert self.prefill_positions is not None
            assert self.prefill_delta is not None
            prefill_positions = self.prefill_positions.gpu
            prefill_delta = self.prefill_delta.gpu
        _prepare_rope_positions_kernel[(num_reqs,)](
            self.positions,
            self.positions.stride(0),
            prefill_positions,
            self.num_dims * self.max_model_len,
            self.max_model_len,
            prefill_delta,
            idx_mapping,
            query_start_loc,
            prefill_lens,
            num_computed_tokens,
            BLOCK_SIZE=1024,
            NUM_DIMS=self.num_dims,
            LINEAR_PREFILL=self.linear_prefill_positions,
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
    if not model_config.uses_mrope:
        return None

    assert isinstance(model, SupportsMRoPE)
    # Without multimodal ingress every request is text-only, so an opted-in
    # model's positions are sequential in every channel. This excludes
    # --enable-mm-embeds, which keeps multimodal ingress enabled.
    linear_prefill_positions = (
        model.supports_linear_text_mrope and not model_config.supports_multimodal_inputs
    )
    return RopeState(
        num_dims=model_config.mrope_num_dims,
        max_num_reqs=max_num_reqs,
        max_num_tokens=max_num_tokens,
        max_model_len=max_model_len,
        device=device,
        linear_prefill_positions=linear_prefill_positions,
    )


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
    LINEAR_PREFILL: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    prefill_len = tl.load(prefill_lens_ptr + req_state_idx)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    is_prefill = num_computed < prefill_len

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    if not LINEAR_PREFILL:
        delta = tl.load(prefill_delta_ptr + req_state_idx)

    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        orig_pos = num_computed + block

        for j in tl.static_range(NUM_DIMS):
            if LINEAR_PREFILL:
                pos = orig_pos
            elif is_prefill:
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
